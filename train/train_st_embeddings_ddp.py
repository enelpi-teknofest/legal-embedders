#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU training for Turkish legal retrieval using Hugging Face Datasets + Accelerate.
Implements:
  - Multiple Negatives Ranking Loss (InfoNCE) with cross-device in-batch negatives
  - Optional Triplet loss if hard negatives are provided
  - E5/BGE-style prompts: "query: ..." / "passage: ..."
Launch:
  accelerate config  # run once to set up (use multi-GPU, fp16/bf16 as you like)
  accelerate launch train_st_embeddings_ddp.py \
    --data_path /mnt/data/tiny.csv \
    --model_name intfloat/multilingual-e5-base \
    --output_dir /mnt/data/models/e5-legal-tr-accel \
    --batch_size 64 --epochs 1 --lr 2e-5 --max_query_len 96 --max_passage_len 384 \
    --use_triplet_if_negs --margin 0.25
"""
import argparse
import math
import os
import random
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from accelerate import Accelerator

def seed_all(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return F.normalize(summed / counts, p=2, dim=1)

def fmt_query(q: str): return f"query: {q}"
def fmt_passage(p: str): return f"passage: {p}"

class PairDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_q_len=96, max_p_len=384, include_triplets=False):
        self.rows = rows
        self.tok = tokenizer
        self.max_q = max_q_len
        self.max_p = max_p_len
        self.include_triplets = include_triplets

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        q = fmt_query(str(r["q"]).strip())
        p = fmt_passage(str(r["pos_text"]).strip())

        item = {
            "q_input": self.tok(q, truncation=True, max_length=self.max_q, padding=False, return_tensors=None),
            "p_pos_input": self.tok(p, truncation=True, max_length=self.max_p, padding=False, return_tensors=None),
        }
        if self.include_triplets and r.get("hard_negs"):
            negs_raw = r["hard_negs"]
            if isinstance(negs_raw, list):
                negs = [str(x).strip() for x in negs_raw if str(x).strip()]
            else:
                s = str(negs_raw)
                if "|||" in s: negs = s.split("|||")
                elif "||" in s: negs = s.split("||")
                elif "|" in s: negs = s.split("|")
                else: negs = [s]
                negs = [n.strip() for n in negs if n.strip()]
            if len(negs) > 0:
                n = fmt_passage(negs[0])
                item["p_neg_input"] = self.tok(n, truncation=True, max_length=self.max_p, padding=False, return_tensors=None)
        return item

def collate_fn(batch, pad):
    keys = ["input_ids", "attention_mask", "token_type_ids"]
    def pad_pack(items):
        # items is a list of tokenized dicts (may lack token_type_ids)
        if len(items) == 0: return None
        out = {}
        for k in keys:
            tensors = [torch.tensor(x[k]) for x in items if k in x]
            if len(tensors) == 0: continue
            out[k] = pad(tensors, padding=True, return_tensors="pt")["input_ids" if k=="input_ids" else k]
        return out

    q_items = [b["q_input"] for b in batch]
    p_pos_items = [b["p_pos_input"] for b in batch]
    p_neg_items = [b.get("p_neg_input") for b in batch if "p_neg_input" in b]

    q = pad(q_items, padding=True, return_tensors="pt")
    p_pos = pad(p_pos_items, padding=True, return_tensors="pt")
    p_neg = pad(p_neg_items, padding=True, return_tensors="pt") if len(p_neg_items) > 0 else None

    return {"q": q, "p_pos": p_pos, "p_neg": p_neg}

def info_nce_loss(accelerator, q_emb, p_pos_emb, temperature=0.05):
    # Gather embeddings from all processes for cross-device negatives
    q_all = accelerator.gather(q_emb)
    p_all = accelerator.gather(p_pos_emb)
    # Select only the local slice for labels
    global_batch = q_all.size(0)
    sim = (q_all @ p_all.t()) / temperature  # cosine since we normalized
    labels = torch.arange(global_batch, device=sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss

def triplet_loss(q, pos, neg, margin=0.25):
    # Cosine distance; embeddings already normalized
    d_pos = 1 - (q * pos).sum(dim=1)
    d_neg = 1 - (q * neg).sum(dim=1)
    return F.relu(margin + d_pos - d_neg).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-base")
    ap.add_argument("--output_dir", type=str, default="./models/out-accel")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_query_len", type=int, default=96)
    ap.add_argument("--max_passage_len", type=int, default=384)
    ap.add_argument("--use_triplet_if_negs", action="store_true")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=1, log_with=None)
    seed_all(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    ext = args.data_path.split(".")[-1].lower()
    if ext == "csv":
        ds = load_dataset("csv", data_files=args.data_path)["train"]
    elif ext in {"json", "jsonl"}:
        ds = load_dataset("json", data_files=args.data_path, split="train")
    else:
        ds = load_dataset(args.data_path, split="train")

    rows = [r for r in ds if r.get("q") and r.get("pos_text")]
    random.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    # Some multilingual models benefit from pooling of first token vs mean; we'll do mean pooling
    model.train()

    include_triplets = args.use_triplet_if_negs and "hard_negs" in ds.column_names
    dataset = PairDataset(rows, tokenizer, args.max_query_len, args.max_passage_len, include_triplets)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = math.ceil(len(dl) * args.epochs)
    num_warmup = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_training_steps)

    model, optimizer, dl, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                # Encode query & pos
                q_out = model(**batch["q"])
                p_out = model(**batch["p_pos"])
                q_emb = mean_pooling(q_out.last_hidden_state, batch["q"]["attention_mask"])
                p_pos_emb = mean_pooling(p_out.last_hidden_state, batch["p_pos"]["attention_mask"])

                # Normalize for cosine
                q_emb = F.normalize(q_emb, p=2, dim=1)
                p_pos_emb = F.normalize(p_pos_emb, p=2, dim=1)

                loss = info_nce_loss(accelerator, q_emb, p_pos_emb)

                # Optional triplet if negatives exist in batch
                if include_triplets and batch["p_neg"] is not None:
                    n_out = model(**batch["p_neg"])
                    n_emb = mean_pooling(n_out.last_hidden_state, batch["p_neg"]["attention_mask"])
                    n_emb = F.normalize(n_emb, p=2, dim=1)
                    loss = loss + triplet_loss(q_emb, p_pos_emb, n_emb, margin=args.margin)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        if accelerator.is_main_process:
            # Save checkpoint per epoch
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    if accelerator.is_main_process:
        print("✅ Training complete. Saved to:", args.output_dir)

if __name__ == "__main__":
    main()
