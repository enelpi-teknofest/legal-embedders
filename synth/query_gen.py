from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import yaml
import argparse
import re

with open("prompts.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cfg_idx = 0
model_name = "google/gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, max_model_len=8192)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
]
terminators = [t for t in terminators if t is not None]
llm.llm_engine.tokenizer.eos_token_id = terminators

sampling_params = SamplingParams(
    max_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    stop_token_ids=terminators,
)

def process_texts(texts):
    prompts = []
    valid_indices = []

    for idx, text in enumerate(tqdm(texts)):
        messages = [
            {"role": "user", "content": sys + text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        tokenized = tokenizer(prompt, return_tensors=None, add_special_tokens=False)
        if len(tokenized["input_ids"]) >= 8192:
            print(f"[WARN] Truncating idx={idx}, prompt too long ({len(tokenized['input_ids'])} tokens)")
            # Truncate the input_ids to fit within the limit
            max_input_length = 500  # Leave some room for generation
            truncated_ids = tokenized["input_ids"][:max_input_length]
            truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            prompts.append(truncated_prompt)
        else:
            prompts.append(prompt)
        valid_indices.append(idx)
    
    return prompts, valid_indices

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process dataset and prompt configuration.")

    parser.add_argument("--prompt", type=str, default="ygty_prompt", help="Which prompt tp get")
    parser.add_argument("--start", type=int, default=0, help="Start index of config splits.")
    parser.add_argument("--end", type=int, default=10, help="End index of config splits.")
    parser.add_argument("--ds_name", type=str, default="fikriokan/ygty", help="Dataset name or path.")
    parser.add_argument("--out_ds_name", type=str, default="ygty_post1", help="Output Dataset name or path.")
    parser.add_argument("--split", type=str, default=None, help="Output Dataset name or path.")
    parser.add_argument("--col_name", type=str, default='genel_soru', help="Output Column Name.")

    args = parser.parse_args()
    sys = config[args.prompt]

    # Load dataset and configs
    ds = load_dataset(args.ds_name)
    cfgs = set(list(ds.keys())[args.start:args.end])

    try:
        ds_out = load_dataset(args.out_ds_name)
        cfgs_out = set(ds_out.keys())
        cfgs = list(cfgs - cfgs_out)
    except:
        print("[INFO] Output Dataset Doesn't Exist.")
    
    if len(cfgs) <= 0:
        print("[INFO] No Configs To Process.")
        exit(0)


    print("[INFO] Processing Configs:", "\t".join(cfgs))
    print("[INFO] Processing Configs:", "\t".join(cfgs))

    for i, cfg in enumerate(cfgs):
        print('[INFO] Current Idx:', args.start + i)
        ds_cur = ds[cfg]# .select(range(100)) # Debugging Purposes
        texts = ds_cur['text']

        # Process prompts and get indices of valid ones
        prompts, valid_indices = process_texts(texts)

        # Only run LLM on valid prompts
        out = llm.generate(prompts, sampling_params)
        out = [o.outputs[0].text for o in out]

        codeblock_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

        parsed_outputs = []
        failed_parsing = 0

        for output in out:
            try:
                # Try to extract the first code block
                match = codeblock_pattern.search(output.strip())
                if match:
                    output = match.group(1)  # Extract just the JSON part

                parsed_json = json.loads(output)
                parsed_outputs.append(parsed_json)

            except json.JSONDecodeError:
                failed_parsing += 1
                parsed_outputs.append(None)
                print("[DEBUG] Failed To Parse:", output)

        # Map parsed questions back to the dataset using valid_indices
        questions = [None] * len(texts)
        for i, idx in enumerate(valid_indices):
            parsed = parsed_outputs[i]
            try:
                questions[idx] = parsed[args.col_name] if parsed else None
            except:
                questions[idx] = None

        # Filter the dataset to rows that had valid prompts (and thus valid output space)
        # Filter dataset first
        print("[DEBUG]  Parsed Output:", parsed_outputs[0])
        ds_cur = ds_cur.select(valid_indices)

        # Map directly to selected dataset
        questions = []
        for output in parsed_outputs:
            if output and args.col_name in output:
                questions.append(output[args.col_name])
            elif output:
                questions.append(output)
            else:
                questions.append(None)

        # Make sure lengths match
        assert len(questions) == len(ds_cur), "Mismatch between questions and dataset rows"

        # Add column
        ds_cur = ds_cur.add_column(args.col_name, questions)
        print("[DEBUG] Output Dataset 1st Element:", ds_cur[0][args.col_name])

        # Get all column names except 'genel_soru' (which we're adding)
        original_columns = {col: ds_cur[col] for col in ds_cur.column_names}
        original_columns[args.col_name] = ds_cur[args.col_name]
        
        ds_cur = Dataset.from_dict(original_columns)

        if args.split:
            ds_cur.push_to_hub(args.out_ds_name, split=args.split)
        else:
            ds_cur.push_to_hub(args.out_ds_name, split=cfg)
            print("[INFO] Pushed name=", args.out_ds_name, "split=", cfg)
