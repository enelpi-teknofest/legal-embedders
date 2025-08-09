from datasets import load_dataset, Dataset
from datasets import Dataset
import random

def build_q_pos_neg(example, dataset, all_texts):
    q = example["genel_soru"]
    pos_text = example["text"]
    
    
    # Avoid selecting the same text as pos_text
    negatives = random.sample(
        [t for t in all_texts if t != pos_text],
        k=5
    )
    
    # Merge original fields with new fields
    return {
        "q": q,
        "pos_text": pos_text,
        "hard_negs": negatives,
        **example,  # keep all original fields
    }


if __name__ == "__main__":

    # ds_name = "fikriokan/sonbahcem-tblg-batch-1-processed-1"
    # new_ds_name = "fikriokan/sonbahcem-tblg-batch-1-qposneg-1"

    ds_name = "fikriokan/sonbahcem-krm-batch-2-processed-1"
    new_ds_name = "fikriokan/sonbahcem-krm-qposneg-1"

    ds = load_dataset(ds_name)
    cfgs = list(ds.keys())

    for i,cfg in enumerate(cfgs):
        print(f"Current Cfg: {i}/{len(cfgs)}")
        ds_cur = ds[cfg]#.select(range(0, 2000))
        all_texts = ds_cur["text"]
        new_ds = ds_cur.map(lambda ex: build_q_pos_neg(ex, ds_cur, all_texts), num_proc=16*4)
        new_ds = new_ds.remove_columns(['text', 'genel_soru'])
        new_ds.push_to_hub(new_ds_name, cfg)
