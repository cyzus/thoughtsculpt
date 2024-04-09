from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd

def load_dataset_from_disk(fname):
    ds = load_from_disk(f"{fname}")
    return ds

def prepare_dataset(content_evaluator, fname, save=False, outname=None):
    ds = load_dataset("json", data_files=f"{fname}.jsonl", split="train")
    ds = ds.map(lambda x: {"new_outlines":x["new_outlines"][:len(x["mask"])]})
    ds = ds.map(lambda x: {"input_prompt": content_evaluator.create_prompt(x["new_outlines"], x["mask"]), "label": 1 if x["interesting"] else 0})
    tokenized_ds = ds.map(lambda x: content_evaluator.tokenizer(x["input_prompt"], padding="max_length", max_length=512, truncation=True, return_tensors="pt"), batched=True)
    if save:
        tokenized_ds.save_to_disk(f"datasets/{outname}")
    return tokenized_ds

def prepare_paired_dataset(content_evaluator, fname, save=False, outname=None):
    ds = load_dataset("json", data_files=f"{fname}.jsonl", split="train")
    ds = ds.map(lambda x: {"new_outlines":x["new_outlines"][:len(x["mask"])]})
    ds = ds.map(lambda x: {"input_prompt": content_evaluator.create_prompt(x["new_outlines"], x["mask"]), "label": 1 if x["interesting"] else 0})

    interesting_ds = ds.select(range(0, len(ds), 2)).to_pandas()
    non_interesting_ds = ds.select(range(1, len(ds), 2)).to_pandas()
    combined_ds = pd.DataFrame(
        {"interesting_outlines": interesting_ds["new_outlines"], 
        "non_interesting_outlines": non_interesting_ds["new_outlines"],
        "original_outlines": interesting_ds["original_outlines"],
        "input_prompt0": non_interesting_ds["input_prompt"],
        "input_prompt1": interesting_ds["input_prompt"],
        "label0": non_interesting_ds["label"],
        "label1": interesting_ds["label"],
        }
    )
    combined_ds = Dataset.from_pandas(combined_ds)
    # tokenize input_prompt0 and rename the columns as input_ids0 and attention_mask0
    combined_ds = combined_ds.map(lambda x: content_evaluator.tokenizer(x["input_prompt0"], padding="max_length", max_length=512, truncation=True, return_tensors="pt"), batched=True)
    combined_ds = combined_ds.map(lambda x: {"input_ids0": x["input_ids"], "attention_mask0": x["attention_mask"]})
    combined_ds = combined_ds.map(lambda x: {"labels" : 0 if x["label0"] == x["label1"] else 1})

    # tokenize input_prompt1 and rename the columns as input_ids1 and attention_mask1
    combined_ds = combined_ds.map(lambda x: content_evaluator.tokenizer(x["input_prompt1"], padding="max_length", max_length=512, truncation=True, return_tensors="pt"), batched=True)
    combined_ds = combined_ds.map(lambda x: {"input_ids1": x["input_ids"], "attention_mask1": x["attention_mask"]})
    # concatenate 0 and 1 to input_ids and attention_mask
    combined_ds = combined_ds.map(lambda x: {"input_ids": x["input_ids0"] + x["input_ids1"], 
                                          "attention_mask": x["attention_mask0"] + x["attention_mask1"],})
    
    if save:
        combined_ds.save_to_disk(f"datasets/{outname}")
    return combined_ds

