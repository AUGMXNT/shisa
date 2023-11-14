import gc
import os
import glob
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer

# Load a sampling of the JA dataset sample.
dataset = concatenate_datasets(
    [
        Dataset.from_json(path)
        for path in glob.glob('/mnt/data/madlad-ja-sampled/*.jsonl')
    ]
).shuffle(seed=42).select(range(1000000))

# Yield dataset in batches.
batch_size = 1000
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
        gc.collect()

# Train a new mistral tokenizer from the dataset.
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=32000, max_token_length=8)
new_tokenizer.save_pretrained("/mnt/data/mistral-7b-tokenizer-ja")
