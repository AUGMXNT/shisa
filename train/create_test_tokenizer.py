import gc
import glob
import os
import random
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer

# Select a random sampling of 3 parquet files.
paths = list(map(str, glob.glob("/mnt/data/datasets/izumi-lab_mc4-ja-filter-ja-normal/data/*.parquet")))
random.shuffle(paths)
dataset = concatenate_datasets([Dataset.from_parquet(path) for path in paths[0:2]])
print(dataset)
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
gc.collect()

# Yield dataset in batches.
batch_size = 1000
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
        gc.collect()

# Train a new mistral tokenizer from the dataset.
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/models/mistralai_Mistral-7B-v0.1")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=65536)
new_tokenizer.save_pretrained("/home/ubuntu/tokenizers/mistral-7b-mc4-tokenizer")
