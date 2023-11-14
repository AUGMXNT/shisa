import gc
import os
import glob
import random
import json
from copy import deepcopy
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Load a sampling of the JA dataset sample.
dataset = concatenate_datasets(
    [
        Dataset.from_json(path)
        for path in glob.glob('/mnt/data/madlad-ja-sampled/*.jsonl')
        if os.stat(path).st_size
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
new_tokenizer.save_pretrained("/mnt/data/mistral-7b-tokenizer-ja-temp")

# Load the original tokenizer.
snapshot_download(repo_id='mistralai/Mistral-7B-v0.1', local_dir='mistral-7b', cache_dir='.cache', local_dir_use_symlinks=False, allow_patterns=['tokenizer.json'])
with open("mistral-7b/tokenizer.json") as f:
    original = json.load(f)

# Load the updated tokenizer we just trained.
with open("mistral-7b-tokenizer-ja-temp/tokenizer.json") as f:
    append = json.load(f)

def merge_tokenizer(data1: dict, data2: dict):
    vocab1 = data1["model"]["vocab"]
    vocab2 = data2["model"]["vocab"]

    merges1 = data1["model"]["merges"]
    merges2 = data2["model"]["merges"]

    # 出力用の変数を定義
    vocab_out = deepcopy(vocab1)
    data_out = deepcopy(data1)

    # merge前の最大idxを取得
    idx = max(vocab_out.values())

    # vocab2のうちvocab1にないものを、idxをインクリメントしつつvocab_outに追加
    for token in vocab2.keys():
        if token not in vocab1:
            idx += 1
            vocab_out[token] = idx

    # vocab_out中の全てのtokenについて、それをある位置で区切ったときの左右それぞれの要素がいずれもvocab_outに含まれる場合、merges_outに追加
    # 参考: https://github.com/huggingface/transformers/pull/17199
    merges_out = []
    for candidate, piece_id in vocab_out.items():
        for i in range(1, len(candidate)):
            left, right = candidate[:i], candidate[i:]

            left_id = vocab_out.get(left, None)
            right_id = vocab_out.get(right, None)

            if left_id is not None and right_id is not None:
                merges_out += [f"{left} {right}"]

    data_out["model"]["vocab"] = vocab_out
    data_out["model"]["merges"] = merges_out

    tokenizer.save_pretrained("mistral-7b-tokenizer-ja")
    with open("mistral-7b-tokenizer-ja/tokenizer.json", "w") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)

# 上で定義した関数により、元のtokenizerと追加したいtokenizerをmerge
merge_tokenizer(data1=original, data2=append)
