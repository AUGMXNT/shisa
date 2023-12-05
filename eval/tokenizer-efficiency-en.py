# Download a single sample file from CulturaX dataset.
from datasets import Dataset
import glob
from epochraft import CheckpointableDataset
from transformers import AutoTokenizer, LlamaTokenizer
import pandas as pd
from huggingface_hub import snapshot_download
import sys

# Download a single sample file from CulturaX dataset.
snapshot_download(repo_id='uonlp/CulturaX', local_dir='CulturaX', allow_patterns=['*en_part_00004.parquet'], repo_type='dataset')
dataset = Dataset.from_parquet('CulturaX/en/en_part_00004.parquet')
dataset.to_json('test.jsonl')

# For some reason Qwen tokenizer doesn't load from the function, fine whatever...
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-14b', use_fast=True, trust_remote_code=True)
dataset = CheckpointableDataset.from_files("test.jsonl").tokenize(tokenizer, parallel=False).take(50000)
n_chars = 0
n_tokens = 0
for sample in dataset:
    n_chars += len(sample["text"])
    n_tokens += len(sample["input_ids"])
print(f"Compression rate: {n_chars / n_tokens} chars / token ({n_chars} / {n_tokens})")
print("Vocab Size:", tokenizer.vocab_size)
print()

def evaluate(tokenizer):
    dataset = CheckpointableDataset.from_files("test.jsonl").tokenize(tokenizer, parallel=False).take(50000)
    n_chars = 0
    n_tokens = 0
    for sample in dataset:
        n_chars += len(sample["text"])
        n_tokens += len(sample["input_ids"])
    try:
        print(f"Compression rate: {n_chars / n_tokens} chars / token ({n_chars} / {n_tokens})")
        return n_chars / n_tokens
    except:
        return 0


TOKENIZERS = [
    ("llm-jp/llm-jp-13b-v1.0", AutoTokenizer, "llm-jp-13b-v1.0 (LLM-jp)"),
    ("EleutherAI/gpt-neox-20b", AutoTokenizer, "weblab-10b (Matsuo Lab)"),
    ("meta-llama/Llama-2-7b-hf", AutoTokenizer, "Japanese-Llama-2-7b (ELYZA)"),
    ("elyza/ELYZA-japanese-Llama-2-7b-fast", AutoTokenizer, "Japanese-Llama-2-7b-fast (ELYZA)"),
    ("novelai/nerdstash-tokenizer-v1", LlamaTokenizer, "Japanese StableLM Alpha (Stability AI)"),
    ("rinna/japanese-gpt-neox-3.6b", AutoTokenizer, "Japanese-GPT-NeoX-3.6B (Rinna)"),
    ("rinna/bilingual-gpt-neox-4b", AutoTokenizer, "Bilingual-GPT-NeoX-4B (Rinna)"),
    ("rinna/youri-7b", AutoTokenizer, "Youri 7B (Rinna)"),
    ("line-corporation/japanese-large-lm-3.6b", AutoTokenizer, "Japanese LargeLM (LINE)"),
    ("cyberagent/open-calm-7b", AutoTokenizer, "OpenCALM (CyberAgent)"),
    ("stabilityai/japanese-stablelm-base-gamma-7b", AutoTokenizer, "Japanese StableLM Gamma (Stability AI)"),
    ("stabilityai/japanese-stablelm-base-ja_vocab-beta-7b", AutoTokenizer, "Japanese StableLM Beta JAVocab (Stability AI)"),
    ("cyberagent/calm2-7b", AutoTokenizer, "CALM2-7B (CyberAgent)"),
    ("/models/shisa-7b-v1-5e-7", AutoTokenizer, "Shisa 7B (AUGMXNT)"),
    ("/models/mistral-7b-ja-v0.0", AutoTokenizer, "mistral-7b-ja-v0.0 (AUGMXNT)"),
]

def generate_row(tokenizer_url, tokenizer_cls, tokenizer_name):
    print(tokenizer_name)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_url)
    if 'custom-tokenizer' in tokenizer_url:
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_url, use_fast=True)
    return {
        "LLM": tokenizer_name,
        "Tokenizer": tokenizer_url,
        "Vocab Size": tokenizer.vocab_size,
        "Avg Char/Token": evaluate(tokenizer)
    }

result = pd.DataFrame(
    [
        generate_row(*args)
        for args in TOKENIZERS
    ]
)
print(result)
result.to_markdown('tokenizer-eval-en.md')
