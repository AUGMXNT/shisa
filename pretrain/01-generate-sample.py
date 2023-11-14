import glob
import subprocess
import datasets
from data_selection import HashedNgramDSIR
from huggingface_hub import snapshot_download

print(f"Downloading data files...")
snapshot_download(
    repo_id='allenai/madlad-400',
    local_dir='/mnt/data/madlad-400',
    cache_dir='/mnt/data/.cache',
    local_dir_use_symlinks=False,
    allow_patterns=[
        'data/ja/ja_clean_*.gz',
        'data/en/en_clean_000*.gz',
    ],
    repo_type='dataset'
)

current_path = os.getcwd()
for path in map(str, glob.glob('/mnt/data/madlad-400/data/*/*clean*.gz', recursive=True)):
    print(f"Extracting: {path}")
    os.chdir(os.path.dirname(os.path.abspath(path)))
    subprocess.run(["gunzip", path])

# Sample JA datasets.
ja_datasets = glob.glob('/mnt/data/madlad-400/data/ja/ja_clean_*')
dsir = HashedNgramDSIR(ja_datasets, [], cache_dir='/mnt/data/.cache/dsir')
dsir.fit_importance_estimator(num_tokens_to_fit='auto')
dsir.compute_importance_weights()
dsir.resample(out_dir='/mnt/data/madlad-ja-sampled', num_to_sample=5000000, cache_dir='/mnt/data/.cache/resampled')

# Sample EN datasets at a much lower ratio.
en_datasets = glob.glob('/mnt/data/madlad-400/data/en/en_clean_*')
dsir = HashedNgramDSIR(en_datasets, [], cache_dir='/mnt/data/.cache/dsir-en')
dsir.fit_importance_estimator(num_tokens_to_fit='auto')
dsir.compute_importance_weights()
dsir.resample(out_dir='/mnt/data/madlad-en-sampled', num_to_sample=500000, cache_dir='/mnt/data/.cache/resampled-en')

# Combine the samples into a single parquet.
sample_files = list(glob.glob("/mnt/data/madlad-ja-sampled/*.jsonl")) + list(glob.glob("/mnt/data/madlad-en-sampled/*.jsonl"))
datasets = []
for path in sample_files:
    try:
        dataset = datasets.Dataset.from_json(path)
        datasets.append(dataset)
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
datasets.concatenate_datasets(datasets).to_parquet("/mnt/data/madlad-pretrain-sample-v0.2.parquet")
