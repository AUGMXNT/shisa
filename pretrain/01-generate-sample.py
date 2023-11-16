import glob
import subprocess
import datasets
import os
import re
from data_selection import HashedNgramDSIR
from huggingface_hub import snapshot_download
from loguru import logger
from tqdm import tqdm

logger.info("Downloading data files...")
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

logger.info("Extracting gzips...")
current_path = os.getcwd()
for path in map(
    str, glob.glob("/mnt/data/madlad-400/data/*/*clean*.gz", recursive=True)
):
    logger.info(f"Extracting: {path}")
    os.chdir(os.path.dirname(os.path.abspath(path)))
    subprocess.run(["gunzip", path])

# Sample JA datasets.
logger.info("Sampling JA datasets...")
with open("/mnt/data/madlad-400-ja-sample.jsonl", "a+") as outfile:
    ...
ja_datasets = glob.glob("/mnt/data/madlad-400/data/ja/ja_clean_*")
dsir = HashedNgramDSIR(
    ja_datasets,
    ["/mnt/data/madlad-400-ja-sample.jsonl"],
    cache_dir="/mnt/data/.cache/dsir",
)
dsir.fit_importance_estimator(num_tokens_to_fit="auto")
dsir.compute_importance_weights()
dsir.resample(
    out_dir="/mnt/data/madlad-ja-sampled",
    num_to_sample=2500000,
    cache_dir="/mnt/data/.cache/resampled",
)

# Sample EN datasets at a much lower ratio.
logger.info("Sampling EN datasets...")
with open("/mnt/data/madlad-400-en-sample.jsonl", "a+") as outfile:
    ...
en_datasets = glob.glob("/mnt/data/madlad-400/data/en/en_clean_*")
dsir = HashedNgramDSIR(
    en_datasets,
    ["/mnt/data/madlad-400-en-sample.jsonl"],
    cache_dir="/mnt/data/.cache/dsir-en",
)
dsir.fit_importance_estimator(num_tokens_to_fit="auto")
dsir.compute_importance_weights()
dsir.resample(
    out_dir="/mnt/data/madlad-en-sampled",
    num_to_sample=250000,
    cache_dir="/mnt/data/.cache/resampled-en",
)

# Load the various EN/JA files.
logger.info("Unifying dataset...")
sample_files = list(glob.glob("/mnt/data/madlad-ja-sampled/*.jsonl")) + list(
    glob.glob("/mnt/data/madlad-en-sampled/*.jsonl")
)
all_datasets = []
for path in sample_files:
    if os.stat(path).st_size:
        dataset = datasets.Dataset.from_json(path)
        all_datasets.append(dataset)

# Add in the JP training data from lm-eval-jp:
for path in glob.glob("/mnt/data/llm-eval-train-ds/tuning/*.json"):
    dataset = datasets.Dataset.from_json(path)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col != "text"]
    )
    all_datasets.append(dataset)

# Add in the *train* splits of some common benchmarks to hopefully reduce the risk of catastrophic forgetting.
mmlu_segments = [
    "high_school_european_history",
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]

# MMLU
logger.info("Loading MMLU training data...")
mmlu = datasets.concatenate_datasets(
    [
        datasets.load_dataset("lukaemon/mmlu", segment, split="train")
        for segment in mmlu_segments
    ]
)
training_data = []
for item in tqdm(mmlu):
    answer_keys = [key for key in item if re.match(r"^[A-Z]$", key)]
    text = "\n".join(
        [
            item["input"],
            "\n".join([f"{key}. {item[key]}" for key in answer_keys]),
            item["target"],
        ]
    )
    training_data.append({"text": text})

# DROP
logger.info("Loading DROP training data...")
for item in tqdm(datasets.load_dataset("drop", split="train")):
    text = "\n".join(
        [
            item["passage"],
            item["question"],
            ", ".join([val for val in item["answers_spans"]["spans"]]),
        ]
    )
    training_data.append({"text": text})

# Natural instructions.
logger.info("Loading natural instructions training data...")
for item in tqdm(
    datasets.load_dataset("Muennighoff/natural-instructions", split="train")
    .shuffle(seed=42)
    .select(range(50000))
):
    text = "\n".join(
        [
            item["definition"],
            item["inputs"],
            item["targets"],
        ]
    )
    training_data.append({"text": text})

# GSM8k
logger.info("Loading gsm8k training data...")
for item in datasets.load_dataset("gsm8k", "main", split="train"):
    text = "\n".join(
        [
            item["question"],
            item["answer"],
        ]
    )
    training_data.append({"text": text})

# P3
logger.info("Loading P3 training data...")
for item in tqdm(
    datasets.load_dataset("Muennighoff/P3", split="train")
    .shuffle(seed=42)
    .select(range(50000))
):
    text = "\n".join(
        [
            item["inputs"],
            item["targets"],
        ]
    )
    training_data.append({"text": text})

# winogrande
logger.info("Loading winogrande training data...")
for item in (
    tqdm(datasets.load_dataset("winogrande", "winogrande_xl", split="train"))
    .shuffle(seed=42)
    .select(range(20000))
):
    text = "\n".join(
        [
            item["sentence"],
            item["option1"] if str(item["answer"]) == "1" else item["option2"],
        ]
    )
    training_data.append({"text": text})

# ARC-Challenge.
logger.info("Loading ARC-Challenge training data...")
for item in datasets.load_dataset("ai2_arc", "ARC-Challenge", split="train"):
    text = "\n".join(
        [
            item["question"],
            "\n".join(
                [
                    f"{item['choices']['label'][idx]}. {item['choices']['text'][idx]}"
                    for idx in range(len(item["choices"]["label"]))
                ]
            ),
            item["answerKey"],
        ]
    )
    training_data.append({"text": text})

# Python.
for item in datasets.load_dataset("Vezora/Tested-188k-Python-Alpaca", split="train"):
    if (item["input"] or "").strip():
        continue
    text = "\n".join(
        [
            item["instruction"],
            item["output"],
        ]
    )
    training_data.append({"text": text})

# Add in the combined bench train samples.
all_datasets.append(datasets.Dataset.from_list(training_data))

# Combine everything.
datasets.concatenate_datasets(all_datasets).shuffle(seed=42).to_parquet(
    "/mnt/data/madlad-pretrain-sample-v0.2.parquet"
)
