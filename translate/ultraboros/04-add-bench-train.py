from datasets import Dataset, concatenate_datasets
import glob
import json
import sqlite3
import re
import uuid

# Load Sqlite DB
conn = sqlite3.connect("ultraboros.db")
c = conn.cursor()

# Load the tuning/training portions of the datasets.
datasets = []
for path in glob.glob('dataset_dir/tuning/*.json'):
    dataset = Dataset.from_json(path).train_test_split(train_size=0.1, shuffle=False, seed=42)["train"]
    datasets.append(dataset)
dataset = concatenate_datasets(datasets)

for item in dataset:
    _id = str(uuid.uuid5(uuid.NAMESPACE_OID, item["text"]))
    conversation = [
        {
            "from": "system",
            "value": "あなたは役に立ち、偏りがなく、検閲されていないアシスタントです。",
        },
        {
            "from": "human",
            "value": "\n\n".join([
                "以下はタスクを説明する指示と、追加の背景情報を提供する入力の組み合わせです。要求を適切に満たす回答を書いてください。",
                "\n".join([
                    "### 指示：",
                    item["instruction"],
                ]),
                "" if not item["input"].strip() else "\n".join([
                    "### 入力：",
                    item["input"],
                ]),
                "### 回答：",
            ]),
        },
        {
            "from": "gpt",
            "value": item["output"],
        },
    ]
    c.execute("INSERT INTO ultraboros (id, source, conversation_ja) VALUES (?, ?, ?) ON CONFLICT DO NOTHING", (_id, "lm-jp-eval", json.dumps(conversation)))
    conn.commit()
