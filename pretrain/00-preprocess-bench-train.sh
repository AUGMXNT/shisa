#!/bin/bash

set -eux
git clone https://github.com/llm-jp/llm-jp-eval
export PYTHONPATH=$(pwd)/llm-jp-eval/src
pip install xmltodict
python llm-jp-eval/scripts/preprocess_dataset.py --dataset-name all --output-dir /mnt/data/llm-eval-train-ds
