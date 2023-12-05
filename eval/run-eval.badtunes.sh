#!/bin/bash


MODEL_PATHS=(
    "/mnt/data/shisa/shisa-bad-7b-v1"
)


for MODEL in "${MODEL_PATHS[@]}"; do
    LOG="logs/$(basename "$MODEL")"

    time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
done
