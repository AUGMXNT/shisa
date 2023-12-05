#!/bin/bash


MODEL_PATHS=(
    # "models/shisa/shisa-mega-7b-v1" \
    # "/models/shisa/shisa-mega-7b-v1.1" \
    # "/models/shisa/shisa-mega-dpo-7b-v1" \
    # "/mnt/data/shisa/shisa-mega-dpo-7b-v1.1" \
    # "/mnt/data/shisa/shisa-mega-7b-v1.2" \
    # "/mnt/data/shisa/shisa-gamma-7b-v1" \
    "/mnt/data/shisa/shisa-mega-dpo-7b-v1.1-zeroed-extra"
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
