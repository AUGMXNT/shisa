MODEL='/mnt/data/shisa/mistral-7b-ja-v0.1'
LOG='logs/shisa-mistral-7b-ja-v0.1'

export CUDA_VISIBLE_DEVICES=0

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
	tokenizer.use_fast=true \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
