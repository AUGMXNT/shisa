MODEL='/mnt/data/models/01-ai_Yi-34B-Chat'
LOG='logs/01-ai_Yi-34B-Chat'

# 4 cards, let's go
export CUDA_VISIBLE_DEVICES=4,5,6,7

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
