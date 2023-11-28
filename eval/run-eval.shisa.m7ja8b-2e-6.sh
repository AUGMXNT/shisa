MODEL='/mnt/data/shisa/shisa-7b-v1-2e-6'
LOG='logs/shisa-7b-v1-2e-6'

export CUDA_VISIBLE_DEVICES=1

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
