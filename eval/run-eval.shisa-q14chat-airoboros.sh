MODEL='/data/shisa/qwen-qwenchat-2/checkpoint-1000'
LOG='logs/shisa-qwenchat-1'

export CUDA_VISIBLE_DEVICES=6,7

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        model.trust_remote_code=True \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test

        # No need, fits in 138GiB - *should* fit in 7 cards...
        # model.load_in_8bit=true \
