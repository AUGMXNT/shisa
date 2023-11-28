MODEL='/mnt/data/models/Qwen_Qwen-14B-Chat'
MODEL='/mnt/data/shisa/ultraboros-14b-ja-v0_3'
LOG='logs/ultraboros-14b-ja-v0_3'

export CUDA_VISIBLE_DEVICES=0,1

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        model.trust_remote_code=True \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test

        # No need, fits in 138GiB - *should* fit in 7 cards...
        # model.load_in_8bit=true \
