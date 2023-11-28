MODEL='/mnt/data/models/stabilityai_japanese-stablelm-base-beta-70b'
LOG='logs/stabilityai_japanese-stablelm-base-beta-70b'

# 7 cards, let's go
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test

        # No need, fits in 138GiB - *should* fit in 7 cards...
        # model.load_in_8bit=true \
