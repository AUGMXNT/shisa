MODEL='/mnt/data/models/migtissera_Tess-M-v1.0'
LOG='logs/migtissera_Tess-M-v1.0'

# 4 cards, let's go
export CUDA_VISIBLE_DEVICES=0,1,2,3

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
        model.pretrained_model_name_or_path="$MODEL" \
        tokenizer.pretrained_model_name_or_path="$MODEL" \
        target_dataset=all \
        log_dir="$LOG" \
        dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
