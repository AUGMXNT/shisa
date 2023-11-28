MODEL='/data/models/meta-llama_Llama-2-7b-hf'
LOG='logs/llama2-7b-3'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path="$MODEL" \
tokenizer.pretrained_model_name_or_path="$MODEL" \
target_dataset=all \
log_dir="$LOG" \
dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
