# MODEL='/data/models/Qwen_Qwen-14B-Chat-llamafied'
# TOKENIZER='/data/models/Qwen_Qwen-14B-Chat'
# LOG='logs/qwen-qwen-14b-chat-llamafied'

MODEL='/data/models/Qwen_Qwen-14B-llamafied'
TOKENIZER='/data/models/Qwen_Qwen-14B'
LOG='logs/qwen-qwen-14b-llamafied'

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path="$MODEL" \
tokenizer.pretrained_model_name_or_path="$TOKENIZER" \
target_dataset=all \
log_dir="$LOG" \
dataset_dir=llm-jp-eval/dataset_dir/evaluation/test

MODEL='/data/models/Qwen_Qwen-14B'
TOKENIZER='/data/models/Qwen_Qwen-14B'
LOG='logs/qwen-qwen-14b-2'

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path="$MODEL" \
tokenizer.pretrained_model_name_or_path="$TOKENIZER" \
target_dataset=all \
log_dir="$LOG" \
dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
