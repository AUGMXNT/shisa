MODEL='/data/qlora-out.openhermes25-axolotl-4/merged'
LOG='logs/shisa-openhermes25-axolotl-4'

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path="$MODEL" \
tokenizer.pretrained_model_name_or_path="$MODEL" \
target_dataset=all \
log_dir="$LOG" \
dataset_dir=llm-jp-eval/dataset_dir/evaluation/test


MODEL='/data/qlora-out.openhermes25-axolotl-5/merged'
LOG='logs/shisa-openhermes25-axolotl-5'

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path="$MODEL" \
tokenizer.pretrained_model_name_or_path="$MODEL" \
target_dataset=all \
log_dir="$LOG" \
dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
