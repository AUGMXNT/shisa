MODEL='/models/llm/hf/rinna_youri-7b-instruction'
LOG='logs/youri-7b-instruction'

export CUDA_VISIBLE_DEVICES=0

time python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
	model.pretrained_model_name_or_path="$MODEL" \
	tokenizer.pretrained_model_name_or_path="$MODEL" \
	target_dataset=all \
	log_dir="$LOG" \
	dataset_dir=llm-jp-eval/dataset_dir/evaluation/test
	# target_dataset=jamp \
