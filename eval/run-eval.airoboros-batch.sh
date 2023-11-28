# sleep like 3h+
# echo "sleeping 3h20m"
# sleep 12000

# mv logs logs.llama2-70b
# mkdir logs

time python scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
tokenizer.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
dataset_dir=/home/ubuntu/llm-jp-eval/dataset_dir/evaluation/test
mv logs logs.airoboros
mkdir logs

time python scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
model.load_in_8bit=true \
tokenizer.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
dataset_dir=/home/ubuntu/llm-jp-eval/dataset_dir/evaluation/test
mv logs logs.airoboros-8bit
mkdir logs

time python scripts/evaluate_llm.py -cn config.yaml \
model.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
model.load_in_4bit=true \
tokenizer.pretrained_model_name_or_path=/mnt/data/models/jondurbin_airoboros-l2-c70b-3.1.2 \
dataset_dir=/home/ubuntu/llm-jp-eval/dataset_dir/evaluation/test
mv logs logs.airoboros-4bit
mkdir logs
