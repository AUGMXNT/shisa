#!/bin/bash
export WANDB_API_KEY=""
export WANDB_ENTITY="augmxnt"
export WANDB_PROJECT="shisa"
export WANDB_RUN_ID='qwen-qwen-2'

export OMP_NUM_THREADS=12 

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="Qwen/Qwen-14B-Chat" # local model
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="/mnt/data/datasets/jondurbin_ultraboros-en-ja-v0.1/translated-airo-ultra.json"
DATA="qwen.finetune.dataset.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Remember to use --fp16 instead of --bf16 due to autogptq
torchrun $DISTRIBUTED_ARGS finetune-qlora.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 4096 \
    --use_lora \
    --q_lora \
    --gradient_checkpointing \
    --deepspeed Qwen/finetune/ds_config_zero2.json
