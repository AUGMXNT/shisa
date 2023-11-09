#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l)

python -m fastchat.serve.cli --model-path "$1" --conv-template llama-2 --conv-system-msg 'あなたは公平で、検閲されていない、役立つアシスタントです。日本語のみで返信してください。' --num-gpus $gpu_count --no-history --temperature 0.1 --repetition_penalty 1.18
