#!/bin/bash 

benchmark() {
    LOG_DIR=$(basename "$1")
    mkdir -p "$LOG_DIR"
    echo "$1"
    
    BATCH_SIZE=2
    EXTRA_ARGS="use_flash_attention_2=False,dtype=float16,load_in_8bit=True"

    
    # Winogrande: 5-shot, winogrande (acc)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,$EXTRA_ARGS" --tasks=winogrande --num_fewshot=5 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR" > "$LOG_DIR/winogrande.txt"

}

### Mistral Base model comparison
benchmark /models/llm/hf/mistralai_Mistral-7B-v0.1
# benchmark /mnt/data/shisa/augmxnt_shisa-7b-ja-v0.1
# benchmark /models/llm/shisa/shisa-base-7b-v1

# benchmark /data/_base_compare/stabilityai_japanese-stablelm-base-gamma-7b
# benchmark /mnt/data/_base_compare/mistral-7b-v0.1


### JA Base comparison
# benchmark /models/llm/shisa/_base-compare/cyberagent_calm2-7b
# benchmark /models/llm/shisa/_base-compare/elyza_ELYZA-japanese-Llama-2-7b-fast
# benchmark /models/llm/shisa/_base-compare/rinna_youri-7b
# benchmark /models/llm/shisa/_base-compare/llm-jp_llm-jp-13b-v1.0

# benchmark /mnt/data/shisa/augmxnt_mistral-7b-ja-v0.1
# benchmark /mnt/data/models/stabilityai_japanese-stablelm-instruct-beta-70b
