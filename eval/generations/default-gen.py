import argparse
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        print('---')
        self.print_help()
        sys.exit(2)

# parser = argparse.ArgumentParser(description="Default Reply Benchmarker")
parser = CustomArgumentParser()
parser.add_argument('-m', required=True, help='Require a model path/name')
parser.add_argument('-t', default=0.1, help='Set a temperature (defaults to 0.1)')
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Error: {e}")
    parser.print_help()
    exit(1)

model_name = args.m
# model_name = "augmxnt/shisa-7b-v1"
# model_name = "/mnt/data/shisa/allsources-7b-ja-v0.4"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
streamer = TextStreamer(tokenizer, skip_prompt=True)

# llama-2 chat format prompt template is included in the tokenizer config, but reproduced here for convenience
tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"


tasks = json.load(open("tasks.json"))
responses = []
for task in tasks:
    if True:
    # if task["type"] == "benchmark" or task["type"] == "japanese" or "translation" in task["type"]:
        print(task)
        chat = []
    
        if task["user_ja"]:
            chat.append({"role": "system", "content": task["prompt_ja"]})
            chat.append({"role": "user", "content": task["user_ja"]})
        elif task["user_en"]:
            chat.append({"role": "system", "content": task["prompt_en"]})
            chat.append({"role": "user", "content": task["user_en"]})

        # apply_chat_template drops our starting token
        # inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
        no_encode = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer.encode(no_encode, return_tensors="pt")

        # For multi-GPU, find the device of the first parameter of the model
        first_param_device = next(model.parameters()).device
        inputs = inputs.to(first_param_device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=500,
                temperature=0.2,
                repetition_penalty=1.15,
                top_p=0.95,
                do_sample=True,
                streamer=streamer,
            )

        # Add just the new tokens to our chat
        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        chat.append({"role": "assistant", "content": response})

        responses.append(chat[2]["content"])
        print('===')
        print()
        print(chat[2]["content"])
        print()

# Write to JSON
name = model_name.split('/')[-1]
with open(f'{name}.json', 'w') as file:
    json.dump(responses, file)
