import json
import sys
import time
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from   vllm import LLM, SamplingParams

# Model
model = 'allsources-7b-ja-v0.4'
models = {
    'allsources-7b-ja-v0.4': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/allsources-7b-ja-v0.4',
        'format': 'llama-2',
    },
}
MODEL = models[model]['model']
FORMAT = models[model]['format']

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

if FORMAT == 'llama-2':
	tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
elif FORMAT == 'tess':
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{message['role'].upper() + ': ' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
else:
	# default to chatml
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Lets load our tasks

# Sample prompts.
with open('tasks.json', 'r') as file:
    tasks = json.load(file)


chat_token_ids = []
for task in tasks:
    if task['user_ja']: 
        chat = []
        chat.append({'role': 'system', 'content': task['prompt_ja']})
        chat.append({'role': 'user', 'content': task['user_ja']})
        tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        chat_token_ids.append(tokens)

    if task['user_en']: 
        chat = []
        chat.append({'role': 'system', 'content': task['prompt_en']})
        chat.append({'role': 'user', 'content': task['user_en']})
        tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        chat_token_ids.append(tokens)

# Create a sampling params object.
sampling_params = SamplingParams(
    max_tokens=500,
    temperature=0.8, 
    # top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.15,
    skip_special_tokens=True,
)

# Create an LLM.
start = time.time()
llm = LLM(model=MODEL, tensor_parallel_size=8)
print(f"loading took: {time.time()-start:.2f} s")
print()


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start = time.time()
outputs = llm.generate(prompt_token_ids=chat_token_ids, sampling_params=sampling_params, use_tqdm=True)
# Print the outputs.
for output in outputs:
    prompt = tokenizer.decode(output.prompt_token_ids, skip_special_tokens=True)
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print()
print(f"inferencing took: {time.time()-start:.2f} s")
