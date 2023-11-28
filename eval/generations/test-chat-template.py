import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_name = "/data/shisa/shisa-7b-v1-2e-6"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
streamer = TextStreamer(tokenizer, skip_prompt=True)

# llama-2 chat format prompt template is included in the tokenizer config, but reproduced here for convenience
# tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- bos_token + '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"

# You are an avid Pokemon fanatic.
prompt = "あなたは熱狂的なポケモンファンです。"
chat = [{"role": "system", "content": prompt}]

# Who is the most powerful Pokemon? Explain your choice.
user_input = "最強のポケモンは誰ですか？その選択理由を説明してください。"
chat.append({"role": "user", "content": user_input})


# Generate - add_generation_prompt to make sure it continues as assistant
inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
print("### apply_chat_template tokens:")
print(inputs)

# debug -works!
no_encode = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
print(no_encode)
inputs = tokenizer.encode(no_encode, return_tensors="pt")
print("### tokenizer.encode tokens:")
print(inputs)

# For multi-GPU, find the device of the first parameter of the model
first_param_device = next(model.parameters()).device
inputs = inputs.to(first_param_device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        temperature=0.8,
        repetition_penalty=1.15,
        top_p=0.95,
        do_sample=True,
        streamer=streamer,
    )

# Add just the new tokens to our chat
new_tokens = outputs[0, inputs.size(1):]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
chat.append({"role": "assistant", "content": response})

print(outputs)
print(tokenizer.decode(outputs[0]))
