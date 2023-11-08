import torch
from   transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'


MODEL = 'qlora-out.openhermes25-axolotl-3/merged'
FORMAT = 'chatml'

MODEL = "/models/llm/hf/stabilityai_japanese-stablelm-base-beta-7b"
FORMAT = 'llama-2'


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
	MODEL, 
	torch_dtype=torch.float16, 
	low_cpu_mem_usage=True, 
	device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# this is for reproducibility.
# feel free to change to get different result
seed = 42  
torch.manual_seed(seed)


if FORMAT == 'llama-2':
	tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
else:
	# default to chatml
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"



# system, user, assistant
chat = [{"role": "system", "content": PROMPT}]


def chat_with_model():
    while True:
        # Get input from the user
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        # Append the user input to the chat
        chat.append({"role": "user", "content": user_input})

        # Generate
        text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to("cuda")
        tokens = model.generate(
            input,
            max_new_tokens=2000,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)

        print(f"Assistant: {output}")

        # Append the assistant's response to the chat
        chat.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_model()
