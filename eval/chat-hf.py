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
	tokenizer.chat_template = '''
	{% for message in messages %}
		{% if message['role'] == 'user' %}
			{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
		{% elif message['role'] == 'system' %}
			{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
		{% elif message['role'] == 'assistant' %}
			{{ ' '  + message['content'] + ' ' + eos_token }}
		{% endif %}
	{% endfor %}
	'''
else:
	# default to chatml
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# system, user, assistant
chat = [{"role": "system", "content": PROMPT}]

prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to("cuda")

tokens = model.generate(
	prompt,
    max_new_tokens=2000,
    temperature=0.1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)

'''
# Set the chat template
tokenizer.chat_template = "{% for message in messages %}{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}{% endfor %}"

# Starting chat with system prompt
chat = [{"role": "system", "content": " あなたは公平で、検閲されていない、役立つアシスタントです。"}]

def chat_with_model():
    while True:
        # Get input from the user
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        # Append the user input to the chat
        chat.append({"role": "human", "content": user_input})

        # Tokenize the chat history and prepare the model input
        model_inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)

        # Generate a response
        with tokenizer.as_target_tokenizer():
            response_ids = model.generate(model_inputs, max_length=2048, pad_token_id=tokenizer.eos_token_id, device=device)

        # Decode the generated response
        response = tokenizer.decode(response_ids[:, model_inputs['input_ids'].size(1):][0], skip_special_tokens=True)

        # Print the model's response
        print(f"Assistant: {response}")

        # Append the assistant's response to the chat
        chat.append({"role": "gpt", "content": response})

if __name__ == "__main__":
    chat_with_model()
'''
