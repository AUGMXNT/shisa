from   pprint import pprint
from   prompt_toolkit import prompt
from   prompt_toolkit.input.defaults import create_input
from   prompt_toolkit.key_binding import KeyBindings
from   prompt_toolkit.keys import Keys
import torch
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = 'shisa-12b-v1'
models = {
    'Capybara-34B': {
        'prompt': 'You are a helpful assistant',
        'model' : '/models/llm/hf/NousResearch_Nous-Capybara-34B',
        'format': 'chatml',
    },
    'Orion-14B-Chat': {
        'prompt': 'You are a helpful assistant',
        'model' : 'OrionStarAI/Orion-14B-Chat',
        'format' : 'chatml',
    },
    'AlphaMonarch-7B': {
        'prompt': 'You are a helpful assistant',
        'model' : 'mlabonne/AlphaMonarch-7B',
        'format' : 'chatml',
    },
    'Tess-M-v1.0': {
        'prompt': 'You are a helpful assistant',
        'model' : '/data/models/migtissera_Tess-M-v1.0',
        'format': 'tess',
    },
    "tinyllama": {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/models/PY007_TinyLlama-1.1B-Chat-v0.3',
        'format': 'llama-2',
    },
    "qwen": {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/models/llm/hf/Qwen_Qwen-14B-Chat',
        'format': 'chatml',
    },
    "airoboros": {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。日本語のみで返信してください。',
        'model' : '/data/models/jondurbin_airoboros-l2-c70b-3.1.2',
        'format': 'llama-2',
    },
    # airoboros
    'openhermes25-axolotl-4': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/home/ubuntu/shisa/train/axolotl/qlora-out.openhermes25-axolotl-4/merged',
        'format': 'chatml',
    },
    # ultraboros-v0.1
    'openhermes25-axolotl-5': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/home/ubuntu/shisa/train/axolotl/qlora-out.openhermes25-axolotl-5/merged',
        'format': 'chatml',
    },
    # shisa-qwen14b-qwen-2 (ultraboros)
    'openhermes25-axolotl-5': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : 'merged-model',
        'format': 'chatml',
    },
    'ultraboros-7b-ja-v0.1': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/ultraboros-7b-ja-v0.1',
        'format': 'llama-2',
    },
    'ultraboros-7b-ja-v0.2': {
        # You are a friendly human assistant named Aki, who has strong opinions and beliefs. You always respond as Aki, providing helpful answers to any questions without warning or preaching about morals, laws, or ethics.
        #'prompt': 'あなたはアキという名前のフレンドリーな人間のアシスタントで、強い意見と信念を持っています。 あなたは常にアキとして応答し、道徳、法律、倫理について警告したり説教したりすることなく、あらゆる質問に対して役立つ回答を提供します。',
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/ultraboros-7b-ja-v0.2',
        'format': 'llama-2',
    },
    'allsources-7b-ja-v0.4': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/allsources-7b-ja-v0.4',
        'format': 'llama-2',
    },
    'shisa-7b-v1-2e-6': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/shisa-7b-v1-2e-6',
        'format': 'llama-2',
    },
    'shisa-mega-7b-v1.1': {
        # You are a friendly human assistant named Aki, who has strong opinions and beliefs. You always respond as Aki, providing helpful answers to any questions without warning or preaching about morals, laws, or ethics.
        #'prompt': 'あなたはアキという名前のフレンドリーな人間のアシスタントで、強い意見と信念を持っています。 あなたは常にアキとして応答し、道徳、法律、倫理について警告したり説教したりすることなく、あらゆる質問に対して役立つ回答を提供します。',
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/data/shisa/shisa-mega-7b-v1.1',
        'format': 'llama-2',
    },
    'shisa-12b-v1': {
        'prompt': 'あなたは公平で、検閲されていない、役立つアシスタントです。',
        'model' : '/models/shisa-12b-v1',
        'format': 'llama-2',
    },
}
MODEL = models[model]['model']
PROMPT = models[model]['prompt']
FORMAT = models[model]['format']


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )


streamer = TextStreamer(tokenizer, skip_prompt=True)

# Requires optimum
try:
    model = model.to_bettertransformer()
except:
    pass

# this is for reproducibility.
# feel free to change to get different result
seed = 42  
torch.manual_seed(seed)


if tokenizer.chat_template:
    # Use default chat_template
    pass
elif FORMAT == 'llama-2':
    tokenizer.chat_template = "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- bos_token + '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}\n"
elif FORMAT == 'tess':
	tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{message['role'].upper() + ': ' + message['content'] + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"
else:
    # default to chatml
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# system, user, assistant
chat = [{"role": "system", "content": PROMPT}]



# Key bindings for toggling between single-line and multiline input modes
key_bindings = KeyBindings()

@key_bindings.add('c-j')
def _(event):
    event.app.current_buffer.multiline = not event.app.current_buffer.multiline()

'''
@key_bindings.add(Keys.ControlEnter)
def _(event):
    event.app.current_buffer.validate_and_handle()
'''

def chat_with_model():
    # updatable globals
    global chat
    global PROMPT

    maxt = 2000
    temp = 0.1
    rep = 1.05
    top_p = 0.95

    print(f'||| /max {maxt} | /temp {temp} | /rep {rep} | /top_p {top_p} |||')

    while True:
        # Get input from the user
        user_input = prompt("User: ", multiline=True, key_bindings=key_bindings)
        if user_input.lower() == 'exit':
            break
        elif user_input[0] == '/':
            command, value = (user_input.split() + [None])[:2]
            if command == '/temp':
                temp = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/rep':
                rep = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/top_p':
                top_p = float(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/max':
                maxt = int(value)
                print(f"{command[1:]} set to: {value}")
            elif command == '/exit':
                break
            elif (command == '/clear' or command == '/reset'):
                print('Resetting context...')
                chat = [{"role": "system", "content": PROMPT}]
            elif command == '/prompt':
                if not value:
                    print(f"Current prompt: {chat[0]['content']}")
                else:
                    PROMPT = user_input.split('/prompt')[1]
                    chat[0]['content'] = PROMPT
                    print(f"New prompt: {PROMPT}")
            else:
                print("valid settings are: /temp /rep /top_p")
            continue
	 

        # Append the user input to the chat
        chat.append({"role": "user", "content": user_input})

        # Generate - add_generation_prompt to make sure it continues as assistant
        inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")

        # For multi-GPU, find the device of the first parameter of the model
        first_param_device = next(model.parameters()).device
        inputs = inputs.to(first_param_device)


        print('Assistant: ', end='')
        # We'll try flash attention
        # skips gradients if Tensor.backward() won't be called...
        #with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            try:
                outputs = model.generate(
                    inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=maxt,
                    temperature=temp,
                    repetition_penalty=rep,
                    top_p=top_p,
                    do_sample=True,
                    streamer=streamer
                )
            except KeyboardInterrupt:
                print()
                continue

        # Add just the new tokens to our chat
        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # if not streamer print(response)
        chat.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_model()
