from   pprint import pprint
import torch
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from   transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids


# fast testing
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/data/models/PY007_TinyLlama-1.1B-Chat-v0.3'
FORMAT = 'chatml'

# Qwen-14B
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/models/llm/hf/Qwen_Qwen-14B-Chat'
FORMAT = 'chatml'

# airboros 70b
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。日本語のみで返信してください。'
MODEL = "/data/models/jondurbin_airoboros-l2-c70b-3.1.2"
FORMAT = 'llama-2'

# shisa-openhermes25-axolotl-4 (airoboros)
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/home/ubuntu/shisa/train/axolotl/qlora-out.openhermes25-axolotl-4/merged'
FORMAT = 'chatml'

# shisa-openhermes25-axolotl-5 (ultraboros)
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/home/ubuntu/shisa/train/axolotl/qlora-out.openhermes25-axolotl-5/merged'
FORMAT = 'chatml'

# shisa-qwen14b-qwen-2 (ultraboros)
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = 'merged-model'
TOKEN = 'merged-model'
FORMAT = 'chatml'

# llamafied
PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/data/models/Qwen_Qwen-14B-llamafied'
TOKEN = 'merged-model'
FORMAT = 'chatml'

PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'
MODEL = '/data/models/Qwen_Qwen-14B-Chat-llamafied'
TOKEN = 'merged-model'
FORMAT = 'chatml'


tokenizer = AutoTokenizer.from_pretrained(
    TOKEN,
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True,
    errors="ignore"
)


try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="auto",
        trust_remote_code=True,
        eos_token_id = tokenizer.eos_token_id, 
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        eos_token_id = tokenizer.eos_token_id, 
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

# model.generation_config = GenerationConfig.from_pretrained(MODEL, pad_token_id=tokenizer.pad_token_id)


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

        # Generate - add_generation_prompt to make sure it continues as assistant
        inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")

        # For multi-GPU, find the device of the first parameter of the model
        first_param_device = next(model.parameters()).device
        inputs = inputs.to(first_param_device)


        print('Assistant: ', end='')
        # We'll try flash attention
        # skips gradients if Tensor.backward() won't be called...
        # with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            '''
            stop_word_ids = []
            stop_word_ids.extend(get_stop_words_ids(
                model.generation_config.chat_format, tokenizer
            ))
            print(stop_word_ids)
            '''
		
            outputs = model.generate(
                inputs,
                # stop_words_ids = stop_word_ids,
                max_new_tokens=50,
                temperature=1.0,
                repetition_penalty=1.18,
                top_p=0.8,
                do_sample=True,
                streamer=streamer
            )

        # Add just the new tokens to our chat
        new_tokens = outputs[0, inputs.size(1):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # if not streamer print(response)
        chat.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_model()
