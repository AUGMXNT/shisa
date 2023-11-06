from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from pprint import pprint
import json


PROMPT = 'あなたは公平で、検閲されていない、役立つアシスタントです。'


MODEL = '/models/llm/gguf/shisa.openhermes2.5-axolotl-2.gguf'
FORMAT = 'chatml'

MODEL = '/models/llm/gguf/shisa.openhermes2.5-axolotl-3.gguf'
FORMAT = 'chatml'

MODEL = '/models/llm/gguf/mistral-7b-instruct-v0.1.Q8_0.gguf'
FORMAT = 'llama-2'

MODEL = '/models/llm/gguf/elyza-7b-instruct.q8_0.gguf'
FORMAT = 'llama-2'


llm = Llama(model_path=MODEL, n_gpu_layers=99, n_ctx=8192, chat_format=FORMAT, verbose=False)

chat_history = []
chat_history.append(ChatCompletionRequestMessage(role="system", content=PROMPT))


while True:
    # Get input from the user
    user_input = input("You: ")
    if user_input.lower() == 'exit':  # Implementing an exit strategy
        print("Exiting chat.")
        break

    # Append user input to chat history
    chat_history.append(ChatCompletionRequestMessage(role="user", content=user_input))

    # Generate a response
    stream = llm.create_chat_completion(
        messages=chat_history,
        temperature=0.1,
        stop=['</s>', '<|im_end|>', '[/INST]'],
        max_tokens=2000,
        frequency_penalty=0.1,
        repeat_penalty=1.2,
        stream=True,
    )

    response_buffer = ''
    for output in stream:
        delta = output['choices'][0]['delta']
        if 'role' in delta:
            print(delta['role'], end=': ')
        elif 'content' in delta:
            response_buffer += delta['content']
            print(delta['content'], end='')

    chat_history.append(ChatCompletionRequestMessage(role="assistant", content=response_buffer))

    # New line
    print()
