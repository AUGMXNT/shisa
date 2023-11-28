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


chat = "Happy birthday "
inputs = tokenizer.encode(chat, return_tensors="pt")
first_param_device = next(model.parameters()).device
inputs = inputs.to(first_param_device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=90,
        temperature=0.7,
        repetition_penalty=1.15,
        top_p=0.95,
        do_sample=True,
        streamer=streamer,
    )

# Add just the new tokens to our chat
new_tokens = outputs[0, inputs.size(1):]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
