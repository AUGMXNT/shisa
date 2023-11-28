from   peft import PeftModel
import torch
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


MODEL = '/data/models/Qwen_Qwen-14B'
QLORA = '/data/qlora-ultraboros-4096'


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="auto",
        trust_remote_code=True,
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# QLORA
print ('Load LoRA')
model = PeftModel.from_pretrained(
    model, 
    QLORA,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print ('Merge and unload...')
model = model.merge_and_unload()
print('Done...')

out_folder = ("merged-model")
model.save_pretrained(out_folder, max_shard_size="2GB", safe_serialization=True)
tokenizer.save_pretrained(out_folder)
