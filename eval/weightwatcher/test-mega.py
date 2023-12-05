import torch
import weightwatcher as ww
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


base_model = AutoModelForCausalLM.from_pretrained(
    "/models/augmxnt_mistral-7b-ja-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "/models/shisa/shisa-mega-7b-v1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

watcher = ww.WeightWatcher()
details = watcher.analyze(model=model, base_model=base_model)
print(details)
summary = watcher.get_summary(details)
details.to_feather('mega-v1.feather')
details.to_csv('mega-v1.csv')
