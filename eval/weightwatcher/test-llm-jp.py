import torch
import weightwatcher as ww
from   transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


base_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/data/models/llm-jp_llm-jp-13b-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/data/models/llm-jp_llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

watcher = ww.WeightWatcher()
details = watcher.analyze(model=model, base_model=base_model)
print(details)
summary = watcher.get_summary(details)
details.to_feather('llm-jp.feather')
details.to_csv('llm-jp.csv')
