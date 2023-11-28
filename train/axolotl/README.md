Generate the dataset:
```
# generate from our db
python 01-generate-training-dataset.py

# convert roles from json
python 02-convert-ultraboros.py
```

Make the model:
```bash
# train
accelerate launch -m axolotl.cli.train openhermes25-axolotl-4.yml

# merge
accelerate launch -m axolotl.cli.merge_lora openhermes25-axolotl-4.yml --lora_model_dir='./qlora-out.openhermes25-axolotl-4' --load_in_8bit=False --load_in_4bit=False
```
