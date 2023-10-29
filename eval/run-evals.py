import os
import json
import shutil
import subprocess
import time
import pandas as pd


def main():
    dataset_dir = "/home/ubuntu/llm-jp-eval/dataset_dir/evaluation/test"
    config_file = "/home/ubuntu/llm-jp-eval/config/config.yaml"
    csv_file_path = "LLM_scores.csv"

    model_list = [
        {"name": "model1", 
         "model_path": "/path/to/model1", 
         "tokenizer_path": "/path/to/tokenizer1", 
         "run": True
        },
        {"name": "model1", 
         "model_path": "/path/to/model1", 
         "tokenizer_path": "/path/to/tokenizer1", 
         "run": False
        },
        # Add more models here
    ]

    for model_info in model_list:
        if model_info["run"]:
            start_time = time.time()
            run_evaluation_script(model_info, dataset_dir, config_file)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Model {model_info['name']} took {elapsed_time} seconds to evaluate.")

            move_logs(model_info['name'])

            json_file_path = f"logs/{model_info['name']}/score_eval.json"
            update_scores_csv(csv_file_path, model_info['name'], json_file_path)

def load_json_scores(model_name, json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return {model_name: data}

def update_scores_csv(csv_file_path, model_name, json_file_path):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, index_col=0)
    else:
        df = pd.DataFrame()

    new_scores = load_json_scores(model_name, json_file_path)
    df = df.append(pd.DataFrame(new_scores).set_index(pd.Index([model_name])))
    df.to_csv(csv_file_path)

def run_evaluation_script(model_info, dataset_dir, config_file):
    command = [
        "time", "python", "scripts/evaluate_llm.py", "-cn", config_file,
        f"model.pretrained_model_name_or_path={model_info['model_path']}",
        f"tokenizer.pretrained_model_name_or_path={model_info['tokenizer_path']}",
        f"dataset_dir={dataset_dir}"
    ]

    subprocess.run(command)

def move_logs(model_name, src_dir='llm-jp-eval/logs/'):
    dest_dir = f'logs/{model_name}/'
    os.makedirs(dest_dir, exist_ok=True)

    for json_file in os.listdir(src_dir):
        if json_file.endswith('.json'):
            shutil.move(os.path.join(src_dir, json_file), os.path.join(dest_dir, json_file))

if __name__ == "__main__":
    main()

