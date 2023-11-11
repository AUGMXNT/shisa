import json

def convert_from_field(data):
    for item in data:
        for conversation in item['conversations']:
            if conversation['from'] == 'gpt':
                conversation['from'] = 'assistant'
            elif conversation['from'] == 'human':
                conversation['from'] = 'user'
    return data

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    modified_data = convert_from_field(data)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(modified_data, file, ensure_ascii=False, indent=2)

input_file = 'ultraboros-en-ja-v0.1.json'
output_file = 'qwen-ultraboros-en-ja-v0.1.json'

process_file(input_file, output_file)
