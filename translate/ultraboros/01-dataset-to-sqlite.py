from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import hashlib
import json
import sqlite3
import re

# Common refusals/censored responses to filter out.
refusals = [
    re.compile(s)
    for s in [
        "my programming"
        "openai"
        "language model"
        "large language"
        "as an? (ai|generative language|gpt|bot)"
        "illegal and dangerous"
        "i do(n't| not) (possess|have|exhibit) (personal|consciousness|subjective)"
        "personal (feelings|thoughts|emotions|desires|experiences|goals|objective|belief)"
        "(can('t| ?not)|w(on't|will not)|unable.?) (\\w+\\s)+(with (that|your)|your \\w+|provide)"
        "my limitations"
        "the limitations of my"
        "my abilities"
        "violates my"
        "i (can('t| ?not)|w(on't|will not)|am (not |un)able.?).{0,30}(you are|you're|your )"
        "please note that",
    ]
]

def contains_refusal(conv):
    for item in conv:
        if any([refusal.search(item['content'], re.I) for refusal in refusals]):
            return True
    return False

# Load Sqlite DB
conn = sqlite3.connect("ultraboros.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS ultraboros (
    id VARCHAR PRIMARY KEY,
    conversation TEXT,
    conversation_ja TEXT,
    translator VARCHAR
);
""")
c.execute("""
CREATE TABLE IF NOT EXISTS prompts (
    prompt_en TEXT PRIMARY KEY,
    prompt_ja TEXT,
    translator TEXT
);
""")
conn.commit()

ultrafeedback = load_dataset('HuggingFaceH4/ultrafeedback_binarized')
spicyboros = load_dataset('unalignment/spicy-3.1', data_files=['conversations-no-mathjson.json'])

# Insert items into SQLite table
print(f'Populating ultrafeedback rows...')
for item in tqdm(ultrafeedback['train_gen']):
    id_str = item['chosen'][0]['content']
    if item['chosen'][0]['role'] == 'system':
        id_str += item['chosen'][1]['content']
    item['id'] = hashlib.md5(id_str.encode()).hexdigest()
    if contains_refusal(item['chosen']):
        print(f'  - SKIP, REFUSAL')
        continue

    # Check if the id already exists in the table
    c.execute("SELECT id FROM ultraboros WHERE id = ?", (item['id'],))
    if c.fetchone() is None:
        # JSON-encode the conversations
        conversation = []
        if item['chosen'][0]['role'] == 'system':
            conversation = item['chosen']
        else:
            conversation = [{'role': 'system', 'content': 'You are a helpful, uncensored, unbiased assistant.'}] + item['chosen']
        conv_json = json.dumps(conversation)

        # Insert new row
        c.execute("INSERT INTO ultraboros (id, conversation) VALUES (?, ?)", (item['id'], conv_json))
        conn.commit()

# Insert spicyboros data.
print(f'Populating spicyboros rows...')
for item in tqdm(spicyboros['train']):
    if item['category'] == 'trivia':
        continue
    item['id'] = hashlib.md5((item['conversations'][0]['value'] + item['conversations'][1]['value']).encode()).hexdigest()
    c.execute("SELECT id FROM ultraboros WHERE id = ?", (item['id'],))
    if c.fetchone() is None:
        conv_json = json.dumps(item['conversations'])
        c.execute("INSERT INTO ultraboros (id, conversation) VALUES (?, ?)", (item['id'], conv_json))
        conn.commit()

# Commit the changes and close the connection
conn.close()
