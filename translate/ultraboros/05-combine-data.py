import os
import uuid
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import random
import hashlib
import json
import sqlite3
import re
from loguru import logger

# Load Sqlite DB
conn = sqlite3.connect("ultraboros.db")
c = conn.cursor()

c.execute("SELECT id, conversation, conversation_ja FROM ultraboros WHERE conversation_ja IS NOT NULL")
rows = c.fetchall()

formatted = []
for row in tqdm(rows):
    _id = row[0]
    conv = None
    conv_ja = None
    if row[1]:
        conv = json.loads(row[1])
        for turn in conv:
            if 'role' in turn:
                turn['from'] = turn.pop('role')
            turn['from'] = 'human' if turn['from'] == 'user' else turn['from']
            turn['from'] = 'gpt' if turn['from'] == 'assistant' else turn['from']
            if 'content' in turn:
                turn['value'] = turn.pop('content')
    if row[2]:
        conv_ja = json.loads(row[2])
        for turn in conv_ja:
            if 'role' in turn:
                turn['from'] = turn.pop('role')
            turn['from'] = 'human' if turn['from'] == 'user' else turn['from']
            turn['from'] = 'gpt' if turn['from'] == 'assistant' else turn['from']
            if 'content' in turn:
                turn['value'] = turn.pop('content')
    if conv:
        formatted.append({
            'id': _id,
            'conversations': conv,
        })
    if conv_ja:
        formatted.append({
            'id': f'{_id}-ja',
            'converations': conv_ja,
        })

with open('ultraboros-en-ja-v0.1.json', 'w') as outfile:
    outfile.write(json.dumps(formatted, indent=2))
