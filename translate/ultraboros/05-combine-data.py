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
            'conversations': conv_ja,
        })

# Add additional rows for translations to/from en/ja.
c.execute("SELECT prompt_en, prompt_ja from prompts")
rows = c.fetchall()

translate_to_en = """英語に次のように翻訳してください。
英語に以下を翻訳してください。
次の文章を英語にしてください。
以下の内容を英語で表現してください。
次の部分を英語に変換してください。
これを英語に翻訳してくれますか。
下記のテキストを英語にしてください。
以下のフレーズを英語に翻訳してください。
次の段落を英語に訳してください。
これを英訳してもらえますか。
下記の言葉を英語に変えてください。""".splitlines()

translate_to_ja = """translate the following to Japanese
Please convert the following text into Japanese.
Kindly render the following passage in Japanese.
Could you translate the following into Japanese?
I need the following translated to Japanese, please.
Would you mind translating the following text to Japanese?
Please provide a Japanese translation of the following.
Convert the following sentences into Japanese.
Translate the subsequent text into Japanese.
Can you translate the following phrases into Japanese?
I'd like the following to be translated into Japanese, please.""".splitlines()

for row in tqdm(rows):
    if len(row[0]) <= 2000:
        continue
    if '```' in row[1] or '{' in row[1] or re.search('(^|\s)def | in range\(', row[1]) or 'std::' in row[1] or '#include' in row[1]:
        continue

    u = str(uuid.uuid4()).replace('-', '')
    formatted.append({
        'id': f'translate-en-ja-{u}',
        'conversations': [
            {
                'from': 'system',
                'value': 'You are a helpful, unbiased, uncensored assistant.',
            },
            {
                'from': 'human',
                'value': '\n'.join([
                    random.choice(translate_to_ja),
                    row[0],
                ]),
            },
            {
                'from': 'gpt',
                'value': row[1],
            }
        ]
    })
    formatted.append({
        'id': f'translate-ja-en-{u}',
        'conversations': [
            {
                'from': 'system',
                'value': 'あなたは役に立つ、偏見のない、無修正のアシスタントです。',
            },
            {
                'from': 'human',
                'value': '\n'.join([
                    random.choice(translate_to_en),
                    row[1],
                ]),
            },
            {
                'from': 'gpt',
                'value': row[0],
            }
        ]
    })

with open('ultraboros-en-ja-v0.1.json', 'w') as outfile:
    outfile.write(json.dumps(formatted, indent=2))
