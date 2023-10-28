from   collections import defaultdict
from   pprint import pprint
import openai
import json
import sqlite3
import sys
import tiktoken
import time

# Connect to the SQLite database
conn = sqlite3.connect("airoboros.db")

# Create a cursor object
c = conn.cursor()
c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL")
rows = c.fetchall()
conn.close()

# There are only 3 'from' roles: 'system', 'human', 'gpt' and we'll leave those in English
# We only want to translate the 'value' and then we will reassemble

prompts = defaultdict(int)
for row in rows:
    conversation = json.loads(row[2])
    for turn in conversation:
        if turn["from"] == "system":
            prompts[turn["value"]] += 1

repeat_prompts = {prompt: count for prompt, count in prompts.items() if count > 1}


# Time to make the donuts
conn = sqlite3.connect("airoboros.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS prompts (
    prompt_en TEXT PRIMARY KEY,
    prompt_ja TEXT
);
""")
conn.commit()

# TODO: if we run this again, we should insert the prompts first, then SELECT
# only empty prompts...

for prompt in repeat_prompts:
    print('>', prompt)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Translate the following to Japanese:"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    prompt_ja = response["choices"][0]["message"]["content"]
    print('>', prompt_ja)

    # Insert into DB
    c.execute("INSERT OR REPLACE INTO prompts (prompt_en, prompt_ja) VALUES (?, ?)", (prompt, prompt_ja))
    conn.commit()

    print()
