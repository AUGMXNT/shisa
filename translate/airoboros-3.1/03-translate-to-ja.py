import backoff
import openai
import json
import sqlite3
import sys
import tiktoken
import time

# We may use a faster model!
model = 'gpt-4'

# Connect to the SQLite database
conn = sqlite3.connect("airoboros.db")

# Create a cursor object
c = conn.cursor()
c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL")
rows = c.fetchall()

# Define OpenAI function
@backoff.on_exception(backoff.expo, Exception)
def call_openai(text):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Translate the following to Japanese:"
            },
            {
                "role": "user",
                "content": turn["value"]
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

# There are only 3 'from' roles: 'system', 'human', 'gpt' and we'll leave those in English
# We only want to translate the 'value' and then we will reassemble

enc = tiktoken.get_encoding("cl100k_base")
tokencount = []
for row in rows:
    ttt = time.time()
    id = row[0]

    print("===")
    print(f"Translating {id} ({row[1]}):")
    conversation = json.loads(row[2])
    conversation_ja = []
    for turn in conversation:
        # Token counts
        # tokens = enc.encode(turn['value'])
        # tokencount.append(len(tokens))
        # continue
        
        value_ja = ""
        print('>', turn)

        # First let's try to get a value
        if turn["from"] == "system":
            c.execute("SELECT prompt_ja FROM prompts WHERE prompt_en = ?", (turn["value"],))
            row = c.fetchone()
            if row is not None:
                value_ja = row[0]

        if not value_ja:
            value_ja = call_openai(turn["value"])
            time.sleep(1)

        print('>>', value_ja)
        turn['value'] = value_ja
        conversation_ja.append(turn)

    # OK, lets 
    conv_json = json.dumps(conversation_ja)
    c.execute("UPDATE airoboros_31 SET conversation_ja = ?, translator=? WHERE id = ?", (conv_json, model, id))
    conn.commit()

    ttt = time.time() - ttt
    print(f"# {ttt:.2f} s")
    print()

    time.sleep(1)


conn.close()
