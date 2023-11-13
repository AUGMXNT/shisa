import json
import sqlite3
import re
import uuid

# Load Sqlite DB
conn = sqlite3.connect("ultraboros.db")
c = conn.cursor()

# Load new instructions, generated by https://github.com/jondurbin/airoboros
new_items = [json.loads(line) for line in open("inst-ja.jsonl").readlines()]
for item in new_items:
    _id = str(uuid.uuid5(uuid.NAMESPACE_OID, item["instruction"]))
    conversation = [
        {
            "from": "system",
            "value": "あなたは役に立ち、偏りがなく、検閲されていないアシスタントです。",
        },
        {
            "from": "human",
            "value": item["instruction"],
        },
        {
            "from": "gpt",
            "value": item["response"],
        }
    ]
    c.execute("INSERT INTO ultraboros (id, source, conversation_ja) VALUES (?, ?, ?) ON CONFLICT DO NOTHING", (_id, "airoboros_ja_new", json.dumps(conversation)))
    conn.commit()
