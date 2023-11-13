from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import hashlib
import json
import sqlite3
import re

# Load Sqlite DB
conn = sqlite3.connect("ultraboros.db")
conn2 = sqlite3.connect("airoboros.db")
c = conn.cursor()
c2 = conn2.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS ultraboros (
    id VARCHAR PRIMARY KEY,
    source TEXT,
    conversation TEXT,
    conversation_ja TEXT,
    translator VARCHAR
);
""")
c.execute("""
CREATE TABLE IF NOT EXISTS reroll (
    prompt TEXT NOT NULL PRIMARY KEY,
    response TEXT
);
""")
conn.commit()

c2.execute("SELECT id, conversation, conversation_ja FROM airoboros_31 WHERE conversation_ja IS NOT NULL")
rows = c2.fetchall()
for row in tqdm(rows):
    id_, conversation, conversation_ja = row
    conv = json.loads(conversation_ja)
    valid = True
    if any(['日本' in turn['value'] for turn in conv]):
        continue
    c.execute("SELECT id, conversation_ja FROM ultraboros WHERE conversation = ?", (conversation,))
    row2 = c.fetchone()
    if row2:
        if valid:
            c.execute("UPDATE ultraboros SET conversation_ja = ? WHERE id = ?", (conversation_ja, row2[0]))
            conn.commit()
    else:
        if valid:
            c.execute("INSERT INTO ultraboros (id, source, conversation, conversation_ja) VALUES (?, ?, ?, ?)", (id_, "airoboros", conversation, conversation_ja))
        else:
            c.execute("INSERT INTO ultraboros (id, source, conversation) VALUES (?, ?, ?)", (id_, "airoboros", conversation))
        conn.commit()
