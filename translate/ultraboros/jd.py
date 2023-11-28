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

c2.execute("SELECT * FROM prompts WHERE prompt_ja IS NOT NULL")
rows = c.fetchall()
for row in tqdm(rows):
    c.execute("INSERT INTO prompts (prompt_en, prompt_ja) VALUES (?, ?)", (row['prompt_en'], row['prompt_ja']))
    conn.commit()
