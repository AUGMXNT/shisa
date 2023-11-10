import json
import sqlite3
import sys

# alpaca_w_system.load_open_orca

conn = sqlite3.connect("shisa/translate/airoboros-3.1/airoboros.db")

# Create a cursor object
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT * FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NOT NULL")
rows = c.fetchall()

'''
# sharegpt: conversations where from is human/gpt
{"conversations": [{"from": "...", "value": "..."}]}
'''

# Function to transform the conversation format
def transform_conversation(conversation_str):
    conversation = json.loads(conversation_str)
    transformed = []
    
    for item in conversation:
        # We won't convert
        '''
        if item["from"] in ["system"]:
            item["from"] = "system_prompt"
        if item["from"] in ["gpt"]:
            item["from"] = "response"
        elif item["from"] == "human":
            item["from"] = "question"
        '''
        transformed.append(item)
    
    return transformed

output = []
for row in rows:
    conversation = transform_conversation(row["conversation"])
    conversation_ja = transform_conversation(row["conversation_ja"])

    # Add both conversations to the output
    output.append({"id": f"identity_{row['id']}", "conversations": conversation})
    output.append({"id": f"identity_{row['id']}_ja", "conversations": conversation_ja})

# Write to a JSON file
with open('airoboros-en-ja.dataset.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

conn.close()
