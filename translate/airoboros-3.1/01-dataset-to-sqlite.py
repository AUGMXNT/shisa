from   datasets import load_dataset
import json
import sqlite3

# Load Sqlite DB
conn = sqlite3.connect("airoboros.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS airoboros_31 (
    id VARCHAR PRIMARY KEY,
    category VARCHAR,
    conversation TEXT,
    conversation_ja TEXT,
    translator VARCHAR
);
""")
conn.commit()

ds = load_dataset('jondurbin/airoboros-3.1')
i = 0
print(ds)
# Insert items into SQLite table
for item in ds['train']:
    print(f"{i:>5} : {item['id']} : {item['category']}")
    i+=1
    # Check if the id already exists in the table
    c.execute("SELECT id FROM airoboros_31 WHERE id = ?", (item['id'],))
    if c.fetchone() is None:
        # JSON-encode the conversations
        conv_json = json.dumps(item['conversations'])
        
        # Insert new row
        c.execute("INSERT INTO airoboros_31 (id, category, conversation) VALUES (?, ?, ?)",
                  (item['id'], item['category'], conv_json))
        conn.commit()

# Commit the changes and close the connection
conn.close()
