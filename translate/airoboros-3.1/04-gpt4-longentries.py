import backoff
from   concurrent.futures import ThreadPoolExecutor
from   loguru import logger
import json
import openai
import sqlite3
import sys
import traceback
import threading
import tiktoken
import time


# Simultaneous calls
THREADS = 2
lock = threading.Lock()

# We may use a faster model!
model = 'gpt-4'
price_input = 0.03
price_output = 0.06

# 32K 0.06 in, 0.12 out

# Connect to the SQLite database
DB = 'airoboros.db'

# Set logger level
DEBUG = 0
if not DEBUG:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


# Execute Threads
def main():
    # > 20K characters
    # rows = get_long_rows()

    # Untranslated non mathjson!
    rows = get_untranslated()

    # Retranslate conversation_ja/conversation ratio >3.0, <1.0...
    # rows = get_sus_lengths()


    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for row in rows:
            executor.submit(process_conversation, row)

        try:
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
                logger.info("Keyboard Interrupt. Canceling tasks...")
                executor.shutdown(wait=False)
                raise


'''
This is our initial run of 1077 long items
'''
def get_long_rows():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    if DEBUG:
        c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND translator != 'gpt-4' AND LENGTH(conversation_ja) >= 20000 LIMIT 1")
    else:
        c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND translator != 'gpt-4' AND LENGTH(conversation_ja) >= 20000")
    rows = c.fetchall()
    conn.close()

    logger.debug("=== DEBUG MODE (1 item, no saving to DB) ===")
    logger.info(f"=== Processing {len(rows)} items ===")
    return rows

'''
Somehow there are 998 missing translations:

sqlite> SELECT COUNT(*), category FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL GROUP BY category;

2|awareness
1|card
90|coding
23|contextual
20|counterfactual_contextual
1|detailed_writing
6|editor
3|experience
188|general
10|gtkm
110|joke
31|misconception
10|multiple_choice
61|multiturn
17|orca
2|quiz
36|riddle
91|roleplay
1|song
21|stylized_response
187|summarization
5|theory_of_mind
10|trivia
8|wordgame
64|writing
'''
def get_untranslated():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    if DEBUG:
        c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL ORDER BY LENGTH(conversation) ASC LIMIT 1")
    else:
        c.execute("SELECT id, category, conversation FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NULL ORDER BY LENGTH(conversation) ASC")
    rows = c.fetchall()
    conn.close()

    logger.debug("=== DEBUG MODE (1 item, no saving to DB) ===")
    logger.info(f"=== Processing {len(rows)} items ===")
    return rows

def get_sus_lengths():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if DEBUG:
        c.execute("SELECT id, category, conversation, conversation_ja FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NOT NULL AND translator != 'gpt-4' LIMIT 1")
    else:
        c.execute("SELECT id, category, conversation, conversation_ja FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NOT NULL AND translator != 'gpt-4'")
    rows = c.fetchall()
    conn.close()

    # This will take about 30s to filter...
    def get_tokencount(conversation_json):
        c = json.loads(conversation_json)
        enc = tiktoken.get_encoding("cl100k_base")

        tokencount = 0
        for turn in c:
            tokens = enc.encode(turn['value'])
            tokencount += len(tokens)

        return tokencount

    sus = []
    for row in rows:
        tc = get_tokencount(row['conversation'])
        tc_ja = get_tokencount(row['conversation_ja'])
        ratio = tc_ja/tc

        if ratio >= 3.0 or ratio < 0.5:
            sus.append(row)

    logger.debug("=== DEBUG MODE (1 item, no saving to DB) ===")
    logger.info(f"=== Processing {len(sus)} items ===")
    return sus




thread_dict = {}
def get_thread_number():
    thread_id = threading.get_ident()
    if thread_id not in thread_dict:
        thread_dict[thread_id] = len(thread_dict) + 1
    return thread_dict[thread_id]


# We handle each conversation as a row
def process_conversation(row):
    try:
        # Since connections aren't threadsafe...
        conn = sqlite3.connect(DB)
        c = conn.cursor()

        # Timer
        ttt = time.time()

        id = row[0]
        category = row[1]
        conversation = json.loads(row[2])

        thread_id = get_thread_number()
        logger.info(f"{thread_id}: START {id} ({category})...")

        conversation_ja = []
        for turn in conversation:
            value_ja = ""
            logger.debug(turn)

            # First let's try to get a value - we won't cache this
            '''
            if turn["from"] == "system":
                c.execute("SELECT prompt_ja FROM prompts WHERE prompt_en = ?", (turn["value"],))
                row = c.fetchone()
                if row is not None:
                    value_ja = row[0]
            '''

            if not value_ja:
                logger.debug(f"{thread_id}: before call_open")
                value_ja = call_openai(turn["value"])
                logger.debug(f"{thread_id}: after call_open")

            turn['value'] = value_ja
            conversation_ja.append(turn)

        conv_json = json.dumps(conversation_ja)
        logger.debug(conv_json)

        # We don't update the DB if debug...
        if not DEBUG:
            with lock:
                c.execute("UPDATE airoboros_31 SET conversation_ja = ?, translator=? WHERE id = ?", (conv_json, model, id))
                c.execute("REPLACE INTO translate_history (id, translator, translation) VALUES (?, ?, ?)", (id, model, conv_json))
                conn.commit()

        ttt = time.time() - ttt
        logger.info(f"{thread_id}: END   {id} ({ttt:.2f} s)")

    except Exception as e:
        # tb_str = ''.join(traceback.format_exception(None, exception, exception.__traceback__))
        logger.error(f"{thread_id}: EXCEPTION in {id}: {e}")
        traceback.print_exc()  # This will print the full traceback to the log
        # logger.error(f"Traceback: {tb_str}")
        # traceback.print_exc()  # This will print the full traceback to the log
    finally:
        conn.close()


# Function to log traceback with Loguru on backoff event
def loguru_traceback(details):
    thread_id = get_thread_number()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(details['args'][0])
    details["args"] = f"({len(tokens)} tokens)"
    logger.warning(f"{thread_id}: Backoff triggered due to an exception: {details}") #, exc_info=True)
    # exc_info=True not working
    # logger.warning("Exception occurred", exc_info=True)


# Define OpenAI function
@backoff.on_exception(backoff.expo, Exception, base=5, factor=2, max_value=120, max_tries=10, on_backoff=loguru_traceback)
def call_openai(text):
    thread_id = get_thread_number()
    enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    price = len(tokens)/1000.0*price_input
    logger.info(f"{thread_id}: SEND {len(tokens)} tokens (${price:.4f})")
    t = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Translate the text to Japanese."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    tokens_ja = enc.encode(response["choices"][0]["message"]["content"])
    price = len(tokens)/1000.0*price_output
    t = time.time()-t
    logger.info(f"{thread_id}: RECIEVED {len(tokens_ja)} JA tokens (${price:.4f}; {t:.2f} s)")

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    main()
