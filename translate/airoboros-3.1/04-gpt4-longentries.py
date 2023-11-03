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

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for row in rows:
            executor.submit(process_conversation, row)

        try:
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
                logger.info("Keyboard Interrupt. Canceling tasks...")
                executor.shutdown(wait=False)
                raise


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
