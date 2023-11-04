import backoff
from   concurrent.futures import ThreadPoolExecutor
from   loguru import logger
import json
import openai
import plotext as plt
from   pprint import pprint
import sqlite3
import sys
from   tabulate import tabulate
import traceback
import threading
import tiktoken
import time


# Connect to the SQLite database
DB = 'airoboros.db'

# Set logger level
DEBUG = 0
if not DEBUG:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


# Execute Threads
def main():
    # Get all rows
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, category, conversation, conversation_ja, translator FROM airoboros_31 WHERE category != 'mathjson' AND conversation_ja IS NOT NULL")
    rows = c.fetchall()
    conn.close()

    sus_long = []
    sus_short = []

    logger.info(f"=== Processing {len(rows)} items ===")



    table = []
    for row in rows:
        r = dict(row)
        r['tokencount_conversation'] =  get_tokencount(row['conversation'])
        r['tokencount_conversation_ja'] =  get_tokencount(row['conversation_ja'])
        r['tokencount_ratio'] = r['tokencount_conversation_ja']/r['tokencount_conversation'] 

        if r['tokencount_ratio'] >= 3.0:
            sus_long.append(r)
        elif r['tokencount_ratio'] <= 1.0:
            sus_short.append(r)

        table.append(r)

    '''
    Play around with histograms until we get some good cutoffs

    # https://pypi.org/project/plotext/
    # https://pypi.org/project/plotille/

    ratios = [row['tokencount_ratio'] for row in table]
    plt.plotsize(None, 30)
    plt.hist(ratios, bins=20000, orientation='horizontal')
    plt.xlim(0, 100)
    plt.ylim(0.7, 1.2)
    plt.title('Histogram of Token Count Ratios')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.show()
    sys.exit()
    '''

    # Print tables
    def print_table(table):
        header_labels = {
            'tokencount_conversation': 'tc',
            'tokencount_conversation_ja': 'tc_ja',
            'tokencount_ratio': 'ratio',
        }
        selected_fields = ['id', 'category', 'translator', 'tokencount_conversation', 'tokencount_conversation_ja', 'tokencount_ratio'] 
        selected_data = [[row[field] for field in selected_fields] for row in table]
        relabeled_headers = [header_labels.get(field, field) for field in selected_fields]
        print(tabulate(selected_data, headers=relabeled_headers, floatfmt='.2f'))

    print('long:', len(sus_long))
    print_table(sus_long)
    print('short:', len(sus_short))
    print_table(sus_short)


def get_tokencount(conversation_json):
    c = json.loads(conversation_json)
    enc = tiktoken.get_encoding("cl100k_base")

    tokencount = 0
    for turn in c:
        tokens = enc.encode(turn['value'])
        tokencount += len(tokens)

    return tokencount


if __name__ == "__main__":
    main()
