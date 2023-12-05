import pandas as pd
from   multiprocessing import Pool
from   transformers import AutoTokenizer
import sys


# Function to tokenize and count tokens in each item
'''
def count_tokens(item):
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], use_fast=True)
    tokens = tokenizer.tokenize(item['value'])  # Assuming 'value' is the key with text
    return len(tokens)

# Function to process each row
def process_row(row):
    conversations = row['conversations']
    with Pool(processes=8) as pool:  # Adjust the number of processes as needed
        token_counts = pool.map(count_tokens, conversations)
    return sum(token_counts)
'''

if len(sys.argv) != 2:
    print("Usage: count-dataset-tokens .py </tokenizer/path>")
    sys.exit(1)

df = pd.read_parquet('/mnt/dev-03-data/ultra-orca-boros-en-ja/ultra-orca-boros-en-ja-v0.6.parquet')
# multithreaded
# total_token_count = sum(df.apply(process_row, axis=1))

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], use_fast=True)

# Initialize a variable to hold the total token count
total_token_count = 0

# Iterate over each row in the DataFrame
for _, row in df.iterrows():
    # Access the 'conversations' column (or replace with your relevant column)
    conversations = row['conversations']  # Assuming 'conversations' is the column of interest

    # Iterate over each item in the 'conversations' list
    for item in conversations:
        # Tokenize the value of each item
        tokens = tokenizer.tokenize(item['value'])  # Assuming 'value' is the key with text

        # Update the total token count
        total_token_count += len(tokens)

print("Total token count:", total_token_count)
