#!/usr/bin/env python

import sys
import re

def main(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Regex to find "acc": [number],
        pattern = r'"acc":\s*(\d+\.\d+)'
        acc_values = [float(match) for match in re.findall(pattern, content)]

        if acc_values:
            avg_acc = sum(acc_values) / len(acc_values)
            print(f'Average MMLU Score: {avg_acc:.4f}')
            print(len(acc_values), 'tests')
        else:
            print("No accuracy values found.")

    except FileNotFoundError:
        print(f"File {filename} not found.")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else 'mmlu.txt'
    main(filename)
