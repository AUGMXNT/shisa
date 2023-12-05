#!/bin/bash

# 5min x 4 x 5

# Define an array of models
models=(
    # "/mnt/data/shisa/zero-extra/shisa-mega-7b-v1"
    # "/mnt/data/shisa/zero-extra/shisa-mega-7b-v1.1"
    # "/mnt/data/shisa/zero-extra/shisa-mega-7b-v1.2"
    # "/mnt/data/shisa/zero-extra/shisa-mega-dpo-7b-v1"
    # "/mnt/data/shisa/zero-extra/shisa-mega-dpo-7b-v1.1"
    "/mnt/data/shisa/augmxnt_shisa-mega-7b-v1.2-dpo"
)

# Array of temperatures
temperatures=(0.1 0.4 0.7 1.0)

# Number of runs per model per temperature
num_runs=5

# Loop over each model
for model in "${models[@]}"; do
    model_basename=$(basename "$model")

    # Loop over each temperature
    for temp in "${temperatures[@]}"; do

        # Loop for the number of runs
        for run in $(seq 1 $num_runs); do
            output_file="gen/${model_basename}_temp-${temp}_run-${run}.json"
            # Run the generation command
            time python default-gen.py -t "$temp" -m "$model" -o "$output_file"
        done

    done
done
