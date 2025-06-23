#!/usr/bin/env bash

DATASET=$1
if [ -z "$DATASET" ]; then
  echo "DATASET not specified"
  exit 1
fi

echo "DATASET: $DATASET"

COMMAND="poetry run python /Users/hamza/OpenHands/evaluation/benchmarks/readme_bench/run_analysis.py \
  --dataset $DATASET "

# Run the command
eval $COMMAND
