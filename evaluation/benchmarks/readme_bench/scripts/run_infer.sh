#!/usr/bin/env bash
set -eo pipefail

MODEL_CONFIG=$1
AGENT=$2
EVAL_LIMIT=$3
MAX_ITER=$4
NUM_WORKERS=$5
DATASET=$6
SPLIT=$7
N_RUNS=${8}
CONFIG_PATH=$9

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=10
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  MAX_ITER=10
  echo "MAX_ITER not specified, use default $MAX_ITER"
  
fi

if [ -z "$RUN_WITH_BROWSING" ]; then
  echo "RUN_WITH_BROWSING not specified, use default false"
  RUN_WITH_BROWSING=false
fi


if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default islamham/test-dataset"
  DATASET="islamham/bugswarm_python"
fi


if [ -z "$SPLIT" ]; then
  echo "HF SPLIT not specified, use default train"
  SPLIT="train"
fi

# if [ -z "$CONFIG_PATH" ]; then
#   echo "CONFIG_PATH not specified, use default config"
#   CONFIG_PATH="/Users/hamza/OpenHands/openhands/config.toml"
# fi

export RUN_WITH_BROWSING=$RUN_WITH_BROWSING
echo "RUN_WITH_BROWSING: $RUN_WITH_BROWSING"

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "HF SPLIT: $SPLIT"

# Default to NOT use Hint
if [ -z "$USE_HINT_TEXT" ]; then
  export USE_HINT_TEXT=false
fi


echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE=$1
# if not using Hint, add -no-hint to the eval note
if [ "$USE_HINT_TEXT" = false ]; then
  EVAL_NOTE="$EVAL_NOTE-no-hint"
fi

if [ "$RUN_WITH_BROWSING" = true ]; then
  EVAL_NOTE="$EVAL_NOTE-with-browsing"
fi

if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi
EVAL_NOTE="$EVAL_NOTE-numworker_$NUM_WORKERS"
EVAL_NOTE="$EVAL_NOTE-$(date +%s)"

function run_eval() {

  local eval_note=$1

  COMMAND="poetry run python /Users/hamza/OpenHands/evaluation/benchmarks/readme_bench/run_infer.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $eval_note \
    --dataset $DATASET \
    --split $SPLIT "


  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  # Run the command
  eval $COMMAND
}

unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, use default $N_RUNS"
fi

for i in $(seq 1 $N_RUNS); do
  current_eval_note="$EVAL_NOTE-run_$i"
  echo "EVAL_NOTE: $current_eval_note"
  run_eval $current_eval_note
done
