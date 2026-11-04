#!/usr/bin/env bash
set -e

MODEL_TYPE="${MODEL_TYPE:-hybrid}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"

uv run python -m src.train \
    --model-type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    "$@"
