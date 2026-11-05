#!/usr/bin/env bash
set -e

MODEL_TYPE="${MODEL_TYPE:-hybrid}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"

if [ -d ".venv" ]; then
    VENV_NVIDIA_LIBS=$(find .venv -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    uv run python -m src.train \
        --model-type "$MODEL_TYPE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        "$@"
else
    python -m src.train \
        --model-type "$MODEL_TYPE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        "$@"
fi
