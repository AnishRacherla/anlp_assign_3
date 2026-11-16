#!/usr/bin/env bash
set -e

MODEL="${MODEL:-${MODEL_TYPE:-combo}}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-8e-4}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -d "$REPO_DIR/.venv" ]; then
    if ! find "$REPO_DIR/.venv" -name "libcudnn.so.9" 2>/dev/null | grep -q .; then
        echo "[test_training.sh] Auto-installing missing nvidia-cudnn-cu12 library..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install "nvidia-cudnn-cu12>=9.0" 2>/dev/null || true
        elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
            "$REPO_DIR/.venv/bin/pip" install "nvidia-cudnn-cu12>=9.0" 2>/dev/null || true
        fi
    fi

    if ! find "$REPO_DIR/.venv" -name "libnccl.so.2" 2>/dev/null | grep -q .; then
        echo "[test_training.sh] Installing missing nvidia-nccl-cu12 into virtual environment..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install "nvidia-nccl-cu12" || true
        elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
            "$REPO_DIR/.venv/bin/pip" install "nvidia-nccl-cu12" || true
        fi
    fi

    VENV_NVIDIA_LIBS=$(find "$REPO_DIR/.venv" -type d -name "lib" 2>/dev/null | grep -E "site-packages/nvidia|site-packages/torch" | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi

    NCCL_LIB=$(find "$REPO_DIR/.venv" -name "libnccl.so.2" 2>/dev/null | head -n 1)
    if [ -n "$NCCL_LIB" ]; then
        export LD_PRELOAD="${NCCL_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
fi

if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    "$REPO_DIR/.venv/bin/python" -m src.train \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        "$@"
elif command -v uv >/dev/null 2>&1; then
    uv run python -m src.train \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        "$@"
else
    python -m src.train \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        "$@"
fi
