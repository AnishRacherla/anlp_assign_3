#!/usr/bin/env bash
set -e

ROLL_NUMBER="${ROLL_NUMBER:-2024101046}"
HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"

EXTRA_ARGS=()
if [ -n "$HF_REPO_ID" ]; then
    EXTRA_ARGS+=(--hf-repo-id "$HF_REPO_ID")
fi
if [ -n "$MODEL" ]; then
    EXTRA_ARGS+=(--model "$MODEL")
elif [ -n "$MODEL_TYPE" ]; then
    EXTRA_ARGS+=(--model "$MODEL_TYPE")
fi
if [ -n "$CHECKPOINT_PATH" ]; then
    EXTRA_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(hostname 2>/dev/null)" =~ (ada|gnode) ]] || [ -d "/home2" ]; then
    export http_proxy="${http_proxy:-http://proxy.iiit.ac.in:8080}"
    export https_proxy="${https_proxy:-http://proxy.iiit.ac.in:8080}"
    export HTTP_PROXY="${HTTP_PROXY:-http://proxy.iiit.ac.in:8080}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.iiit.ac.in:8080}"
fi

if [ -d "$REPO_DIR/.venv" ]; then
    if ! find "$REPO_DIR/.venv" -name "libcudnn.so.9" 2>/dev/null | grep -q .; then
        echo "[test_assn.sh] Installing missing nvidia-cudnn-cu12 into virtual environment..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
        elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
            "$REPO_DIR/.venv/bin/pip" install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
        fi
    fi

    if ! find "$REPO_DIR/.venv" -name "libnccl.so*" 2>/dev/null | grep -q .; then
        echo "[test_assn.sh] Installing missing nvidia-nccl-cu12 into virtual environment..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --reinstall "nvidia-nccl-cu12" || true
        elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
            "$REPO_DIR/.venv/bin/pip" install --force-reinstall "nvidia-nccl-cu12" || true
        fi
    fi

    VENV_NVIDIA_LIBS=$(find "$REPO_DIR/.venv" -type d -name "lib" 2>/dev/null | grep -E "site-packages/nvidia|site-packages/torch" | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi

    NCCL_LIB=$(find "$REPO_DIR/.venv" -name "libnccl.so*" 2>/dev/null | grep -E "libnccl\.so(\.[0-9]+)*$" | head -n 1)
    if [ -n "$NCCL_LIB" ]; then
        export LD_PRELOAD="${NCCL_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
fi

if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    "$REPO_DIR/.venv/bin/python" -m src.evaluate \
        --roll-number "$ROLL_NUMBER" \
        "${EXTRA_ARGS[@]}" \
        "$@"
elif command -v uv >/dev/null 2>&1; then
    uv run python -m src.evaluate \
        --roll-number "$ROLL_NUMBER" \
        "${EXTRA_ARGS[@]}" \
        "$@"
else
    python -m src.evaluate \
        --roll-number "$ROLL_NUMBER" \
        "${EXTRA_ARGS[@]}" \
        "$@"
fi
