#!/usr/bin/env bash
set -e

ROLL_NUMBER="${ROLL_NUMBER:-2024XXXXXX}"
HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"

EXTRA_ARGS=()
if [ -n "$HF_REPO_ID" ]; then
    EXTRA_ARGS+=(--hf-repo-id "$HF_REPO_ID")
fi
if [ -n "$MODEL_TYPE" ]; then
    EXTRA_ARGS+=(--model-type "$MODEL_TYPE")
fi
if [ -n "$CHECKPOINT_PATH" ]; then
    EXTRA_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
fi

if [ -d ".venv" ]; then
    VENV_NVIDIA_LIBS=$(find .venv -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi
fi

if command -v uv >/dev/null 2>&1; then
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
