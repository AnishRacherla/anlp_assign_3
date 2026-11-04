#!/usr/bin/env bash
set -e

ROLL_NUMBER="${ROLL_NUMBER:-2024XXXXXX}"

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
