#!/usr/bin/env bash
set -e

if command -v uv >/dev/null 2>&1; then
    uv run python -m src.gnn_pipeline "$@"
else
    python -m src.gnn_pipeline "$@"
fi

