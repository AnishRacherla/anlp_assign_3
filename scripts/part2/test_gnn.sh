#!/usr/bin/env bash
set -e

uv run python -m src.gnn_pipeline "$@"

