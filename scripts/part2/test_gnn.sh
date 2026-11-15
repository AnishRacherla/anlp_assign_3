#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(hostname 2>/dev/null)" =~ (ada|gnode) ]] || [ -d "/home2" ]; then
    export http_proxy="${http_proxy:-http://proxy.iiit.ac.in:8080}"
    export https_proxy="${https_proxy:-http://proxy.iiit.ac.in:8080}"
    export HTTP_PROXY="${HTTP_PROXY:-http://proxy.iiit.ac.in:8080}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.iiit.ac.in:8080}"
fi

if [ -d "$REPO_DIR/.venv" ]; then
    if ! find "$REPO_DIR/.venv" -name "libcudnn.so.9" 2>/dev/null | grep -q .; then
        echo "[test_gnn.sh] Installing missing nvidia-cudnn-cu12 into virtual environment..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
        elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
            "$REPO_DIR/.venv/bin/pip" install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
        fi
    fi

    VENV_NVIDIA_LIBS=$(find "$REPO_DIR/.venv" -type d -name "lib" 2>/dev/null | grep -E "site-packages/nvidia|site-packages/torch" | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi
fi

if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    "$REPO_DIR/.venv/bin/python" -m src.gnn_pipeline "$@"
elif command -v uv >/dev/null 2>&1; then
    uv run python -m src.gnn_pipeline "$@"
else
    python -m src.gnn_pipeline "$@"
fi

