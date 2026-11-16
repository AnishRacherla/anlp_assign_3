#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

find "$REPO_DIR/src" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$REPO_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [[ "$(hostname 2>/dev/null)" =~ (ada|gnode) ]] || [ -d "/home2" ]; then
    export http_proxy="${http_proxy:-http://proxy.iiit.ac.in:8080}"
    export https_proxy="${https_proxy:-http://proxy.iiit.ac.in:8080}"
    export HTTP_PROXY="${HTTP_PROXY:-http://proxy.iiit.ac.in:8080}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.iiit.ac.in:8080}"
fi

SEARCH_DIRS=()
if [ -n "$VIRTUAL_ENV" ] && [ -d "$VIRTUAL_ENV" ]; then
    SEARCH_DIRS+=("$VIRTUAL_ENV")
fi
if [ -d "$REPO_DIR/.venv" ]; then
    SEARCH_DIRS+=("$REPO_DIR/.venv")
fi
SYS_PREFIX="$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [ -n "$SYS_PREFIX" ] && [ -d "$SYS_PREFIX" ]; then
    SEARCH_DIRS+=("$SYS_PREFIX")
fi

FOUND_NCCL=""
FOUND_CUDNN=""
for sdir in "${SEARCH_DIRS[@]}"; do
    if [ -z "$FOUND_CUDNN" ]; then
        FOUND_CUDNN=$(find "$sdir" -name "libcudnn.so.9" 2>/dev/null | head -n 1)
    fi
    if [ -z "$FOUND_NCCL" ]; then
        FOUND_NCCL=$(find "$sdir" -name "libnccl.so*" 2>/dev/null | grep -E "libnccl\.so(\.[0-9]+)*$" | head -n 1)
    fi
done

if [ -z "$FOUND_CUDNN" ]; then
    echo "[test_gnn.sh] Installing missing nvidia-cudnn-cu12..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
    elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
        "$REPO_DIR/.venv/bin/pip" install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
    elif command -v pip >/dev/null 2>&1; then
        pip install --upgrade "nvidia-cudnn-cu12>=9.1.0" || true
    fi
fi

if [ -z "$FOUND_NCCL" ]; then
    echo "[test_gnn.sh] Installing missing nvidia-nccl-cu12..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --reinstall "nvidia-nccl-cu12" || true
    elif [ -f "$REPO_DIR/.venv/bin/pip" ]; then
        "$REPO_DIR/.venv/bin/pip" install --force-reinstall "nvidia-nccl-cu12" || true
    elif command -v pip >/dev/null 2>&1; then
        pip install --force-reinstall "nvidia-nccl-cu12" || true
    fi
fi

for sdir in "${SEARCH_DIRS[@]}"; do
    VENV_NVIDIA_LIBS=$(find "$sdir" -type d -name "lib" 2>/dev/null | grep -E "site-packages/nvidia|site-packages/torch" | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi
    if [ -z "$FOUND_NCCL" ]; then
        FOUND_NCCL=$(find "$sdir" -name "libnccl.so*" 2>/dev/null | grep -E "libnccl\.so(\.[0-9]+)*$" | head -n 1)
    fi
done

if [ -n "$FOUND_NCCL" ]; then
    export LD_PRELOAD="${FOUND_NCCL}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    "$REPO_DIR/.venv/bin/python" -m src.gnn_pipeline "$@"
elif command -v uv >/dev/null 2>&1; then
    uv run python -m src.gnn_pipeline "$@"
else
    python -m src.gnn_pipeline "$@"
fi
