#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO_DIR/models/part1"

HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"

if [[ "$(hostname 2>/dev/null)" =~ (ada|gnode) ]] || [ -d "/home2" ]; then
    export http_proxy="${http_proxy:-http://proxy.iiit.ac.in:8080}"
    export https_proxy="${https_proxy:-http://proxy.iiit.ac.in:8080}"
    export HTTP_PROXY="${HTTP_PROXY:-http://proxy.iiit.ac.in:8080}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.iiit.ac.in:8080}"
fi

RUN_CMD="python"
if [ -f "$REPO_DIR/.venv/bin/python" ]; then
    RUN_CMD="$REPO_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    RUN_CMD="uv run python"
fi

$RUN_CMD -c "
import shutil, os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = '$HF_REPO_ID'
models = ['hybrid_best.pt', 'transformer_best.pt', 'mamba_best.pt', 'deltanet_best.pt']
dest_dir = Path('$REPO_DIR/models/part1')
dest_dir.mkdir(parents=True, exist_ok=True)

print(f'Pulling models from Hugging Face Hub: {repo}')
for m in models:
    try:
        print(f'  Downloading {m}...')
        downloaded = hf_hub_download(repo_id=repo, filename=m)
        target = dest_dir / m
        shutil.copyfile(downloaded, target)
        size_mb = os.path.getsize(target) / (1024 * 1024)
        print(f'  Saved to {target} ({size_mb:.1f} MB)')
    except Exception as e:
        print(f'  Warning: Failed to download {m}: {e}')

print('All available models downloaded successfully!')
"
