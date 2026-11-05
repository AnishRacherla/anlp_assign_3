#!/usr/bin/env bash
#SBATCH --job-name=gnn_upfd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/part2/gnn_%j.log
#SBATCH --error=logs/part2/gnn_%j.log

set -e
export PYTHONUNBUFFERED=1
mkdir -p logs/part2

echo "=== Hostname: $(hostname) ==="
nvidia-smi || true

if ! command -v uv >/dev/null 2>&1; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

echo "=== Syncing venv via uv sync ==="
uv sync

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

VENV_NVIDIA_LIBS=$(find .venv -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
fi

echo "=== Verifying CUDA Compatibility ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== Running GNN Pipeline on GPU ==="
python -m src.gnn_pipeline --epochs 100 --lr 0.005 --batch-size 64 --device cuda
