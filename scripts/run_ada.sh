#!/usr/bin/env bash
#SBATCH --job-name=anlp_a3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/part1/ada_%j.log

set -e
export PYTHONUNBUFFERED=1

# ==============================================================================
# ANLP Assignment 3 - Ada Cluster Runner
# Slurm Job Script for IIIT-H Ada Cluster
# Usage:
#   sbatch scripts/run_ada.sh
# Or run interactively:
#   bash scripts/run_ada.sh
# ==============================================================================

echo "=== Node Hostname: $(hostname) ==="
echo "=== GPU Allocation ==="
nvidia-smi

# Ensure uv is in PATH
if ! command -v uv >/dev/null 2>&1; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Load .env secrets if present
if [ -f ".env" ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
fi

HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"
WANDB_PROJECT="${WANDB_PROJECT:-anlp-a3-part1}"

HF_FLAGS=""
if [ -n "$HF_REPO_ID" ]; then
    HF_FLAGS="--push-to-hf --hf-repo-id $HF_REPO_ID"
fi

WANDB_FLAGS=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_FLAGS="--wandb-project $WANDB_PROJECT"
fi

echo "=== uv version: $(uv --version) ==="

# Sync virtual environment from pyproject.toml
echo "=== Syncing Environment via uv sync ==="
uv sync

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SITE_PKG" ]; then
    export LD_LIBRARY_PATH="$SITE_PKG/nvidia/cudnn/lib:$SITE_PKG/nvidia/nccl/lib:$SITE_PKG/nvidia/cublas/lib:$LD_LIBRARY_PATH"
fi

# Ensure local venv nvidia libraries take priority over any conflicting system libraries
VENV_NVIDIA_LIBS=$(find .venv -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
if [ -n "$VENV_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
fi

echo "=== Verifying CUDA Compatibility ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

mkdir -p logs/part1 models/part1 output/part1 results/part1

# Verify dataset
if [ ! -f "data/part1/train/cipher.jsonl" ]; then
    echo "========================================================================"
    echo "ERROR: data/part1/train/cipher.jsonl not found!"
    echo "Please ensure the dataset is placed in data/part1/ (train, validation, test)."
    echo "Example:"
    echo "  mkdir -p data/part1"
    echo "  cp -r /path/to/extracted/train data/part1/"
    echo "  cp -r /path/to/extracted/validation data/part1/"
    echo "  cp -r /path/to/extracted/test data/part1/"
    echo "========================================================================"
    exit 1
fi

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
ROLL_NUMBER="${ROLL_NUMBER:-2024101046}"
LR="${LR:-8e-4}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-100000}"
EXTRA_FLAGS="--max-train-samples $MAX_TRAIN_SAMPLES"

TARGET_MODEL="${1:-deltanet}"
if [ "$TARGET_MODEL" == "--all" ]; then
    RUN_MODELS=("transformer" "mamba" "deltanet" "hybrid")
else
    RUN_MODELS=("$TARGET_MODEL")
fi

echo "=== Target Models to Run: ${RUN_MODELS[*]} ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')"

for M in "${RUN_MODELS[@]}"; do
    echo "--- Training $M ---"
    python -m src.train \
        --model-type "$M" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --device cuda:0 \
        $WANDB_FLAGS --wandb-run-name "${M}_ada" $HF_FLAGS $EXTRA_FLAGS
        
    echo "--- Evaluating $M ---"
    python -m src.evaluate \
        --model-type "$M" \
        --checkpoint-path "models/part1/${M}_best.pt" \
        --roll-number "$ROLL_NUMBER" \
        --device cuda:0
done

echo ""
echo "========================================================================"
echo "=== All Models Finished Successfully! ==="
echo "========================================================================"
echo "Results:"
for f in results/part1/*.json; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        cat "$f"
        echo ""
    fi
done

