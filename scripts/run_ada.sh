#!/usr/bin/env bash
#SBATCH --job-name=anlp_a3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/part1/ada_%j.out
#SBATCH --error=logs/part1/ada_%j.err

set -e

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

echo "=== uv version: $(uv --version) ==="

# Sync virtual environment from pyproject.toml
echo "=== Syncing Environment via uv sync ==="
uv sync

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

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
ROLL_NUMBER="${ROLL_NUMBER:-2024XXXXXX}"

EXTRA_FLAGS=""
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    EXTRA_FLAGS="--max-train-samples $MAX_TRAIN_SAMPLES"
fi

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "Detected GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "=== Running Dual-GPU Parallel Pipelines (GPU 0: Transformer+Hybrid, GPU 1: Mamba+DeltaNet) ==="
    
    (
        set -e
        echo "[GPU 0] Training Transformer..."
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.train \
            --model-type transformer --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" --device cuda:0 $EXTRA_FLAGS > logs/part1/ada_gpu0_transformer.log 2>&1
        echo "[GPU 0] Evaluating Transformer..."
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.evaluate \
            --model-type transformer --checkpoint-path models/part1/transformer_best.pt \
            --roll-number "$ROLL_NUMBER" --device cuda:0 >> logs/part1/ada_gpu0_transformer.log 2>&1

        echo "[GPU 0] Training Hybrid (Mamba-DeltaNet)..."
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.train \
            --model-type hybrid --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" --device cuda:0 $EXTRA_FLAGS > logs/part1/ada_gpu0_hybrid.log 2>&1
        echo "[GPU 0] Evaluating Hybrid..."
        CUDA_VISIBLE_DEVICES=0 uv run python -m src.evaluate \
            --model-type hybrid --checkpoint-path models/part1/hybrid_best.pt \
            --roll-number "$ROLL_NUMBER" --device cuda:0 >> logs/part1/ada_gpu0_hybrid.log 2>&1
    ) &
    PID_0=$!

    (
        set -e
        echo "[GPU 1] Training Mamba..."
        CUDA_VISIBLE_DEVICES=1 uv run python -m src.train \
            --model-type mamba --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" --device cuda:0 $EXTRA_FLAGS > logs/part1/ada_gpu1_mamba.log 2>&1
        echo "[GPU 1] Evaluating Mamba..."
        CUDA_VISIBLE_DEVICES=1 uv run python -m src.evaluate \
            --model-type mamba --checkpoint-path models/part1/mamba_best.pt \
            --roll-number "$ROLL_NUMBER" --device cuda:0 >> logs/part1/ada_gpu1_mamba.log 2>&1

        echo "[GPU 1] Training DeltaNet..."
        CUDA_VISIBLE_DEVICES=1 uv run python -m src.train \
            --model-type deltanet --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" --device cuda:0 $EXTRA_FLAGS > logs/part1/ada_gpu1_deltanet.log 2>&1
        echo "[GPU 1] Evaluating DeltaNet..."
        CUDA_VISIBLE_DEVICES=1 uv run python -m src.evaluate \
            --model-type deltanet --checkpoint-path models/part1/deltanet_best.pt \
            --roll-number "$ROLL_NUMBER" --device cuda:0 >> logs/part1/ada_gpu1_deltanet.log 2>&1
    ) &
    PID_1=$!

    wait $PID_0
    wait $PID_1
else
    echo "=== Running Single-GPU Sequential Pipeline ==="
    MODELS=("transformer" "mamba" "deltanet" "hybrid")
    for M in "${MODELS[@]}"; do
        echo "--- Training $M ---"
        uv run python -m src.train \
            --model-type "$M" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --max-seq-len "$MAX_SEQ_LEN" \
            $EXTRA_FLAGS
            
        echo "--- Evaluating $M ---"
        uv run python -m src.evaluate \
            --model-type "$M" \
            --checkpoint-path "models/part1/${M}_best.pt" \
            --roll-number "$ROLL_NUMBER"
    done
fi

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

