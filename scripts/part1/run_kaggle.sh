#!/usr/bin/env bash
set -e

# ==============================================================================
# Kaggle Dual-GPU Parallel Training & Evaluation Runner
# GPU 0: Transformer -> Hybrid
# GPU 1: Mamba -> DeltaNet
# ==============================================================================

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/AnishRacherla/anlp_assign_3.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"
WANDB_PROJECT="${WANDB_PROJECT:-anlp-a3-part1}"
ROLL_NUMBER="${ROLL_NUMBER:-2024XXXXXX}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

if [ -f ".env" ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
fi

echo "=== System GPU Check ==="
nvidia-smi

# Setup working directory if not already inside repo
if [ ! -d "src" ] && [ -n "$GIT_REPO_URL" ]; then
    echo "Cloning repository from $GIT_REPO_URL ..."
    git clone -b "$GIT_BRANCH" "$GIT_REPO_URL" repo
    cd repo
fi

# Auto-link dataset from /kaggle/input if missing
if [ ! -f "data/part1/train/cipher.jsonl" ]; then
    echo "Dataset not found in data/part1/train/cipher.jsonl. Searching /kaggle/input/ ..."
    FOUND_CIPHER=$(find /kaggle/input -name "cipher.jsonl" 2>/dev/null | grep -E "train|part1" | head -n 1)
    if [ -z "$FOUND_CIPHER" ]; then
        FOUND_CIPHER=$(find /kaggle/input -name "cipher.jsonl" 2>/dev/null | head -n 1)
    fi
    if [ -n "$FOUND_CIPHER" ]; then
        DATA_SRC=$(dirname "$FOUND_CIPHER")
        echo "Found dataset near: $DATA_SRC"
        mkdir -p data/part1
        if [ -d "$DATA_SRC/train" ]; then
            cp -rn "$DATA_SRC"/* data/part1/ 2>/dev/null || true
        elif [ -d "$(dirname "$DATA_SRC")/train" ]; then
            cp -rn "$(dirname "$DATA_SRC")"/* data/part1/ 2>/dev/null || true
        else
            cp -rn "$DATA_SRC" data/part1/train 2>/dev/null || true
        fi
    fi
fi

export PYTHONUNBUFFERED=1

# Verify dataset exists before starting
if [ ! -f "data/part1/train/cipher.jsonl" ]; then
    echo "========================================================================"
    echo "ERROR: data/part1/train/cipher.jsonl does not exist!"
    echo "Please ensure the dataset is copied into data/part1/ before running."
    echo "Run the dataset setup cell first."
    echo "========================================================================"
    exit 1
fi

echo "=== Installing Dependencies ==="
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || true
pip install --quiet torch-geometric wandb huggingface-hub tqdm editdistance pyyaml scipy

mkdir -p logs/part1 models/part1 output/part1 results/part1

SAMPLE_FLAG=""
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    SAMPLE_FLAG="--max-train-samples $MAX_TRAIN_SAMPLES"
fi

HF_FLAG=""
if [ -n "$HF_REPO_ID" ]; then
    HF_FLAG="--push-to-hf --hf-repo-id $HF_REPO_ID"
fi

echo "=== Launching Dual-GPU Parallel Pipelines ==="

# ------------------------------------------------------------------------------
# GPU 0 Pipeline: Transformer -> Hybrid
# ------------------------------------------------------------------------------
(
    set -e
    echo "[GPU 0] Starting Model 1: Transformer..."
    if ! CUDA_VISIBLE_DEVICES=0 python -m src.train \
        --model-type transformer \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --device cuda:0 \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "transformer_gpu0" \
        $HF_FLAG $SAMPLE_FLAG > logs/part1/gpu0_transformer.log 2>&1; then
        echo "[GPU 0] ERROR: Transformer training failed! Recent log:"
        tail -n 25 logs/part1/gpu0_transformer.log
        exit 1
    fi

    echo "[GPU 0] Transformer training complete. Evaluating..."
    if ! CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
        --model-type transformer \
        --checkpoint-path models/part1/transformer_best.pt \
        --roll-number "$ROLL_NUMBER" \
        --device cuda:0 >> logs/part1/gpu0_transformer.log 2>&1; then
        echo "[GPU 0] ERROR: Transformer evaluation failed! Recent log:"
        tail -n 25 logs/part1/gpu0_transformer.log
        exit 1
    fi
    echo "[GPU 0] Transformer evaluated successfully."

    echo "[GPU 0] Starting Model 2: Hybrid (Mamba-DeltaNet)..."
    if ! CUDA_VISIBLE_DEVICES=0 python -m src.train \
        --model-type hybrid \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --device cuda:0 \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "hybrid_gpu0" \
        $HF_FLAG $SAMPLE_FLAG > logs/part1/gpu0_hybrid.log 2>&1; then
        echo "[GPU 0] ERROR: Hybrid training failed! Recent log:"
        tail -n 25 logs/part1/gpu0_hybrid.log
        exit 1
    fi

    echo "[GPU 0] Hybrid training complete. Evaluating..."
    if ! CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
        --model-type hybrid \
        --checkpoint-path models/part1/hybrid_best.pt \
        --roll-number "$ROLL_NUMBER" \
        --device cuda:0 >> logs/part1/gpu0_hybrid.log 2>&1; then
        echo "[GPU 0] ERROR: Hybrid evaluation failed! Recent log:"
        tail -n 25 logs/part1/gpu0_hybrid.log
        exit 1
    fi
    echo "[GPU 0] Hybrid evaluated successfully."
    echo "[GPU 0] All assigned models finished successfully!"
) &
PID_GPU0=$!

# ------------------------------------------------------------------------------
# GPU 1 Pipeline: Mamba -> DeltaNet
# ------------------------------------------------------------------------------
(
    set -e
    echo "[GPU 1] Starting Model 3: Mamba..."
    if ! CUDA_VISIBLE_DEVICES=1 python -m src.train \
        --model-type mamba \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --device cuda:0 \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "mamba_gpu1" \
        $HF_FLAG $SAMPLE_FLAG > logs/part1/gpu1_mamba.log 2>&1; then
        echo "[GPU 1] ERROR: Mamba training failed! Recent log:"
        tail -n 25 logs/part1/gpu1_mamba.log
        exit 1
    fi

    echo "[GPU 1] Mamba training complete. Evaluating..."
    if ! CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
        --model-type mamba \
        --checkpoint-path models/part1/mamba_best.pt \
        --roll-number "$ROLL_NUMBER" \
        --device cuda:0 >> logs/part1/gpu1_mamba.log 2>&1; then
        echo "[GPU 1] ERROR: Mamba evaluation failed! Recent log:"
        tail -n 25 logs/part1/gpu1_mamba.log
        exit 1
    fi
    echo "[GPU 1] Mamba evaluated successfully."

    echo "[GPU 1] Starting Model 4: DeltaNet..."
    if ! CUDA_VISIBLE_DEVICES=1 python -m src.train \
        --model-type deltanet \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --device cuda:0 \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-run-name "deltanet_gpu1" \
        $HF_FLAG $SAMPLE_FLAG > logs/part1/gpu1_deltanet.log 2>&1; then
        echo "[GPU 1] ERROR: DeltaNet training failed! Recent log:"
        tail -n 25 logs/part1/gpu1_deltanet.log
        exit 1
    fi

    echo "[GPU 1] DeltaNet training complete. Evaluating..."
    if ! CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
        --model-type deltanet \
        --checkpoint-path models/part1/deltanet_best.pt \
        --roll-number "$ROLL_NUMBER" \
        --device cuda:0 >> logs/part1/gpu1_deltanet.log 2>&1; then
        echo "[GPU 1] ERROR: DeltaNet evaluation failed! Recent log:"
        tail -n 25 logs/part1/gpu1_deltanet.log
        exit 1
    fi
    echo "[GPU 1] DeltaNet evaluated successfully."
    echo "[GPU 1] All assigned models finished successfully!"
) &
PID_GPU1=$!

echo "Processes launched in background (GPU 0: PID $PID_GPU0, GPU 1: PID $PID_GPU1)."
echo "Waiting for both pipelines to complete..."

wait $PID_GPU0
wait $PID_GPU1

echo ""
echo "========================================================================"
echo "=== All 4 Models Successfully Trained and Evaluated! ==="
echo "========================================================================"
echo ""
echo "=== RESULTS SUMMARY (results/part1/) ==="
for f in results/part1/*.json; do
    if [ -f "$f" ]; then
        echo "--- File: $f ---"
        cat "$f"
        echo ""
    fi
done

