#!/usr/bin/env bash
set -e

ROLL_NUMBER="${ROLL_NUMBER:-2024101046}"
HF_REPO_ID="${HF_REPO_ID:-AnishRacherla/anlp-a3-models}"
MODELS=("transformer" "mamba" "deltanet" "hybrid")

if [ -f ".env" ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
fi

if [ -d ".venv" ]; then
    VENV_NVIDIA_LIBS=$(find .venv -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | tr '\n' ':')
    if [ -n "$VENV_NVIDIA_LIBS" ]; then
        export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
    fi
fi

mkdir -p results/part1 output/part1

echo "========================================================================"
echo "=== Running Full Test & Heldout Evaluation for All 4 Models ==="
echo "Roll Number: $ROLL_NUMBER"
echo "HuggingFace Hub: $HF_REPO_ID"
echo "========================================================================"

for m in "${MODELS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo ">>> Evaluating Model: $m"
    echo "------------------------------------------------------------------------"

    EXTRA_FLAGS=()
    if [ -f "models/part1/${m}_best.pt" ]; then
        echo "Found local checkpoint: models/part1/${m}_best.pt"
        EXTRA_FLAGS+=(--checkpoint-path "models/part1/${m}_best.pt")
    else
        echo "Local checkpoint not found. Pulling from Hugging Face: $HF_REPO_ID"
        EXTRA_FLAGS+=(--hf-repo-id "$HF_REPO_ID")
    fi

    python -m src.evaluate \
        --model-type "$m" \
        --split test \
        --roll-number "$ROLL_NUMBER" \
        "${EXTRA_FLAGS[@]}"

    # Also generate held-out predictions
    if [ -d "data/part1/heldout" ]; then
        echo "Generating held-out decryption with $m..."
        python -m src.evaluate \
            --model-type "$m" \
            --heldout \
            --roll-number "$ROLL_NUMBER" \
            "${EXTRA_FLAGS[@]}" >/dev/null 2>&1 || true
    fi
done

echo ""
echo "========================================================================"
echo "=== ALL 4 EVALUATIONS COMPLETED ==="
echo "========================================================================"
echo ""

python -c "
import json
from pathlib import Path

models = ['transformer', 'mamba', 'deltanet', 'hybrid']
rows = []

for m in models:
    p = Path(f'results/part1/{m}_test_metrics.json')
    if not p.exists():
        p = Path(f'results/part1/{m}.json')
    if p.exists():
        d = json.loads(p.read_text())
        rows.append({
            'Model': m.capitalize(),
            'Params': f\"{d.get('total_parameters', 0):,}\",
            'GFLOPs': f\"{d.get('flops_per_sequence', 0) / 1e9:.2f}\",
            'Char Acc': f\"{d.get('character_accuracy', 0)*100:.2f}%\",
            'Word Acc': f\"{d.get('word_accuracy', 0)*100:.2f}%\",
            'F1': f\"{d.get('f1', 0)*100:.2f}%\",
            'Levenshtein': f\"{d.get('levenshtein_distance', 0):.2f}\",
        })

if rows:
    header = ['Model', 'Params', 'GFLOPs', 'Char Acc', 'Word Acc', 'F1', 'Levenshtein']
    print(f\"{header[0]:<14} {header[1]:<12} {header[2]:<9} {header[3]:<10} {header[4]:<10} {header[5]:<9} {header[6]:<12}\")
    print('-' * 80)
    for r in rows:
        print(f\"{r['Model']:<14} {r['Params']:<12} {r['GFLOPs']:<9} {r['Char Acc']:<10} {r['Word Acc']:<10} {r['F1']:<9} {r['Levenshtein']:<12}\")
else:
    print('No metrics files found in results/part1/')
"

echo ""
echo "Metrics files saved in: results/part1/"
echo "Predictions saved in:   output/part1/"
