#!/usr/bin/env bash
#SBATCH --job-name=eval_part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/part1/eval_%j.log
#SBATCH --error=logs/part1/eval_%j.log

set -e
export PYTHONUNBUFFERED=1
mkdir -p logs/part1 results/part1 output/part1

echo "=== Hostname: $(hostname) ==="
nvidia-smi || true

MODEL="${1:-deltanet}"

if [ "$MODEL" == "--all" ]; then
    echo "=== Running Full Evaluation on All 4 Models ==="
    bash scripts/part1/eval_all.sh
else
    echo "=== Running Evaluation for Model: $MODEL ==="
    bash scripts/part1/test_assn.sh --model-type "$MODEL" --checkpoint-path "models/part1/${MODEL}_best.pt" "${@:2}"
fi
