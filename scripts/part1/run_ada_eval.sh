#!/usr/bin/env bash
#SBATCH --job-name=eval_part1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/part1/eval_%j.log
#SBATCH --error=logs/part1/eval_%j.log

set -e
export PYTHONUNBUFFERED=1
mkdir -p logs/part1

echo "=== Hostname: $(hostname) ==="
nvidia-smi || true

bash scripts/part1/test_assn.sh
