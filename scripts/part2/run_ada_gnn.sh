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
mkdir -p logs/part2 output/part2 results/part2

echo "=== Hostname: $(hostname) ==="
nvidia-smi || true

echo "=== Running Part 2 GNN Pipeline on GPU ==="
bash scripts/part2/test_gnn.sh "$@"
