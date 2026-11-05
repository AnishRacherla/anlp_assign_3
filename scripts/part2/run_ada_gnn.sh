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
mkdir -p logs/part2

echo "=== Hostname: $(hostname) ==="
nvidia-smi || true

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

python -m src.gnn_pipeline --epochs 100 --lr 0.005 --batch-size 64 --device cuda
