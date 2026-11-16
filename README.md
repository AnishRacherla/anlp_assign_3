```bash
# Environment Setup
uv sync

# Pull Checkpoints from Hugging Face
bash scripts/part1/download_models.sh

# Part 1: Training
bash scripts/part1/test_training.sh --model transformer
bash scripts/part1/test_training.sh --model mamba
bash scripts/part1/test_training.sh --model deltanet
bash scripts/part1/test_training.sh --model combo

# Part 1: Evaluation (Test Split)
bash scripts/part1/test_assn.sh --model transformer
bash scripts/part1/test_assn.sh --model mamba
bash scripts/part1/test_assn.sh --model deltanet
bash scripts/part1/test_assn.sh --model combo

# Part 1: Held-out Evaluation (Default: Combo)
bash scripts/part1/test_assn.sh --heldout

# Part 1: Analysis (run after the evaluation commands above)
bash scripts/part1/eval_all.sh
uv run python scripts/part1/analyze_predictions.py
uv run python scripts/part1/noise_sweep.py
uv run python scripts/part1/measure_memory.py

# Part 2: GNN Evaluation
bash ./scripts/part2/test_gnn.sh
bash ./scripts/part2/test_gnn.sh --ta
bash ./scripts/part2/test_gnn.sh --use-cached
```
