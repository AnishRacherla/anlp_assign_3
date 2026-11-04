```bash
# Environment Setup
uv sync

# Part 1: Training
bash scripts/part1/test_training.sh --model-type transformer
bash scripts/part1/test_training.sh --model-type mamba
bash scripts/part1/test_training.sh --model-type deltanet
bash scripts/part1/test_training.sh --model-type hybrid

# Part 1: Evaluation (Test Split)
bash scripts/part1/test_assn.sh --model-type transformer
bash scripts/part1/test_assn.sh --model-type mamba
bash scripts/part1/test_assn.sh --model-type deltanet
bash scripts/part1/test_assn.sh --model-type hybrid

# Part 1: Held-out Evaluation
bash scripts/part1/test_assn.sh --heldout

# Part 2: GNN Evaluation
bash ./scripts/part2/test_gnn.sh
bash ./scripts/part2/test_gnn.sh --ta
```
