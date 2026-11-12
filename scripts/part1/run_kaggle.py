import os
import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd: str, log_file: Path):
    print(f"[RUNNING] {cmd}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== CMD: {cmd} ===\n")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc

def main():
    gpu_count = 0
    try:
        import torch
        gpu_count = torch.cuda.device_count()
    except Exception:
        pass
    print(f"Detected {gpu_count} CUDA GPUs.")

    epochs = os.environ.get("EPOCHS", "3")
    batch_size = os.environ.get("BATCH_SIZE", "32")
    max_seq_len = os.environ.get("MAX_SEQ_LEN", "512")
    wandb_proj = os.environ.get("WANDB_PROJECT", "anlp-a3-part1")
    hf_repo = os.environ.get("HF_REPO_ID", "")
    max_samples = os.environ.get("MAX_TRAIN_SAMPLES", "50000")

    sample_flag = f"--max-train-samples {max_samples}" if max_samples else ""
    hf_flag = f"--push-to-hf --hf-repo-id {hf_repo}" if hf_repo else ""

    Path("logs/part1").mkdir(parents=True, exist_ok=True)
    Path("models/part1").mkdir(parents=True, exist_ok=True)
    Path("output/part1").mkdir(parents=True, exist_ok=True)
    Path("results/part1").mkdir(parents=True, exist_ok=True)

    gpu0_cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python -m src.train --model-type transformer "
        f"--epochs {epochs} --batch-size {batch_size} --max-seq-len {max_seq_len} "
        f"--device cuda:0 --wandb-project {wandb_proj} --wandb-run-name transformer_gpu0 "
        f"{hf_flag} {sample_flag} && "
        f"CUDA_VISIBLE_DEVICES=0 python -m src.evaluate --model-type transformer "
        f"--checkpoint-path models/part1/transformer_best.pt --device cuda:0 && "
        f"CUDA_VISIBLE_DEVICES=0 python -m src.train --model-type hybrid "
        f"--epochs {epochs} --batch-size {batch_size} --max-seq-len {max_seq_len} "
        f"--device cuda:0 --wandb-project {wandb_proj} --wandb-run-name hybrid_gpu0 "
        f"{hf_flag} {sample_flag} && "
        f"CUDA_VISIBLE_DEVICES=0 python -m src.evaluate --model-type hybrid "
        f"--checkpoint-path models/part1/hybrid_best.pt --device cuda:0"
    )

    gpu1_device = "1" if gpu_count >= 2 else "0"
    gpu1_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu1_device} python -m src.train --model-type mamba "
        f"--epochs {epochs} --batch-size {batch_size} --max-seq-len {max_seq_len} "
        f"--device cuda:0 --wandb-project {wandb_proj} --wandb-run-name mamba_gpu1 "
        f"{hf_flag} {sample_flag} && "
        f"CUDA_VISIBLE_DEVICES={gpu1_device} python -m src.evaluate --model-type mamba "
        f"--checkpoint-path models/part1/mamba_best.pt --device cuda:0 && "
        f"CUDA_VISIBLE_DEVICES={gpu1_device} python -m src.train --model-type deltanet "
        f"--epochs {epochs} --batch-size {batch_size} --max-seq-len {max_seq_len} "
        f"--device cuda:0 --wandb-project {wandb_proj} --wandb-run-name deltanet_gpu1 "
        f"{hf_flag} {sample_flag} && "
        f"CUDA_VISIBLE_DEVICES={gpu1_device} python -m src.evaluate --model-type deltanet "
        f"--checkpoint-path models/part1/deltanet_best.pt --device cuda:0"
    )

    print("Starting GPU 0: Transformer -> Hybrid")
    p0 = run_cmd(gpu0_cmd, Path("logs/part1/gpu0_pipeline.log"))

    print(f"Starting GPU {gpu1_device}: Mamba -> DeltaNet")
    p1 = run_cmd(gpu1_cmd, Path("logs/part1/gpu1_pipeline.log"))

    p0.wait()
    p1.wait()

    print("\n=== All Pipelines Finished ===")
    print("GPU 0 Exit Code:", p0.returncode)
    print("GPU 1 Exit Code:", p1.returncode)

    if p0.returncode != 0:
        print("[ERROR] Check logs/part1/gpu0_pipeline.log for details.")
    if p1.returncode != 0:
        print("[ERROR] Check logs/part1/gpu1_pipeline.log for details.")

    if p0.returncode == 0 and p1.returncode == 0:
        print("Success! Generated metrics:")
        for res_file in Path("results/part1").glob("*.json"):
            print(f" - {res_file.name}")

if __name__ == "__main__":
    main()

