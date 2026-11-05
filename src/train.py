import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .dataset import get_dataloader
from .deltanet import DeltaNet
from .hub import init_wandb, push_to_hub
from .hybrid import HybridMambaDeltaNet
from .mamba import Mamba
from .transformer import Transformer
from .vocab import CharTokenizer, NUM_TARGET_CLASSES, PAD_TARGET_ID

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

def get_model(
    model_type: str,
    vocab_size: int,
    num_classes: int = NUM_TARGET_CLASSES,
    d_model: int = 192,
    n_layers: int = 8,
    max_seq_len: int = 512,
) -> nn.Module:
    if model_type == "transformer":
        return Transformer(vocab_size=vocab_size, num_classes=num_classes, d_model=d_model, n_layers=n_layers, max_seq_len=max_seq_len)
    elif model_type == "mamba":
        return Mamba(vocab_size=vocab_size, num_classes=num_classes, d_model=d_model, n_layers=n_layers, max_seq_len=max_seq_len)
    elif model_type == "deltanet":
        return DeltaNet(vocab_size=vocab_size, num_classes=num_classes, d_model=d_model, n_layers=n_layers, max_seq_len=max_seq_len)
    elif model_type == "hybrid":
        return HybridMambaDeltaNet(vocab_size=vocab_size, num_classes=num_classes, d_model=d_model, n_layers=n_layers, max_seq_len=max_seq_len)
    raise ValueError(f"Unknown model_type: {model_type}")

def lr_schedule(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, device: str) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:
            continue
        if labels is None:
            continue
        if hasattr(model, "max_seq_len") and inputs.size(1) > model.max_seq_len:
            inputs = inputs[:, : model.max_seq_len]
            labels = labels[:, : model.max_seq_len]
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        b, t, v = logits.shape
        loss = F.cross_entropy(logits.view(-1, v), labels.view(-1), ignore_index=PAD_TARGET_ID, reduction="sum")
        n_tokens = (labels != PAD_TARGET_ID).sum().item()
        total_loss += loss.item()
        total_tokens += max(1, n_tokens)
    model.train()
    return total_loss / max(1, total_tokens)

def train(
    model_type: str = "transformer",
    train_cipher: str = "data/part1/train/cipher.jsonl",
    train_cleaned: str = "data/part1/train/cleaned.jsonl",
    val_cipher: str = "data/part1/validation/cipher.jsonl",
    val_cleaned: str = "data/part1/validation/cleaned.jsonl",
    d_model: int = 192,
    n_layers: int = 8,
    max_seq_len: int = 512,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    checkpoint_dir: str = "models/part1",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    push_hf: bool = False,
    hf_repo_id: Optional[str] = None,
    device: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}) but torch.cuda.is_available() is False!")
        print(f"GPU device: {torch.cuda.get_device_name(device)}")

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size

    train_loader = get_dataloader(
        cipher_path=train_cipher,
        cleaned_path=train_cleaned,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_seq_len,
        shuffle=True,
        is_train=True,
        max_samples=max_train_samples,
    )
    val_loader = get_dataloader(
        cipher_path=val_cipher,
        cleaned_path=val_cleaned,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_seq_len,
        shuffle=False,
        is_train=False,
        max_samples=max_val_samples or 1000,
    )

    model = get_model(
        model_type=model_type,
        vocab_size=vocab_size,
        num_classes=NUM_TARGET_CLASSES,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = epochs * len(train_loader)

    use_wandb = wandb_project is not None and _HAS_WANDB
    if use_wandb:
        wb_run = init_wandb(
            project=wandb_project,
            name=wandb_run_name or f"{model_type}_run",
            config={
                "model_type": model_type,
                "d_model": d_model,
                "n_layers": n_layers,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
            },
        )
        use_wandb = wb_run is not None

    is_cuda = "cuda" in str(device)
    dev_type = "cuda" if is_cuda else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"{model_type}_best.pt"

    step = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=dev_type, enabled=is_cuda):
                logits = model(inputs)
                b, t, v = logits.shape
                loss = F.cross_entropy(logits.view(-1, v), labels.view(-1), ignore_index=PAD_TARGET_ID)

            cur_lr = lr * lr_schedule(step, warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

            if use_wandb and step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": cur_lr}, step=step)

        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        if use_wandb:
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_type": model_type,
                    "vocab_size": vocab_size,
                    "num_classes": NUM_TARGET_CLASSES,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "max_seq_len": max_seq_len,
                },
            }
            torch.save(ckpt_data, best_ckpt_path)
            print(f"Saved best model to {best_ckpt_path}")

    if push_hf and hf_repo_id and best_ckpt_path.exists():
        try:
            push_to_hub(
                local_path=str(best_ckpt_path),
                repo_id=hf_repo_id,
                filename=f"{model_type}_best.pt",
            )
        except Exception as e:
            print(f"Warning: push_to_hub failed: {e}")

    if use_wandb:
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "mamba", "deltanet", "hybrid"])
    p.add_argument("--train-cipher", type=str, default="data/part1/train/cipher.jsonl")
    p.add_argument("--train-cleaned", type=str, default="data/part1/train/cleaned.jsonl")
    p.add_argument("--val-cipher", type=str, default="data/part1/validation/cipher.jsonl")
    p.add_argument("--val-cleaned", type=str, default="data/part1/validation/cleaned.jsonl")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--checkpoint-dir", type=str, default="models/part1")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--push-to-hf", action="store_true")
    p.add_argument("--hf-repo-id", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        model_type=args.model_type,
        train_cipher=args.train_cipher,
        train_cleaned=args.train_cleaned,
        val_cipher=args.val_cipher,
        val_cleaned=args.val_cleaned,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        push_hf=args.push_to_hf,
        hf_repo_id=args.hf_repo_id,
        device=args.device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
