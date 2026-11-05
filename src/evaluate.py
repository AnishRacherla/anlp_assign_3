import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

for _p in sys.path:
    if "site-packages" in _p:
        for _so in Path(_p).glob("nvidia/*/lib/*.so*"):
            try:
                ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

import torch
import torch.nn as nn
from tqdm import tqdm

from .dataset import get_dataloader
from .deltanet import DeltaNet
from .hub import pull_from_hub
from .hybrid import HybridMambaDeltaNet
from .mamba import Mamba
from .metrics import count_parameters, measure_flops_forward, evaluate_predictions
from .transformer import Transformer
from .vocab import CharTokenizer, NUM_TARGET_CLASSES

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

def load_checkpoint(
    checkpoint_path: str,
    device: str,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
):
    ckpt_file = Path(checkpoint_path)
    if (not ckpt_file.exists() or hf_repo_id) and hf_repo_id:
        try:
            filename = hf_filename or ckpt_file.name
            downloaded = pull_from_hub(repo_id=hf_repo_id, filename=filename)
            ckpt_file = Path(downloaded)
        except Exception as e:
            if not ckpt_file.exists():
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} and HuggingFace download failed: {e}")

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}. Ensure models are trained or pulled from Hugging Face.")

    try:
        raw = torch.load(ckpt_file, map_location=device, weights_only=True)
    except Exception:
        raw = torch.load(ckpt_file, map_location=device, weights_only=False)

    config = {}
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        config = raw.get("config", {})
    else:
        state_dict = raw
    return state_dict, config

def run_evaluation(
    model_type: str = "hybrid",
    checkpoint_path: Optional[str] = None,
    cipher_path: str = "data/part1/test/cipher.jsonl",
    cleaned_path: Optional[str] = "data/part1/test/cleaned.jsonl",
    split: str = "test",
    heldout: bool = False,
    heldout_path: str = "data/part1/heldout",
    roll_number: str = "2023XXXX",
    output_dir: str = "output/part1",
    results_dir: str = "results/part1",
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
    d_model: int = 192,
    n_layers: int = 8,
    max_seq_len: int = 512,
    batch_size: int = 32,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    if checkpoint_path is None:
        checkpoint_path = f"models/part1/{model_type}_best.pt"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hf_repo_id is None:
        hf_repo_id = os.environ.get("HF_REPO_ID", "AnishRacherla/anlp-a3-models")

    state_dict, config = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
    )

    d_model = config.get("d_model", d_model)
    n_layers = config.get("n_layers", n_layers)
    max_seq_len = config.get("max_seq_len", max_seq_len)
    model_type = config.get("model_type", model_type)

    tokenizer = CharTokenizer()
    vocab_size = tokenizer.vocab_size

    model = get_model(
        model_type=model_type,
        vocab_size=vocab_size,
        num_classes=NUM_TARGET_CLASSES,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if heldout:
        target_cipher = Path(heldout_path)
        if target_cipher.is_dir():
            candidates = list(target_cipher.glob("*.jsonl"))
            target_cipher = candidates[0] if candidates else target_cipher / "cipher.jsonl"
        cleaned_path = None
        eval_loader = get_dataloader(
            cipher_path=target_cipher,
            cleaned_path=None,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_seq_len,
            shuffle=False,
            is_train=False,
            max_samples=max_samples,
        )
    else:
        eval_loader = get_dataloader(
            cipher_path=cipher_path,
            cleaned_path=cleaned_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_seq_len,
            shuffle=False,
            is_train=False,
            max_samples=max_samples,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if heldout:
        out_filename = output_dir / f"heldout_result_{roll_number}.jsonl"
    else:
        out_filename = output_dir / f"{split}_decrypt.jsonl"

    references = []
    hypotheses = []

    with open(out_filename, "w", encoding="utf-8") as f_out:
        for inputs, labels, ids, raw_ciphers, raw_plains, lengths in tqdm(eval_loader, desc=f"Evaluating {model_type}"):
            inputs = inputs.to(device)
            batch_preds = model.predict(inputs, lengths=lengths)
            hypotheses.extend(batch_preds)
            if raw_plains and raw_plains[0]:
                references.extend(raw_plains)

            for id_val, hyp in zip(ids, batch_preds):
                f_out.write(json.dumps({"id": id_val, "text": hyp}, ensure_ascii=False) + "\n")
            hypotheses.extend(batch_preds)
            if raw_plains and raw_plains[0]:
                references.extend(raw_plains)

            for id_val, hyp in zip(ids, batch_preds):
                f_out.write(json.dumps({"id": id_val, "text": hyp}, ensure_ascii=False) + "\n")

    print(f"Decrypted outputs saved to {out_filename}")

    metrics = {}
    if references and len(references) == len(hypotheses):
        metrics = evaluate_predictions(references, hypotheses)
        total_p, non_emb_p = count_parameters(model)
        sample_tensor = torch.randint(0, vocab_size, (1, min(max_seq_len, 256)), device=device)
        flops = measure_flops_forward(model, sample_tensor)

        metrics["total_parameters"] = total_p
        metrics["non_embedding_parameters"] = non_emb_p
        metrics["flops_per_sequence"] = flops

        res_path = results_dir / f"{model_type}_{split}_metrics.json"
        with open(res_path, "w", encoding="utf-8") as f_res:
            json.dump(metrics, f_res, indent=2)

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:,}")

    return metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", type=str, default="hybrid", choices=["transformer", "mamba", "deltanet", "hybrid"])
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--cipher-path", type=str, default="data/part1/test/cipher.jsonl")
    p.add_argument("--cleaned-path", type=str, default="data/part1/test/cleaned.jsonl")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--heldout-path", type=str, default="data/part1/heldout")
    p.add_argument("--roll-number", type=str, default="2023XXXX")
    p.add_argument("--output-dir", type=str, default="output/part1")
    p.add_argument("--results-dir", type=str, default="results/part1")
    p.add_argument("--hf-repo-id", type=str, default="AnishRacherla/anlp-a3-models")
    p.add_argument("--hf-filename", type=str, default=None)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        cipher_path=args.cipher_path,
        cleaned_path=args.cleaned_path,
        split=args.split,
        heldout=args.heldout,
        heldout_path=args.heldout_path,
        roll_number=args.roll_number,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        hf_repo_id=args.hf_repo_id,
        hf_filename=args.hf_filename,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
