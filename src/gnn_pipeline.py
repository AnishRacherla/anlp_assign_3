import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from .gnn_models import build_gnn_model

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

SEEDS = [42, 1337, 2024, 7, 99]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(y_true == y_pred))
    classes = np.unique(y_true)
    if len(classes) <= 2:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        precs, recs, f1s = [], [], []
        for c in classes:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            precs.append(p)
            recs.append(r)
            f1s.append(f)
        prec = float(np.mean(precs))
        rec = float(np.mean(recs))
        f1 = float(np.mean(f1s))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

@torch.no_grad()
def evaluate_gnn(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        ptr = getattr(batch, "ptr", None)
        logits = model(batch.x, batch.edge_index, batch.batch, ptr=ptr)
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch.y.cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)
    return compute_classification_metrics(np.array(all_targets), np.array(all_preds))

def train_single_run(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    in_channels: int,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 0.001,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, float]:
    set_seed(seed)
    model = build_gnn_model(model_name, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            ptr = getattr(batch, "ptr", None)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, ptr=ptr)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_gnn(model, val_loader, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_metrics = evaluate_gnn(model, test_loader, device)
    return test_metrics

def run_experiment_across_seeds(
    model_name: str,
    feature_name: str,
    root_data_dir: str,
    epochs: int,
    batch_size: int,
    device: str,
    seeds: List[int],
    lr: float = 0.005,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    transform = None if model_name.lower() == "bigcn" else ToUndirected()
    train_data = UPFD(root=root_data_dir, name="gossipcop", feature=feature_name, split="train", transform=transform)
    val_data = UPFD(root=root_data_dir, name="gossipcop", feature=feature_name, split="val", transform=transform)
    test_data = UPFD(root=root_data_dir, name="gossipcop", feature=feature_name, split="test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    in_channels = train_data.num_features

    seed_results = []
    for s in seeds:
        res = train_single_run(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_channels=in_channels,
            epochs=epochs,
            lr=lr,
            device=device,
            seed=s,
        )
        seed_results.append(res)

    means = {}
    stds = {}
    for metric_key in ["accuracy", "precision", "recall", "f1"]:
        vals = [r[metric_key] for r in seed_results]
        means[metric_key] = round(float(np.mean(vals)), 4)
        stds[metric_key] = round(float(np.std(vals)), 4)

    return means, stds

def run_all(
    ta_mode: bool = False,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.005,
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = "data/part2/upfd"
    out_dir = Path("output/part2")
    res_dir = Path("results/part2")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    results_template = {
        "part_a": {
            "mlp": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "gcn": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "sage": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "gat": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "bigcn": {"accuracy": None, "precision": None, "recall": None, "f1": None},
        },
        "part_b": {
            "profile": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "spacy": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "bert": {"accuracy": None, "precision": None, "recall": None, "f1": None},
            "content": {"accuracy": None, "precision": None, "recall": None, "f1": None},
        },
    }

    detailed_report = {"part_a": {}, "part_b": {}}

    models_part_a = ["mlp", "gcn", "sage", "gat", "bigcn"]
    for m in models_part_a:
        print(f"\n[ARCH EXPERIMENT] Running {m.upper()} on BERT features across 5 seeds...")
        means, stds = run_experiment_across_seeds(
            model_name=m,
            feature_name="bert",
            root_data_dir=root_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seeds=SEEDS,
        )
        results_template["part_a"][m] = means
        detailed_report["part_a"][m] = {"mean": means, "std": stds}
        print(f"  {m.upper()} Mean: {means}")
        print(f"  {m.upper()} Std:  {stds}")

    features_part_b = ["profile", "spacy", "bert", "content"]
    for f in features_part_b:
        if f == "bert" and "sage" in detailed_report["part_a"]:
            means = results_template["part_a"]["sage"]
            stds = detailed_report["part_a"]["sage"]["std"]
            results_template["part_b"][f] = means
            detailed_report["part_b"][f] = {"mean": means, "std": stds}
            print(f"\n[FEATURE ABLATION] GraphSAGE (BERT) reused from architecture comparison: {means}")
            continue

        print(f"\n[FEATURE ABLATION] Running GraphSAGE on {f.upper()} features across 5 seeds...")
        means, stds = run_experiment_across_seeds(
            model_name="sage",
            feature_name=f,
            root_data_dir=root_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seeds=SEEDS,
        )
        results_template["part_b"][f] = means
        detailed_report["part_b"][f] = {"mean": means, "std": stds}
        print(f"  GraphSAGE ({f.upper()}) Mean: {means}")
        print(f"  GraphSAGE ({f.upper()}) Std:  {stds}")

    target_json = out_dir / ("results_ta.json" if ta_mode else "results.json")
    with open(target_json, "w", encoding="utf-8") as f_out:
        json.dump(results_template, f_out, indent=2)
    print(f"\nSaved required results template to: {target_json}")

    detailed_json = res_dir / "gnn_metrics_detailed.json"
    with open(detailed_json, "w", encoding="utf-8") as f_det:
        json.dump(detailed_report, f_det, indent=2)
    print(f"Saved detailed metrics (mean ± std) to: {detailed_json}")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE: ARCHITECTURE COMPARISON (BERT FEATURES)")
    print("=" * 60)
    print(f"{'Model':<10} | {'Accuracy':<18} | {'Precision':<18} | {'Recall':<18} | {'F1-Score':<18}")
    print("-" * 60)
    for m in models_part_a:
        mean = detailed_report["part_a"][m]["mean"]
        std = detailed_report["part_a"][m]["std"]
        print(f"{m.upper():<10} | {mean['accuracy']*100:.2f} ± {std['accuracy']*100:.2f}%     | {mean['precision']*100:.2f} ± {std['precision']*100:.2f}%     | {mean['recall']*100:.2f} ± {std['recall']*100:.2f}%     | {mean['f1']*100:.2f} ± {std['f1']*100:.2f}%")

    print("\n" + "=" * 60)
    print("SUMMARY TABLE: FEATURE ABLATION (GRAPHSAGE)")
    print("=" * 60)
    print(f"{'Feature':<10} | {'Accuracy':<18} | {'Precision':<18} | {'Recall':<18} | {'F1-Score':<18}")
    print("-" * 60)
    for f in features_part_b:
        mean = detailed_report["part_b"][f]["mean"]
        std = detailed_report["part_b"][f]["std"]
        print(f"{f.upper():<10} | {mean['accuracy']*100:.2f} ± {std['accuracy']*100:.2f}%     | {mean['precision']*100:.2f} ± {std['precision']*100:.2f}%     | {mean['recall']*100:.2f} ± {std['recall']*100:.2f}%     | {mean['f1']*100:.2f} ± {std['f1']*100:.2f}%")
    print("=" * 60)

    wb_proj = wandb_project or os.environ.get("WANDB_PROJECT")
    if wb_proj and _HAS_WANDB:
        key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
        wandb.init(project=wb_proj, name=f"part2_gnn_{'ta' if ta_mode else 'benchmark'}", reinit=True)
        for m, vals in results_template["part_a"].items():
            for metric_k, metric_v in vals.items():
                wandb.log({f"arch_{m}/{metric_k}": metric_v})
        for f, vals in results_template["part_b"].items():
            for metric_k, metric_v in vals.items():
                wandb.log({f"feature_{f}/{metric_k}": metric_v})
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ta", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(
        ta_mode=args.ta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        wandb_project=args.wandb_project,
    )
