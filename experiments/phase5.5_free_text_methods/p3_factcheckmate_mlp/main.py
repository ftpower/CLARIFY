"""P3: FactCheckmate MLP — Pre-decode hallucination prediction on TriviaQA.

Trains a lightweight MLP on the LAST INPUT TOKEN's hidden state (before any
generation) to predict whether the model will hallucinate on the answer.

MLP architecture: d_model → 128 → 64 → 1 (sigmoid)
Only needs a single forward pass per sample (no generation).

FactCheckmate (NeurIPS 2025): 70-77% detection accuracy across 8 LM families.

Usage:
    python main.py --n_samples 200
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import load_model_and_data, format_prompt


def extract_input_hidden_states(model, samples, device, layer_idx, n_layers):
    """Extract hidden state at last INPUT token for specified layer.

    Only does a single forward pass per sample — no generation.
    """
    hidden_states = []
    labels = []

    for sample in tqdm(samples, desc="Hidden states"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        storage = {}

        def _hook(name):
            def hook(act, hook):
                storage[name] = act[:, -1, :].detach()

            return hook

        fwd_hooks = [
            (f"blocks.{layer_idx}.hook_resid_post", _hook("h")),
        ]

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        hidden_states.append(storage["h"].squeeze(0).cpu().numpy())
        labels.append(sample["is_correct"])

    return np.array(hidden_states, dtype=np.float64), np.array(labels, dtype=np.int32)


def train_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dims: list[int] = [128, 64],
    n_epochs: int = 50,
    lr: float = 1e-3,
    n_folds: int = 5,
    device: str = "cuda",
) -> dict:
    """Train MLP with stratified k-fold CV, reporting AUROC."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(device)
        X_test = torch.tensor(X_scaled[test_idx], dtype=torch.float32).to(device)
        y_test = y[test_idx]

        # Build MLP
        d_in = X.shape[1]
        layers = []
        prev_dim = d_in
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        mlp = nn.Sequential(*layers).to(device)

        opt = torch.optim.Adam(mlp.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        # Train
        mlp.train()
        for epoch in range(n_epochs):
            opt.zero_grad()
            pred = mlp(X_train).squeeze(-1)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt.step()

        # Evaluate
        mlp.eval()
        with torch.no_grad():
            y_pred = mlp(X_test).squeeze(-1).cpu().numpy()
        try:
            auc = roc_auc_score(1 - y_test, y_pred)
            fold_aurocs.append(float(auc))
        except ValueError:
            fold_aurocs.append(0.5)

        print(f"    Fold {fold_idx + 1}: AUROC = {fold_aurocs[-1]:.4f}")

    return {
        "mean": float(np.mean(fold_aurocs)),
        "std": float(np.std(fold_aurocs)),
        "per_fold": fold_aurocs,
    }


def main(n_samples=200, device="cuda", model_id="Qwen/Qwen3-1.7B",
         output_dir="outputs", seed=42):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, samples, _, _ = load_model_and_data(
        n_samples=n_samples, seed=seed, device=device, model_id=model_id
    )
    n_layers = model.cfg.n_layers

    # ── Load labels from Phase 5.1 ─────────────────────────────────────
    phase5_json = (_sys_parent / "phase5_cross_task" / "outputs" / "triviaqa_features.json")
    with open(phase5_json) as f:
        p5_labels = [s["is_correct"] for s in json.load(f)["per_sample"]]
    # Attach labels to samples for extract function
    for i, s in enumerate(samples):
        s["is_correct"] = p5_labels[i] if i < len(p5_labels) else False

    # ── Scan all layers ────────────────────────────────────────────────
    print(f"\nScanning {n_layers} layers for best MLP detection layer...")
    all_layer_results = {}
    best_layer, best_auroc = -1, 0.5

    for li in range(n_layers):
        print(f"\n  Layer {li}/{n_layers - 1}:")
        X, y = extract_input_hidden_states(
            model, samples, device, li, n_layers
        )
        y_binary = (y == 1).astype(np.int32)

        result = train_mlp_cv(X, y_binary, device=device)
        all_layer_results[f"L{li}"] = result
        print(f"    CV AUROC: {result['mean']:.4f} ± {result['std']:.4f}")

        if result["mean"] > best_auroc:
            best_auroc = result["mean"]
            best_layer = li

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Best: L{best_layer} CV AUROC = {best_auroc:.4f}")
    print(f"Num samples: {n_samples}")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples, "model_id": model_id,
            "hidden_dims": [128, 64], "seed": seed,
        },
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "per_layer": all_layer_results,
    }

    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / 'results.json'}")
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P3: FactCheckmate MLP on TriviaQA")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device, model_id=args.model,
         output_dir=args.output_dir, seed=args.seed)
