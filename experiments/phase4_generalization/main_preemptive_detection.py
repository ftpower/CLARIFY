"""P0 Part C: Preemptive Detection — FactCheckmate-style pre-decode hallucination screening.

Trains a lightweight MLP on last-input-token hidden states (L16) to predict whether
the model will answer correctly, BEFORE any generation happens. This is Stage 1 of
a two-stage detection framework: fast pre-screening → selective deep verification.

Compares preemptive AUROC with post-hoc D2 JS AUROC and evaluates joint performance.

Usage:
    python main_preemptive_detection.py --n_samples 500 --device cuda
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "phase2_entropy"))
from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt

sys.path.insert(0, str(Path(__file__).parent))
from phase4_utils.generalization_features import (
    compute_d2_js_topk,
    compute_d2_js_score,
    select_top_js_pairs,
)
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states


def extract_preemptive_features(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    preemptive_layer: int = 16,
    js_early: int = 17,
    js_late: int = 26,
) -> dict:
    """Extract preemptive hidden states (no decoding) and post-hoc JS features.

    For each HellaSwag sample:
      1. Forward pass on the full input prefix (question + 4 options)
      2. Extract last input token hidden state at preemptive_layer
      3. Also compute post-hoc D2 JS for comparison

    Returns:
        labels: [N] int
        p_correct: [N] float
        preemptive_states: [N, d_model] — last input token hidden state at target layer
        js_scores: [N] — D2 JS divergence between js_early and js_late
        max_p: [N] — max_p at final layer
        choice_probs: [N, n_layers, 4]
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]

    N = len(samples)
    labels = np.zeros(N, dtype=np.int32)
    p_correct_arr = np.zeros(N, dtype=np.float32)

    preemptive_states = np.zeros((N, d_model), dtype=np.float32)
    all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)

    for idx, sample in enumerate(tqdm(samples, desc="Extracting preemptive features")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        sub_states = extract_all_sub_layer_states(model, prompt)

        # Preemptive: last input token hidden state at target layer
        preemptive_states[idx, :] = (
            sub_states["hidden"][preemptive_layer][0, :].detach().cpu().to(torch.float32)
        )

        # Post-hoc: per-layer 4-choice softmax for JS
        for li in range(n_layers):
            h = sub_states["hidden"][li].to(W_U.device)
            logits_L = h @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)

            probs_L = torch.softmax(logits_L.float(), dim=-1)
            all_maxp[idx, li] = probs_L.max().item()

            choice_logits = logits_L[0, letter_tok_ids]
            choice_probs_L = torch.softmax(choice_logits.float(), dim=-1)
            all_choice_probs[idx, li, :] = choice_probs_L.detach().cpu().to(torch.float32)

        # Correctness from final logits
        logits_last = sub_states["logits"]
        choice_logits_final = logits_last[letter_tok_ids]
        probs_final = torch.softmax(choice_logits_final.float(), dim=-1)
        pred_idx = probs_final.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        labels[idx] = int(is_correct)
        p_correct_arr[idx] = probs_final[letters.index(correct_letter)].item()

    # D2 JS scores at best pair
    js_scores = compute_d2_js_score(all_choice_probs, js_early, js_late)

    return {
        "labels": labels,
        "p_correct": p_correct_arr,
        "preemptive_states": preemptive_states,
        "js_scores": js_scores,
        "max_p": all_maxp,
        "choice_probs": all_choice_probs,
    }


class PreemptiveMLP(torch.nn.Module):
    """Lightweight MLP for preemptive hallucination detection.

    Architecture: d_model → 128 → 64 → 1 (sigmoid)
    Trained with binary cross-entropy, no GPU requirement after feature extraction.
    """

    def __init__(self, d_model: int, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_preemptive_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train PreemptiveMLP with stratified K-fold CV.

    Returns:
        dict with "cv_scores", "cv_probs" (out-of-fold predictions), "models"
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    d_model = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs = np.zeros(len(y), dtype=np.float32)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y)
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Convert to tensors
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)

        model = PreemptiveMLP(d_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Class balance weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

        model.train()
        for _epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_train_t, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        all_probs[val_idx] = val_probs
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            auc = float("nan")
        cv_scores.append(auc)
        print(f"  Fold {fold + 1}: AUROC = {auc:.4f}")

    return {
        "cv_scores": cv_scores,
        "cv_mean_auroc": float(np.nanmean(cv_scores)),
        "cv_std_auroc": float(np.nanstd(cv_scores)),
        "cv_probs": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="P0 Part C: Preemptive Detection"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preemptive_layer", type=int, default=16)
    parser.add_argument("--js_early", type=int, default=17)
    parser.add_argument("--js_late", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "preemptive_features.npz"

    if args.skip_extract and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data = {
            "labels": cached["labels"],
            "p_correct": cached["p_correct"],
            "preemptive_states": cached["preemptive_states"],
            "js_scores": cached["js_scores"],
            "max_p": cached["max_p"],
        }
    else:
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(
                f" {letter}", add_special_tokens=False
            )
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
        print(f"Letter token IDs: {letter_ids}")

        print(f"\nLoading HellaSwag ({args.n_samples} samples)...")
        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        data = extract_preemptive_features(
            model,
            samples,
            letter_ids,
            preemptive_layer=args.preemptive_layer,
            js_early=args.js_early,
            js_late=args.js_late,
        )

        np.savez_compressed(
            cache_path,
            labels=data["labels"],
            p_correct=data["p_correct"],
            preemptive_states=data["preemptive_states"],
            js_scores=data["js_scores"],
            max_p=data["max_p"],
        )
        print(f"Cached features to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Knowledge filter ──
    labels = data["labels"]
    p_correct = data["p_correct"]
    preemptive_states = data["preemptive_states"]
    js_scores = data["js_scores"]
    max_p = data["max_p"]

    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_preemptive = preemptive_states[filt_mask]
    filt_js = js_scores[filt_mask]
    filt_maxp = max_p[filt_mask]
    n_filt = filt_mask.sum()

    print(f"\nFull: {labels.sum()}/{len(labels)} = {labels.mean():.4f}")
    print(f"Filtered (P>0.3): {n_filt} samples, acc={filt_labels.mean():.4f}")

    # ── Post-hoc baselines ──
    print(f"\n{'=' * 60}")
    print("Post-Hoc Detection Baselines (for reference)")
    print(f"{'=' * 60}")

    # Best max_p layer
    n_layers = max_p.shape[1]
    best_mp_auroc = 0.0
    best_mp_layer = -1
    for li in range(n_layers):
        try:
            auc = roc_auc_score(filt_labels, filt_maxp[:, li])
        except ValueError:
            auc = 0.5
        if auc > best_mp_auroc:
            best_mp_auroc = auc
            best_mp_layer = li
    print(f"Post-hoc max_p (L{best_mp_layer}):  AUROC = {best_mp_auroc:.4f}")

    try:
        js_auroc = roc_auc_score(filt_labels, filt_js)
    except ValueError:
        js_auroc = 0.5
    print(f"Post-hoc D2 JS (L{args.js_early} vs L{args.js_late}): AUROC = {js_auroc:.4f}")

    # Joint post-hoc LR
    X_post = np.stack([filt_maxp[:, best_mp_layer], filt_js], axis=1)
    try:
        lr_post = LogisticRegression(max_iter=1000)
        joint_post_auroc = cross_val_score(
            lr_post, X_post, filt_labels, cv=5, scoring="roc_auc"
        ).mean()
        print(f"Post-hoc Joint (max_p + JS):        AUROC = {joint_post_auroc:.4f}")
    except Exception as e:
        joint_post_auroc = float("nan")
        print(f"Post-hoc Joint LR failed: {e}")

    # ── Preemptive MLP ──
    print(f"\n{'=' * 60}")
    print(f"Preemptive MLP (L{args.preemptive_layer}, d_model→128→64→1)")
    print(f"{'=' * 60}")

    mlp_result = train_preemptive_mlp_cv(
        filt_preemptive,
        filt_labels,
        n_splits=5,
        epochs=args.epochs,
        seed=args.seed,
    )

    preemptive_auroc = mlp_result["cv_mean_auroc"]
    print(f"\nPreemptive MLP (5-fold CV):  AUROC = {preemptive_auroc:.4f} "
          f"± {mlp_result['cv_std_auroc']:.4f}")

    # ── Joint: preemptive + post-hoc ──
    print(f"\n{'=' * 60}")
    print("Joint: Preemptive + Post-hoc")
    print(f"{'=' * 60}")

    X_joint = np.stack(
        [mlp_result["cv_probs"], filt_maxp[:, best_mp_layer], filt_js],
        axis=1,
    )
    try:
        lr_joint = LogisticRegression(max_iter=1000)
        joint_all_auroc = cross_val_score(
            lr_joint, X_joint, filt_labels, cv=5, scoring="roc_auc"
        ).mean()
        print(f"Joint (preemptive + max_p + JS): AUROC = {joint_all_auroc:.4f}")
    except Exception as e:
        joint_all_auroc = float("nan")
        print(f"Joint LR failed: {e}")

    # ── Efficiency analysis ──
    print(f"\n{'=' * 60}")
    print("Efficiency Analysis (Stage 1 → Stage 2 cascade)")
    print(f"{'=' * 60}")

    preemptive_probs = mlp_result["cv_probs"]

    for tau_low in [0.2, 0.3, 0.4]:
        skip_mask = preemptive_probs < tau_low
        skip_rate = skip_mask.mean()
        # Among skipped samples, how many were actually correct?
        if skip_mask.sum() > 0:
            false_skip_rate = 1.0 - filt_labels[skip_mask].mean()
        else:
            false_skip_rate = 0.0

        # Recall: what fraction of hallucinations (label=0) are caught?
        halluc_mask = filt_labels == 0
        if halluc_mask.sum() > 0:
            recall = 1.0 - skip_mask[halluc_mask].mean()
        else:
            recall = float("nan")

        print(
            f"  τ_low={tau_low:.1f}: skip={skip_rate:.1%}, "
            f"false_skip={false_skip_rate:.1%}, "
            f"recall@τ={recall:.3f}"
        )

    # ── Save results ──
    output = {
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "preemptive_layer": args.preemptive_layer,
            "js_early": args.js_early,
            "js_late": args.js_late,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "full_set": {
            "n_samples": len(labels),
            "accuracy": float(labels.mean()),
        },
        "filtered_set": {
            "n_samples": int(n_filt),
            "accuracy": float(filt_labels.mean()),
        },
        "post_hoc": {
            "max_p_auroc": float(best_mp_auroc),
            "max_p_layer": best_mp_layer,
            "js_auroc": float(js_auroc),
            "joint_post_hoc_auroc": float(joint_post_auroc)
            if not np.isnan(joint_post_auroc)
            else None,
        },
        "preemptive": {
            "auroc_cv_mean": float(preemptive_auroc),
            "auroc_cv_std": float(mlp_result["cv_std_auroc"]),
            "cv_scores": [float(s) for s in mlp_result["cv_scores"]],
        },
        "joint_all_auroc": float(joint_all_auroc)
        if not np.isnan(joint_all_auroc)
        else None,
    }

    out_path = output_dir / "preemptive_detection_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Preemptive (Stage 1):        AUROC = {preemptive_auroc:.4f}")
    print(f"Post-hoc (Stage 2):          AUROC = {joint_post_auroc:.4f}")
    print(f"Joint (Stage 1 + Stage 2):   AUROC = {joint_all_auroc:.4f}")
    gain = (
        float(joint_all_auroc) - float(joint_post_auroc)
        if not np.isnan(joint_all_auroc) and not np.isnan(joint_post_auroc)
        else float("nan")
    )
    print(f"Δ (joint − post-hoc only):   {gain:+.4f}")
    if not np.isnan(gain) and gain < 0.01:
        print("WARNING: Preemptive adds negligible value over post-hoc alone.")
        print("Consider: Stage 1 AUROC < 0.70 → abandon two-stage framework.")


if __name__ == "__main__":
    main()
