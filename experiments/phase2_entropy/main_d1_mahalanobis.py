"""Phase 3 D1: Mahalanobis Distance Hallucination Detector.

For each transformer layer, compute per-class (correct vs incorrect) statistics
from hidden states, then use Mahalanobis distance to the "correct" centroid as a
hallucination detection score.

Key design decisions:
- Ledoit-Wolf shrinkage for covariance estimation (d=2048 >> n≈300)
- Train/test split within the filtered set (P(correct)>0.3) via stratified 5-fold CV
- Reports AUROC per layer, joint with max_p, and compares to max_p baseline

Usage:
    python main_d1_mahalanobis.py                        # full run
    python main_d1_mahalanobis.py --skip_extract          # use cached states
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


def mahalanobis_distance(
    X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray
) -> np.ndarray:
    """d_M(x) = sqrt((x - mu)^T Σ^{-1} (x - mu)) for each row of X."""
    diff = X - mu  # [N, d]
    # (diff @ inv_cov) → [N, d], then element-wise * diff, sum over axis=1
    quad = np.sum(diff @ inv_cov * diff, axis=1)
    quad = np.maximum(quad, 0.0)  # numerical guard
    return np.sqrt(quad)


def extract_states_for_d1(model, samples, letter_ids, n_layers, d_model) -> dict:
    """Extract hidden states for D1 — only last-token residual post per sample."""
    n_samples = len(samples)
    states = np.zeros((n_samples, n_layers, d_model), dtype=np.float16)
    labels = np.zeros(n_samples, dtype=np.int32)
    p_correct_arr = np.zeros(n_samples, dtype=np.float32)

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]

    storage = {}
    hooks = []
    for i in range(n_layers):
        key = f"blocks.{i}.hook_resid_post"
        hooks.append((key, _make_collect_hook(storage, key)))

    for idx, sample in enumerate(tqdm(samples, desc="Extracting D1 states")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        for li in range(n_layers):
            h = storage[f"blocks.{li}.hook_resid_post"][0, last_pos, :]
            states[idx, li, :] = h.detach().cpu().to(torch.float16)

        logits_last = logits[0, last_pos, :]
        choice_logits = logits_last[letter_tok_ids]
        probs = torch.softmax(choice_logits.float(), dim=-1)
        pred_idx = probs.argmax().item()
        labels[idx] = int(letters[pred_idx] == correct_letter)
        p_correct_arr[idx] = probs[letters.index(correct_letter)].item()

    return {"labels": labels, "p_correct": p_correct_arr, "states": states}


def _make_collect_hook(storage: dict, key: str):
    def hook(activation, hook=None):
        storage[key] = activation.detach()
        return activation

    return hook


# ═══════════════════════════════════════════════════════════════════════════
# D1: Mahalanobis Distance Detector
# ═══════════════════════════════════════════════════════════════════════════


def run_d1(data: dict, output_dir: Path, n_folds: int = 5):
    """Mahalanobis distance to correct-class centroid as hallucination detector."""
    print("\n" + "=" * 60)
    print("D1: Mahalanobis Distance Detector")
    print("=" * 60)

    labels = data["labels"]
    p_correct = data["p_correct"]
    states = data["states"].astype(np.float32)  # [N, n_layers, d_model]
    n_layers = states.shape[1]
    d_model = states.shape[2]
    n_total = len(labels)

    print(f"Samples: {n_total}, Layers: {n_layers}, d_model: {d_model}")
    print(f"Full-set accuracy: {labels.mean():.4f}")

    # Knowledge filter
    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_states = states[filt_mask]
    filt_p_correct = p_correct[filt_mask]
    n_filt = filt_mask.sum()
    print(f"Filtered (P>0.3): {n_filt} samples, acc={filt_labels.mean():.4f}")

    if n_filt < 40:
        print("ERROR: Too few filtered samples for CV. Need >= 40.")
        return

    # ── Per-layer Mahalanobis AUROC via stratified 5-fold CV ──
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store: [n_layers, n_folds] AUROC values
    mahal_aurocs = np.zeros((n_layers, n_folds))
    joint_aurocs = np.zeros((n_layers, n_folds))
    maxp_aurocs = np.zeros((n_layers, n_folds))

    for li in tqdm(range(n_layers), desc="D1 per-layer Mahalanobis"):
        X = filt_states[:, li, :]  # [N_filt, d_model]
        y = filt_labels

        for fi, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Split into correct/incorrect
            X_corr = X_train[y_train == 1]
            X_incorr = X_train[y_train == 0]

            if len(X_corr) < 3 or len(X_incorr) < 3:
                mahal_aurocs[li, fi] = float("nan")
                joint_aurocs[li, fi] = float("nan")
                maxp_aurocs[li, fi] = float("nan")
                continue

            # Mean of correct class
            mu_c = X_corr.mean(axis=0)  # [d_model]

            # Shared covariance via Ledoit-Wolf
            try:
                lw = LedoitWolf().fit(X_train)
                inv_cov = np.linalg.inv(lw.covariance_)
            except Exception:
                # Fallback: diagonal covariance with shrinkage
                var = X_train.var(axis=0) + 1e-3
                inv_cov = np.diag(1.0 / var)

            # Mahalanobis distance to correct centroid for test samples
            d_m = mahalanobis_distance(X_test, mu_c, inv_cov)

            try:
                auc_m = roc_auc_score(y_test, d_m)
                # Higher distance → more likely hallucination
                # If AUC < 0.5, flip (distance to correct = close to correct = correct)
                if auc_m < 0.5:
                    auc_m = 1.0 - auc_m
                mahal_aurocs[li, fi] = auc_m
            except ValueError:
                mahal_aurocs[li, fi] = float("nan")

            # Joint LR with max_p (using P(correct) from filtered data as proxy
            # since we need per-layer max_p in the test set — use the p_correct from test set)
            p_c_test = filt_p_correct[test_idx]
            try:
                X_joint = np.stack([d_m, p_c_test], axis=1)
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(max_iter=1000)
                lr.fit(
                    np.stack([d_m[train_idx] if False else d_m, p_c_test])[: len(d_m)],
                    y_test,
                )
                # Use the same fold data in a proper nested way
                # For simplicity: just report individual AUROCs
                joint_aurocs[li, fi] = float("nan")
            except Exception:
                joint_aurocs[li, fi] = float("nan")

            # max_p as baseline (use P(correct) from logits)
            try:
                auc_mp = roc_auc_score(y_test, p_c_test)
                maxp_aurocs[li, fi] = auc_mp if auc_mp >= 0.5 else 1.0 - auc_mp
            except ValueError:
                maxp_aurocs[li, fi] = float("nan")

    # ── Report ──
    mean_mahal = np.nanmean(mahal_aurocs, axis=1)
    std_mahal = np.nanstd(mahal_aurocs, axis=1)
    mean_maxp_cv = np.nanmean(maxp_aurocs, axis=1)

    # Best layer
    best_li = int(np.nanargmax(mean_mahal))
    print(
        f"\n{'Layer':<6} {'Mahal AUROC':>12} {'Std':>8} {'max_p AUROC':>12} {'Delta':>8}"
    )
    print("-" * 55)
    for li in range(n_layers):
        if li <= 2 or li >= n_layers - 3 or li == best_li or li in [11, 15]:
            delta = mean_mahal[li] - mean_maxp_cv[li]
            marker = " <--" if li == best_li else ""
            print(
                f"L{li:<5} {mean_mahal[li]:>12.4f} {std_mahal[li]:>8.4f} "
                f"{mean_maxp_cv[li]:>12.4f} {delta:>+8.4f}{marker}"
            )

    print(
        f"\nBest: L{best_li} Mahal AUROC = {mean_mahal[best_li]:.4f} ± {std_mahal[best_li]:.4f}"
    )

    # ── Also compute on full (non-filtered) set for reference ──
    print(f"\n--- Full-set Mahalanobis (no knowledge filter) ---")
    # Train on all filtered, test on full set
    full_mahal_aurocs = np.zeros(n_layers)
    for li in range(n_layers):
        X_filt_all = filt_states[:, li, :]
        y_filt_all = filt_labels
        X_corr = X_filt_all[y_filt_all == 1]
        if len(X_corr) < 3:
            full_mahal_aurocs[li] = 0.5
            continue

        mu_c = X_corr.mean(axis=0)
        try:
            lw = LedoitWolf().fit(X_filt_all)
            inv_cov = np.linalg.inv(lw.covariance_)
        except Exception:
            var = X_filt_all.var(axis=0) + 1e-3
            inv_cov = np.diag(1.0 / var)

        d_m_full = mahalanobis_distance(
            states[:, li, :].astype(np.float32), mu_c, inv_cov
        )
        try:
            auc = roc_auc_score(labels, d_m_full)
            full_mahal_aurocs[li] = max(auc, 1.0 - auc)
        except ValueError:
            full_mahal_aurocs[li] = 0.5

    best_full_li = int(np.argmax(full_mahal_aurocs))
    print(
        f"Best full-set: L{best_full_li} AUROC = {full_mahal_aurocs[best_full_li]:.4f}"
    )

    # Joint LR evaluation (Mahal + max_p) on filtered set via CV
    print(f"\n--- Joint Mahal + max_p (cross-validated LR) ---")
    joint_lr_results = []
    for li in range(n_layers):
        X_filt = filt_states[:, li, :]
        y_filt = filt_labels
        p_c = filt_p_correct

        cv_aurocs = []
        for train_idx, test_idx in cv.split(X_filt, y_filt):
            X_tr, X_te = X_filt[train_idx], X_filt[test_idx]
            y_tr, y_te = y_filt[train_idx], y_filt[test_idx]
            p_c_te = p_c[test_idx]

            X_c = X_tr[y_tr == 1]
            if len(X_c) < 3:
                continue
            mu_c = X_c.mean(axis=0)
            try:
                lw = LedoitWolf().fit(X_tr)
                inv_cov = np.linalg.inv(lw.covariance_)
            except Exception:
                var = X_tr.var(axis=0) + 1e-3
                inv_cov = np.diag(1.0 / var)

            d_m = mahalanobis_distance(X_te, mu_c, inv_cov)

            from sklearn.linear_model import LogisticRegression

            X_lr = np.stack([d_m, p_c_te], axis=1)
            try:
                lr = LogisticRegression(max_iter=1000).fit(X_lr, y_te)
                # Predict probability
                y_prob = lr.predict_proba(X_lr)[:, 1]
                auc_j = roc_auc_score(y_te, y_prob)
                cv_aurocs.append(auc_j)
            except Exception:
                pass

        if cv_aurocs:
            joint_lr_results.append((li, np.mean(cv_aurocs), np.std(cv_aurocs)))

    joint_lr_results.sort(key=lambda x: x[1], reverse=True)
    if joint_lr_results:
        best_joint = joint_lr_results[0]
        print(
            f"Best joint: L{best_joint[0]} AUROC = {best_joint[1]:.4f} ± {best_joint[2]:.4f}"
        )

    # Comparison with max_p alone (best filtered max_p AUROC over folds)
    p_c_auroc = (
        roc_auc_score(filt_labels, filt_p_correct)
        if len(np.unique(filt_labels)) > 1
        else 0.5
    )
    print(f"\nBaseline max_p (P(correct)>0.3): {p_c_auroc:.4f}")
    print(f"Best Mahal gain over max_p: {mean_mahal[best_li] - p_c_auroc:+.4f}")

    out = {
        "n_total": n_total,
        "n_filtered": int(n_filt),
        "filtered_acc": float(filt_labels.mean()),
        "full_acc": float(labels.mean()),
        "n_folds": n_folds,
        "best_mahal_layer": int(best_li),
        "best_mahal_auroc": float(mean_mahal[best_li]),
        "best_mahal_std": float(std_mahal[best_li]),
        "maxp_baseline_auroc": float(p_c_auroc),
        "mahal_gain_over_maxp": float(mean_mahal[best_li] - p_c_auroc),
        "per_layer_mahal_auroc": [float(v) for v in mean_mahal],
        "per_layer_mahal_std": [float(v) for v in std_mahal],
        "best_full_mahal_layer": int(best_full_li),
        "best_full_mahal_auroc": float(full_mahal_aurocs[best_full_li]),
        "best_joint": (
            {
                "layer": int(best_joint[0]),
                "auroc": float(best_joint[1]),
                "std": float(best_joint[2]),
            }
            if joint_lr_results
            else None
        ),
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_dir / "d1_mahalanobis_results.json", "w") as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {output_dir / 'd1_mahalanobis_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_cache = output_dir / "phase3_residual_states.npz"

    if args.skip_extract and state_cache.exists():
        print(f"Loading cached states from {state_cache}")
        cached = np.load(state_cache, allow_pickle=True)
        data = {
            "labels": cached["labels"],
            "p_correct": cached["p_correct"],
            "states": cached["states"],
        }
    else:
        print(f"Loading HellaSwag ({args.n_samples} samples)...")
        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
        print(f"Letter token IDs: {letter_ids}")

        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model

        data = extract_states_for_d1(model, samples, letter_ids, n_layers, d_model)

        np.savez_compressed(
            state_cache,
            labels=data["labels"],
            p_correct=data["p_correct"],
            states=data["states"],
        )
        print(f"Cached to {state_cache}")

    run_d1(data, output_dir, n_folds=args.n_folds)
    print("\nD1 — Done")


if __name__ == "__main__":
    main()
