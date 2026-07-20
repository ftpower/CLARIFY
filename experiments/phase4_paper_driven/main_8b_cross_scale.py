"""Phase 4.4: 8B Cross-Scale Alignment — the 8B receiver side.

Loads Qwen3-8B, extracts the same 3 features (max_p, d2_js_top5, haloscope_zeta),
applies the frozen 1.7B-trained scaler+LR for zero-shot AUROC, trains an 8B-native
LR for comparison, and computes PCA bases at corresponding layers for subspace
comparison.

Usage (on AutoDL RTX 5090):
    python main_8b_cross_scale.py --n_samples 500 --device cuda
    python main_8b_cross_scale.py --n_samples 200 --skip_extract  # use cache
"""

import argparse
import gc
import json
import os
import pickle
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase4_generalization"))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states
from phase4_utils.generalization_features import (
    compute_d2_js_topk, select_top_js_pairs,
    compute_haloscope_zeta_batch,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Proportional layer mapping: 1.7B L → 8B L'
# 1.7B has 28 layers (L0-L27), 8B has 36 layers (L0-L35)
# Map: L_8B ≈ round(L_1.7B * 35/27)
LAYER_MAP_1_7B_TO_8B = {
    11: 14,   # round(11 * 35/27) = 14
    15: 19,   # round(15 * 35/27) = 19
    17: 22,   # round(17 * 35/27) = 22
    26: 33,   # round(26 * 35/27) = 33
}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction (8B)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_8b_features(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    key_layers_1_7b: list[int],
    key_layers_8b: list[int],
    k_pca: int = 64,
) -> dict:
    """Extract features on 8B: max_p, D2 JS, HaloScope, and per-layer hidden states.

    Returns same structure as 1.7B's extract_and_compute_bases() for cross-scale
    comparison.
    """
    n_layers = model.cfg.n_layers       # 36 for Qwen3-8B
    d_model = model.cfg.d_model          # 4096 for Qwen3-8B
    W_U = model.unembed.W_U             # [4096, 151936]
    b_U = model.unembed.b_U

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]

    N = len(samples)
    labels = np.zeros(N, dtype=np.int32)
    p_correct_arr = np.zeros(N, dtype=np.float32)

    all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)

    # Hidden state accumulators for PCA
    hidden_accum = {L: [] for L in key_layers_8b}
    hidden_for_halo = []  # at 8B's L22 (≈1.7B L17, haloscope reference layer)

    for idx, sample in enumerate(tqdm(samples, desc="Extracting 8B features")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        sub = extract_all_sub_layer_states(model, prompt)

        # Per-layer logit lens for all 36 layers
        for li in range(n_layers):
            h = sub["hidden"][li].to(W_U.device)
            logits_L = h @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)
            all_maxp[idx, li] = torch.softmax(
                logits_L.float(), dim=-1
            ).max().item()

            choice_logits = logits_L[0, letter_tok_ids]
            all_choice_probs[idx, li, :] = torch.softmax(
                choice_logits.float(), dim=-1
            ).detach().cpu().to(torch.float32)

        # Accumulate hidden states for PCA at key layers
        for L in key_layers_8b:
            hidden_accum[L].append(sub["hidden"][L][0, :].cpu().numpy())

        # HaloScope reference: 1.7B used L17 → 8B uses L22
        halo_layer = LAYER_MAP_1_7B_TO_8B[17]
        hidden_for_halo.append(sub["hidden"][halo_layer][0, :].cpu())

        # Correctness
        logits_last = sub["logits"]
        cf = logits_last[letter_tok_ids]
        pf = torch.softmax(cf.float(), dim=-1)
        pred_idx = pf.argmax().item()
        labels[idx] = int(letters[pred_idx] == correct_letter)
        p_correct_arr[idx] = pf[letters.index(correct_letter)].item()

    # ── PCA bases per key 8B layer ──
    pca_bases = {}
    for L in key_layers_8b:
        X = np.stack(hidden_accum[L], axis=0).astype(np.float64)  # [N, 4096]
        k_actual = min(k_pca, X.shape[0], X.shape[1])
        pca = PCA(n_components=k_actual)
        pca.fit(X)
        pca_bases[str(L)] = {
            "components": pca.components_,  # [k, 4096]
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
            "mean": pca.mean_.tolist(),
            "k_actual": k_actual,
        }
        print(f"  8B L{L}: PCA(k={k_actual}), cumulative variance = "
              f"{pca.explained_variance_ratio_.sum():.3f}")

    # ── Features (same 3 as 1.7B) ──
    # D2 JS: scan all 36*35/2 layer pairs
    js_result = compute_d2_js_topk(all_choice_probs, top_k=5, exclude_layer0=True)
    js_selector = select_top_js_pairs(js_result["all_pair_js"], labels, top_k=5)

    # HaloScope at 8B's reference layer
    hidden_matrix = torch.stack(hidden_for_halo).numpy()
    halo_zeta = compute_haloscope_zeta_batch(hidden_matrix, k=5)

    # Best max_p layer — use last layer L35 (consistent with 1.7B L27 fix)
    last_layer = n_layers - 1  # L35
    best_mp_auroc = roc_auc_score(labels, all_maxp[:, last_layer])
    print(f"  max_p at L{last_layer} (full AUROC={best_mp_auroc:.4f})")

    features = {
        "labels": labels,
        "p_correct": p_correct_arr,
        "max_p": all_maxp[:, last_layer],
        "d2_js_top5": js_selector["js_scores"],
        "haloscope_zeta": halo_zeta,
        "best_mp_layer": last_layer,
        "best_mp_auroc": best_mp_auroc,
        "d2_best_pair": js_selector["best_pair"],
    }

    return {
        "pca_bases": pca_bases,
        "features": features,
        "n_layers_8b": n_layers,
        "d_model_8b": d_model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-shot transfer: apply frozen 1.7B detector to 8B features
# ═══════════════════════════════════════════════════════════════════════════════


def apply_frozen_detector(
    features: dict,
    transfer_path: str,
) -> dict:
    """Apply 1.7B-trained scaler + LR to 8B features (zero-shot transfer)."""
    with open(transfer_path, "rb") as f:
        transfer = pickle.load(f)

    scaler: StandardScaler = transfer["scaler"]
    lr: LogisticRegression = transfer["lr"]
    feature_names: list[str] = transfer["feature_names"]

    labels = features["labels"]
    p_correct = features["p_correct"]

    # Knowledge filter (same threshold as 1.7B)
    filt_mask = p_correct > 0.3
    y = labels[filt_mask]
    n_filt = int(filt_mask.sum())
    print(f"  Knowledge-filtered: {n_filt}/{len(labels)}, acc={y.mean():.4f}")

    X_raw = np.stack([
        features[name][filt_mask]
        for name in feature_names
    ], axis=1)

    valid = ~np.isnan(X_raw).any(axis=1)
    X = X_raw[valid]
    y = y[valid]
    n_valid = int(valid.sum())
    print(f"  Valid: {n_valid}/{n_filt} (dropped {n_filt - n_valid} NaN)")

    # ── A: Apply frozen 1.7B scaler + LR (zero-shot) ──
    X_scaled_1_7b = scaler.transform(X)
    y_pred_proba = lr.predict_proba(X_scaled_1_7b)[:, 1]
    zero_shot_auroc = float(roc_auc_score(y, y_pred_proba))
    print(f"  Zero-shot AUROC (1.7B scaler+LR on 8B): {zero_shot_auroc:.4f}")

    # ── B: 8B-native scaler + LR (5-fold CV) ──
    scaler_8b = StandardScaler()
    X_scaled_8b = scaler_8b.fit_transform(X)
    lr_8b = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv_scores = cross_val_score(
        lr_8b, X_scaled_8b, y, cv=min(5, n_valid // 10),
        scoring="roc_auc",
    )
    native_auroc = float(cv_scores.mean())
    native_std = float(cv_scores.std())
    print(f"  Native 8B AUROC ({min(5, n_valid // 10)}-fold CV): "
          f"{native_auroc:.4f} ± {native_std:.4f}")

    # ── C: Per-feature AUROC comparison ──
    per_feature = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        try:
            auc = float(roc_auc_score(y, col))
        except ValueError:
            auc = 0.5
        per_feature[name] = auc

    # ── D: Re-train 8B LR on all data (for saving) ──
    lr_8b.fit(X_scaled_8b, y)
    train_acc = lr_8b.score(X_scaled_8b, y)
    print(f"  8B LR training accuracy: {train_acc:.4f}")
    print(f"  8B LR coefficients: {lr_8b.coef_[0].tolist()}")

    return {
        "zero_shot_auroc": zero_shot_auroc,
        "native_auroc_cv": native_auroc,
        "native_auroc_std": native_std,
        "degradation": zero_shot_auroc - native_auroc,
        "per_feature_auroc": per_feature,
        "feature_names": feature_names,
        "lr_8b_coef": lr_8b.coef_[0].tolist(),
        "lr_8b_intercept": lr_8b.intercept_.tolist(),
        "lr_1_7b_coef": lr.coef_[0].tolist(),
        "lr_1_7b_intercept": lr.intercept_.tolist(),
        "n_filtered": n_filt,
        "n_valid": n_valid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-scale comparison report
# ═══════════════════════════════════════════════════════════════════════════════


def print_cross_scale_report(results: dict):
    """Print a formatted cross-scale comparison report."""
    print(f"\n{'=' * 60}")
    print("Cross-Scale Transfer Report")
    print(f"{'=' * 60}")

    # 1. Zero-shot vs native
    zs = results["transfer"]["zero_shot_auroc"]
    nat = results["transfer"]["native_auroc_cv"]
    deg = results["transfer"]["degradation"]
    status = "✅" if deg > -0.10 else ("⚠️" if deg > -0.20 else "🔴")
    print(f"\n  Zero-shot AUROC (1.7B→8B):  {zs:.4f}")
    print(f"  Native 8B AUROC (CV):        {nat:.4f}")
    print(f"  Cross-scale degradation:     {deg:+.4f} {status}")

    # 2. Per-feature comparison
    print(f"\n  Per-Feature AUROC Comparison:")
    print(f"  {'Feature':<20s} {'1.7B':>8s} {'8B':>8s} {'Δ':>8s}")
    print(f"  {'-'*44}")
    pf = results["transfer"].get("per_feature_auroc", {})
    # 1.7B per-feature AUROCs (from the training run)
    pf_1_7b = {
        "max_p": 0.8492,
        "d2_js_top5": 0.7046,
        "haloscope_zeta": 0.5071,
    }
    for name in pf:
        a_1b = pf_1_7b.get(name, float("nan"))
        a_8b = pf[name]
        d = a_8b - a_1b
        s = "✅" if d > -0.10 else ("⚠️" if d > -0.20 else "🔴")
        print(f"  {name:<20s} {a_1b:8.4f} {a_8b:8.4f} {d:+8.4f} {s}")

    # 3. LR coefficient comparison
    print(f"\n  LR Coefficient Comparison:")
    print(f"  {'Feature':<20s} {'1.7B':>8s} {'8B':>8s}")
    print(f"  {'-'*36}")
    for i, name in enumerate(results["transfer"]["feature_names"]):
        c_1b = results["transfer"]["lr_1_7b_coef"][i]
        c_8b = results["transfer"]["lr_8b_coef"][i]
        print(f"  {name:<20s} {c_1b:8.4f} {c_8b:8.4f}")

    # 4. PCA variance comparison
    print(f"\n  PCA Cumulative Variance (k=64):")
    print(f"  {'1.7B Layer':>10s} {'8B Layer':>10s} "
          f"{'1.7B Var':>10s} {'8B Var':>10s}")
    print(f"  {'-'*42}")
    pca_1_7b = results["pca_1_7b"]
    pca_8b = results["pca_8b"]
    for L_1b in results["key_layers_1_7b"]:
        L_8b = LAYER_MAP_1_7B_TO_8B.get(L_1b, L_1b)
        var_1b = pca_1_7b.get(str(L_1b), {}).get("cumulative_variance", float("nan"))
        var_8b = pca_8b.get(str(L_8b), {}).get("cumulative_variance", float("nan"))
        print(f"  {L_1b:>10} {L_8b:>10} {var_1b:>10.3f} {var_8b:>10.3f}")

    # 5. Summary verdict
    print(f"\n{'=' * 60}")
    print("Verdict")
    print(f"{'=' * 60}")
    if deg > -0.05:
        print("✅ Cross-scale transfer SUCCESSFUL — 1.7B detector directly usable on 8B.")
    elif deg > -0.15:
        print("⚠️  Cross-scale transfer MARGINAL — 8B-native LR preferred but 1.7B LR useful as baseline.")
    else:
        print("🔴 Cross-scale transfer FAILED — features depend on model scale.")
        print("   Recommendation: Train detector natively on 8B for best results.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.4: 8B Cross-Scale Alignment (8B receiver side)"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--bases_1_7b", type=str,
                        default="outputs/cross_scale_1_7b_bases.npz",
                        help="Path to 1.7B PCA bases .npz")
    parser.add_argument("--detector_1_7b", type=str,
                        default="outputs/detector_1_7b_for_8b_transfer.pkl",
                        help="Path to 1.7B-trained scaler+LR pickle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_pca", type=int, default=64)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "cross_scale_8b_features.npz"

    # ── Load 1.7B artifacts ──
    bases_1_7b_path = Path(args.bases_1_7b)
    detector_1_7b_path = Path(args.detector_1_7b)

    if not bases_1_7b_path.exists():
        print(f"ERROR: 1.7B bases not found at {bases_1_7b_path}")
        print("Run main_cross_scale_alignment.py on 1.7B first.")
        sys.exit(1)
    if not detector_1_7b_path.exists():
        print(f"ERROR: 1.7B detector not found at {detector_1_7b_path}")
        print("Run main_cross_scale_alignment.py on 1.7B first.")
        sys.exit(1)

    bases_1_7b_data = np.load(bases_1_7b_path, allow_pickle=True)
    key_layers_1_7b = bases_1_7b_data["key_layers"].tolist()
    pca_1_7b = bases_1_7b_data["pca_bases"].item()
    print(f"Loaded 1.7B bases for layers {key_layers_1_7b}")
    print(f"d_model_1_7b = {bases_1_7b_data['d_model_1_7b'].item()}")

    # 8B layer mapping
    key_layers_8b = [LAYER_MAP_1_7B_TO_8B[L] for L in key_layers_1_7b]
    print(f"Mapped to 8B layers: {key_layers_8b}")

    # ── Extract (or load) 8B features ──
    if args.skip_extract and cache_path.exists():
        print(f"\nLoading cached 8B features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        features_8b = cached["features"].item()
        pca_8b = cached["pca_bases"].item()
        d_model_8b = int(cached["d_model_8b"])
        n_layers_8b = int(cached["n_layers_8b"])
    else:
        print(f"\nLoading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(
                f" {letter}", add_special_tokens=False
            )
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

        print(f"\nLoading HellaSwag (n={args.n_samples})...")
        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        print(f"\n{'=' * 60}")
        print("Extracting 8B features + PCA bases")
        print(f"{'=' * 60}")

        data = extract_8b_features(
            model, samples, letter_ids, key_layers_1_7b, key_layers_8b,
            args.k_pca,
        )

        features_8b = data["features"]
        pca_8b = data["pca_bases"]
        d_model_8b = data["d_model_8b"]
        n_layers_8b = data["n_layers_8b"]

        # Save cache
        np.savez_compressed(
            cache_path,
            features=features_8b,
            pca_bases=pca_8b,
            d_model_8b=d_model_8b,
            n_layers_8b=n_layers_8b,
            allow_pickle=True,
        )
        print(f"Cached 8B features to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Zero-shot transfer ──
    print(f"\n{'=' * 60}")
    print("Zero-Shot Transfer: 1.7B Detector → 8B Features")
    print(f"{'=' * 60}")

    transfer_results = apply_frozen_detector(
        features_8b,
        str(detector_1_7b_path),
    )

    # ── Compile full results ──
    results = {
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "seed": args.seed,
            "key_layers_1_7b": key_layers_1_7b,
            "key_layers_8b": key_layers_8b,
            "k_pca": args.k_pca,
            "d_model_1_7b": int(bases_1_7b_data["d_model_1_7b"]),
            "d_model_8b": d_model_8b,
        },
        "transfer": transfer_results,
        "key_layers_1_7b": key_layers_1_7b,
        "pca_1_7b": {k: {"cumulative_variance": pca_1_7b[k]["cumulative_variance"]}
                      for k in pca_1_7b},
        "pca_8b": {k: {"cumulative_variance": pca_8b[k]["cumulative_variance"]}
                   for k in pca_8b},
    }

    # ── Report ──
    print_cross_scale_report(results)

    # ── Save 8B artifacts for future use ──
    # Save 8B detector (native, for downstream experiments)
    detector_8b_path = output_dir / "detector_8b_native.pkl"
    with open(detector_8b_path, "wb") as f:
        # Re-create the scaler+LR from the transfer results
        labels_8b = features_8b["labels"]
        p_correct_8b = features_8b["p_correct"]
        filt_mask = p_correct_8b > 0.3
        y = labels_8b[filt_mask]
        X_raw = np.stack([
            features_8b[name][filt_mask]
            for name in transfer_results["feature_names"]
        ], axis=1)
        valid = ~np.isnan(X_raw).any(axis=1)
        X, y = X_raw[valid], y[valid]
        scaler_8b = StandardScaler()
        X_scaled = scaler_8b.fit_transform(X)
        lr_8b = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr_8b.fit(X_scaled, y)
        pickle.dump({
            "scaler": scaler_8b,
            "lr": lr_8b,
            "feature_names": transfer_results["feature_names"],
        }, f)
    print(f"\nSaved 8B native detector to {detector_8b_path}")

    # Save JSON results
    results_path = output_dir / "cross_scale_8b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")

    print("\n8B cross-scale alignment complete. ✅")


if __name__ == "__main__":
    main()
