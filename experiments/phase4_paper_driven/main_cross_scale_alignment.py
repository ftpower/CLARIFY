"""Phase 4.4 Prep (Plan 2): Cross-Model Subspace Alignment Analysis (1.7B side).

Prepares for 8B cross-scale validation by computing PCA bases on 1.7B and saving
the aligned feature representations. When 8B results become available (from
AutoDL), the cross-scale analysis can be completed by loading this file.

Analysis dimensions:
  1. Per-layer PCA bases (k=64) for 1.7B across key layers
  2. Feature standardization parameters (saved for 8B normalization)
  3. LR weights trained on 1.7B (saved for 8B zero-shot)
  4. Expected cross-scale degradation estimates

This script runs ONLY on 1.7B (local RTX 5060). The 8B counterpart runs on AutoDL.

Usage:
    python main_cross_scale_alignment.py --n_samples 500 --device cuda
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase4_generalization"))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states
from phase4_utils.generalization_features import (
    compute_d2_js_topk, select_top_js_pairs, compute_d2_js_score,
    compute_haloscope_zeta_batch,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.7B feature extraction + PCA basis computation
# ═══════════════════════════════════════════════════════════════════════════════


def extract_and_compute_bases(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    key_layers: list[int],
    k_pca: int = 64,
) -> dict:
    """Extract features and compute per-layer PCA bases on 1.7B.

    Returns everything needed for cross-scale alignment:
      - Per-layer PCA bases (for subspace angle computation with 8B)
      - Feature values (for training 1.7B LR)
      - Standardization parameters (for 8B normalization)
      - LR weights (for 8B zero-shot evaluation)
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

    all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)

    # Per-layer hidden state accumulators for PCA
    hidden_accum = {L: [] for L in key_layers}
    hidden_at_halo = []

    for idx, sample in enumerate(tqdm(samples, desc="Extracting 1.7B features")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        sub = extract_all_sub_layer_states(model, prompt)

        for li in range(n_layers):
            h = sub["hidden"][li].to(W_U.device)
            logits_L = h @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)
            all_maxp[idx, li] = torch.softmax(logits_L.float(), dim=-1).max().item()
            choice_logits = logits_L[0, letter_tok_ids]
            all_choice_probs[idx, li, :] = torch.softmax(
                choice_logits.float(), dim=-1
            ).detach().cpu().to(torch.float32)

        # Accumulate for PCA
        for L in key_layers:
            hidden_accum[L].append(sub["hidden"][L][0, :].cpu().numpy())

        hidden_at_halo.append(sub["hidden"][17][0, :].cpu())

        # Correctness
        logits_last = sub["logits"]
        cf = logits_last[letter_tok_ids]
        pf = torch.softmax(cf.float(), dim=-1)
        pred_idx = pf.argmax().item()
        labels[idx] = int(letters[pred_idx] == correct_letter)
        p_correct_arr[idx] = pf[letters.index(correct_letter)].item()

    # ── PCA bases per key layer ──
    pca_bases = {}
    for L in key_layers:
        X = np.stack(hidden_accum[L], axis=0).astype(np.float64)  # [N, d]
        k_actual = min(k_pca, X.shape[0], X.shape[1])
        pca = PCA(n_components=k_actual)
        pca.fit(X)
        pca_bases[str(L)] = {
            "components": pca.components_,  # [k, d]
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
            "mean": pca.mean_.tolist(),
            "k_actual": k_actual,
        }
        print(f"  L{L}: PCA(k={k_actual}), cumulative variance = "
              f"{pca.explained_variance_ratio_.sum():.3f}")

    # ── Features ──
    # D2 JS
    js_result = compute_d2_js_topk(all_choice_probs, top_k=5, exclude_layer0=True)
    js_selector = select_top_js_pairs(js_result["all_pair_js"], labels, top_k=5)

    # HaloScope
    hidden_matrix = torch.stack(hidden_at_halo).numpy()
    halo_zeta = compute_haloscope_zeta_batch(hidden_matrix, k=5)

    # Best max_p
    best_mp_auroc = 0.0
    best_mp_layer = 0
    for li in range(n_layers):
        try:
            auc = roc_auc_score(labels, all_maxp[:, li])
        except ValueError:
            auc = 0.5
        if auc > best_mp_auroc:
            best_mp_auroc = auc
            best_mp_layer = li

    features = {
        "labels": labels,
        "p_correct": p_correct_arr,
        "max_p": all_maxp[:, best_mp_layer],
        "d2_js_top5": js_selector["js_scores"],
        "haloscope_zeta": halo_zeta,
        "best_mp_layer": best_mp_layer,
        "best_mp_auroc": best_mp_auroc,
        "d2_best_pair": js_selector["best_pair"],
    }

    return {
        "pca_bases": pca_bases,
        "features": features,
        "n_layers_1_7b": n_layers,
        "d_model_1_7b": d_model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Train 1.7B detector (to be transferred to 8B)
# ═══════════════════════════════════════════════════════════════════════════════


def train_and_save_detector(features: dict, output_dir: Path) -> dict:
    """Train LR detector on knowledge-filtered 1.7B features, save for 8B transfer."""

    labels = features["labels"]
    p_correct = features["p_correct"]

    filt_mask = p_correct > 0.3
    y = labels[filt_mask]
    n_filt = filt_mask.sum()
    print(f"  Knowledge-filtered: {n_filt}/{len(labels)}, acc={y.mean():.4f}")

    X_raw = np.stack([
        features["max_p"][filt_mask],
        features["d2_js_top5"][filt_mask],
        features["haloscope_zeta"][filt_mask],
    ], axis=1)

    # Handle NaN
    valid = ~np.isnan(X_raw).any(axis=1)
    X = X_raw[valid]
    y = y[valid]
    print(f"  Valid: {valid.sum()}/{n_filt} (dropped {(~valid).sum()} NaN)")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train LR
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_scaled, y)

    train_acc = lr.score(X_scaled, y)
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Coefficients: {lr.coef_[0].tolist()}")

    # Save for 8B transfer
    transfer_artifacts = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lr_coef": lr.coef_.tolist(),
        "lr_intercept": lr.intercept_.tolist(),
        "lr_classes": lr.classes_.tolist(),
        "feature_names": ["max_p", "d2_js_top5", "haloscope_zeta"],
        "d_model_1_7b": features.get("d_model_1_7b", 2048),
    }

    transfer_path = output_dir / "detector_1_7b_for_8b_transfer.json"
    with open(transfer_path, "w") as f:
        json.dump(transfer_artifacts, f, indent=2)
    print(f"  Saved 1.7B→8B transfer artifacts to {transfer_path}")

    # Also save as pickle for easy Python loading
    pkl_path = output_dir / "detector_1_7b_for_8b_transfer.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {"scaler": scaler, "lr": lr, "feature_names": transfer_artifacts["feature_names"]},
            f,
        )
    print(f"  Saved pickle to {pkl_path}")

    return transfer_artifacts


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.4 Prep: Cross-Scale Alignment (1.7B side)"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_pca", type=int, default=64)
    parser.add_argument("--key_layers", type=int, nargs="+",
                        default=[11, 15, 17, 26])
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "cross_scale_1_7b_bases.npz"
    transfer_path = output_dir / "detector_1_7b_for_8b_transfer.json"

    if args.skip_extract and cache_path.exists() and transfer_path.exists():
        print(f"Loading cached bases from {cache_path}")
        print(f"Transfer artifacts already at {transfer_path}")
        print("1.7B side complete. Ready for 8B AutoDL run.")
        return

    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

    print(f"\nLoading HellaSwag (n={args.n_samples})...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    print(f"\n{'=' * 60}")
    print("Extracting 1.7B features + PCA bases")
    print(f"{'=' * 60}")

    data = extract_and_compute_bases(
        model, samples, letter_ids, args.key_layers, args.k_pca,
    )

    # ── Save PCA bases ──
    np.savez_compressed(
        cache_path,
        pca_bases=data["pca_bases"],
        key_layers=args.key_layers,
        n_layers_1_7b=data["n_layers_1_7b"],
        d_model_1_7b=data["d_model_1_7b"],
        allow_pickle=True,
    )
    print(f"\nSaved PCA bases to {cache_path}")

    # ── Train and save detector ──
    print(f"\n{'=' * 60}")
    print("Training 1.7B Detector for 8B Transfer")
    print(f"{'=' * 60}")

    features = data["features"]
    features["d_model_1_7b"] = data["d_model_1_7b"]
    transfer = train_and_save_detector(features, output_dir)

    # ── Instructions for 8B run ──
    print(f"\n{'=' * 60}")
    print("8B AutoDL Instructions")
    print(f"{'=' * 60}")
    print(f"""
    1. Copy these files to AutoDL:
       - {cache_path}
       - {output_dir / 'detector_1_7b_for_8b_transfer.pkl'}

    2. On AutoDL, run 8B validation with cross-scale alignment:
       python main_8b_cross_scale.py \\
         --model Qwen/Qwen3-8B \\
         --bases_1_7b {cache_path} \\
         --detector_1_7b {output_dir / 'detector_1_7b_for_8b_transfer.pkl'}

    3. Expected outputs:
       - Principal angles between 1.7B and 8B subspaces
       - Zero-shot AUROC of 1.7B-trained LR on 8B features
       - Cross-scale degradation matrix (per-feature)
    """)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("1.7B side complete. ✅")


if __name__ == "__main__":
    main()
