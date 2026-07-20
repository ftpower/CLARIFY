"""Phase 4.1 (Plan 2): Multi-Signal Orthogonality Verification.

Core scientific question: Are EigenScore, D2 JS, and HaloScope ζ measuring
orthogonal aspects of hallucination risk, or are they redundant?

Three signals, three variance sources:
  - EigenScore: sampling variance (K-shot semantic consistency)
  - D2 JS:      layer-wise variance (inter-layer prediction disagreement)
  - HaloScope ζ: population variance (sample outlier in dominant directions)

Hypothesis: pairwise Pearson r < 0.3 for all three pairs.

Analysis pipeline:
  Step 1: Compute all features on HellaSwag n=500 (knowledge-filtered)
  Step 2: Pairwise Pearson + Spearman correlation matrix
  Step 3: Conditional AUROC — Δ = AUROC(X | known Y) − AUROC(Y alone)
  Step 4: Joint LR with all features vs max_p baseline
  Step 5: Ablation — drop each feature one at a time, measure AUROC drop

Reuses feature functions from phase4_generalization/phase4_utils/.

Usage:
    python main_orthogonality_analysis.py --n_samples 500 --device cuda
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Path setup ──
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PHASE2 = _SCRIPT_DIR.parent / "phase2_entropy"
_PHASE4_UTILS = _SCRIPT_DIR.parent / "phase4_generalization"

sys.path.insert(0, str(_PHASE2))
sys.path.insert(0, str(_PHASE4_UTILS))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt
from phase4_utils.generalization_features import (
    compute_d2_js_topk,
    select_top_js_pairs,
    compute_haloscope_zeta_batch,
    compute_attn_ffn_ratio,
)
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction (focused on the 3 target signals + max_p baseline)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_three_signals(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    eig_layer: int = 17,
    halo_layer: int = 17,
    ratio_layer: int = 17,
) -> dict:
    """Extract EigenScore, D2 JS, HaloScope ζ, Attn/FFN ratio, and max_p.

    Note: EigenScore requires K temperature-sampled forward passes per sample,
    which is expensive. This function extracts all OTHER features and leaves
    EigenScore to be filled in separately (or skipped).

    Returns:
        labels: [N]
        p_correct: [N]
        features: dict[str, np.ndarray] — one entry per feature
        choice_probs: [N, n_layers, 4] — for D2 JS computation
        hidden_at_halo: list[np.ndarray] — for HaloScope computation
        attn_at_ratio: list[np.ndarray]
        ffn_at_ratio: list[np.ndarray]
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]

    N = len(samples)
    labels = np.zeros(N, dtype=np.int32)
    p_correct_arr = np.zeros(N, dtype=np.float32)

    all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)
    all_entropy_arr = np.zeros((N, n_layers), dtype=np.float32)

    hidden_at_halo = []
    attn_at_ratio = []
    ffn_at_ratio = []

    for idx, sample in enumerate(tqdm(samples, desc="Extracting 3-signal features")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        sub_states = extract_all_sub_layer_states(model, prompt)

        # Per-layer logit lens
        for li in range(n_layers):
            h = sub_states["hidden"][li].to(W_U.device)
            logits_L = h @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)

            probs_L = torch.softmax(logits_L.float(), dim=-1)
            all_maxp[idx, li] = probs_L.max().item()

            log_probs_L = torch.log_softmax(logits_L.float(), dim=-1)
            all_entropy_arr[idx, li] = float(
                -(probs_L * log_probs_L).sum(dim=-1).item()
            )

            choice_logits = logits_L[0, letter_tok_ids]
            all_choice_probs[idx, li, :] = torch.softmax(
                choice_logits.float(), dim=-1
            ).detach().cpu().to(torch.float32)

        # Sub-layer states at target layers
        hidden_at_halo.append(sub_states["hidden"][halo_layer][0, :].cpu())
        attn_at_ratio.append(sub_states["attn"][ratio_layer][0, :].cpu())
        ffn_at_ratio.append(sub_states["ffn"][ratio_layer][0, :].cpu())

        # Correctness
        logits_last = sub_states["logits"]
        choice_logits_final = logits_last[letter_tok_ids]
        probs_final = torch.softmax(choice_logits_final.float(), dim=-1)
        pred_idx = probs_final.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        labels[idx] = int(is_correct)
        p_correct_arr[idx] = probs_final[letters.index(correct_letter)].item()

    # ── Build feature arrays ──
    features = {}

    # max_p at best layer
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
    features["max_p"] = all_maxp[:, best_mp_layer]
    features["entropy"] = all_entropy_arr[:, best_mp_layer]

    # D2 JS
    js_result = compute_d2_js_topk(all_choice_probs, top_k=5, exclude_layer0=True)
    js_selector = select_top_js_pairs(js_result["all_pair_js"], labels, top_k=5)
    features["d2_js_top5"] = js_selector["js_scores"]
    features["d2_js_best_pair"] = js_selector["best_pair"]

    # HaloScope ζ
    hidden_matrix = torch.stack(hidden_at_halo).numpy()
    features["haloscope_zeta"] = compute_haloscope_zeta_batch(hidden_matrix, k=5)

    # Attn/FFN ratio
    ratios = np.array(
        [
            compute_attn_ffn_ratio(a, f)
            for a, f in zip(attn_at_ratio, ffn_at_ratio)
        ],
        dtype=np.float32,
    )
    features["attn_ffn_ratio"] = ratios

    return {
        "labels": labels,
        "p_correct": p_correct_arr,
        "features": features,
        "choice_probs": all_choice_probs,
        "best_mp_layer": best_mp_layer,
        "best_mp_auroc": best_mp_auroc,
        "d2_best_pair": js_selector["best_pair"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Orthogonality analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_orthogonality(
    feature_dict: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict:
    """Compute pairwise correlations and conditional AUROC increments.

    Args:
        feature_dict: {name: [N] array} for each feature.
        labels: [N] binary array.

    Returns:
        dict with correlation_matrix, conditional_auroc, joint_auroc, ablation.
    """
    feature_names = list(feature_dict.keys())
    # Only keep 1D array features (skip metadata like d2_js_best_pair tuple)
    # Also skip features that are all NaN
    feature_names = [n for n in feature_names
                     if isinstance(feature_dict[n], np.ndarray)
                     and feature_dict[n].ndim == 1
                     and not np.all(np.isnan(feature_dict[n]))]
    n_features = len(feature_names)

    # Build clean feature matrix (drop NaN rows)
    X = np.stack([feature_dict[n] for n in feature_names], axis=1)  # [N, F]
    valid = ~np.isnan(X).any(axis=1)
    X_clean = X[valid]
    y_clean = labels[valid]
    n_dropped = (~valid).sum()
    print(f"  Valid samples: {valid.sum()}/{len(labels)} (dropped {n_dropped} NaN)")

    # ── Pairwise correlations ──
    pearson_matrix = np.zeros((n_features, n_features))
    spearman_matrix = np.zeros((n_features, n_features))
    pearson_pvals = np.zeros((n_features, n_features))
    spearman_pvals = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                pearson_matrix[i, j] = 1.0
                spearman_matrix[i, j] = 1.0
                pearson_pvals[i, j] = 0.0
                spearman_pvals[i, j] = 0.0
            else:
                xi, xj = X_clean[:, i], X_clean[:, j]
                r_p, p_p = pearsonr(xi, xj)
                r_s, p_s = spearmanr(xi, xj)
                pearson_matrix[i, j] = r_p
                spearman_matrix[i, j] = r_s
                pearson_pvals[i, j] = p_p
                spearman_pvals[i, j] = p_s

    # ── Individual AUROCs ──
    individual_auroc = {}
    for i, name in enumerate(feature_names):
        try:
            auc = roc_auc_score(y_clean, X_clean[:, i])
        except ValueError:
            auc = float("nan")
        individual_auroc[name] = float(auc) if not np.isnan(auc) else None

    # ── Conditional AUROC increments ──
    conditional = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    for i, name_i in enumerate(feature_names):
        for j, name_j in enumerate(feature_names):
            if i >= j:
                continue
            # AUROC(name_i | known name_j) = joint AUROC - AUROC(name_j alone)
            X_pair = X_scaled[:, [i, j]]
            try:
                lr = LogisticRegression(max_iter=2000)
                joint_ij = cross_val_score(
                    lr, X_pair, y_clean, cv=5, scoring="roc_auc"
                ).mean()
            except Exception:
                joint_ij = float("nan")

            auc_i = individual_auroc.get(name_i, 0) or 0
            auc_j = individual_auroc.get(name_j, 0) or 0

            delta_i_given_j = joint_ij - auc_j
            delta_j_given_i = joint_ij - auc_i

            conditional[f"{name_i}|{name_j}"] = {
                "joint": float(joint_ij),
                "auroc_i": float(auc_i),
                "auroc_j": float(auc_j),
                "delta_i_given_j": float(delta_i_given_j),
                "delta_j_given_i": float(delta_j_given_i),
            }

    # ── Joint LR with all features ──
    try:
        lr_all = LogisticRegression(max_iter=2000)
        joint_all_auroc = cross_val_score(
            lr_all, X_scaled, y_clean, cv=5, scoring="roc_auc"
        ).mean()
    except Exception:
        joint_all_auroc = float("nan")

    # ── Ablation: drop each feature one at a time ──
    ablation = {}
    for i, name in enumerate(feature_names):
        # Exclude column i
        cols = [c for c in range(n_features) if c != i]
        X_ablate = X_scaled[:, cols]
        try:
            lr = LogisticRegression(max_iter=2000)
            auc_without = cross_val_score(
                lr, X_ablate, y_clean, cv=5, scoring="roc_auc"
            ).mean()
        except Exception:
            auc_without = float("nan")
        ablation[name] = {
            "auroc_without": float(auc_without),
            "delta": float(joint_all_auroc - auc_without),
        }

    return {
        "n_valid": int(valid.sum()),
        "n_dropped": int(n_dropped),
        "feature_names": feature_names,
        "pearson_matrix": {
            feature_names[i]: {
                feature_names[j]: float(pearson_matrix[i, j])
                for j in range(n_features)
            }
            for i in range(n_features)
        },
        "spearman_matrix": {
            feature_names[i]: {
                feature_names[j]: float(spearman_matrix[i, j])
                for j in range(n_features)
            }
            for i in range(n_features)
        },
        "individual_auroc": individual_auroc,
        "conditional_auroc": conditional,
        "joint_all_auroc": float(joint_all_auroc),
        "ablation": ablation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_orthogonality_report(results: dict):
    """Pretty-print the orthogonality analysis results."""

    feature_names = results["feature_names"]
    n = len(feature_names)

    # Correlation matrix
    print(f"\n{'=' * 60}")
    print("Pairwise Pearson Correlation Matrix")
    print(f"{'=' * 60}")
    header = f"{'':<16}" + "".join(f"{name:<14}" for name in feature_names)
    print(header)
    print("-" * len(header))
    for i, name_i in enumerate(feature_names):
        row = f"{name_i:<16}"
        for j in range(n):
            r = results["pearson_matrix"][name_i][feature_names[j]]
            symbol = "█" if abs(r) > 0.5 else ("▆" if abs(r) > 0.3 else "·")
            row += f"{r:+.4f} {symbol}  "
        print(row)

    # Orthogonality check
    print(f"\n--- Orthogonality Check ---")
    target_pairs = [
        ("d2_js_top5", "haloscope_zeta"),
        ("d2_js_top5", "attn_ffn_ratio"),
        ("haloscope_zeta", "attn_ffn_ratio"),
    ]
    all_orthogonal = True
    for n1, n2 in target_pairs:
        if n1 in feature_names and n2 in feature_names:
            r = results["pearson_matrix"][n1][n2]
            status = "✅ ORTHOGONAL" if abs(r) < 0.3 else "⚠ CORRELATED"
            if abs(r) >= 0.3:
                all_orthogonal = False
            print(f"  r({n1}, {n2}) = {r:+.4f}  {status}")

    if all_orthogonal:
        print("\n✅ All three signals are mutually orthogonal (r < 0.3).")
        print("   This supports Claim 1 — they capture independent aspects.")
    else:
        print("\n⚠ Some signals are correlated (|r| ≥ 0.3).")
        print("   Consider dropping the redundant signal(s).")

    # Conditional AUROC
    print(f"\n{'=' * 60}")
    print("Conditional AUROC Increments (Δ = joint − single)")
    print(f"{'=' * 60}")
    for key, info in results["conditional_auroc"].items():
        delta_i = info["delta_i_given_j"]
        delta_j = info["delta_j_given_i"]
        sig_i = "✅ +signal" if delta_i > 0.02 else "· negligible"
        sig_j = "✅ +signal" if delta_j > 0.02 else "· negligible"
        print(
            f"  {key}: joint={info['joint']:.4f}, "
            f"Δ_i|j={delta_i:+.4f} {sig_i}, "
            f"Δ_j|i={delta_j:+.4f} {sig_j}"
        )

    # Joint
    print(f"\n{'=' * 60}")
    print("Joint Detection Performance")
    print(f"{'=' * 60}")
    best_single = max(
        (v for v in results["individual_auroc"].values() if v is not None), default=0
    )
    gain = results["joint_all_auroc"] - best_single
    print(f"  Best single AUROC:    {best_single:.4f}")
    print(f"  Joint all-feature:    {results['joint_all_auroc']:.4f}")
    print(f"  Δ over best single:   {gain:+.4f}")

    # Ablation
    print(f"\n--- Ablation: Feature Importance (drop-one) ---")
    print(f"  {'Feature':<20} {'AUROC w/o':>10} {'Δ (drop cost)':>14}")
    print(f"  {'-' * 44}")
    for name, info in sorted(
        results["ablation"].items(), key=lambda x: x[1]["delta"], reverse=True
    ):
        print(f"  {name:<20} {info['auroc_without']:>10.4f} {info['delta']:>+14.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.1: Multi-Signal Orthogonality Verification"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_eigenscore", action="store_true",
                        help="Skip EigenScore (K=10 temp. sampling, ~30min)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "orthogonality_features.npz"

    if args.skip_extract and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data = {
            "labels": cached["labels"],
            "p_correct": cached["p_correct"],
            "features": cached["features"].item(),
            "choice_probs": cached["choice_probs"],
        }
    else:
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

        print(f"Loading HellaSwag (n={args.n_samples})...")
        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        print(f"\n{'=' * 60}")
        print("Extracting 3-signal features")
        print(f"{'=' * 60}")
        data = extract_three_signals(model, samples, letter_ids)

        # EigenScore (optional, expensive)
        if not args.skip_eigenscore:
            print(f"\n{'=' * 60}")
            print("Computing EigenScore (K=10 temperature sampling)")
            print(f"{'=' * 60}")
            from phase4_utils.generalization_features import compute_eigenscore

            es_scores = np.zeros(len(samples), dtype=np.float32)
            for idx in tqdm(range(len(samples)), desc="EigenScore"):
                prompt = format_prompt(
                    samples[idx]["question"],
                    samples[idx]["context"],
                    dataset="hellaswag",
                )
                try:
                    es_scores[idx] = compute_eigenscore(
                        model, prompt, layer_idx=17, K=10, temperature=0.5
                    )
                except Exception as e:
                    print(f"  WARNING: EigenScore failed for sample {idx}: {e}")
                    es_scores[idx] = np.nan
            data["features"]["eigenscore"] = es_scores
        else:
            print("\nSkipping EigenScore (--skip_eigenscore)")
            data["features"]["eigenscore"] = np.full(
                len(data["labels"]), np.nan, dtype=np.float32
            )

        # Save cache
        np.savez_compressed(
            cache_path,
            labels=data["labels"],
            p_correct=data["p_correct"],
            features=data["features"],
            choice_probs=data["choice_probs"],
        )
        print(f"Cached to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Knowledge filter ──
    labels = data["labels"]
    p_correct = data["p_correct"]
    features = data["features"]

    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    n_filt = filt_mask.sum()

    print(f"\nKnowledge filter: {n_filt}/{len(labels)} samples, "
          f"acc={filt_labels.mean():.4f}")

    # Filter features (skip non-array entries like d2_js_best_pair tuple)
    filt_features = {}
    for name, arr in features.items():
        if isinstance(arr, np.ndarray):
            filt_features[name] = arr[filt_mask]
        else:
            filt_features[name] = arr  # keep as-is (e.g., d2_js_best_pair tuple)

    # ── Orthogonality analysis ──
    print(f"\n{'=' * 60}")
    print("Orthogonality Analysis")
    print(f"{'=' * 60}")

    results = analyze_orthogonality(filt_features, filt_labels)
    print_orthogonality_report(results)

    # ── Save ──
    output = {
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "n_filtered": int(n_filt),
            "filtered_acc": float(filt_labels.mean()),
            "seed": args.seed,
        },
        "orthogonality": results,
        "interpretation": {
            "orthogonal": all(
                abs(results["pearson_matrix"][n1][n2]) < 0.3
                for n1, n2 in [
                    ("d2_js_top5", "haloscope_zeta"),
                    ("d2_js_top5", "attn_ffn_ratio"),
                    ("haloscope_zeta", "attn_ffn_ratio"),
                ]
                if n1 in results["feature_names"] and n2 in results["feature_names"]
            ),
            "claim1_supported": (
                results["joint_all_auroc"]
                > max(
                    v for v in results["individual_auroc"].values() if v is not None
                )
                + 0.02
            ),
        },
    }

    out_path = output_dir / "orthogonality_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
