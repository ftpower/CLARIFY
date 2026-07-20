"""P0 Part A: Generalization Feature Set — cross-model cross-dataset evaluation.

Computes 7 detection features on HellaSwag (knowledge-filtered), trains a logistic
regression detector, and evaluates zero-shot on TriviaQA and SQuAD.

Features:
  1. EigenScore (K=10 temperature-sampled forward passes) at L17
  2. HaloScope ζ (cross-sample SVD projection) at L17
  3. Attn/FFN L2 norm ratio at L17
  4. D2 JS top-5 mean (layer-pair consistency score)
  5. max_p at best layer (baseline)
  6. Entropy at best layer
  7. Top-5 mass at best layer

Usage:
    python main_generalization_features.py --n_samples 500 --device cuda
    python main_generalization_features.py --n_samples 200 --skip_extract  # use cache
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import from phase2_entropy
sys.path.insert(0, str(Path(__file__).parent.parent / "phase2_entropy"))
from src.model_loader import load_model
from src.data_loader import (
    load_hellaswag,
    load_triviaqa,
    load_squad,
    format_prompt,
)

# Import from local phase4_utils
sys.path.insert(0, str(Path(__file__).parent))
from phase4_utils.generalization_features import (
    compute_d2_js_topk,
    compute_d2_js_score,
    select_top_js_pairs,
    compute_haloscope_zeta_batch,
    compute_attn_ffn_ratio,
    compute_max_prob_per_layer,
    compute_entropy_per_layer,
    compute_top5_mass_per_layer,
)
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract_features(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    eig_layer: int = 17,
    halo_layer: int = 17,
    ratio_layer: int = 17,
    js_early: int = 17,
    js_late: int = 26,
    output_dir: Path | None = None,
) -> dict:
    """Extract all 7 features for a list of samples.

    Returns dict with:
        labels: [N] int — 1=correct, 0=incorrect
        p_correct: [N] float
        features: [N, 7] float32 — feature matrix
        feature_names: list[str] — column names
        choice_probs: [N, n_layers, 4] — for D2 JS
        hidden_at_eig: list of [d_model] — for HaloScope
        attn_at_ratio: list of [d_model] — for ratio feature
        ffn_at_ratio: list of [d_model] — for ratio feature
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

    # Per-sample accumulators
    all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)
    all_entropy_arr = np.zeros((N, n_layers), dtype=np.float32)
    all_top5 = np.zeros((N, n_layers), dtype=np.float32)

    hidden_at_eig = []
    attn_at_ratio = []
    ffn_at_ratio = []

    for idx, sample in enumerate(tqdm(samples, desc="Extracting features")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        # Extract all sub-layer states in one forward pass
        sub_states = extract_all_sub_layer_states(model, prompt)

        # ── Logit lens metrics (all layers) ──
        for li in range(n_layers):
            h = sub_states["hidden"][li].to(W_U.device)  # [1, d_model]
            logits_L = h @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)

            probs_L = torch.softmax(logits_L.float(), dim=-1)
            all_maxp[idx, li] = probs_L.max().item()

            log_probs_L = torch.log_softmax(logits_L.float(), dim=-1)
            all_entropy_arr[idx, li] = float(
                -(probs_L * log_probs_L).sum(dim=-1).item()
            )

            top5_vals, _ = torch.topk(probs_L, k=5, dim=-1)
            all_top5[idx, li] = top5_vals.sum().item()

            # 4-choice softmax
            choice_logits = logits_L[0, letter_tok_ids]
            choice_probs_L = torch.softmax(choice_logits.float(), dim=-1)
            all_choice_probs[idx, li, :] = choice_probs_L.detach().cpu().to(torch.float32)

        # ── Sub-layer states at key layers ──
        hidden_at_eig.append(sub_states["hidden"][eig_layer][0, :].cpu())
        attn_at_ratio.append(sub_states["attn"][ratio_layer][0, :].cpu())
        ffn_at_ratio.append(sub_states["ffn"][ratio_layer][0, :].cpu())

        # ── Correctness from final logits ──
        logits_last = sub_states["logits"]
        choice_logits_final = logits_last[letter_tok_ids]
        probs_final = torch.softmax(choice_logits_final.float(), dim=-1)
        pred_idx = probs_final.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        labels[idx] = int(is_correct)
        p_correct_arr[idx] = probs_final[letters.index(correct_letter)].item()

    # ── Build 7-feature matrix ──
    feature_names = [
        "eigenscore",
        "haloscope_zeta",
        "attn_ffn_ratio",
        "d2_js_top5",
        "max_p_best",
        "entropy_best",
        "top5_mass_best",
    ]

    features = np.zeros((N, 7), dtype=np.float32)

    # F1: EigenScore — NOT computed in batch (requires K× forward passes per sample)
    # We set it to NaN initially; caller can fill it via compute_eigenscore()
    features[:, 0] = np.nan

    # F2: HaloScope ζ at halo_layer
    hidden_matrix = torch.stack(hidden_at_eig).float().numpy()  # [N, d_model]
    features[:, 1] = compute_haloscope_zeta_batch(hidden_matrix, k=5)

    # F3: Attn/FFN ratio at ratio_layer
    for i, (a, f) in enumerate(zip(attn_at_ratio, ffn_at_ratio)):
        features[i, 2] = compute_attn_ffn_ratio(a, f)

    # F4: D2 JS top-5 — compute all pairs first, then select top-K by AUROC
    js_result = compute_d2_js_topk(all_choice_probs, top_k=5, exclude_layer0=True)
    js_selector = select_top_js_pairs(
        js_result["all_pair_js"], labels, top_k=5
    )
    features[:, 3] = js_selector["js_scores"]

    # F5-F7: Use last layer (closest to actual model output) for max_p/entropy/top5
    # Using the FULL dataset for layer selection picks different layers than
    # filtered data (e.g., L25 vs L27), which cripples AUROC on filtered eval.
    # L_{n_layers-1} (logit lens of final residual) is consistently best.
    last_layer = n_layers - 1  # L27 for 28-layer model
    features[:, 4] = all_maxp[:, last_layer]
    best_mp_auroc = roc_auc_score(labels, all_maxp[:, last_layer])
    print(f"  max_p at L{last_layer} (full AUROC={best_mp_auroc:.4f})")

    # F6: Entropy at last layer
    features[:, 5] = all_entropy_arr[:, last_layer]

    # F7: Top-5 mass at last layer
    features[:, 6] = all_top5[:, last_layer]
    best_mp_layer = last_layer

    result = {
        "labels": labels,
        "p_correct": p_correct_arr,
        "features": features,
        "feature_names": feature_names,
        "choice_probs": all_choice_probs,
        "all_maxp": all_maxp,
        "all_entropy": all_entropy_arr,
        "all_top5": all_top5,
        "hidden_at_eig": [h.numpy() for h in hidden_at_eig],
        "attn_at_ratio": [a.numpy() for a in attn_at_ratio],
        "ffn_at_ratio": [f.numpy() for f in ffn_at_ratio],
        "best_mp_layer": best_mp_layer,
        "best_mp_auroc": best_mp_auroc,
        "js_best_pair": js_selector["best_pair"],
        "js_best_auroc": js_selector["best_auroc"],
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Cross-dataset zero-shot evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_cross_dataset(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    scaler: StandardScaler,
    lr_model: LogisticRegression,
    feature_cols: list[int],
    dataset_name: str,
    halo_layer: int = 17,
    ratio_layer: int = 17,
) -> dict:
    """Extract features on a new dataset and evaluate with pre-trained LR.

    Only computes features that don't require per-sample K× forward passes
    (skips EigenScore to keep evaluation fast).
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
    all_top5 = np.zeros((N, n_layers), dtype=np.float32)

    hidden_at_halo = []
    attn_at_ratio = []
    ffn_at_ratio = []

    for idx, sample in enumerate(tqdm(samples, desc=f"Evaluating {dataset_name}")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset=dataset_name
        )

        if dataset_name in ("triviaqa", "squad"):
            # For open-ended datasets, correctness is judged by generation
            from src.data_loader import check_correct as _check_correct

            # Use greedy generation
            from src.hidden_state import generate_answer

            answer = generate_answer(model, prompt, max_new_tokens=20)
            is_correct = _check_correct(answer, sample["answers"], dataset=dataset_name)
            labels[idx] = int(is_correct)
            # P(correct) approximate from generation
            p_correct_arr[idx] = 1.0 if is_correct else 0.0
        else:
            correct_letter = sample["answers"][1].upper()

        # Extract states
        sub_states = extract_all_sub_layer_states(model, prompt)

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

            top5_vals, _ = torch.topk(probs_L, k=5, dim=-1)
            all_top5[idx, li] = top5_vals.sum().item()

            choice_logits = logits_L[0, letter_tok_ids]
            choice_probs_L = torch.softmax(choice_logits.float(), dim=-1)
            all_choice_probs[idx, li, :] = choice_probs_L.detach().cpu().to(torch.float32)

        hidden_at_halo.append(sub_states["hidden"][halo_layer][0, :].cpu())
        attn_at_ratio.append(sub_states["attn"][ratio_layer][0, :].cpu())
        ffn_at_ratio.append(sub_states["ffn"][ratio_layer][0, :].cpu())

        if dataset_name in ("triviaqa", "squad"):
            # Already set correctness above from generation
            pass
        else:
            logits_last = sub_states["logits"]
            choice_logits_final = logits_last[letter_tok_ids]
            probs_final = torch.softmax(choice_logits_final.float(), dim=-1)
            pred_idx = probs_final.argmax().item()
            is_correct = letters[pred_idx] == correct_letter
            labels[idx] = int(is_correct)
            p_correct_arr[idx] = probs_final[letters.index(correct_letter)].item()

    # Build feature matrix (skip EigenScore — column 0)
    N_feat = len(feature_cols)
    features = np.zeros((N, N_feat), dtype=np.float32)

    col_map = {orig_col: new_col for new_col, orig_col in enumerate(feature_cols)}

    # F2: HaloScope
    if 1 in col_map:
        hidden_matrix = torch.stack(hidden_at_halo).numpy()
        features[:, col_map[1]] = compute_haloscope_zeta_batch(hidden_matrix, k=5)

    # F3: Attn/FFN ratio
    if 2 in col_map:
        for i, (a, f) in enumerate(zip(attn_at_ratio, ffn_at_ratio)):
            features[i, col_map[2]] = compute_attn_ffn_ratio(a, f)

    # F4: D2 JS (best pair from training)
    if 3 in col_map:
        js_result = compute_d2_js_topk(all_choice_probs, top_k=5, exclude_layer0=True)
        js_selector = select_top_js_pairs(
            js_result["all_pair_js"], labels, top_k=5
        )
        features[:, col_map[3]] = js_selector["js_scores"]

    # F5: max_p at best layer
    if 4 in col_map:
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
        features[:, col_map[4]] = all_maxp[:, best_mp_layer]

    # F6/F7: Entropy/Top5 at same layer
    if 5 in col_map:
        features[:, col_map[5]] = all_entropy_arr[:, best_mp_layer]
    if 6 in col_map:
        features[:, col_map[6]] = all_top5[:, best_mp_layer]

    # Handle NaN rows
    valid_mask = ~np.isnan(features).any(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped} samples with NaN features")

    X = features[valid_mask]
    y = labels[valid_mask]

    if len(np.unique(y)) < 2:
        print(f"  WARNING: Only one class in {dataset_name} labels, AUROC undefined")
        return {"dataset": dataset_name, "n_samples": N, "auroc": float("nan")}

    # Standardize and predict
    X_scaled = scaler.transform(X)
    try:
        y_prob = lr_model.predict_proba(X_scaled)[:, 1]
        auroc = roc_auc_score(y, y_prob)
    except Exception as e:
        print(f"  ERROR: {e}")
        auroc = float("nan")

    print(f"  {dataset_name}: N={len(y)}, AUROC={auroc:.4f}")

    return {
        "dataset": dataset_name,
        "n_samples": N,
        "n_valid": len(y),
        "n_dropped": int(n_dropped),
        "auroc": float(auroc) if not np.isnan(auroc) else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="P0 Part A: Generalization Feature Set"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_cross", type=int, default=200,
                        help="Samples per cross-dataset for zero-shot eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true",
                        help="Load cached features instead of extracting")
    parser.add_argument("--skip_eigenscore", action="store_true",
                        help="Skip EigenScore computation (saves ~30min)")
    parser.add_argument("--skip_cross_dataset", action="store_true",
                        help="Skip cross-dataset zero-shot evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "generalization_features.npz"

    if args.skip_extract and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data = {
            "labels": cached["labels"],
            "p_correct": cached["p_correct"],
            "features": cached["features"],
            "feature_names": list(cached["feature_names"]),
            "choice_probs": cached["choice_probs"],
        }
    else:
        # ── Load model ──
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

        # ── Load HellaSwag ──
        print(f"\nLoading HellaSwag ({args.n_samples} samples)...")
        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        # ── Extract features ──
        print(f"\n{'=' * 60}")
        print("Phase 1: Feature Extraction")
        print(f"{'=' * 60}")

        data = extract_features(
            model,
            samples,
            letter_ids,
            eig_layer=17,
            halo_layer=17,
            ratio_layer=17,
            js_early=17,
            js_late=26,
            output_dir=output_dir,
        )

        # ── EigenScore (optional, slow) ──
        if not args.skip_eigenscore:
            print(f"\n{'=' * 60}")
            print("Phase 1b: EigenScore Computation (K=10 temperature sampling)")
            print(f"{'=' * 60}")

            from phase4_utils.generalization_features import compute_eigenscore

            for idx in tqdm(
                range(len(samples)), desc="EigenScore"
            ):
                prompt = format_prompt(
                    samples[idx]["question"],
                    samples[idx]["context"],
                    dataset="hellaswag",
                )
                try:
                    es = compute_eigenscore(
                        model, prompt, layer_idx=17, K=10, temperature=0.5
                    )
                    data["features"][idx, 0] = es
                except Exception as e:
                    print(f"  WARNING: EigenScore failed for sample {idx}: {e}")
                    data["features"][idx, 0] = np.nan
        else:
            print("\nSkipping EigenScore (--skip_eigenscore)")

        # Save cache
        np.savez_compressed(
            cache_path,
            labels=data["labels"],
            p_correct=data["p_correct"],
            features=data["features"],
            feature_names=np.array(data["feature_names"]),
            choice_probs=data["choice_probs"],
        )
        print(f"Cached features to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Knowledge filter ──
    labels = data["labels"]
    p_correct = data["p_correct"]
    features = data["features"]
    feature_names = data["feature_names"]

    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_features = features[filt_mask]
    n_filt = filt_mask.sum()

    print(f"\nFull: {labels.sum()}/{len(labels)} = {labels.mean():.4f}")
    print(f"Filtered (P>0.3): {n_filt} samples, acc={filt_labels.mean():.4f}")

    # ── Handle NaN in features ──
    valid_cols = []
    for ci in range(features.shape[1]):
        col_data = filt_features[:, ci]
        nan_frac = np.isnan(col_data).mean()
        if nan_frac > 0.5:
            print(f"  WARNING: {feature_names[ci]} has {nan_frac:.1%} NaN, excluding")
        else:
            valid_cols.append(ci)

    if len(valid_cols) == 0:
        print("ERROR: No valid features after NaN filtering")
        return

    valid_feature_names = [feature_names[ci] for ci in valid_cols]

    # Impute NaN with column median
    X = filt_features[:, valid_cols].copy()
    for ci in range(X.shape[1]):
        col = X[:, ci]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = np.nanmedian(col)

    y = filt_labels

    print(f"\nValid features ({len(valid_cols)}): {valid_feature_names}")

    # ── Per-feature AUROC ──
    print(f"\n{'=' * 60}")
    print("Per-Feature AUROC (knowledge-filtered)")
    print(f"{'=' * 60}")
    print(f"{'Feature':<20} {'AUROC':>8}")
    print("-" * 30)

    per_feature_auroc = {}
    for ci, name in enumerate(valid_feature_names):
        try:
            auc = roc_auc_score(y, X[:, ci])
        except ValueError:
            auc = float("nan")
        per_feature_auroc[name] = float(auc) if not np.isnan(auc) else None
        print(f"{name:<20} {auc:>8.4f}")

    # ── Joint LR with 5-fold CV ──
    print(f"\n{'=' * 60}")
    print("Joint Logistic Regression (5-fold CV)")
    print(f"{'=' * 60}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    try:
        joint_auroc = cross_val_score(
            lr, X_scaled, y, cv=cv, scoring="roc_auc"
        ).mean()
        print(f"Joint AUROC (all {len(valid_cols)} features): {joint_auroc:.4f}")
    except Exception as e:
        print(f"Joint LR CV failed: {e}")
        joint_auroc = float("nan")

    # ── Fit final LR on all filtered data ──
    lr.fit(X_scaled, y)

    # ── Feature importance ──
    print(f"\nFeature coefficients:")
    for ci, name in enumerate(valid_feature_names):
        print(f"  {name:<20}: {lr.coef_[0][ci]:+.4f}")

    # ── Cross-dataset zero-shot ──
    cross_results = []
    if not args.skip_cross_dataset:
        print(f"\n{'=' * 60}")
        print("Cross-Dataset Zero-Shot Evaluation")
        print(f"{'=' * 60}")

        print("\nLoading model for cross-dataset eval...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(
                f" {letter}", add_special_tokens=False
            )
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

        for dataset_name, loader_fn in [
            ("triviaqa", load_triviaqa),
            ("squad", load_squad),
        ]:
            print(f"\n--- {dataset_name.upper()} ---")
            try:
                ds_samples = loader_fn(n_samples=args.n_cross, seed=args.seed + 1)
                result = evaluate_cross_dataset(
                    model,
                    ds_samples,
                    letter_ids,
                    scaler,
                    lr,
                    valid_cols,
                    dataset_name,
                    halo_layer=17,
                    ratio_layer=17,
                )
                cross_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                cross_results.append(
                    {"dataset": dataset_name, "error": str(e)}
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ──
    output = {
        "config": {
            "n_samples": args.n_samples,
            "n_cross": args.n_cross,
            "model": args.model,
            "seed": args.seed,
            "feature_names": valid_feature_names,
        },
        "full_set": {
            "n_samples": len(labels),
            "accuracy": float(labels.mean()),
            "n_correct": int(labels.sum()),
        },
        "filtered_set": {
            "n_samples": int(n_filt),
            "accuracy": float(filt_labels.mean()),
            "n_correct": int(filt_labels.sum()),
        },
        "per_feature_auroc": per_feature_auroc,
        "joint_auroc_cv": float(joint_auroc) if not np.isnan(joint_auroc) else None,
        "lr_coefficients": {
            name: float(lr.coef_[0][ci])
            for ci, name in enumerate(valid_feature_names)
        },
        "cross_dataset_zero_shot": cross_results,
    }

    out_path = output_dir / "generalization_features_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Best single feature: {max(per_feature_auroc, key=lambda k: per_feature_auroc[k] or 0)} "
          f"= {per_feature_auroc[max(per_feature_auroc, key=lambda k: per_feature_auroc[k] or 0)]:.4f}")
    print(f"Joint LR (5-fold CV): {joint_auroc:.4f}")
    if cross_results:
        for cr in cross_results:
            auroc_str = f"{cr.get('auroc', 'N/A'):.4f}" if cr.get('auroc') is not None else "N/A"
            print(f"Zero-shot {cr['dataset']}: AUROC = {auroc_str}")


if __name__ == "__main__":
    main()
