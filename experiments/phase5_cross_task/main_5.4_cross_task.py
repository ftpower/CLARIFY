"""Exp 5.4: Task-Agnostic Detector — zero-shot cross-task LR.

Tests whether hallucination detection features generalize across tasks:
  - Train on HellaSwag → Test on TriviaQA (target: AUROC > 0.7)
  - Train on TriviaQA → Test on HellaSwag
  - Joint training with cross-validation

Uses aligned features:
  - d2_js: layer-wise JS divergence (4-choice on HellaSwag, full-vocab on TriviaQA)
  - max_p: maximum softmax probability (4-choice vs full-vocab)
  - entropy: logit-lens entropy (4-choice vs full-vocab)

Because features operate on different probability spaces (4-choice vs 152K vocab),
values are NOT directly comparable in absolute terms. StandardScaler is applied
per-dataset to handle scale differences.

Usage:
    python main_5.4_cross_task.py
    python main_5.4_cross_task.py --hellaswag_file ../phase4_generalization/outputs/generalization_features.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# phase5_utils not needed for 5.4 — only uses numpy/sklearn
sys.path.insert(0, str(Path(__file__).parent))


def main(
    triviaqa_file: str = "outputs/triviaqa_features.json",
    hellaswag_file: str = "../phase4_generalization/outputs/generalization_features.npz",
    output_dir: str = "outputs",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load TriviaQA features ─────────────────────────────────────────
    print("Loading TriviaQA features...")
    with open(triviaqa_file) as f:
        tqa_data = json.load(f)

    tqa_samples = tqa_data["per_sample"]
    tqa_config = tqa_data["config"]

    # Extract last-token features from valid samples
    tqa_X, tqa_y = _extract_triviaqa_features(tqa_samples)
    print(f"  TriviaQA: X={tqa_X.shape}, y={tqa_y.shape}, "
          f"correct={tqa_y.sum()}/{len(tqa_y)}")

    # ── Load HellaSwag features ────────────────────────────────────────
    print(f"Loading HellaSwag features from {hellaswag_file}...")
    hs_path = Path(hellaswag_file)
    if not hs_path.exists():
        print(f"  ERROR: HellaSwag file not found at {hs_path}")
        print("  Run Phase 4's main_generalization_features.py first, or specify --hellaswag_file")
        return

    hs_data = np.load(hs_path, allow_pickle=True)
    hs_features = hs_data["features"]  # [N, 7]
    hs_feature_names = list(hs_data["feature_names"])
    hs_labels = hs_data["labels"]

    # Feature indices in HellaSwag's 7-feature matrix:
    #   0: eigenscore (all NaN, skip)
    #   1: haloscope_zeta
    #   2: attn_ffn_ratio
    #   3: d2_js_top5
    #   4: max_p_best
    #   5: entropy_best
    #   6: top5_mass_best
    hs_d2_js = hs_features[:, 3]   # d2_js_top5
    hs_max_p = hs_features[:, 4]    # max_p_best
    hs_entropy = hs_features[:, 5]  # entropy_best

    # Build HellaSwag feature matrix [N, 3]
    hs_X = np.column_stack([hs_d2_js, hs_max_p, hs_entropy])
    hs_y = hs_labels.astype(np.int32)

    # Drop NaN rows
    hs_valid = np.all(np.isfinite(hs_X), axis=1)
    hs_X = hs_X[hs_valid]
    hs_y = hs_y[hs_valid]

    print(f"  HellaSwag: X={hs_X.shape}, y={hs_y.shape}, "
          f"correct={hs_y.sum()}/{len(hs_y)}")
    print(f"  Feature names: [d2_js_top5, max_p_best, entropy_best]")

    # ── Check class balance ────────────────────────────────────────────
    for name, y in [("HellaSwag", hs_y), ("TriviaQA", tqa_y)]:
        frac = y.sum() / len(y)
        if frac < 0.1 or frac > 0.9:
            print(f"  WARNING: {name} class imbalance: {frac:.3f} correct")

    # ── Experiment A: HellaSwag → TriviaQA ────────────────────────────
    print("\n" + "=" * 72)
    print("Experiment A: Train HellaSwag → Test TriviaQA (zero-shot)")
    print("-" * 72)
    expA_auroc = _cross_dataset_auroc(hs_X, hs_y, tqa_X, tqa_y)
    print(f"  Zero-shot AUROC: {expA_auroc:.4f}")

    # ── Experiment B: TriviaQA → HellaSwag ────────────────────────────
    print("\n" + "=" * 72)
    print("Experiment B: Train TriviaQA → Test HellaSwag (zero-shot)")
    print("-" * 72)
    expB_auroc = _cross_dataset_auroc(tqa_X, tqa_y, hs_X, hs_y)
    print(f"  Zero-shot AUROC: {expB_auroc:.4f}")

    # ── Experiment C: In-domain baselines ──────────────────────────────
    print("\n" + "=" * 72)
    print("Experiment C: In-domain baselines (5-fold CV)")
    print("-" * 72)
    expC_hs = _in_domain_cv(hs_X, hs_y, "HellaSwag")
    expC_tqa = _in_domain_cv(tqa_X, tqa_y, "TriviaQA")
    print(f"  HellaSwag in-domain CV AUROC: {expC_hs:.4f}")
    print(f"  TriviaQA in-domain CV AUROC: {expC_tqa:.4f}")

    # ── Experiment D: Joint training ──────────────────────────────────
    print("\n" + "=" * 72)
    print("Experiment D: Joint training (HellaSwag + TriviaQA, 5-fold CV)")
    print("-" * 72)
    joint_X = np.concatenate([hs_X, tqa_X], axis=0)
    joint_y = np.concatenate([hs_y, tqa_y], axis=0)
    # Also need per-dataset scaling before joint
    scaler = StandardScaler()
    joint_X_scaled = scaler.fit_transform(joint_X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            lr, joint_X_scaled, joint_y, cv=cv, scoring="roc_auc"
        )
        expD_auroc = float(cv_scores.mean())
        expD_std = float(cv_scores.std())
        print(f"  Joint CV AUROC: {expD_auroc:.4f} ± {expD_std:.4f}")
    except Exception as e:
        print(f"  Joint CV failed: {e}")
        expD_auroc = float("nan")
        expD_std = float("nan")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Cross-Task Generalization Summary")
    print("=" * 72)
    print(f"  {'Experiment':<40} {'AUROC':<10}")
    print(f"  {'-' * 40} {'-' * 10}")
    print(f"  {'A: HellaSwag → TriviaQA (zero-shot)':<40} {expA_auroc:.4f}")
    print(f"  {'B: TriviaQA → HellaSwag (zero-shot)':<40} {expB_auroc:.4f}")
    print(f"  {'C: HellaSwag in-domain (5-fold CV)':<40} {expC_hs:.4f}")
    print(f"  {'C: TriviaQA in-domain (5-fold CV)':<40} {expC_tqa:.4f}")
    print(f"  {'D: Joint training (5-fold CV)':<40} {expD_auroc:.4f}")

    # ── Interpretation ─────────────────────────────────────────────────
    print("\n--- Interpretation ---")
    if expA_auroc > 0.7 and expB_auroc > 0.7:
        print("  ✓ BOTH zero-shot directions > 0.7 — genuine cross-task generalization!")
    elif expA_auroc > 0.7 or expB_auroc > 0.7:
        direction = "HellaSwag→TriviaQA" if expA_auroc > 0.7 else "TriviaQA→HellaSwag"
        print(f"  ~ One-way transfer: {direction} succeeds, the other doesn't")
    else:
        print("  ✗ Zero-shot AUROC < 0.7 in both directions.")
        if expA_auroc > 0.6 and expB_auroc > 0.6:
            print("    Moderate transfer — features carry some signal across tasks")
        else:
            print("    Features are task-specific — cross-task detection is weak")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "config": {
            "triviaqa_file": str(triviaqa_file),
            "hellaswag_file": str(hellaswag_file),
            "tqa_config": tqa_config,
            "features": ["d2_js", "max_p", "entropy"],
            "note": "HellaSwag features: 4-choice softmax space. TriviaQA features: full-vocab softmax space. Different absolute scales.",
            "hallaswag_in_domain_d2_js_auroc": _get_phase4_auroc("d2_js_top5"),
            "hallaswag_in_domain_max_p_auroc": _get_phase4_auroc("max_p_best"),
        },
        "cross_task": {
            "A_hellaswag_to_triviaqa": expA_auroc,
            "B_triviaqa_to_hellaswag": expB_auroc,
            "C_hellaswag_in_domain_cv": expC_hs,
            "C_triviaqa_in_domain_cv": expC_tqa,
            "D_joint_cv_mean": expD_auroc,
            "D_joint_cv_std": expD_std,
        },
    }

    output_file = output_path / "cross_task_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def _extract_triviaqa_features(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned features from TriviaQA per-sample data.

    Returns (X, y) where X has columns: [d2_js_last, max_p_L27, entropy_L27].
    """
    d2_js_vals = []
    max_p_vals = []
    entropy_vals = []
    labels = []

    for s in samples:
        if not s["per_token"]:
            continue
        last = s["per_token"][-1]
        d2_js_vals.append(last["d2_js"])
        # L27 is index 27 (28 layers, 0-indexed)
        max_p_vals.append(last["max_p"][-1])
        entropy_vals.append(last["entropy"][-1])
        labels.append(s["is_correct"])

    X = np.column_stack([
        np.array(d2_js_vals, dtype=np.float64),
        np.array(max_p_vals, dtype=np.float64),
        np.array(entropy_vals, dtype=np.float64),
    ])
    y = np.array(labels, dtype=np.int32)
    return X, y


def _cross_dataset_auroc(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    """Train on one dataset, test on another. Returns AUROC."""
    # Standardize: fit on train, apply to both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LR
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        lr.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"    LR fit failed: {e}")
        return float("nan")

    # Predict on test
    try:
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        auroc = float(roc_auc_score(y_test, y_pred_proba))
    except Exception as e:
        print(f"    Prediction failed: {e}")
        return float("nan")

    print(f"    LR coef: {lr.coef_[0].tolist()}")
    return auroc


def _in_domain_cv(X: np.ndarray, y: np.ndarray, name: str) -> float:
    """5-fold CV within a single dataset."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())
    except Exception as e:
        print(f"    {name} CV failed: {e}")
        return float("nan")


def _get_phase4_auroc(feature_name: str) -> float | None:
    """Try to read Phase 4 per-feature AUROC from saved results."""
    try:
        hs_path = Path(
            "../phase4_generalization/outputs/generalization_features_results.json"
        )
        with open(hs_path) as f:
            data = json.load(f)
        return data.get("per_feature_auroc", {}).get(feature_name)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.4: Cross-task zero-shot generalization"
    )
    parser.add_argument(
        "--triviaqa_file", type=str, default="outputs/triviaqa_features.json"
    )
    parser.add_argument(
        "--hellaswag_file", type=str,
        default="../phase4_generalization/outputs/generalization_features.npz",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    main(
        triviaqa_file=args.triviaqa_file,
        hellaswag_file=args.hellaswag_file,
        output_dir=args.output_dir,
    )
