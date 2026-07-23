"""P2-2: Per-Token Trajectory Shape Features on TriviaQA.

Extracts shape features from per-token max_p/entropy trajectories.
Phase 2 had cross-LAYER trajectory features; this adapts them to cross-TOKEN.

Features (per layer × per sample):
  - slope: linear fit slope over tokens
  - curvature: mean of second differences
  - half_life: token index where feature crosses 50% of range
  - saturation: token index where feature reaches 90% of final value
  - early_mean / late_mean: split at half point
  - variance: std across tokens

Then best layer × best feature → AUROC. Optional LR on all features.

Pure CPU — reads Phase 5 JSON directly. No GPU needed.

Usage:
    python main.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_token_trajectory_features(
    per_token_values: list[float],
) -> dict:
    """Extract shape features from a per-token value sequence.

    Args:
        per_token_values: list of scalar values, one per generated token.

    Returns:
        dict of feature_name → float (NaN if not applicable).
    """
    arr = np.array(per_token_values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return {f: float("nan") for f in [
            "slope", "curvature", "half_life", "saturation",
            "early_mean", "late_mean", "variance", "delta", "range",
        ]}

    features = {}

    # Slope: linear fit
    x = np.arange(n, dtype=np.float64)
    features["slope"] = float(np.polyfit(x, arr, 1)[0])

    # Curvature: mean of second differences
    if n >= 3:
        d2 = np.diff(arr, n=2)
        features["curvature"] = float(np.mean(np.abs(d2)))
    else:
        features["curvature"] = 0.0

    # Range
    features["range"] = float(arr.max() - arr.min())

    # Delta (last - first)
    features["delta"] = float(arr[-1] - arr[0])

    # Half-life: token index where value crosses halfway between min and max
    halfway = (arr.min() + arr.max()) / 2
    diff = np.abs(arr - halfway)
    features["half_life"] = float(np.argmin(diff)) / n

    # Saturation: token index where value reaches 90% of max deviation
    threshold = arr.min() + 0.9 * (arr.max() - arr.min())
    saturated = np.where(arr >= threshold)[0]
    features["saturation"] = float(saturated[0]) / n if len(saturated) > 0 else 1.0

    # Early vs late mean
    half = max(1, n // 2)
    features["early_mean"] = float(arr[:half].mean())
    features["late_mean"] = float(arr[half:].mean())

    # Variance
    features["variance"] = float(arr.var(ddof=1))

    return features


def main():
    # ── Load Phase 5 data ─────────────────────────────────────────────
    phase5_json = (
        Path(__file__).parent.parent.parent
        / "phase5_cross_task"
        / "outputs"
        / "triviaqa_features.json"
    )
    with open(phase5_json) as f:
        p5 = json.load(f)

    samples = p5["per_sample"]
    n_layers = p5["config"]["n_layers"]
    labels = np.array([s["is_correct"] for s in samples], dtype=np.int32)
    print(f"  Samples: {len(samples)}, layers: {n_layers}")
    print(f"  Accuracy: {labels.sum()}/{len(labels)} = {labels.sum()/len(labels):.4f}")

    # Feature names for per-token trajectory
    feat_names = [
        "slope", "curvature", "half_life", "saturation",
        "early_mean", "late_mean", "variance", "delta", "range",
    ]
    n_feats = len(feat_names)

    # ── Extract features: [N, n_layers, n_feats] ──────────────────────
    print(f"\nExtracting per-token trajectory features for max_p...")
    all_feats_maxp = np.full(
        (len(samples), n_layers, n_feats), np.nan, dtype=np.float64
    )

    for si, sample in enumerate(samples):
        per_token = sample["per_token"]
        if not per_token:
            continue
        for li in range(n_layers):
            values = [t["max_p"][li] for t in per_token]
            feats = extract_token_trajectory_features(values)
            for fi, fn in enumerate(feat_names):
                all_feats_maxp[si, li, fi] = feats[fn]

    # ── Per-layer × per-feature AUROC ─────────────────────────────────
    print(f"\n{'Layer':<6} {'Feature':<14} {'AUROC':<10}")
    print("-" * 32)

    best_layer, best_feat, best_auroc = -1, "", 0.5

    for li in range(n_layers):
        for fi, fn in enumerate(feat_names):
            scores = all_feats_maxp[:, li, fi]
            valid = ~np.isnan(scores)
            if valid.sum() < 10:
                continue

            try:
                auc = roc_auc_score(
                    1 - labels[valid],
                    scores[valid],
                )
            except ValueError:
                auc = 0.5

            if auc > best_auroc:
                best_auroc = auc
                best_layer = li
                best_feat = fn
                marker = " *** BEST"
            else:
                marker = ""

            if auc > best_auroc * 0.95:  # Show top performers
                print(f"  L{li:<4} {fn:<14} {auc:<10.4f}{marker}")

    print(f"\n  Best: L{best_layer} {best_feat} = {best_auroc:.4f}")

    # ── Joint LR: all non-NaN features ─────────────────────────────────
    print(f"\n--- Joint LR (all layers × all features) ---")
    # Flatten: [N, n_layers * n_feats]
    X_flat = all_feats_maxp.reshape(len(samples), -1)
    valid_cols = ~np.all(np.isnan(X_flat), axis=0)
    X_valid = X_flat[:, valid_cols]
    valid_rows = ~np.any(np.isnan(X_valid), axis=1)
    X_valid = X_valid[valid_rows]
    y_valid = labels[valid_rows]

    print(f"  Samples: {len(y_valid)}/{len(samples)} (no NaN rows)")
    print(f"  Features: {X_valid.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(lr, X_scaled, y_valid, cv=cv, scoring="roc_auc")
        joint_auroc = float(cv_scores.mean())
        joint_std = float(cv_scores.std())
        print(f"  Joint CV AUROC: {joint_auroc:.4f} ± {joint_std:.4f}")
    except Exception as e:
        print(f"  CV failed: {e}")
        joint_auroc = float("nan")
        joint_std = float("nan")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "config": {"n_samples": len(samples), "n_layers": n_layers,
                    "n_features_per_layer": n_feats},
        "best_single": {"layer": best_layer, "feature": best_feat,
                         "auroc": best_auroc},
        "joint_lr_cv": {"mean": joint_auroc, "std": joint_std},
    }

    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / 'results.json'}")


if __name__ == "__main__":
    main()
