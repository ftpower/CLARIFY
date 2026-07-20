"""Direction 3: Information Flow Trajectory Shape Analysis.

Plan A: Hand-crafted shape features + Logistic Regression / Random Forest.
Plan C: Group-level trajectory difference visualization.

Usage:
    python main_trajectory_shape.py
    python main_trajectory_shape.py --input outputs/entropy_diagnosis.json
"""

import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.trajectory_features import extract_trajectory_features


def load_data(input_path: str) -> tuple[list[dict], dict]:
    """Load per-sample trajectory data from diagnosis JSON."""
    with open(input_path) as f:
        data = json.load(f)
    samples = data["per_sample"]
    config = data["config"]
    print(
        f"Loaded {len(samples)} samples (accuracy={config['accuracy']:.4f}, "
        f"n_layers={config['n_layers']})"
    )
    return samples, config


def extract_all_features(
    samples: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract trajectory features for all samples."""
    feature_list = []
    labels = []
    for s in samples:
        feats = extract_trajectory_features(s["entropy"], s["max_prob"], s["top5_mass"])
        feature_list.append(feats)
        labels.append(1 if s["is_correct"] else 0)

    # Build matrix
    feature_names = sorted(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])
    y = np.array(labels)

    print(f"Feature matrix: {X.shape} ({len(feature_names)} features)")
    return X, y, feature_names


def print_feature_stats(feature_names: list[str], X: np.ndarray, y: np.ndarray):
    """Print per-feature group means and Cohen's d."""
    mask_c = y == 1
    mask_i = y == 0
    print(f"\n{'Feature':<28s} {'Correct':>10s} {'Incorrect':>10s} {'|d|':>8s}")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        mc = X[mask_c, i].mean()
        mi = X[mask_i, i].mean()
        sc = X[mask_c, i].std()
        si = X[mask_i, i].std()
        pooled_std = np.sqrt((sc**2 + si**2) / 2)
        d = abs(mc - mi) / pooled_std if pooled_std > 0 else 0.0
        print(f"{name:<28s} {mc:10.4f} {mi:10.4f} {d:8.4f}")


def classify_lr(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """5-fold CV Logistic Regression. Returns AUROC, accuracy, F1, feature importance."""
    print("\n--- Logistic Regression (5-fold CV) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(
        LogisticRegression(max_iter=2000, random_state=42),
        X_scaled,
        y,
        cv=cv,
        method="predict",
    )
    y_proba = cross_val_predict(
        LogisticRegression(max_iter=2000, random_state=42),
        X_scaled,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    auroc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"  AUROC: {auroc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1: {f1:.4f}")

    # Feature importance from LR coefficients (fit on full data for interpretability)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_scaled, y)
    importance = np.abs(lr.coef_[0])
    ranked = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print("\n  Top-10 features (|coef|):")
    for i, (name, imp) in enumerate(ranked[:10]):
        print(f"    {i + 1}. {name}: {imp:.4f}")

    return {
        "model": "LogisticRegression",
        "auroc": float(auroc),
        "accuracy": float(acc),
        "f1": float(f1),
        "feature_importance": [(name, float(imp)) for name, imp in ranked],
    }


def classify_rf(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """5-fold CV Random Forest as validation."""
    print("\n--- Random Forest (5-fold CV) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X,
        y,
        cv=cv,
        method="predict",
    )
    y_proba = cross_val_predict(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    auroc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"  AUROC: {auroc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1: {f1:.4f}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    ranked = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])

    print("\n  Top-10 features (RF importance):")
    for i, (name, imp) in enumerate(ranked[:10]):
        print(f"    {i + 1}. {name}: {imp:.4f}")

    return {
        "model": "RandomForest",
        "auroc": float(auroc),
        "accuracy": float(acc),
        "f1": float(f1),
        "feature_importance": [(name, float(imp)) for name, imp in ranked],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plan C: Group-level trajectory visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_trajectory_comparison(samples: list[dict], config: dict, output_path: Path):
    """Generate 2x3 panel figure: trajectory comparison + statistical tests."""
    n_total = config["n_total_layers"]
    layers = np.arange(n_total)

    correct = [s for s in samples if s["is_correct"]]
    incorrect = [s for s in samples if not s["is_correct"]]

    ent_c = np.array([s["entropy"] for s in correct])
    ent_i = np.array([s["entropy"] for s in incorrect])
    mp_c = np.array([s["max_prob"] for s in correct])
    mp_i = np.array([s["max_prob"] for s in incorrect])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Entropy trajectory: mean ± SE
    ax = axes[0, 0]
    _plot_band(ax, layers, ent_c, "blue", "Correct")
    _plot_band(ax, layers, ent_i, "red", "Incorrect")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Logit Lens Entropy H(l)")
    ax.set_title("(a) Entropy Trajectory (mean ± 1 SE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) max_prob trajectory
    ax = axes[0, 1]
    _plot_band(ax, layers, mp_c, "blue", "Correct")
    _plot_band(ax, layers, mp_i, "red", "Incorrect")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Probability")
    ax.set_title("(b) Max Prob Trajectory (mean ± 1 SE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Per-layer Cohen's d
    ax = axes[0, 2]
    d_values = []
    for li in range(n_total):
        mc, mi = ent_c[:, li].mean(), ent_i[:, li].mean()
        sc, si = ent_c[:, li].std(ddof=1), ent_i[:, li].std(ddof=1)
        pooled = np.sqrt((sc**2 + si**2) / 2)
        d = (mc - mi) / pooled if pooled > 0 else 0.0
        d_values.append(abs(d))
    colors = [
        "#2ecc71" if d < 0.2 else "#f39c12" if d < 0.5 else "#e74c3c" for d in d_values
    ]
    ax.bar(layers, d_values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="small (0.2)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="medium (0.5)")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.9, label="large (0.8)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's |d|")
    ax.set_title("(c) Per-Layer Effect Size (entropy)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Per-layer KS test
    ax = axes[1, 0]
    from scipy.stats import ks_2samp

    p_values = []
    for li in range(n_total):
        _, p = ks_2samp(ent_c[:, li], ent_i[:, li])
        p_values.append(p)
    neg_log_p = [-np.log10(max(p, 1e-30)) for p in p_values]
    bonferroni = -np.log10(0.05 / n_total)
    colors = ["#e74c3c" if nlp > bonferroni else "#bdc3c7" for nlp in neg_log_p]
    ax.bar(layers, neg_log_p, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(
        y=bonferroni,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Bonferroni ({0.05 / n_total:.1e})",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("-log10(p)")
    ax.set_title("(d) KS Test: Correct vs Incorrect Entropy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (e) Cumulative difference
    ax = axes[1, 1]
    mean_c = ent_c.mean(axis=0)
    mean_i = ent_i.mean(axis=0)
    abs_diff = np.abs(mean_c - mean_i)
    cum_diff = np.cumsum(abs_diff)
    ax.fill_between(layers, 0, cum_diff, alpha=0.3, color="purple")
    ax.plot(layers, cum_diff, "purple", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative |ΔH|")
    ax.set_title("(e) Cumulative Absolute Difference")
    ax.grid(True, alpha=0.3)

    # (f) First derivative comparison
    ax = axes[1, 2]
    d1_c = -np.diff(mean_c)  # negative diff = entropy DROP
    d1_i = -np.diff(mean_i)
    ax.plot(layers[1:], d1_c, "b-", linewidth=2, label="Correct")
    ax.plot(layers[1:], d1_i, "r-", linewidth=2, label="Incorrect")
    # Mark layer with max derivative difference
    diff_d1 = np.abs(d1_c - d1_i)
    max_diff_layer = np.argmax(diff_d1) + 1
    ax.axvline(
        x=max_diff_layer,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"max Δ at L{max_diff_layer}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("-ΔH / Δlayer (entropy drop per layer)")
    ax.set_title("(f) First Derivative: Convergence Rate")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Trajectory Shape Analysis — {config['model_id']} {config['dataset'].upper()} "
        f"(n={len(samples)}, acc={config['accuracy']:.3f})",
        fontsize=13,
    )
    fig.tight_layout()

    plot_file = output_path / "trajectory_shape.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_file}")


def _plot_band(ax, layers, data, color, label):
    """Plot mean ± 1 SE band."""
    mean = data.mean(axis=0)
    se = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    ax.plot(layers, mean, color=color, linewidth=2, label=label)
    ax.fill_between(layers, mean - se, mean + se, alpha=0.15, color=color)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(
    input_path: str = "outputs/entropy_diagnosis.json", output_dir: str = "outputs"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    samples, config = load_data(input_path)

    # ── Plan A: Feature extraction + classification ──────────────────────
    print("\n" + "=" * 60)
    print("Plan A: Shape Feature Classification")
    print("=" * 60)

    X, y, feature_names = extract_all_features(samples)
    print_feature_stats(feature_names, X, y)

    lr_result = classify_lr(X, y, feature_names)
    rf_result = classify_rf(X, y, feature_names)

    # Baseline comparison
    print(f"\n{'=' * 60}")
    print(f"Baseline (single-point max_p at L28): AUROC = 0.68")
    print(f"Trajectory shape LR:                 AUROC = {lr_result['auroc']:.4f}")
    print(f"Trajectory shape RF:                 AUROC = {rf_result['auroc']:.4f}")
    delta = lr_result["auroc"] - 0.68
    if delta > 0.01:
        print(f"✓ Trajectory shape BEATS single-point baseline by +{delta:.4f}")
    elif delta > -0.01:
        print(f"≈ Trajectory shape TIED with single-point baseline (Δ={delta:+.4f})")
    else:
        print(
            f"✗ Trajectory shape does NOT beat single-point baseline (Δ={delta:+.4f})"
        )

    # ── Plan C: Group-level visualization ────────────────────────────────
    print("\n" + "=" * 60)
    print("Plan C: Trajectory Comparison Visualization")
    print("=" * 60)
    plot_trajectory_comparison(samples, config, output_path)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "config": config,
        "plan_a": {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "logistic_regression": lr_result,
            "random_forest": rf_result,
            "baseline_auroc": 0.68,
            "baseline_description": "single-point max_p at L28",
        },
    }

    results_file = output_path / "trajectory_shape_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/entropy_diagnosis.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args.input, args.output_dir)
