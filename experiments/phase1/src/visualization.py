"""Visualization for q^(ℓ) results."""

from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_q_curve(results: list[dict], save_path: Path):
    """Plot q^(ℓ) vs layer index for all four estimators."""
    layers = [r["layer"] for r in results]
    q_overlap = [r.get("q_overlap", np.nan) for r in results]
    q_kl = [r.get("q_kl", np.nan) for r in results]
    q_bc = [r.get("q_bhattacharyya", np.nan) for r in results]
    q_auroc = [r.get("q_auroc", np.nan) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        layers,
        q_overlap,
        "o-",
        label="B: Overlap Area (primary)",
        linewidth=2,
        markersize=6,
    )
    ax.plot(layers, q_kl, "s--", label="A: KL Divergence", alpha=0.7)
    ax.plot(layers, q_bc, "^--", label="C: Bhattacharyya", alpha=0.7)
    ax.plot(layers, q_auroc, "d--", label="D: 1-AUROC", alpha=0.7)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("q^(ℓ) (hallucination channel width)", fontsize=12)
    ax.set_title(
        "Per-Layer Hallucination Channel Width q^(ℓ)\nPythia-1B + TriviaQA", fontsize=14
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate embedding layer
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.text(
        0, 1.02, "embed", ha="center", fontsize=9, transform=ax.get_xaxis_transform()
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distribution_overlap(confidences: list[dict], save_path: Path):
    """Plot confidence distributions for correct vs incorrect at layers 0, mid, final."""
    n_total = len(confidences)
    plot_layers = [0, n_total // 2, n_total - 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, layer_idx in zip(axes, plot_layers):
        conf_c = np.array(confidences[layer_idx]["correct"])
        conf_i = np.array(confidences[layer_idx]["incorrect"])

        if len(conf_c) < 3 or len(conf_i) < 3:
            ax.set_title(f"Layer {layer_idx} (insufficient data)")
            continue

        x_min = max(0, min(conf_c.min(), conf_i.min()) - 0.01)
        x_max = min(1, max(conf_c.max(), conf_i.max()) + 0.05)
        grid = np.linspace(x_min, x_max, 200)

        kde_c = gaussian_kde(conf_c, bw_method="scott")
        kde_i = gaussian_kde(conf_i, bw_method="scott")
        p_c = kde_c(grid)
        p_i = kde_i(grid)

        ax.fill_between(grid, p_c, alpha=0.4, label="Correct", color="green")
        ax.fill_between(grid, p_i, alpha=0.4, label="Incorrect", color="red")
        ax.plot(grid, p_c, color="green", linewidth=1.5)
        ax.plot(grid, p_i, color="red", linewidth=1.5)

        # Overlap area
        overlap = np.trapz(np.minimum(p_c / p_c.sum(), p_i / p_i.sum()), grid)
        ax.set_title(f"Layer {layer_idx}\nOverlap q = {overlap:.3f}", fontsize=11)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Confidence Distributions: Correct vs Incorrect", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_estimator_consistency(results: list[dict], save_path: Path):
    """Scatter matrix of all four estimators to check consistency."""
    valid = [r for r in results if not np.isnan(r["q_overlap"])]
    if len(valid) < 4:
        return

    estimators = {
        "Overlap": np.array([r.get("q_overlap", np.nan) for r in valid]),
        "KL": np.array([r.get("q_kl", np.nan) for r in valid]),
        "BC": np.array([r.get("q_bhattacharyya", np.nan) for r in valid]),
        "AUROC": np.array([r.get("q_auroc", np.nan) for r in valid]),
    }

    names = list(estimators.keys())
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(estimators[names[i]], bins=15, alpha=0.7)
                ax.set_title(names[i], fontsize=10)
            else:
                ax.scatter(estimators[names[j]], estimators[names[i]], alpha=0.7, s=20)
                # Correlation
                corr = np.corrcoef(estimators[names[j]], estimators[names[i]])[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"r={corr:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                )
            if j == 0:
                ax.set_ylabel(names[i], fontsize=10)
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=10)

    fig.suptitle("Estimator Consistency Matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
