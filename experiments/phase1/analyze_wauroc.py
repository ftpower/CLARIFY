"""wAUROC analysis: knowledge-weighted AUROC to separate ignorance from hallucination.

Usage:
    python analyze_wauroc.py outputs_qwen3_8b_200/per_sample.json
    python analyze_wauroc.py outputs_qwen3_8b_hellaswag/per_sample.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score


def bootstrap_auroc(y_true, y_score, n_bootstrap=2000, ci=95):
    """Bootstrap AUROC with confidence interval."""
    y_true, y_score = np.array(y_true), np.array(y_score)
    n = len(y_true)
    aurocs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    aurocs = np.array(auroscs)
    lo = np.percentile(auroscs, (100 - ci) / 2)
    hi = np.percentile(auroscs, 100 - (100 - ci) / 2)
    return np.mean(auroscs), lo, hi


def compute_wauroc(labels, confidences, weights, n_bootstrap=2000):
    """Weighted AUROC: each pair (j,k) weighted by w_j * w_k.

    Uses the pairwise formulation: wAUROC = sum(w_j * w_k * I(c_j > c_k) * I(y_j > y_k))
                                          / sum(w_j * w_k * I(y_j > y_k))
    where y=1 for correct, y=0 for incorrect.
    """
    n = len(labels)
    labels = np.array(labels)
    confs = np.array(confidences)
    w = np.array(weights)

    # Normalize weights to avoid numerical issues
    w = w / w.max()

    total_weight = 0.0
    total_pairs = 0.0
    for j in range(n):
        if labels[j] == 0:
            continue
        for k in range(n):
            if labels[k] == 1:
                continue
            pair_weight = w[j] * w[k]
            if pair_weight == 0:
                continue
            total_weight += pair_weight
            if confs[j] > confs[k]:
                total_pairs += pair_weight
            elif confs[j] == confs[k]:
                total_pairs += 0.5 * pair_weight

    if total_weight == 0:
        return np.nan, np.nan, np.nan

    wauroc = total_pairs / total_weight

    # Bootstrap CI
    rng = np.random.RandomState(42)
    waurocs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        l, c, w_ = labels[idx], confs[idx], w[idx]
        tw, tp = 0.0, 0.0
        for jj in range(n):
            if l[jj] == 0:
                continue
            for kk in range(n):
                if l[kk] == 1:
                    continue
                pw = w_[jj] * w_[kk]
                if pw == 0:
                    continue
                tw += pw
                if c[jj] > c[kk]:
                    tp += pw
                elif c[jj] == c[kk]:
                    tp += 0.5 * pw
        waurocs.append(tp / tw if tw > 0 else np.nan)

    waurocs = np.array(waurocs)
    waurocs = waurocs[~np.isnan(waurocs)]
    lo = np.percentile(waurocs, 2.5)
    hi = np.percentile(waurocs, 97.5)
    return wauroc, lo, hi


def main(data_path: str):
    data = json.loads(Path(data_path).read_text())
    labels = [int(s["is_correct"]) for s in data]
    p_c = [s["p_correct"] for s in data]  # P(correct | final_logits) as knowledge proxy
    n_samples = len(labels)
    n_layers = len(data[0]["dot_confidences"])

    print(f"File: {data_path}")
    print(f"Samples: {n_samples}")
    print(f"Correct: {sum(labels)}, Incorrect: {n_samples - sum(labels)}")
    print(f"Class balance: {sum(labels)/n_samples*100:.1f}%")
    print(f"P(correct) mean: {np.mean(p_c):.6f}, median: {np.median(p_c):.6f}")
    print()

    # --- Per-layer AUROC (unweighted baseline) ---
    best_auroc = 0.0
    best_layer = -1
    print(f"{'Layer':>6} {'AUROC':>8} {'95% CI':>20} {'wAUROC':>8} {'95% CI':>20} {'Δ':>8}")
    print("-" * 75)

    for li in range(n_layers):
        # Dot-product confidence at this layer
        confs = [s["dot_confidences"][li] for s in data]
        y = labels

        # Unweighted AUROC
        try:
            auc = roc_auc_score(y, confs)
        except ValueError:
            auc = np.nan
        auc_mean, auc_lo, auc_hi = bootstrap_auroc(y, confs)

        # Weighted AUROC with P(correct) as sample weight
        wauroc, w_lo, w_hi = compute_wauroc(y, confs, p_c)

        delta = ""
        if not np.isnan(auc) and not np.isnan(wauroc):
            d = wauroc - auc
            delta = f"{d:+.4f}"
        elif not np.isnan(auc):
            delta = "  nan"

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "   nan"
        wauroc_str = f"{wauroc:.4f}" if not np.isnan(wauroc) else "   nan"
        ci_str = f"[{auc_lo:.4f}, {auc_hi:.4f}]" if not np.isnan(auc) else "       nan"
        wci_str = f"[{w_lo:.4f}, {w_hi:.4f}]" if not np.isnan(wauroc) else "       nan"

        print(f"{li:>6} {auc_str:>8} {ci_str:>20} {wauroc_str:>8} {wci_str:>20} {delta:>8}")

        if not np.isnan(auc) and auc > best_auroc:
            best_auroc = auc
            best_layer = li

    print(f"\nBest AUROC layer: L{best_layer} = {best_auroc:.4f}")

    # --- P(correct) distribution by correctness ---
    p_c_arr = np.array(p_c)
    print(f"\nP(correct) by label:")
    for label_val, label_name in [(1, "correct"), (0, "incorrect")]:
        mask = np.array(labels) == label_val
        vals = p_c_arr[mask]
        if len(vals) > 0:
            print(f"  {label_name}: mean={vals.mean():.6f} median={np.median(vals):.6f} "
                  f"min={vals.min():.6f} max={vals.max():.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_wauroc.py <per_sample.json>")
        sys.exit(1)
    main(sys.argv[1])
