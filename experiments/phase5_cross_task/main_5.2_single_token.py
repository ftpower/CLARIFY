"""Exp 5.2: Single-Token Feature Migration — last-token AUROC evaluation.

Reads the output of Exp 5.1 (triviaqa_features.json) and evaluates per-feature
AUROC at the LAST generated token position. This is the closest analog to
HellaSwag's single-token evaluation.

Evaluated features:
  - max_p at each layer (logit lens)
  - entropy at each layer (logit lens)
  - d2_js (L_early vs L_late full-vocab JS at the last token)
  - d2_js all layer pairs (scans all 378 pairs for best pair)

Compares results with known HellaSwag findings from Phase 4.

Usage:
    python main_5.2_single_token.py
    python main_5.2_single_token.py --input_file outputs/triviaqa_features.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main(
    input_file: str = "outputs/triviaqa_features.json",
    output_dir: str = "outputs",
    js_early: int = 11,
    js_late: int = 27,
):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    config = data["config"]
    per_sample = data["per_sample"]
    n_samples = len(per_sample)
    n_layers = config["n_layers"]
    n_correct = config["correct_count"]
    n_incorrect = config["incorrect_count"]

    print(f"  Samples: {n_samples} ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"  Accuracy: {config['accuracy']:.4f}")
    print(f"  Layers: {n_layers}")

    # ── Extract labels ─────────────────────────────────────────────────
    labels = np.array([s["is_correct"] for s in per_sample], dtype=np.int32)
    if labels.sum() == 0 or labels.sum() == n_samples:
        print("WARNING: All samples have the same label. AUROC is undefined.")
        return

    # ── Extract last-token features ────────────────────────────────────
    # Filter out samples with no generated tokens
    valid_indices = [
        i for i, s in enumerate(per_sample) if len(s["per_token"]) > 0
    ]
    if len(valid_indices) < n_samples:
        n_skipped = n_samples - len(valid_indices)
        print(f"  Skipping {n_skipped} samples with zero generated tokens")

    labels_valid = labels[valid_indices]
    samples_valid = [per_sample[i] for i in valid_indices]

    # Last token max_p: [N, n_layers]
    max_p_last = np.array([
        s["per_token"][-1]["max_p"] for s in samples_valid
    ], dtype=np.float64)

    # Last token entropy: [N, n_layers]
    entropy_last = np.array([
        s["per_token"][-1]["entropy"] for s in samples_valid
    ], dtype=np.float64)

    # Last token d2_js: [N]
    d2_js_last = np.array([
        s["per_token"][-1]["d2_js"] for s in samples_valid
    ], dtype=np.float64)

    print(f"\n  Feature shapes: max_p {max_p_last.shape}, "
          f"entropy {entropy_last.shape}, d2_js {d2_js_last.shape}")

    # ── Per-layer AUROC for max_p ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("max_p AUROC (per layer, last token)")
    print("-" * 72)
    max_p_aurocs = {}
    for li in range(n_layers):
        scores = max_p_last[:, li]
        if np.all(np.isfinite(scores)):
            try:
                auc = roc_auc_score(labels_valid, scores)
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")
        max_p_aurocs[li] = auc

    for li in range(n_layers):
        auc = max_p_aurocs[li]
        marker = " *** BEST" if li == max(max_p_aurocs, key=max_p_aurocs.get) else ""
        print(f"  L{li:>2}: AUROC = {auc:.4f}" + (marker if not np.isnan(auc) else ""))

    best_max_p_layer = max(
        (li for li in range(n_layers) if not np.isnan(max_p_aurocs[li])),
        key=lambda li: max_p_aurocs[li],
        default=-1,
    )
    best_max_p_auroc = max_p_aurocs.get(best_max_p_layer, float("nan"))

    # ── Per-layer AUROC for entropy ────────────────────────────────────
    print("\n" + "=" * 72)
    print("Entropy AUROC (per layer, last token)")
    print("-" * 72)
    entropy_aurocs = {}
    for li in range(n_layers):
        scores = entropy_last[:, li]
        if np.all(np.isfinite(scores)):
            try:
                auc = roc_auc_score(labels_valid, scores)
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")
        entropy_aurocs[li] = auc

    best_entropy_layer = max(
        (li for li in range(n_layers) if not np.isnan(entropy_aurocs[li])),
        key=lambda li: entropy_aurocs[li],
        default=-1,
    )
    best_entropy_auroc = entropy_aurocs.get(best_entropy_layer, float("nan"))
    print(f"  Best: L{best_entropy_layer} = {best_entropy_auroc:.4f}")
    print(f"  L27:  = {entropy_aurocs.get(27, float('nan')):.4f}")

    # ── d2_js at last token (L_early vs L_late) ────────────────────────
    print("\n" + "=" * 72)
    print(f"d2_js AUROC (L{js_early} vs L{js_late}, last token)")
    print("-" * 72)
    try:
        d2_js_auroc = roc_auc_score(labels_valid, d2_js_last)
        print(f"  AUROC = {d2_js_auroc:.4f}")
    except ValueError:
        d2_js_auroc = float("nan")
        print("  AUROC = nan (all same label)")

    # ── Best layer pair from all-pair JS ───────────────────────────────
    print("\n" + "=" * 72)
    print("Best layer pair from last_token_js_all_pairs")
    print("-" * 72)
    all_pair_aurocs = _evaluate_all_pairs(samples_valid, labels_valid, n_layers)
    if all_pair_aurocs:
        sorted_pairs = sorted(all_pair_aurocs.items(), key=lambda x: x[1], reverse=True)
        print(f"  Best:  {sorted_pairs[0][0]} AUROC = {sorted_pairs[0][1]:.4f}")
        for pair_str, auc in sorted_pairs[1:6]:
            print(f"         {pair_str} AUROC = {auc:.4f}")
        best_pair_str, best_pair_auroc = sorted_pairs[0]
    else:
        best_pair_str = "(none)"
        best_pair_auroc = float("nan")
        print("  No valid pairs found")

    # ── Comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Summary: TriviaQA Single-Token Feature AUROC")
    print("=" * 72)
    print(f"  {'Feature':<25} {'Best/AUROC':<25} {'L27/Default':<15}")
    print(f"  {'-' * 25} {'-' * 25} {'-' * 15}")
    print(f"  {'max_p':<25} L{best_max_p_layer} = {best_max_p_auroc:<22.4f} "
          f"L27 = {max_p_aurocs.get(27, float('nan')):.4f}")
    print(f"  {'entropy':<25} L{best_entropy_layer} = {best_entropy_auroc:<21.4f} "
          f"L27 = {entropy_aurocs.get(27, float('nan')):.4f}")
    print(f"  {'d2_js (L11/L27)':<25} {d2_js_auroc:<24.4f} --")
    if all_pair_aurocs:
        print(f"  {'d2_js (best pair)':<25} {best_pair_str} = {best_pair_auroc:.4f}")

    # ── HellaSwag comparison ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Cross-task comparison: TriviaQA vs HellaSwag (Phase 4, 1.7B)")
    print("-" * 72)
    # Known HellaSwag results from Phase 4
    hellaswag_baselines = {
        "max_p (best)": 0.85,    # Phase 4 D2 result
        "entropy (best)": 0.65,  # approximate from Phase 2 D1
        "d2_js (L11/L27)": 0.82, # approximate from Phase 4 D2
    }
    print(f"  {'Feature':<25} {'HellaSwag':<12} {'TriviaQA':<12} {'Δ':<10}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(f"  {'max_p':<25} {hellaswag_baselines['max_p (best)']:<12.4f} "
          f"{best_max_p_auroc:<12.4f} {best_max_p_auroc - hellaswag_baselines['max_p (best)']:+.4f}")
    print(f"  {'entropy':<25} {hellaswag_baselines['entropy (best)']:<12.4f} "
          f"{best_entropy_auroc:<12.4f} {best_entropy_auroc - hellaswag_baselines['entropy (best)']:+.4f}")
    print(f"  {'d2_js (L11/L27)':<25} {hellaswag_baselines['d2_js (L11/L27)']:<12.4f} "
          f"{d2_js_auroc:<12.4f} {d2_js_auroc - hellaswag_baselines['d2_js (L11/L27)']:+.4f}")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "config": config,
        "n_valid_samples": len(valid_indices),
        "n_correct": int(labels_valid.sum()),
        "n_incorrect": int(len(labels_valid) - labels_valid.sum()),
        "auroc": {
            "max_p_per_layer": {str(li): auc for li, auc in max_p_aurocs.items()},
            "max_p_best_layer": best_max_p_layer,
            "max_p_best_auroc": best_max_p_auroc,
            "max_p_L27": max_p_aurocs.get(27, float("nan")),
            "entropy_per_layer": {str(li): auc for li, auc in entropy_aurocs.items()},
            "entropy_best_layer": best_entropy_layer,
            "entropy_best_auroc": best_entropy_auroc,
            "entropy_L27": entropy_aurocs.get(27, float("nan")),
            "d2_js_L11_L27": d2_js_auroc,
            "d2_js_best_pair": best_pair_str,
            "d2_js_best_pair_auroc": best_pair_auroc,
            "d2_js_top5_pairs": [
                {"pair": p, "auroc": a}
                for p, a in sorted(all_pair_aurocs.items(), key=lambda x: x[1], reverse=True)[:5]
            ] if all_pair_aurocs else [],
        },
    }

    output_file = output_path / "single_token_auroc.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


def _evaluate_all_pairs(
    samples: list[dict],
    labels: np.ndarray,
    n_layers: int,
) -> dict:
    """Compute AUROC for each layer pair from saved all-pair JS values.

    Reads per-sample last_token_js_all_pairs dicts and evaluates per-pair AUROC.
    """
    # Collect all pair keys
    pair_keys = None
    for s in samples:
        if s.get("last_token_js_all_pairs"):
            pair_keys = set(s["last_token_js_all_pairs"].keys())
            break

    if pair_keys is None:
        return {}

    pair_aurocs = {}
    for pair_key in sorted(pair_keys):
        js_values = []
        valid_labels = []
        for s, lab in zip(samples, labels):
            js_dict = s.get("last_token_js_all_pairs", {})
            if pair_key in js_dict:
                js_value = js_dict[pair_key]
                if np.isfinite(js_value):
                    js_values.append(js_value)
                    valid_labels.append(lab)

        if len(js_values) < 2:
            continue
        js_values = np.array(js_values, dtype=np.float64)
        valid_labels = np.array(valid_labels, dtype=np.int32)
        if valid_labels.sum() == 0 or valid_labels.sum() == len(valid_labels):
            continue

        try:
            auc = roc_auc_score(valid_labels, js_values)
            pair_aurocs[pair_key] = auc
        except ValueError:
            pass

    return pair_aurocs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.2: Single-token feature AUROC on TriviaQA"
    )
    parser.add_argument(
        "--input_file", type=str, default="outputs/triviaqa_features.json"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--js_early", type=int, default=11)
    parser.add_argument("--js_late", type=int, default=27)
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        js_early=args.js_early,
        js_late=args.js_late,
    )
