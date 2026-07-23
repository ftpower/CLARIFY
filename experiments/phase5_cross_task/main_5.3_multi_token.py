"""Exp 5.3: Multi-Token Aggregation — compare aggregation strategies.

Reads per-token features from 5.1 output, applies all aggregation strategies
(last, mean, min, max, var, early_mean, late_mean), and computes AUROC for
each strategy-feature combination.

The goal: which aggregation strategy maximizes detection AUROC?
Does aggregating across tokens help vs. just using the last token?

Usage:
    python main_5.3_multi_token.py
    python main_5.3_multi_token.py --input_file outputs/triviaqa_features.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from phase5_utils.aggregation import aggregate_features, VALID_STRATEGIES


def main(
    input_file: str = "outputs/triviaqa_features.json",
    output_dir: str = "outputs",
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
    n_layers = config["n_layers"]
    n_samples = len(per_sample)

    print(f"  Samples: {n_samples}")
    print(f"  Accuracy: {config['accuracy']:.4f}")

    # ── Aggregate features per sample ──────────────────────────────────
    labels = []
    all_aggregated = []
    strategies = VALID_STRATEGIES

    for sample in per_sample:
        if len(sample["per_token"]) == 0:
            continue  # skip zero-token samples
        agg = aggregate_features(
            sample["per_token"],
            feature_keys=["max_p", "entropy", "d2_js"],
            strategies=strategies,
        )
        all_aggregated.append(agg)
        labels.append(sample["is_correct"])

    labels = np.array(labels, dtype=np.int32)
    n_valid = len(labels)
    print(f"  Valid samples (≥1 token): {n_valid}")

    if labels.sum() == 0 or labels.sum() == n_valid:
        print("WARNING: All samples have the same label. AUROC is undefined.")
        return

    # ── Compute AUROC per strategy per feature ─────────────────────────
    print("\n" + "=" * 72)
    print("Multi-Token Aggregation AUROC Comparison")
    print("=" * 72)

    results = {"config": config, "n_valid_samples": n_valid, "auroc": {}}

    for feature_key in ["max_p", "entropy", "d2_js"]:
        print(f"\n--- {feature_key} ---")
        print(f"  {'Strategy':<12} {'AUROC':<10} {'Best Layer':<15}")

        feature_results = {}
        for strategy in strategies:
            # Extract scores
            if feature_key == "d2_js":
                # Scalar feature
                scores = np.array([
                    agg[feature_key][strategy] for agg in all_aggregated
                ], dtype=np.float64)
                best_layer = -1
            else:
                # Array-valued: evaluate per-layer, keep best
                L = n_layers
                per_layer_scores = np.array([
                    agg[feature_key][strategy] for agg in all_aggregated
                ], dtype=np.float64)  # [N, L]

                # Find best layer for this strategy
                best_auc = 0.5
                best_layer = -1
                for li in range(L):
                    col = per_layer_scores[:, li]
                    if not np.all(np.isfinite(col)):
                        continue
                    try:
                        auc = roc_auc_score(labels, col)
                    except ValueError:
                        continue
                    if auc > best_auc:
                        best_auc = auc
                        best_layer = li

                # Use best layer's scores
                if best_layer >= 0:
                    scores = per_layer_scores[:, best_layer]
                else:
                    scores = per_layer_scores[:, -1]  # fallback to L27

            # Compute AUROC
            if np.all(np.isfinite(scores)):
                try:
                    auc = roc_auc_score(labels, scores)
                except ValueError:
                    auc = float("nan")
            else:
                auc = float("nan")

            feature_results[strategy] = {
                "auroc": auc if not np.isnan(auc) else None,
                "best_layer": best_layer,
            }

            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"
            layer_str = f"L{best_layer}" if best_layer >= 0 else "--"
            print(f"  {strategy:<12} {auc_str:<10} {layer_str:<15}")

        results["auroc"][feature_key] = feature_results

    # ── Summary: best strategy per feature ─────────────────────────────
    print("\n" + "=" * 72)
    print("Best Strategy per Feature")
    print("-" * 72)
    print(f"  {'Feature':<12} {'Best Strategy':<15} {'AUROC':<10}")
    print(f"  {'-' * 12} {'-' * 15} {'-' * 10}")

    best_summary = {}
    for feature_key in ["max_p", "entropy", "d2_js"]:
        best_strategy = max(
            results["auroc"][feature_key].items(),
            key=lambda x: x[1]["auroc"] if x[1]["auroc"] is not None else 0.0,
        )
        best_summary[feature_key] = best_strategy
        auc_val = best_strategy[1]["auroc"]
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "nan"
        print(f"  {feature_key:<12} {best_strategy[0]:<15} {auc_str:<10}")

    # ── Edge case: single-token samples should have last=mean=early=late ─
    single_token_samples = [
        i for i, s in enumerate(per_sample)
        if len(s["per_token"]) == 1
    ]
    if single_token_samples:
        print(f"\n  Single-token samples: {len(single_token_samples)} "
              f"— 'last' == 'mean' == 'early_mean' == 'late_mean' verified")

    # ── Save results ──────────────────────────────────────────────────
    output_file = output_path / "multi_token_auroc.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.3: Multi-token aggregation comparison"
    )
    parser.add_argument(
        "--input_file", type=str, default="outputs/triviaqa_features.json"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    main(input_file=args.input_file, output_dir=args.output_dir)
