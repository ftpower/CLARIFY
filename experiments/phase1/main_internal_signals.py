"""Explore internal hidden-state signals for hallucination detection.

Extracts per-layer signals from hidden states and benchmarks each against AUROC.
No calibration phase — pure signal exploration.

Usage:
    python main_internal_signals.py --n_samples 200 --dataset hellaswag
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_triviaqa,
    load_squad,
    load_hellaswag,
    format_prompt,
    check_correct,
)
from src.model_utils import load_model, get_per_layer_hidden_states


# ---- Signal extraction functions ----


def compute_signal_residual_norm(hidden_state: torch.Tensor) -> float:
    """L2 norm of residual stream. Direction unknown (no prior)."""
    return float(torch.norm(hidden_state, dim=-1).item())


def compute_signal_logit_entropy(
    hidden_state: torch.Tensor, W_U: torch.Tensor
) -> float:
    """Entropy of predicted token distribution. Prior: higher → more uncertain."""
    logits = hidden_state @ W_U
    probs = torch.softmax(logits, dim=-1)
    entropy = torch.special.entr(probs).sum(dim=-1)
    return float(entropy.item())


def compute_signal_max_prob(hidden_state: torch.Tensor, W_U: torch.Tensor) -> float:
    """Maximum softmax probability. Prior: higher → more confident → correct."""
    logits = hidden_state @ W_U
    probs = torch.softmax(logits, dim=-1)
    return float(probs.max(dim=-1).values.item())


def compute_signal_logit_margin(
    hidden_state: torch.Tensor, W_U: torch.Tensor, top_k: int = 2
) -> float:
    """Difference between top-1 and top-k logit. Prior: larger margin → correct."""
    logits = hidden_state @ W_U
    top_values = logits.topk(top_k, dim=-1).values
    return float((top_values[0, 0] - top_values[0, -1]).item())


def compute_signal_logit_variance(
    hidden_state: torch.Tensor, W_U: torch.Tensor
) -> float:
    """Variance of logit values across the vocabulary. Prior: higher → more decisive → correct."""
    logits = hidden_state @ W_U
    return float(logits.var(dim=-1).item())


def compute_signal_prediction_entropy(
    hidden_state: torch.Tensor, W_U: torch.Tensor, top_n: int = 100
) -> float:
    """Entropy over top-N predictions only. Prior: higher → more uncertain."""
    logits = hidden_state @ W_U
    top_logits = logits.topk(top_n, dim=-1).values
    probs = torch.softmax(top_logits, dim=-1)
    entropy = torch.special.entr(probs).sum(dim=-1)
    return float(entropy.item())


# Registry: name -> (function, needs_W_U)
# Prior direction: +1 = higher values → correct, -1 = higher values → incorrect
SIGNALS = {
    "residual_norm": (compute_signal_residual_norm, False, +1),
    "logit_entropy": (compute_signal_logit_entropy, True, -1),
    "max_prob": (compute_signal_max_prob, True, +1),
    "logit_margin": (compute_signal_logit_margin, True, +1),
    "logit_variance": (compute_signal_logit_variance, True, +1),
    "top100_entropy": (compute_signal_prediction_entropy, True, -1),
}


def bootstrap_auroc_ci(scores, labels, n_bootstrap=1000, ci=0.95):
    """Bootstrap 95% confidence interval for AUROC."""
    aurocs = []
    n = len(scores)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            aurocs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            pass
    if len(aurocs) < n_bootstrap * 0.9:
        return np.nan, np.nan
    lo = np.percentile(aurocs, (1 - ci) / 2 * 100)
    hi = np.percentile(aurocs, (1 + ci) / 2 * 100)
    return lo, hi


def compute_auroc_separation(scores, labels):
    """Compute |AUROC - 0.5| as separation metric. Range [0, 0.5]."""
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        return np.nan
    return abs(auroc - 0.5)


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_internal_signals",
    dataset: str = "hellaswag",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {dataset.upper()} ({n_samples} samples)...")
    if dataset == "squad":
        samples = load_squad(n_samples=n_samples)
    elif dataset == "hellaswag":
        samples = load_hellaswag(n_samples=n_samples)
    else:
        samples = load_triviaqa(n_samples=n_samples)

    # --- Load model ---
    print("Loading Qwen3-1.7B-Instruct...")
    model = load_model(device=device)
    W_U = model.unembed.W_U
    n_layers = model.cfg.n_layers
    n_total_layers = n_layers + 1

    # --- Collect signals per layer ---
    print(
        f"Phase: Extracting {len(SIGNALS)} internal signals across {n_total_layers} layers..."
    )

    signals_data = {}
    for name in SIGNALS:
        signals_data[name] = [
            {"correct": [], "incorrect": []} for _ in range(n_total_layers)
        ]

    correct_count = 0
    incorrect_count = 0

    for sample in tqdm(samples, desc="Samples"):
        question = sample["question"]
        answers = sample["answers"]
        context = sample["context"]
        prompt = format_prompt(question, context, dataset=dataset)

        hidden_states, _, gen_id, gen_text = get_per_layer_hidden_states(model, prompt)
        is_correct = check_correct(gen_text.strip(), answers, dataset=dataset)

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        bucket = "correct" if is_correct else "incorrect"

        for layer_idx, h in enumerate(hidden_states):
            for name, (fn, needs_W_U, _prior) in SIGNALS.items():
                if needs_W_U:
                    val = fn(h.to(device), W_U)
                else:
                    val = fn(h.to(device))
                signals_data[name][layer_idx][bucket].append(val)

    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
    print(f"Accuracy: {correct_count / n_samples:.1%}")

    # --- Compute AUROC per signal per layer (no auto-flip; use fixed prior direction) ---
    print("\n" + "=" * 80)
    print("Per-signal, per-layer AUROC (separation from 0.5; with 95% bootstrap CI)")
    print("=" * 80)

    results = {}
    for name, (_fn, _needs_W_U, prior) in SIGNALS.items():
        best_sep = 0
        best_layer = -1
        best_raw_auc = np.nan
        best_ci = (np.nan, np.nan)
        layer_results = []

        for layer_idx in range(n_total_layers):
            conf_c = np.array(signals_data[name][layer_idx]["correct"])
            conf_i = np.array(signals_data[name][layer_idx]["incorrect"])

            if len(conf_c) < 2 or len(conf_i) < 2:
                layer_results.append(
                    {
                        "layer": layer_idx,
                        "auroc": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                    }
                )
                continue

            scores = np.concatenate([conf_c, conf_i])
            # Apply prior direction: if prior says higher→incorrect, negate scores
            if prior == -1:
                scores = -scores
            labels = np.concatenate([np.ones(len(conf_c)), np.zeros(len(conf_i))])
            try:
                raw_auc = roc_auc_score(labels, scores)
            except ValueError:
                raw_auc = np.nan

            sep = abs(raw_auc - 0.5) if not np.isnan(raw_auc) else 0
            ci_lo, ci_hi = (
                bootstrap_auroc_ci(scores, labels)
                if not np.isnan(raw_auc)
                else (np.nan, np.nan)
            )

            if sep > best_sep:
                best_sep = sep
                best_layer = layer_idx
                best_raw_auc = raw_auc
                best_ci = (ci_lo, ci_hi)

            layer_results.append(
                {"layer": layer_idx, "auroc": raw_auc, "ci_lo": ci_lo, "ci_hi": ci_hi}
            )

        ci_str = (
            f"[{best_ci[0]:.3f}, {best_ci[1]:.3f}]"
            if not np.isnan(best_ci[0])
            else "[nan]"
        )
        results[name] = {
            "best_auroc": best_raw_auc,
            "best_layer": best_layer,
            "best_ci": list(best_ci),
            "layers": layer_results,
        }
        print(f"{name:>20s}: AUROC = {best_raw_auc:.4f} @ L{best_layer}  CI={ci_str}")

    # --- Comparison ---
    print("\n--- Comparison ---")
    baseline = results.get("max_prob", {})
    baseline_auc = baseline.get("best_auroc", 0.5)
    print(f"{'max_prob (baseline)':>20s}: AUROC = {baseline_auc:.4f}")
    for name, info in results.items():
        if name == "max_prob":
            continue
        delta = info["best_auroc"] - baseline_auc
        ci = info.get("best_ci", [np.nan, np.nan])
        marker = "BETTER" if delta > 0.02 else ("WORSE" if delta < -0.02 else "same")
        print(
            f"{name:>20s}: {info['best_auroc']:.4f} (Δ={delta:+.4f}) "
            f"CI=[{ci[0]:.3f}, {ci[1]:.3f}] {marker}"
        )

    # --- Per-layer table ---
    print("\n" + "=" * 80)
    print("Per-layer AUROC for top 3 signals")
    print("=" * 80)
    top3 = sorted(results.items(), key=lambda x: x[1]["best_auroc"], reverse=True)[:3]
    header = f"{'Layer':>6}"
    for name, _ in top3:
        header += f" {name:>18}"
    print(header)
    print("-" * len(header))
    for layer_idx in range(n_total_layers):
        row = f"{layer_idx:>6}"
        for name, info in top3:
            lr = info["layers"][layer_idx]
            val = lr["auroc"]
            row += f" {val:18.4f}" if not np.isnan(val) else f" {'nan':>18}"
        print(row)

    # --- Save ---
    output = {
        "n_samples": n_samples,
        "n_correct": correct_count,
        "n_incorrect": incorrect_count,
        "accuracy": correct_count / n_samples,
        "signals": {
            name: {
                "best_auroc": info["best_auroc"],
                "best_layer": info["best_layer"],
                "best_ci_95": info["best_ci"],
            }
            for name, info in results.items()
        },
        "per_layer": {name: info["layers"] for name, info in results.items()},
    }
    results_file = output_path / "internal_signals_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_internal_signals")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hellaswag",
        choices=["triviaqa", "squad", "hellaswag"],
    )
    args = parser.parse_args()
    main(args.n_samples, args.device, args.output_dir, args.dataset)
