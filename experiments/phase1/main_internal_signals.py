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
import torch.nn.functional as F
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


def compute_signal_residual_norm(hidden_state: torch.Tensor) -> float:
    """L2 norm of residual stream."""
    return float(torch.norm(hidden_state, dim=-1).item())


def compute_signal_logit_entropy(
    hidden_state: torch.Tensor, W_U: torch.Tensor
) -> float:
    """Entropy of the predicted token distribution: H(p) = -sum(p * log(p))."""
    logits = hidden_state @ W_U  # [1, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return float(entropy.item())


def compute_signal_max_prob(hidden_state: torch.Tensor, W_U: torch.Tensor) -> float:
    """Maximum softmax probability (baseline confidence signal)."""
    logits = hidden_state @ W_U
    probs = torch.softmax(logits, dim=-1)
    return float(probs.max(dim=-1).values.item())


def compute_signal_logit_margin(
    hidden_state: torch.Tensor, W_U: torch.Tensor, top_k: int = 2
) -> float:
    """Difference between top-1 and top-k logit (margin)."""
    logits = hidden_state @ W_U  # [1, vocab_size]
    top_values = logits.topk(top_k, dim=-1).values
    return float((top_values[0, 0] - top_values[0, -1]).item())


def compute_signal_logit_variance(
    hidden_state: torch.Tensor, W_U: torch.Tensor
) -> float:
    """Variance of logit values across the vocabulary."""
    logits = hidden_state @ W_U  # [1, vocab_size]
    return float(logits.var(dim=-1).item())


def compute_signal_prediction_entropy(
    hidden_state: torch.Tensor, W_U: torch.Tensor, top_n: int = 100
) -> float:
    """Entropy over top-N predictions only (excludes long-tail noise)."""
    logits = hidden_state @ W_U  # [1, vocab_size]
    top_logits = logits.topk(top_n, dim=-1).values
    probs = torch.softmax(top_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return float(entropy.item())


# Registry: name -> (function, needs_W_U)
SIGNALS = {
    "residual_norm": (compute_signal_residual_norm, False),
    "logit_entropy": (compute_signal_logit_entropy, True),
    "max_prob": (compute_signal_max_prob, True),
    "logit_margin": (compute_signal_logit_margin, True),
    "logit_variance": (compute_signal_logit_variance, True),
    "top100_entropy": (compute_signal_prediction_entropy, True),
}


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_internal_signals",
    dataset: str = "hellaswag",
    seed: int = 42,
):
    np.random.seed(seed)

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
    W_U = model.unembed.W_U  # [d_model, vocab_size]
    n_layers = model.cfg.n_layers
    n_total_layers = n_layers + 1  # embed + layers

    # --- Collect signals per layer ---
    print(
        f"Phase: Extracting {len(SIGNALS)} internal signals across {n_total_layers} layers..."
    )

    # signals_data[name][layer] = {"correct": [...], "incorrect": [...]}
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
        is_correct = check_correct(gen_text.strip(), answers)

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        bucket = "correct" if is_correct else "incorrect"

        for layer_idx, h in enumerate(hidden_states):
            h = h.to(device)
            for name, (fn, needs_W_U) in SIGNALS.items():
                if needs_W_U:
                    val = fn(h, W_U)
                else:
                    val = fn(h)
                signals_data[name][layer_idx][bucket].append(val)

    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
    print(f"Accuracy: {correct_count / n_samples:.1%}")

    # --- Compute AUROC per signal per layer ---
    print("\n" + "=" * 80)
    print("Per-signal, per-layer AUROC (higher = better, >0.5 = signal works)")
    print("=" * 80)

    results = {}
    for name in SIGNALS:
        best_auroc = 0
        best_layer = -1
        layer_results = []

        for layer_idx in range(n_total_layers):
            conf_c = np.array(signals_data[name][layer_idx]["correct"])
            conf_i = np.array(signals_data[name][layer_idx]["incorrect"])

            if len(conf_c) < 2 or len(conf_i) < 2:
                layer_results.append({"layer": layer_idx, "auroc": np.nan})
                continue

            scores = np.concatenate([conf_c, conf_i])
            labels = np.concatenate([np.ones(len(conf_c)), np.zeros(len(conf_i))])
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                auroc = np.nan

            # For signals where LOW values indicate correctness, flip
            # (entropy, variance should be lower for correct answers)
            if auroc < 0.5 and not np.isnan(auroc):
                auroc_flipped = roc_auc_score(labels, -scores)
                if auroc_flipped > auroc:
                    auroc = auroc_flipped

            if not np.isnan(auroc) and auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer_idx

            layer_results.append({"layer": layer_idx, "auroc": auroc})

        results[name] = {
            "best_auroc": best_auroc,
            "best_layer": best_layer,
            "layers": layer_results,
        }
        print(f"{name:>20s}: best AUROC = {best_auroc:.4f} @ layer {best_layer}")

    # --- Compare with dot-product baseline from softmax confidence ---
    print("\n--- Comparison ---")
    baseline_auroc = results.get("max_prob", {}).get("best_auroc", 0)
    print(f"{'Softmax max_prob (baseline)':>20s}: {baseline_auroc:.4f}")
    for name, info in results.items():
        if name == "max_prob":
            continue
        delta = info["best_auroc"] - baseline_auroc
        marker = (
            "↑ BETTER" if delta > 0.02 else ("↓ WORSE" if delta < -0.02 else "≈ same")
        )
        print(f"{name:>20s}: {info['best_auroc']:.4f} (Δ={delta:+.4f}) {marker}")

    # --- Print per-layer table for top signals ---
    print("\n" + "=" * 80)
    print("Per-layer AUROC for promising signals")
    print("=" * 80)
    top_signals = sorted(
        results.items(), key=lambda x: x[1]["best_auroc"], reverse=True
    )[:4]
    header = f"{'Layer':>6}"
    for name, _ in top_signals:
        header += f" {name:>16}"
    print(header)
    print("-" * len(header))
    for layer_idx in range(n_total_layers):
        row = f"{layer_idx:>6}"
        for name, info in top_signals:
            lr = info["layers"][layer_idx]
            val = lr["auroc"]
            row += f" {val:16.4f}" if not np.isnan(val) else f" {'nan':>16}"
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
