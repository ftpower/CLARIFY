"""Direction 1: Logit Lens Entropy as Hallucination Diagnostic.

Usage:
    python main_diagnosis.py --n_samples 200
    python main_diagnosis.py --n_samples 50 --model Qwen/Qwen3-8B
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, load_triviaqa, load_squad, format_prompt, check_correct
from src.hidden_state import extract_hidden_states, generate_answer
from src.entropy import (
    compute_logit_lens_entropy,
    compute_per_layer_auroc,
    compute_collapse_stats,
)


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    dataset: str = "hellaswag",
    output_dir: str = "outputs",
    seed: int = 42,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading {dataset.upper()} ({n_samples} samples)...")
    if dataset == "hellaswag":
        samples = load_hellaswag(n_samples=n_samples, seed=seed)
    elif dataset == "squad":
        samples = load_squad(n_samples=n_samples, seed=seed)
    else:
        samples = load_triviaqa(n_samples=n_samples, seed=seed)

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    n_layers = model.cfg.n_layers
    n_total = n_layers + 1  # embed + blocks

    # ── Per-sample extraction ──────────────────────────────────────────
    print(f"Processing {n_samples} samples...")
    per_sample_results = []
    correct_count = 0

    for sample in tqdm(samples, desc="Samples"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset=dataset
        )

        # Extract per-layer hidden states + final logits
        hidden_states, logits_final, gen_id, gen_text = extract_hidden_states(
            model, prompt
        )

        # Multi-token answer for QA datasets
        if dataset in ("triviaqa", "squad"):
            answer_text = generate_answer(model, prompt, max_new_tokens=20)
        else:
            answer_text = gen_text.strip()

        is_correct = check_correct(answer_text, sample["answers"], dataset=dataset)
        if is_correct:
            correct_count += 1

        # Logit lens entropy
        metrics = compute_logit_lens_entropy(hidden_states, W_U, b_U, temperature=1.0)

        per_sample_results.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "generated_text": answer_text,
            "is_correct": is_correct,
            "entropy": metrics["entropy"],
            "max_prob": metrics["max_prob"],
            "top5_mass": metrics["top5_mass"],
            "top1_ids": metrics["top1_ids"],
        })

    accuracy = correct_count / n_samples
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{n_samples})")

    # ── Per-layer AUROC ────────────────────────────────────────────────
    print("\n--- Per-layer AUROC (entropy as detection score) ---")
    entropy_auroc = compute_per_layer_auroc(per_sample_results, n_total, metric="entropy")
    maxprob_auroc = compute_per_layer_auroc(per_sample_results, n_total, metric="max_prob")
    top5_auroc = compute_per_layer_auroc(per_sample_results, n_total, metric="top5_mass")

    print(f"{'Layer':>6} {'AUROC(H)':>10} {'AUROC(max_p)':>12} {'AUROC(top5)':>12}")
    print("-" * 44)
    for li in range(n_total):
        h_str = f"{entropy_auroc['aurocs'][li]:.4f}" if not np.isnan(entropy_auroc["aurocs"][li]) else "   nan"
        m_str = f"{maxprob_auroc['aurocs'][li]:.4f}" if not np.isnan(maxprob_auroc["aurocs"][li]) else "   nan"
        t_str = f"{top5_auroc['aurocs'][li]:.4f}" if not np.isnan(top5_auroc["aurocs"][li]) else "   nan"
        print(f"{li:>6} {h_str:>10} {m_str:>12} {t_str:>12}")

    print(f"\nBest AUROC(H): L{entropy_auroc['best_layer']} = {entropy_auroc['best_auroc']:.4f}")
    print(f"Best AUROC(max_p): L{maxprob_auroc['best_layer']} = {maxprob_auroc['best_auroc']:.4f}")
    print(f"Best AUROC(top5): L{top5_auroc['best_layer']} = {top5_auroc['best_auroc']:.4f}")

    # ── Entropy collapse analysis ──────────────────────────────────────
    collapse = compute_collapse_stats(per_sample_results, n_total)

    print(f"\n--- Entropy Collapse Layer ℓ* = argmin H(ℓ) ---")
    if collapse["collapse_layers_correct"]:
        mean_c = np.mean(collapse["collapse_layers_correct"])
        print(f"Correct:   mean ℓ* = {mean_c:.2f}")
    if collapse["collapse_layers_incorrect"]:
        mean_i = np.mean(collapse["collapse_layers_incorrect"])
        print(f"Incorrect: mean ℓ* = {mean_i:.2f}")

    # Wilcoxon test for ℓ* difference
    if collapse["collapse_layers_correct"] and collapse["collapse_layers_incorrect"]:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(
            collapse["collapse_layers_correct"],
            collapse["collapse_layers_incorrect"],
            alternative="two-sided",
        )
        print(f"Mann-Whitney U test: p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples,
            "model_id": model_id,
            "dataset": dataset,
            "n_layers": n_layers,
            "n_total_layers": n_total,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "incorrect_count": n_samples - correct_count,
        },
        "entropy_auroc": {
            "aurocs": entropy_auroc["aurocs"],
            "best_layer": entropy_auroc["best_layer"],
            "best_auroc": entropy_auroc["best_auroc"],
        },
        "maxprob_auroc": {
            "aurocs": maxprob_auroc["aurocs"],
            "best_layer": maxprob_auroc["best_layer"],
            "best_auroc": maxprob_auroc["best_auroc"],
        },
        "top5_auroc": {
            "aurocs": top5_auroc["aurocs"],
            "best_layer": top5_auroc["best_layer"],
            "best_auroc": top5_auroc["best_auroc"],
        },
        "collapse": {
            "collapse_layers_correct": collapse["collapse_layers_correct"],
            "collapse_layers_incorrect": collapse["collapse_layers_incorrect"],
            "mean_entropy_correct": collapse["mean_entropy_correct"],
            "std_entropy_correct": collapse["std_entropy_correct"],
            "mean_entropy_incorrect": collapse["mean_entropy_incorrect"],
            "std_entropy_incorrect": collapse["std_entropy_incorrect"],
        },
        "per_sample": [
            {
                "question": r["question"],
                "answers": r["answers"],
                "generated_text": r["generated_text"],
                "is_correct": r["is_correct"],
                "entropy": r["entropy"],
                "max_prob": r["max_prob"],
                "top5_mass": r["top5_mass"],
            }
            for r in per_sample_results
        ],
    }

    results_file = output_path / "entropy_diagnosis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ── Plots ──────────────────────────────────────────────────────────
    try:
        _plot_results(
            collapse, entropy_auroc, maxprob_auroc, top5_auroc,
            model_id, dataset, accuracy, n_total, output_path,
        )
    except Exception as e:
        print(f"Plot failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()


def _plot_results(
    collapse, entropy_auroc, maxprob_auroc, top5_auroc,
    model_id, dataset, accuracy, n_total, output_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_total))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) Entropy trajectory ──
    ax = axes[0, 0]
    mean_c = collapse["mean_entropy_correct"]
    std_c = collapse["std_entropy_correct"]
    mean_i = collapse["mean_entropy_incorrect"]
    std_i = collapse["std_entropy_incorrect"]

    ax.plot(layers, mean_c, "b-", label="Correct", linewidth=2)
    ax.fill_between(layers,
                    [m - s for m, s in zip(mean_c, std_c)],
                    [m + s for m, s in zip(mean_c, std_c)],
                    alpha=0.15, color="blue")
    ax.plot(layers, mean_i, "r-", label="Incorrect", linewidth=2)
    ax.fill_between(layers,
                    [m - s for m, s in zip(mean_i, std_i)],
                    [m + s for m, s in zip(mean_i, std_i)],
                    alpha=0.15, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Logit Lens Entropy H(ℓ)")
    ax.set_title("Per-Layer Entropy Trajectory (mean ± 1σ)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (b) AUROC comparison ──
    ax = axes[0, 1]
    ax.plot(layers, entropy_auroc["aurocs"], "g-o", label="AUROC(H)", markersize=4)
    ax.plot(layers, maxprob_auroc["aurocs"], "m-s", label="AUROC(max_p)", markersize=4)
    ax.plot(layers, top5_auroc["aurocs"], "c-^", label="AUROC(top5)", markersize=4)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Layer AUROC (Entropy vs MaxProb vs Top5)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (c) Collapse layer distribution ──
    ax = axes[1, 0]
    bins = np.arange(0, n_total + 1) - 0.5
    if collapse["collapse_layers_correct"]:
        ax.hist(collapse["collapse_layers_correct"], bins=bins, alpha=0.6,
                color="blue", label="Correct", density=True)
    if collapse["collapse_layers_incorrect"]:
        ax.hist(collapse["collapse_layers_incorrect"], bins=bins, alpha=0.6,
                color="red", label="Incorrect", density=True)
    ax.set_xlabel("Layer ℓ* (argmin H)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy Collapse Layer Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (d) Summary text ──
    ax = axes[1, 1]
    ax.axis("off")
    best_h = entropy_auroc["best_auroc"]
    best_mp = maxprob_auroc["best_auroc"]
    summary = (
        f"Model: {model_id}\n"
        f"Dataset: {dataset.upper()}\n"
        f"Samples: {len(collapse['collapse_layers_correct']) + len(collapse['collapse_layers_incorrect'])}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Best AUROC(H): L{entropy_auroc['best_layer']} = {best_h:.4f}\n"
        f"Best AUROC(max_p): L{maxprob_auroc['best_layer']} = {best_mp:.4f}\n"
        f"Best AUROC(top5): L{top5_auroc['best_layer']} = {top5_auroc['best_auroc']:.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace")

    fig.suptitle(f"Logit Lens Entropy Diagnosis — {model_id} {dataset.upper()}",
                 fontsize=14)
    fig.tight_layout()

    plot_file = output_path / "entropy_diagnosis.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="hellaswag",
                        choices=["hellaswag", "triviaqa", "squad"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_samples, args.device, args.model, args.dataset, args.output_dir, args.seed)
