"""Plan A: Fine-grained α scan around L14 to verify accuracy improvement.

Verifies the L14 α=0.7 accuracy boost observed in the 1.7B n=100 experiment
with larger sample size and neighboring layers as controls.

Usage:
    python planA_l14_scaling.py --n_samples 300
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
from scipy.stats import binomtest
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.model_utils import load_model, make_scale_hook


def layer_to_hook(idx: int, n_layers: int) -> str:
    """Map layer index to hook point name."""
    if idx == 0:
        return "blocks.0.hook_resid_pre"
    if idx <= n_layers - 1:
        return f"blocks.{idx - 1}.hook_resid_post"
    return f"blocks.{n_layers - 1}.hook_resid_post"


def run_scale_sweep(
    model, samples, prompts, layer_indices, alphas, n_layers, baseline_correct
):
    """Run scaling sweep across layers and alphas.

    Returns:
        results: dict[layer_label] = list of {alpha, accuracy, delta, per_sample_correct}
    """
    results = {}
    total = len(samples)
    baseline_acc = sum(baseline_correct) / total

    for li in layer_indices:
        hook_point = layer_to_hook(li, n_layers)
        label = f"L{li}" if li > 0 else "L0"
        print(f"\n--- Scale @ {label} ({hook_point}) ---")
        layer_results = []

        for alpha in alphas:
            scale_fn = make_scale_hook(alpha)
            hooks = [(hook_point, scale_fn)]
            per_sample = []

            for prompt in tqdm(prompts, desc=f"α={alpha:.3f}", leave=False):
                tokens = model.to_tokens(prompt, prepend_bos=True)
                if tokens.shape[1] > 1024:
                    tokens = tokens[:, :1024]
                last_pos = tokens.shape[1] - 1

                with model.hooks(fwd_hooks=hooks):
                    with torch.no_grad():
                        logits = model(tokens)

                gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
                gen_text = model.tokenizer.decode(gen_id).strip()
                per_sample.append(gen_text)

            # Score
            correct = [check_correct(p, s["answers"], dataset="hellaswag")
                       for p, s in zip(per_sample, samples)]
            acc = sum(correct) / total
            delta = baseline_acc - acc  # positive = degradation

            layer_results.append({
                "alpha": alpha,
                "accuracy": acc,
                "delta_accuracy": delta,
                "n_correct": sum(correct),
                "n_incorrect": total - sum(correct),
                "per_sample_correct": correct,
            })
            print(f"  α={alpha:.3f}: acc={acc:.4f} (Δ={delta:+.4f}), "
                  f"correct={sum(correct)}/{total}")

            gc.collect()
            torch.cuda.empty_cache()

        results[label] = layer_results

    return results


def mcnemar_paired(baseline_correct, intervention_correct):
    """McNemar test for paired binary data.

    Compares baseline vs intervention on the same samples.
    Returns p-value (two-sided).
    """
    n = len(baseline_correct)
    # Count discordant pairs
    b = 0  # baseline correct, intervention incorrect
    c = 0  # baseline incorrect, intervention correct
    for bc, ic in zip(baseline_correct, intervention_correct):
        if bc and not ic:
            b += 1
        elif not bc and ic:
            c += 1

    if b + c == 0:
        return 1.0

    # Binomial test: under H0, P(b out of b+c) = 0.5
    result = binomtest(b, b + c, p=0.5, alternative="two-sided")
    return result.pvalue


def main(
    n_samples: int = 300,
    device: str = "cuda",
    output_dir: str = "outputs_planA",
    seed: int = 42,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)
    prompts = [format_prompt(s["question"], s["context"], dataset="hellaswag")
               for s in samples]
    answers_list = [s["answers"] for s in samples]

    # --- Load model ---
    print("Loading Qwen/Qwen3-1.7B...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    n_layers = model.cfg.n_layers

    # --- Layers to test ---
    layer_indices = [11, 12, 13, 14, 15, 16]  # L11=detection ref, L14=target
    alphas = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    print(f"Model: Qwen3-1.7B, n_layers={n_layers}")
    print(f"Layers: {[f'L{li}' for li in layer_indices]}")
    print(f"Alphas: {alphas}")

    # --- Baseline ---
    print(f"\n{'='*50} Baseline {'='*50}")
    baseline_correct = []
    for prompt in tqdm(prompts, desc="Baseline"):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model(tokens)
        gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
        gen_text = model.tokenizer.decode(gen_id).strip()
        baseline_correct.append(
            check_correct(gen_text, answers_list[len(baseline_correct)],
                          dataset="hellaswag")
        )

    baseline_acc = sum(baseline_correct) / n_samples
    print(f"Baseline accuracy: {baseline_acc:.4f} ({sum(baseline_correct)}/{n_samples})")

    # --- Scale sweep ---
    print(f"\n{'='*50} Scale Sweep {'='*50}")
    results = run_scale_sweep(
        model, samples, prompts, layer_indices, alphas, n_layers, baseline_correct
    )

    # --- Statistical analysis ---
    print(f"\n{'='*50} Statistical Tests {'='*50}")
    stats = {}

    for label, layer_results in results.items():
        layer_stats = []
        print(f"\n{label} vs Baseline (McNemar test):")
        for r in layer_results:
            p_val = mcnemar_paired(baseline_correct, r["per_sample_correct"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            direction = "BETTER" if r["accuracy"] > baseline_acc else "WORSE"
            acc_improvement = r["accuracy"] > baseline_acc

            print(f"  α={r['alpha']:.3f}: acc={r['accuracy']:.4f} "
                  f"({'↑' if acc_improvement else '↓'}{abs(r['delta_accuracy']):.4f}) "
                  f"p={p_val:.4f} {sig} ({direction})")

            layer_stats.append({
                "alpha": r["alpha"],
                "accuracy": r["accuracy"],
                "delta_accuracy": r["delta_accuracy"],
                "mcnemar_p": float(p_val),
                "significant_05": bool(p_val < 0.05),
                "direction": direction,
            })
        stats[label] = layer_stats

    # --- Save ---
    output = {
        "config": {
            "n_samples": n_samples,
            "model_id": "Qwen/Qwen3-1.7B",
            "n_layers": n_layers,
            "layers": [f"L{li}" for li in layer_indices],
            "alphas": alphas,
            "baseline_accuracy": baseline_acc,
            "baseline_n_correct": sum(baseline_correct),
            "seed": seed,
        },
        "results": {
            label: [{k: v for k, v in r.items() if k != "per_sample_correct"}
                    for r in layer_results]
            for label, layer_results in results.items()
        },
        "statistics": stats,
    }

    results_file = output_path / "planA_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))

        # Dose-response curves
        for idx, (label, layer_results) in enumerate(results.items()):
            a_vals = [r["alpha"] for r in layer_results]
            accs = [r["accuracy"] for r in layer_results]
            ax1.plot(a_vals, accs, "o-", color=colors[idx], label=label, markersize=6)

        ax1.axhline(y=baseline_acc, color="black", linestyle="--", alpha=0.7,
                    label=f"baseline ({baseline_acc:.3f})")
        ax1.axhline(y=0.25, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax1.set_xlabel("Activation scale (α)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("L14 Region — Scaling Dose-Response")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Significance heatmap
        p_matrix = []
        for label in sorted(results.keys(), key=lambda x: int(x[1:])):
            p_vals = []
            for r in stats[label]:
                p_vals.append(r["mcnemar_p"])
            p_matrix.append(p_vals)

        p_array = np.array(p_matrix)
        im = ax2.imshow(-np.log10(np.clip(p_array, 1e-15, 1)), aspect="auto",
                         cmap="RdYlGn", vmin=0, vmax=3)
        ax2.set_xticks(range(len(alphas)))
        ax2.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45)
        ax2.set_yticks(range(len(p_matrix)))
        ax2.set_yticklabels(sorted(results.keys(), key=lambda x: int(x[1:])))
        ax2.set_xlabel("α")
        ax2.set_title("-log10(p) — green = significant deviation from baseline")
        plt.colorbar(im, ax=ax2, label="-log10(p)")

        # Annotate significant cells
        for i in range(len(p_matrix)):
            for j in range(len(alphas)):
                if p_matrix[i][j] < 0.05:
                    marker = "***" if p_matrix[i][j] < 0.001 else \
                             "**" if p_matrix[i][j] < 0.01 else "*"
                    # White text for dark cells (high -log10(p))
                    text_color = "white" if -np.log10(p_matrix[i][j]) > 1.5 else "black"
                    ax2.text(j, i, marker, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_color)

        fig.suptitle("Plan A: L14 Scaling Enhancement Verification\n"
                     f"Qwen3-1.7B, HellaSwag n={n_samples}",
                     fontsize=13)
        fig.tight_layout()

        plot_file = output_path / "planA_dose_response.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_planA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
