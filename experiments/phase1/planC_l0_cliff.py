"""Plan C: Precise L0 noise cliff location.

The n=100 experiment showed accuracy collapses from 0.50→0.13 between σ=0.01→0.05
at the embedding layer. This script maps the exact threshold with fine-grained sampling.

Usage:
    python planC_l0_cliff.py --n_samples 200
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

from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.model_utils import load_model, make_noise_hook


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_planC",
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

    # --- Load model ---
    print("Loading Qwen/Qwen3-1.7B...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")

    # L0 = embedding output = blocks.0.hook_resid_pre
    hook_point = "blocks.0.hook_resid_pre"

    # Fine-grained σ around the cliff region
    sigmas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
              0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]

    print(f"Target: L0 (embedding) → {hook_point}")
    print(f"σ levels: {sigmas}")

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
        idx = len(baseline_correct)
        baseline_correct.append(
            check_correct(gen_text, samples[idx]["answers"], dataset="hellaswag")
        )

    baseline_acc = sum(baseline_correct) / n_samples
    print(f"Baseline accuracy: {baseline_acc:.4f} ({sum(baseline_correct)}/{n_samples})")

    # --- Noise sweep ---
    print(f"\n{'='*50} L0 Noise Cliff Sweep {'='*50}")
    results = []

    for sigma in sigmas:
        noise_hook = make_noise_hook(sigma)
        hooks = [(hook_point, noise_hook)]
        per_sample = []

        for prompt in tqdm(prompts, desc=f"σ={sigma:.4f}", leave=False):
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

        correct = [check_correct(p, s["answers"], dataset="hellaswag")
                   for p, s in zip(per_sample, samples)]
        acc = sum(correct) / n_samples
        delta = baseline_acc - acc

        results.append({
            "sigma": sigma,
            "accuracy": acc,
            "delta_accuracy": delta,
            "n_correct": sum(correct),
            "n_incorrect": n_samples - sum(correct),
        })
        print(f"  σ={sigma:.4f}: acc={acc:.4f} (Δ={delta:+.4f}), "
              f"correct={sum(correct)}/{n_samples}")

        # Early stop: if accuracy already at 0, no need for larger sigmas
        if acc == 0.0 and sigma >= 0.1:
            print("  → accuracy hit floor, stopping sweep")
            break

        gc.collect()
        torch.cuda.empty_cache()

    # --- Find cliff midpoint ---
    print(f"\n{'='*50} Cliff Analysis {'='*50}")
    accs = np.array([r["accuracy"] for r in results])
    sigmas_arr = np.array([r["sigma"] for r in results])

    # Find σ where accuracy crosses 50% of baseline (half-max)
    half_max = baseline_acc / 2
    transitions = []
    for i in range(1, len(accs)):
        if accs[i - 1] >= half_max and accs[i] <= half_max:
            # Linear interpolation for σ_half
            frac = (half_max - accs[i - 1]) / (accs[i] - accs[i - 1])
            sigma_half = sigmas_arr[i - 1] + frac * (sigmas_arr[i] - sigmas_arr[i - 1])
            transitions.append(float(sigma_half))

    if transitions:
        print(f"Half-max accuracy σ: {transitions[0]:.5f}")
        print(f"  (accuracy drops from {accs[0]:.4f} to {half_max:.4f})")

    # Find steepest drop
    diffs = -np.diff(accs)
    steepest_idx = int(np.argmax(diffs))
    print(f"Steepest drop: σ={sigmas_arr[steepest_idx]:.4f}→{sigmas_arr[steepest_idx + 1]:.4f}, "
          f"acc {accs[steepest_idx]:.4f}→{accs[steepest_idx + 1]:.4f} (Δ={diffs[steepest_idx]:.4f})")

    # --- Save ---
    output = {
        "config": {
            "n_samples": n_samples,
            "model_id": "Qwen/Qwen3-1.7B",
            "hook_point": hook_point,
            "sigmas": sigmas[:len(results)],
            "baseline_accuracy": baseline_acc,
            "seed": seed,
        },
        "analysis": {
            "half_max_sigma": transitions[0] if transitions else None,
            "steepest_drop_from": float(sigmas_arr[steepest_idx]),
            "steepest_drop_to": float(sigmas_arr[steepest_idx + 1]),
            "steepest_drop_magnitude": float(diffs[steepest_idx]),
        },
        "results": results,
    }

    results_file = output_path / "planC_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Full range
        ax1.plot(sigmas_arr, accs, "o-", color="#e74c3c", markersize=6)
        ax1.axhline(y=baseline_acc, color="black", linestyle="--", alpha=0.7,
                    label=f"baseline ({baseline_acc:.3f})")
        ax1.axhline(y=0.25, color="gray", linestyle=":", alpha=0.5, label="chance")
        if transitions:
            ax1.axvline(x=transitions[0], color="red", linestyle="--", alpha=0.5,
                        label=f"σ_half={transitions[0]:.4f}")
        ax1.set_xlabel("Gaussian noise std (σ)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("L0 (Embedding) Noise Cliff — Full Range")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Zoom into cliff region: σ ∈ [0, 0.06]
        zoom_mask = sigmas_arr <= 0.06
        ax2.plot(sigmas_arr[zoom_mask], accs[zoom_mask], "o-", color="#e74c3c",
                 markersize=8, linewidth=2)
        ax2.axhline(y=baseline_acc, color="black", linestyle="--", alpha=0.7,
                    label=f"baseline ({baseline_acc:.3f})")
        if transitions:
            ax2.axvline(x=transitions[0], color="red", linestyle="--", alpha=0.5,
                        label=f"σ_half={transitions[0]:.4f}")
        ax2.set_xlabel("Gaussian noise std (σ)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("L0 (Embedding) Noise Cliff — Zoom [0, 0.06]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Plan C: L0 Embedding Noise Cliff Mapping\n"
                     f"Qwen3-1.7B, HellaSwag n={n_samples}",
                     fontsize=13)
        fig.tight_layout()

        plot_file = output_path / "planC_l0_cliff.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_planC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
