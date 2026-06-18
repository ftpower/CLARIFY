"""Plan B: Directional noise injection at L11 to locate detection signal subspace.

Tests whether detection-relevant information at L11 is concentrated in a low-dimensional
subspace by comparing accuracy degradation under:
  1. Noise parallel to the correct-answer unembedding direction
  2. Noise parallel to model-predicted token unembedding direction
  3. Noise orthogonal to correct-answer direction (all d-1 dims except u_correct)
  4. Isotropic noise (baseline, same as main intervention experiment)

If (1) degrades accuracy more than (4) at the same σ, the detection signal is
concentrated along the unembedding direction of the correct answer.

Usage:
    python planB_directional.py --n_samples 200
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
from src.model_utils import load_model


LETTER_IDS = None  # populated in main, cached globally for hook fn access


def make_directional_noise_hook(sigma: float, direction: torch.Tensor, mode: str):
    """Hook that adds noise in/orthogonal to a specific direction.

    Args:
        sigma: noise standard deviation
        direction: unit vector [d_model] on GPU
        mode: 'parallel' (only along direction),
              'orthogonal' (only perpendicular to direction),
              'isotropic' (same as make_noise_hook)
    """
    d_vec = direction  # [d_model]

    def hook(activation: torch.Tensor, hook) -> torch.Tensor:
        if mode == "isotropic":
            return activation + torch.randn_like(activation) * sigma

        # activation: [batch, seq, d_model]
        noise_raw = torch.randn_like(activation)  # ~ N(0, 1)

        if mode == "parallel":
            # Project noise onto direction: ⟨noise, d⟩ · d
            # dot product along last dim: [batch, seq]
            proj = (noise_raw * d_vec).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
            noise = proj * d_vec  # [batch, seq, d_model]
        elif mode == "orthogonal":
            # Remove component along direction: noise - ⟨noise, d⟩ · d
            proj = (noise_raw * d_vec).sum(dim=-1, keepdim=True)
            noise = noise_raw - proj * d_vec
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return activation + noise * sigma

    return hook


def get_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else 0


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_planB",
    seed: int = 42,
):
    global LETTER_IDS
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
    n_layers = model.cfg.n_layers
    W_U = model.W_U  # [d_model, vocab_size]
    LETTER_IDS = {
        letter: get_token_id(model.tokenizer, letter)
        for letter in ["A", "B", "C", "D"]
    }

    target_layer = 11  # wAUROC-optimal for HellaSwag 1.7B
    target_hook = f"blocks.{target_layer - 1}.hook_resid_post"
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"Target: L{target_layer} → {target_hook}")
    print(f"Letter token IDs: {LETTER_IDS}")
    print(f"σ levels: {sigmas}")

    # --- Baseline ---
    print(f"\n{'='*50} Baseline {'='*50}")
    baseline_correct = []
    baseline_predictions = []  # store predicted letter per sample

    for prompt in tqdm(prompts, desc="Baseline"):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model(tokens)

        gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
        gen_text = model.tokenizer.decode(gen_id).strip()
        pred_letter = gen_text[0].upper() if gen_text else ""
        baseline_predictions.append(pred_letter)

        idx = len(baseline_correct)
        baseline_correct.append(
            check_correct(gen_text, samples[idx]["answers"], dataset="hellaswag")
        )

    baseline_acc = sum(baseline_correct) / n_samples
    print(f"Baseline accuracy: {baseline_acc:.4f} ({sum(baseline_correct)}/{n_samples})")

    # Pre-compute correct-answer unembedding directions per sample
    correct_directions = []  # unit vector in d_model per sample
    pred_directions = []     # unit vector for model prediction
    for s, pred_letter in zip(samples, baseline_predictions):
        correct_letter = s["answers"][1].strip().upper()
        c_id = LETTER_IDS.get(correct_letter, LETTER_IDS["A"])
        u_c = W_U[:, c_id].float()  # [d_model]
        correct_directions.append(u_c / (u_c.norm() + 1e-8))

        p_id = LETTER_IDS.get(pred_letter, LETTER_IDS["A"])
        u_p = W_U[:, p_id].float()
        pred_directions.append(u_p / (u_p.norm() + 1e-8))

    # --- Directional noise sweep ---
    noise_modes = ["isotropic", "parallel_correct", "parallel_pred", "orthogonal_correct"]
    mode_labels = {
        "isotropic": "Isotropic",
        "parallel_correct": "Parallel (correct ans)",
        "parallel_pred": "Parallel (predicted)",
        "orthogonal_correct": "Orthogonal (correct ans)",
    }
    all_results = {}

    for mode in noise_modes:
        print(f"\n{'='*50} {mode_labels[mode]} {'='*50}")
        mode_results = []

        for sigma in sigmas:
            per_sample_correct = []

            for i, (prompt, sample) in enumerate(
                tqdm(zip(prompts, samples), total=n_samples, desc=f"{mode} σ={sigma}")
            ):
                # Build hook for this sample (direction depends on mode)
                if mode == "isotropic":
                    direction = torch.zeros(W_U.shape[0], device=device)
                elif mode == "parallel_correct":
                    direction = correct_directions[i]
                elif mode == "parallel_pred":
                    direction = pred_directions[i]
                elif mode == "orthogonal_correct":
                    direction = correct_directions[i]
                else:
                    raise ValueError(mode)

                if mode == "isotropic":
                    hook_mode = "isotropic"
                elif mode in ("parallel_correct", "parallel_pred"):
                    hook_mode = "parallel"
                elif mode == "orthogonal_correct":
                    hook_mode = "orthogonal"
                else:
                    raise ValueError(mode)
                noise_hook = make_directional_noise_hook(sigma, direction, hook_mode)
                hooks = [(target_hook, noise_hook)]

                tokens = model.to_tokens(prompt, prepend_bos=True)
                if tokens.shape[1] > 1024:
                    tokens = tokens[:, :1024]
                last_pos = tokens.shape[1] - 1

                with model.hooks(fwd_hooks=hooks):
                    with torch.no_grad():
                        logits = model(tokens)

                gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
                gen_text = model.tokenizer.decode(gen_id).strip()
                per_sample_correct.append(
                    check_correct(gen_text, sample["answers"], dataset="hellaswag")
                )

            acc = sum(per_sample_correct) / n_samples
            delta = baseline_acc - acc
            print(f"  σ={sigma:.4f}: acc={acc:.4f} (Δ={delta:+.4f}), "
                  f"correct={sum(per_sample_correct)}/{n_samples}")

            mode_results.append({
                "sigma": sigma,
                "accuracy": acc,
                "delta_accuracy": delta,
                "n_correct": sum(per_sample_correct),
                "n_incorrect": n_samples - sum(per_sample_correct),
                "per_sample_correct": per_sample_correct,
            })

            gc.collect()
            torch.cuda.empty_cache()

        all_results[mode] = mode_results

    # --- Analysis: compare parallel vs isotropic degradation ---
    print(f"\n{'='*50} Analysis {'='*50}")
    print(f"\n{'Mode':<28} {'σ':>8} {'Acc':>8} {'Δ Acc':>8}")
    print("-" * 56)
    for mode in noise_modes:
        for r in all_results[mode]:
            print(f"{mode_labels[mode]:<28} {r['sigma']:>8.3f} {r['accuracy']:>8.4f} "
                  f"{r['delta_accuracy']:>+8.4f}")

    # Compare parallel_correct vs isotropic at moderate σ (0.5)
    key_sigma = 0.5
    comparisons = []
    for mode in noise_modes:
        for r in all_results[mode]:
            if abs(r["sigma"] - key_sigma) < 1e-6:
                comparisons.append((mode, r["accuracy"], r["delta_accuracy"]))

    print(f"\nAt σ={key_sigma}:")
    for mode, acc, delta in comparisons:
        print(f"  {mode_labels[mode]}: acc={acc:.4f}")

    # --- Save ---
    output = {
        "config": {
            "n_samples": n_samples,
            "model_id": "Qwen/Qwen3-1.7B",
            "n_layers": n_layers,
            "target_layer": target_layer,
            "target_hook": target_hook,
            "sigmas": sigmas,
            "baseline_accuracy": baseline_acc,
            "seed": seed,
        },
        "results": {
            mode: [{k: v for k, v in r.items() if k != "per_sample_correct"}
                   for r in mode_results]
            for mode, mode_results in all_results.items()
        },
    }

    results_file = output_path / "planB_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {"isotropic": "gray", "parallel_correct": "#e74c3c",
                  "parallel_pred": "#3498db", "orthogonal_correct": "#2ecc71"}
        markers = {"isotropic": "o", "parallel_correct": "s",
                   "parallel_pred": "D", "orthogonal_correct": "^"}

        for mode in noise_modes:
            s_vals = [r["sigma"] for r in all_results[mode]]
            accs = [r["accuracy"] for r in all_results[mode]]
            ax.plot(s_vals, accs, f"{markers[mode]}-", color=colors[mode],
                    label=mode_labels[mode], markersize=7, linewidth=2)

        ax.axhline(y=baseline_acc, color="black", linestyle="--", alpha=0.7,
                   label=f"baseline ({baseline_acc:.3f})")
        ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax.set_xlabel("Gaussian noise std (σ)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Plan B: Directional vs Isotropic Noise at L11\n"
                     f"Qwen3-1.7B, HellaSwag n={n_samples}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("symlog", linthresh=0.05)

        fig.tight_layout()
        plot_file = output_path / "planB_directional.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_planB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
