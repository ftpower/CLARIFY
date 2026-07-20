"""Knowledge-filtered fixed-σ noise injection experiment.

Retests Phase 5 on the knowledge-filtered subset (P(correct) > 0.3, acc=74.6%).
Hypothesis: noise injection failed on the full set (52% acc) because "incorrect"
answers were mostly random noise, not true hallucinations. On the filtered subset,
incorrect answers are more likely to be real hallucinations with a distinct pathway
that noise could disrupt.

Usage:
    python main_filtered_noise.py
    python main_filtered_noise.py --n_samples 500 --thr 0.3
"""

import gc
import json
import os
import sys
from itertools import product
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import format_prompt, check_correct
from src.adaptive_noise import run_fixed_sigma


def main(
    n_samples: int = 500,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    thr: float = 0.3,
    output_dir: str = "outputs",
    seed: int = 42,
):
    output_path = Path(output_dir)
    cache_file = output_path / "knowledge_filtered_data.json"

    # ── Load filtered samples ──────────────────────────────────────────
    if not cache_file.exists():
        print(
            f"Cache file {cache_file} not found. Run main_knowledge_filtered.py first."
        )
        return

    with open(cache_file) as f:
        per_sample = json.load(f)

    from src.data_loader import load_hellaswag

    raw_samples = load_hellaswag(n_samples=n_samples, seed=seed)
    question_to_ctx = {s["question"]: s["context"] for s in raw_samples}
    for s in per_sample:
        s["context"] = question_to_ctx.get(s["question"], "")

    # Filter
    filtered = [s for s in per_sample if s["p_correct"] > thr]
    n_filt = len(filtered)
    n_correct = sum(1 for s in filtered if s["is_correct"])
    n_incorrect = n_filt - n_correct
    baseline_acc = n_correct / n_filt
    print(f"Filtered subset (P(correct) > {thr}): ")
    print(
        f"  n={n_filt}, correct={n_correct}, incorrect={n_incorrect}, "
        f"acc={baseline_acc:.4f}"
    )

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    n_layers = model.cfg.n_layers
    n_total = n_layers + 1

    # ── Config grid ────────────────────────────────────────────────────
    layers = [0, 3, 7, 11, 15, 20, 24, 28]
    sigmas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    configs = list(product(layers, sigmas))
    print(f"Configs: {len(layers)} layers × {len(sigmas)} σ = {len(configs)}")
    print(f"Total forward passes: {n_filt} × {len(configs)} = {n_filt * len(configs)}")

    # ── Run noise injection ────────────────────────────────────────────
    heatmap = np.full((len(sigmas), len(layers)), np.nan)
    correct_before = np.full((len(sigmas), len(layers)), 0)
    correct_after = np.full((len(sigmas), len(layers)), 0)
    flips_wrong_to_correct = np.full((len(sigmas), len(layers)), 0)
    flips_correct_to_wrong = np.full((len(sigmas), len(layers)), 0)

    for si, sigma in enumerate(sigmas):
        for li, layer in enumerate(layers):
            n_flip_w2c = 0
            n_flip_c2w = 0
            n_corr = 0

            for sample in tqdm(filtered, desc=f"σ={sigma} L{layer}", leave=False):
                prompt = format_prompt(
                    sample["question"], sample["context"], dataset="hellaswag"
                )
                result = run_fixed_sigma(
                    model,
                    prompt,
                    sample["answers"],
                    "hellaswag",
                    layer,
                    sigma,
                    n_layers,
                )
                was_correct = sample["is_correct"]
                now_correct = result["is_correct"]
                if now_correct:
                    n_corr += 1
                if not was_correct and now_correct:
                    n_flip_w2c += 1
                if was_correct and not now_correct:
                    n_flip_c2w += 1

            acc = n_corr / n_filt
            heatmap[si, li] = acc
            flips_wrong_to_correct[si, li] = n_flip_w2c
            flips_correct_to_wrong[si, li] = n_flip_c2w
            net_benefit = n_flip_w2c - n_flip_c2w
            print(
                f"  σ={sigma:>5} L{layer:>2}: acc={acc:.4f} "
                f"(Δ={acc - baseline_acc:+.4f}) "
                f"flips: W→C={n_flip_w2c} C→W={n_flip_c2w} net={net_benefit:+d}"
            )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"Heatmap: σ × Layer → Accuracy (baseline={baseline_acc:.4f})")
    print(f"{'=' * 65}")
    header = "sigma/L  " + "  ".join(f"L{l:<5}" for l in layers)
    print(header)
    print(header)
    sep = "-" * len(header)
    print(sep)
    for si, sigma in enumerate(sigmas):
        vals = "  ".join(f"{heatmap[si, li]:.4f}" for li in range(len(layers)))
        print(f"  {sigma:<8.2f} {vals}")

    # Best result
    best_idx = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
    best_acc = heatmap[best_idx]
    best_sigma = sigmas[best_idx[0]]
    best_layer = layers[best_idx[1]]
    print(
        f"\nBest: sigma={best_sigma}, L{best_layer} -> acc={best_acc:.4f} "
        f"(delta={best_acc - baseline_acc:+.4f})"
    )

    # Flips summary
    print("\nNet benefit (W->C minus C->W):")
    print(header)
    print(sep)
    for si, sigma in enumerate(sigmas):
        net = flips_wrong_to_correct[si] - flips_correct_to_wrong[si]
        vals = "  ".join(f"{int(net[li]):>5d}" for li in range(len(layers)))
        print(f"  {sigma:<8.2f} {vals}")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "model": model_id,
        "dataset": "hellaswag",
        "thr": thr,
        "n_filtered": n_filt,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "baseline_acc": float(baseline_acc),
        "layers": layers,
        "sigmas": sigmas,
        "heatmap": heatmap.tolist(),
        "flips_w2c": flips_wrong_to_correct.tolist(),
        "flips_c2w": flips_correct_to_wrong.tolist(),
        "best_accuracy": float(best_acc),
        "best_sigma": float(best_sigma),
        "best_layer": int(best_layer),
        "best_delta": float(best_acc - baseline_acc),
    }

    results_file = output_path / "filtered_noise_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_file}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--thr", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_samples, args.device, args.model, args.thr, args.output_dir, args.seed)
