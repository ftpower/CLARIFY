"""Knowledge-filtered noise injection — single-pass fast version.

Uses single forward pass (no multi-token generation) since HellaSwag
answers are single-letter (A/B/C/D).
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
from src.data_loader import load_hellaswag, format_prompt, check_correct


def hidden_state_index_to_hook(idx, n_layers):
    if idx == 0:
        return "blocks.0.hook_resid_pre"
    if idx <= n_layers - 1:
        return f"blocks.{idx - 1}.hook_resid_post"
    return f"blocks.{n_layers - 1}.hook_resid_post"


def single_pass_noisy_answer(
    model, prompt, answers, dataset, inject_idx, sigma, n_layers
):
    """Single forward pass with noise at inject_idx. Returns is_correct."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    last_pos = tokens.shape[1] - 1

    hook_point = hidden_state_index_to_hook(inject_idx, n_layers)

    def noise_hook(act, hook):
        return act + torch.randn_like(act) * sigma

    with model.hooks(fwd_hooks=[(hook_point, noise_hook)]):
        with torch.no_grad():
            logits = model(tokens)

    gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
    gen_text = model.tokenizer.decode(gen_id).strip()
    return check_correct(gen_text, answers, dataset=dataset)


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

    # ── Load & filter ───────────────────────────────────────────────────
    with open(cache_file) as f:
        per_sample = json.load(f)

    from src.data_loader import load_hellaswag

    raw_samples = load_hellaswag(n_samples=n_samples, seed=seed)
    q2ctx = {s["question"]: s["context"] for s in raw_samples}
    for s in per_sample:
        s["context"] = q2ctx.get(s["question"], "")
        s["prompt"] = format_prompt(s["question"], s["context"], dataset="hellaswag")

    filtered = [s for s in per_sample if s["p_correct"] > thr]
    n_filt = len(filtered)
    n_correct = sum(1 for s in filtered if s["is_correct"])
    n_incorrect = n_filt - n_correct
    baseline_acc = n_correct / n_filt
    print(
        f"Filtered (thr={thr}): n={n_filt}, C={n_correct}, I={n_incorrect}, "
        f"acc={baseline_acc:.4f}"
    )

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    n_layers = model.cfg.n_layers

    # ── Config grid (reduced: most promising layers + sigmas) ────────────
    layers = [3, 7, 11, 15, 20]
    sigmas = [0.05, 0.1, 0.5, 1.0]
    configs = list(product(layers, sigmas))
    n_pass = n_filt * len(configs)
    print(f"Configs: {len(layers)}L × {len(sigmas)}σ = {len(configs)}, passes={n_pass}")

    # ── Baseline (no noise) ─────────────────────────────────────────────
    print("Computing baseline (no noise)...")
    n_baseline_correct = 0
    for sample in tqdm(filtered, desc="Baseline"):
        if single_pass_noisy_answer(
            model, sample["prompt"], sample["answers"], "hellaswag", 28, 0.0, n_layers
        ):
            n_baseline_correct += 1
    baseline_acc_check = n_baseline_correct / n_filt
    print(
        f"  Baseline (recomputed): {baseline_acc_check:.4f} "
        f"(cached: {baseline_acc:.4f})"
    )

    # ── Noise sweep ─────────────────────────────────────────────────────
    results = {}
    print(
        f"\n{'sigma':>8} {'layer':>6} {'acc':>8} {'delta':>8} {'w2c':>6} {'c2w':>6} {'net':>6}"
    )
    print("-" * 58)

    for layer, sigma in tqdm(configs, desc="Sweep"):
        n_corr = 0
        n_w2c = 0
        n_c2w = 0
        for sample in filtered:
            was_correct = sample["is_correct"]
            now_correct = single_pass_noisy_answer(
                model,
                sample["prompt"],
                sample["answers"],
                "hellaswag",
                layer,
                sigma,
                n_layers,
            )
            if now_correct:
                n_corr += 1
            if not was_correct and now_correct:
                n_w2c += 1
            if was_correct and not now_correct:
                n_c2w += 1

        acc = n_corr / n_filt
        delta = acc - baseline_acc
        net = n_w2c - n_c2w
        print(
            f"{sigma:>8.2f} L{layer:>5} {acc:>8.4f} {delta:>+8.4f} "
            f"{n_w2c:>6d} {n_c2w:>6d} {net:>+6d}"
        )
        results[f"L{layer}_s{sigma}"] = {
            "layer": layer,
            "sigma": sigma,
            "accuracy": acc,
            "delta": delta,
            "w2c": n_w2c,
            "c2w": n_c2w,
            "net": net,
        }

    # ── Summary ─────────────────────────────────────────────────────────
    best = max(results.values(), key=lambda x: x["delta"])
    print(
        f"\nBest net benefit: sigma={best['sigma']}, L{best['layer']}, "
        f"delta={best['delta']:+.4f}, net={best['net']:+d}"
    )

    # Mini heatmap
    print(f"\n{'sigma':>8}", end="")
    for l in layers:
        print(f" L{l:>6}", end="")
    print(f"\n{'-' * (8 + 7 * len(layers))}")
    for s in sigmas:
        print(f"{s:>8.2f}", end="")
        for l in layers:
            key = f"L{l}_s{s}"
            print(f" {results[key]['delta']:+7.4f}", end="")
        print()

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "model": model_id,
        "dataset": "hellaswag",
        "thr": thr,
        "n_filtered": n_filt,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "baseline_acc": baseline_acc,
        "baseline_acc_recomputed": baseline_acc_check,
        "best": best,
        "results": results,
    }
    results_file = output_path / "filtered_noise_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_file}")

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
