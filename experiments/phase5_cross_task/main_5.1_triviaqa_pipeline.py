"""Exp 5.1: TriviaQA Data Pipeline — end-to-end feature extraction for free-text QA.

Extracts per-token hallucination features during greedy generation, computes
correctness labels, and saves everything to JSON for downstream experiments.

Usage:
    python main_5.1_triviaqa_pipeline.py --n_samples 200
    python main_5.1_triviaqa_pipeline.py --n_samples 5   # quick test
    python main_5.1_triviaqa_pipeline.py --n_samples 500 --output_dir outputs
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

# Cross-phase imports (Phase 4 pattern: phase2_entropy/src for model/data,
# phase5_utils for local modules)
_sys_parent = Path(__file__).parent
sys.path.insert(0, str(_sys_parent.parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent))

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct
from phase5_utils.generation_features import (
    generate_with_per_token_features,
    compute_all_pair_js,
)


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    output_dir: str = "outputs",
    seed: int = 42,
    js_early: int = 11,
    js_late: int = 27,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading model {model_id}...")
    model = load_model(device=device, model_id=model_id)
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    print(f"  Model loaded: {n_layers} layers, d_model={model.cfg.d_model}")

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading TriviaQA ({n_samples} samples)...")
    samples = load_triviaqa(n_samples=n_samples, seed=seed)
    print(f"  Loaded {len(samples)} samples")

    # ── Per-sample extraction ──────────────────────────────────────────
    print(f"Extracting per-token features (max_new_tokens={max_new_tokens})...")
    per_sample_results = []
    correct_count = 0
    js_pairs_computed = 0

    for idx, sample in enumerate(tqdm(samples, desc="Samples")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        # ── Generate with per-token features ───────────────────────
        result = generate_with_per_token_features(
            model=model,
            prompt=prompt,
            W_U=W_U,
            b_U=b_U,
            max_new_tokens=max_new_tokens,
            js_early_layer=js_early,
            js_late_layer=js_late,
            temperature=temperature,
        )

        # ── Check correctness ──────────────────────────────────────
        is_correct = check_correct(
            result["answer_text"], sample["answers"], dataset="triviaqa"
        )
        if is_correct:
            correct_count += 1

        # ── Compute all-pair JS at the last generated token ───────
        if result["last_token_vocab_probs"] is not None:
            all_pair_js = compute_all_pair_js(
                result["last_token_vocab_probs"], n_layers, exclude_layer0=True
            )
            js_pairs_computed += 1
        else:
            all_pair_js = {}

        # ── Build per-sample record (discard full vocab probs) ────
        per_sample_results.append({
            "sample_id": idx,
            "question": sample["question"],
            "answers": sample["answers"],
            "generated_text": result["answer_text"],
            "is_correct": is_correct,
            "n_generated_tokens": result["n_tokens"],
            "per_token": result["per_token"],
            "last_token_js_all_pairs": all_pair_js,
        })

        # Periodic cleanup
        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    accuracy = correct_count / n_samples if n_samples > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct_count}/{n_samples})")
    print(f"JS all-pair computed for: {js_pairs_computed}/{n_samples} samples")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples,
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "js_early_layer": js_early,
            "js_late_layer": js_late,
            "temperature": temperature,
            "seed": seed,
            "n_layers": n_layers,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "incorrect_count": n_samples - correct_count,
        },
        "per_sample": per_sample_results,
    }

    output_file = output_path / "triviaqa_features.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # ── Quick stats ───────────────────────────────────────────────────
    n_tokens_list = [s["n_generated_tokens"] for s in per_sample_results]
    print(f"Generated tokens: mean={np.mean(n_tokens_list):.1f}, "
          f"min={np.min(n_tokens_list)}, max={np.max(n_tokens_list)}")

    if per_sample_results and per_sample_results[0]["per_token"]:
        n_layers_actual = len(per_sample_results[0]["per_token"][0]["max_p"])
        print(f"Features per token: {n_layers_actual} layers × (max_p + entropy) + d2_js")

    print("Done.")
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5.1: TriviaQA data extraction pipeline"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--js_early", type=int, default=11)
    parser.add_argument("--js_late", type=int, default=27)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        model_id=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        js_early=args.js_early,
        js_late=args.js_late,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
