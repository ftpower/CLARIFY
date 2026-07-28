"""P2: Cosine similarity between gradient g_L and truth direction v.

Prediction: |cos(g_L, v)| ≈ 0 — the gradient direction (which lies in the
control subspace) is nearly orthogonal to the truth direction v (which lies
in the readout subspace).

Validation:
  - Compute g_L = ∇_{h_L} log P(y_true | h_L) for ~20 samples via autograd
  - Compare cos(g_L, v) vs cos(g_L, random_direction) baseline
  - P2 confirmed if mean(|cos(g_L, v)|) < 0.15

Note: at d=2048, expected |cos| between random vectors ≈ sqrt(2/π)/sqrt(d) ≈ 0.018.
A |cos| < 0.15 is still substantially above random — we're looking for any
meaningful alignment, not just statistical significance.

Usage:
    python validate_p2_cosine.py --n_calibrate 200 --n_test 20 --layer 27
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Path setup ──────────────────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    compute_v,
    get_first_answer_token_id,
    compute_g_L,
    extract_h_at_layer,
    greedy_generate,
)


def main():
    parser = argparse.ArgumentParser(description="P2: cos(g_L, v) validation")
    parser.add_argument(
        "--n_calibrate", type=int, default=200, help="Samples for computing v (seed=42)"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=20,
        help="Test samples for cosine computation (seed=123)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=27,
        help="Layer to evaluate (default: L27, last layer)",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=10,
        help="Random unit vectors for baseline comparison",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("P2: Gradient-Direction Cosine Similarity")
    print(f"Layer: L{args.layer}, Calibrate: {args.n_calibrate}, Test: {args.n_test}")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────
    print("\n[1/4] Loading model + unembed...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(
        f"  Model: {model.cfg.model_name}, d={d_model}, "
        f"loaded in {time.time() - t0:.1f}s"
    )

    # ── 2. Compute truth direction v ─────────────────────────────
    print(f"\n[2/4] Computing truth direction v at L{args.layer}...")
    v_tensor, v_stats = compute_v(
        model, tokenizer, args.n_calibrate, device, args.layer
    )
    v_np = v_tensor.float().cpu().numpy()
    v_norm = float(np.linalg.norm(v_np))
    print(
        f"  v_norm={v_norm:.6f}, correct={v_stats['n_correct']}, "
        f"incorrect={v_stats['n_incorrect']}"
    )

    # ── 3. Compute cos(g_L, v) per test sample ───────────────────
    print(f"\n[3/4] Computing cos(g_L, v) for {args.n_test} samples...")

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]  # ensure exact count

    results = []
    skipped = 0

    for i, sample in enumerate(tqdm(test_samples, desc="  Test samples")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        # Extract h_L
        h_L, logits, tokens, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )

        # Get y_true token ID
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            skipped += 1
            continue

        # Generate for correctness label
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        # Compute exact gradient
        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        g_np = g_L.float().numpy()
        g_norm = float(np.linalg.norm(g_np))

        # cos(g, v)
        cos_g_v = float(np.dot(g_np, v_np) / (g_norm * v_norm + 1e-10))

        # cos(g, random) baseline — n_random unit vectors
        cos_g_rand = []
        rng = np.random.RandomState(args.seed + i * 1000)
        for _ in range(args.n_random):
            r = rng.randn(d_model).astype(np.float32)
            r = r / (np.linalg.norm(r) + 1e-10)
            c = float(np.dot(g_np, r) / (g_norm + 1e-10))
            cos_g_rand.append(c)

        results.append(
            {
                "sample_id": i,
                "question": sample["question"][:120],
                "is_correct": is_correct,
                "y_true_token": tokenizer.decode([y_true_id]) if y_true_id else None,
                "g_norm": g_norm,
                "cos_g_v": cos_g_v,
                "abs_cos_g_v": abs(cos_g_v),
                "cos_g_random": cos_g_rand,
                "abs_cos_g_random_mean": float(np.mean(np.abs(cos_g_rand))),
            }
        )

    if skipped:
        print(f"  Skipped {skipped} samples with un-tokenizable answers")

    # ── 4. Summary ───────────────────────────────────────────────
    print(f"\n[4/4] Summary")

    abs_cos_g_v = np.array([r["abs_cos_g_v"] for r in results])
    abs_cos_rand = np.array([r["abs_cos_g_random_mean"] for r in results])

    summary = {
        "n_valid": len(results),
        "n_skipped": skipped,
        "n_correct": sum(1 for r in results if r["is_correct"]),
        "abs_cos_g_v": {
            "mean": float(np.mean(abs_cos_g_v)),
            "median": float(np.median(abs_cos_g_v)),
            "std": float(np.std(abs_cos_g_v)),
            "min": float(np.min(abs_cos_g_v)),
            "max": float(np.max(abs_cos_g_v)),
            "frac_below_0_1": float((abs_cos_g_v < 0.1).mean()),
            "frac_below_0_15": float((abs_cos_g_v < 0.15).mean()),
        },
        "abs_cos_g_random": {
            "mean": float(np.mean(abs_cos_rand)),
            "median": float(np.median(abs_cos_rand)),
        },
        "d_model": d_model,
        "random_theoretical_expected": float(np.sqrt(2 / np.pi) / np.sqrt(d_model)),
        "p2_passes": bool(np.mean(abs_cos_g_v) < 0.15),
    }

    for k, v in summary["abs_cos_g_v"].items():
        print(f"  abs_cos_g_v.{k}: {v}")
    print(f"  abs_cos_g_random.mean: {summary['abs_cos_g_random']['mean']:.6f}")
    print(f"  random_theoretical:    {summary['random_theoretical_expected']:.6f}")
    print(
        f"\n  P2 PASSES: {summary['p2_passes']} "
        f"(mean |cos| = {summary['abs_cos_g_v']['mean']:.4f} vs threshold 0.15)"
    )

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "layer": args.layer,
            "d_model": d_model,
            "seed": args.seed,
            "n_random": args.n_random,
        },
        "v_stats": v_stats,
        "per_sample": results,
        "summary": summary,
    }

    results_path = output_dir / "p2_cosine_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
