"""P3: Gradient-based vs v-based single-layer intervention.

Prediction: Intervening with g_L (gradient direction) produces non-zero
control effect (Δ_accuracy > 0), while intervening with v produces zero
effect (Δ_accuracy ≈ 0).

Method:
  - v:  h_L += alpha * v_unit    (traditional truth direction)
  - g:  h_L += alpha * g_unit    (gradient direction, one-shot oracle)
  - g is computed from the ORIGINAL h_L using the true answer token.
    This is an oracle correction — valid for proof-of-concept validation.
  - P3 confirmed if any g-intervention Δ > +5% AND all v-intervention
    |Δ| < 3%.

Usage:
    python validate_p3_gradient_intervention.py --n_calibrate 200 \\
        --n_test 30 --layer 27 --alphas -1.0 -0.5 0.5 1.0
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


def generate_with_intervention(
    model, tokenizer, prompt, device, layer, direction, alpha
):
    """Single-layer intervention: h_L += alpha * direction_unit.

    The intervention is applied on the FIRST forward pass only.
    Subsequent autoregressive steps are hook-free.

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        prompt: string
        device: "cuda" or "cpu"
        layer: layer index for hook point
        direction: torch.Tensor [d_model] — unit-norm direction (float32 on device)
        alpha: float — scaling factor

    Returns:
        generated_text: str
    """
    d_f16 = direction.to(dtype=torch.float16)  # match model dtype

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"
    return greedy_generate(
        model, tokenizer, prompt, device, fwd_hooks=[(hook_name, _intervene)]
    )


def main():
    parser = argparse.ArgumentParser(description="P3: Gradient vs v-based intervention")
    parser.add_argument(
        "--n_calibrate", type=int, default=200, help="Samples for computing v (seed=42)"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=30,
        help="Test samples for intervention (seed=123)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=27,
        help="Layer for intervention (L27 = last layer)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[-1.0, -0.5, 0.5, 1.0],
        help="Alpha values to sweep",
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
    print("P3: Gradient vs v-based Single-Layer Intervention")
    print(f"Layer: L{args.layer}, Test: {args.n_test}, Alphas: {args.alphas}")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────
    print("\n[1/5] Loading model + unembed...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, loaded in {time.time() - t0:.1f}s")

    # ── 2. Compute truth direction v ─────────────────────────────
    print(f"\n[2/5] Computing truth direction v at L{args.layer}...")
    v_tensor, v_stats = compute_v(
        model, tokenizer, args.n_calibrate, device, args.layer
    )
    print(f"  correct={v_stats['n_correct']}, incorrect={v_stats['n_incorrect']}")

    # ── 3. Run baseline ─────────────────────────────────────────
    print(f"\n[3/5] Baseline generation on {args.n_test} samples...")

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    baseline_correct = 0
    baseline_results = []

    for i, sample in enumerate(tqdm(test_samples, desc="  Baseline")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")
        if correct:
            baseline_correct += 1
        baseline_results.append(
            {
                "sample_id": i,
                "generated": gen_text,
                "is_correct": correct,
            }
        )

    baseline_rate = baseline_correct / args.n_test
    print(f"  Baseline: {baseline_correct}/{args.n_test} = {baseline_rate:.1%}")

    # ── 4. Run interventions (g-based + v-based) ────────────────
    print(
        f"\n[4/5] Running interventions ({len(args.alphas)} alphas × 2 directions)..."
    )

    # Pre-compute: extract h_L + compute g_L for each test sample
    # g_L is the one-shot oracle gradient (uses true answer token)
    precomputed = []
    skipped_idx = set()

    for i, sample in enumerate(tqdm(test_samples, desc="  Pre-compute g")):
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
            skipped_idx.add(i)
            precomputed.append(None)
            continue

        # Compute gradient
        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        g_np = g_L.float().numpy()
        g_norm = float(np.linalg.norm(g_np))
        g_unit = g_np / (g_norm + 1e-10)
        g_unit_tensor = torch.from_numpy(g_unit).float().to(device)

        precomputed.append(
            {
                "g_unit_tensor": g_unit_tensor,
                "g_norm": g_norm,
                "y_true_token": tokenizer.decode([y_true_id]),
                "prompt": prompt,
            }
        )

    # Recompute baseline rate on VALID samples only (exclude skipped)
    if skipped_idx:
        n_valid = args.n_test - len(skipped_idx)
        baseline_correct_valid = sum(
            1
            for i in range(args.n_test)
            if i not in skipped_idx and baseline_results[i]["is_correct"]
        )
        baseline_rate_valid = baseline_correct_valid / n_valid
        print(
            f"  Baseline (valid): {baseline_correct_valid}/{n_valid} = "
            f"{baseline_rate_valid:.1%}"
        )
    else:
        n_valid = args.n_test
        baseline_rate_valid = baseline_rate

    # Run interventions
    all_results = {}

    # v-based interventions
    for alpha in args.alphas:
        key = f"v_alpha={alpha:+.1f}"
        correct = 0

        for i, sample in enumerate(tqdm(test_samples, desc=f"  {key}", leave=False)):
            if i in skipped_idx:
                continue
            pc = precomputed[i]
            gen_text = generate_with_intervention(
                model, tokenizer, pc["prompt"], device, args.layer, v_tensor, alpha
            )
            if check_correct(gen_text, sample["answers"], dataset="triviaqa"):
                correct += 1

        rate = correct / n_valid if n_valid > 0 else 0.0
        delta = rate - baseline_rate_valid
        all_results[key] = {
            "correct": correct,
            "total": n_valid,
            "rate": rate,
            "delta": float(delta),
        }
        print(f"  {key}: {correct}/{n_valid} = {rate:.1%} (Δ={delta:+.1%})")

    # g-based interventions
    for alpha in args.alphas:
        key = f"g_alpha={alpha:+.1f}"
        correct = 0

        for i, sample in enumerate(tqdm(test_samples, desc=f"  {key}", leave=False)):
            if i in skipped_idx:
                continue
            pc = precomputed[i]
            gen_text = generate_with_intervention(
                model,
                tokenizer,
                pc["prompt"],
                device,
                args.layer,
                pc["g_unit_tensor"],
                alpha,
            )
            if check_correct(gen_text, sample["answers"], dataset="triviaqa"):
                correct += 1

        n_valid = args.n_test - len(skipped_idx)
        rate = correct / n_valid if n_valid > 0 else 0.0
        delta = rate - baseline_rate_valid
        all_results[key] = {
            "correct": correct,
            "total": n_valid,
            "rate": rate,
            "delta": float(delta),
        }
        print(f"  {key}: {correct}/{n_valid} = {rate:.1%} (Δ={delta:+.1%})")

    if skipped_idx:
        print(
            f"  Skipped {len(skipped_idx)} samples with un-tokenizable "
            f"answers: {sorted(skipped_idx)}"
        )

    # ── 5. Summary ───────────────────────────────────────────────
    print(f"\n[5/5] Summary")

    v_best_delta = max(v["delta"] for k, v in all_results.items() if k.startswith("v_"))
    v_best_key = max(
        ((k, v) for k, v in all_results.items() if k.startswith("v_")),
        key=lambda x: x[1]["delta"],
    )[0]

    g_best_delta = max(v["delta"] for k, v in all_results.items() if k.startswith("g_"))
    g_best_key = max(
        ((k, v) for k, v in all_results.items() if k.startswith("g_")),
        key=lambda x: x[1]["delta"],
    )[0]

    # Gate: any g-intervention Δ > +5%, all v-intervention |Δ| < 3%
    v_max_abs_delta = max(
        abs(v["delta"]) for k, v in all_results.items() if k.startswith("v_")
    )
    p3_g_works = g_best_delta > 0.05
    p3_v_fails = v_max_abs_delta < 0.03
    p3_passes = p3_g_works and p3_v_fails

    summary = {
        "n_test": args.n_test,
        "n_skipped": len(skipped_idx),
        "layer": args.layer,
        "d_model": d_model,
        "baseline_rate_all": baseline_rate,
        "baseline_rate_valid": baseline_rate_valid,
        "baseline_correct": baseline_correct,
        "v_best_delta": v_best_delta,
        "v_best_key": v_best_key,
        "v_max_abs_delta": v_max_abs_delta,
        "g_best_delta": g_best_delta,
        "g_best_key": g_best_key,
        "g_any_over_5pct": p3_g_works,
        "v_all_under_3pct": p3_v_fails,
        "p3_passes": p3_passes,
        "all_results": all_results,
    }

    print(f"  Baseline:          {baseline_rate:.1%}")
    print(f"  Best v-intervention:  {v_best_key} Δ={v_best_delta:+.1%}")
    print(f"  Best g-intervention:  {g_best_key} Δ={g_best_delta:+.1%}")
    print(f"  v max |Δ| = {v_max_abs_delta:.3f} (need <0.03)")
    print(f"  g best Δ  = {g_best_delta:.3f} (need >0.05)")
    print(f"\n  P3 PASSES: {p3_passes}")
    if not p3_g_works:
        print(f"    FAIL: no g-intervention exceeds +5%")
    if not p3_v_fails:
        print(f"    FAIL: some v-intervention |Δ| >= 3%")

    # Additional diagnostics
    g_norms = [pc["g_norm"] for pc in precomputed if pc is not None]
    summary["g_norm_stats"] = {
        "mean": float(np.mean(g_norms)),
        "median": float(np.median(g_norms)),
        "std": float(np.std(g_norms)),
        "min": float(np.min(g_norms)),
        "max": float(np.max(g_norms)),
    }
    print(
        f"\n  g_norm stats: mean={summary['g_norm_stats']['mean']:.4f}, "
        f"median={summary['g_norm_stats']['median']:.4f}"
    )

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "layer": args.layer,
            "alphas": args.alphas,
            "d_model": d_model,
            "seed": args.seed,
        },
        "v_stats": v_stats,
        "baseline": {
            "rate": baseline_rate,
            "correct": baseline_correct,
            "total": args.n_test,
        },
        "summary": summary,
    }

    results_path = output_dir / "p3_intervention_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
