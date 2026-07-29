"""A.5.1 + A.5.2: Δ log P diagnosis + amplitude calibration.

A.5.1 — Δ log P diagnosis:
  Measure log P(y_true) before and after g-intervention at L27.
  Gate: Δ log P > 1.0 nats → direction correct, amplitude problem.
        Δ log P < 0.5 nats → first-order approximation failed.

A.5.2 — Amplitude calibration:
  Sweep α ∈ {±0.5, ±1.0, ±2.0, ±5.0, ±10.0}, measure Δ log P vs α.
  Also measure h norm distribution at L20 and L27.
  Find where linear approximation breaks.

Usage:
    python validate_a5_logprob.py --n_test 30 --layer 27
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

from src.data_loader import load_triviaqa, format_prompt
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    compute_g_L,
    extract_h_at_layer,
)


# ═════════════════════════════════════════════════════════════════════════════
# Core: log P measurement with intervention
# ═════════════════════════════════════════════════════════════════════════════


def compute_log_p_with_intervention(
    model,
    tokens,
    layer,
    direction,
    alpha,
    y_true_id,
):
    """Compute log P(y_true) with single-layer intervention h_l += alpha * direction.

    The intervention is applied at blocks.{layer}.hook_resid_post (last token).
    The model continues through layers l+1..L, ln_final, and W_U.

    Args:
        model: HookedTransformer
        tokens: token tensor [1, seq_len]
        layer: intervention layer
        direction: [d_model] unit-norm direction (float32 on device)
        alpha: scaling factor
        y_true_id: ground-truth answer's first token ID

    Returns:
        log_p: float — log P(y_true | h + α·direction)
    """
    d_f16 = direction.to(dtype=torch.float16)

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _intervene)])

    log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    return float(log_probs[y_true_id].item())


def measure_h_norm_distribution(model, tokenizer, samples, device, layers):
    """Measure hidden state norm distribution at specified layers.

    Returns:
        dict: layer -> {mean, std, median, min, max, p5, p95, values: [...]}
    """
    results = {}
    for layer in layers:
        norms = []
        hook_name = f"blocks.{layer}.hook_resid_post"

        for sample in tqdm(samples, desc=f"  h-norm L{layer}"):
            prompt = format_prompt(
                sample["question"], sample["context"], dataset="triviaqa"
            )
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]

            residual = {}

            def _hook(act, hook=None):
                residual["h"] = act[:, -1, :].detach()
                return act

            with torch.no_grad():
                model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

            h_vec = residual["h"].float()
            norms.append(float(h_vec.norm().item()))

        norms_arr = np.array(norms)
        results[f"L{layer}"] = {
            "mean": float(np.mean(norms_arr)),
            "std": float(np.std(norms_arr)),
            "median": float(np.median(norms_arr)),
            "min": float(np.min(norms_arr)),
            "max": float(np.max(norms_arr)),
            "p5": float(np.percentile(norms_arr, 5)),
            "p95": float(np.percentile(norms_arr, 95)),
            "n_samples": len(norms),
        }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# A.5.1: Δ log P diagnosis
# ═════════════════════════════════════════════════════════════════════════════


def run_a5_1_logprob_diagnosis(
    model, tokenizer, W_U, b_U, ln_final, device, test_samples, layer, alphas
):
    """Measure Δ log P(y_true) for g-intervention and random-direction control.

    For each test sample:
      - Extract h_L
      - Compute g_L (analytical gradient)
      - Measure baseline log P(y_true)
      - Measure log P(y_true) after g-intervention at each α
      - Measure log P(y_true) after random-direction intervention (control)
    """
    d_model = model.cfg.d_model
    results = {
        "layer": layer,
        "n_samples": len(test_samples),
        "alphas": alphas,
        "samples": [],
    }

    # Precompute
    precomputed = []
    skipped_idx = set()

    for i, sample in enumerate(tqdm(test_samples, desc="  Pre-compute g_L")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        h_L, logits, tokens, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, layer
        )

        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            skipped_idx.add(i)
            precomputed.append(None)
            continue

        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        g_np = g_L.float().numpy()
        g_norm = float(np.linalg.norm(g_np))
        g_unit = g_np / (g_norm + 1e-10)
        g_unit_tensor = torch.from_numpy(g_unit).float().to(device)

        # Random direction (same norm as g)
        rng = np.random.RandomState(42 + i)
        r = rng.randn(d_model).astype(np.float32)
        r = r / (np.linalg.norm(r) + 1e-10)
        r_tensor = torch.from_numpy(r).float().to(device)

        # Baseline log P
        log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        baseline_log_p = float(log_probs[y_true_id].item())

        precomputed.append(
            {
                "g_unit_tensor": g_unit_tensor,
                "g_norm": g_norm,
                "r_tensor": r_tensor,
                "y_true_token": tokenizer.decode([y_true_id]),
                "y_true_id": y_true_id,
                "tokens": tokens,
                "baseline_log_p": baseline_log_p,
                "question": sample["question"],
            }
        )

    if skipped_idx:
        print(f"  Skipped {len(skipped_idx)} samples: {sorted(skipped_idx)}")

    # Measure Δ log P for each α
    print(f"\n  Measuring Δ log P ({len(alphas)} alphas × 2 directions)...")

    for i, pc in enumerate(tqdm(precomputed, desc="  Δ log P")):
        if pc is None:
            results["samples"].append(None)
            continue

        sample_result = {
            "sample_id": i,
            "baseline_log_p": pc["baseline_log_p"],
            "g_norm": pc["g_norm"],
            "g_interventions": {},
            "r_interventions": {},
        }

        for alpha in alphas:
            # g-direction intervention
            log_p_g = compute_log_p_with_intervention(
                model,
                pc["tokens"],
                layer,
                pc["g_unit_tensor"],
                alpha,
                pc["y_true_id"],
            )
            sample_result["g_interventions"][f"alpha={alpha:+.1f}"] = {
                "log_p": log_p_g,
                "delta": log_p_g - pc["baseline_log_p"],
            }

            # Random-direction intervention (control)
            log_p_r = compute_log_p_with_intervention(
                model,
                pc["tokens"],
                layer,
                pc["r_tensor"],
                alpha,
                pc["y_true_id"],
            )
            sample_result["r_interventions"][f"alpha={alpha:+.1f}"] = {
                "log_p": log_p_r,
                "delta": log_p_r - pc["baseline_log_p"],
            }

        results["samples"].append(sample_result)

    # Aggregate statistics
    valid_samples = [s for s in results["samples"] if s is not None]
    n_valid = len(valid_samples)

    aggregate = {
        "n_valid": n_valid,
        "n_skipped": len(skipped_idx),
        "baseline_log_p_mean": float(
            np.mean([s["baseline_log_p"] for s in valid_samples])
        ),
        "baseline_log_p_std": float(
            np.std([s["baseline_log_p"] for s in valid_samples])
        ),
        "g_norm_mean": float(np.mean([s["g_norm"] for s in valid_samples])),
        "g_norm_std": float(np.std([s["g_norm"] for s in valid_samples])),
        "by_alpha": {},
    }

    for alpha in alphas:
        key = f"alpha={alpha:+.1f}"
        g_deltas = [s["g_interventions"][key]["delta"] for s in valid_samples]
        r_deltas = [s["r_interventions"][key]["delta"] for s in valid_samples]

        aggregate["by_alpha"][key] = {
            "g_delta_mean": float(np.mean(g_deltas)),
            "g_delta_std": float(np.std(g_deltas)),
            "g_delta_median": float(np.median(g_deltas)),
            "g_delta_min": float(np.min(g_deltas)),
            "g_delta_max": float(np.max(g_deltas)),
            "r_delta_mean": float(np.mean(r_deltas)),
            "r_delta_std": float(np.std(r_deltas)),
        }

    results["aggregate"] = aggregate

    # Print summary
    print(f"\n  ── A.5.1 Δ log P Summary (L{layer}, {n_valid} valid samples) ──")
    print(
        f"  Baseline log P: {aggregate['baseline_log_p_mean']:.3f} ± "
        f"{aggregate['baseline_log_p_std']:.3f}"
    )
    print(f"  g norm: {aggregate['g_norm_mean']:.3f} ± {aggregate['g_norm_std']:.3f}")

    for alpha in alphas:
        key = f"alpha={alpha:+.1f}"
        gd = aggregate["by_alpha"][key]
        rd = aggregate["by_alpha"][key]
        print(
            f"  {key}: g Δ={gd['g_delta_mean']:+.4f}±{gd['g_delta_std']:.4f}  "
            f"r Δ={rd['r_delta_mean']:+.4f}±{rd['r_delta_std']:.4f}"
        )

    # Gate check for A.5.1
    key_a1 = "alpha=+1.0"
    g_delta_mean = aggregate["by_alpha"][key_a1]["g_delta_mean"]
    if g_delta_mean > 1.0:
        gate_result = "direction_correct_amplitude_insufficient"
    elif g_delta_mean < 0.5:
        gate_result = "first_order_approximation_failed"
    else:
        gate_result = "marginal"

    print(f"\n  Gate (α=+1.0): Δ log P = {g_delta_mean:+.4f} → {gate_result}")
    results["gate_a51"] = {
        "delta_log_p_at_alpha_1": g_delta_mean,
        "result": gate_result,
    }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# A.5.2: Amplitude calibration (extended α sweep)
# ═════════════════════════════════════════════════════════════════════════════


def run_a5_2_amplitude_calibration(
    model,
    tokenizer,
    W_U,
    b_U,
    ln_final,
    device,
    test_samples,
    layer,
    large_alphas,
    n_subset=10,
):
    """Sweep larger α values to find linearity boundary.

    Uses a subset of samples for efficiency.
    Measures Δ log P for α ∈ {±0.5, ±1.0, ±2.0, ±5.0, ±10.0}.
    """
    d_model = model.cfg.d_model
    subset = test_samples[:n_subset]

    print(f"  Using {len(subset)}-sample subset for efficiency")

    # Precompute
    precomputed = []
    for i, sample in enumerate(tqdm(subset, desc="  Pre-compute")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        h_L, logits, tokens, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, layer
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            precomputed.append(None)
            continue

        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        g_np = g_L.float().numpy()
        g_norm = float(np.linalg.norm(g_np))
        g_unit = g_np / (g_norm + 1e-10)
        g_unit_tensor = torch.from_numpy(g_unit).float().to(device)

        log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        baseline_log_p = float(log_probs[y_true_id].item())

        precomputed.append(
            {
                "g_unit_tensor": g_unit_tensor,
                "g_norm": g_norm,
                "y_true_id": y_true_id,
                "tokens": tokens,
                "baseline_log_p": baseline_log_p,
                "question": sample["question"],
            }
        )

    # Measure Δ log P for large α sweep
    results = {"alphas": large_alphas, "samples": []}

    for i, pc in enumerate(tqdm(precomputed, desc="  Large α sweep")):
        if pc is None:
            results["samples"].append(None)
            continue

        sample_result = {
            "sample_id": i,
            "baseline_log_p": pc["baseline_log_p"],
            "g_norm": pc["g_norm"],
            "interventions": {},
        }

        for alpha in large_alphas:
            log_p = compute_log_p_with_intervention(
                model,
                pc["tokens"],
                layer,
                pc["g_unit_tensor"],
                alpha,
                pc["y_true_id"],
            )
            sample_result["interventions"][f"alpha={alpha:+.1f}"] = {
                "log_p": log_p,
                "delta": log_p - pc["baseline_log_p"],
            }

        results["samples"].append(sample_result)

    # Analyze linearity
    valid = [s for s in results["samples"] if s is not None]
    n_valid = len(valid)

    # For each sample, compute the ratio Δ/α across alphas
    # Perfect linearity → Δ/α = constant
    linearity_analysis = {}
    for alpha in large_alphas:
        if alpha == 0:
            continue
        key = f"alpha={alpha:+.1f}"
        deltas = [s["interventions"][key]["delta"] for s in valid]
        ratios = [d / abs(alpha) for d in deltas]  # normalized by |α|
        linearity_analysis[key] = {
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            # Predicted from first-order: Δ ≈ α * ||g||²
            "predicted_delta": alpha * np.mean([s["g_norm"] ** 2 for s in valid]),
        }

    results["linearity_analysis"] = linearity_analysis
    results["n_valid"] = n_valid

    print(f"\n  ── A.5.2 Amplitude Calibration (L{layer}, {n_valid} valid) ──")
    print(f"  {'α':>8s}  {'Δ mean':>10s}  {'Δ/|α|':>10s}  {'Predicted':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for alpha in large_alphas:
        key = f"alpha={alpha:+.1f}"
        la = linearity_analysis[key]
        print(
            f"  {alpha:+8.1f}  {la['delta_mean']:+10.4f}  "
            f"{la['ratio_mean']:+10.4f}  {la['predicted_delta']:+10.4f}"
        )

    # Check for linearity breakdown
    # If ratio is stable across α → linear. If it degrades → nonlinear regime.
    ratios_by_alpha = [
        linearity_analysis[f"alpha={a:+.1f}"]["ratio_mean"] for a in large_alphas
    ]
    ratio_variation = np.std(ratios_by_alpha) / (abs(np.mean(ratios_by_alpha)) + 1e-10)
    print(
        f"\n  Ratio variation (CV): {ratio_variation:.3f} "
        f"({'linear' if ratio_variation < 0.2 else 'nonlinear regime'})"
    )

    results["ratio_cv"] = float(ratio_variation)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="A.5.1 + A.5.2: Δ log P diagnosis + amplitude calibration"
    )
    parser.add_argument("--n_test", type=int, default=30)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[-1.0, -0.5, 0.5, 1.0],
    )
    parser.add_argument(
        "--large_alphas",
        type=float,
        nargs="*",
        default=[-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--n_h_norm",
        type=int,
        default=50,
        help="Samples for h-norm distribution measurement",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--skip_a52", action="store_true", help="Skip A.5.2 amplitude calibration"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("A.5.1 + A.5.2: Δ log P Diagnosis + Amplitude Calibration")
    print(f"Layer: L{args.layer}, Test: {args.n_test}")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/5] Loading model + unembed...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, loaded in {time.time() - t0:.1f}s")

    # ── Load test samples ──
    print(f"\n[2/5] Loading {args.n_test} test samples (seed={args.seed})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    # ── Measure h norm distribution ──
    print(
        f"\n[3/5] Measuring h-norm distribution at L20, L27 "
        f"({args.n_h_norm} samples)..."
    )
    h_norm_samples = load_triviaqa(n_samples=args.n_h_norm, seed=99)
    h_norm_results = measure_h_norm_distribution(
        model, tokenizer, h_norm_samples[: args.n_h_norm], device, [20, 27]
    )
    for layer_key, stats in h_norm_results.items():
        print(
            f"  {layer_key}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"median={stats['median']:.2f}, p5={stats['p5']:.2f}, p95={stats['p95']:.2f}"
        )

    # ── A.5.1: Δ log P diagnosis ──
    print(f"\n[4/5] A.5.1: Δ log P diagnosis (α ∈ {args.alphas})...")
    a51_results = run_a5_1_logprob_diagnosis(
        model,
        tokenizer,
        W_U,
        b_U,
        ln_final,
        device,
        test_samples,
        args.layer,
        args.alphas,
    )

    # ── A.5.2: Amplitude calibration ──
    if args.skip_a52:
        print("\n[5/5] A.5.2: SKIPPED")
        a52_results = None
    else:
        print(f"\n[5/5] A.5.2: Amplitude calibration (α ∈ {args.large_alphas})...")
        a52_results = run_a5_2_amplitude_calibration(
            model,
            tokenizer,
            W_U,
            b_U,
            ln_final,
            device,
            test_samples,
            args.layer,
            args.large_alphas,
            n_subset=10,
        )

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "layer": args.layer,
            "alphas": args.alphas,
            "large_alphas": args.large_alphas,
            "d_model": d_model,
            "seed": args.seed,
        },
        "h_norm_distribution": h_norm_results,
        "a51_logprob_diagnosis": {
            "gate": a51_results["gate_a51"],
            "aggregate": a51_results["aggregate"],
            "n_samples": a51_results["n_samples"],
        },
    }
    if a52_results is not None:
        output["a52_amplitude_calibration"] = {
            "linearity_analysis": a52_results["linearity_analysis"],
            "ratio_cv": a52_results["ratio_cv"],
            "n_valid": a52_results["n_valid"],
        }

    # Save full results (with per-sample data)
    full_path = output_dir / "a5_logprob_full.json"
    with open(full_path, "w") as f:
        json.dump(
            {**output, "a51_full": a51_results, "a52_full": a52_results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nFull results saved to {full_path}")

    # Save summary (without per-sample data)
    summary_path = output_dir / "a5_logprob_summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    # ── Final gate assessment ──
    print("\n" + "=" * 60)
    print("Gate Assessment")
    print("=" * 60)
    gate_a51 = a51_results["gate_a51"]
    print(
        f"A.5.1: Δ log P(α=+1.0) = {gate_a51['delta_log_p_at_alpha_1']:+.4f} → "
        f"{gate_a51['result']}"
    )

    if a52_results is not None:
        print(
            f"A.5.2: Ratio CV = {a52_results['ratio_cv']:.3f} → "
            f"{'linear regime' if a52_results['ratio_cv'] < 0.2 else 'nonlinear regime'}"
        )

    # Practical interpretation
    if gate_a51["result"] == "direction_correct_amplitude_insufficient":
        print("\n→ 方向正确但幅度不足。多层级联或更大 α 可能有效。")
    elif gate_a51["result"] == "first_order_approximation_failed":
        print("\n→ 一阶近似失效。即使梯度方向也未能显著增加 P(y_true)。")
        print("→ 单层线性修正范式在根本上不足以影响 argmax。")
    else:
        print("\n→ 边缘结果。需结合幅度校准数据判断。")


if __name__ == "__main__":
    main()
