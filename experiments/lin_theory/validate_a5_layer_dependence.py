"""A.5.3: Layer dependence — g-intervention at L15, L20, L27.

Compare g-based intervention across early (L15), middle (L20), and late (L27)
layers. Measure both Δ accuracy (generation) and Δ log P(y_true).

Hypotheses:
  A: L20 most effective — truth signal strongest at detection-optimal layer;
     perturbation amplified through 7+ downstream transformer layers.
  B: L27 most effective — closest to output, perturbation not attenuated.
  C: All zero — single-layer shift ineffective at any depth.

Usage:
    python validate_a5_layer_dependence.py --n_test 30
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
    get_first_answer_token_id,
    compute_g_L,
    extract_h_at_layer,
    greedy_generate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═════════════════════════════════════════════════════════════════════════════


def generate_with_g_intervention(
    model, tokenizer, prompt, device, layer, g_unit, alpha
):
    """Generate with g-based intervention at specified layer."""
    d_f16 = g_unit.to(dtype=torch.float16)

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"
    return greedy_generate(
        model, tokenizer, prompt, device, fwd_hooks=[(hook_name, _intervene)]
    )


def compute_log_p_with_g_intervention(
    model,
    tokens,
    layer,
    g_unit,
    alpha,
    y_true_id,
):
    """Compute log P(y_true) with g-intervention at specified layer."""
    d_f16 = g_unit.to(dtype=torch.float16)

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _intervene)])

    log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    return float(log_probs[y_true_id].item())


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="A.5.3: Layer dependence of g-intervention"
    )
    parser.add_argument("--n_test", type=int, default=30)
    parser.add_argument(
        "--layers", type=int, nargs="*", default=[15, 20, 27], help="Layers to compare"
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
    print("A.5.3: Layer Dependence — g-intervention at L15/L20/L27")
    print(f"Layers: {args.layers}, Test: {args.n_test}, Alphas: {args.alphas}")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/4] Loading model + unembed...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, loaded in {time.time() - t0:.1f}s")

    # ── Load test samples ──
    print(f"\n[2/4] Loading {args.n_test} test samples (seed={args.seed})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    # ── Baseline generation ──
    print(f"\n[3/4] Running baseline + layer-dependent g-interventions...")

    # Baseline (no intervention, same for all layers)
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
                "prompt": prompt,
            }
        )

    baseline_rate = baseline_correct / args.n_test
    print(f"  Baseline: {baseline_correct}/{args.n_test} = {baseline_rate:.1%}")

    # ── For each layer: precompute g_L + run interventions ──
    all_layer_results = {}

    for layer in args.layers:
        print(f"\n  ── Layer L{layer} ──")

        # Precompute g_L for each sample at this layer
        precomputed = []
        skipped_idx = set()

        for i, sample in enumerate(
            tqdm(test_samples, desc=f"    Pre-compute g_L{layer}")
        ):
            prompt = baseline_results[i]["prompt"]

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

            # Baseline log P at this layer
            log_probs = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
            baseline_log_p = float(log_probs[y_true_id].item())

            precomputed.append(
                {
                    "g_unit_tensor": g_unit_tensor,
                    "g_norm": g_norm,
                    "y_true_id": y_true_id,
                    "tokens": tokens,
                    "baseline_log_p": baseline_log_p,
                    "prompt": prompt,
                    "y_true_token": tokenizer.decode([y_true_id]),
                }
            )

        n_valid = args.n_test - len(skipped_idx)
        baseline_correct_valid = sum(
            1
            for i in range(args.n_test)
            if i not in skipped_idx and baseline_results[i]["is_correct"]
        )
        baseline_rate_valid = baseline_correct_valid / n_valid if n_valid > 0 else 0.0

        if skipped_idx:
            print(f"    Skipped {len(skipped_idx)}: {sorted(skipped_idx)}")
        print(
            f"    Valid samples: {n_valid}, baseline: "
            f"{baseline_correct_valid}/{n_valid} = {baseline_rate_valid:.1%}"
        )

        # g-norm stats
        g_norms = [pc["g_norm"] for pc in precomputed if pc is not None]
        print(f"    g_norm: mean={np.mean(g_norms):.3f}, std={np.std(g_norms):.3f}")

        # Run interventions
        layer_results = {
            "layer": layer,
            "n_valid": n_valid,
            "n_skipped": len(skipped_idx),
            "baseline_rate_valid": baseline_rate_valid,
            "baseline_correct_valid": baseline_correct_valid,
            "g_norm_mean": float(np.mean(g_norms)),
            "g_norm_std": float(np.std(g_norms)),
            "interventions": {},
        }

        for alpha in args.alphas:
            key = f"g_alpha={alpha:+.1f}"
            correct = 0
            deltas_log_p = []

            for i, sample in enumerate(
                tqdm(test_samples, desc=f"    {key}", leave=False)
            ):
                if i in skipped_idx:
                    continue
                pc = precomputed[i]

                # Generation
                gen_text = generate_with_g_intervention(
                    model,
                    tokenizer,
                    pc["prompt"],
                    device,
                    layer,
                    pc["g_unit_tensor"],
                    alpha,
                )
                if check_correct(gen_text, sample["answers"], dataset="triviaqa"):
                    correct += 1

                # Δ log P
                log_p = compute_log_p_with_g_intervention(
                    model,
                    pc["tokens"],
                    layer,
                    pc["g_unit_tensor"],
                    alpha,
                    pc["y_true_id"],
                )
                deltas_log_p.append(log_p - pc["baseline_log_p"])

            rate = correct / n_valid if n_valid > 0 else 0.0
            delta_acc = rate - baseline_rate_valid

            layer_results["interventions"][key] = {
                "correct": correct,
                "total": n_valid,
                "rate": rate,
                "delta_accuracy": float(delta_acc),
                "delta_log_p_mean": float(np.mean(deltas_log_p)),
                "delta_log_p_std": float(np.std(deltas_log_p)),
                "delta_log_p_median": float(np.median(deltas_log_p)),
            }

            print(
                f"    {key}: acc={rate:.1%} (Δ={delta_acc:+.1%})  "
                f"ΔlogP={np.mean(deltas_log_p):+.4f}±{np.std(deltas_log_p):.4f}"
            )

        all_layer_results[f"L{layer}"] = layer_results

    # ── Cross-layer comparison ──
    print(f"\n[4/4] Cross-layer comparison")
    print(
        f"\n  {'Layer':>6s}  {'Best Δ acc':>12s}  {'Best Δ logP':>14s}  {'g_norm':>8s}"
    )
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 14}  {'-' * 8}")

    cross_layer_summary = {}
    for layer in args.layers:
        lr = all_layer_results[f"L{layer}"]
        best_acc = max(v["delta_accuracy"] for v in lr["interventions"].values())
        best_logp = max(v["delta_log_p_mean"] for v in lr["interventions"].values())
        best_logp_key = max(
            lr["interventions"].items(),
            key=lambda x: x[1]["delta_log_p_mean"],
        )[0]
        print(
            f"  L{layer:>5d}  {best_acc:+12.1%}  {best_logp:+14.4f}  "
            f"{lr['g_norm_mean']:>8.3f}"
        )

        cross_layer_summary[f"L{layer}"] = {
            "best_delta_accuracy": best_acc,
            "best_delta_log_p": best_logp,
            "best_delta_log_p_alpha": best_logp_key,
            "g_norm_mean": lr["g_norm_mean"],
        }

    # Determine which hypothesis is supported
    best_acc_layer = max(
        cross_layer_summary.items(), key=lambda x: x[1]["best_delta_accuracy"]
    )
    best_logp_layer = max(
        cross_layer_summary.items(), key=lambda x: x[1]["best_delta_log_p"]
    )

    all_zero_acc = all(
        v["best_delta_accuracy"] == 0.0 for v in cross_layer_summary.values()
    )

    print(
        f"\n  Best Δ accuracy layer: {best_acc_layer[0]} "
        f"(Δ={best_acc_layer[1]['best_delta_accuracy']:+.1%})"
    )
    print(
        f"  Best Δ log P layer:   {best_logp_layer[0]} "
        f"(Δ={best_logp_layer[1]['best_delta_log_p']:+.4f})"
    )

    if all_zero_acc:
        print(
            "\n  → Hypothesis C supported: single-layer g-intervention "
            "zero-effect at ALL depths"
        )
    elif best_acc_layer[0] == "L20":
        print("\n  → Hypothesis A supported: L20 (detection-optimal) most effective")
    elif best_acc_layer[0] == "L27":
        print("\n  → Hypothesis B supported: L27 (closest to output) most effective")

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "layers": args.layers,
            "alphas": args.alphas,
            "d_model": d_model,
            "seed": args.seed,
        },
        "baseline": {
            "rate": baseline_rate,
            "correct": baseline_correct,
            "total": args.n_test,
        },
        "cross_layer_summary": cross_layer_summary,
        "all_zero_accuracy": all_zero_acc,
        "layer_results": all_layer_results,
    }

    results_path = output_dir / "a5_layer_dependence.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
