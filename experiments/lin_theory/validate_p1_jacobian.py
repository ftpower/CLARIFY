"""P1: Jacobian-vector product ||J_l @ d|| via finite differences.

Prediction: ||J @ v|| << ||J @ r|| for random unit vectors r.
v lies in the near-null space of the Jacobian (readout subspace),
while random directions have much larger effects on logits.

Method:
  - JVP: (logits(h + eps*d) - logits(h)) / eps  for direction d
  - Compare ||Jv|| / median(||Jr_i||) for n_random random directions
  - Also sweep epsilon to confirm linear regime.
  - P1 confirmed if median ratio < 0.1.

Note on precision: We compute logits in float16 (model dtype), then cast to
float32 for the difference and norm. With eps=0.1 and ||v||=1, the perturbation
in hidden space is 0.1, which relative to typical hidden state norms (~5-20)
is 0.5-2% — well within float16 precision.

Usage:
    python validate_p1_jacobian.py --n_calibrate 200 --n_test 10 \\
        --layer 20 --n_random 10
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
from common import load_model_and_unembed, compute_v


def compute_jvp_norm_fd(model, tokens, device, layer, direction_np, epsilon):
    """Compute ||J_l @ d|| via finite-difference Jacobian-vector product.

    JVP:  J @ d = (f(h + eps * d) - f(h)) / eps

    We use a pass-through hook on baseline to ensure identical code paths,
    then add eps * d for the perturbed forward pass.

    Args:
        model: HookedTransformer
        tokens: token tensor [1, seq_len]
        device: "cuda" or "cpu"
        layer: layer index for hook point
        direction_np: numpy array [d_model], unit norm
        epsilon: float, perturbation magnitude

    Returns:
        jvp_norm: float — ||J @ d||_2
        logits_base: numpy array [vocab_size] — baseline logits (float32)
        logits_pert: numpy array [vocab_size] — perturbed logits (float32)
    """
    d_tensor = torch.from_numpy(direction_np).float().to(device)
    d_f16 = d_tensor.to(dtype=torch.float16)
    last_pos = tokens.shape[1] - 1
    hook_name = f"blocks.{layer}.hook_resid_post"

    # ── Baseline: pass-through hook (identical code path) ──
    def _pass_through(act, hook=None):
        return act

    with torch.no_grad():
        logits_base = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_name, _pass_through)]
        )

    # ── Perturbed: add eps * d ──
    def _perturb(act, hook=None):
        act[:, -1, :] += epsilon * d_f16.unsqueeze(0)
        return act

    with torch.no_grad():
        logits_pert = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _perturb)])

    # Cast to float32 for numerical stability
    logits_base_f = logits_base[0, last_pos, :].float()
    logits_pert_f = logits_pert[0, last_pos, :].float()

    # Finite-difference JVP
    jvp = (logits_pert_f - logits_base_f) / epsilon
    jvp_norm = torch.norm(jvp).item()

    return jvp_norm, logits_base_f.cpu().numpy(), logits_pert_f.cpu().numpy()


def generate_random_directions(d_model, n_random, seed):
    """Generate n_random random unit vectors in R^d_model."""
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(n_random):
        r = rng.randn(d_model).astype(np.float32)
        r = r / (np.linalg.norm(r) + 1e-10)
        directions.append(r)
    return directions


def main():
    parser = argparse.ArgumentParser(
        description="P1: Jacobian-vector product validation"
    )
    parser.add_argument(
        "--n_calibrate", type=int, default=200, help="Samples for computing v (seed=42)"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10,
        help="Test samples for JVP computation (seed=123)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer for JVP (L20 = best detection layer)",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=10,
        help="Random directions per sample for baseline",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Finite-difference epsilon"
    )
    parser.add_argument(
        "--epsilon_sweep",
        type=float,
        nargs="*",
        default=[0.05, 0.1, 0.2],
        help="Epsilon values to sweep for linearity check",
    )
    parser.add_argument(
        "--n_linearity_check",
        type=int,
        default=3,
        help="Number of samples for epsilon sweep",
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
    print("P1: Jacobian-Vector Product — ||J @ d||")
    print(f"Layer: L{args.layer}, Test samples: {args.n_test}, Epsilon: {args.epsilon}")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, loaded in {time.time() - t0:.1f}s")

    # ── 2. Compute truth direction v at test layer ───────────────
    print(f"\n[2/5] Computing truth direction v at L{args.layer}...")
    v_tensor, v_stats = compute_v(
        model, tokenizer, args.n_calibrate, device, args.layer
    )
    v_np = v_tensor.float().cpu().numpy()
    print(
        f"  v_norm={float(np.linalg.norm(v_np)):.6f}, "
        f"correct={v_stats['n_correct']}, incorrect={v_stats['n_incorrect']}"
    )

    # ── 3. Epsilon linearity check on pilot samples ──────────────
    print(f"\n[3/5] Epsilon linearity check ({args.n_linearity_check} samples)...")

    pilot_samples = load_triviaqa(n_samples=args.n_linearity_check, seed=999)
    linearity_data = []

    for i, sample in enumerate(pilot_samples):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        for eps in args.epsilon_sweep:
            jvp_v = compute_jvp_norm_fd(model, tokens, device, args.layer, v_np, eps)[0]
            # One random direction for reference
            r = np.random.RandomState(999 + i).randn(d_model).astype(np.float32)
            r = r / (np.linalg.norm(r) + 1e-10)
            jvp_r = compute_jvp_norm_fd(model, tokens, device, args.layer, r, eps)[0]
            linearity_data.append(
                {
                    "sample": i,
                    "epsilon": eps,
                    "jvp_v": jvp_v,
                    "jvp_r": jvp_r,
                    "ratio": jvp_v / (jvp_r + 1e-10),
                }
            )

    print("  Epsilon sweep results:")
    for eps in args.epsilon_sweep:
        ratios = [d["ratio"] for d in linearity_data if d["epsilon"] == eps]
        jvp_vs = [d["jvp_v"] for d in linearity_data if d["epsilon"] == eps]
        jvp_rs = [d["jvp_r"] for d in linearity_data if d["epsilon"] == eps]
        print(
            f"    eps={eps:.2f}: ratio={np.median(ratios):.6f}  "
            f"||Jv||={np.median(jvp_vs):.4f}  ||Jr||={np.median(jvp_rs):.4f}"
        )

    # ── 4. Main JVP experiment ───────────────────────────────────
    print(
        f"\n[4/5] Computing ||Jv|| / ||Jr|| for {args.n_test} samples "
        f"(eps={args.epsilon})..."
    )

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    per_sample_results = []

    for i, sample in enumerate(tqdm(test_samples, desc="  JVP test")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # ||J @ v||
        jvp_v, _, _ = compute_jvp_norm_fd(
            model, tokens, device, args.layer, v_np, args.epsilon
        )

        # ||J @ r_i|| for random directions
        rand_dirs = generate_random_directions(
            d_model, args.n_random, args.seed + i * 1000
        )
        jvp_r_list = []
        for r in rand_dirs:
            jvp_r, _, _ = compute_jvp_norm_fd(
                model, tokens, device, args.layer, r, args.epsilon
            )
            jvp_r_list.append(jvp_r)

        median_jvr = float(np.median(jvp_r_list))
        ratio = jvp_v / (median_jvr + 1e-10)

        per_sample_results.append(
            {
                "sample_id": i,
                "question": sample["question"][:120],
                "jvp_v": jvp_v,
                "jvp_r_list": jvp_r_list,
                "jvp_r_median": median_jvr,
                "ratio": ratio,
            }
        )

    # ── 5. Summary ───────────────────────────────────────────────
    print(f"\n[5/5] Summary")

    ratios = np.array([r["ratio"] for r in per_sample_results])
    median_ratio = float(np.median(ratios))
    mean_ratio = float(np.mean(ratios))

    summary = {
        "n_test": args.n_test,
        "n_random": args.n_random,
        "layer": args.layer,
        "d_model": d_model,
        "epsilon": args.epsilon,
        "ratio_stats": {
            "mean": mean_ratio,
            "median": median_ratio,
            "std": float(np.std(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "values": [float(x) for x in ratios.tolist()],
        },
        "linearity_check": linearity_data,
        "p1_passes": bool(median_ratio < 0.10),
    }

    print(f"  Ratio ||Jv|| / median(||Jr||):")
    print(
        f"    mean={mean_ratio:.6f}  median={median_ratio:.6f}  "
        f"std={summary['ratio_stats']['std']:.6f}"
    )
    for i, r in enumerate(ratios):
        print(f"    sample {i}: ratio={r:.6f}")

    print(
        f"\n  P1 PASSES: {summary['p1_passes']} "
        f"(median ratio={median_ratio:.6f} vs threshold 0.10)"
    )

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "layer": args.layer,
            "epsilon": args.epsilon,
            "n_random": args.n_random,
            "d_model": d_model,
            "seed": args.seed,
        },
        "v_stats": v_stats,
        "per_sample": per_sample_results,
        "summary": summary,
    }

    results_path = output_dir / "p1_jacobian_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
