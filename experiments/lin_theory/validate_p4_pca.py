"""P4: PCA on gradient vectors g_L across different questions.

Prediction: g_L vectors from different questions share a low-rank structure,
supporting the feasibility of a low-rank bottleneck in LIN (rank r=8 to 32).

Validation:
  - Compute g_L for ~50 samples, stack into matrix G=[N, d_model]
  - Run PCA: report explained variance ratio for top-k components
  - Effective rank = min k such that cumulative var > 0.90
  - P4 supported if effective_rank_90 < 64

Usage:
    python validate_p4_pca.py --n_samples 50 --layer 27
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
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
    compute_v,
)


def main():
    parser = argparse.ArgumentParser(description="P4: PCA on gradient vectors")
    parser.add_argument(
        "--n_samples", type=int, default=50, help="Number of samples for PCA"
    )
    parser.add_argument(
        "--layer", type=int, default=27, help="Layer to extract g_L from"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Max PCA components (default: min(N, d_model))",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    print("P4: PCA on Gradient Vectors g_L")
    print(f"Layer: L{args.layer}, Samples: {args.n_samples}")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────
    print("\n[1/3] Loading model + unembed...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, loaded in {time.time() - t0:.1f}s")

    # ── 2. Compute g_L for all samples ───────────────────────────
    print(f"\n[2/3] Computing g_L for {args.n_samples} samples...")

    test_samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    test_samples = test_samples[: args.n_samples]

    G_list = []  # list of [d_model] numpy arrays
    skipped = 0

    for i, sample in enumerate(tqdm(test_samples, desc="  Extracting g_L")):
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

        # Compute exact gradient
        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        G_list.append(g_L.float().numpy())

    if len(G_list) == 0:
        raise RuntimeError(
            f"All {args.n_samples} samples had un-tokenizable answers. "
            "Cannot run PCA on empty gradient set."
        )

    G = np.stack(G_list, axis=0)  # [N_valid, d_model]
    n_valid = G.shape[0]
    print(f"  Collected {n_valid} gradients (skipped {skipped})")

    # ── 3. PCA ───────────────────────────────────────────────────
    print(f"\n[3/3] Running PCA on {n_valid} x {d_model} matrix...")

    # Center the data
    G_centered = G - G.mean(axis=0, keepdims=True)

    max_components = min(args.n_components or n_valid, n_valid, d_model)
    pca = PCA(n_components=max_components, random_state=args.seed)
    pca.fit(G_centered)

    evr = pca.explained_variance_ratio_  # [max_components]
    cumsum = np.cumsum(evr)

    # Effective rank at 90% variance
    effective_rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)

    # Select key k values for reporting
    report_k = [1, 2, 4, 8, 16, 32, 64]
    report_k = [k for k in report_k if k <= max_components]

    key_evr = {}
    for k in report_k:
        key_evr[str(k)] = {
            "individual": float(evr[k - 1]) if k <= len(evr) else None,
            "cumulative": float(cumsum[k - 1]) if k <= len(cumsum) else None,
        }

    # Also find exact k values for common thresholds
    thresholds = [0.50, 0.75, 0.90, 0.95]
    threshold_k = {}
    for t in thresholds:
        k = int(np.searchsorted(cumsum, t) + 1)
        threshold_k[f"k_at_{int(t * 100)}pct"] = k

    # ── Summary ───────────────────────────────────────────────
    summary = {
        "n_valid": n_valid,
        "n_skipped": skipped,
        "d_model": d_model,
        "max_components": max_components,
        "effective_rank_90": effective_rank_90,
        "top10_evr": [float(x) for x in evr[:10].tolist()],
        "top10_cumsum": [float(x) for x in cumsum[:10].tolist()],
        "key_components": key_evr,
        "threshold_ranks": threshold_k,
        "p4_supported": bool(effective_rank_90 < 64),
    }

    print(f"\n  Top-k explained variance ratio:")
    for k in report_k[:6]:
        if str(k) in key_evr:
            kv = key_evr[str(k)]
            print(
                f"    k={k:2d}:  indiv={kv['individual']:.4f}  "
                f"cum={kv['cumulative']:.4f}"
            )
    print(f"\n  Effective rank (90%): {effective_rank_90}")
    print(
        f"  P4 SUPPORTED: {summary['p4_supported']} "
        f"(effective_rank_90={effective_rank_90} vs threshold 64)"
    )

    # ── Compare top PC with v (optional) ────────────────────────
    if n_valid >= 10:
        # Quick v computation with disjoint calibration set (different seed)
        v_tensor, v_stats = compute_v(
            model,
            tokenizer,
            n_calibrate=50,
            device=device,
            layer=args.layer,
            seed=9999,
        )
        v_np = v_tensor.float().cpu().numpy()
        top_pc = pca.components_[0]  # [d_model] — direction of max variance
        cos_top_pc_v = float(
            np.dot(top_pc, v_np)
            / (np.linalg.norm(top_pc) * np.linalg.norm(v_np) + 1e-10)
        )
        summary["cos_top_pc_v"] = cos_top_pc_v
        print(f"  cos(top_PC, v): {cos_top_pc_v:.6f}")

    # ── Save ─────────────────────────────────────────────────────
    # Save full EVR for offline analysis
    output = {
        "config": {
            "n_samples": args.n_samples,
            "layer": args.layer,
            "seed": args.seed,
            "n_components": max_components,
        },
        "explained_variance_ratio": [float(x) for x in evr.tolist()],
        "cumulative_variance": [float(x) for x in cumsum.tolist()],
        "summary": summary,
    }

    results_path = output_dir / "p4_pca_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
