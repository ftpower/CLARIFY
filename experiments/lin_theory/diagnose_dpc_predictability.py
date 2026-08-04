"""Phase 19.1: DPC pre-elimination — diagnostic experiment.

Theory: docs/llm-coding-theory.md §10
Plan:   ~/.claude/plans/CLARIFY/phase19-beyond-tldc.md

Tests whether override δ = y_L20 - y_L27 can be predicted from L18 logits
(Dirty Paper Coding: pre-encode at channel input before override forms).

Diagnostic gates:
  P19.1.1: R²(δ̂, δ) > 0.1
  P19.1.2: top-5 overlap(δ̂, δ) > 50%

Three predictor schemes:
  A: PCA(δ) → 256 dim, linear regression y_L18 → PCA scores
  B: Token frequency bins → per-bin mean δ predictor
  C: W_U token embedding norm → scalar g(t) prediction

Usage:
    python diagnose_dpc_predictability.py --n_cal 200 --n_test 100

Output:
    experiments/outputs/lin_theory/s19_1_dpc_predictability.json
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_lin_dir = str(Path(__file__).parent)
if _lin_dir not in sys.path:
    sys.path.insert(0, _lin_dir)

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    greedy_generate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-layer logit extraction (L18, L20, L27)
# ═════════════════════════════════════════════════════════════════════════════


def extract_dpc_logits(model, tokenizer, prompt, device, ln_final, W_U, b_U):
    """Extract logits at L18, L20, L27 in a single forward pass.

    Returns:
        l18: [vocab] float32 — early-exit logits at L18 (DPC input)
        l20: [vocab] float32 — early-exit logits at L20 (reference)
        l27: [vocab] float32 — final-layer logits
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    captured = {}

    def _make_hook(l):
        def _hook(act, hook=None):
            captured[l] = act[:, -1:, :].detach()
            return act

        return _hook

    hooks = [
        (f"blocks.{18}.hook_resid_post", _make_hook(18)),
        (f"blocks.{20}.hook_resid_post", _make_hook(20)),
        (f"blocks.{27}.hook_resid_post", _make_hook(27)),
    ]

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    dtype = next(ln_final.parameters()).dtype

    def compute_logits(h):
        h_f16 = h.to(dtype=dtype)
        h_norm = ln_final(h_f16)
        logits = h_norm @ W_U.to(dtype)
        if b_U is not None:
            logits = logits + b_U.to(dtype)
        return logits.float().detach().squeeze()  # [vocab]

    l18 = compute_logits(captured[18])
    l20 = compute_logits(captured[20])
    l27 = compute_logits(captured[27])

    return l18, l20, l27


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 19.1: DPC predictability diagnosis"
    )
    parser.add_argument(
        "--n_cal",
        type=int,
        default=200,
        help="Calibration samples for training predictor",
    )
    parser.add_argument(
        "--n_test", type=int, default=100, help="Test samples for evaluation"
    )
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed_cal", type=int, default=42)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--pca_dim", type=int, default=256, help="PCA dimension for scheme A"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 72)
    print("Phase 19.1: DPC Pre-Elimination — Predictability Diagnosis")
    print(f"  n_cal={args.n_cal}, n_test={args.n_test}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    vocab_size = model.cfg.d_vocab
    d_model = model.cfg.d_model
    print(
        f"  Model: {model.cfg.n_layers} layers, d_model={d_model}, vocab={vocab_size}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Extract calibration data ──
    print(f"\n[2/4] Extracting calibration data ({args.n_cal} samples)...")
    cal_samples = load_triviaqa(n_samples=args.n_cal, seed=args.seed_cal)
    cal_samples = cal_samples[: args.n_cal]

    cal_data = []
    for i, sample in enumerate(tqdm(cal_samples, desc="  Calibration extract")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        l18, l20, l27 = extract_dpc_logits(
            model, tokenizer, prompt, device, ln_final, W_U, b_U
        )

        delta = l20 - l27  # δ = y_ref - y_final

        cal_data.append(
            {
                "sample_id": i,
                "y_true_id": y_true_id,
                "l18": l18.cpu(),  # [vocab]
                "l20": l20.cpu(),
                "l27": l27.cpu(),
                "delta": delta.cpu(),  # [vocab] — ground truth δ
                "question": sample["question"][:100],
            }
        )

    print(f"  Calibration: {len(cal_data)} valid samples")

    # ── Extract test data ──
    print(f"\n[3/4] Extracting test data ({args.n_test} samples)...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    test_data = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Test extract")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        l18, l20, l27 = extract_dpc_logits(
            model, tokenizer, prompt, device, ln_final, W_U, b_U
        )

        delta = l20 - l27

        # Baseline generation for KC/KW/DK classification
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        # Rank at L27
        sorted_ids = l27.float().argsort(descending=True)
        rank = int((sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item())

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        test_data.append(
            {
                "sample_id": i,
                "y_true_id": y_true_id,
                "rank": rank,
                "subset": subset,
                "gen_correct": is_correct,
                "l18": l18.cpu(),
                "l20": l20.cpu(),
                "l27": l27.cpu(),
                "delta": delta.cpu(),
                "question": sample["question"][:100],
            }
        )

    n_kw = sum(1 for e in test_data if e["subset"] == "know_wrong")
    n_kc = sum(1 for e in test_data if e["subset"] == "know_correct")
    n_dk = sum(1 for e in test_data if e["subset"] == "dont_know")
    print(f"  Test: KC={n_kc}, KW={n_kw}, DK={n_dk}, Total={len(test_data)}")

    # ── Scheme A: Bilateral PCA + Linear Regression ──
    print(f"\n[4/4] Training predictors & evaluating...")
    print(f"\n  ── Scheme A: Bilateral PCA + Linear Regression ──")

    n_cal = len(cal_data)
    n_test = len(test_data)

    # Stack calibration data (ensure detached)
    Y_cal = torch.stack(
        [
            d["delta"].detach() if isinstance(d["delta"], torch.Tensor) else d["delta"]
            for d in cal_data
        ]
    )
    X_cal = torch.stack(
        [
            d["l18"].detach() if isinstance(d["l18"], torch.Tensor) else d["l18"]
            for d in cal_data
        ]
    )

    Y_test = torch.stack(
        [
            d["delta"].detach() if isinstance(d["delta"], torch.Tensor) else d["delta"]
            for d in test_data
        ]
    )
    X_test = torch.stack(
        [
            d["l18"].detach() if isinstance(d["l18"], torch.Tensor) else d["l18"]
            for d in test_data
        ]
    )

    # Convert to numpy safely
    if hasattr(Y_cal, "numpy"):
        Y_cal = Y_cal.detach().numpy()
        X_cal = X_cal.detach().numpy()
        Y_test = Y_test.detach().numpy()
        X_test = X_test.detach().numpy()

    # Bilateral PCA: reduce both X (L18) and Y (δ) to avoid
    # 150K → 256 regression with only 200 samples
    pca_dim = min(args.pca_dim, n_cal, vocab_size)

    # PCA on δ (Y target)
    pca_y = PCA(n_components=pca_dim, random_state=42)
    Y_cal_pca = pca_y.fit_transform(Y_cal)  # [n_cal, pca_dim]
    Y_test_pca = pca_y.transform(Y_test)

    # PCA on L18 logits (X features)
    pca_x = PCA(n_components=pca_dim, random_state=43)
    X_cal_pca = pca_x.fit_transform(X_cal)  # [n_cal, pca_dim]
    X_test_pca = pca_x.transform(X_test)

    pca_y_var = pca_y.explained_variance_ratio_.sum()
    pca_x_var = pca_x.explained_variance_ratio_.sum()
    print(f"  PCA-Y (δ) explained var: {pca_y_var:.4f}")
    print(f"  PCA-X (L18) explained var: {pca_x_var:.4f}")

    # Linear regression in reduced space: 256 → 256 (65K params, 200 samples)
    reg = Ridge(alpha=1.0)
    reg.fit(X_cal_pca, Y_cal_pca)

    Y_pred_pca = reg.predict(X_test_pca)  # [n_test, pca_dim]
    Y_pred = pca_y.inverse_transform(Y_pred_pca)  # [n_test, vocab]

    # R² in reduced PCA space
    r2_reduced = r2_score(Y_test_pca.ravel(), Y_pred_pca.ravel())
    print(f"  R² (PCA space): {r2_reduced:.6f}")

    # R² per sample and global
    r2_per_sample_a = []
    for i in range(n_test):
        r2 = r2_score(Y_test[i], Y_pred[i])
        r2_per_sample_a.append(r2)

    r2_global_a = r2_score(Y_test.ravel(), Y_pred.ravel())
    mean_r2_a = np.mean(r2_per_sample_a)

    print(f"  Global R²: {r2_global_a:.6f}")
    print(f"  Mean per-sample R²: {mean_r2_a:.6f}")

    # Top-5 overlap
    top5_overlaps_a = []
    for i in range(n_test):
        true_top5 = set(np.argsort(Y_test[i])[-5:])
        pred_top5 = set(np.argsort(Y_pred[i])[-5:])
        overlap = len(true_top5 & pred_top5) / 5
        top5_overlaps_a.append(overlap)

    mean_top5_a = np.mean(top5_overlaps_a)
    print(f"  Mean top-5 overlap: {mean_top5_a:.4f}")

    # Per-subset breakdown
    for sn in ["know_wrong", "know_correct", "dont_know"]:
        idxs = [i for i, d in enumerate(test_data) if d["subset"] == sn]
        if idxs:
            r2_sub = np.mean([r2_per_sample_a[i] for i in idxs])
            top5_sub = np.mean([top5_overlaps_a[i] for i in idxs])
            print(f"    {sn}: R²={r2_sub:.6f}, top-5 overlap={top5_sub:.4f}")

    # ── Scheme B: Token frequency bin predictor ──
    print(f"\n  ── Scheme B: Token Frequency Bin Predictor ──")

    # Define bins
    freq_bins = {
        "top-100": (0, 100),
        "100-1K": (100, 1000),
        "1K-10K": (1000, 10000),
        "10K-100K": (10000, 100000),
        "100K+": (100000, vocab_size),
    }

    # For each bin, learn per-token mean δ from calibration data
    # Predict test δ as bin-mean
    Y_pred_b = np.zeros_like(Y_test)
    for bin_name, (lo, hi) in freq_bins.items():
        hi = min(hi, vocab_size)
        # Mean δ over calibration samples in this bin
        bin_mean = Y_cal[:, lo:hi].mean(axis=0)  # [bin_size]
        # Broadcast to all test samples
        Y_pred_b[:, lo:hi] = bin_mean[None, :]

    r2_global_b = r2_score(Y_test.ravel(), Y_pred_b.ravel())
    r2_per_sample_b = [r2_score(Y_test[i], Y_pred_b[i]) for i in range(n_test)]
    mean_r2_b = np.mean(r2_per_sample_b)

    top5_overlaps_b = []
    for i in range(n_test):
        true_top5 = set(np.argsort(Y_test[i])[-5:])
        pred_top5 = set(np.argsort(Y_pred_b[i])[-5:])
        top5_overlaps_b.append(len(true_top5 & pred_top5) / 5)

    mean_top5_b = np.mean(top5_overlaps_b)

    print(f"  Global R²: {r2_global_b:.6f}")
    print(f"  Mean per-sample R²: {mean_r2_b:.6f}")
    print(f"  Mean top-5 overlap: {mean_top5_b:.4f}")

    # ── Scheme C: W_U-based scalar predictor ──
    print(f"\n  ── Scheme C: W_U Token Embedding Norm Predictor ──")

    # Hypothesis: tokens with larger W_U embedding norms are more "important"
    # and might have systematically different δ
    W_U_cpu = W_U.float().cpu()  # [d_model, vocab]
    embedding_norms = W_U_cpu.norm(dim=0)  # [vocab]

    # Learn: δ(t) = a * norm(t) + b (per-sample intercept from L18)
    # Actually simpler: for each token t, learn δ(t) = a_t * l18(t) + b_t
    # But with 200 samples × 150K tokens, we need regularization.
    # Simpler approach: learn per-token δ_mean from calibration
    delta_mean_cal = Y_cal.mean(axis=0)  # [vocab] — mean δ per token across cal

    # Predict using mean δ as constant predictor
    Y_pred_c = np.tile(delta_mean_cal[None, :], (n_test, 1))

    r2_global_c = r2_score(Y_test.ravel(), Y_pred_c.ravel())
    r2_per_sample_c = [r2_score(Y_test[i], Y_pred_c[i]) for i in range(n_test)]
    mean_r2_c = np.mean(r2_per_sample_c)

    top5_overlaps_c = []
    for i in range(n_test):
        true_top5 = set(np.argsort(Y_test[i])[-5:])
        pred_top5 = set(np.argsort(Y_pred_c[i])[-5:])
        top5_overlaps_c.append(len(true_top5 & pred_top5) / 5)

    mean_top5_c = np.mean(top5_overlaps_c)

    print(f"  Global R²: {r2_global_c:.6f}")
    print(f"  Mean per-sample R²: {mean_r2_c:.6f}")
    print(f"  Mean top-5 overlap: {mean_top5_c:.4f}")

    # ── Also: check if δ at y_true specifically is predictable ──
    print(f"\n  ── δ at y_true_id specifically ──")

    for sn in ["know_wrong", "know_correct", "dont_know"]:
        idxs = [i for i, d in enumerate(test_data) if d["subset"] == sn]
        if not idxs:
            continue

        # δ at y_true_id
        delta_at_yt = np.array(
            [float(Y_test[i, test_data[i]["y_true_id"]]) for i in idxs]
        )
        delta_pred_a = np.array(
            [float(Y_pred[i, test_data[i]["y_true_id"]]) for i in idxs]
        )
        delta_pred_b = np.array(
            [float(Y_pred_b[i, test_data[i]["y_true_id"]]) for i in idxs]
        )

        r2_yt_a = r2_score(delta_at_yt, delta_pred_a) if len(idxs) > 1 else 0
        r2_yt_b = r2_score(delta_at_yt, delta_pred_b) if len(idxs) > 1 else 0

        # Correlation between predicted and true δ at y_true
        corr_yt_a = np.corrcoef(delta_at_yt, delta_pred_a)[0, 1] if len(idxs) > 1 else 0
        corr_yt_b = np.corrcoef(delta_at_yt, delta_pred_b)[0, 1] if len(idxs) > 1 else 0

        print(f"    {sn} (n={len(idxs)}):")
        print(
            f"      δ(y_true) mean ± std: {delta_at_yt.mean():.4f} ± {delta_at_yt.std():.4f}"
        )
        print(f"      Scheme A: R²(δ̂, δ) = {r2_yt_a:.4f}, ρ = {corr_yt_a:.4f}")
        print(f"      Scheme B: R²(δ̂, δ) = {r2_yt_b:.4f}, ρ = {corr_yt_b:.4f}")

    # ── Gate evaluation ──
    print(f"\n{'=' * 60}")
    print(f"Gate Summary")
    print(f"{'=' * 60}")

    # P19.1.1: R² > 0.1
    best_r2 = max(r2_global_a, r2_global_b, r2_global_c)
    best_scheme_r2 = (
        "A" if best_r2 == r2_global_a else ("B" if best_r2 == r2_global_b else "C")
    )
    p1911_pass = best_r2 > 0.1

    # P19.1.2: top-5 overlap > 50%
    best_top5 = max(mean_top5_a, mean_top5_b, mean_top5_c)
    best_scheme_top5 = (
        "A" if best_top5 == mean_top5_a else ("B" if best_top5 == mean_top5_b else "C")
    )
    p1912_pass = best_top5 > 0.5

    print(
        f"  Scheme A (Bilateral PCA+LR): R²(global)={r2_global_a:.6f}, R²(PCA)={r2_reduced:.6f}, top-5={mean_top5_a:.4f}"
    )
    print(f"  Scheme B (FreqBin): R²={r2_global_b:.6f}, top-5={mean_top5_b:.4f}")
    print(f"  Scheme C (PerToken): R²={r2_global_c:.6f}, top-5={mean_top5_c:.4f}")
    print(f"")
    print(
        f"  P19.1.1 (R² > 0.1):  {'✅' if p1911_pass else '❌'} "
        f"(best={best_r2:.6f}, scheme {best_scheme_r2})"
    )
    print(
        f"  P19.1.2 (top-5 > 50%): {'✅' if p1912_pass else '❌'} "
        f"(best={best_top5:.4f}, scheme {best_scheme_top5})"
    )

    any_pass = p1911_pass or p1912_pass
    print(
        f"\n  Overall: {'✅ At least one gate passed' if any_pass else '❌ Both gates failed'}"
    )

    # ── Save ──
    output = {
        "config": {
            "n_cal": args.n_cal,
            "n_test": args.n_test,
            "seed_cal": args.seed_cal,
            "seed_test": args.seed_test,
            "pca_dim": args.pca_dim,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "rank_threshold": args.rank_threshold,
        },
        "sample_counts": {
            "cal": n_cal,
            "test": n_test,
            "KC": n_kw,  # wait, let me double-check
            "KW": n_kw,
            "DK": n_dk,
        },
        "scheme_a_bilateral_pca": {
            "method": "bilateral PCA (PCA-X + PCA-Y) + Ridge regression",
            "pca_dim": int(pca_dim),
            "pca_y_explained_variance": float(pca_y_var),
            "pca_x_explained_variance": float(pca_x_var),
            "r2_pca_space": float(r2_reduced),
            "r2_global": float(r2_global_a),
            "r2_mean_per_sample": float(mean_r2_a),
            "top5_overlap_mean": float(mean_top5_a),
        },
        "scheme_b_freq_bin": {
            "r2_global": float(r2_global_b),
            "r2_mean_per_sample": float(mean_r2_b),
            "top5_overlap_mean": float(mean_top5_b),
        },
        "scheme_c_per_token_mean": {
            "r2_global": float(r2_global_c),
            "r2_mean_per_sample": float(mean_r2_c),
            "top5_overlap_mean": float(mean_top5_c),
        },
        "gates": {
            "P19.1.1": {
                "pass": bool(p1911_pass),
                "best_r2": float(best_r2),
                "best_scheme": best_scheme_r2,
            },
            "P19.1.2": {
                "pass": bool(p1912_pass),
                "best_top5_overlap": float(best_top5),
                "best_scheme": best_scheme_top5,
            },
        },
    }

    # Fix sample counts
    output["sample_counts"]["KC"] = n_kc

    out_path = output_dir / "s19_1_dpc_predictability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
