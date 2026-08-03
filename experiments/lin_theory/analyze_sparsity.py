"""Phase 17.1: TLDC diagnostic experiments — sparsity + channel layer distribution.

17.1a: Override sparsity diagnosis
  Hypothesis H11: g(t|x) = l_L20(t) - l_L27(t) is sparse — top-5 tokens
  account for >80% of total ||g||² mass.

  Metrics:
    1. top-5 / top-10 / top-50 proportion of total ||g||²
    2. Gini coefficient (0 = uniform, 1 = concentrated on single token)
    3. KW vs KC vs DK subset comparison

17.1b: Channel gain layer distribution
  Hypothesis H14: ḡ_KW(ℓ) / ḡ_KC(ℓ) is higher in later layers (L24-L27)
  than in early layers (L20-L24), where ḡ(ℓ) = mean of g_ℓ at L27 argmax token.

  Gates:
    C1: ratio monotonically increases from L20 → L27
    C2: ratio[L25:L27] > 2× ratio[L21:L24]

Usage:
    python analyze_sparsity.py --n_calibrate 200 --seed 42
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

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

# Also add lin_theory dir for common.py
_lin_dir = str(Path(__file__).parent)
if _lin_dir not in sys.path:
    sys.path.insert(0, _lin_dir)

from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
)
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def compute_early_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state via RMSNorm + W_U (early exit)."""
    dtype = next(ln_final.parameters()).dtype
    device = h.device
    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def gini_coefficient(values):
    """Compute Gini coefficient of a 1D array.

    Gini = 0 → perfectly uniform
    Gini = 1 → all mass in one element

    Uses the relative mean absolute difference formula.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or np.sum(v) == 0:
        return 0.0
    # Sort and compute cumulative
    v_sorted = np.sort(v)
    n = len(v)
    # G = (2 * sum(i * v_i) / (n * sum(v))) - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * v_sorted)) / (n * np.sum(v_sorted)) - (n + 1.0) / n
    return float(gini)


def classify_sample(model, tokenizer, sample, device, rank_threshold=50):
    """Classify a single sample into KC/KW/DK.

    Returns dict with subset, rank, y_true_id, prompt, or None if y_true_id not found.
    """
    y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
    if y_true_id is None:
        return None

    prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits_final = model(tokens)

    sorted_ids = logits_final[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

    # Greedy generation for correctness check
    nid = int(logits_final[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(19):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits_final = model(tokens)
        nid = int(logits_final[0, -1, :].argmax().item())
        gids.append(nid)
    gen_text = tokenizer.decode(gids).strip()
    is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

    if rank <= rank_threshold:
        subset = "know_correct" if is_correct else "know_wrong"
    else:
        subset = "dont_know"

    return {
        "sample_id": sample.get("id", 0),
        "rank": rank,
        "subset": subset,
        "prompt": prompt,
        "answers": sample["answers"],
        "question": sample["question"][:100],
        "y_true_id": y_true_id,
        "is_correct": is_correct,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 17.1a: Override sparsity
# ═════════════════════════════════════════════════════════════════════════════


def compute_sparsity_stats(l_early, l_final):
    """Compute sparsity metrics for g(t|x) = l_early - l_final.

    Args:
        l_early: [1, 1, vocab] or [1, vocab] — early-layer logits
        l_final: [1, 1, vocab] or [1, vocab] — final-layer logits

    Returns:
        dict with top-5/10/50/100 proportion, Gini, n_nonzero
    """
    # Squeeze to [vocab]
    g = (l_early.float() - l_final.float()).squeeze().detach()
    # Use |g|² (squared deviation weight) for concentration analysis
    g_sq = (g**2).cpu().numpy().astype(np.float64)
    total_mass = np.sum(g_sq)
    if total_mass == 0:
        return {
            "top5_prop": 0.0,
            "top10_prop": 0.0,
            "top50_prop": 0.0,
            "top100_prop": 0.0,
            "top500_prop": 0.0,
            "top1000_prop": 0.0,
            "gini": 0.0,
            "total_mass": 0.0,
            "vocab_size": len(g_sq),
        }

    g_sq_sorted = np.sort(g_sq)[::-1]
    cumsum = np.cumsum(g_sq_sorted)

    def prop(k):
        return float(cumsum[min(k - 1, len(cumsum) - 1)] / total_mass)

    return {
        "top5_prop": prop(5),
        "top10_prop": prop(10),
        "top50_prop": prop(50),
        "top100_prop": prop(100),
        "top500_prop": prop(500),
        "top1000_prop": prop(1000),
        "gini": gini_coefficient(g_sq),
        "total_mass": float(total_mass),
        "vocab_size": len(g_sq),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 17.1b: Channel layer distribution
# ═════════════════════════════════════════════════════════════════════════════


def capture_multilayer_logits(
    model, tokenizer, prompt, device, layers, ln_final, W_U, b_U
):
    """Capture hidden states at multiple layers in one forward pass.

    Returns:
        logits_dict: {layer: logits_tensor[1, 1, vocab]} for each layer
        final_logits: [1, 1, vocab] — actual final layer output logits
    """
    captured = {}

    def make_hook(layer_idx):
        def _hook(act, hook=None):
            captured[layer_idx] = act[:, -1:, :].detach()
            return act

        return _hook

    hooks = [(f"blocks.{l}.hook_resid_post", make_hook(l)) for l in layers]

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        final_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Compute early-exit logits for each captured layer
    logits_dict = {}
    for l in layers:
        if l in captured:
            logits_dict[l] = compute_early_logits(captured[l], ln_final, W_U, b_U)

    return logits_dict, final_logits


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 17.1: TLDC diagnostic experiments"
    )
    parser.add_argument(
        "--n_calibrate",
        type=int,
        default=200,
        help="Number of calibration samples (separate seed from test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for calibration data"
    )
    parser.add_argument(
        "--layer_early",
        type=int,
        default=20,
        help="Reference layer ℓ* for sparsity analysis (default L20)",
    )
    parser.add_argument(
        "--rank_threshold",
        type=int,
        default=50,
        help="Rank threshold for know vs don't-know",
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

    # ── Config ──
    early_layer = args.layer_early  # L20 for sparsity analysis
    # Layers for channel distribution (17.1b): from L10 to L27, with higher density near top
    channel_layers = [10, 15] + list(range(20, 28))  # L10, L15, L20-L27

    print("=" * 72)
    print("Phase 17.1: TLDC Diagnostic Experiments")
    print(f"  17.1a: Override sparsity  (ℓ* = L{early_layer})")
    print(f"  17.1b: Channel layer dist  (ℓ ∈ {channel_layers})")
    print(f"  n_calibrate={args.n_calibrate}, seed={args.seed}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    print(f"  Model: {n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Final layer: L{final_layer}")
    print(f"  Vocab size: {W_U.shape[1]}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Load & classify calibration samples ──
    print(f"\n[2/4] Loading & classifying {args.n_calibrate} calibration samples...")
    calib_samples = load_triviaqa(n_samples=args.n_calibrate, seed=args.seed)
    calib_samples = calib_samples[: args.n_calibrate]

    entries = []
    for i, sample in enumerate(tqdm(calib_samples, desc="  Classify")):
        sample["id"] = i
        entry = classify_sample(model, tokenizer, sample, device, args.rank_threshold)
        if entry is not None:
            entries.append(entry)

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]
    print(f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}, Total={len(entries)}")

    if len(kw) < 5:
        print("  ⚠️  WARNING: < 5 KW samples — sparsity stats will be noisy.")
        print("  Consider increasing --n_calibrate or using a different --seed.")

    # ── Run multi-layer capture on all entries ──
    print(f"\n[3/4] Capturing multi-layer logits for {len(entries)} samples...")

    # Store results per sample
    sparsity_results = []  # 17.1a: per-sample sparsity stats
    channel_results = []  # 17.1b: per-sample per-layer g stats

    for entry in tqdm(entries, desc="  Forward pass"):
        logits_dict, final_logits = capture_multilayer_logits(
            model,
            tokenizer,
            entry["prompt"],
            device,
            channel_layers,
            ln_final,
            W_U,
            b_U,
        )

        l_final = final_logits[0, -1:, :]  # [1, vocab]

        # ── 17.1a: Sparsity at reference layer (L20) ──
        if early_layer in logits_dict:
            l_early = logits_dict[early_layer]  # [1, 1, vocab]
            sp_stats = compute_sparsity_stats(l_early, l_final)
            sp_stats["sample_id"] = entry["sample_id"]
            sp_stats["subset"] = entry["subset"]
            sp_stats["rank"] = entry["rank"]
            sp_stats["question"] = entry["question"]
            sparsity_results.append(sp_stats)

        # ── 17.1b: Per-layer channel gains ──
        # For each layer ℓ, compute g_ℓ = l_ℓ - l_L
        l_final_flat = l_final.float().squeeze().detach()  # [vocab]
        final_argmax_id = int(l_final_flat.argmax().item())
        y_true_id = entry["y_true_id"]

        channel_entry = {
            "sample_id": entry["sample_id"],
            "subset": entry["subset"],
            "rank": entry["rank"],
            "y_true_id": y_true_id,
            "final_argmax_id": final_argmax_id,
            "layers": {},
        }

        for l in channel_layers:
            if l not in logits_dict:
                continue
            l_layer = logits_dict[l].float().squeeze().detach()  # [vocab]
            g_layer = l_layer - l_final_flat  # [vocab]

            # Metrics:
            # 1. g at final argmax token (the "over-hype" signal)
            g_at_argmax = float(g_layer[final_argmax_id].item())
            # 2. g at y_true token (the "truth retention" signal)
            g_at_ytrue = float(g_layer[y_true_id].item())
            # 3. Mean |g| across vocabulary (overall layer divergence)
            g_abs_mean = float(g_layer.abs().mean().item())
            # 4. Std of g across vocabulary
            g_std = float(g_layer.std().item())

            channel_entry["layers"][str(l)] = {
                "g_at_argmax": g_at_argmax,
                "g_at_ytrue": g_at_ytrue,
                "g_abs_mean": g_abs_mean,
                "g_std": g_std,
            }

        channel_results.append(channel_entry)

    # ── 17.1a: Aggregate sparsity stats ──
    print(f"\n[4a/4] ── 17.1a: Override Sparsity Diagnosis ──\n")
    print(f"  Reference layer: L{early_layer}, Final layer: L{final_layer}")
    print(f"  g(t|x) = l_L{early_layer}(t) - l_L{final_layer}(t)")
    print()

    for subset_name, subset_entries in [("KW", kw), ("KC", kc), ("DK", dk)]:
        subset_sp = [s for s in sparsity_results if s["subset"] == subset_name]
        if not subset_sp:
            continue

        top5s = [s["top5_prop"] for s in subset_sp]
        top10s = [s["top10_prop"] for s in subset_sp]
        top50s = [s["top50_prop"] for s in subset_sp]
        top100s = [s["top100_prop"] for s in subset_sp]
        top500s = [s["top500_prop"] for s in subset_sp]
        top1000s = [s["top1000_prop"] for s in subset_sp]
        ginis = [s["gini"] for s in subset_sp]
        masses = [s["total_mass"] for s in subset_sp]

        print(f"  {subset_name} (n={len(subset_sp)}):")
        print(f"    ||g||² mass concentration:")
        print(
            f"      Top-5:    {np.mean(top5s) * 100:.1f}% ± {np.std(top5s) * 100:.1f}%  "
            f"[min={np.min(top5s) * 100:.1f}%, max={np.max(top5s) * 100:.1f}%]"
        )
        print(
            f"      Top-10:   {np.mean(top10s) * 100:.1f}% ± {np.std(top10s) * 100:.1f}%"
        )
        print(
            f"      Top-50:   {np.mean(top50s) * 100:.1f}% ± {np.std(top50s) * 100:.1f}%"
        )
        print(
            f"      Top-100:  {np.mean(top100s) * 100:.1f}% ± {np.std(top100s) * 100:.1f}%"
        )
        print(
            f"      Top-500:  {np.mean(top500s) * 100:.1f}% ± {np.std(top500s) * 100:.1f}%"
        )
        print(
            f"      Top-1000: {np.mean(top1000s) * 100:.1f}% ± {np.std(top1000s) * 100:.1f}%"
        )
        print(
            f"    Gini coefficient: {np.mean(ginis):.4f} ± {np.std(ginis):.4f}  "
            f"[0=uniform, 1=concentrated]"
        )
        print(f"    ||g||² total mass: {np.mean(masses):.1f} ± {np.std(masses):.1f}")
        print()

    # Gate P1
    kw_sp = [s for s in sparsity_results if s["subset"] == "know_wrong"]
    if kw_sp:
        mean_top5 = np.mean([s["top5_prop"] for s in kw_sp])
        print(f"  ── Gate P1: Override sparsity ──")
        print(f"  KW top-5 ||g||² proportion: {mean_top5 * 100:.1f}%")
        if mean_top5 > 0.80:
            print(f"  ✅ P1 PASSED: top-5 > 80% → override HIGHLY SPARSE")
            print(f"     → 17.2 SIC, L1 sparse, and channel probing all benefit")
        elif mean_top5 > 0.50:
            print(f"  ⚠️  P1 MARGINAL: top-5 > 50% but < 80%")
            print(f"     → SIC still promising; L1 benefit reduced")
        else:
            print(f"  ❌ P1 FAILED: top-5 < 50% → override NOT sparse")
            print(f"     → Abandon L1 direction; SIC becomes exploratory")

    # ── KW sample-level detail ──
    if kw_sp:
        print(f"\n  KW sample-level detail:")
        print(
            f"  {'ID':>4} {'Rank':>5} {'Top-5%':>8} {'Top-10%':>8} {'Top-50%':>8} {'Gini':>7} {'Question':>40}"
        )
        print(
            f"  {'─' * 4} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 40}"
        )
        for s in sorted(kw_sp, key=lambda x: x["top5_prop"], reverse=True):
            print(
                f"  {s['sample_id']:>4} {s['rank']:>5} "
                f"{s['top5_prop'] * 100:>7.1f}% {s['top10_prop'] * 100:>7.1f}% "
                f"{s['top50_prop'] * 100:>7.1f}% {s['gini']:>7.4f} "
                f"{s['question'][:40]:>40}"
            )

    # ── 17.1b: Channel layer distribution ──
    print(f"\n[4b/4] ── 17.1b: Channel Gain Layer Distribution ──\n")
    print(f"  Layers analyzed: {channel_layers}")
    print(f"  g_ℓ(t|x) = l_ℓ(t) - l_L{final_layer}(t)")
    print()

    # Aggregate per-layer per-subset
    layer_subset_stats = {}  # {layer: {subset: {metric: [values]}}}
    for l in channel_layers:
        layer_subset_stats[l] = {
            "KW": {"g_at_argmax": [], "g_at_ytrue": [], "g_abs_mean": []},
            "KC": {"g_at_argmax": [], "g_at_ytrue": [], "g_abs_mean": []},
            "DK": {"g_at_argmax": [], "g_at_ytrue": [], "g_abs_mean": []},
        }

    for cr in channel_results:
        subset = {"know_wrong": "KW", "know_correct": "KC", "dont_know": "DK"}[
            cr["subset"]
        ]
        for l_str, l_data in cr["layers"].items():
            l = int(l_str)
            layer_subset_stats[l][subset]["g_at_argmax"].append(l_data["g_at_argmax"])
            layer_subset_stats[l][subset]["g_at_ytrue"].append(l_data["g_at_ytrue"])
            layer_subset_stats[l][subset]["g_abs_mean"].append(l_data["g_abs_mean"])

    # Print per-layer table: mean g at argmax
    print(
        f"  ── Metric: g(t|x) at L27 argmax token (negative = L27 over-hypes relative to ℓ) ──"
    )
    print(
        f"  {'Layer':>6} {'KW mean':>10} {'KW std':>8} {'KC mean':>10} {'KC std':>8} "
        f"{'DK mean':>10} {'DK std':>8} {'|KW|/|KC|':>10}"
    )
    print(
        f"  {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 10}"
    )

    ratios_at_argmax = {}

    for l in channel_layers:
        kw_vals = layer_subset_stats[l]["KW"]["g_at_argmax"]
        kc_vals = layer_subset_stats[l]["KC"]["g_at_argmax"]
        dk_vals = layer_subset_stats[l]["DK"]["g_at_argmax"]

        kw_mean = np.mean(kw_vals) if kw_vals else 0
        kc_mean = np.mean(kc_vals) if kc_vals else 0
        dk_mean = np.mean(dk_vals) if dk_vals else 0
        kw_std = np.std(kw_vals) if kw_vals else 0
        kc_std = np.std(kc_vals) if kc_vals else 0
        dk_std = np.std(dk_vals) if dk_vals else 0

        # Ratio of absolute means: |g_KW| / |g_KC|
        ratio = abs(kw_mean) / (abs(kc_mean) + 1e-10)
        ratios_at_argmax[l] = ratio

        print(
            f"  L{l:<5} {kw_mean:>+10.2f} {kw_std:>8.2f} "
            f"{kc_mean:>+10.2f} {kc_std:>8.2f} "
            f"{dk_mean:>+10.2f} {dk_std:>8.2f} "
            f"{ratio:>10.3f}"
        )

    # Also print g at y_true token
    print(f"\n  ── Metric: g(t|x) at y_true token ──")
    print(
        f"  {'Layer':>6} {'KW mean':>10} {'KW std':>8} {'KC mean':>10} {'KC std':>8} "
        f"{'DK mean':>10} {'DK std':>8}"
    )
    print(f"  {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 8}")

    for l in channel_layers:
        kw_vals = layer_subset_stats[l]["KW"]["g_at_ytrue"]
        kc_vals = layer_subset_stats[l]["KC"]["g_at_ytrue"]
        dk_vals = layer_subset_stats[l]["DK"]["g_at_ytrue"]

        kw_mean = np.mean(kw_vals) if kw_vals else 0
        kc_mean = np.mean(kc_vals) if kc_vals else 0
        dk_mean = np.mean(dk_vals) if dk_vals else 0
        kw_std = np.std(kw_vals) if kw_vals else 0
        kc_std = np.std(kc_vals) if kc_vals else 0
        dk_std = np.std(dk_vals) if dk_vals else 0

        print(
            f"  L{l:<5} {kw_mean:>+10.2f} {kw_std:>8.2f} "
            f"{kc_mean:>+10.2f} {kc_std:>8.2f} "
            f"{dk_mean:>+10.2f} {dk_std:>8.2f}"
        )

    # Gate C1 & C2
    print(f"\n  ── Gate C1/C2: Ratio |g_KW| / |g_KC| at argmax ──")
    early_layers = [l for l in channel_layers if 20 <= l <= 24]
    late_layers = [l for l in channel_layers if 25 <= l <= 27]

    early_ratios = [ratios_at_argmax[l] for l in early_layers if l in ratios_at_argmax]
    late_ratios = [ratios_at_argmax[l] for l in late_layers if l in ratios_at_argmax]

    if early_ratios and late_ratios:
        early_mean = np.mean(early_ratios)
        late_mean = np.mean(late_ratios)
        print(f"  Early layers L20-L24: mean ratio = {early_mean:.3f}")
        print(f"  Late layers  L25-L27: mean ratio = {late_mean:.3f}")

        # Check monotonicity (C1): Spearman rank correlation
        from scipy.stats import spearmanr

        layers_in_range = [
            l for l in channel_layers if 20 <= l <= 27 and l in ratios_at_argmax
        ]
        ratio_list = [ratios_at_argmax[l] for l in layers_in_range]
        if len(ratio_list) >= 3:
            rho, pval = spearmanr(layers_in_range, ratio_list)
            print(f"  Spearman ρ (L20→L27): {rho:.3f} (p={pval:.4f})")
            if rho > 0.7 and pval < 0.05:
                print(f"  ✅ C1 PASSED: Ratio monotonically increases L20→L27")
            elif rho > 0:
                print(f"  ⚠️  C1 WEAK: Positive trend but ρ={rho:.3f}")
            else:
                print(f"  ❌ C1 FAILED: No monotonic increase")
        else:
            print(f"  ⚠️  C1: Insufficient layers for Spearman test")

        # Check C2
        print(
            f"  Ratio[L25:L27] / Ratio[L21:L24] = {late_mean / (early_mean + 1e-10):.2f}x"
        )
        if late_mean > 2.0 * early_mean:
            print(f"  ✅ C2 PASSED: Late-layer ratio > 2× early-layer ratio")
        else:
            print(f"  ❌ C2 FAILED: Late-layer ratio ≤ 2× early-layer ratio")

    # ── Save results ──
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "seed": args.seed,
            "layer_early": early_layer,
            "final_layer": final_layer,
            "channel_layers": channel_layers,
            "rank_threshold": args.rank_threshold,
        },
        "sample_counts": {"KC": len(kc), "KW": len(kw), "DK": len(dk)},
        "sparsity_17_1a": {
            "per_sample": sparsity_results,
            "summary": {},
        },
        "channel_17_1b": {
            "per_layer_stats": {},
            "gates": {},
        },
    }

    # Summary stats for 17.1a
    for subset_name in ["know_wrong", "know_correct", "dont_know"]:
        subset_sp = [s for s in sparsity_results if s["subset"] == subset_name]
        if subset_sp:
            output["sparsity_17_1a"]["summary"][subset_name] = {
                "n": len(subset_sp),
                "top5_prop_mean": float(np.mean([s["top5_prop"] for s in subset_sp])),
                "top5_prop_std": float(np.std([s["top5_prop"] for s in subset_sp])),
                "top10_prop_mean": float(np.mean([s["top10_prop"] for s in subset_sp])),
                "top50_prop_mean": float(np.mean([s["top50_prop"] for s in subset_sp])),
                "top100_prop_mean": float(
                    np.mean([s["top100_prop"] for s in subset_sp])
                ),
                "gini_mean": float(np.mean([s["gini"] for s in subset_sp])),
                "gini_std": float(np.std([s["gini"] for s in subset_sp])),
                "total_mass_mean": float(np.mean([s["total_mass"] for s in subset_sp])),
            }

    # Summary stats for 17.1b
    for l in channel_layers:
        output["channel_17_1b"]["per_layer_stats"][str(l)] = {}
        for subset_key in ["KW", "KC", "DK"]:
            stats = layer_subset_stats[l][subset_key]
            output["channel_17_1b"]["per_layer_stats"][str(l)][subset_key] = {
                "g_at_argmax_mean": float(np.mean(stats["g_at_argmax"]))
                if stats["g_at_argmax"]
                else None,
                "g_at_argmax_std": float(np.std(stats["g_at_argmax"]))
                if stats["g_at_argmax"]
                else None,
                "g_at_ytrue_mean": float(np.mean(stats["g_at_ytrue"]))
                if stats["g_at_ytrue"]
                else None,
                "g_abs_mean_mean": float(np.mean(stats["g_abs_mean"]))
                if stats["g_abs_mean"]
                else None,
            }

    # Gate results
    kw_sp = [s for s in sparsity_results if s["subset"] == "know_wrong"]
    p1_passed = bool(kw_sp and np.mean([s["top5_prop"] for s in kw_sp]) > 0.80)
    output["channel_17_1b"]["gates"] = {
        "P1": {"passed": p1_passed, "description": "KW top-5 > 80% ||g||² mass"},
        "C1": {"passed": None, "description": "Ratio monotonically increases L20→L27"},
        "C2": {"passed": None, "description": "Ratio[L25:L27] > 2× Ratio[L21:L24]"},
    }

    out_path = output_dir / "s17_1_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    # ── Final summary ──
    print(f"\n{'=' * 72}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 72}")
    print(f"  17.1a Override sparsity  → {'PASS' if p1_passed else 'CHECK OUTPUT'}")
    print(f"  17.1b Channel layers     → see gate results above")
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
