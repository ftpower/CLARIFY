"""Phase 19.2: OFDM sub-band decomposition — diagnostic experiment.

Theory: docs/llm-coding-theory.md §11
Plan:   ~/.claude/plans/CLARIFY/phase19-beyond-tldc.md

Tests whether override magnitude differs across token groups (frequency bins
or δ-driven clusters). If some token types experience systematically stronger
override, per-cluster β_c could improve TLDC.

Diagnostic gate:
  P19.2.1: max_c ḡ_c / min_c ḡ_c > 1.5

Usage:
    python diagnose_ofdm_clusters.py --n_test 100

Output:
    experiments/outputs/lin_theory/s19_2_ofdm_clusters.json
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
# Multi-layer logit extraction
# ═════════════════════════════════════════════════════════════════════════════


def extract_multilayer_logits(
    model, tokenizer, prompt, device, layers, final_layer, ln_final, W_U, b_U
):
    """Extract early-exit logits at multiple layers in a single forward pass."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    captured = {}

    def _make_hook(l):
        def _hook(act, hook=None):
            captured[l] = act[:, -1:, :].detach()
            return act

        return _hook

    hooks = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        hooks.append((hook_name, _make_hook(layer)))

    hook_final = f"blocks.{final_layer}.hook_resid_post"

    def _hook_final(act, hook=None):
        captured["final"] = act[:, -1:, :].detach()
        return act

    hooks.append((hook_final, _hook_final))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    dtype = next(ln_final.parameters()).dtype

    def compute_logits(h):
        h_f16 = h.to(dtype=dtype)
        h_norm = ln_final(h_f16)
        logits = h_norm @ W_U.to(dtype)
        if b_U is not None:
            logits = logits + b_U.to(dtype)
        return logits.float().squeeze()  # [vocab]

    logits_dict = {}
    for layer in layers:
        logits_dict[layer] = compute_logits(captured[layer])

    l_final = compute_logits(captured["final"])
    return logits_dict, l_final


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 19.2: OFDM sub-band decomposition diagnosis"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--ref_layer",
        type=int,
        default=20,
        help="Reference layer for TLDC (L20 = peak AUROC)",
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
    print("Phase 19.2: OFDM Sub-band Decomposition — Diagnosis")
    print(f"  Reference layer: L{args.ref_layer}")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/3] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1  # L27
    print(f"  Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Extract logits ──
    print(f"\n[2/3] Extracting logits at L{args.ref_layer} + L{final_layer}...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Logit extraction")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        logits_dict, l_final = extract_multilayer_logits(
            model,
            tokenizer,
            prompt,
            device,
            [args.ref_layer],
            final_layer,
            ln_final,
            W_U,
            b_U,
        )

        # y_true rank (final layer)
        sorted_ids = l_final.float().argsort(descending=True)
        rank = int((sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item())

        # Baseline generation for KC/KW/DK classification
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        # δ = y_ref - y_final
        delta = (logits_dict[args.ref_layer] - l_final).float().cpu()
        # Channel gain g = -δ
        g = -delta

        entry = {
            "sample_id": i,
            "rank": rank,
            "subset": subset,
            "y_true_id": y_true_id,
            "gen_correct": is_correct,
            "question": sample["question"][:100],
            "delta": delta,  # [vocab] — δ(t|x)
            "g": g,  # [vocab] — g(t|x) = -δ(t|x)
            "l_ref": logits_dict[args.ref_layer].float().cpu(),
            "l_final": l_final.float().cpu(),
        }
        entries.append(entry)

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]

    n_kw, n_kc, n_dk = len(kw), len(kc), len(dk)
    print(f"  KC={n_kc}, KW={n_kw}, DK={n_dk}, Total={len(entries)}")

    # ── OFDM analysis ──
    print(f"\n[3/3] OFDM cluster analysis...")
    vocab_size = model.cfg.d_vocab

    # ── Scheme A: Token frequency bins ──
    print(f"\n  ── Scheme A: Frequency bins ──")

    # Get token frequencies from tokenizer vocab
    # For Qwen3, we can approximate frequency by token ID order
    # (smaller IDs ≈ more frequent) or use the tokenizer's vocab directly
    # Actually, we need actual corpus frequencies. Use a heuristic:
    # Bin by token ID range as a proxy (tokenizers sort by frequency roughly)
    # Better: use W_U embedding norm as a proxy for token importance
    W_U_cpu = W_U.float().cpu()  # [d_model, vocab]

    # Bin definitions
    freq_bins = {
        "top-1K": (0, 1000),
        "1K-10K": (1000, 10000),
        "10K-50K": (10000, 50000),
        "50K+": (50000, vocab_size),
    }

    # Per-bin statistics
    # For each sample, compute mean |g(t)| per bin
    bin_g_abs = {bin_name: [] for bin_name in freq_bins}  # [[n_samples] per bin]
    bin_g_mean = {bin_name: [] for bin_name in freq_bins}  # signed mean g(t)

    for e in entries:
        g = e["g"]
        for bin_name, (lo, hi) in freq_bins.items():
            hi = min(hi, vocab_size)
            g_slice = g[lo:hi]
            bin_g_abs[bin_name].append(float(g_slice.abs().mean().item()))
            bin_g_mean[bin_name].append(float(g_slice.mean().item()))

    print(
        f"\n  {'Bin':<12} {'Tokens':>10} {'mean |g|':>12} {'std |g|':>12} "
        f"{'mean g':>12} {'std g':>12}"
    )
    print(f"  {'─' * 12} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    freq_cluster_stats = {}
    for bin_name, (lo, hi) in freq_bins.items():
        hi = min(hi, vocab_size)
        g_abs_arr = np.array(bin_g_abs[bin_name])
        g_mean_arr = np.array(bin_g_mean[bin_name])
        freq_cluster_stats[bin_name] = {
            "token_range": [lo, hi],
            "n_tokens": hi - lo,
            "mean_abs_g": float(g_abs_arr.mean()),
            "std_abs_g": float(g_abs_arr.std()),
            "mean_g": float(g_mean_arr.mean()),
            "std_g": float(g_mean_arr.std()),
        }
        print(
            f"  {bin_name:<12} {hi - lo:>10,} {g_abs_arr.mean():>12.4f} "
            f"{g_abs_arr.std():>12.4f} {g_mean_arr.mean():>12.4f} "
            f"{g_mean_arr.std():>12.4f}"
        )

    # P19.2.1 for Scheme A
    freq_means = [freq_cluster_stats[bn]["mean_abs_g"] for bn in freq_bins]
    freq_max = max(freq_means)
    freq_min = min(freq_means)
    freq_ratio = freq_max / freq_min if freq_min > 0 else float("inf")
    p1921a_pass = freq_ratio > 1.5
    print(
        f"\n  Scheme A max/min |g̅| ratio: {freq_ratio:.4f} "
        f"({'✅' if p1921a_pass else '❌'} threshold=1.5)"
    )

    # ── Scheme B: δ-driven bins ──
    print(f"\n  ── Scheme B: δ-driven bins (by mean |g(t)| across samples) ──")

    # Compute mean |g(t)| across all samples for each token
    g_abs_per_token = torch.zeros(vocab_size)
    g_mean_per_token = torch.zeros(vocab_size)
    n_with_token = torch.zeros(vocab_size)

    for e in entries:
        g = e["g"]
        g_abs_per_token += g.abs()
        g_mean_per_token += g
        n_with_token += 1  # all tokens present in all samples (full logit vector)

    mean_abs_g_per_token = g_abs_per_token / n_with_token  # [vocab]
    mean_g_per_token = g_mean_per_token / n_with_token

    # Sort tokens by mean |g| and bin into quartiles
    sorted_indices = mean_abs_g_per_token.argsort(descending=True)
    quartile_size = vocab_size // 4

    delta_bins = {
        "Q1 (high |g|)": (0, quartile_size),
        "Q2": (quartile_size, 2 * quartile_size),
        "Q3": (2 * quartile_size, 3 * quartile_size),
        "Q4 (low |g|)": (3 * quartile_size, vocab_size),
    }

    print(
        f"\n  {'Bin':<16} {'Tokens':>10} {'mean |g|':>12} {'std |g|':>12} "
        f"{'mean g':>12} {'example tokens':>30}"
    )
    print(f"  {'─' * 16} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 30}")

    delta_cluster_stats = {}
    for bin_name, (lo, hi) in delta_bins.items():
        hi = min(hi, vocab_size)
        bin_indices = sorted_indices[lo:hi]
        bin_g_abs_vals = mean_abs_g_per_token[bin_indices]
        bin_g_mean_vals = mean_g_per_token[bin_indices]

        # Example tokens (top 5 by |g| in this bin)
        top5_in_bin = bin_indices[:5]
        example_tokens = []
        for tid in top5_in_bin:
            try:
                tok_str = tokenizer.decode([int(tid)])
                # Clean up for display
                tok_str = tok_str.replace("\n", "\\n").replace("\t", "\\t")[:15]
                example_tokens.append(tok_str)
            except Exception:
                example_tokens.append(f"<{tid}>")

        delta_cluster_stats[bin_name] = {
            "token_range_in_sorted": [int(lo), int(hi)],
            "n_tokens": int(hi - lo),
            "mean_abs_g": float(bin_g_abs_vals.mean().item()),
            "std_abs_g": float(bin_g_abs_vals.std().item()),
            "mean_g": float(bin_g_mean_vals.mean().item()),
            "std_g": float(bin_g_mean_vals.std().item()),
            "example_tokens": example_tokens,
        }
        print(
            f"  {bin_name:<16} {hi - lo:>10,} {bin_g_abs_vals.mean():>12.4f} "
            f"{bin_g_abs_vals.std():>12.4f} {bin_g_mean_vals.mean():>12.4f} "
            f"{', '.join(example_tokens):>30}"
        )

    # P19.2.1 for Scheme B
    delta_means = [delta_cluster_stats[bn]["mean_abs_g"] for bn in delta_bins]
    delta_max = max(delta_means)
    delta_min = min(delta_means)
    delta_ratio = delta_max / delta_min if delta_min > 0 else float("inf")
    p1921b_pass = delta_ratio > 1.5
    print(
        f"\n  Scheme B max/min |g̅| ratio: {delta_ratio:.4f} "
        f"({'✅' if p1921b_pass else '❌'} threshold=1.5)"
    )

    # ── Also check per-subset clustering ──
    print(f"\n  ── Per-subset |g(t)| profiles ──")
    for subset_name, subset_entries in [("KC", kc), ("KW", kw), ("DK", dk)]:
        if len(subset_entries) == 0:
            continue
        # Mean |g| per token for this subset
        g_abs_subset = torch.zeros(vocab_size)
        for e in subset_entries:
            g_abs_subset += e["g"].abs()
        g_abs_subset /= len(subset_entries)

        # Check if KW has systematically larger |g| in any freq bin
        print(f"  {subset_name} (n={len(subset_entries)}):")
        for bin_name, (lo, hi) in freq_bins.items():
            hi = min(hi, vocab_size)
            g_slice = g_abs_subset[lo:hi]
            print(f"    {bin_name}: mean |g| = {g_slice.mean():.4f}")

    # ── Check g at y_true vs other tokens ──
    print(f"\n  ── |g| at y_true token vs all tokens ──")
    for subset_name, subset_entries in [("KC", kc), ("KW", kw), ("DK", dk)]:
        g_at_yt = []
        g_at_others = []
        for e in subset_entries:
            yt = e["y_true_id"]
            g = e["g"]
            g_at_yt.append(float(g[yt].abs().item()))
            g_at_others.append(float(g.abs().mean().item()))

        g_at_yt = np.array(g_at_yt)
        g_at_others = np.array(g_at_others)
        print(
            f"  {subset_name}: |g(y_true)| = {g_at_yt.mean():.4f} ± {g_at_yt.std():.4f}, "
            f"mean |g(all)| = {g_at_others.mean():.4f} ± {g_at_others.std():.4f}, "
            f"ratio = {g_at_yt.mean() / g_at_others.mean():.2f}"
        )

    # ── Gate summary ──
    print(f"\n{'=' * 60}")
    print(f"Gate Summary")
    print(f"{'=' * 60}")
    print(
        f"  P19.2.1 (Scheme A, freq bins): "
        f"{'✅' if p1921a_pass else '❌'} "
        f"ratio = {freq_ratio:.4f}"
    )
    print(
        f"  P19.2.1 (Scheme B, δ-driven):  "
        f"{'✅' if p1921b_pass else '❌'} "
        f"ratio = {delta_ratio:.4f}"
    )

    any_pass = p1921a_pass or p1921b_pass
    print(f"\n  Overall P19.2.1: {'✅ PASS' if any_pass else '❌ FAIL'}")

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "ref_layer": args.ref_layer,
            "final_layer": final_layer,
            "rank_threshold": args.rank_threshold,
            "vocab_size": vocab_size,
        },
        "sample_counts": {"KC": n_kc, "KW": n_kw, "DK": n_dk, "total": len(entries)},
        "scheme_a_freq_bins": {
            "bins": freq_cluster_stats,
            "max_min_ratio": float(freq_ratio),
            "gate_pass": bool(p1921a_pass),
        },
        "scheme_b_delta_bins": {
            "bins": delta_cluster_stats,
            "max_min_ratio": float(delta_ratio),
            "gate_pass": bool(p1921b_pass),
        },
        "gates": {
            "P19.2.1": {
                "pass": bool(any_pass),
                "scheme_a_ratio": float(freq_ratio),
                "scheme_a_pass": bool(p1921a_pass),
                "scheme_b_ratio": float(delta_ratio),
                "scheme_b_pass": bool(p1921b_pass),
            },
        },
        "per_sample": [
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank": e["rank"],
                "gen_correct": e["gen_correct"],
                "question": e["question"],
            }
            for e in entries
        ],
    }

    out_path = output_dir / "s19_2_ofdm_clusters.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
