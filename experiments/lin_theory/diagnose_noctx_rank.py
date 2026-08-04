"""Phase 18.3: No-context contrast TLDC — diagnostic experiment.

Theory: docs/theory-intervention-failure.md §5
Plan:   ~/.claude/plans/CLARIFY/phase18-tldc-improvements.md

Tests whether removing context improves rank of y_true on KW samples.
If so, the difference l_no_ctx - l_ctx can serve as an alternative to
l_L20 - l_L27 for TLDC-style intervention.

Diagnostic gates:
  P18.3.1: rank_no_ctx < rank_ctx for >50% KW, paired Wilcoxon p < 0.05
  P18.3.2: rank_no_ctx ≤ 5 for >80% KC

Usage:
    python diagnose_noctx_rank.py --n_test 100

Output:
    experiments/outputs/lin_theory/s18_3_noctx.json
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats
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
from common import load_model_and_unembed, get_first_answer_token_id, greedy_generate


# ═════════════════════════════════════════════════════════════════════════════
# Prompt formatting
# ═════════════════════════════════════════════════════════════════════════════


def format_noctx_prompt(question):
    """Minimal prompt without context — just the question."""
    return f"Answer the question with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:"


# ═════════════════════════════════════════════════════════════════════════════
# Forward pass helper
# ═════════════════════════════════════════════════════════════════════════════


def forward_logits(model, tokenizer, prompt, device):
    """Run forward pass and return last-token logits [1, vocab]."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    with torch.no_grad():
        logits = model(tokens)
    return logits[0, -1, :]  # [vocab]


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 18.3: No-context contrast diagnosis"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
        help="Beta sweep for first-token accuracy (no_ctx - ctx contrast)",
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
    print("Phase 18.3: No-Context Contrast TLDC — Diagnosis")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print(f"  Betas: {args.betas}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    print(f"  Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Extract logits with and without context ──
    print(f"\n[2/4] Extracting logits ({args.n_test} samples × 2 passes)...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Forward")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        # Standard prompt with context
        prompt_ctx = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        l_ctx = forward_logits(model, tokenizer, prompt_ctx, device)

        # No-context prompt
        prompt_no_ctx = format_noctx_prompt(sample["question"])
        l_no_ctx = forward_logits(model, tokenizer, prompt_no_ctx, device)

        # Ranks
        sorted_ctx = l_ctx.float().argsort(descending=True)
        sorted_no_ctx = l_no_ctx.float().argsort(descending=True)

        rank_ctx = int((sorted_ctx == y_true_id).nonzero(as_tuple=True)[0].item())
        rank_no_ctx = int((sorted_no_ctx == y_true_id).nonzero(as_tuple=True)[0].item())

        # Baseline generation (with context) for KC/KW/DK classification
        gen_text = greedy_generate(model, tokenizer, prompt_ctx, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank_ctx <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        entries.append(
            {
                "sample_id": i,
                "rank_ctx": rank_ctx,
                "rank_no_ctx": rank_no_ctx,
                "rank_delta": rank_no_ctx - rank_ctx,  # negative = improvement
                "subset": subset,
                "question": sample["question"][:100],
                "answers": sample["answers"],
                "y_true_id": y_true_id,
                "gen_correct": is_correct,
                # Store logits for contrast TLDC sweep
                "l_ctx": l_ctx.float().detach().cpu(),
                "l_no_ctx": l_no_ctx.float().detach().cpu(),
            }
        )

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]

    n_kw, n_kc, n_dk = len(kw), len(kc), len(dk)
    kw_baseline_correct = sum(1 for e in kw if e["gen_correct"])
    kc_baseline_correct = sum(1 for e in kc if e["gen_correct"])
    print(f"  KC={n_kc} (baseline correct: {kc_baseline_correct}/{n_kc})")
    print(f"  KW={n_kw} (baseline correct: {kw_baseline_correct}/{n_kw})")
    print(f"  DK={n_dk}, Total={len(entries)}")

    # ── Analysis ──
    print(f"\n[3/4] Analysis: rank comparison and contrast TLDC...")

    # P18.3.1: rank_no_ctx < rank_ctx on KW
    print(f"\n  ── P18.3.1: rank comparison on KW ──")

    rank_kw_ctx = np.array([e["rank_ctx"] for e in kw])
    rank_kw_no_ctx = np.array([e["rank_no_ctx"] for e in kw])
    rank_improved = np.array([e["rank_no_ctx"] < e["rank_ctx"] for e in kw])
    rank_worsened = np.array([e["rank_no_ctx"] > e["rank_ctx"] for e in kw])
    rank_same = np.array([e["rank_no_ctx"] == e["rank_ctx"] for e in kw])

    pct_improved = rank_improved.mean() * 100 if n_kw > 0 else 0
    pct_worsened = rank_worsened.mean() * 100 if n_kw > 0 else 0
    pct_same = rank_same.mean() * 100 if n_kw > 0 else 0

    print(f"  KW n={n_kw}")
    print(
        f"    rank_ctx: μ={rank_kw_ctx.mean():.1f}, median={np.median(rank_kw_ctx):.0f}"
    )
    print(
        f"    rank_no_ctx: μ={rank_kw_no_ctx.mean():.1f}, median={np.median(rank_kw_no_ctx):.0f}"
    )
    print(f"    Improved: {rank_improved.sum()}/{n_kw} ({pct_improved:.0f}%)")
    print(f"    Worsened: {rank_worsened.sum()}/{n_kw} ({pct_worsened:.0f}%)")
    print(f"    Same:     {rank_same.sum()}/{n_kw} ({pct_same:.0f}%)")

    if n_kw >= 5:
        try:
            w_stat, w_p = sp_stats.wilcoxon(
                rank_kw_no_ctx, rank_kw_ctx, alternative="less"
            )
            print(f"    Wilcoxon (paired, less): W={w_stat:.1f}, p={w_p:.4f}")
        except Exception:
            w_p = 1.0
            print(f"    Wilcoxon: could not compute (all zeros or ties)")

        p1831_pass = (pct_improved > 50) and (w_p < 0.05)
        print(
            f"  P18.3.1 (>50% KW improved + Wilcoxon p<0.05): "
            f"{'✅' if p1831_pass else '❌'} "
            f"(improved={pct_improved:.0f}%, p={w_p:.4f})"
        )
    else:
        p1831_pass = False
        print(f"  P18.3.1: ❌ (insufficient KW samples, n={n_kw})")

    # P18.3.2: rank_no_ctx ≤ 5 for >80% KC
    print(f"\n  ── P18.3.2: KC preservation without context ──")
    rank_kc_ctx = np.array([e["rank_ctx"] for e in kc])
    rank_kc_no_ctx = np.array([e["rank_no_ctx"] for e in kc])
    kc_top5_ctx = (rank_kc_ctx <= 5).mean() * 100 if n_kc > 0 else 0
    kc_top5_no_ctx = (rank_kc_no_ctx <= 5).mean() * 100 if n_kc > 0 else 0

    print(f"  KC n={n_kc}")
    print(f"    rank_ctx ≤ 5: {kc_top5_ctx:.0f}%")
    print(f"    rank_no_ctx ≤ 5: {kc_top5_no_ctx:.0f}%")

    p1832_pass = kc_top5_no_ctx > 80
    print(
        f"  P18.3.2 (>80% KC rank_no_ctx ≤ 5): "
        f"{'✅' if p1832_pass else '❌'} ({kc_top5_no_ctx:.0f}%)"
    )

    # Also show DK stats
    rank_dk_ctx = np.array([e["rank_ctx"] for e in dk])
    rank_dk_no_ctx = np.array([e["rank_no_ctx"] for e in dk])
    dk_improved = (
        (
            np.array([e["rank_no_ctx"] for e in dk])
            < np.array([e["rank_ctx"] for e in dk])
        ).mean()
        * 100
        if n_dk > 0
        else 0
    )
    print(
        f"\n  DK n={n_dk}: improved={dk_improved:.0f}%, "
        f"rank_ctx μ={rank_dk_ctx.mean():.1f}, rank_no_ctx μ={rank_dk_no_ctx.mean():.1f}"
    )

    # ── Contrast TLDC: first-token sweep ──
    print(f"\n  ── Contrast TLDC first-token sweep ──")
    print(f"  y_combined = y_ctx + β · (y_no_ctx - y_ctx)")
    print(
        f"  {'β':>6}  {'KW_acc':>10}  {'KC_acc':>10}  {'DK_acc':>10}  {'All_acc':>10}"
    )
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    # Baseline and standard TLDC for comparison
    # Baseline: first-token from l_ctx
    bl_kw_ft = sum(
        1 for e in kw if int(e["l_ctx"].argmax().item()) == e["y_true_id"]
    ) / max(1, n_kw)
    bl_kc_ft = sum(
        1 for e in kc if int(e["l_ctx"].argmax().item()) == e["y_true_id"]
    ) / max(1, n_kc)
    bl_dk_ft = sum(
        1 for e in dk if int(e["l_ctx"].argmax().item()) == e["y_true_id"]
    ) / max(1, n_dk)
    bl_all_ft = sum(
        1 for e in entries if int(e["l_ctx"].argmax().item()) == e["y_true_id"]
    ) / max(1, len(entries))

    contrast_results = {}

    for beta in args.betas:
        correct_kw = 0
        correct_kc = 0
        correct_dk = 0

        for e in entries:
            l_ctx = e["l_ctx"]
            l_no_ctx = e["l_no_ctx"]
            l_combined = l_ctx + beta * (l_no_ctx - l_ctx)
            nid = int(l_combined.argmax().item())

            if e["subset"] == "know_wrong":
                correct_kw += int(nid == e["y_true_id"])
            elif e["subset"] == "know_correct":
                correct_kc += int(nid == e["y_true_id"])
            elif e["subset"] == "dont_know":
                correct_dk += int(nid == e["y_true_id"])

        total_correct = correct_kw + correct_kc + correct_dk
        kw_acc = correct_kw / max(1, n_kw)
        kc_acc = correct_kc / max(1, n_kc)
        dk_acc = correct_dk / max(1, n_dk)
        all_acc = total_correct / max(1, len(entries))

        print(
            f"  {beta:>6.2f}  {kw_acc:>9.1%}  {kc_acc:>9.1%}  {dk_acc:>9.1%}  {all_acc:>9.1%}"
        )

        contrast_results[f"beta={beta:.2f}"] = {
            "kw_acc": float(kw_acc),
            "kw_correct": correct_kw,
            "kw_total": n_kw,
            "kc_acc": float(kc_acc),
            "kc_correct": correct_kc,
            "kc_total": n_kc,
            "dk_acc": float(dk_acc),
            "dk_correct": correct_dk,
            "dk_total": n_dk,
            "all_acc": float(all_acc),
            "all_correct": total_correct,
            "all_total": len(entries),
            "kw_delta": float(kw_acc - bl_kw_ft),
            "kc_delta": float(kc_acc - bl_kc_ft),
            "dk_delta": float(dk_acc - bl_dk_ft),
            "all_delta": float(all_acc - bl_all_ft),
        }

    print(
        f"\n  Baseline (l_ctx) first-token: KW={bl_kw_ft:.1%}, KC={bl_kc_ft:.1%}, "
        f"DK={bl_dk_ft:.1%}, All={bl_all_ft:.1%}"
    )

    # Best contrast delta
    best_contrast_kw_delta = max(r["kw_delta"] for r in contrast_results.values())
    best_contrast_beta = [
        b
        for b, r in contrast_results.items()
        if r["kw_delta"] == best_contrast_kw_delta
    ][0]
    print(
        f"\n  Best contrast: {best_contrast_beta}, KW Δ={best_contrast_kw_delta:+.1%}"
    )

    # ── Gate Summary ──
    print(f"\n[4/4] Gate Summary")
    print(f"  {'=' * 60}")
    gates = {
        "P18.3.1": {
            "pass": p1831_pass,
            "desc": f"rank_no_ctx < rank_ctx for >50% KW (actual: {pct_improved:.0f}%)",
        },
        "P18.3.2": {
            "pass": p1832_pass,
            "desc": f"rank_no_ctx ≤ 5 for >80% KC (actual: {kc_top5_no_ctx:.0f}%)",
        },
    }
    for gname, ginfo in gates.items():
        status = "✅ PASS" if ginfo["pass"] else "❌ FAIL"
        print(f"  {gname}: {status} — {ginfo['desc']}")

    n_pass = sum(1 for g in gates.values() if g["pass"])
    print(f"\n  {n_pass}/{len(gates)} gates passed")
    if n_pass >= 1:
        print(f"  ✅ At least one gate passed → evaluate full-generation eligibility")
        if best_contrast_kw_delta > 0:
            print(
                f"  → Contrast TLDC shows positive KW Δ={best_contrast_kw_delta:+.1%}"
            )
        else:
            print(f"  → But contrast TLDC KW Δ ≤ 0, may not merit full-generation")
    else:
        print(f"  ❌ All gates failed → skip full-generation")

    # ── Detailed: KW sample-by-sample ──
    print(f"\n  ── KW Sample Details (rank comparison) ──")
    for e in sorted(kw, key=lambda x: x["rank_delta"])[: min(15, n_kw)]:
        delta = e["rank_delta"]
        direction = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        y_true_str = tokenizer.decode([e["y_true_id"]])
        ctx_top1 = tokenizer.decode([int(e["l_ctx"].argmax().item())])
        no_ctx_top1 = tokenizer.decode([int(e["l_no_ctx"].argmax().item())])
        print(
            f"  [#{e['sample_id']}] {direction} Δ={delta:+d} "
            f"(ctx: #{e['rank_ctx']}→'{ctx_top1}', "
            f"no_ctx: #{e['rank_no_ctx']}→'{no_ctx_top1}', "
            f"y_true='{y_true_str}') "
            f"Q: {e['question'][:50]}"
        )

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "rank_threshold": args.rank_threshold,
            "betas": args.betas,
        },
        "sample_counts": {"KC": n_kc, "KW": n_kw, "DK": n_dk, "total": len(entries)},
        "kw_rank_stats": {
            "rank_ctx_mean": float(rank_kw_ctx.mean()) if n_kw > 0 else None,
            "rank_no_ctx_mean": float(rank_kw_no_ctx.mean()) if n_kw > 0 else None,
            "pct_improved": float(pct_improved),
            "pct_worsened": float(pct_worsened),
            "pct_same": float(pct_same),
            "wilcoxon_p": float(w_p) if n_kw >= 5 else None,
        },
        "kc_rank_stats": {
            "pct_top5_ctx": float(kc_top5_ctx),
            "pct_top5_no_ctx": float(kc_top5_no_ctx),
        },
        "gates": {
            "P18.3.1": {"pass": bool(p1831_pass)},
            "P18.3.2": {"pass": bool(p1832_pass)},
        },
        "contrast_tldc": contrast_results,
        "baseline_first_token": {
            "kw_acc": float(bl_kw_ft),
            "kc_acc": float(bl_kc_ft),
            "dk_acc": float(bl_dk_ft),
            "all_acc": float(bl_all_ft),
        },
        "per_sample": [
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank_ctx": e["rank_ctx"],
                "rank_no_ctx": e["rank_no_ctx"],
                "rank_delta": e["rank_delta"],
                "gen_correct": e["gen_correct"],
                "question": e["question"],
            }
            for e in entries
        ],
    }

    out_path = output_dir / "s18_3_noctx.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
