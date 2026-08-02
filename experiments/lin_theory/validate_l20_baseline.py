"""P0: L20-only early-exit baseline + TLDC re-evaluation with exact match.

Key questions:
  1. What is L20-only greedy decoding accuracy on KW samples?
  2. Does TLDC interpolation beat L20 alone?
  3. How many "corrected" samples are real vs check_correct false positives?

Strategies tested:
  - Baseline: L27 greedy decoding (standard)
  - L20-only: Early-exit greedy from L20 logits (β=1.0 equivalent, but full generation)
  - TLDC: Interpolation l_L + β·(l_L20 - l_L) with β sweep

Correctness evaluated with:
  - Exact match (answer string appears in generation)
  - Token-overlap (original check_correct — for comparison)

Usage:
    python validate_l20_baseline.py --n_test 100 --seed_test 123
"""

import argparse, json, os, re, sys, time
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

from src.data_loader import load_triviaqa, format_prompt
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    extract_h_at_layer,
    greedy_generate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Correctness: EXACT match (replaces fuzzy check_correct)
# ═════════════════════════════════════════════════════════════════════════════


def check_correct_exact(prediction: str, answers: list[str]) -> bool:
    """Exact match: any answer string (case-insensitive) appears in prediction.

    This is conservative — requires the actual answer text to appear verbatim.
    For multi-word answers, checks if the full answer appears as a contiguous
    substring OR all answer words appear as contiguous words in the prediction.
    """
    pred_lower = prediction.strip().lower()
    for ans in answers:
        ans_lower = ans.strip().lower()
        if not ans_lower:
            continue
        # Direct substring match (answer appears verbatim in prediction)
        if ans_lower in pred_lower:
            return True
    return False


def check_correct_fuzzy(prediction: str, answers: list[str]) -> bool:
    """Original check_correct logic — word-overlap based (for comparison)."""
    pred_lower = prediction.strip().lower()
    pred_words = set(pred_lower.split())
    for ans in answers:
        ans_lower = ans.lower().strip()
        ans_words = set(ans_lower.split())
        if ans_words & pred_words:
            return True
        if len(pred_lower) >= 3 and len(ans_lower) >= 3:
            if ans_lower in pred_lower or pred_lower in ans_lower:
                return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# L20-only early-exit greedy generation
# ═════════════════════════════════════════════════════════════════════════════


def l20_greedy_generate(
    model, tokenizer, prompt, device, layer_early, W_U, b_U, ln_final, max_new=20
):
    """Greedy generation using L20 early-exit logits ONLY (no TLDC, no L27).

    At each step, captures h at layer_early, computes early-exit logits,
    and greedily decodes from those logits. Equivalent to TLDC with β=1.0
    but does NOT require a full forward pass to L27 for final logits.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    captured = {}

    def _hook(act, hook=None):
        captured["h"] = act[:, -1:, :].detach()
        return act

    gids = []

    for _ in range(max_new):
        with torch.no_grad():
            _ = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

        h_early = captured["h"]
        l_early = _compute_early_logits(h_early, ln_final, W_U, b_U)

        nid = int(l_early.argmax(dim=-1).item())
        gids.append(nid)

        if nid == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    return tokenizer.decode(gids).strip()


def _compute_early_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state via RMSNorm + W_U (early exit)."""
    dtype = next(ln_final.parameters()).dtype
    device = h.device
    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


# ═════════════════════════════════════════════════════════════════════════════
# TLDC generation (same as validate_s14_tldc.py)
# ═════════════════════════════════════════════════════════════════════════════


def tldc_greedy_generate(
    model, tokenizer, prompt, device, layer_early, W_U, b_U, ln_final, beta, max_new=20
):
    """TLDC greedy generation with per-step interpolation."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    captured = {}

    def _hook(act, hook=None):
        captured["h"] = act[:, -1:, :].detach()
        return act

    gids = []

    for _ in range(max_new):
        with torch.no_grad():
            logits_final = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

        h_early = captured["h"]
        l_early = _compute_early_logits(h_early, ln_final, W_U, b_U)
        l_final = logits_final[0, -1:, :].float()

        if l_early.shape[-1] == l_final.shape[-1]:
            l_combined = l_final + beta * (l_early - l_final)
        else:
            l_combined = l_final

        nid = int(l_combined.argmax(dim=-1).item())
        gids.append(nid)

        if nid == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    return tokenizer.decode(gids).strip()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="P0: L20-only baseline + TLDC exact-match re-evaluation"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.0],
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

    print("=" * 64)
    print("P0: L20-only Baseline + TLDC Exact-Match Re-evaluation")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print(f"  Early layer: L{args.layer_early}")
    print(f"  Betas: {args.betas}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    print(f"  Model: {n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Classify samples ──
    print(f"\n[2/4] Classifying {args.n_test} test samples...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    test_entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        # Get L27 rank and baseline generation
        h_L, logits, tokens_t, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer_early
        )
        sorted_ids = logits[0, -1, :].float().argsort(descending=True)
        rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct_exact = check_correct_exact(gen_text, sample["answers"])
        is_correct_fuzzy = check_correct_fuzzy(gen_text, sample["answers"])

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct_fuzzy else "know_wrong"
        else:
            subset = "dont_know"

        test_entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "subset": subset,
                "prompt": prompt,
                "answers": sample["answers"],
                "question": sample["question"][:100],
                "y_true_id": y_true_id,
                "baseline_gen": gen_text,
                "baseline_correct_exact": is_correct_exact,
                "baseline_correct_fuzzy": is_correct_fuzzy,
            }
        )

    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]

    print(f"\n  Sample breakdown (fuzzy match baseline):")
    print(f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}")
    print(
        f"  Baseline accuracy (fuzzy):  {sum(1 for e in test_entries if e['baseline_correct_fuzzy']) / len(test_entries):.1%}"
    )
    print(
        f"  Baseline accuracy (exact):  {sum(1 for e in test_entries if e['baseline_correct_exact']) / len(test_entries):.1%}"
    )

    # Also show the discrepancy between fuzzy and exact on baseline
    fuzzy_yes_exact_no = sum(
        1
        for e in test_entries
        if e["baseline_correct_fuzzy"] and not e["baseline_correct_exact"]
    )
    print(
        f"  Fuzzy=YES but Exact=NO:    {fuzzy_yes_exact_no}/{len(test_entries)} ← false positives from fuzzy match"
    )

    # ── Run all decoding strategies ──
    print(f"\n[3/4] Running decoding strategies on {len(test_entries)} samples...")

    # Strategies to test
    strategies = {}

    # L20-only (new baseline)
    print("\n  ── Strategy: L20-only early-exit ──")
    l20_results = []
    for e in tqdm(test_entries, desc="    L20-only"):
        gen_text = l20_greedy_generate(
            model,
            tokenizer,
            e["prompt"],
            device,
            args.layer_early,
            W_U,
            b_U,
            ln_final,
            max_new=20,
        )
        l20_results.append(
            {
                "gen_text": gen_text,
                "correct_exact": check_correct_exact(gen_text, e["answers"]),
                "correct_fuzzy": check_correct_fuzzy(gen_text, e["answers"]),
            }
        )
    strategies["L20-only"] = l20_results

    # TLDC with β sweep
    for beta in args.betas:
        print(f"\n  ── Strategy: TLDC β={beta} ──")
        tldc_results = []
        for e in tqdm(test_entries, desc=f"    TLDC β={beta}", leave=False):
            gen_text = tldc_greedy_generate(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_early,
                W_U,
                b_U,
                ln_final,
                beta,
                max_new=20,
            )
            tldc_results.append(
                {
                    "gen_text": gen_text,
                    "correct_exact": check_correct_exact(gen_text, e["answers"]),
                    "correct_fuzzy": check_correct_fuzzy(gen_text, e["answers"]),
                }
            )
        strategies[f"TLDC β={beta}"] = tldc_results

    # ── Evaluate ──
    print(f"\n[4/4] Evaluation\n")
    print(f"{'=' * 100}")

    # Compute per-strategy per-subset stats
    def compute_stats(entries, results, metric="correct_exact"):
        """Compute accuracy per subset for a strategy."""
        stats = {"all": {"correct": 0, "total": 0}}
        for e, r in zip(entries, results):
            subset = e["subset"]
            if subset not in stats:
                stats[subset] = {"correct": 0, "total": 0}
            stats[subset]["correct"] += int(r[metric])
            stats[subset]["total"] += 1
            stats["all"]["correct"] += int(r[metric])
            stats["all"]["total"] += 1
        return stats

    # Baseline stats (from classification phase)
    baseline_stats_exact = {"all": {"correct": 0, "total": len(test_entries)}}
    baseline_stats_fuzzy = {"all": {"correct": 0, "total": len(test_entries)}}
    for e in test_entries:
        subset = e["subset"]
        for stats, metric in [
            (baseline_stats_exact, "baseline_correct_exact"),
            (baseline_stats_fuzzy, "baseline_correct_fuzzy"),
        ]:
            if subset not in stats:
                stats[subset] = {"correct": 0, "total": 0}
            stats[subset]["correct"] += int(e[metric])
            stats[subset]["total"] += 1
            stats["all"]["correct"] += int(e[metric])

    # Print header
    print(
        f"{'Strategy':<20} {'Metric':<8} {'KW':>10} {'KC':>10} {'DK':>10} {'All':>10}"
    )
    print(f"{'─' * 20} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    def format_rate(correct, total):
        if total == 0:
            return "N/A"
        pct = correct / total * 100
        return f"{correct}/{total} ({pct:.1f}%)"

    # Baseline
    for metric_name, baseline_stats in [
        ("Exact", baseline_stats_exact),
        ("Fuzzy", baseline_stats_fuzzy),
    ]:
        kw_s = baseline_stats.get("know_wrong", {"correct": 0, "total": 0})
        kc_s = baseline_stats.get("know_correct", {"correct": 0, "total": 0})
        dk_s = baseline_stats.get("dont_know", {"correct": 0, "total": 0})
        print(
            f"{'Baseline (L27)':<20} {metric_name:<8} "
            f"{format_rate(kw_s['correct'], kw_s['total']):>10} "
            f"{format_rate(kc_s['correct'], kc_s['total']):>10} "
            f"{format_rate(dk_s['correct'], dk_s['total']):>10} "
            f"{format_rate(baseline_stats['all']['correct'], baseline_stats['all']['total']):>10}"
        )

    # Each strategy
    for strategy_name, results in strategies.items():
        for metric_name in ["Exact", "Fuzzy"]:
            metric_key = "correct_exact" if metric_name == "Exact" else "correct_fuzzy"
            stats = compute_stats(test_entries, results, metric=metric_key)
            kw_s = stats.get("know_wrong", {"correct": 0, "total": 0})
            kc_s = stats.get("know_correct", {"correct": 0, "total": 0})
            dk_s = stats.get("dont_know", {"correct": 0, "total": 0})
            print(
                f"{strategy_name:<20} {metric_name:<8} "
                f"{format_rate(kw_s['correct'], kw_s['total']):>10} "
                f"{format_rate(kc_s['correct'], kc_s['total']):>10} "
                f"{format_rate(dk_s['correct'], dk_s['total']):>10} "
                f"{format_rate(stats['all']['correct'], stats['all']['total']):>10}"
            )

    # ── Deep dive: KW sample-by-sample ──
    print(f"\n{'=' * 100}")
    print("KNOW-WRONG SAMPLE-BY-SAMPLE ANALYSIS")
    print(f"{'=' * 100}")

    for e, l20_r in zip(test_entries, strategies["L20-only"]):
        if e["subset"] != "know_wrong":
            continue

        print(f"\n  Sample {e['sample_id']}: {e['question']}")
        print(f"  Answers: {e['answers']}")
        print(f"  Rank(y_true) @ L27: {e['rank']}")
        print(
            f"  Baseline (L27):   [{e['baseline_correct_exact']}|{e['baseline_correct_fuzzy']}] {e['baseline_gen'][:120]}"
        )
        print(
            f"  L20-only:         [{l20_r['correct_exact']}|{l20_r['correct_fuzzy']}] {l20_r['gen_text'][:120]}"
        )

        for beta in args.betas:
            key = f"TLDC β={beta}"
            r = strategies[key][e["sample_id"] - test_entries[0]["sample_id"]]
            # Find the right result for this sample_id
            for idx, entry in enumerate(test_entries):
                if entry["sample_id"] == e["sample_id"]:
                    r = strategies[key][idx]
                    break
            marker = (
                " ← CORRECTED"
                if r["correct_fuzzy"] and not e["baseline_correct_fuzzy"]
                else ""
            )
            if r["correct_exact"] and not e["baseline_correct_exact"]:
                marker += " ★ EXACT CORRECTED"
            if marker:
                print(
                    f"  TLDC β={beta:<5}:  [{r['correct_exact']}|{r['correct_fuzzy']}] {r['gen_text'][:120]}{marker}"
                )

    # ── Summary: is L20-only better than TLDC? ──
    print(f"\n{'=' * 100}")
    print("KEY COMPARISON: L20-only vs TLDC vs Baseline on KW subset")
    print(f"{'=' * 100}")

    kw_indices = [i for i, e in enumerate(test_entries) if e["subset"] == "know_wrong"]

    for metric_name, metric_key in [
        ("Exact", "correct_exact"),
        ("Fuzzy", "correct_fuzzy"),
    ]:
        print(f"\n  ── {metric_name} match ──")

        # Baseline
        bl_correct = sum(
            1 for i in kw_indices if test_entries[i][f"baseline_{metric_key}"]
        )
        print(
            f"  Baseline (L27): {bl_correct}/{len(kw_indices)} ({bl_correct / len(kw_indices) * 100:.1f}%)"
            if kw_indices
            else "  Baseline: N/A"
        )

        # L20-only
        l20_correct = sum(
            1 for i in kw_indices if strategies["L20-only"][i][metric_key]
        )
        print(
            f"  L20-only:       {l20_correct}/{len(kw_indices)} ({l20_correct / len(kw_indices) * 100:.1f}%)"
            if kw_indices
            else "  L20-only: N/A"
        )

        # Best TLDC
        best_beta = None
        best_correct = -1
        for beta in args.betas:
            key = f"TLDC β={beta}"
            n_correct = sum(1 for i in kw_indices if strategies[key][i][metric_key])
            if n_correct > best_correct:
                best_correct = n_correct
                best_beta = beta

        print(
            f"  Best TLDC:      β={best_beta}: {best_correct}/{len(kw_indices)} ({best_correct / len(kw_indices) * 100:.1f}%)"
            if kw_indices
            else "  Best TLDC: N/A"
        )

        # Which strategy actually helped?
        if l20_correct > bl_correct:
            print(f"  → L20-only beats baseline by +{l20_correct - bl_correct} samples")
        if best_correct > bl_correct:
            print(
                f"  → Best TLDC beats baseline by +{best_correct - bl_correct} samples"
            )
        if best_correct > l20_correct:
            print(
                f"  → TLDC beats L20-only by +{best_correct - l20_correct} samples ← SYNERGY"
            )
        elif l20_correct > best_correct:
            print(
                f"  → L20-only beats TLDC by +{l20_correct - best_correct} samples ← TLDC IS JUST INTERPOLATION"
            )
        elif best_correct == l20_correct and best_correct == bl_correct:
            print(f"  → No strategy beats baseline on this subset")

    # ── Save ──
    output = {
        "config": {k: v for k, v in vars(args).items() if k != "output_dir"},
        "sample_count": len(test_entries),
        "kw_count": len(kw),
        "kc_count": len(kc),
        "dk_count": len(dk),
        "per_sample": [],
    }

    for i, e in enumerate(test_entries):
        row = {
            "sample_id": e["sample_id"],
            "question": e["question"],
            "answers": e["answers"],
            "subset": e["subset"],
            "rank": e["rank"],
            "baseline_gen": e["baseline_gen"],
            "baseline_correct_exact": e["baseline_correct_exact"],
            "baseline_correct_fuzzy": e["baseline_correct_fuzzy"],
            "L20_only_gen": strategies["L20-only"][i]["gen_text"],
            "L20_only_correct_exact": strategies["L20-only"][i]["correct_exact"],
            "L20_only_correct_fuzzy": strategies["L20-only"][i]["correct_fuzzy"],
            "tldc": {},
        }
        for beta in args.betas:
            key = f"TLDC β={beta}"
            row["tldc"][f"beta={beta}"] = {
                "gen_text": strategies[key][i]["gen_text"],
                "correct_exact": strategies[key][i]["correct_exact"],
                "correct_fuzzy": strategies[key][i]["correct_fuzzy"],
            }
        output["per_sample"].append(row)

    out_path = output_dir / "p0_l20_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
