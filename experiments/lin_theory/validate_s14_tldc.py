"""Phase 14c: Token-Level Dynamic Contrast (TLDC) Decoding.

Theory: docs/theory-intervention-failure.md Section 14.2

Instead of using a pre-computed direction v (which marginalizes out x), TLDC
uses the logit difference between the detection-peak layer (ℓ*) and the final
layer (L) as a dynamic, per-token, per-sample intervention signal:

    logits_adj ← l_L + β·(l_ℓ* - l_L)

Intuition: ℓ* encodes maximal truth-relevant information (AUROC peak). The
shift from ℓ* to L captures how later layers override this truth signal.
Interpolating back toward ℓ* partially undoes the override.

Gates:
  D1: ℓ* ≠ 0 (known: L20 for 1.7B)
  D2: l_ℓ* gives better y_true rank than l_L on know-wrong subset
  D3: TLDC Δ accuracy > 5% on know-wrong subset
  D5: TLDC Δ accuracy ≥ 0% on don't-know subset (no harm)

Usage:
    python validate_s14_tldc.py --n_calibrate 200 --n_test 50
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

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    extract_h_at_layer,
    greedy_generate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def get_y_true_rank(logits, y_true_id):
    """Rank 0 = highest probability."""
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state at any layer via early exit.

    Applies the same RMSNorm + W_U mapping as the final layer.
    Stays in float16 to avoid OOM from W_U.float() (~1.2 GB).
    """
    dtype = next(ln_final.parameters()).dtype
    device = h.device

    h_f16 = h.to(dtype=dtype)  # [..., d_model] float16
    h_norm = ln_final(h_f16)  # [..., d_model] float16
    # Stay in float16 to avoid allocating ~1.2 GB for W_U.float()
    logits = h_norm @ W_U.to(dtype)  # [..., vocab_size] float16
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def tldc_greedy_generate(
    model,
    tokenizer,
    prompt,
    device,
    layer_early,
    W_U,
    b_U,
    ln_final,
    beta,
    max_new=20,
):
    """Greedy generation with TLDC (Token-Level Dynamic Contrast).

    At each generation step:
      1. Forward pass → capture h at layer_early and final (L27) logits
      2. Compute early-exit logits: l_early = W_U @ ln_final(h_early)
      3. Adjusted logits: l = l_final + beta * (l_early - l_final)
      4. Greedy decode from adjusted logits

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        prompt: string
        device: "cuda" or "cpu"
        layer_early: int — detection peak layer (e.g. 20)
        W_U: [d_model, vocab_size] unembedding matrix
        b_U: [vocab_size] bias or None
        ln_final: RMSNorm module
        beta: float — interpolation weight toward early layer
        max_new: max tokens to generate

    Returns:
        generated_text, initial_raw_logits (from final layer, pre-intervention)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hook_early = f"blocks.{layer_early}.hook_resid_post"

    # Storage for captured hidden states
    captured_early = {}

    def _hook_early(act, hook=None):
        captured_early["h"] = act[:, -1:, :].detach()  # [1, 1, d_model]
        return act

    # ── First token (pre-generation) ──
    with torch.no_grad():
        logits_final = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_early, _hook_early)]
        )

    # TLDC adjustment for first token
    h_early = captured_early["h"]  # [1, 1, d_model]
    l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)  # [1, vocab]
    l_final = logits_final[0, -1:, :].float()  # [1, vocab]

    if l_early.shape[-1] == l_final.shape[-1]:
        # Ensure same vocab size
        logits_adj = l_final + beta * (l_early - l_final)  # [1, vocab]
    else:
        # Safety: if vocab size mismatch, fall back to final
        logits_adj = l_final

    initial_raw_logits = logits_final.detach().clone()
    nid = int(logits_adj.argmax(dim=-1).item())

    gids = [nid]

    # ── Subsequent tokens (autoregressive) ──
    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break

        # Append new token
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

        # Forward pass with early-exit hook
        with torch.no_grad():
            logits_final = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_early, _hook_early)]
            )

        h_early = captured_early["h"]  # [1, 1, d_model]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final = logits_final[0, -1:, :].float()

        if l_early.shape[-1] == l_final.shape[-1]:
            logits_adj = l_final + beta * (l_early - l_final)
        else:
            logits_adj = l_final

        nid = int(logits_adj.argmax(dim=-1).item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    return ans, initial_raw_logits


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 14c: TLDC decoding")
    parser.add_argument(
        "--n_calibrate", type=int, default=200, help="Calibration samples"
    )
    parser.add_argument("--n_test", type=int, default=50, help="Test samples")
    parser.add_argument(
        "--layer_early",
        type=int,
        default=20,
        help="Detection peak layer (ℓ*, early exit layer)",
    )
    parser.add_argument(
        "--rank_threshold", type=int, default=50, help="Rank threshold for knowability"
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Beta values to sweep (higher = more weight to early layer)",
    )
    parser.add_argument("--seed_cal", type=int, default=42)
    parser.add_argument("--seed_test", type=int, default=123)
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
    print("Phase 14c: TLDC (Token-Level Dynamic Contrast)")
    print(f"  Early layer ℓ*: L{args.layer_early}")
    print(f"  Final layer L: L27 (last)")
    print(f"  Betas: {args.betas}")
    print(f"  n_cal={args.n_calibrate}, n_test={args.n_test}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1  # L27
    print(f"  Model: {n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Calibration: classify test samples by knowability ──
    print(f"\n[2/5] Classifying {args.n_test} test samples (seed={args.seed_test})...")
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

        h_L, logits, _, _ = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer_early
        )
        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        test_entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "is_correct": is_correct,
                "subset": subset,
                "prompt": prompt,
                "answers": sample["answers"],
                "question": sample["question"][:80],
                "y_true_id": y_true_id,
            }
        )

    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]

    baseline_rate = sum(1 for e in test_entries if e["is_correct"]) / len(test_entries)
    kw_baseline = sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))
    kc_baseline = sum(1 for e in kc if e["is_correct"]) / max(1, len(kc))
    dk_baseline = sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))

    print(
        f"  Know & Correct: {len(kc)}/{len(test_entries)} (baseline={kc_baseline:.1%})"
    )
    print(
        f"  Know & Wrong:   {len(kw)}/{len(test_entries)} (baseline={kw_baseline:.1%})  ← TARGET"
    )
    print(
        f"  Don't Know:     {len(dk)}/{len(test_entries)} (baseline={dk_baseline:.1%})"
    )
    print(f"  All:            {baseline_rate:.1%}")

    # ── Gate D2: Compare L20 vs L27 y_true rank on know-wrong ──
    print(f"\n[3/5] Gate D2: Early vs Final layer y_true rank comparison...")
    d2_results = {"early_better": 0, "final_better": 0, "same": 0, "details": []}

    for e in tqdm(test_entries, desc="  D2"):
        prompt = e["prompt"]
        y_true_id = e["y_true_id"]

        # Get hidden states at both layers in one forward pass
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        captured = {}

        def _hook_early(act, hook=None):
            captured["h_early"] = act[:, -1:, :].detach()
            return act

        def _hook_final(act, hook=None):
            captured["h_final"] = act[:, -1:, :].detach()
            return act

        hook_early = f"blocks.{args.layer_early}.hook_resid_post"
        hook_final = f"blocks.{final_layer}.hook_resid_post"

        with torch.no_grad():
            _ = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_early, _hook_early), (hook_final, _hook_final)],
            )

        # Compute early-exit logits
        h_early = captured["h_early"]
        h_final = captured["h_final"]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final_exit = compute_early_exit_logits(h_final, ln_final, W_U, b_U)

        rank_early = get_y_true_rank(l_early.unsqueeze(1), y_true_id)
        rank_final = get_y_true_rank(l_final_exit.unsqueeze(1), y_true_id)

        if rank_early < rank_final:
            d2_results["early_better"] += 1
        elif rank_final < rank_early:
            d2_results["final_better"] += 1
        else:
            d2_results["same"] += 1

        d2_results["details"].append(
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank_early": rank_early,
                "rank_final": rank_final,
                "question": e["question"],
            }
        )

    print(f"  Early (L{args.layer_early}) better rank: {d2_results['early_better']}")
    print(f"  Final (L{final_layer}) better rank:    {d2_results['final_better']}")
    print(f"  Same rank:                            {d2_results['same']}")

    # D2 on know-wrong specifically
    kw_details = [d for d in d2_results["details"] if d["subset"] == "know_wrong"]
    kw_early_better = sum(1 for d in kw_details if d["rank_early"] < d["rank_final"])
    kw_final_better = sum(1 for d in kw_details if d["rank_final"] < d["rank_early"])
    print(f"  ── Know-Wrong subset ──")
    print(f"  L{args.layer_early} better: {kw_early_better}/{len(kw_details)}")
    print(f"  L{final_layer} better:      {kw_final_better}/{len(kw_details)}")
    d2_pass = kw_early_better > kw_final_better
    print(f"  D2 (early > final on know-wrong): {'✅' if d2_pass else '❌'}")

    # ── Phase 2: TLDC intervention ──
    print(
        f"\n[4/5] TLDC intervention ({len(args.betas)} betas × {len(test_entries)} samples)..."
    )

    all_results = {
        "baseline_rate": baseline_rate,
        "kw_baseline": kw_baseline,
        "kc_baseline": kc_baseline,
        "dk_baseline": dk_baseline,
        "betas": {},
    }

    for beta in args.betas:
        print(f"\n  ── β = {beta:.1f} ──")
        correct_by_subset = defaultdict(int)
        count_by_subset = defaultdict(int)

        for e in tqdm(test_entries, desc=f"    β={beta:.1f}", leave=False):
            subset = e["subset"]

            gen_text, initial_logits = tldc_greedy_generate(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_early,
                W_U,
                b_U,
                ln_final,
                beta,
            )
            is_correct = check_correct(gen_text, e["answers"], dataset="triviaqa")

            if is_correct:
                correct_by_subset[subset] += 1
                correct_by_subset["all"] += 1
            count_by_subset[subset] += 1
            count_by_subset["all"] += 1

        beta_results = {}
        for s in ["know_wrong", "know_correct", "dont_know", "all"]:
            if count_by_subset[s] > 0:
                rate = correct_by_subset[s] / count_by_subset[s]
                # Delta relative to baseline for this subset
                if s == "know_wrong":
                    bl = kw_baseline
                elif s == "know_correct":
                    bl = kc_baseline
                elif s == "dont_know":
                    bl = dk_baseline
                else:
                    bl = baseline_rate
                delta = rate - bl
            else:
                rate, delta = 0.0, 0.0

            beta_results[s] = {
                "correct": correct_by_subset[s],
                "total": count_by_subset[s],
                "rate": rate,
                "delta": delta,
            }

            # Print subset-specific results
            if s in ["know_wrong", "all"]:
                r = beta_results[s]
                print(f"    {s}: {r['correct']}/{r['total']} (Δ={r['delta']:+.1%})")

        all_results["betas"][f"beta={beta:.1f}"] = beta_results

    # ── Gate verification ──
    print(f"\n[5/5] Gate verification")
    print(f"\n{'=' * 80}")

    # D3: Δ accuracy > 5% on know-wrong
    best_kw_delta = max(
        all_results["betas"][k]["know_wrong"]["delta"] for k in all_results["betas"]
    )
    d3_pass = best_kw_delta > 0.05
    print(
        f"\n  D3 (TLDC Δ > 5% on know-wrong): {'✅' if d3_pass else '❌'} "
        f"(best Δ={best_kw_delta:+.1%})"
    )

    # D5: Δ ≥ 0 on don't-know
    worst_dk_delta = min(
        all_results["betas"][k]["dont_know"]["delta"] for k in all_results["betas"]
    )
    d5_pass = worst_dk_delta >= 0.0
    print(
        f"  D5 (TLDC Δ ≥ 0 on don't-know): {'✅' if d5_pass else '❌'} "
        f"(worst Δ={worst_dk_delta:+.1%})"
    )

    # Summary table
    print(f"\n  ── Summary table ──")
    print(f"  {'β':>6}  {'KW Δ':>8}  {'KC Δ':>8}  {'DK Δ':>8}  {'All Δ':>8}")
    print(f"  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for beta in args.betas:
        key = f"beta={beta:.1f}"
        r = all_results["betas"][key]
        print(
            f"  {beta:>6.1f}  "
            f"{r['know_wrong']['delta']:>+8.1%}  "
            f"{r['know_correct']['delta']:>+8.1%}  "
            f"{r['dont_know']['delta']:>+8.1%}  "
            f"{r['all']['delta']:>+8.1%}"
        )

    # Overall verdict
    n_pass = sum([d2_pass, d3_pass, d5_pass])
    print(
        f"\n  Gate summary: {n_pass}/3 passed (D2={'✅' if d2_pass else '❌'}, "
        f"D3={'✅' if d3_pass else '❌'}, D5={'✅' if d5_pass else '❌'})"
    )

    if d3_pass:
        print(f"  ✅ TLDC shows positive effect on know-wrong → proceed to Phase 14d")
    else:
        print(f"  ❌ TLDC no effect → Phase 14c gate not passed")

    # ── Save ──
    output = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "final_layer": final_layer,
        },
        "test": {
            "n_total": len(test_entries),
            "n_know_correct": len(kc),
            "n_know_wrong": len(kw),
            "n_dont_know": len(dk),
            "baseline_rate": float(baseline_rate),
            "kw_baseline_rate": float(kw_baseline),
            "kc_baseline_rate": float(kc_baseline),
            "dk_baseline_rate": float(dk_baseline),
        },
        "d2": {
            "early_better": d2_results["early_better"],
            "final_better": d2_results["final_better"],
            "same": d2_results["same"],
            "kw_early_better": kw_early_better,
            "kw_final_better": kw_final_better,
            "pass": bool(d2_pass),
        },
        "gates": {
            "D2": {"pass": bool(d2_pass)},
            "D3": {"pass": bool(d3_pass), "best_kw_delta": float(best_kw_delta)},
            "D5": {"pass": bool(d5_pass), "worst_dk_delta": float(worst_dk_delta)},
        },
        "results": all_results,
    }

    out_path = output_dir / "s14_tldc.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
