"""Phase 15.2b: TLDC per-token analysis of corrected KW samples.

For the 2 KW samples corrected by TLDC at β=0.10 (seed=123), this script:
  1. Captures l_L20, l_L27, l_combined at EVERY generation step
  2. Prints top-3 tokens for each
  3. Identifies which step's TLDC adjustment flipped the argmax
  4. Computes (l_L20 - l_L27) on y_true token vs distractor
  5. Distinguishes: "push up y_true" vs "push down distractor"

Usage:
    python analyze_tldc_per_token.py --seed_test 123 --beta 0.10 --n_test 50
"""

import argparse, json, os, sys
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
)

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state via early exit (RMSNorm + W_U)."""
    dtype = next(ln_final.parameters()).dtype
    device = h.device
    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def get_topk_info(logits, tokenizer, k=5):
    """Return top-k token strings + logit values from logits tensor [..., vocab]."""
    # Squeeze all leading dims — logits can be [1, vocab], [1, 1, vocab], etc.
    flat = logits.float().squeeze()
    # Now flat should be [vocab]
    vals, idxs = torch.topk(flat, k)
    result = []
    for i in range(k):
        tid = int(idxs[i].item())
        token_str = tokenizer.decode([tid])
        result.append((tid, token_str, float(vals[i].item())))
    return result


def classify_samples(
    model, tokenizer, test_samples, device, layer_early, rank_threshold
):
    """Classify samples into KC, KW, DK and return entries with metadata."""
    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        # Get hidden state and logits at early layer
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        hook_early = f"blocks.{layer_early}.hook_resid_post"
        captured = {}

        def _hook(act, hook=None):
            captured["h"] = act[:, -1:, :].detach()
            return act

        with torch.no_grad():
            logits_final = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

        # Get rank from final logits
        sorted_ids = logits_final[0, -1, :].float().argsort(descending=True)
        rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

        # Baseline generation (no intervention)
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

        entries.append(
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

    return entries


# ═════════════════════════════════════════════════════════════════════════════
# Main: Per-token TLDC analysis
# ═════════════════════════════════════════════════════════════════════════════


def analyze_tldc_per_token(
    model,
    tokenizer,
    entry,
    device,
    layer_early,
    final_layer,
    W_U,
    b_U,
    ln_final,
    beta,
    max_new=20,
    print_tokens=True,
):
    """Run TLDC generation with full per-step logit capture.

    Returns a dict with per-step analysis suitable for printing.
    """
    prompt = entry["prompt"]
    y_true_id = entry["y_true_id"]
    y_true_str = tokenizer.decode([y_true_id])

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    hook_final_name = f"blocks.{final_layer}.hook_resid_post"

    captured = {}

    def _hook_early(act, hook=None):
        captured["h_early"] = act[:, -1:, :].detach()
        return act

    def _hook_final(act, hook=None):
        captured["h_final"] = act[:, -1:, :].detach()
        return act

    # ── Step 0 (pre-generation): capture all three logit spaces ──
    with torch.no_grad():
        _ = model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (hook_early, _hook_early),
                (hook_final_name, _hook_final),
            ],
        )

    h_early = captured["h_early"]
    h_final = captured["h_final"]

    l_early_0 = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
    l_final_0 = compute_early_exit_logits(h_final, ln_final, W_U, b_U)

    if l_early_0.shape[-1] != l_final_0.shape[-1]:
        l_combined_0 = l_final_0
    else:
        l_combined_0 = l_final_0.float() + beta * (
            l_early_0.float() - l_final_0.float()
        )

    steps_log = []
    nid = int(l_combined_0.argmax(dim=-1).item())
    gids = [nid]

    # Record step 0
    steps_log.append(
        {
            "step": 0,
            "l_early": l_early_0.detach().clone(),
            "l_final": l_final_0.detach().clone(),
            "l_combined": l_combined_0.detach().clone(),
            "chosen_id": nid,
        }
    )

    # ── Subsequent steps (autoregressive) ──
    for step in range(1, max_new):
        if nid == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

        with torch.no_grad():
            _ = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (hook_early, _hook_early),
                    (hook_final_name, _hook_final),
                ],
            )

        h_early = captured["h_early"]
        h_final = captured["h_final"]

        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final = compute_early_exit_logits(h_final, ln_final, W_U, b_U)

        if l_early.shape[-1] != l_final.shape[-1]:
            l_combined = l_final
        else:
            l_combined = l_final.float() + beta * (l_early.float() - l_final.float())

        nid = int(l_combined.argmax(dim=-1).item())
        gids.append(nid)

        steps_log.append(
            {
                "step": step,
                "l_early": l_early.detach().clone(),
                "l_final": l_final.detach().clone(),
                "l_combined": l_combined.detach().clone(),
                "chosen_id": nid,
            }
        )

    gen_text = tokenizer.decode(gids).strip()

    # ── Also get baseline generation (l_final only, no TLDC) ──
    tokens_bl = model.to_tokens(prompt, prepend_bos=True)
    if tokens_bl.shape[1] > 1024:
        tokens_bl = tokens_bl[:, :1024]

    with torch.no_grad():
        logits_bl = model(tokens_bl)
    nid_bl = int(logits_bl[0, -1, :].argmax().item())
    gids_bl = [nid_bl]
    for _ in range(max_new - 1):
        if nid_bl == tokenizer.eos_token_id:
            break
        tokens_bl = torch.cat(
            [tokens_bl, torch.tensor([[nid_bl]], device=device)], dim=1
        )
        with torch.no_grad():
            logits_bl = model(tokens_bl)
        nid_bl = int(logits_bl[0, -1, :].argmax().item())
        gids_bl.append(nid_bl)
    baseline_text = tokenizer.decode(gids_bl).strip()

    return {
        "gen_text": gen_text,
        "baseline_text": baseline_text,
        "gids": gids,
        "gids_bl": gids_bl,
        "steps": steps_log,
    }


def print_per_step_analysis(result, entry, tokenizer, beta):
    """Print detailed per-step analysis for one sample."""
    y_true_id = entry["y_true_id"]
    y_true_str = tokenizer.decode([y_true_id])
    question = entry["question"]
    answers = entry["answers"]

    print(f"\n{'=' * 80}")
    print(f"Question: {question}")
    print(f"Ground truth: {answers}")
    print(f"y_true first token: '{y_true_str}' (id={y_true_id})")
    print(f"\nBaseline (no TLDC): {result['baseline_text']}")
    print(f"TLDC (β={beta}):      {result['gen_text']}")

    steps = result["steps"]
    baseline_gids = result["gids_bl"]
    tldc_gids = result["gids"]

    # Print per-step top-3 comparison
    print(f"\n{'─' * 80}")
    print(
        f"{'Step':>5} {'Source':>10} {'Rank':>5} {'Token ID':>8} {'Token':>20} {'Logit':>10} {'Prob':>10}"
    )
    print(f"{'─' * 80}")

    for s in steps:
        step = s["step"]
        chosen_tldc = tldc_gids[step] if step < len(tldc_gids) else None
        chosen_bl = baseline_gids[step] if step < len(baseline_gids) else None

        # Compute probabilities for each source (squeeze all leading dims)
        l_early = s["l_early"].float().squeeze()
        l_final = s["l_final"].float().squeeze()
        l_combined = s["l_combined"].float().squeeze()

        probs_early = torch.softmax(l_early, dim=-1)
        probs_final = torch.softmax(l_final, dim=-1)
        probs_combined = torch.softmax(l_combined, dim=-1)

        # Get top-3 from combined (what we actually decode from)
        topk_combined = get_topk_info(s["l_combined"], tokenizer, k=3)

        for rank, (tid, tok_str, logit_val) in enumerate(topk_combined):
            # Get logit/prob from all three sources
            logit_e = float(l_early[tid].item())
            logit_f = float(l_final[tid].item())
            prob_e = float(probs_early[tid].item())
            prob_f = float(probs_final[tid].item())
            prob_c = float(probs_combined[tid].item())

            marker = ""
            if tid == chosen_tldc:
                marker = " ← TLDC ARGMAX"
            if tid == y_true_id:
                marker += " ★ y_true"

            print(
                f"{step:>5} {'early (L20)':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {logit_e:>10.2f} {prob_e:>10.4f}{marker}"
            )
            if rank == 0:
                print(
                    f"{'':>5} {'final (L27)':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {logit_f:>10.2f} {prob_f:>10.4f}"
                )
                print(
                    f"{'':>5} {'combined':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {logit_val:>10.2f} {prob_c:>10.4f}"
                )

        # Show y_true info if not in top-3
        if y_true_id not in [t[0] for t in topk_combined]:
            logit_e_yt = float(l_early[y_true_id].item())
            logit_f_yt = float(l_final[y_true_id].item())
            logit_c_yt = float(l_combined[y_true_id].item())
            prob_e_yt = float(probs_early[y_true_id].item())
            prob_f_yt = float(probs_final[y_true_id].item())
            prob_c_yt = float(probs_combined[y_true_id].item())

            rank_early = (
                (l_early.argsort(descending=True) == y_true_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            rank_final = (
                (l_final.argsort(descending=True) == y_true_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )
            rank_combined = (
                (l_combined.argsort(descending=True) == y_true_id)
                .nonzero(as_tuple=True)[0]
                .item()
            )

            print(
                f"{step:>5} {'early (L20)':>10} {rank_early + 1:>5} {y_true_id:>8} {y_true_str:>20} {logit_e_yt:>10.2f} {prob_e_yt:>10.4f} ★ y_true (off-list)"
            )
            print(
                f"{'':>5} {'final (L27)':>10} {rank_final + 1:>5} {y_true_id:>8} {y_true_str:>20} {logit_f_yt:>10.2f} {prob_f_yt:>10.4f}"
            )
            print(
                f"{'':>5} {'combined':>10} {rank_combined + 1:>5} {y_true_id:>8} {y_true_str:>20} {logit_c_yt:>10.2f} {prob_c_yt:>10.4f}"
            )

        # Divider between steps
        if step < len(steps) - 1:
            print(f"{'─' * 80}")

    # ── Summary: TLDC effect direction ──
    print(f"\n{'=' * 80}")
    print("SUMMARY: TLDC Effect Direction")
    print(f"{'=' * 80}")

    for s in steps:
        step = s["step"]
        l_early = s["l_early"].float().squeeze()
        l_final = s["l_final"].float().squeeze()
        l_combined = s["l_combined"].float().squeeze()

        # TLDC signal: delta = l_early - l_final
        delta = l_early - l_final

        # Effect on y_true
        delta_yt = float(delta[y_true_id].item())
        logit_early_yt = float(l_early[y_true_id].item())
        logit_final_yt = float(l_final[y_true_id].item())
        logit_combined_yt = float(l_combined[y_true_id].item())

        # Effect on the argmax of final layer (the "distractor")
        final_argmax_id = int(l_final.argmax().item())
        final_argmax_str = tokenizer.decode([final_argmax_id])
        delta_distractor = float(delta[final_argmax_id].item())

        # Effect on the argmax of combined
        combined_argmax_id = int(l_combined.argmax().item())
        combined_argmax_str = tokenizer.decode([combined_argmax_id])

        # Who benefits from TLDC?
        topk_combined_ids = [
            t[0] for t in get_topk_info(s["l_combined"], tokenizer, k=5)
        ]
        topk_final_ids = [t[0] for t in get_topk_info(s["l_final"], tokenizer, k=5)]

        print(f"\n  Step {step}:")
        print(f"    L20  argmax: '{tokenizer.decode([int(l_early.argmax().item())])}'")
        print(f"    L27  argmax: '{final_argmax_str}'")
        print(f"    TLDC argmax: '{combined_argmax_str}'")

        print(f"    TLDC delta on y_true ('{y_true_str}'):      {delta_yt:+.2f}")
        print(
            f"      → L20 logit={logit_early_yt:.2f}, L27 logit={logit_final_yt:.2f}, combined={logit_combined_yt:.2f}"
        )

        if final_argmax_id != y_true_id:
            print(
                f"    TLDC delta on distractor ('{final_argmax_str}'): {delta_distractor:+.2f}"
            )
            logit_e_d = float(l_early[final_argmax_id].item())
            logit_f_d = float(l_final[final_argmax_id].item())
            logit_c_d = float(l_combined[final_argmax_id].item())
            print(
                f"      → L20 logit={logit_e_d:.2f}, L27 logit={logit_f_d:.2f}, combined={logit_c_d:.2f}"
            )

        # Determine: push y_true up or push distractor down?
        if abs(delta_yt) > 0.01 or abs(delta_distractor) > 0.01:
            if delta_yt > 0 and final_argmax_id != y_true_id:
                print(f"    → TLDC PUSHES UP y_true (+{delta_yt:.2f})")
            if delta_distractor < 0 and final_argmax_id != y_true_id:
                print(f"    → TLDC PUSHES DOWN distractor ({delta_distractor:+.2f})")

        # Check if TLDC flipped something
        if final_argmax_id != combined_argmax_id:
            print(
                f"    🔄 ARGMAX FLIP: L27='{final_argmax_str}' → TLDC='{combined_argmax_str}'"
            )

    # ── Net effect verdict ──
    print(f"\n{'─' * 80}")
    print("NET EFFECT (across all steps):")
    yt_deltas = []
    distractor_deltas = []
    for s in steps:
        l_early = s["l_early"].float().squeeze()
        l_final = s["l_final"].float().squeeze()
        delta = l_early - l_final
        yt_deltas.append(float(delta[y_true_id].item()))
        final_am = int(l_final.argmax().item())
        if final_am != y_true_id:
            distractor_deltas.append(float(delta[final_am].item()))

    mean_yt_delta = np.mean(yt_deltas)
    mean_dist_delta = np.mean(distractor_deltas) if distractor_deltas else 0.0
    print(f"  Mean TLDC delta on y_true:      {mean_yt_delta:+.2f}")
    print(f"  Mean TLDC delta on distractor:   {mean_dist_delta:+.2f}")
    if mean_yt_delta > 0 and mean_dist_delta < 0:
        print(f"  → TLDC BOTH pushes up y_true AND pushes down distractor")
    elif mean_yt_delta > 0:
        print(f"  → TLDC primarily PUSHES UP y_true")
    elif mean_dist_delta < 0:
        print(f"  → TLDC primarily PUSHES DOWN distractor")

    # Compare baseline vs TLDC generation sequences
    print(
        f"\n  Baseline token sequence: {[tokenizer.decode([g]) for g in result['gids_bl']]}"
    )
    print(
        f"  TLDC token sequence:     {[tokenizer.decode([g]) for g in result['gids']]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 15.2b: TLDC per-token analysis")
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--rank_threshold", type=int, default=50)
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
    print("Phase 15.2b: TLDC Per-Token Analysis")
    print(f"  β={args.beta}, seed={args.seed_test}, n={args.n_test}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1
    print(f"  Model: {model.cfg.n_layers} layers")

    # ── Classify samples ──
    print(f"\n[2/4] Classifying test samples (seed={args.seed_test})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]
    entries = classify_samples(
        model, tokenizer, test_samples, device, args.layer_early, args.rank_threshold
    )

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]
    print(f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}")

    # ── Run TLDC with full per-token capture on ALL samples ──
    print(f"\n[3/4] Running TLDC β={args.beta} with per-token capture...")

    all_results = []
    for e in tqdm(entries, desc="  TLDC"):
        result = analyze_tldc_per_token(
            model,
            tokenizer,
            e,
            device,
            args.layer_early,
            final_layer,
            W_U,
            b_U,
            ln_final,
            args.beta,
            max_new=20,
            print_tokens=False,
        )
        is_correct = check_correct(result["gen_text"], e["answers"], dataset="triviaqa")
        result["is_correct"] = is_correct
        result["subset"] = e["subset"]
        result["question"] = e["question"]
        result["answers"] = e["answers"]
        result["y_true_id"] = e["y_true_id"]
        result["sample_id"] = e["sample_id"]
        all_results.append(result)

    # ── Find corrected KW samples ──
    print(f"\n[4/4] Deep-dive on corrected KW samples...")

    corrected_kw = [
        r for r in all_results if r["subset"] == "know_wrong" and r["is_correct"]
    ]

    # Also find KC samples to check if TLDC would have broken them
    broken_kc = [
        r for r in all_results if r["subset"] == "know_correct" and not r["is_correct"]
    ]

    print(f"\n  Corrected KW: {len(corrected_kw)}/{len(kw)}")
    print(f"  Broken KC:    {len(broken_kc)}/{len(kc)}")

    # Also print overall stats
    for subset_name, subset_list in [
        ("know_wrong", kw),
        ("know_correct", kc),
        ("dont_know", dk),
    ]:
        subset_results = [r for r in all_results if r["subset"] == subset_name]
        n_correct = sum(1 for r in subset_results if r["is_correct"])
        print(f"  {subset_name}: {n_correct}/{len(subset_list)}")

    # ── Per-sample deep analysis ──
    for r in corrected_kw:
        print_per_step_analysis(r, r, tokenizer, args.beta)

    # Also print broken KC if any
    for r in broken_kc:
        print_per_step_analysis(r, r, tokenizer, args.beta)

    # ── Aggregate summary ──
    print(f"\n{'=' * 80}")
    print("AGGREGATE MECHANISM SUMMARY")
    print(f"{'=' * 80}")

    # For all KW samples, compute mean TLDC delta on y_true vs distractor
    all_yt_deltas = []
    all_dist_deltas = []
    for r in all_results:
        if r["subset"] == "know_wrong":
            for s in r["steps"]:
                l_early = s["l_early"].float().squeeze()
                l_final = s["l_final"].float().squeeze()
                delta = l_early - l_final
                all_yt_deltas.append(float(delta[r["y_true_id"]].item()))
                final_am = int(l_final.argmax().item())
                if final_am != r["y_true_id"]:
                    all_dist_deltas.append(float(delta[final_am].item()))

    print(f"\n  All KW samples (n={len(kw)}):")
    print(f"  Mean TLDC delta on y_true:       {np.mean(all_yt_deltas):+.2f}")
    print(f"  Mean TLDC delta on distractor:    {np.mean(all_dist_deltas):+.2f}")

    if corrected_kw:
        corrected_yt = []
        corrected_dist = []
        for r in corrected_kw:
            for s in r["steps"]:
                l_early = s["l_early"].float().squeeze()
                l_final = s["l_final"].float().squeeze()
                delta = l_early - l_final
                corrected_yt.append(float(delta[r["y_true_id"]].item()))
                final_am = int(l_final.argmax().item())
                if final_am != r["y_true_id"]:
                    corrected_dist.append(float(delta[final_am].item()))

        print(f"\n  Corrected KW only (n={len(corrected_kw)}):")
        print(f"  Mean TLDC delta on y_true:       {np.mean(corrected_yt):+.2f}")
        print(f"  Mean TLDC delta on distractor:    {np.mean(corrected_dist):+.2f}")

        if np.mean(corrected_yt) > 0 and np.mean(corrected_dist) < 0:
            print(
                f"  → MECHANISM: TLDC BOTH pushes up y_true AND pushes down distractor"
            )
        elif np.mean(corrected_yt) > 0:
            print(f"  → MECHANISM: TLDC primarily PUSHES UP y_true logit")
        elif np.mean(corrected_dist) < 0:
            print(f"  → MECHANISM: TLDC primarily PUSHES DOWN distractor logit")

    # ── Compare corrected vs non-corrected KW ──
    non_corrected_kw = [
        r for r in all_results if r["subset"] == "know_wrong" and not r["is_correct"]
    ]
    if corrected_kw and non_corrected_kw:
        print(f"\n  ── Corrected vs Non-corrected KW comparison ──")
        for label, group in [
            ("Corrected", corrected_kw),
            ("Not corrected", non_corrected_kw),
        ]:
            yt_d = []
            dist_d = []
            for r in group:
                for s in r["steps"]:
                    l_early = s["l_early"].float().squeeze()
                    l_final = s["l_final"].float().squeeze()
                    delta = l_early - l_final
                    yt_d.append(float(delta[r["y_true_id"]].item()))
                    final_am = int(l_final.argmax().item())
                    if final_am != r["y_true_id"]:
                        dist_d.append(float(delta[final_am].item()))
            print(
                f"    {label}: Δ_y_true={np.mean(yt_d):+.2f}, Δ_dist={np.mean(dist_d):+.2f}"
            )

    # ── Save ──
    # Save per-step data for corrected KW samples (serializable subset)
    save_data = []
    for r in all_results:
        entry_out = {
            "sample_id": r["sample_id"],
            "subset": r["subset"],
            "question": r["question"],
            "answers": r["answers"],
            "is_correct": r["is_correct"],
            "gen_text": r["gen_text"],
            "baseline_text": r["baseline_text"],
            "gids": r["gids"],
            "gids_bl": r["gids_bl"],
        }
        # For corrected KW, save full per-step logits
        if r["subset"] == "know_wrong":
            steps_out = []
            for s in r["steps"]:
                steps_out.append(
                    {
                        "step": s["step"],
                        "chosen_id": s["chosen_id"],
                        "l_early_top5": get_topk_info(s["l_early"], tokenizer, k=5),
                        "l_final_top5": get_topk_info(s["l_final"], tokenizer, k=5),
                        "l_combined_top5": get_topk_info(
                            s["l_combined"], tokenizer, k=5
                        ),
                    }
                )
            entry_out["steps"] = steps_out
        save_data.append(entry_out)

    out_path = output_dir / "s15_2b_tldc_per_token.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
