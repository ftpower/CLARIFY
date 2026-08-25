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

from src.data_loader import load_triviaqa, format_prompt, check_correct_exact
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


def _compact_step(l_early, l_final, l_combined, y_true_id, tokenizer, step, chosen_id, k=10):
    """Compact per-step summary: top-k per logit space + y_true logits + chosen id.

    Memory fix (2026-08-25): the previous version stored full [vocab] tensors
    (3 × ~152k floats) per step and kept all of them in `all_results` — ~36 MB
    per sample, which OOM'd an 8 GB GPU at ~80/300 samples. Top-k + scalars are
    ~10 KB per step.
    """
    e_f = l_early.float().squeeze()
    f_f = l_final.float().squeeze()
    c_f = l_combined.float().squeeze()
    return {
        "step": step,
        "chosen_id": chosen_id,
        "y_true_id": y_true_id,
        "early_top10": get_topk_info(e_f, tokenizer, k=k),
        "final_top10": get_topk_info(f_f, tokenizer, k=k),
        "combined_top10": get_topk_info(c_f, tokenizer, k=k),
        "yt_logits": [
            float(e_f[y_true_id].item()),
            float(f_f[y_true_id].item()),
            float(c_f[y_true_id].item()),
        ],
    }


def _delta_yt(s):
    """TLDC delta (l_early - l_final) on the y_true token at this step."""
    return s["yt_logits"][0] - s["yt_logits"][1]


def _delta_distractor(s):
    """TLDC delta on the final-layer argmax token (None if it is y_true itself
    or not present in the early-layer top-10)."""
    final_am, _, final_am_logit = s["final_top10"][0]
    if final_am == s["y_true_id"]:
        return None
    early_logit = next((v for tid, _, v in s["early_top10"] if tid == final_am), None)
    if early_logit is None:
        return None
    return early_logit - final_am_logit


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
            tokens = tokens[:, -1024:]  # keep TAIL (Question) — code review Critical 1

        hook_early = f"blocks.{layer_early}.hook_resid_post"
        captured = {}

        def _hook(act, hook=None):
            captured["h"] = act[:, -1:, :].detach()
            return act

        with torch.no_grad():
            logits_final = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

        # Get rank from final logits (1-indexed — code review Medium 5)
        sorted_ids = logits_final[0, -1, :].float().argsort(descending=True)
        rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item() + 1

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
        is_correct = check_correct_exact(gen_text, sample["answers"])  # exact (High 3)

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
        tokens = tokens[:, -1024:]  # keep TAIL (Question) — code review Critical 1

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

    # Record step 0 (compact: top-k + scalars only — GPU-memory fix)
    steps_log.append(
        _compact_step(l_early_0, l_final_0, l_combined_0, y_true_id, tokenizer, 0, nid)
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
            _compact_step(l_early, l_final, l_combined, y_true_id, tokenizer, step, nid)
        )

    gen_text = tokenizer.decode(gids).strip()

    # ── Also get baseline generation (l_final only, no TLDC) ──
    tokens_bl = model.to_tokens(prompt, prepend_bos=True)
    if tokens_bl.shape[1] > 1024:
        tokens_bl = tokens_bl[:, -1024:]  # keep TAIL (Question) — code review Critical 1

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
    """Print detailed per-step analysis for one sample (compact-step format)."""
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

    def _lookup(top10, tid):
        for t_id, _, val in top10:
            if t_id == tid:
                return val
        return None

    def _rank_in(top10, tid):
        for i, (t_id, _, _) in enumerate(top10):
            if t_id == tid:
                return i + 1
        return None

    def _fmt(v):
        return f"{v:>10.2f}" if v is not None else f"{'—':>10}"

    # Print per-step top-3 comparison
    print(f"\n{'─' * 80}")
    print(
        f"{'Step':>5} {'Source':>10} {'Rank':>5} {'Token ID':>8} {'Token':>20} {'Logit':>10}"
    )
    print(f"{'─' * 80}")

    for s in steps:
        step = s["step"]
        chosen_tldc = tldc_gids[step] if step < len(tldc_gids) else None
        chosen_bl = baseline_gids[step] if step < len(baseline_gids) else None

        topk_combined = s["combined_top10"][:3]

        for rank, (tid, tok_str, logit_val) in enumerate(topk_combined):
            logit_e = _lookup(s["early_top10"], tid)
            logit_f = _lookup(s["final_top10"], tid)

            marker = ""
            if tid == chosen_tldc:
                marker += " ← TLDC ARGMAX"
            if tid == chosen_bl:
                marker += " ← BASELINE"
            if tid == y_true_id:
                marker += " ★ y_true"

            print(
                f"{step:>5} {'early (L20)':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {_fmt(logit_e)}{marker}"
            )
            if rank == 0:
                print(
                    f"{'':>5} {'final (L27)':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {_fmt(logit_f)}"
                )
                print(
                    f"{'':>5} {'combined':>10} {rank + 1:>5} {tid:>8} {tok_str:>20} {logit_val:>10.2f}"
                )

        # Show y_true info if not in top-3
        if y_true_id not in [t[0] for t in topk_combined]:
            e_yt, f_yt, c_yt = s["yt_logits"]
            r_e = _rank_in(s["early_top10"], y_true_id)
            r_f = _rank_in(s["final_top10"], y_true_id)
            r_c = _rank_in(s["combined_top10"], y_true_id)

            def _r(v):
                return f"{v:>5}" if v is not None else f"{'>10':>5}"

            print(
                f"{step:>5} {'early (L20)':>10} {_r(r_e)} {y_true_id:>8} {y_true_str:>20} {e_yt:>10.2f} ★ y_true (off-list)"
            )
            print(
                f"{'':>5} {'final (L27)':>10} {_r(r_f)} {y_true_id:>8} {y_true_str:>20} {f_yt:>10.2f}"
            )
            print(
                f"{'':>5} {'combined':>10} {_r(r_c)} {y_true_id:>8} {y_true_str:>20} {c_yt:>10.2f}"
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
        delta_yt = _delta_yt(s)

        final_argmax_id, final_argmax_str, _ = s["final_top10"][0]
        combined_argmax_id, combined_argmax_str, _ = s["combined_top10"][0]
        _, early_argmax_str, _ = s["early_top10"][0]

        print(f"\n  Step {step}:")
        print(f"    L20  argmax: '{early_argmax_str}'")
        print(f"    L27  argmax: '{final_argmax_str}'")
        print(f"    TLDC argmax: '{combined_argmax_str}'")

        e_yt, f_yt, c_yt = s["yt_logits"]
        print(f"    TLDC delta on y_true ('{y_true_str}'):      {delta_yt:+.2f}")
        print(
            f"      → L20 logit={e_yt:.2f}, L27 logit={f_yt:.2f}, combined={c_yt:.2f}"
        )

        delta_distractor = _delta_distractor(s)
        if final_argmax_id != y_true_id:
            if delta_distractor is not None:
                print(
                    f"    TLDC delta on distractor ('{final_argmax_str}'): {delta_distractor:+.2f}"
                )
            else:
                print(
                    f"    TLDC delta on distractor ('{final_argmax_str}'): not in L20 top-10"
                )

        # Determine: push y_true up or push distractor down?
        if abs(delta_yt) > 0.01:
            if delta_yt > 0 and final_argmax_id != y_true_id:
                print(f"    → TLDC PUSHES UP y_true (+{delta_yt:.2f})")
            elif delta_yt < 0:
                print(f"    → TLDC PUSHES DOWN y_true ({delta_yt:+.2f})")
        if delta_distractor is not None and delta_distractor < -0.01 and final_argmax_id != y_true_id:
            print(f"    → TLDC PUSHES DOWN distractor ({delta_distractor:+.2f})")

        # Check if TLDC flipped something
        if final_argmax_id != combined_argmax_id:
            print(
                f"    🔄 ARGMAX FLIP: L27='{final_argmax_str}' → TLDC='{combined_argmax_str}'"
            )

    # ── Net effect verdict ──
    print(f"\n{'─' * 80}")
    print("NET EFFECT (across all steps):")
    yt_deltas = [_delta_yt(s) for s in steps]
    distractor_deltas = [
        _delta_distractor(s)
        for s in steps
        if s["final_top10"][0][0] != y_true_id and _delta_distractor(s) is not None
    ]
    mean_yt_delta = float(np.mean(yt_deltas))
    mean_dist_delta = float(np.mean(distractor_deltas)) if distractor_deltas else 0.0
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
        is_correct = check_correct_exact(result["gen_text"], e["answers"])  # exact (High 3)
        result["is_correct"] = is_correct
        result["baseline_correct"] = check_correct_exact(
            result["baseline_text"], e["answers"]
        )
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
                all_yt_deltas.append(_delta_yt(s))
                dd = _delta_distractor(s)
                if dd is not None:
                    all_dist_deltas.append(dd)

    print(f"\n  All KW samples (n={len(kw)}):")
    print(f"  Mean TLDC delta on y_true:       {np.mean(all_yt_deltas):+.2f}")
    print(f"  Mean TLDC delta on distractor:    {np.mean(all_dist_deltas):+.2f}")

    if corrected_kw:
        corrected_yt = []
        corrected_dist = []
        for r in corrected_kw:
            for s in r["steps"]:
                corrected_yt.append(_delta_yt(s))
                dd = _delta_distractor(s)
                if dd is not None:
                    corrected_dist.append(dd)

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

    # ── Group-level mechanism stats (2026-08-25: extended to KC/DK groups) ──
    # Question: is TLDC "asymmetric" (pushes up y_true only when greedy is wrong)
    # or "symmetric" (pushes up y_true everywhere → breaks KC)?
    non_corrected_kw = [
        r for r in all_results if r["subset"] == "know_wrong" and not r["is_correct"]
    ]
    kept_kc = [r for r in all_results if r["subset"] == "know_correct" and r["is_correct"]]
    broken_kc = [r for r in all_results if r["subset"] == "know_correct" and not r["is_correct"]]
    rescued_dk = [
        r
        for r in all_results
        if r["subset"] == "dont_know"
        and r["is_correct"]
        and not r["baseline_correct"]
    ]
    not_rescued_dk = [
        r
        for r in all_results
        if r["subset"] == "dont_know"
        and not r["is_correct"]
        and not r["baseline_correct"]
    ]

    def _group_deltas(group):
        """Per-step (delta_yt, delta_dist) lists for a group of results."""
        yt, dist = [], []
        for r in group:
            for s in r["steps"]:
                yt.append(_delta_yt(s))
                dd = _delta_distractor(s)
                if dd is not None:
                    dist.append(dd)
        return yt, dist

    print(f"\n  ── Group mechanism table (mean ± std; push-up = Δ_y_true>0 rate) ──")
    print(
        f"  {'group':>14} {'n':>4} {'Δ_y_true':>14} {'Δ_distractor':>14} {'push-up%':>8} {'push-down%':>9}"
    )
    groups = [
        ("KW rescued", corrected_kw),
        ("KW not rescued", non_corrected_kw),
        ("KC kept", kept_kc),
        ("KC broken", broken_kc),
        ("DK rescued", rescued_dk),
        ("DK not rescued", not_rescued_dk),
    ]
    for label, group in groups:
        if not group:
            print(f"  {label:>14} {'0':>4}")
            continue
        yt, dist = _group_deltas(group)
        push_up = 100 * np.mean([1 if d > 0 else 0 for d in yt])
        push_down = 100 * np.mean([1 if d < 0 else 0 for d in dist]) if dist else float("nan")
        print(
            f"  {label:>14} {len(group):>4} {np.mean(yt):>+13.2f}±{np.std(yt):.2f} "
            f"{np.mean(dist) if dist else 0.0:>+13.2f}±{np.std(dist) if dist else 0.0:.2f} "
            f"{push_up:>7.1f}% {push_down:>8.1f}%"
        )

    # Flip-step histogram for rescued groups (step at which TLDC argmax first != baseline argmax)
    def _flip_steps(group):
        steps = []
        for r in group:
            for i, s in enumerate(r["steps"]):
                tldc_g = r["gids"][i] if i < len(r["gids"]) else None
                bl_g = r["gids_bl"][i] if i < len(r["gids_bl"]) else None
                if tldc_g is not None and bl_g is not None and tldc_g != bl_g:
                    steps.append(i)
                    break
        return steps

    for label, group in [("KW rescued", corrected_kw), ("KC broken", broken_kc)]:
        if group:
            fs = _flip_steps(group)
            print(f"  {label}: first-flip step distribution = {fs}")
            print(f"           mean={np.mean(fs):.1f}, step0={sum(1 for s in fs if s == 0)}/{len(fs)}")

    # ── Save ──
    # Save per-step delta data for ALL samples (compact, serializable)
    save_data = []
    for r in all_results:
        entry_out = {
            "sample_id": r["sample_id"],
            "subset": r["subset"],
            "question": r["question"],
            "answers": r["answers"],
            "is_correct": r["is_correct"],
            "baseline_correct": r["baseline_correct"],
            "gen_text": r["gen_text"],
            "baseline_text": r["baseline_text"],
            "gids": r["gids"],
            "gids_bl": r["gids_bl"],
        }
        steps_out = []
        for s in r["steps"]:
            steps_out.append(
                {
                    "step": s["step"],
                    "chosen_id": s["chosen_id"],
                    "delta_yt": _delta_yt(s),
                    "final_argmax_id": s["final_top10"][0][0],
                    "delta_distractor": _delta_distractor(s),
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
