"""Phase 17.2: HARQ-gated TLDC + SIC exploratory intervention.

17.2b HARQ (Hybrid ARQ) Gating:
  Use a per-sample confidence signal to decide whether to apply TLDC.
  Hypothesis: low-confidence samples (DK) should skip intervention to avoid
  degradation, while high-confidence-wrong samples (KW, implicitly) benefit.

  Gating signals tested:
    1. max_prob:   max(softmax(l_L)) — model's own confidence
    2. prob_gap:   top1_prob - top2_prob — decisiveness
    3. g_norm:     ||l_L20 - l_L||_2 — override magnitude
    4. g_max_abs:  max(|l_L20 - l_L|) — peak override

17.2a SIC (Successive Interference Cancellation):
  Instead of uniform proportional penalty β*g(t) for all tokens, iteratively
  find the tokens with largest |g| and apply directed suppression.
  Even though override is not sparse (17.1a: top-5 = 0.1%), directed
  suppression might flip argmax with less total logit perturbation.

Gates:
  H17_1: HARQ DK Δ > TLDC DK Δ  (reduces DK degradation)
  H17_2: HARQ KW Δ ≥ TLDC KW Δ × 0.8  (preserves KW correction)
  S2:    best SIC KW Δ > best TLDC KW Δ
  S3:    SIC KC Δ ≥ 0%

Usage:
    python validate_s17_2_harq_sic.py --n_test 50 --seed_test 123
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

from common import load_model_and_unembed, get_first_answer_token_id, greedy_generate
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═════════════════════════════════════════════════════════════════════════════
# Exact match (from validate_l20_baseline.py)
# ═════════════════════════════════════════════════════════════════════════════


def check_correct_exact(prediction: str, answers: list[str]) -> bool:
    """Exact match: any answer string (case-insensitive) appears in prediction."""
    pred_lower = prediction.strip().lower()
    for ans in answers:
        ans_lower = ans.strip().lower()
        if not ans_lower:
            continue
        if ans_lower in pred_lower:
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Early-exit logits
# ═════════════════════════════════════════════════════════════════════════════


def compute_early_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state via RMSNorm + W_U (early exit)."""
    dtype = next(ln_final.parameters()).dtype
    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


# ═════════════════════════════════════════════════════════════════════════════
# HARQ Gating signals
# ═════════════════════════════════════════════════════════════════════════════


def compute_gating_signals(l_final, l_early):
    """Compute per-sample gating signals from first-token logits.

    Args:
        l_final: [1, vocab] float32 — final-layer logits
        l_early: [1, vocab] float32 — early-layer logits

    Returns:
        dict with max_prob, prob_gap, g_norm, g_max_abs
    """
    lf = l_final.float().squeeze().detach()
    le = l_early.float().squeeze().detach()

    probs = torch.softmax(lf, dim=-1)
    top2_probs, top2_ids = torch.topk(probs, 2)

    g = le - lf  # [vocab] — TLDC delta

    return {
        "max_prob": float(top2_probs[0].item()),
        "prob_gap": float((top2_probs[0] - top2_probs[1]).item()),
        "g_norm": float(g.norm(p=2).item()),
        "g_max_abs": float(g.abs().max().item()),
        "g_mean_abs": float(g.abs().mean().item()),
        "entropy": float((-probs * torch.log(probs + 1e-12)).sum().item()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TLDC baseline (standard proportional penalty)
# ═════════════════════════════════════════════════════════════════════════════


def apply_tldc(l_final, l_early, beta):
    """Apply standard TLDC: l_combined = l_final + beta * (l_early - l_final)."""
    return l_final.float() + beta * (l_early.float() - l_final.float())


# ═════════════════════════════════════════════════════════════════════════════
# HARQ-gated TLDC
# ═════════════════════════════════════════════════════════════════════════════


def apply_harq_tldc(l_final, l_early, beta, signals, signal_key, tau):
    """Apply TLDC only if gating signal exceeds threshold.

    Args:
        l_final, l_early: logit tensors [1, vocab]
        beta: TLDC interpolation coefficient
        signals: dict from compute_gating_signals
        signal_key: which signal to use for gating
        tau: threshold — apply TLDC if signal > tau, else skip (beta=0)

    Returns:
        l_combined: [1, vocab] float32
        applied: bool — whether TLDC was applied
    """
    if signals[signal_key] > tau:
        return apply_tldc(l_final, l_early, beta), True
    else:
        return l_final.float(), False


# ═════════════════════════════════════════════════════════════════════════════
# SIC: Successive Interference Cancellation
# ═════════════════════════════════════════════════════════════════════════════


def apply_sic(l_final, l_early, beta, gamma, K, y_true_id=None):
    """SIC: iterative directed suppression of top-K over-hyped tokens.

    Algorithm:
      l ← l_L + β·(l_ℓ* - l_L)     # initial TLDC
      for k = 1..K:
          t_k = argmin_t g(t|x)     # most over-hyped (most negative g)
          l[t_k] -= γ · |g(t_k|x)|  # directed suppression
          if y_true_id and argmax(l) == y_true_id: break

    Args:
        l_final: [1, vocab] final logits
        l_early: [1, vocab] early logits
        beta: initial TLDC coefficient
        gamma: suppression strength per iteration
        K: max iterations
        y_true_id: optional — early stop if argmax flips to y_true

    Returns:
        l_combined: [1, vocab] logits after SIC
        n_iterations: number of iterations actually applied
        suppressed_tokens: list of (token_id, g_value) suppressed
    """
    lf = l_final.float().squeeze().clone()
    le = l_early.float().squeeze().clone()
    g = le - lf  # TLDC delta [vocab]

    # Step 1: apply initial TLDC
    l_cur = lf + beta * g

    suppressed = []
    n_iter = 0

    for k in range(K):
        if y_true_id is not None:
            if int(l_cur.argmax().item()) == y_true_id:
                break

        # Find most over-hyped token (largest |g| where g < 0, i.e., L27 > L20)
        # Since g = l_early - l_final, negative g means l_final > l_early (over-hyped)
        # Use g directly: the most negative g is the most over-hyped token
        t_k = int(g.argmin().item())
        g_val = float(g[t_k].item())

        if abs(g_val) < 0.01:  # negligible override
            break

        # Directed suppression: reduce over-hyped token's logit
        penalty = gamma * abs(g_val)
        l_cur[t_k] -= penalty
        suppressed.append((t_k, g_val, penalty))
        n_iter += 1

    return l_cur.unsqueeze(0), n_iter, suppressed


# ═════════════════════════════════════════════════════════════════════════════
# Full generation with per-step intervention
# ═════════════════════════════════════════════════════════════════════════════


def generate_with_strategy(
    model,
    tokenizer,
    prompt,
    device,
    layer_early,
    ln_final,
    W_U,
    b_U,
    final_layer,
    strategy_fn,
    strategy_args,
    max_new=20,
):
    """Autoregressive generation with a per-step logit intervention strategy.

    Args:
        strategy_fn: callable(l_final, l_early, step, **strategy_args) → l_combined
        strategy_args: dict of additional args for strategy_fn
        max_new: max tokens to generate

    Returns:
        generated_text: str
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
    for step in range(max_new):
        with torch.no_grad():
            logits_final = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

        h_early = captured["h"]
        l_early = compute_early_logits(h_early, ln_final, W_U, b_U).float()
        l_final = logits_final[0, -1:, :].float()

        if l_early.shape[-1] != l_final.shape[-1]:
            l_combined = l_final
        else:
            l_combined = strategy_fn(l_final, l_early, step, **strategy_args)

        nid = int(l_combined.squeeze().argmax().item())
        gids.append(nid)

        if nid == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    return tokenizer.decode(gids).strip()


# Strategy functions for full generation


def strategy_baseline(l_final, l_early, step):
    """No intervention — just use l_final."""
    return l_final


def strategy_tldc(l_final, l_early, step, beta):
    """Standard TLDC at every step."""
    return l_final + beta * (l_early - l_final)


def strategy_harq(l_final, l_early, step, beta, signal_key, tau):
    """HARQ-gated TLDC: apply only if signal > tau."""
    signals = compute_gating_signals(l_final, l_early)
    if signals[signal_key] > tau:
        return l_final + beta * (l_early - l_final)
    return l_final


def strategy_sic(l_final, l_early, step, beta, gamma, K, y_true_id):
    """SIC at every step (y_true_id unused at inference, but accepted for consistency)."""
    l_sic, _, _ = apply_sic(l_final, l_early, beta, gamma, K, y_true_id=None)
    return l_sic


# ═════════════════════════════════════════════════════════════════════════════
# Sample classification
# ═════════════════════════════════════════════════════════════════════════════


def classify_sample(model, tokenizer, sample, device, layer_early, rank_threshold=50):
    """Classify a sample into KC/KW/DK. Returns entry dict or None."""
    y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
    if y_true_id is None:
        return None

    prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")

    # Get final logits and rank
    hook_early = f"blocks.{layer_early}.hook_resid_post"
    captured = {}

    def _hook(act, hook=None):
        captured["h"] = act[:, -1:, :].detach()
        return act

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits_final = model.run_with_hooks(tokens, fwd_hooks=[(hook_early, _hook)])

    sorted_ids = logits_final[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

    # Baseline generation
    gen_text = greedy_generate(model, tokenizer, prompt, device)
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
        "l_final_first": logits_final[0, -1, :].float().detach(),  # [vocab]
        "h_early_first": captured["h"].detach(),  # [1, 1, d_model]
        "baseline_correct": is_correct,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 17.2: HARQ-gated TLDC + SIC exploration"
    )
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
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
    print("Phase 17.2: HARQ-gated TLDC + SIC Exploration")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print(f"  Early layer: L{args.layer_early}")
    print(f"  TLDC betas: {args.betas}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1
    print(f"  Model: {model.cfg.n_layers} layers, loaded in {time.time() - t0:.1f}s")

    # ── Classify samples ──
    print(f"\n[2/4] Classifying {args.n_test} test samples...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        sample["id"] = i
        entry = classify_sample(
            model, tokenizer, sample, device, args.layer_early, args.rank_threshold
        )
        if entry is not None:
            # Compute early-exit logits at L20 from captured hidden state
            entry["l_early_first"] = (
                compute_early_logits(entry["h_early_first"], ln_final, W_U, b_U)
                .float()
                .squeeze()
                .detach()
            )  # [vocab]
            entries.append(entry)

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]
    print(f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}, Total={len(entries)}")

    # ── Compute gating signals for all entries ──
    print(f"\n[3/4] Computing gating signals + running strategies...")

    for e in entries:
        e["signals"] = compute_gating_signals(e["l_final_first"], e["l_early_first"])

    # Print gating signal stats by subset
    print(f"\n  ── Gating signal statistics by subset ──")
    for sig_name in ["max_prob", "prob_gap", "g_norm", "g_max_abs", "entropy"]:
        print(f"  {sig_name}:")
        for subset_name, subset_entries in [("KW", kw), ("KC", kc), ("DK", dk)]:
            vals = [e["signals"][sig_name] for e in subset_entries]
            if vals:
                print(
                    f"    {subset_name}: μ={np.mean(vals):.4f}, σ={np.std(vals):.4f}, "
                    f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                )

    # ── Run all strategies ──
    # Strategies to evaluate:
    #   1. Baseline (no intervention)
    #   2. Standard TLDC at each beta
    #   3. HARQ-gated TLDC: signal × tau × beta combinations
    #   4. SIC: beta × gamma × K combinations

    # Pre-compute baseline correctness for all entries
    for e in entries:
        e["correct_exact"] = check_correct_exact(
            greedy_generate(model, tokenizer, e["prompt"], device), e["answers"]
        )

    # ── Strategy: Standard TLDC ──
    tldc_results = {}  # {beta: [is_correct_per_entry]}

    for beta in tqdm(args.betas, desc="  TLDC", leave=False):
        results = []
        for e in entries:
            l_combined = apply_tldc(e["l_final_first"], e["l_early_first"], beta)
            nid = int(l_combined.squeeze().argmax().item())
            # For first-token evaluation, check if argmax token alone is correct
            # This is a first-token-only proxy — not full generation
            gen_text = tokenizer.decode([nid])
            is_correct = check_correct_exact(gen_text, e["answers"])
            results.append(is_correct)
        tldc_results[beta] = results

    # ── Strategy: HARQ-gated TLDC ──
    signal_names = ["max_prob", "prob_gap", "g_norm", "g_max_abs"]
    # Thresholds as percentiles of the signal distribution across all entries
    tau_percentiles = [50, 60, 70, 80, 90]
    # Compute per-signal thresholds
    thresholds = {}
    for sig_name in signal_names:
        all_vals = [e["signals"][sig_name] for e in entries]
        thresholds[sig_name] = {
            p: float(np.percentile(all_vals, p)) for p in tau_percentiles
        }

    harq_results = {}  # {(beta, signal, tau_pct): [is_correct_per_entry]}

    harq_combos = [
        (beta, sig, tau_pct)
        for beta in args.betas
        for sig in signal_names
        for tau_pct in tau_percentiles
    ]

    for beta, sig, tau_pct in tqdm(harq_combos, desc="  HARQ", leave=False):
        tau = thresholds[sig][tau_pct]
        results = []
        for e in entries:
            l_combined, applied = apply_harq_tldc(
                e["l_final_first"], e["l_early_first"], beta, e["signals"], sig, tau
            )
            nid = int(l_combined.squeeze().argmax().item())
            gen_text = tokenizer.decode([nid])
            is_correct = check_correct_exact(gen_text, e["answers"])
            results.append(is_correct)
        harq_results[(beta, sig, tau_pct)] = results

    # ── Strategy: SIC Exploratory ──
    sic_betas = [0.0, 0.03, 0.05, 0.08, 0.10]
    sic_gammas = [0.5, 1.0, 1.5, 2.0]
    sic_Ks = [1, 3, 5, 10]

    sic_results = {}  # {(beta, gamma, K): [is_correct_per_entry]}

    sic_combos = [
        (beta, gamma, K) for beta in sic_betas for gamma in sic_gammas for K in sic_Ks
    ]

    for beta, gamma, K in tqdm(sic_combos, desc="  SIC", leave=False):
        results = []
        for e in entries:
            l_combined, n_iter, suppressed = apply_sic(
                e["l_final_first"],
                e["l_early_first"],
                beta,
                gamma,
                K,
                y_true_id=e["y_true_id"],
            )
            nid = int(l_combined.squeeze().argmax().item())
            gen_text = tokenizer.decode([nid])
            is_correct = check_correct_exact(gen_text, e["answers"])
            results.append(is_correct)
        sic_results[(beta, gamma, K)] = results

    # ── Evaluate ──
    print(f"\n[4/4] Evaluation\n")
    print(f"{'=' * 100}")

    def compute_subset_stats(results_list, entries):
        """Compute per-subset accuracy from list of bools."""
        stats = {
            "KW": {"correct": 0, "total": 0},
            "KC": {"correct": 0, "total": 0},
            "DK": {"correct": 0, "total": 0},
            "All": {"correct": 0, "total": 0},
        }
        for e, correct in zip(entries, results_list):
            s = {"know_wrong": "KW", "know_correct": "KC", "dont_know": "DK"}[
                e["subset"]
            ]
            stats[s]["correct"] += int(correct)
            stats[s]["total"] += 1
            stats["All"]["correct"] += int(correct)
            stats["All"]["total"] += 1
        return stats

    def format_rate(correct, total):
        if total == 0:
            return "N/A"
        return f"{correct}/{total} ({correct / total * 100:.1f}%)"

    # Baseline stats
    baseline_correct = [e["correct_exact"] for e in entries]
    baseline_stats = compute_subset_stats(baseline_correct, entries)

    print(f"  ── First-token exact-match accuracy ──")
    print(f"  {'Strategy':<30} {'KW':>15} {'KC':>15} {'DK':>15} {'All':>15}")
    print(f"  {'─' * 30} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15}")

    def print_row(name, stats):
        print(
            f"  {name:<30} "
            f"{format_rate(stats['KW']['correct'], stats['KW']['total']):>15} "
            f"{format_rate(stats['KC']['correct'], stats['KC']['total']):>15} "
            f"{format_rate(stats['DK']['correct'], stats['DK']['total']):>15} "
            f"{format_rate(stats['All']['correct'], stats['All']['total']):>15}"
        )

    print_row("Baseline (L27 greedy)", baseline_stats)

    # Best TLDC
    best_tldc_beta = None
    best_tldc_kw_delta = -999
    best_tldc_stats = None
    for beta in args.betas:
        stats = compute_subset_stats(tldc_results[beta], entries)
        kw_rate = (
            stats["KW"]["correct"] / stats["KW"]["total"] if stats["KW"]["total"] else 0
        )
        bl_kw_rate = (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        kw_delta = kw_rate - bl_kw_rate
        if kw_delta > best_tldc_kw_delta:
            best_tldc_kw_delta = kw_delta
            best_tldc_beta = beta
            best_tldc_stats = stats

    if best_tldc_stats:
        print_row(f"Best TLDC (β={best_tldc_beta})", best_tldc_stats)

    # Best HARQ
    best_harq_key = None
    best_harq_kw_delta = -999
    best_harq_dk_delta = None

    for (beta, sig, tau_pct), results in harq_results.items():
        stats = compute_subset_stats(results, entries)
        bl_kw_rate = (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        kw_rate = (
            stats["KW"]["correct"] / stats["KW"]["total"] if stats["KW"]["total"] else 0
        )
        kw_delta = kw_rate - bl_kw_rate

        bl_dk_rate = (
            baseline_stats["DK"]["correct"] / baseline_stats["DK"]["total"]
            if baseline_stats["DK"]["total"]
            else 0
        )
        dk_rate = (
            stats["DK"]["correct"] / stats["DK"]["total"] if stats["DK"]["total"] else 0
        )
        dk_delta = dk_rate - bl_dk_rate

        # Prefer HARQ configs that preserve KW while improving DK
        if kw_delta > best_harq_kw_delta:
            best_harq_kw_delta = kw_delta
            best_harq_key = (beta, sig, tau_pct)
            best_harq_dk_delta = dk_delta

    if best_harq_key:
        beta, sig, tau_pct = best_harq_key
        best_harq_stats = compute_subset_stats(harq_results[best_harq_key], entries)
        print_row(f"Best HARQ (β={beta}, {sig}, τ=p{tau_pct})", best_harq_stats)

    # Best SIC
    best_sic_key = None
    best_sic_kw_delta = -999

    for (beta, gamma, K), results in sic_results.items():
        stats = compute_subset_stats(results, entries)
        bl_kw_rate = (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        kw_rate = (
            stats["KW"]["correct"] / stats["KW"]["total"] if stats["KW"]["total"] else 0
        )
        kw_delta = kw_rate - bl_kw_rate
        if kw_delta > best_sic_kw_delta:
            best_sic_kw_delta = kw_delta
            best_sic_key = (beta, gamma, K)

    if best_sic_key:
        beta, gamma, K = best_sic_key
        best_sic_stats = compute_subset_stats(sic_results[best_sic_key], entries)
        print_row(f"Best SIC (β={beta}, γ={gamma}, K={K})", best_sic_stats)

    # ── Gate evaluation ──
    print(f"\n  ── Gate Evaluation ──\n")

    # H17: HARQ gates
    if best_harq_key and best_tldc_stats:
        harq_kw_rate = (
            best_harq_stats["KW"]["correct"] / best_harq_stats["KW"]["total"]
            if best_harq_stats["KW"]["total"]
            else 0
        )
        tldc_kw_rate = (
            best_tldc_stats["KW"]["correct"] / best_tldc_stats["KW"]["total"]
            if best_tldc_stats["KW"]["total"]
            else 0
        )
        harq_dk_rate = (
            best_harq_stats["DK"]["correct"] / best_harq_stats["DK"]["total"]
            if best_harq_stats["DK"]["total"]
            else 0
        )
        tldc_dk_rate = (
            best_tldc_stats["DK"]["correct"] / best_tldc_stats["DK"]["total"]
            if best_tldc_stats["DK"]["total"]
            else 0
        )
        bl_dk_rate = (
            baseline_stats["DK"]["correct"] / baseline_stats["DK"]["total"]
            if baseline_stats["DK"]["total"]
            else 0
        )

        harq_kw_delta = harq_kw_rate - (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        tldc_kw_delta = tldc_kw_rate - (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        harq_dk_delta = harq_dk_rate - bl_dk_rate
        tldc_dk_delta = tldc_dk_rate - bl_dk_rate

        print(
            f"  H17_1: HARQ DK Δ = {harq_dk_delta:+.1%} vs TLDC DK Δ = {tldc_dk_delta:+.1%}"
        )
        if harq_dk_delta > tldc_dk_delta:
            print(f"    ✅ H17_1 PASSED: HARQ reduces DK degradation")
        else:
            print(f"    ❌ H17_1 FAILED: HARQ does not improve DK over TLDC")

        print(
            f"  H17_2: HARQ KW Δ = {harq_kw_delta:+.1%} vs TLDC KW Δ = {tldc_kw_delta:+.1%}"
        )
        if harq_kw_delta >= tldc_kw_delta * 0.8:
            print(f"    ✅ H17_2 PASSED: HARQ preserves ≥80% of TLDC KW correction")
        else:
            print(f"    ❌ H17_2 FAILED: HARQ loses too much KW correction")

    # S2, S3: SIC gates
    if best_sic_key and best_tldc_stats:
        sic_kw_rate = (
            best_sic_stats["KW"]["correct"] / best_sic_stats["KW"]["total"]
            if best_sic_stats["KW"]["total"]
            else 0
        )
        tldc_kw_rate = (
            best_tldc_stats["KW"]["correct"] / best_tldc_stats["KW"]["total"]
            if best_tldc_stats["KW"]["total"]
            else 0
        )

        sic_kw_delta = sic_kw_rate - (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )
        tldc_kw_delta = tldc_kw_rate - (
            baseline_stats["KW"]["correct"] / baseline_stats["KW"]["total"]
            if baseline_stats["KW"]["total"]
            else 0
        )

        sic_kc_rate = (
            best_sic_stats["KC"]["correct"] / best_sic_stats["KC"]["total"]
            if best_sic_stats["KC"]["total"]
            else 0
        )
        bl_kc_rate = (
            baseline_stats["KC"]["correct"] / baseline_stats["KC"]["total"]
            if baseline_stats["KC"]["total"]
            else 0
        )

        print(
            f"\n  S2: SIC KW Δ = {sic_kw_delta:+.1%} vs TLDC KW Δ = {tldc_kw_delta:+.1%}"
        )
        if sic_kw_delta > tldc_kw_delta:
            print(f"    ✅ S2 PASSED: SIC beats TLDC on KW correction")
        else:
            print(f"    ❌ S2 FAILED: SIC does not beat TLDC")

        print(f"  S3: SIC KC Δ = {sic_kc_rate - bl_kc_rate:+.1%}")
        if sic_kc_rate >= bl_kc_rate:
            print(f"    ✅ S3 PASSED: SIC does not harm KC")
        else:
            print(f"    ❌ S3 FAILED: SIC degrades KC")

    # ── Detailed: KW sample-by-sample ──
    print(f"\n  ── KW Sample-by-Sample (best strategies) ──")
    kw_indices = [i for i, e in enumerate(entries) if e["subset"] == "know_wrong"]

    for idx in kw_indices:
        e = entries[idx]
        print(f"\n  Sample {e['sample_id']}: rank={e['rank']}, Q={e['question'][:60]}")
        print(f"    Answer: {e['answers'][:3]}")
        print(
            f"    Signals: max_prob={e['signals']['max_prob']:.4f}, "
            f"g_norm={e['signals']['g_norm']:.1f}, g_max_abs={e['signals']['g_max_abs']:.1f}"
        )

        # Baseline first token
        bl_nid = int(e["l_final_first"].squeeze().argmax().item())
        print(
            f"    Baseline 1st token: '{tokenizer.decode([bl_nid])}' "
            f"(correct={e['correct_exact']})"
        )

        # Best TLDC
        if best_tldc_beta is not None:
            tl = apply_tldc(e["l_final_first"], e["l_early_first"], best_tldc_beta)
            tl_nid = int(tl.squeeze().argmax().item())
            tl_correct = tldc_results[best_tldc_beta][idx]
            print(
                f"    TLDC β={best_tldc_beta}: '{tokenizer.decode([tl_nid])}' "
                f"(correct={tl_correct})"
            )

        # Best HARQ
        if best_harq_key:
            beta, sig, tau_pct = best_harq_key
            tau = thresholds[sig][tau_pct]
            l_hq, applied = apply_harq_tldc(
                e["l_final_first"], e["l_early_first"], beta, e["signals"], sig, tau
            )
            hq_nid = int(l_hq.squeeze().argmax().item())
            hq_correct = harq_results[best_harq_key][idx]
            print(
                f"    HARQ β={beta},{sig}>p{tau_pct}({tau:.3f}): "
                f"'{tokenizer.decode([hq_nid])}' (applied={applied}, correct={hq_correct})"
            )

        # Best SIC
        if best_sic_key:
            beta, gamma, K = best_sic_key
            l_sic, n_iter, suppressed = apply_sic(
                e["l_final_first"], e["l_early_first"], beta, gamma, K, e["y_true_id"]
            )
            sic_nid = int(l_sic.squeeze().argmax().item())
            sic_correct = sic_results[best_sic_key][idx]
            supp_str = ", ".join(
                f"'{tokenizer.decode([tid])}'(g={gv:.1f},pen={p:.1f})"
                for tid, gv, p in suppressed[:3]
            )
            print(
                f"    SIC β={beta},γ={gamma},K={K}: '{tokenizer.decode([sic_nid])}' "
                f"(n_iter={n_iter}, correct={sic_correct})"
            )
            if suppressed:
                print(f"      Suppressed: {supp_str}")

    # ═════════════════════════════════════════════════════════════════════════
    # Full-generation verification (best configs only)
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n  ── Full-Generation Verification (best configs) ──\n")

    full_gen_results = {}

    # Baseline
    print("  Running full-generation: Baseline...")
    bl_full = []
    for e in tqdm(entries, desc="    Baseline", leave=False):
        gen = greedy_generate(model, tokenizer, e["prompt"], device)
        bl_full.append(check_correct_exact(gen, e["answers"]))
    full_gen_results["baseline"] = compute_subset_stats(bl_full, entries)

    # Best TLDC
    if best_tldc_beta is not None:
        print(f"  Running full-generation: TLDC β={best_tldc_beta}...")
        tl_full = []
        for e in tqdm(entries, desc="    TLDC", leave=False):
            gen = generate_with_strategy(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_early,
                ln_final,
                W_U,
                b_U,
                final_layer,
                strategy_tldc,
                {"beta": best_tldc_beta},
            )
            tl_full.append(check_correct_exact(gen, e["answers"]))
        full_gen_results["tldc"] = compute_subset_stats(tl_full, entries)

    # Best HARQ
    if best_harq_key is not None:
        beta, sig, tau_pct = best_harq_key
        tau = thresholds[sig][tau_pct]
        print(f"  Running full-generation: HARQ β={beta},{sig}>p{tau_pct}...")
        hq_full = []
        for e in tqdm(entries, desc="    HARQ", leave=False):
            gen = generate_with_strategy(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_early,
                ln_final,
                W_U,
                b_U,
                final_layer,
                strategy_harq,
                {"beta": beta, "signal_key": sig, "tau": tau},
            )
            hq_full.append(check_correct_exact(gen, e["answers"]))
        full_gen_results["harq"] = compute_subset_stats(hq_full, entries)

    # Best SIC
    if best_sic_key is not None:
        beta, gamma, K = best_sic_key
        print(f"  Running full-generation: SIC β={beta},γ={gamma},K={K}...")
        sic_full = []
        for e in tqdm(entries, desc="    SIC", leave=False):
            gen = generate_with_strategy(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_early,
                ln_final,
                W_U,
                b_U,
                final_layer,
                strategy_sic,
                {"beta": beta, "gamma": gamma, "K": K, "y_true_id": None},
            )
            sic_full.append(check_correct_exact(gen, e["answers"]))
        full_gen_results["sic"] = compute_subset_stats(sic_full, entries)

    # ── Full-gen results table ──
    print(f"\n  ── Full-Generation Exact-Match Accuracy ──")
    print(f"  {'Strategy':<30} {'KW':>15} {'KC':>15} {'DK':>15} {'All':>15}")
    print(f"  {'─' * 30} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15}")

    for name, stats in full_gen_results.items():
        label = {
            "baseline": "Baseline (full gen)",
            "tldc": f"TLDC β={best_tldc_beta}",
            "harq": f"HARQ β={best_harq_key[0]},{best_harq_key[1]}>p{best_harq_key[2]}"
            if best_harq_key
            else "HARQ",
            "sic": f"SIC β={best_sic_key[0]},γ={best_sic_key[1]},K={best_sic_key[2]}"
            if best_sic_key
            else "SIC",
        }[name]
        print_row(label, stats)

    # ── Re-evaluate gates with full-gen results ──
    bl_fg = full_gen_results["baseline"]
    bl_kw_rate = (
        bl_fg["KW"]["correct"] / bl_fg["KW"]["total"] if bl_fg["KW"]["total"] else 0
    )
    bl_dk_rate = (
        bl_fg["DK"]["correct"] / bl_fg["DK"]["total"] if bl_fg["DK"]["total"] else 0
    )
    bl_kc_rate = (
        bl_fg["KC"]["correct"] / bl_fg["KC"]["total"] if bl_fg["KC"]["total"] else 0
    )

    print(f"\n  ── Gate Re-evaluation (full generation) ──\n")

    if "tldc" in full_gen_results and "harq" in full_gen_results:
        tl_fg = full_gen_results["tldc"]
        hq_fg = full_gen_results["harq"]
        tl_kw_rate = (
            tl_fg["KW"]["correct"] / tl_fg["KW"]["total"] if tl_fg["KW"]["total"] else 0
        )
        tl_dk_rate = (
            tl_fg["DK"]["correct"] / tl_fg["DK"]["total"] if tl_fg["DK"]["total"] else 0
        )
        hq_kw_rate = (
            hq_fg["KW"]["correct"] / hq_fg["KW"]["total"] if hq_fg["KW"]["total"] else 0
        )
        hq_dk_rate = (
            hq_fg["DK"]["correct"] / hq_fg["DK"]["total"] if hq_fg["DK"]["total"] else 0
        )

        hq_kw_delta = hq_kw_rate - bl_kw_rate
        tl_kw_delta = tl_kw_rate - bl_kw_rate
        hq_dk_delta = hq_dk_rate - bl_dk_rate
        tl_dk_delta = tl_dk_rate - bl_dk_rate

        print(
            f"  H17_1 (full gen): HARQ DK Δ = {hq_dk_delta:+.1%} vs TLDC DK Δ = {tl_dk_delta:+.1%}"
        )
        print(f"    {'✅' if hq_dk_delta > tl_dk_delta else '❌'} H17_1")

        print(
            f"  H17_2 (full gen): HARQ KW Δ = {hq_kw_delta:+.1%} vs TLDC KW Δ = {tl_kw_delta:+.1%}"
        )
        print(f"    {'✅' if hq_kw_delta >= tl_kw_delta * 0.8 else '❌'} H17_2")

    if "tldc" in full_gen_results and "sic" in full_gen_results:
        tl_fg = full_gen_results["tldc"]
        sc_fg = full_gen_results["sic"]
        tl_kw_rate = (
            tl_fg["KW"]["correct"] / tl_fg["KW"]["total"] if tl_fg["KW"]["total"] else 0
        )
        sc_kw_rate = (
            sc_fg["KW"]["correct"] / sc_fg["KW"]["total"] if sc_fg["KW"]["total"] else 0
        )
        sc_kc_rate = (
            sc_fg["KC"]["correct"] / sc_fg["KC"]["total"] if sc_fg["KC"]["total"] else 0
        )

        sc_kw_delta = sc_kw_rate - bl_kw_rate
        tl_kw_delta = tl_kw_rate - bl_kw_rate
        sc_kc_delta = sc_kc_rate - bl_kc_rate

        print(
            f"\n  S2 (full gen):  SIC KW Δ = {sc_kw_delta:+.1%} vs TLDC KW Δ = {tl_kw_delta:+.1%}"
        )
        print(f"    {'✅' if sc_kw_delta > tl_kw_delta else '❌'} S2")
        print(f"  S3 (full gen):  SIC KC Δ = {sc_kc_delta:+.1%}")
        print(f"    {'✅' if sc_kc_rate >= bl_kc_rate else '❌'} S3")

    # ── Save results ──
    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "layer_early": args.layer_early,
            "final_layer": final_layer,
            "betas": args.betas,
        },
        "sample_counts": {"KC": len(kc), "KW": len(kw), "DK": len(dk)},
        "baseline": {
            "stats": {k: v for k, v in baseline_stats.items()},
        },
        "tldc": {
            str(beta): {
                k: v
                for k, v in compute_subset_stats(tldc_results[beta], entries).items()
            }
            for beta in args.betas
        },
        "harq": {
            "best_key": str(best_harq_key),
            "best_stats": {k: v for k, v in best_harq_stats.items()}
            if best_harq_stats
            else None,
            "thresholds": {
                sig: {str(p): v for p, v in thresh.items()}
                for sig, thresh in thresholds.items()
            },
        },
        "sic": {
            "best_key": str(best_sic_key),
            "best_stats": {k: v for k, v in best_sic_stats.items()}
            if best_sic_stats
            else None,
        },
        "gates": {
            "H17_1": None,
            "H17_2": None,
            "S2": None,
            "S3": None,
        },
        "full_gen": {
            name: {k: v for k, v in stats.items()}
            for name, stats in full_gen_results.items()
        },
    }

    out_path = output_dir / "s17_2_harq_sic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    print(f"\n{'=' * 72}")
    print("PHASE 17.2 COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
