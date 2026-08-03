"""Phase 17.3b: Adaptive β TLDC via JS-divergence CQI.

Phase A (diagnostic): Compute JS(P_L20 || P_L27) per sample and check whether
  it distinguishes KW from KC/DK. Gates: A1 (t-test/Cohen's d), A2 (AUROC > 0.55).

Phase B (intervention): β(x) = β₀ · (JS(x) / median_JS)^α — per-sample adaptive
  beta based on the cross-layer distribution shift.

Phase C (gating, if B passes): skip intervention when JS is below threshold.

Usage:
    python validate_s17_3b_adaptive_beta.py --n_test 100 --seed_test 123
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
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
# Early-exit logits (same as validate_s17_2_harq_sic.py)
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
# JS Divergence (L20 vs L27 distribution shift per sample)
# ═════════════════════════════════════════════════════════════════════════════


def compute_js_divergence(logits_early, logits_late):
    """Compute Jensen-Shannon divergence between early and late layer distributions.

    Args:
        logits_early: [vocab] or [1, vocab] float32 — early-layer logits
        logits_late:  [vocab] or [1, vocab] float32 — late-layer logits

    Returns:
        js: float — JS divergence in nats
    """
    # Use log_softmax for numerical stability
    log_p_early = F.log_softmax(logits_early.float().squeeze(), dim=-1)
    log_p_late = F.log_softmax(logits_late.float().squeeze(), dim=-1)

    p_early = log_p_early.exp()
    p_late = log_p_late.exp()

    # m = 0.5 * (p + q); log_m = log(p + q) - log(2)
    log_m = torch.log(p_early + p_late + 1e-12) - np.log(2)

    kl_early = (p_early * (log_p_early - log_m)).sum().item()
    kl_late = (p_late * (log_p_late - log_m)).sum().item()

    js = 0.5 * (kl_early + kl_late)
    return float(max(js, 0.0))  # clamp to avoid tiny negative from numerical error


# ═════════════════════════════════════════════════════════════════════════════
# TLDC formulas (same as validate_s17_2_harq_sic.py)
# ═════════════════════════════════════════════════════════════════════════════


def apply_tldc(l_final, l_early, beta):
    """Standard TLDC: l_combined = l_final + beta * (l_early - l_final)."""
    return l_final.float() + beta * (l_early.float() - l_final.float())


def apply_adaptive_tldc(l_final, l_early, beta_0, alpha, median_js, sample_js):
    """Adaptive TLDC: beta(x) = beta_0 * (JS(x) / median_JS)^alpha."""
    if median_js <= 0 or sample_js <= 0:
        beta = beta_0
    else:
        beta = beta_0 * (sample_js / median_js) ** alpha
    return apply_tldc(l_final, l_early, beta), beta


# ═════════════════════════════════════════════════════════════════════════════
# Sample classification with JS extraction
# ═════════════════════════════════════════════════════════════════════════════


def classify_with_js(
    model,
    tokenizer,
    sample,
    device,
    layer_early,
    final_layer,
    ln_final,
    W_U,
    b_U,
    rank_threshold=50,
):
    """Classify sample (KC/KW/DK) AND extract JS divergence.

    Uses dual hooks to capture both L20 and L27 hidden states in one pass,
    then computes early-exit logits at both layers and computes JS.

    Returns None if y_true cannot be tokenized.
    """
    y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
    if y_true_id is None:
        return None

    prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    hook_final = f"blocks.{final_layer}.hook_resid_post"
    captured = {}

    def _hook_early(act, hook=None):
        captured["h_early"] = act[:, -1:, :].detach()
        return act

    def _hook_final(act, hook=None):
        captured["h_final"] = act[:, -1:, :].detach()
        return act

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits_final = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_early, _hook_early), (hook_final, _hook_final)],
        )

    # Rank of y_true in final-layer logits
    sorted_ids = logits_final[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

    # Baseline generation
    gen_text = greedy_generate(model, tokenizer, prompt, device)
    is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

    if rank <= rank_threshold:
        subset = "know_correct" if is_correct else "know_wrong"
    else:
        subset = "dont_know"

    # Compute early-exit logits at both layers
    l_early = compute_early_logits(captured["h_early"], ln_final, W_U, b_U)
    l_final_exit = compute_early_logits(captured["h_final"], ln_final, W_U, b_U)

    # JS divergence
    js = compute_js_divergence(l_early, l_final_exit)

    return {
        "sample_id": sample.get("id", 0),
        "rank": rank,
        "subset": subset,
        "prompt": prompt,
        "answers": sample["answers"],
        "question": sample["question"][:100],
        "y_true_id": y_true_id,
        "l_final_first": logits_final[0, -1, :].float().detach(),  # from model output
        "l_early_first": l_early.float().squeeze().detach(),  # early-exit L20
        "l_final_exit": l_final_exit.float().squeeze().detach(),  # early-exit L27
        "js_div": js,
        "baseline_correct": is_correct,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Full generation with per-step intervention (from validate_s17_2_harq_sic.py)
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
    """Autoregressive generation with a per-step logit intervention strategy."""
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


# Strategy functions


def strategy_baseline(l_final, l_early, step):
    return l_final


def strategy_tldc(l_final, l_early, step, beta):
    return l_final + beta * (l_early - l_final)


def strategy_adaptive(l_final, l_early, step, beta_0, alpha, median_js, sample_js):
    l_combined, _ = apply_adaptive_tldc(
        l_final, l_early, beta_0, alpha, median_js, sample_js
    )
    return l_combined


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════


def compute_subset_stats(results_list, entries):
    """Compute per-subset accuracy from list of bools."""
    stats = {
        "KW": {"correct": 0, "total": 0},
        "KC": {"correct": 0, "total": 0},
        "DK": {"correct": 0, "total": 0},
        "All": {"correct": 0, "total": 0},
    }
    for e, correct in zip(entries, results_list):
        s = {"know_wrong": "KW", "know_correct": "KC", "dont_know": "DK"}[e["subset"]]
        stats[s]["correct"] += int(correct)
        stats[s]["total"] += 1
        stats["All"]["correct"] += int(correct)
        stats["All"]["total"] += 1
    return stats


def format_rate(correct, total):
    if total == 0:
        return "N/A"
    return f"{correct}/{total} ({correct / total * 100:.1f}%)"


def print_row(name, stats):
    print(
        f"  {name:<35} "
        f"{format_rate(stats['KW']['correct'], stats['KW']['total']):>15} "
        f"{format_rate(stats['KC']['correct'], stats['KC']['total']):>15} "
        f"{format_rate(stats['DK']['correct'], stats['DK']['total']):>15} "
        f"{format_rate(stats['All']['correct'], stats['All']['total']):>15}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Phase A: JS diagnostic statistics
# ═════════════════════════════════════════════════════════════════════════════


def run_phase_a_diagnostics(entries):
    """Compute JS divergence statistics and gate checks.

    Returns dict with keys: per_subset_stats, t_test, cohens_d, auroc, gates
    """
    kw_js = [e["js_div"] for e in entries if e["subset"] == "know_wrong"]
    kc_js = [e["js_div"] for e in entries if e["subset"] == "know_correct"]
    dk_js = [e["js_div"] for e in entries if e["subset"] == "dont_know"]

    per_subset = {}
    for name, vals in [("KW", kw_js), ("KC", kc_js), ("DK", dk_js)]:
        if vals:
            per_subset[name] = {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            }
        else:
            per_subset[name] = {"n": 0, "mean": None, "std": None}

    # Gate A1: t-test KW vs KC
    t_stat, p_value = None, None
    cohens_d = None
    if len(kw_js) >= 2 and len(kc_js) >= 2:
        from scipy import stats as scipy_stats

        t_stat, p_value = scipy_stats.ttest_ind(kw_js, kc_js, equal_var=False)
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(kw_js) - 1) * np.var(kw_js) + (len(kc_js) - 1) * np.var(kc_js))
            / (len(kw_js) + len(kc_js) - 2)
        )
        if pooled_std > 0:
            cohens_d = float((np.mean(kw_js) - np.mean(kc_js)) / pooled_std)

    gate_a1 = (cohens_d is not None and abs(cohens_d) > 0.3) or (
        p_value is not None and p_value < 0.05
    )

    # Gate A2: AUROC of JS for KW vs (KC+DK)
    auroc = None
    if len(kw_js) >= 2:
        rest_js = kc_js + dk_js
        if len(rest_js) >= 2:
            labels = np.array([1] * len(kw_js) + [0] * len(rest_js))
            scores = np.array(kw_js + rest_js)
            auroc = _compute_auroc(labels, scores)

    gate_a2 = auroc is not None and auroc > 0.55

    return {
        "per_subset": per_subset,
        "t_test": {"t_stat": t_stat, "p_value": p_value, "cohens_d": cohens_d},
        "auroc": auroc,
        "gates": {
            "A1": {
                "passed": gate_a1,
                "description": "JS_KW significantly different from JS_KC (Cohen's d > 0.3 or p < 0.05)",
                "details": f"Cohen's d={cohens_d:.4f}" if cohens_d else "N/A",
            },
            "A2": {
                "passed": gate_a2,
                "description": "AUROC(JS, KW vs rest) > 0.55",
                "details": f"AUROC={auroc:.4f}" if auroc else "N/A",
            },
        },
    }


def _compute_auroc(labels, scores):
    """Simple AUROC without sklearn dependency."""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    auc = 0.0
    tp = 0
    fp = 0
    prev_fp = 0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1
            if i > 0 and pairs[i - 1][1] == 1:
                auc += (fp - prev_fp) * tp
                prev_fp = fp

    auc += (n_neg - prev_fp) * tp
    return auc / (n_pos * n_neg)


# ═════════════════════════════════════════════════════════════════════════════
# Phase B: Adaptive β first-token sweep
# ═════════════════════════════════════════════════════════════════════════════


def run_phase_b_sweep(entries, betas, alphas):
    """First-token sweep of adaptive beta vs fixed beta.

    Returns:
        fixed_results:  {beta: {stats_dict, per_sample_booleans}}
        adaptive_results: {(beta_0, alpha): {stats_dict, per_sample_booleans}}
        median_js: float — median JS across all entries
    """
    all_js = [e["js_div"] for e in entries]
    median_js = float(np.median(all_js))

    # Fixed beta baseline
    fixed_results = {}
    for beta in tqdm(betas, desc="  Fixed β first-token", leave=False):
        results = []
        for e in entries:
            l_combined = apply_tldc(e["l_final_first"], e["l_early_first"], beta)
            nid = int(l_combined.squeeze().argmax().item())
            gen_text = tokenizer_decode_safe(nid, e)
            is_correct = check_correct_exact(gen_text, e["answers"])
            results.append(is_correct)
        fixed_results[beta] = {
            "stats": compute_subset_stats(results, entries),
            "per_sample": results,
        }

    # Adaptive beta
    adaptive_results = {}
    combos = [(b0, a) for b0 in betas for a in alphas]
    for beta_0, alpha in tqdm(combos, desc="  Adaptive β first-token", leave=False):
        results = []
        for e in entries:
            l_combined, _beta_used = apply_adaptive_tldc(
                e["l_final_first"],
                e["l_early_first"],
                beta_0,
                alpha,
                median_js,
                e["js_div"],
            )
            nid = int(l_combined.squeeze().argmax().item())
            gen_text = tokenizer_decode_safe(nid, e)
            is_correct = check_correct_exact(gen_text, e["answers"])
            results.append(is_correct)
        adaptive_results[(beta_0, alpha)] = {
            "stats": compute_subset_stats(results, entries),
            "per_sample": results,
        }

    return fixed_results, adaptive_results, median_js


# ═════════════════════════════════════════════════════════════════════════════
# Full generation comparison (best configs)
# ═════════════════════════════════════════════════════════════════════════════


def run_full_generation(
    model,
    tokenizer,
    entries,
    device,
    layer_early,
    ln_final,
    W_U,
    b_U,
    final_layer,
    median_js,
    best_fixed_betas,
    best_adaptive_configs,
):
    """Run full autoregressive generation for best fixed and adaptive configs.

    Returns dict: {config_key: {stats, per_sample}}
    """
    full_results = {}

    # Baseline
    print("  Generating: baseline ...")
    baseline_correct = []
    for e in tqdm(entries, desc="    baseline", leave=False):
        gen = greedy_generate(model, tokenizer, e["prompt"], device)
        baseline_correct.append(check_correct_exact(gen, e["answers"]))
    full_results["baseline"] = {
        "stats": compute_subset_stats(baseline_correct, entries),
        "per_sample": baseline_correct,
    }

    # Fixed beta
    for beta in best_fixed_betas:
        key = f"tldc_b{beta}"
        print(f"  Generating: {key} ...")
        correct = []
        for e in tqdm(entries, desc=f"    {key}", leave=False):
            gen = generate_with_strategy(
                model,
                tokenizer,
                e["prompt"],
                device,
                layer_early,
                ln_final,
                W_U,
                b_U,
                final_layer,
                strategy_tldc,
                {"beta": beta},
            )
            correct.append(check_correct_exact(gen, e["answers"]))
        full_results[key] = {
            "stats": compute_subset_stats(correct, entries),
            "per_sample": correct,
        }

    # Adaptive beta
    for beta_0, alpha in best_adaptive_configs:
        key = f"adaptive_b{beta_0}_a{alpha}"
        print(f"  Generating: {key} ...")
        correct = []
        for e in tqdm(entries, desc=f"    {key}", leave=False):
            gen = generate_with_strategy(
                model,
                tokenizer,
                e["prompt"],
                device,
                layer_early,
                ln_final,
                W_U,
                b_U,
                final_layer,
                strategy_adaptive,
                {
                    "beta_0": beta_0,
                    "alpha": alpha,
                    "median_js": median_js,
                    "sample_js": e["js_div"],
                },
            )
            correct.append(check_correct_exact(gen, e["answers"]))
        full_results[key] = {
            "stats": compute_subset_stats(correct, entries),
            "per_sample": correct,
        }

    return full_results


def tokenizer_decode_safe(nid, entry):
    """Safe token decode for first-token evaluation. Returns string."""
    # We just need the token text for substring matching
    # Use a dummy tokenizer reference set at module level
    return _global_tokenizer.decode([nid])


# ═════════════════════════════════════════════════════════════════════════════
# Gate evaluation
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_gates(full_results, baseline_stats, entries):
    """Evaluate Phase B gates: B1 (KW delta), B2 (KC preservation)."""

    def kw_rate(stats):
        s = stats["KW"]
        return s["correct"] / s["total"] if s["total"] > 0 else 0.0

    def kc_rate(stats):
        s = stats["KC"]
        return s["correct"] / s["total"] if s["total"] > 0 else 0.0

    def kw_delta(stats):
        return kw_rate(stats) - kw_rate(baseline_stats)

    bl_kw = kw_rate(baseline_stats)
    bl_kc = kc_rate(baseline_stats)

    best_fixed_kw_delta = -999
    best_fixed_kw_key = None
    best_adaptive_kw_delta = -999
    best_adaptive_kw_key = None

    for key, result in full_results.items():
        if key == "baseline":
            continue
        d = kw_delta(result["stats"])
        if "tldc_b" in key and d > best_fixed_kw_delta:
            best_fixed_kw_delta = d
            best_fixed_kw_key = key
        if "adaptive" in key and d > best_adaptive_kw_delta:
            best_adaptive_kw_delta = d
            best_adaptive_kw_key = key

    gate_b1 = best_adaptive_kw_delta > best_fixed_kw_delta

    # B2: best adaptive KC delta >= best fixed KC delta * 0.8
    if best_fixed_kw_key:
        fixed_kc_delta = kc_rate(full_results[best_fixed_kw_key]["stats"]) - bl_kc
        adaptive_kc_delta = (
            kc_rate(full_results[best_adaptive_kw_key]["stats"]) - bl_kc
            if best_adaptive_kw_key
            else 0.0
        )
        gate_b2 = adaptive_kc_delta >= fixed_kc_delta * 0.8
    else:
        gate_b2 = None

    return {
        "baseline": {"KW_rate": bl_kw, "KC_rate": bl_kc},
        "best_fixed": {"key": best_fixed_kw_key, "kw_delta": best_fixed_kw_delta},
        "best_adaptive": {
            "key": best_adaptive_kw_key,
            "kw_delta": best_adaptive_kw_delta,
        },
        "gates": {
            "B1": {
                "passed": gate_b1,
                "description": "Adaptive β KW Δ > fixed β KW Δ",
                "details": (
                    f"adaptive Δ={best_adaptive_kw_delta:.4f} vs "
                    f"fixed Δ={best_fixed_kw_delta:.4f}"
                ),
            },
            "B2": {
                "passed": gate_b2,
                "description": "Adaptive β KC Δ ≥ 0.8 × fixed β KC Δ",
                "details": (
                    f"adaptive KC Δ={adaptive_kc_delta:.4f} vs "
                    f"fixed KC Δ={fixed_kc_delta:.4f}"
                )
                if best_fixed_kw_key
                else "N/A (no fixed baseline)",
            },
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

_global_tokenizer = None  # set in main for tokenizer_decode_safe


def main():
    global _global_tokenizer

    parser = argparse.ArgumentParser(
        description="Phase 17.3b: Adaptive β TLDC via JS-divergence CQI"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.05, 0.08, 0.10, 0.15],
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 2.0],
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
    print("Phase 17.3b: Adaptive β TLDC via JS-divergence CQI")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print(f"  Early layer: L{args.layer_early}")
    print(f"  β₀ ∈ {args.betas}, α ∈ {args.alphas}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    _global_tokenizer = tokenizer
    final_layer = model.cfg.n_layers - 1
    print(
        f"  Model: {model.cfg.n_layers} layers (L{final_layer} final), "
        f"loaded in {time.time() - t0:.1f}s"
    )

    # ── Classify samples + extract JS ──
    print(f"\n[2/5] Classifying {args.n_test} samples + extracting JS...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    skipped = 0
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify+JS")):
        sample["id"] = i
        entry = classify_with_js(
            model,
            tokenizer,
            sample,
            device,
            args.layer_early,
            final_layer,
            ln_final,
            W_U,
            b_U,
            args.rank_threshold,
        )
        if entry is not None:
            entries.append(entry)
        else:
            skipped += 1

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]
    print(
        f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}, "
        f"Total={len(entries)}, Skipped={skipped}"
    )

    # ── Phase A: JS diagnostics ──
    print(f"\n[3/5] Phase A: JS divergence diagnostics")
    diag = run_phase_a_diagnostics(entries)

    print(f"\n  ── JS divergence by subset ──")
    for name in ["KW", "KC", "DK"]:
        s = diag["per_subset"][name]
        if s["n"] > 0:
            print(
                f"  {name} (n={s['n']}): μ={s['mean']:.6f}, σ={s['std']:.6f}, "
                f"median={s['median']:.6f}, range=[{s['min']:.6f}, {s['max']:.6f}]"
            )
        else:
            print(f"  {name}: no samples")

    print(f"\n  ── Gate A1 (t-test KW vs KC) ──")
    tt = diag["t_test"]
    print(
        f"  t={tt['t_stat']:.4f}, p={tt['p_value']:.4f}, Cohen's d={tt['cohens_d']:.4f}"
        if tt["t_stat"] is not None
        else "  N/A (insufficient samples)"
    )
    print(f"  Gate A1: {'✅ PASS' if diag['gates']['A1']['passed'] else '❌ FAIL'}")

    print(f"\n  ── Gate A2 (AUROC KW vs rest) ──")
    print(f"  AUROC={diag['auroc']:.4f}" if diag["auroc"] is not None else "  N/A")
    print(f"  Gate A2: {'✅ PASS' if diag['gates']['A2']['passed'] else '❌ FAIL'}")

    if not (diag["gates"]["A1"]["passed"] or diag["gates"]["A2"]["passed"]):
        print("\n  ⚠ Both A1 and A2 failed — JS does not distinguish KW from KC/DK.")
        print("    Skipping Phase B (adaptive β unlikely to help).")
        phase_b_skipped = True
        fixed_results = {}
        adaptive_results = {}
        median_js = float(np.median([e["js_div"] for e in entries]))
        full_results = {}
        gates = {}
    else:
        phase_b_skipped = False
        print(f"\n  ✅ At least one gate passed — proceeding to Phase B.")

        # ── Phase B: First-token sweep ──
        print(f"\n[4/5] Phase B: Adaptive β first-token sweep")
        print(f"  β₀ ∈ {args.betas}, α ∈ {args.alphas}")
        fixed_results, adaptive_results, median_js = run_phase_b_sweep(
            entries,
            args.betas,
            args.alphas,
        )

        # Print first-token results
        print(f"\n  ── First-token exact-match accuracy ──")
        print(f"  {'Strategy':<35} {'KW':>15} {'KC':>15} {'DK':>15} {'All':>15}")
        print(f"  {'─' * 35} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15}")

        for beta in args.betas:
            print_row(f"Fixed β={beta}", fixed_results[beta]["stats"])

        # Find best adaptive per beta_0
        best_per_b0 = {}
        for beta_0 in args.betas:
            best_kw_delta = -999
            best_alpha = None
            for alpha in args.alphas:
                s = adaptive_results[(beta_0, alpha)]["stats"]
                kw_r = s["KW"]["correct"] / s["KW"]["total"] if s["KW"]["total"] else 0
                # Use first fixed beta as baseline
                bl_kw_r = fixed_results[args.betas[0]]["stats"]["KW"]["correct"] / max(
                    fixed_results[args.betas[0]]["stats"]["KW"]["total"], 1
                )
                if kw_r - bl_kw_r > best_kw_delta:
                    best_kw_delta = kw_r - bl_kw_r
                    best_alpha = alpha
            if best_alpha is not None:
                best_per_b0[beta_0] = best_alpha
                print_row(
                    f"Adaptive β₀={beta_0}, α={best_alpha}",
                    adaptive_results[(beta_0, best_alpha)]["stats"],
                )

        # ── Full generation for best configs ──
        print(f"\n[5/5] Full-generation verification...")
        best_fixed_betas = args.betas  # test all fixed betas
        best_adaptive_configs = [(b0, a) for b0, a in best_per_b0.items()]

        full_results = run_full_generation(
            model,
            tokenizer,
            entries,
            device,
            args.layer_early,
            ln_final,
            W_U,
            b_U,
            final_layer,
            median_js,
            best_fixed_betas,
            best_adaptive_configs,
        )

        # Print full-generation results
        print(f"\n  ── Full-generation exact-match accuracy ──")
        print(f"  {'Strategy':<35} {'KW':>15} {'KC':>15} {'DK':>15} {'All':>15}")
        print(f"  {'─' * 35} {'─' * 15} {'─' * 15} {'─' * 15} {'─' * 15}")
        for key in full_results:
            print_row(key, full_results[key]["stats"])

        # Evaluate gates
        baseline_stats = full_results["baseline"]["stats"]
        gates = evaluate_gates(full_results, baseline_stats, entries)

        print(f"\n  ── Gates ──")
        for g_name, g_info in gates["gates"].items():
            status = (
                "✅ PASS"
                if g_info["passed"]
                else ("❌ FAIL" if g_info["passed"] is False else "⚠ N/A")
            )
            print(f"  {g_name}: {status} — {g_info['details']}")

    # ── Save output ──
    print(f"\n{'=' * 72}")
    print(f"Saving results...")

    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "layer_early": args.layer_early,
            "final_layer": final_layer,
            "rank_threshold": args.rank_threshold,
            "betas": args.betas,
            "alphas": args.alphas,
            "phase_b_skipped": phase_b_skipped,
        },
        "sample_counts": {"KC": len(kc), "KW": len(kw), "DK": len(dk)},
        "phase_a_diagnostics": {
            "per_subset": diag["per_subset"],
            "t_test": {
                k: float(v) if v is not None and not isinstance(v, bool) else v
                for k, v in diag["t_test"].items()
            },
            "auroc": diag["auroc"],
            "gates": diag["gates"],
        },
    }

    if not phase_b_skipped:
        # Store first-token results as summary stats only (per_sample arrays are large)
        output["phase_b_first_token"] = {
            "fixed": {
                str(beta): {"stats": r["stats"]} for beta, r in fixed_results.items()
            },
            "adaptive": {
                f"b{beta_0}_a{alpha}": {"stats": r["stats"]}
                for (beta_0, alpha), r in adaptive_results.items()
            },
            "median_js": float(median_js) if "median_js" in dir() else None,
        }

        output["full_generation"] = {
            key: {
                "stats": r["stats"],
                "per_sample": r["per_sample"],
            }
            for key, r in full_results.items()
        }

        output["gates"] = gates

    # Also save per-sample JS values for reference
    output["per_sample_js"] = [
        {
            "sample_id": e["sample_id"],
            "subset": e["subset"],
            "rank": e["rank"],
            "js_div": e["js_div"],
            "question": e["question"],
        }
        for e in entries
    ]

    out_path = output_dir / "s17_3b_adaptive_beta.json"

    # Sanitize: convert numpy bools to Python bools
    def sanitize(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    output = sanitize(output)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to {out_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
