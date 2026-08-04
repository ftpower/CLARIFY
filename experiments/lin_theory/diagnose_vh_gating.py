"""Phase 18.1: v·h gating for TLDC — diagnostic experiment.

Theory: docs/theory-intervention-failure.md §5
Plan:   ~/.claude/plans/CLARIFY/phase18-tldc-improvements.md

Uses the existing truthfulness detection direction v (mean correct - mean wrong)
at L27 as a per-sample gate for whether to apply TLDC.

Diagnostic gates:
  P5.1 (sanity): μ_KW[s(x)] < μ_KC[s(x)] — v·h is a truthfulness detector
  P5.2 (core):   TLDC corrections concentrated in low-s (low v·h) subset

Usage:
    python diagnose_vh_gating.py --n_calibrate 200 --n_test 100

Output:
    experiments/outputs/lin_theory/s18_1_vh_gating.json
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
from common import (
    load_model_and_unembed,
    compute_v,
    get_first_answer_token_id,
    extract_h_at_layer,
    greedy_generate,
)


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
# Extract both L20 and L27 hidden states + logits in one pass
# ═════════════════════════════════════════════════════════════════════════════


def extract_all(model, tokenizer, prompt, device, layer_early=20, final_layer=27):
    """Extract h_L20, h_L27, l_L20, l_L27 in a single forward pass.

    Returns:
        h_early: [1, 1, d_model] on device
        h_final: [1, 1, d_model] on device
        l_early: [1, vocab] float32, early-exit logits
        l_final: [1, vocab] float32, final logits
        rank: int — y_true rank (or -1 if y_true_id is None)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    captured = {}

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    hook_final = f"blocks.{final_layer}.hook_resid_post"

    def _hook_early(act, hook=None):
        captured["h_early"] = act[:, -1:, :].detach()
        return act

    def _hook_final(act, hook=None):
        captured["h_final"] = act[:, -1:, :].detach()
        return act

    with torch.no_grad():
        logits_final = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_early, _hook_early), (hook_final, _hook_final)],
        )

    return (
        captured["h_early"],  # [1, 1, d_model]
        captured["h_final"],  # [1, 1, d_model]
        logits_final[0, -1:, :],  # [1, vocab] from final layer
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 18.1: v·h gating diagnosis")
    parser.add_argument("--n_calibrate", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--v_layer", type=int, default=27, help="Layer for v direction")
    parser.add_argument("--rank_threshold", type=int, default=50)
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

    print("=" * 72)
    print("Phase 18.1: v·h Gating for TLDC — Diagnosis")
    print(f"  v layer: L{args.v_layer}")
    print(f"  Early layer ℓ*: L{args.layer_early}")
    print(f"  n_cal={args.n_calibrate}, n_test={args.n_test}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1  # L27
    print(f"  Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Compute or load v vector at L27 ──
    print(f"\n[2/5] Computing truth direction v at L{args.v_layer}...")
    t0 = time.time()
    v, v_stats = compute_v(
        model, tokenizer, args.n_calibrate, device, args.v_layer, seed=args.seed_cal
    )
    print(
        f"  v computed: n_correct={v_stats['n_correct']}, n_incorrect={v_stats['n_incorrect']}"
    )
    print(f"  v_norm_raw={v_stats['v_norm_raw']:.5f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Extract test sample data ──
    print(
        f"\n[3/5] Extracting test sample data ({args.n_test} samples, seed={args.seed_test})..."
    )
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Extract")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        h_early, h_final, l_final_raw = extract_all(
            model, tokenizer, prompt, device, args.layer_early, final_layer
        )

        # Compute early-exit logits
        l_early = compute_early_logits(
            h_early, ln_final, W_U, b_U
        ).float()  # [1, vocab]
        l_final_exit = compute_early_logits(
            h_final, ln_final, W_U, b_U
        ).float()  # [1, vocab]

        # y_true rank (final layer)
        sorted_ids = l_final_raw[0, :].float().argsort(descending=True)
        rank = int((sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item())

        # v·h score
        h_final_vec = h_final.squeeze().float()  # [d_model]
        s = float(torch.dot(v.float(), h_final_vec).item())

        # Baseline generation for KC/KW/DK classification
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        # TLDC first-token effect (β=0.10): does it flip argmax to y_true?
        l_tldc = l_final_exit.float() + 0.10 * (l_early.float() - l_final_exit.float())
        tldc_nid = int(l_tldc.squeeze().argmax().item())
        tldc_corrects = tldc_nid == y_true_id

        # Baseline first-token
        bl_nid = int(l_final_exit.squeeze().argmax().item())
        bl_first_correct = bl_nid == y_true_id

        entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "subset": subset,
                "prompt": prompt,
                "answers": sample["answers"],
                "question": sample["question"][:100],
                "y_true_id": y_true_id,
                "s": s,  # v·h score
                "bl_first_token_id": bl_nid,  # baseline first token
                "bl_first_correct": bl_first_correct,
                "tldc_first_correct": tldc_corrects,
                "gen_correct": is_correct,
            }
        )

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]

    n_kw, n_kc, n_dk = len(kw), len(kc), len(dk)
    print(f"  KC={n_kc}, KW={n_kw}, DK={n_dk}, Total={len(entries)}")

    # ── Analysis ──
    print(f"\n[4/5] Analysis: v·h score distribution and TLDC gating...")

    # P5.1: s(x) distribution by subset
    s_all = np.array([e["s"] for e in entries])
    s_kw = np.array([e["s"] for e in kw])
    s_kc = np.array([e["s"] for e in kc])
    s_dk = np.array([e["s"] for e in dk])

    print(f"\n  ── P5.1: v·h score distribution by subset ──")
    for label, arr in [("KW", s_kw), ("KC", s_kc), ("DK", s_dk)]:
        if len(arr) > 0:
            print(
                f"  {label}: μ={arr.mean():.5f}, σ={arr.std():.5f}, "
                f"min={arr.min():.5f}, max={arr.max():.5f}"
            )

    # One-tailed t-test: H0: μ_KW = μ_KC vs H1: μ_KW < μ_KC
    if n_kw >= 2 and n_kc >= 2:
        t_stat, p_value = sp_stats.ttest_ind(s_kw, s_kc, alternative="less")
        p5_1_pass = (p_value < 0.05) and (s_kw.mean() < s_kc.mean())
        print(
            f"\n  P5.1 t-test: t={t_stat:.4f}, p={p_value:.4f} (one-tailed, H1: μ_KW < μ_KC)"
        )
        print(f"  P5.1 (μ_KW < μ_KC): {'✅' if p5_1_pass else '❌'}")
    else:
        p5_1_pass = None
        print(f"\n  P5.1: SKIP (insufficient samples)")

    # P5.2: TLDC corrections concentrated in low-s subset
    print(f"\n  ── P5.2: TLDC correction vs v·h score ──")

    # Pre-compute s_median (used by both P5.2 analysis and save section)
    s_median = float(np.median(s_all))

    # Find KW samples that TLDC corrects (first-token argmax flips to y_true)
    kw_corrected = [
        e for e in kw if e["tldc_first_correct"] and not e["bl_first_correct"]
    ]
    kw_not_corrected = [e for e in kw if not e["tldc_first_correct"]]

    print(f"  KW TLDC-corrected: {len(kw_corrected)}/{n_kw}")
    print(f"  KW not corrected:  {len(kw_not_corrected)}/{n_kw}")

    if kw_corrected:
        s_corrected = np.array([e["s"] for e in kw_corrected])
        s_not_corrected = (
            np.array([e["s"] for e in kw_not_corrected])
            if kw_not_corrected
            else np.array([])
        )

        print(f"  Corrected s: μ={s_corrected.mean():.5f}, σ={s_corrected.std():.5f}")
        if len(s_not_corrected) > 0:
            print(
                f"  Not-corrected s: μ={s_not_corrected.mean():.5f}, σ={s_not_corrected.std():.5f}"
            )

        # Key test: are corrected samples disproportionately in low-s region?
        low_s_entries = [e for e in entries if e["s"] < s_median]
        low_s_kw = [e for e in kw if e["s"] < s_median]
        low_s_corrected = [e for e in kw_corrected if e["s"] < s_median]

        p_corrected = len(kw_corrected) / max(1, n_kw)
        p_corrected_low_s = len(low_s_corrected) / max(1, len(low_s_kw))

        print(f"\n  s median threshold = {s_median:.5f}")
        print(f"  Low-s (< median): {len(low_s_entries)}/{len(entries)} total samples")
        print(f"  Low-s KW: {len(low_s_kw)}/{n_kw}")
        print(
            f"  Corrected in low-s: {len(low_s_corrected)}/{len(low_s_kw)} "
            f"({p_corrected_low_s:.1%})"
        )
        print(f"  Overall correction rate: {p_corrected:.1%}")

        # P5.2: TLDC correction rate higher in low-s
        if p_corrected > 0:
            p5_2_pass = p_corrected_low_s > p_corrected
            print(
                f"  P5.2 (concentration in low-s): {'✅' if p5_2_pass else '❌'} "
                f"({p_corrected_low_s:.1%} vs {p_corrected:.1%} overall)"
            )
        else:
            p5_2_pass = False
            print(f"  P5.2: ❌ (no TLDC corrections to analyze, n_KW={n_kw})")
    else:
        p5_2_pass = False
        print(f"  P5.2: ❌ (no TLDC corrections in KW subset)")

    # ── τ sweep ──
    print(f"\n  ── τ sweep (first-token accuracy by gate threshold) ──")

    tau_percentiles = [25, 33, 50, 67, 75]
    tau_values = {p: float(np.percentile(s_all, p)) for p in tau_percentiles}

    print(
        f"  {'τ_pct':>6}  {'τ_val':>10}  {'Applied':>8}  {'KW_acc':>10}  {'KC_acc':>10}  {'DK_acc':>10}  {'All_acc':>10}"
    )
    print(
        f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}"
    )

    tau_results = {}
    for pct in tau_percentiles:
        tau = tau_values[pct]
        correct_kw = 0
        correct_kc = 0
        correct_dk = 0
        n_applied = 0

        for e in entries:
            # Gate: apply TLDC only if s < τ
            if e["s"] < tau:
                is_correct = e["tldc_first_correct"]
                n_applied += 1
            else:
                is_correct = e["bl_first_correct"]

            if e["subset"] == "know_wrong":
                correct_kw += int(is_correct)
            elif e["subset"] == "know_correct":
                correct_kc += int(is_correct)
            elif e["subset"] == "dont_know":
                correct_dk += int(is_correct)

        total = len(entries)
        correct_all = correct_kw + correct_kc + correct_dk

        kw_acc = correct_kw / max(1, n_kw) * 100
        kc_acc = correct_kc / max(1, n_kc) * 100
        dk_acc = correct_dk / max(1, n_dk) * 100
        all_acc = correct_all / max(1, total) * 100

        print(
            f"  {pct:>6}  {tau:>10.5f}  {n_applied:>4}/{total:<4}  "
            f"{kw_acc:>9.1f}%  {kc_acc:>9.1f}%  {dk_acc:>9.1f}%  {all_acc:>9.1f}%"
        )

        tau_results[str(pct)] = {
            "tau": float(tau),
            "n_applied": n_applied,
            "kw_correct": correct_kw,
            "kw_total": n_kw,
            "kw_acc": kw_acc / 100,
            "kc_correct": correct_kc,
            "kc_total": n_kc,
            "kc_acc": kc_acc / 100,
            "dk_correct": correct_dk,
            "dk_total": n_dk,
            "dk_acc": dk_acc / 100,
            "all_correct": correct_all,
            "all_total": total,
            "all_acc": all_acc / 100,
        }

    # Baseline (no intervention) and TLDC (all samples)
    bl_kw_acc = sum(1 for e in kw if e["bl_first_correct"]) / max(1, n_kw)
    tldc_kw_acc = sum(1 for e in kw if e["tldc_first_correct"]) / max(1, n_kw)
    bl_kc_acc = sum(1 for e in kc if e["bl_first_correct"]) / max(1, n_kc)
    tldc_kc_acc = sum(1 for e in kc if e["tldc_first_correct"]) / max(1, n_kc)
    bl_all_acc = sum(1 for e in entries if e["bl_first_correct"]) / max(1, len(entries))
    tldc_all_acc = sum(1 for e in entries if e["tldc_first_correct"]) / max(
        1, len(entries)
    )

    print(
        f"\n  Baseline first-token: KW={bl_kw_acc:.1%}, KC={bl_kc_acc:.1%}, All={bl_all_acc:.1%}"
    )
    print(
        f"  TLDC β=0.10 all-samples: KW={tldc_kw_acc:.1%}, KC={tldc_kc_acc:.1%}, All={tldc_all_acc:.1%}"
    )

    # ── Continuous β(s) analysis (JSCC-like soft gating) ──
    print(f"\n  ── Continuous β(s) soft gating (exploratory) ──")
    # β(s) = β₀ · σ(α(τ - s)) where σ is sigmoid, soft transition
    for alpha in [1.0, 5.0, 10.0]:
        for pct in [33, 50, 67]:
            tau = float(np.percentile(s_all, pct))
            correct_kw_cont = 0
            correct_kc_cont = 0
            correct_all_cont = 0

            for e in entries:
                # Soft gate: β(x) = β₀ · sigmoid(α(τ - s))
                beta_soft = 0.10 / (1 + np.exp(-alpha * (tau - e["s"])))
                if beta_soft < 0.001:
                    is_correct = e["bl_first_correct"]
                elif beta_soft > 0.099:
                    is_correct = e["tldc_first_correct"]
                else:
                    # Interpolate logits
                    # (We don't have intermediate logits cached, so approximate:
                    #  if beta_soft > 0.05 use TLDC result, else baseline)
                    is_correct = (
                        e["tldc_first_correct"]
                        if beta_soft > 0.05
                        else e["bl_first_correct"]
                    )

                if e["subset"] == "know_wrong":
                    correct_kw_cont += int(is_correct)
                elif e["subset"] == "know_correct":
                    correct_kc_cont += int(is_correct)
                correct_all_cont += int(is_correct)

            kw_cont_acc = correct_kw_cont / max(1, n_kw)
            kc_cont_acc = correct_kc_cont / max(1, n_kc)
            all_cont_acc = correct_all_cont / max(1, len(entries))
            print(
                f"  α={alpha:>4.0f}, τ=p{pct}: KW={kw_cont_acc:.1%}, KC={kc_cont_acc:.1%}, All={all_cont_acc:.1%}"
            )

    # ── Gate summary ──
    print(f"\n[5/5] Gate Summary")
    print(f"  {'=' * 60}")
    gates = {
        "P5.1": {
            "pass": p5_1_pass,
            "desc": "μ_KW[s] < μ_KC[s] (truthfulness detection sanity)",
        },
        "P5.2": {
            "pass": p5_2_pass,
            "desc": "TLDC corrections concentrated in low-s subset",
        },
    }
    for gname, ginfo in gates.items():
        status = (
            "✅ PASS"
            if ginfo["pass"]
            else ("❌ FAIL" if ginfo["pass"] is False else "⏸️ SKIP")
        )
        print(f"  {gname}: {status} — {ginfo['desc']}")

    n_pass = sum(1 for g in gates.values() if g["pass"])
    print(f"\n  {n_pass}/{len(gates)} gates passed")
    if p5_2_pass:
        print(f"  ✅ Phase 18.1 diagnostic PASSED → proceed to full-generation")
    else:
        print(f"  ❌ Phase 18.1 diagnostic FAILED → skip full-generation")

    # ── Save ──
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "v_layer": args.v_layer,
            "layer_early": args.layer_early,
            "final_layer": final_layer,
            "rank_threshold": args.rank_threshold,
            "seed_cal": args.seed_cal,
            "seed_test": args.seed_test,
        },
        "v_stats": {
            "n_correct": v_stats["n_correct"],
            "n_incorrect": v_stats["n_incorrect"],
            "v_norm_raw": v_stats["v_norm_raw"],
        },
        "sample_counts": {"KC": n_kc, "KW": n_kw, "DK": n_dk, "total": len(entries)},
        "s_distribution": {
            "KW": {
                "mean": float(s_kw.mean()),
                "std": float(s_kw.std()),
                "min": float(s_kw.min()),
                "max": float(s_kw.max()),
            },
            "KC": {
                "mean": float(s_kc.mean()),
                "std": float(s_kc.std()),
                "min": float(s_kc.min()),
                "max": float(s_kc.max()),
            },
            "DK": {
                "mean": float(s_dk.mean()),
                "std": float(s_dk.std()),
                "min": float(s_dk.min()),
                "max": float(s_dk.max()),
            },
        },
        "gates": {
            "P5.1": {
                "pass": bool(p5_1_pass) if p5_1_pass is not None else None,
                "t_stat": float(t_stat) if n_kw >= 2 and n_kc >= 2 else None,
                "p_value": float(p_value) if n_kw >= 2 and n_kc >= 2 else None,
            },
            "P5.2": {
                "pass": bool(p5_2_pass),
                "kw_corrected": len(kw_corrected) if kw_corrected else 0,
                "s_corrected_mean": float(s_corrected.mean()) if kw_corrected else None,
                "s_median": s_median,
                "p_corrected_overall": float(p_corrected) if kw_corrected else None,
                "p_corrected_low_s": float(p_corrected_low_s)
                if kw_corrected and "p_corrected_low_s" in dir()
                else None,
            },
        },
        "tau_sweep": tau_results,
        "baseline_first_token": {
            "kw_acc": float(bl_kw_acc),
            "kc_acc": float(bl_kc_acc),
            "all_acc": float(bl_all_acc),
        },
        "tldc_all_first_token": {
            "kw_acc": float(tldc_kw_acc),
            "kc_acc": float(tldc_kc_acc),
            "all_acc": float(tldc_all_acc),
        },
        "per_sample": [
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "s": e["s"],
                "rank": e["rank"],
                "bl_first_correct": e["bl_first_correct"],
                "tldc_first_correct": e["tldc_first_correct"],
                "gen_correct": e["gen_correct"],
                "question": e["question"],
            }
            for e in entries
        ],
    }

    out_path = output_dir / "s18_1_vh_gating.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
