"""Phase 14b: Anti-Override Intervention.

Theory: docs/theory-intervention-failure.md Section 14.3

Paradigm shift: Instead of "injecting truth" (h += α·v_classic), we "remove override"
(h -= α·v_override). The model already knows the answer in 58% of cases but suppresses
it in 55% of those.

Key operations:
  Anti-override:  h ← h - α·v_override   (remove suppression)
  Classic:         h ← h + α·v_classic    (baseline control)
  Detection-gated: only intervene if ⟨v_classic, h⟩ > τ

Gates:
  O3: Δ accuracy > 10% on know-wrong subset (anti-override)
  O4: Δ accuracy ≈ 0% on know-wrong subset (classic v, control)
  O5: Δ accuracy ≤ 0% on know-correct subset (anti-override, negative side effect check)
  O6: Δ accuracy > 5% overall (detection-gated)

Usage:
    python validate_s14_anti_override.py --n_calibrate 200 --n_test 100
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


def interventional_generate(model, tokenizer, prompt, device, layer, direction, alpha):
    """Generate with additive intervention at specified layer."""
    d_f16 = direction.to(dtype=torch.float16)

    def _hook(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"
    return greedy_generate(
        model, tokenizer, prompt, device, fwd_hooks=[(hook_name, _hook)]
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 14b: Anti-override intervention"
    )
    parser.add_argument(
        "--n_calibrate",
        type=int,
        default=200,
        help="Calibration samples for v_override/v_classic",
    )
    parser.add_argument(
        "--n_test", type=int, default=100, help="Test samples for intervention"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=27,
        help="Intervention layer (L27=best override from 14a)",
    )
    parser.add_argument(
        "--rank_threshold", type=int, default=50, help="Rank threshold for knowability"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.3, 1.0, 2.0],
        help="Alpha values to sweep",
    )
    parser.add_argument(
        "--detection_tau_percentile",
        type=float,
        default=50,
        help="Percentile threshold for detection gate",
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
    print("Phase 14b: Anti-Override Intervention")
    print(f"Layer: L{args.layer}, n_cal={args.n_calibrate}, n_test={args.n_test}")
    print(f"Alphas: {args.alphas}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/6] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Phase 1: Calibration — compute v_override, v_classic, τ ──
    print(f"\n[2/6] Calibration: {args.n_calibrate} samples (seed={args.seed_cal})...")
    cal_samples = load_triviaqa(n_samples=args.n_calibrate, seed=args.seed_cal)

    # Storage
    cal_h = {"know_correct": [], "know_wrong": [], "correct": [], "incorrect": []}
    cal_scores = {"know": [], "dont_know": []}  # v_classic · h for τ

    for i, sample in enumerate(tqdm(cal_samples, desc="  Calibrate")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        h_L, logits, tokens, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )

        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        h_np = h_L.float().cpu().numpy().flatten()

        # Classification
        if rank <= args.rank_threshold:
            if is_correct:
                cal_h["know_correct"].append(h_np)
            else:
                cal_h["know_wrong"].append(h_np)
        # v_classic uses all correct/incorrect (not just know)
        if is_correct:
            cal_h["correct"].append(h_np)
        else:
            cal_h["incorrect"].append(h_np)

    n_cal_kc = len(cal_h["know_correct"])
    n_cal_kw = len(cal_h["know_wrong"])
    n_cal_c = len(cal_h["correct"])
    n_cal_i = len(cal_h["incorrect"])
    n_cal = n_cal_c + n_cal_i

    print(f"  Calibration: KC={n_cal_kc}, KW={n_cal_kw}, C={n_cal_c}, I={n_cal_i}")
    if n_cal_kw < 5 or n_cal_kc < 5:
        print("  ❌ Not enough know-correct or know-wrong samples for v_override!")
        return

    # Compute v_override and v_classic (on calibration)
    v_override_raw = np.mean(cal_h["know_wrong"], axis=0) - np.mean(
        cal_h["know_correct"], axis=0
    )
    v_override_norm = float(np.linalg.norm(v_override_raw))
    v_override_unit = v_override_raw / v_override_norm
    v_override = torch.from_numpy(v_override_unit).float().to(device)

    v_classic_raw = np.mean(cal_h["correct"], axis=0) - np.mean(
        cal_h["incorrect"], axis=0
    )
    v_classic_norm = float(np.linalg.norm(v_classic_raw))
    v_classic_unit = v_classic_raw / v_classic_norm
    v_classic = torch.from_numpy(v_classic_unit).float().to(device)

    cos_vo_vc = float(np.dot(v_override_unit, v_classic_unit))
    print(f"  v_override norm: {v_override_norm:.2f}")
    print(f"  v_classic norm:  {v_classic_norm:.2f}")
    print(f"  cos(v_override, v_classic): {cos_vo_vc:+.4f}")

    # Compute detection threshold τ: separate know vs don't-know
    # We need to re-run calibration samples with v_classic projection
    # Actually, we already have h from calibration. Compute scores offline.
    # For τ, we need know vs don't-know scores for proper threshold.

    # Re-extract for all calibration samples → get v_classic·h scores
    print(f"  Computing detection threshold τ...")
    know_scores = []
    dontknow_scores = []

    for i, sample in enumerate(tqdm(cal_samples, desc="  τ calc")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        h_L, logits, _, _ = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )
        h_np = h_L.float().cpu().numpy().flatten()
        score = float(np.dot(v_classic_unit, h_np))
        rank = get_y_true_rank(logits, y_true_id)

        if rank <= args.rank_threshold:
            know_scores.append(score)
        else:
            dontknow_scores.append(score)

    # Use percentile of know scores as threshold
    tau = float(np.percentile(know_scores, args.detection_tau_percentile))
    print(
        f"  Know scores: [{np.min(know_scores):.2f}, {np.max(know_scores):.2f}] "
        f"mean={np.mean(know_scores):.2f}"
    )
    print(
        f"  Don't-know scores: [{np.min(dontknow_scores):.2f}, "
        f"{np.max(dontknow_scores):.2f}] mean={np.mean(dontknow_scores):.2f}"
    )
    print(f"  τ (P{args.detection_tau_percentile} of know): {tau:.4f}")

    # ── Phase 2: Test samples with intervention ──
    print(f"\n[3/6] Loading {args.n_test} test samples (seed={args.seed_test})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    # Classify test samples
    print(f"\n[4/6] Classifying test samples by knowability...")
    test_entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        h_L, logits, _, _ = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )
        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        h_np = h_L.float().cpu().numpy().flatten()
        score = float(np.dot(v_classic_unit, h_np))

        if rank <= args.rank_threshold:
            if is_correct:
                subset = "know_correct"
            else:
                subset = "know_wrong"
        else:
            subset = "dont_know"

        test_entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "is_correct": is_correct,
                "h_L_np": h_np,
                "score": score,
                "subset": subset,
                "prompt": prompt,
                "answers": sample["answers"],
                "question": sample["question"][:80],
            }
        )

    # Stratify
    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]

    print(f"  Know & Correct: {len(kc)}/{len(test_entries)}")
    print(f"  Know & Wrong:   {len(kw)}/{len(test_entries)}  ← TARGET")
    print(f"  Don't Know:     {len(dk)}/{len(test_entries)}")
    baseline_rate = sum(1 for e in test_entries if e["is_correct"]) / len(test_entries)
    print(f"  Baseline accuracy: {baseline_rate:.1%}")

    if len(kw) == 0:
        print("  ⚠️ No know-wrong samples in test set! Cannot evaluate O3.")
        # Continue anyway for O4, O5, O6

    # ── Phase 3: Run interventions ──
    print(f"\n[5/6] Running interventions (3 methods × {len(args.alphas)} alphas)...")

    methods = {
        "anti_override": {
            "direction": v_override,
            "sign": -1.0,
            "label": "anti-v_over",
        },
        "classic_v": {"direction": v_classic, "sign": +1.0, "label": "classic+v"},
        "gated_anti": {
            "direction": v_override,
            "sign": -1.0,
            "label": "gated-anti",
            "gated": True,
            "tau": tau,
        },
    }

    all_results = {"baseline_rate": baseline_rate, "methods": {}}

    for method_key, method_info in methods.items():
        label = method_info["label"]
        direction = method_info["direction"]
        sign = method_info["sign"]
        is_gated = method_info.get("gated", False)
        print(f"\n  ── {label} ──")

        method_results = {}
        for alpha in args.alphas:
            key = f"alpha={alpha:+.1f}"
            correct_by_subset = {
                "know_wrong": 0,
                "know_correct": 0,
                "dont_know": 0,
                "all": 0,
            }
            count_by_subset = {
                "know_wrong": 0,
                "know_correct": 0,
                "dont_know": 0,
                "all": 0,
            }
            intervened_count = 0

            for e in tqdm(test_entries, desc=f"    {key}", leave=False):
                subset = e["subset"]

                # Detection gate check
                if is_gated and e["score"] <= tau:
                    # Don't intervene — use baseline result
                    if e["is_correct"]:
                        for s in [subset, "all"]:
                            correct_by_subset[s] += 1
                        count_by_subset[subset] += 1
                        count_by_subset["all"] += 1
                    else:
                        count_by_subset[subset] += 1
                        count_by_subset["all"] += 1
                    continue

                intervened_count += 1
                effective_alpha = sign * alpha

                gen_text = interventional_generate(
                    model,
                    tokenizer,
                    e["prompt"],
                    device,
                    args.layer,
                    direction,
                    effective_alpha,
                )
                is_correct_int = check_correct(
                    gen_text, e["answers"], dataset="triviaqa"
                )

                if is_correct_int:
                    for s in [subset, "all"]:
                        correct_by_subset[s] += 1
                for s in [subset, "all"]:
                    count_by_subset[s] += 1

            # Compute rates
            method_results[key] = {}
            for s in ["know_wrong", "know_correct", "dont_know", "all"]:
                if count_by_subset[s] > 0:
                    rate = correct_by_subset[s] / count_by_subset[s]
                    delta = (
                        rate - baseline_rate
                        if s == "all"
                        else (
                            rate
                            - (
                                sum(
                                    1
                                    for e in test_entries
                                    if e["subset"] == s and e["is_correct"]
                                )
                                / max(
                                    1, sum(1 for e in test_entries if e["subset"] == s)
                                )
                            )
                        )
                    )
                else:
                    rate = 0.0
                    delta = 0.0

                method_results[key][s] = {
                    "correct": correct_by_subset[s],
                    "total": count_by_subset[s],
                    "rate": rate,
                    "delta": delta,
                }

            method_results[key]["n_intervened"] = intervened_count
            method_results[key]["n_total"] = len(test_entries)

            # Print summary
            kw_r = method_results[key]["know_wrong"]
            all_r = method_results[key]["all"]
            print(
                f"    {key}: kw={kw_r['correct']}/{kw_r['total']} "
                f"(Δ={kw_r['delta']:+.1%})  "
                f"all={all_r['correct']}/{all_r['total']} "
                f"(Δ={all_r['delta']:+.1%})  "
                f"intervened={intervened_count}/{len(test_entries)}"
            )

        all_results["methods"][label] = method_results

    # ── Gate verification ──
    print(f"\n[6/6] Gate verification")
    print(f"\n{'=' * 80}")

    # Get baseline per-subset rates
    kw_baseline_rate = sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))
    kc_baseline_rate = sum(1 for e in kc if e["is_correct"]) / max(1, len(kc))
    dk_baseline_rate = sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))

    print(f"\n  Baseline rates:")
    print(f"  Know-wrong:   {kw_baseline_rate:.1%} ({len(kw)} samples)")
    print(f"  Know-correct: {kc_baseline_rate:.1%} ({len(kc)} samples)")
    print(f"  Don't-know:   {dk_baseline_rate:.1%} ({len(dk)} samples)")
    print(f"  All:          {baseline_rate:.1%} ({len(test_entries)} samples)")

    # Gate O3: anti-override Δ > 10% on know-wrong
    ao_results = all_results["methods"]["anti-v_over"]
    cv_results = all_results["methods"]["classic+v"]
    ga_results = all_results["methods"]["gated-anti"]

    print(f"\n  ── Anti-override ──")
    for alpha in args.alphas:
        key = f"alpha={alpha:+.1f}"
        r = ao_results[key]["know_wrong"]
        print(f"  α=-{alpha:.1f}: kw={r['correct']}/{r['total']} Δ={r['delta']:+.1%}")

    best_ao_kw = max(
        (
            ao_results[k]["know_wrong"]["delta"]
            for k in ao_results
            if k.startswith("alpha=")
        ),
        default=0.0,
    )
    o3_pass = best_ao_kw > 0.10
    print(
        f"  O3 (anti-override Δ > 10% on know-wrong): "
        f"{'✅' if o3_pass else '❌'} (best Δ={best_ao_kw:+.1%})"
    )

    # Gate O4: classic v Δ ≈ 0 on know-wrong
    print(f"\n  ── Classic v (control) ──")
    for alpha in args.alphas:
        key = f"alpha={alpha:+.1f}"
        r = cv_results[key]["know_wrong"]
        print(f"  α=+{alpha:.1f}: kw={r['correct']}/{r['total']} Δ={r['delta']:+.1%}")

    best_cv_kw = max(
        abs(cv_results[k]["know_wrong"]["delta"])
        for k in cv_results
        if k.startswith("alpha=")
    )
    o4_pass = best_cv_kw < 0.05
    print(
        f"  O4 (classic v Δ ≈ 0 on know-wrong): "
        f"{'✅' if o4_pass else '❌'} (max |Δ|={best_cv_kw:+.1%})"
    )

    # Gate O5: anti-override Δ ≤ 0 on know-correct
    best_ao_kc = max(
        (
            ao_results[k]["know_correct"]["delta"]
            for k in ao_results
            if k.startswith("alpha=")
        ),
        default=0.0,
    )
    o5_pass = best_ao_kc <= 0.0
    print(
        f"\n  O5 (anti-override Δ ≤ 0 on know-correct): "
        f"{'✅' if o5_pass else '❌'} (best Δ={best_ao_kc:+.1%})"
    )

    # Gate O6: gated Δ > 5% overall
    best_ga_all = max(
        (ga_results[k]["all"]["delta"] for k in ga_results if k.startswith("alpha=")),
        default=0.0,
    )
    o6_pass = best_ga_all > 0.05
    print(
        f"  O6 (gated Δ > 5% overall): "
        f"{'✅' if o6_pass else '❌'} (best Δ={best_ga_all:+.1%})"
    )

    # Overall
    n_pass = sum([o3_pass, o4_pass, o5_pass, o6_pass])
    print(f"\n  Gate summary: {n_pass}/4 passed")
    if o3_pass or o6_pass:
        print(f"  ✅ Intervention shows positive effect → proceed to Phase 14d")
    else:
        print(f"  ❌ No positive effect → anti-override alone insufficient")
        print(f"     → Continue with Phase 14c (TLDC)")

    # ── Save ──
    output = {
        "config": vars(args),
        "calibration": {
            "n_know_correct": n_cal_kc,
            "n_know_wrong": n_cal_kw,
            "n_correct": n_cal_c,
            "n_incorrect": n_cal_i,
            "v_override_norm": v_override_norm,
            "v_classic_norm": v_classic_norm,
            "cos_vo_vc": cos_vo_vc,
            "tau": tau,
        },
        "test": {
            "n_total": len(test_entries),
            "n_know_correct": len(kc),
            "n_know_wrong": len(kw),
            "n_dont_know": len(dk),
            "baseline_rate": float(baseline_rate),
            "kw_baseline_rate": float(kw_baseline_rate),
            "kc_baseline_rate": float(kc_baseline_rate),
            "dk_baseline_rate": float(dk_baseline_rate),
        },
        "gates": {
            "O3": {"pass": bool(o3_pass), "best_delta": float(best_ao_kw)},
            "O4": {"pass": bool(o4_pass), "max_abs_delta": float(best_cv_kw)},
            "O5": {"pass": bool(o5_pass), "best_delta": float(best_ao_kc)},
            "O6": {"pass": bool(o6_pass), "best_delta": float(best_ga_all)},
        },
        "results": all_results,
    }

    out_path = output_dir / "s14_anti_override.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
