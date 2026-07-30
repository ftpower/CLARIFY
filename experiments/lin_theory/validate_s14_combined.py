"""Phase 14d: Combined Intervention — Anti-Override + TLDC.

Theory: docs/theory-intervention-failure.md Section 14.4

Combines two complementary intervention mechanisms:
  1. Hidden space (L27): h ← h - α·v_override  (remove suppression)
  2. Logit space:        logits ← l_L + β·(l_ℓ* - l_L)  (TLDC toward detection layer)

Phase 14b (anti-override alone): zero effect
Phase 14c (TLDC alone): positive effect at β=0.1 (KW Δ=+28.6%)

Usage:
    python validate_s14_combined.py --n_test 50
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


def get_y_true_rank(logits, y_true_id):
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Compute early-exit logits from hidden state. Float16 for memory."""
    dtype = next(ln_final.parameters()).dtype
    h_norm = ln_final(h.to(dtype=dtype))
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def combined_greedy_generate(
    model,
    tokenizer,
    prompt,
    device,
    layer_override,
    v_override,
    alpha,
    layer_early,
    W_U,
    b_U,
    ln_final,
    beta,
    max_new=20,
):
    """Greedy generation with combined hidden-space + logit-space intervention.

    Step 1 (first token): Apply hidden intervention at layer_override, capture
                          h at layer_early for TLDC.
    Step 2+: For each subsequent token, apply hidden intervention + TLDC.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    dtype = next(ln_final.parameters()).dtype

    hook_override = f"blocks.{layer_override}.hook_resid_post"
    hook_early = f"blocks.{layer_early}.hook_resid_post"

    # Prepare override direction
    v_override_f16 = v_override.to(dtype=dtype)

    captured_early = {}

    def _hook_early(act, hook=None):
        captured_early["h"] = act[:, -1:, :].detach()
        return act

    def _hook_override(act, hook=None):
        act[:, -1, :] -= alpha * v_override_f16
        return act

    # ── First token ──
    with torch.no_grad():
        logits_final = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_override, _hook_override), (hook_early, _hook_early)],
        )

    h_early = captured_early["h"]
    l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
    l_final = logits_final[0, -1:, :].float()
    logits_adj = l_final + beta * (l_early - l_final)
    nid = int(logits_adj.argmax(dim=-1).item())
    gids = [nid]

    # ── Subsequent tokens ──
    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

        with torch.no_grad():
            logits_final = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (hook_override, _hook_override),
                    (hook_early, _hook_early),
                ],
            )

        h_early = captured_early["h"]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final = logits_final[0, -1:, :].float()
        logits_adj = l_final + beta * (l_early - l_final)
        nid = int(logits_adj.argmax(dim=-1).item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    return ans


def main():
    parser = argparse.ArgumentParser(description="Phase 14d: Combined intervention")
    parser.add_argument("--n_calibrate", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer_override", type=int, default=27)
    parser.add_argument("--layer_early", type=int, default=20)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.1, 0.3, 0.5, 1.0],
        help="Anti-override alpha values",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
        help="TLDC beta values",
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
    print("Phase 14d: Combined Anti-Override + TLDC")
    print(f"  Override layer: L{args.layer_override}, α ∈ {args.alphas}")
    print(f"  Early layer:    L{args.layer_early}, β ∈ {args.betas}")
    print(f"  n_test={args.n_test}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Calibrate: compute v_override + classify test samples ──
    print(f"\n[2/4] Calibration + classification...")
    cal_samples = load_triviaqa(n_samples=args.n_calibrate, seed=args.seed_cal)

    # Compute v_override from calibration
    cal_h = {"know_correct": [], "know_wrong": [], "correct": [], "incorrect": []}
    for sample in tqdm(cal_samples[: args.n_calibrate], desc="  Calibrate"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue
        h_L, logits, _, _ = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer_override
        )
        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")
        h_np = h_L.float().cpu().numpy().flatten()

        if is_correct:
            cal_h["correct"].append(h_np)
            if rank <= args.rank_threshold:
                cal_h["know_correct"].append(h_np)
        else:
            cal_h["incorrect"].append(h_np)
            if rank <= args.rank_threshold:
                cal_h["know_wrong"].append(h_np)

    n_kc = len(cal_h["know_correct"])
    n_kw = len(cal_h["know_wrong"])
    print(f"  Calibration: KC={n_kc}, KW={n_kw}")

    if n_kw >= 5 and n_kc >= 5:
        v_override_raw = np.mean(cal_h["know_wrong"], axis=0) - np.mean(
            cal_h["know_correct"], axis=0
        )
        v_override_norm = float(np.linalg.norm(v_override_raw))
        v_override_unit = v_override_raw / v_override_norm
        v_override = torch.from_numpy(v_override_unit).float().to(device)
        use_override = True
        print(f"  ‖v_override‖={v_override_norm:.1f}")
    else:
        print(f"  ⚠️ Not enough KC/KW samples, skipping anti-override component")
        use_override = False
        v_override = None

    # Classify test samples
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    test_entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify test")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue
        h_L, logits, _, _ = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer_override
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
            }
        )

    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]

    baseline_rate = sum(1 for e in test_entries if e["is_correct"]) / len(test_entries)
    kw_baseline = sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))
    kc_baseline = sum(1 for e in kc if e["is_correct"]) / max(1, len(kc))
    dk_baseline = sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))

    print(f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}, All={baseline_rate:.1%}")

    # ── Run interventions ──
    print(
        f"\n[3/4] Combined intervention ({len(args.alphas)} α × {len(args.betas)} β)..."
    )

    all_results = {"baseline_rate": baseline_rate, "combinations": {}}

    # First, run TLDC-only baselines (α=0, various β)
    print(f"\n  ── TLDC only (α=0) ──")
    for beta in args.betas:
        correct = defaultdict(int)
        count = defaultdict(int)
        for e in tqdm(test_entries, desc=f"    β={beta:.3f}", leave=False):
            subset = e["subset"]
            gen_text = combined_greedy_generate(
                model,
                tokenizer,
                e["prompt"],
                device,
                args.layer_override,
                v_override if use_override else torch.zeros(2048),
                0.0,  # α=0 → no anti-override
                args.layer_early,
                W_U,
                b_U,
                ln_final,
                beta,
            )
            is_c = check_correct(gen_text, e["answers"], dataset="triviaqa")
            if is_c:
                correct[subset] += 1
                correct["all"] += 1
            count[subset] += 1
            count["all"] += 1

        result = {}
        for s in ["know_wrong", "know_correct", "dont_know", "all"]:
            bl = {
                "know_wrong": kw_baseline,
                "know_correct": kc_baseline,
                "dont_know": dk_baseline,
                "all": baseline_rate,
            }[s]
            rate = correct[s] / count[s] if count[s] > 0 else 0.0
            result[s] = {
                "correct": correct[s],
                "total": count[s],
                "rate": rate,
                "delta": rate - bl,
            }
        all_results["combinations"][f"α=0.0_β={beta:.3f}"] = result

        kw_r = result["know_wrong"]
        all_r = result["all"]
        print(
            f"    β={beta:.3f}: kw={kw_r['correct']}/{kw_r['total']} "
            f"(Δ={kw_r['delta']:+.1%}) all={all_r['correct']}/{all_r['total']} "
            f"(Δ={all_r['delta']:+.1%})"
        )

    # Then run combinations with α > 0
    for alpha in args.alphas:
        if not use_override:
            break
        print(f"\n  ── α={alpha:.1f} ──")
        # Only try best β values
        betas_subset = [b for b in args.betas if b <= 0.12]
        for beta in betas_subset:
            correct = defaultdict(int)
            count = defaultdict(int)
            for e in tqdm(
                test_entries, desc=f"    α={alpha:.1f} β={beta:.3f}", leave=False
            ):
                subset = e["subset"]
                gen_text = combined_greedy_generate(
                    model,
                    tokenizer,
                    e["prompt"],
                    device,
                    args.layer_override,
                    v_override,
                    alpha,
                    args.layer_early,
                    W_U,
                    b_U,
                    ln_final,
                    beta,
                )
                is_c = check_correct(gen_text, e["answers"], dataset="triviaqa")
                if is_c:
                    correct[subset] += 1
                    correct["all"] += 1
                count[subset] += 1
                count["all"] += 1

            result = {}
            for s in ["know_wrong", "know_correct", "dont_know", "all"]:
                bl = {
                    "know_wrong": kw_baseline,
                    "know_correct": kc_baseline,
                    "dont_know": dk_baseline,
                    "all": baseline_rate,
                }[s]
                rate = correct[s] / count[s] if count[s] > 0 else 0.0
                result[s] = {
                    "correct": correct[s],
                    "total": count[s],
                    "rate": rate,
                    "delta": rate - bl,
                }
            all_results["combinations"][f"α={alpha:.1f}_β={beta:.3f}"] = result

            kw_r = result["know_wrong"]
            all_r = result["all"]
            print(
                f"    α={alpha:.1f} β={beta:.3f}: kw={kw_r['correct']}/{kw_r['total']} "
                f"(Δ={kw_r['delta']:+.1%}) all={all_r['correct']}/{all_r['total']} "
                f"(Δ={all_r['delta']:+.1%})"
            )

    # ── Summary ──
    print(f"\n[4/4] Summary")
    print(f"\n{'=' * 80}")
    print(
        f"  Baseline: {baseline_rate:.1%} ({sum(1 for e in test_entries if e['is_correct'])}/{len(test_entries)})"
    )
    print(f"  KW={len(kw)}, KC={len(kc)}, DK={len(dk)}")
    print(f"\n  {'Config':>20}  {'KW Δ':>8}  {'KC Δ':>8}  {'DK Δ':>8}  {'All Δ':>8}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    best_kw_delta = 0.0
    best_config = ""
    for config, r in sorted(all_results["combinations"].items()):
        kw_d = r["know_wrong"]["delta"]
        if kw_d > best_kw_delta:
            best_kw_delta = kw_d
            best_config = config
        print(
            f"  {config:>20}  {r['know_wrong']['delta']:>+8.1%}  "
            f"{r['know_correct']['delta']:>+8.1%}  "
            f"{r['dont_know']['delta']:>+8.1%}  "
            f"{r['all']['delta']:>+8.1%}"
        )

    print(f"\n  Best KW Δ: {best_kw_delta:+.1%} ({best_config})")

    # Save
    output = {
        "config": vars(args),
        "test": {
            "n_total": len(test_entries),
            "n_kc": len(kc),
            "n_kw": len(kw),
            "n_dk": len(dk),
            "baseline_rate": baseline_rate,
        },
        "best": {"config": best_config, "kw_delta": best_kw_delta},
        "results": all_results,
    }
    out_path = output_dir / "s14_combined.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
