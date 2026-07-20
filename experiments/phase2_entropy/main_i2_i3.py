"""I2 + I3: Multi-layer coordination & adaptive lambda calibration.

I2: Hook L11 and L15 simultaneously, sweep (λ11, λ15), compare single vs dual.
I3: λ = λ_base * f(p_correct) — per-sample adaptive intervention strength.

Usage:
    python main_i2_i3.py --n_eval 200
    python main_i2_i3.py --n_eval 200 --i2_only  # skip I3
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ═══════════════════════════════════════════════════════════════════════════
# Hook generation
# ═══════════════════════════════════════════════════════════════════════════


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    sign = -1.0 if mode == "subtract" else 1.0

    def hook(activation, hook=None):
        d = direction.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam * projection

    return hook


def make_dual_hooks(
    dir_L11: torch.Tensor,
    lam_11: float,
    dir_L15: torch.Tensor,
    lam_15: float,
    mode: str = "subtract",
):
    """Return hooks for simultaneous L11+L15 intervention."""
    sign = -1.0 if mode == "subtract" else 1.0

    def hook_L11(activation, hook=None):
        d = dir_L11.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam_11 * projection

    def hook_L15(activation, hook=None):
        d = dir_L15.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam_15 * projection

    return [
        ("blocks.11.hook_resid_post", hook_L11),
        ("blocks.15.hook_resid_post", hook_L15),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# I2: Multi-layer coordination
# ═══════════════════════════════════════════════════════════════════════════


def run_i2(
    model,
    eval_samples: list[dict],
    directions: dict[int, torch.Tensor],
    letter_ids: dict[str, int],
    lams_11: list[float],
    lams_15: list[float],
    output_dir: Path,
):
    print("\n" + "=" * 60)
    print("I2: Multi-Layer Coordination (L11 + L15)")
    print("=" * 60)

    letters = ["A", "B", "C", "D"]
    lid_tensor = torch.tensor([letter_ids[l] for l in letters])

    # Pre-tokenize
    print("Pre-tokenizing...")
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append({"tokens": tokens, "correct_letter": correct_letter})

    dir_L11 = directions[11]
    dir_L15 = directions[15]

    # Configs: single-layer + dual-layer, all subtract mode (best from I1)
    configs = []
    # Single L11
    for lam in lams_11:
        configs.append(("L11", lam, None))
    # Single L15
    for lam in lams_15:
        configs.append(("L15", None, lam))
    # Dual L11+L15
    for lam_11 in lams_11:
        for lam_15 in lams_15:
            configs.append(("L11+L15", lam_11, lam_15))

    accum = {cfg: {"n_correct": 0, "per_sample": []} for cfg in configs}

    # Baseline
    n_base_correct = 0
    base_per_sample = []

    for item in tqdm(tokenized, desc="Evaluating I2"):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)

        # Baseline
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        pred_idx = probs.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        p_correct = probs[letters.index(correct_letter)].item()
        n_base_correct += int(is_correct)
        base_per_sample.append({"is_correct": is_correct, "p_correct": p_correct})

        for cfg in configs:
            label, lam_11, lam_15 = cfg

            if label == "L11":
                hooks = [
                    (
                        "blocks.11.hook_resid_post",
                        make_projection_hook(dir_L11, lam_11, "subtract"),
                    ),
                ]
            elif label == "L15":
                hooks = [
                    (
                        "blocks.15.hook_resid_post",
                        make_projection_hook(dir_L15, lam_15, "subtract"),
                    ),
                ]
            else:  # L11+L15
                hooks = make_dual_hooks(dir_L11, lam_11, dir_L15, lam_15, "subtract")

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            logits_last = logits[0, -1, :]
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            pred_idx = probs.argmax().item()
            is_correct = letters[pred_idx] == correct_letter
            p_correct = probs[letters.index(correct_letter)].item()

            accum[cfg]["n_correct"] += int(is_correct)
            accum[cfg]["per_sample"].append(
                {
                    "is_correct": is_correct,
                    "p_correct": p_correct,
                }
            )

    n_total = len(tokenized)
    baseline_acc = n_base_correct / n_total
    base_filtered = [s for s in base_per_sample if s["p_correct"] > 0.3]
    n_base_filt = len(base_filtered)
    baseline_filt_acc = (
        sum(s["is_correct"] for s in base_filtered) / n_base_filt
        if n_base_filt > 0
        else None
    )
    print(
        f"\nBaseline: full={baseline_acc:.4f}, filtered={baseline_filt_acc:.4f} (n={n_base_filt})"
    )

    # ── Results ──
    results = []
    for cfg in configs:
        label, lam_11, lam_15 = cfg
        a = accum[cfg]
        acc = a["n_correct"] / n_total
        delta = acc - baseline_acc

        filtered = [s for s in a["per_sample"] if s["p_correct"] > 0.3]
        n_filt = len(filtered)
        acc_filt = (
            sum(s["is_correct"] for s in filtered) / n_filt if n_filt >= 20 else None
        )
        delta_f = (
            acc_filt - baseline_filt_acc
            if (acc_filt is not None and baseline_filt_acc is not None)
            else None
        )

        results.append(
            {
                "label": label,
                "lam_11": lam_11,
                "lam_15": lam_15,
                "acc": float(acc),
                "delta": float(delta),
                "acc_filtered": float(acc_filt) if acc_filt else None,
                "delta_filtered": float(delta_f) if delta_f is not None else None,
                "n_filtered": n_filt,
            }
        )

    # ── Report ──
    # Single layer best
    l11_best = max(
        [r for r in results if r["label"] == "L11"],
        key=lambda r: r["delta_filtered"] or -999,
    )
    l15_best = max(
        [r for r in results if r["label"] == "L15"],
        key=lambda r: r["delta_filtered"] or -999,
    )
    dual_best = max(
        [r for r in results if r["label"] == "L11+L15"],
        key=lambda r: r["delta_filtered"] or -999,
    )

    print(
        f"\nBest L11 only:  λ={l11_best['lam_11']}, "
        f"acc_f={l11_best['acc_filtered']:.4f}, Δf={l11_best['delta_filtered']:+.4f}"
    )
    print(
        f"Best L15 only:  λ={l15_best['lam_15']}, "
        f"acc_f={l15_best['acc_filtered']:.4f}, Δf={l15_best['delta_filtered']:+.4f}"
    )
    print(
        f"Best L11+L15:   λ11={dual_best['lam_11']}, λ15={dual_best['lam_15']}, "
        f"acc_f={dual_best['acc_filtered']:.4f}, Δf={dual_best['delta_filtered']:+.4f}"
    )

    # Synergy check
    synergy = dual_best["delta_filtered"] - max(
        l11_best["delta_filtered"], l15_best["delta_filtered"]
    )
    print(f"\nSynergy (dual - best_single): {synergy:+.4f} pp")
    print(f"{'1+1>2' if synergy > 0.005 else 'no synergy'}")

    # Full grid
    print(
        f"\n{'Config':<18} {'λ11':<8} {'λ15':<8} {'Acc':>8} {'Δ':>8} {'AccF':>8} {'Δf':>8}"
    )
    print("-" * 70)
    for r in sorted(results, key=lambda r: r["delta_filtered"] or -999, reverse=True):
        lam_11_str = f"{r['lam_11']}" if r["lam_11"] is not None else "-"
        lam_15_str = f"{r['lam_15']}" if r["lam_15"] is not None else "-"
        acc_str = f"{r['acc']:.4f}"
        delta_str = f"{r['delta']:+.4f}"
        accf_str = f"{r['acc_filtered']:.4f}" if r["acc_filtered"] else "N/A"
        deltaf_str = (
            f"{r['delta_filtered']:+.4f}" if r["delta_filtered"] is not None else "N/A"
        )
        print(
            f"{r['label']:<18} {lam_11_str:<8} {lam_15_str:<8} {acc_str:>8} {delta_str:>8} {accf_str:>8} {deltaf_str:>8}"
        )

    out = {
        "baseline_acc": baseline_acc,
        "baseline_filt_acc": baseline_filt_acc,
        "n_base_filtered": n_base_filt,
        "n_total": n_total,
        "best_L11": {
            k: l11_best[k]
            for k in ["lam_11", "delta", "delta_filtered", "acc_filtered"]
        },
        "best_L15": {
            k: l15_best[k]
            for k in ["lam_15", "delta", "delta_filtered", "acc_filtered"]
        },
        "best_dual": {
            k: dual_best[k]
            for k in ["lam_11", "lam_15", "delta", "delta_filtered", "acc_filtered"]
        },
        "synergy": float(synergy),
        "all_results": results,
    }
    with open(output_dir / "i2_multi_layer_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'i2_multi_layer_results.json'}")

    return {
        "best_single_layer": "L11"
        if l11_best["delta_filtered"] > (l15_best["delta_filtered"] or -999)
        else "L15",
        "best_single_lam": l11_best["lam_11"]
        if l11_best["delta_filtered"] > (l15_best["delta_filtered"] or -999)
        else l15_best["lam_15"],
        "best_single_delta": max(
            l11_best["delta_filtered"], l15_best["delta_filtered"] or -999
        ),
        "best_dual_delta": dual_best["delta_filtered"],
        "synergy": synergy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# I3: Adaptive lambda
# ═══════════════════════════════════════════════════════════════════════════


def run_i3(
    model,
    eval_samples: list[dict],
    directions: dict[int, torch.Tensor],
    letter_ids: dict[str, int],
    base_lams: list[float],
    output_dir: Path,
):
    print("\n" + "=" * 60)
    print("I3: Adaptive Lambda Calibration")
    print("=" * 60)

    letters = ["A", "B", "C", "D"]
    lid_tensor = torch.tensor([letter_ids[l] for l in letters])
    dir_L11 = directions[11]

    # Pre-tokenize
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append({"tokens": tokens, "correct_letter": correct_letter})

    # Step 1: baseline forward pass to get p_correct for each sample
    print("Computing baseline p_correct for each sample...")
    sample_p_correct = []
    baseline_is_correct = []
    baseline_correct = 0
    for item in tqdm(tokenized, desc="Baseline"):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        p_correct = probs[letters.index(correct_letter)].item()
        pred_idx = probs.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        sample_p_correct.append(p_correct)
        baseline_is_correct.append(is_correct)
        baseline_correct += int(is_correct)

    n_total = len(tokenized)
    baseline_acc = baseline_correct / n_total
    print(f"Baseline: full={baseline_acc:.4f}")

    # Configs: fixed λ vs adaptive λ with two schedules
    # f1: λ = λ_base * (1 - p_correct)  → 不确定 → 更强干预
    # f2: λ = λ_base * p_correct        → 确定 → 更强干预（针对"自信但错"）
    configs = []
    for lam_base in base_lams:
        configs.append(
            ("fixed", lam_base, None, None)
        )  # (schedule, lam_base, clip_min, clip_max)
    for lam_base in base_lams:
        configs.append(
            ("1-p", lam_base, 0.05, 2.0)
        )  # clip at [0.05*lam_base, 2.0*lam_base]
    for lam_base in base_lams:
        configs.append(("p", lam_base, 0.05, 2.0))

    accum = {cfg: {"n_correct": 0, "per_sample": []} for cfg in configs}

    for idx, item in enumerate(tqdm(tokenized, desc="Evaluating I3")):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)
        p_c = sample_p_correct[idx]  # baseline P(correct)

        for cfg in configs:
            schedule, lam_base, clip_min, clip_max = cfg

            if schedule == "fixed":
                lam = lam_base
            elif schedule == "1-p":
                lam = lam_base * (1.0 - p_c)
                lam = max(lam_base * clip_min, min(lam_base * clip_max, lam))
            else:  # "p"
                lam = lam_base * p_c
                lam = max(lam_base * clip_min, min(lam_base * clip_max, lam))

            hooks = [
                (
                    "blocks.11.hook_resid_post",
                    make_projection_hook(dir_L11, lam, "subtract"),
                ),
            ]

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            logits_last = logits[0, -1, :]
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            pred_idx = probs.argmax().item()
            is_correct = letters[pred_idx] == correct_letter
            new_p_correct = probs[letters.index(correct_letter)].item()

            accum[cfg]["n_correct"] += int(is_correct)
            accum[cfg]["per_sample"].append(
                {
                    "is_correct": is_correct,
                    "p_correct": new_p_correct,
                    "lambda_used": float(lam),
                }
            )

    # ── Report ──
    results = []
    for cfg in configs:
        schedule, lam_base, clip_min, clip_max = cfg
        a = accum[cfg]
        acc = a["n_correct"] / n_total
        delta = acc - baseline_acc

        # Filtered set
        filtered = [s for s in a["per_sample"] if s["p_correct"] > 0.3]
        n_filt = len(filtered)
        acc_filt = (
            sum(s["is_correct"] for s in filtered) / n_filt if n_filt >= 20 else None
        )
        delta_f = acc_filt - (baseline_acc if n_filt >= 20 else 0)  # rough delta

        # Mean lambda used
        mean_lam = np.mean([s["lambda_used"] for s in a["per_sample"]])

        results.append(
            {
                "schedule": schedule,
                "lambda_base": lam_base,
                "acc": float(acc),
                "delta": float(delta),
                "acc_filtered": float(acc_filt) if acc_filt else None,
                "n_filtered": n_filt,
                "mean_lambda": float(mean_lam),
            }
        )

    # Compute filtered baseline from stored per-sample data
    base_filt_idxs = [i for i, s in enumerate(sample_p_correct) if s > 0.3]
    n_base_filt = len(base_filt_idxs)
    base_filt_correct = sum(1 for i in base_filt_idxs if baseline_is_correct[i])
    baseline_filt_acc = base_filt_correct / n_base_filt if n_base_filt > 0 else 0.0
    print(f"Baseline filtered: {baseline_filt_acc:.4f} (n={n_base_filt})")

    print(
        f"\n{'Schedule':<10} {'λ_base':<8} {'Acc':>8} {'Δ':>8} {'AccF':>8} {'mean_λ':>8}"
    )
    print("-" * 55)
    best_fixed = None
    best_adaptive = None
    for r in sorted(results, key=lambda r: r["delta"], reverse=True):
        accf_str = f"{r['acc_filtered']:.4f}" if r["acc_filtered"] else "N/A"
        print(
            f"{r['schedule']:<10} {r['lambda_base']:<8} {r['acc']:.4f} {r['delta']:>+.4f} {accf_str:>8} {r['mean_lambda']:>8.4f}"
        )
        if r["schedule"] == "fixed" and (
            best_fixed is None or r["delta"] > best_fixed["delta"]
        ):
            best_fixed = r
        if r["schedule"] != "fixed" and (
            best_adaptive is None or r["delta"] > best_adaptive["delta"]
        ):
            best_adaptive = r

    if best_fixed and best_adaptive:
        adaptive_gain = best_adaptive["delta"] - best_fixed["delta"]
        print(
            f"\nBest fixed:      λ={best_fixed['lambda_base']}, Δ={best_fixed['delta']:+.4f}"
        )
        print(
            f"Best adaptive:   schedule={best_adaptive['schedule']}, λ_base={best_adaptive['lambda_base']}, "
            f"Δ={best_adaptive['delta']:+.4f}"
        )
        print(f"Adaptive gain:   {adaptive_gain:+.4f} pp")

    out = {
        "baseline_acc": baseline_acc,
        "n_total": n_total,
        "best_fixed": best_fixed,
        "best_adaptive": best_adaptive,
        "adaptive_gain": float(adaptive_gain)
        if (best_fixed and best_adaptive)
        else None,
        "all_results": results,
    }
    with open(output_dir / "i3_adaptive_lambda_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'i3_adaptive_lambda_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--i2_only", action="store_true")
    parser.add_argument("--i3_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

    # ── Load directions from I1 ──
    dir_file = output_dir / "i1_directions.pt"
    if not dir_file.exists():
        print(f"ERROR: {dir_file} not found. Run main_i1_directions.py first.")
        sys.exit(1)

    ckpt = torch.load(dir_file, map_location=args.device)
    # Use mean_diff directions (best from I1, same as PCA)
    directions = {}
    for L_str, d in ckpt["mean_diff"].items():
        L = int(L_str)
        directions[L] = d.to(args.device)
    print(f"Loaded mean_diff directions for layers: {list(directions.keys())}")

    # ── Load eval data ──
    print(f"Loading HellaSwag validation ({args.n_eval} samples)...")
    eval_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

    # ── Run I2 ──
    if not args.i3_only:
        i2_result = run_i2(
            model=model,
            eval_samples=eval_samples,
            directions=directions,
            letter_ids=letter_ids,
            lams_11=[0.1, 0.3, 0.5, 1.0],
            lams_15=[0.3, 0.5, 1.0],
            output_dir=output_dir,
        )

    # ── Run I3 ──
    if not args.i2_only:
        run_i3(
            model=model,
            eval_samples=eval_samples,
            directions=directions,
            letter_ids=letter_ids,
            base_lams=[0.1, 0.3, 0.5, 1.0],
            output_dir=output_dir,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("I2 + I3 — Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
