"""Direction D: Truth Direction Intervention — lean version.

Tests three intervention modes at the best detection layer (L20).
Uses pre-extracted hidden states to compute v, then generates with intervention.

Modes:
  D.0: Baseline (no intervention)
  D.1: Additive — h' = h + α·v  (shift toward correctness)
  D.2: Amplify — h' = h + α·(v·h)·v  (scale truth component)
  D.3: Subtract — h' = h - α·v  (shift away, control)

Usage:
    python D_intervention_lean.py --n_samples 30 --layer 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_sys_parent = Path(__file__).parent
for _p in [
    str(_sys_parent.parent / "phase2_entropy"),
    str(_sys_parent.parent / "phase4_generalization"),
    str(_sys_parent.parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct


def compute_v(model, tokenizer, n_calibrate, device, layer):
    """Compute truth direction v from a small calibration set."""
    samples = load_triviaqa(n_samples=n_calibrate, seed=42)
    h_correct = []
    h_incorrect = []

    for s in tqdm(samples, desc="Calibrate v"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        residual = {}
        def _hook(act, hook=None):
            residual["h"] = act[:, -1, :].detach()
            return act
        fwd_hooks = [(f"blocks.{layer}.hook_resid_post", _hook)]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        # Quick generation for correctness check
        for _ in range(19):
            if nid == model.tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = model.tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")

        h_vec = residual["h"].float().cpu().numpy().flatten()
        if is_correct:
            h_correct.append(h_vec)
        else:
            h_incorrect.append(h_vec)

    h_correct = np.stack(h_correct, axis=0)
    h_incorrect = np.stack(h_incorrect, axis=0)
    v = h_correct.mean(axis=0) - h_incorrect.mean(axis=0)
    v = v / np.linalg.norm(v)
    return torch.from_numpy(v).float().to(device)


def generate_with_intervention(model, tokenizer, samples, dataset, device,
                                layer, v, alpha, mode, max_new_tokens=20):
    """Generate with intervention. Returns (accuracy, answers, details)."""
    correct = 0
    results = []

    # Cast v to match model dtype (usually float16)
    v_local = v.to(dtype=torch.float16)

    for s in tqdm(samples, desc=f"{mode} α={alpha:+0.1f}", leave=False):
        prompt = format_prompt(s["question"], s["context"], dataset=dataset)
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        def _intervene(act, hook=None):
            h = act[:, -1, :]
            if mode == "additive":
                h_new = h + alpha * v_local
            elif mode == "amplify":
                proj = (h @ v_local)
                h_new = h + alpha * proj * v_local
            elif mode == "subtract":
                h_new = h - alpha * v_local
            else:
                h_new = h
            act[:, -1, :] = h_new
            return act

        fwd_hooks = [(f"blocks.{layer}.hook_resid_post", _intervene)]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(max_new_tokens - 1):
            if nid == model.tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = model.tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset=dataset)
        if is_correct:
            correct += 1
        results.append({"answer": ans, "is_correct": is_correct,
                        "n_tokens": len(gids)})

    return correct / len(samples), correct, results


def main():
    parser = argparse.ArgumentParser(description="D: Intervention — lean")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--n_calibrate", type=int, default=200)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nDirection D: Truth Direction Intervention")
    print(f"  Layer: L{args.layer}  Test samples: {args.n_samples}")
    print(f"  Calibration samples: {args.n_calibrate}")

    # ── Load model ──
    print("\nLoading model...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    tokenizer = model.tokenizer

    # ── Compute v ──
    print(f"Computing truth direction v at L{args.layer}...")
    t0 = time.time()
    v = compute_v(model, tokenizer, args.n_calibrate, device, args.layer)
    print(f"  Done in {time.time()-t0:.0f}s")

    # ── Load test samples ──
    test_samples = load_triviaqa(n_samples=args.n_samples, seed=123)

    # ── D.0: Baseline ──
    print(f"\nD.0: Baseline (no intervention)")
    base_acc, base_correct, base_results = generate_with_intervention(
        model, tokenizer, test_samples, "triviaqa", device,
        args.layer, v, alpha=0.0, mode="none",
    )
    print(f"  Accuracy: {base_acc:.3f} ({base_correct}/{args.n_samples})")

    # ── D.1-D.3: Sweep ──
    configs = [
        # (mode, alpha)
        ("additive", -1.0),
        ("additive", -0.5),
        ("additive", -0.2),
        ("additive", 0.2),
        ("additive", 0.5),
        ("additive", 1.0),
        ("amplify", -1.0),
        ("amplify", -0.5),
        ("amplify", 0.5),
        ("amplify", 1.0),
        ("subtract", -0.5),
        ("subtract", 0.5),
    ]

    print(f"\nD.1-D.3: Intervention sweep")
    print(f"  {'Mode':>12s}  {'α':>6s}  {'Acc':>8s}  {'Δ base':>8s}  {'Correct':>8s}")
    print(f"  {'─'*52}")

    all_results = []
    best_acc = base_acc
    best_config = ("none", 0.0)

    for mode, alpha in configs:
        acc, correct, _ = generate_with_intervention(
            model, tokenizer, test_samples, "triviaqa", device,
            args.layer, v, alpha=alpha, mode=mode,
        )
        delta = acc - base_acc
        marker = " ✅" if delta > 0.05 else (" ⬆" if delta > 0 else "")
        print(f"  {mode:>12s}  {alpha:>+6.1f}  {acc:>8.3f}  {delta:>+8.3f}{marker}  "
              f"{correct:>8d}")

        all_results.append({
            "mode": mode, "alpha": alpha, "accuracy": acc,
            "delta": delta, "correct": correct,
        })

        if acc > best_acc:
            best_acc = acc
            best_config = (mode, alpha)

    print(f"\n  Baseline: {base_acc:.3f}")
    print(f"  Best:     {best_config[0]}, α={best_config[1]:+.1f} → {best_acc:.3f} "
          f"(Δ={best_acc-base_acc:+.3f})")

    if best_acc > base_acc + 0.05:
        print(f"  ✅ Intervention IMPROVES accuracy!")
    elif best_acc > base_acc - 0.05:
        print(f"  ≈ Intervention has NO SIGNIFICANT EFFECT (within ±5%)")
    else:
        print(f"  ❌ Intervention DEGRADES accuracy")

    # ── Save ──
    save_path = output_dir / "D_intervention_lean.json"
    with open(save_path, "w") as f:
        json.dump({
            "layer": args.layer,
            "n_samples": args.n_samples,
            "n_calibrate": args.n_calibrate,
            "baseline_accuracy": base_acc,
            "baseline_correct": base_correct,
            "best_mode": best_config[0],
            "best_alpha": best_config[1],
            "best_accuracy": best_acc,
            "best_delta": best_acc - base_acc,
            "all_results": all_results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
