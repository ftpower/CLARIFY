"""C2 Cross-Task Transfer + Joint Detection with max_p.

Two experiments in one run:

Experiment 1 — Cross-Task Transfer:
  Train truth direction v on HellaSwag (4-choice), test projection on TriviaQA (free-text).
  If AUROC remains high → truth direction is task-independent.

Experiment 2 — C2 + max_p Joint:
  Combine TriviaQA truth direction score with per-token max_p.
  Simple fusion: z-score normalize both, average, compute AUROC.

Efficiency: HS extraction + max_p computation both happen during the same generation
forward passes — no extra overhead over the C2 baseline.

Usage:
    python C2_transfer_and_joint.py --n_samples 200
    python C2_transfer_and_joint.py --n_samples 100 --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
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
from src.data_loader import load_triviaqa, load_hellaswag, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Core: extract HS + max_p during generation (single pass)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hs_and_maxp(
    model, tokenizer, samples: list[dict], dataset: str,
    device: str, layers_to_scan: list[int],
    max_new_tokens: int = 20,
) -> dict:
    """Generate answers and extract hidden states + per-token max_p.

    For each sample:
      - First forward pass: hook ALL layers → get HS at last prompt token
      - Every forward pass: compute max_p from final-layer logits
      - Check correctness for label

    Returns:
        dict with:
          - all_h: {layer: np.ndarray [N, d]}
          - max_p_scores: np.ndarray [N] (early_mean of per-token max_p)
          - labels: np.ndarray [N] (1=correct, 0=incorrect)
          - correct_count: int
          - elapsed: float
    """
    all_h = {li: [] for li in layers_to_scan}
    max_p_scores = []
    labels = []
    correct_count = 0

    t0 = time.time()
    for s in tqdm(samples, desc=f"Extract {dataset}"):
        prompt = format_prompt(s["question"], s["context"], dataset=dataset)
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # ── First forward pass: get HS from ALL layers + first logits ──
        residual = {}
        fwd_hooks = []
        for li in layers_to_scan:
            def _hook(act, hook=None, _layer=li):
                residual[_layer] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_resid_post", _hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Store hidden states
        for li in layers_to_scan:
            all_h[li].append(residual[li].float().cpu().numpy().flatten())

        # max_p from first token
        probs = F.softmax(logits[0, -1, :].float(), dim=-1)
        per_token_maxp = [float(probs.max().item())]

        # Get first generated token
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]

        # ── Continue autoregressive generation ──
        for _ in range(max_new_tokens - 1):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)
            probs = F.softmax(logits[0, -1, :].float(), dim=-1)
            per_token_maxp.append(float(probs.max().item()))

        # Aggregate max_p: early_mean
        n_early = max(1, len(per_token_maxp) // 3)
        max_p_scores.append(float(np.mean(per_token_maxp[:n_early])))

        # Check correctness
        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset=dataset)
        if is_correct:
            correct_count += 1
        labels.append(1 if is_correct else 0)

    # Stack HS arrays
    for li in layers_to_scan:
        all_h[li] = np.stack(all_h[li], axis=0)

    elapsed = time.time() - t0
    return {
        "all_h": all_h,
        "max_p_scores": np.array(max_p_scores),
        "labels": np.array(labels),
        "correct_count": correct_count,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Truth direction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_truth_direction(h_correct: np.ndarray, h_incorrect: np.ndarray) -> np.ndarray:
    """v = mean(h_correct) - mean(h_incorrect), L2-normalized."""
    v = h_correct.mean(axis=0) - h_incorrect.mean(axis=0)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    return v


def project_and_auroc(H: np.ndarray, v: np.ndarray, labels: np.ndarray) -> float:
    """Project H onto v, compute AUROC (|direction| ignored)."""
    scores = H @ v  # [N]
    valid = np.isfinite(scores)
    if valid.sum() < 10 or labels[valid].std() == 0:
        return float("nan")
    auroc = float(roc_auc_score(labels[valid], scores[valid]))
    return max(auroc, 1 - auroc)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C2 Cross-Task Transfer + Joint Detection")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_hellaswag", action="store_true",
                        help="Skip HellaSwag extraction (use cached)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"C2: Cross-Task Transfer + Joint Detection")
    print(f"  Model: {args.model}  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # ── Load model ──
    print("Loading model...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    layers_to_scan = list(range(n_layers))
    print(f"  Loaded in {time.time()-t0:.0f}s  ({n_layers} layers)")

    # ── Load data ──
    hs_samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)
    tq_samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed + 1)
    print(f"  HellaSwag: {len(hs_samples)} samples")
    print(f"  TriviaQA:  {len(tq_samples)} samples")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Extract HS + max_p from both datasets
    # ═══════════════════════════════════════════════════════════════════════════

    # ── HellaSwag ──
    print(f"\n{'─'*40}")
    print(f"Phase 1a: HellaSwag extraction (4-choice)")
    hs_data = extract_hs_and_maxp(
        model, tokenizer, hs_samples, "hellaswag",
        device, layers_to_scan,
    )
    print(f"  Correct: {hs_data['correct_count']}/{args.n_samples} "
          f"({hs_data['correct_count']/args.n_samples:.1%})")
    print(f"  Time: {hs_data['elapsed']:.0f}s")

    # ── TriviaQA ──
    print(f"\n{'─'*40}")
    print(f"Phase 1b: TriviaQA extraction (free-text)")
    tq_data = extract_hs_and_maxp(
        model, tokenizer, tq_samples, "triviaqa",
        device, layers_to_scan,
    )
    print(f"  Correct: {tq_data['correct_count']}/{args.n_samples} "
          f"({tq_data['correct_count']/args.n_samples:.1%})")
    print(f"  Time: {tq_data['elapsed']:.0f}s")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: Per-layer analysis
    # ═══════════════════════════════════════════════════════════════════════════

    # ── max_p baseline ──
    maxp_valid = np.isfinite(tq_data["max_p_scores"])
    maxp_auroc = float(roc_auc_score(
        tq_data["labels"][maxp_valid],
        tq_data["max_p_scores"][maxp_valid],
    ))
    maxp_auroc = max(maxp_auroc, 1 - maxp_auroc)
    print(f"\n  max_p baseline (TriviaQA): AUROC = {maxp_auroc:.4f}")

    print(f"\n{'─'*40}")
    print(f"Phase 2: Per-layer AUROC")
    print(f"  {'Layer':>5s}  {'H→T (xfer)':>12s}  {'T→T (in-task)':>14s}  "
          f"{'C2+max_p':>10s}  {'Δ joint':>8s}")
    print(f"  {'─'*55}")

    best_xfer_layer, best_xfer_auroc = -1, 0.0
    best_intask_layer, best_intask_auroc = -1, 0.0
    best_joint_layer, best_joint_auroc = -1, 0.0
    layer_results = []

    for li in sorted(layers_to_scan):
        H_hs = hs_data["all_h"][li]  # HellaSwag HS
        H_tq = tq_data["all_h"][li]  # TriviaQA HS
        labels_hs = hs_data["labels"]
        labels_tq = tq_data["labels"]

        mask_hs_c = labels_hs == 1
        mask_hs_i = labels_hs == 0
        mask_tq_c = labels_tq == 1
        mask_tq_i = labels_tq == 0

        if (mask_hs_c.sum() < 2 or mask_hs_i.sum() < 2 or
            mask_tq_c.sum() < 2 or mask_tq_i.sum() < 2):
            continue

        # ── 2a: Cross-task (HellaSwag v → TriviaQA projection) ──
        v_hs = compute_truth_direction(H_hs[mask_hs_c], H_hs[mask_hs_i])
        auroc_xfer = project_and_auroc(H_tq, v_hs, labels_tq)

        # ── 2b: In-task (TriviaQA v → TriviaQA projection) ──
        v_tq = compute_truth_direction(H_tq[mask_tq_c], H_tq[mask_tq_i])
        auroc_intask = project_and_auroc(H_tq, v_tq, labels_tq)

        # ── 2c: Joint C2 + max_p ──
        # z-score normalize both signals, then average
        c2_scores = H_tq @ v_tq  # [N]
        valid = np.isfinite(c2_scores) & np.isfinite(tq_data["max_p_scores"])

        c2_z = (c2_scores[valid] - c2_scores[valid].mean()) / (c2_scores[valid].std() + 1e-10)
        mp_z = (tq_data["max_p_scores"][valid] - tq_data["max_p_scores"][valid].mean()) / (
            tq_data["max_p_scores"][valid].std() + 1e-10)
        joint_scores = c2_z + mp_z  # equal-weight fusion

        auroc_joint = float("nan")
        if valid.sum() >= 10 and labels_tq[valid].std() > 0:
            auroc_joint = float(roc_auc_score(labels_tq[valid], joint_scores))
            auroc_joint = max(auroc_joint, 1 - auroc_joint)

        delta_joint = (auroc_joint - auroc_intask) if not np.isnan(auroc_joint) else float("nan")

        print(f"  {li:>5d}  {auroc_xfer:>12.4f}  {auroc_intask:>14.4f}  "
              f"{auroc_joint:>10.4f}  {delta_joint:>+8.4f}")

        layer_results.append({
            "layer": li,
            "auroc_xfer": auroc_xfer if not np.isnan(auroc_xfer) else None,
            "auroc_intask": auroc_intask if not np.isnan(auroc_intask) else None,
            "auroc_joint": auroc_joint if not np.isnan(auroc_joint) else None,
            "delta_joint": delta_joint if not np.isnan(delta_joint) else None,
        })

        if not np.isnan(auroc_xfer) and auroc_xfer > best_xfer_auroc:
            best_xfer_auroc = auroc_xfer
            best_xfer_layer = li
        if not np.isnan(auroc_intask) and auroc_intask > best_intask_auroc:
            best_intask_auroc = auroc_intask
            best_intask_layer = li
        if not np.isnan(auroc_joint) and auroc_joint > best_joint_auroc:
            best_joint_auroc = auroc_joint
            best_joint_layer = li

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  max_p baseline (TriviaQA):        AUROC = {maxp_auroc:.4f}")
    print(f"  Cross-task (H→T):  L{best_xfer_layer}  AUROC = {best_xfer_auroc:.4f}")
    print(f"  In-task   (T→T):   L{best_intask_layer}  AUROC = {best_intask_auroc:.4f}")
    print(f"  C2+max_p joint:    L{best_joint_layer}  AUROC = {best_joint_auroc:.4f}")

    if best_intask_auroc > 0:
        xfer_ratio = best_xfer_auroc / best_intask_auroc * 100
        print(f"\n  Cross-task retention: {xfer_ratio:.1f}% of in-task AUROC")
        if xfer_ratio > 90:
            print(f"  → Truth direction is HIGHLY task-independent ✅")
        elif xfer_ratio > 70:
            print(f"  → Truth direction has MODERATE task transfer")
        else:
            print(f"  → Truth direction is mostly task-specific")

    if not np.isnan(best_joint_auroc):
        delta = best_joint_auroc - max(best_intask_auroc, maxp_auroc)
        if delta > 0.01:
            print(f"\n  Joint > best single: +{delta:.4f} — signals are complementary ✅")
        elif delta > -0.01:
            print(f"\n  Joint ≈ best single: {delta:+.4f} — signals are redundant")
        else:
            print(f"\n  Joint < best single: {delta:+.4f} — fusion degrades performance")

    # ── Save ──
    save_path = output_dir / "C2_transfer_and_joint.json"
    with open(save_path, "w") as f:
        json.dump({
            "max_p_auroc": maxp_auroc,
            "best_xfer_layer": best_xfer_layer,
            "best_xfer_auroc": best_xfer_auroc,
            "best_intask_layer": best_intask_layer,
            "best_intask_auroc": best_intask_auroc,
            "best_joint_layer": best_joint_layer,
            "best_joint_auroc": best_joint_auroc,
            "per_layer": sorted(layer_results,
                                key=lambda x: (x.get("auroc_intask") or 0), reverse=True),
            "n_samples": args.n_samples,
            "hs_correct": int(hs_data["labels"].sum()),
            "tq_correct": int(tq_data["labels"].sum()),
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"C2 transfer + joint complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
