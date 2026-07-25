"""Direction A+D: Truth Subspace + Truth Direction Intervention.

Direction A — Truth Subspace (1D → k-D):
  Extends Truth Direction from a single direction to a k-dimensional subspace.
  Method: Orthogonal Sequential Discriminant (OSD) directions.
    v₁ = μ_correct - μ_incorrect
    For j=2..k: project data to remove v₁..v_{j-1}, recompute mean difference.
  This captures additional discriminative dimensions beyond the main truth axis.

Direction D — Truth Direction Intervention (detect → correct):
  Modifies the residual stream along v during generation to see if hallucination
  can be reduced. Three modes:
    D.1 - Additive: h' = h + α·v  (shift toward correctness)
    D.2 - Subtractive: h' = h - α·v  (shift away, control)
    D.3 - Amplify: h' = h + α·(v·h)·v  (scale the truth component)

Usage:
    python AD_exploration.py --n_samples 200
    python AD_exploration.py --n_samples 100 --skip_subspace
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
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Extract HS + labels
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hidden_states(model, tokenizer, samples, dataset, device,
                          layers_to_scan, max_new_tokens=20):
    """Extract HS from all layers during generation. Returns all_h, labels, metadata."""
    all_h = {li: [] for li in layers_to_scan}
    labels = []
    correct_count = 0
    answers_list = []

    t0 = time.time()
    for s in tqdm(samples, desc=f"Extract {dataset}"):
        prompt = format_prompt(s["question"], s["context"], dataset=dataset)
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # First forward pass: hook ALL layers
        residual = {}
        fwd_hooks = []
        for li in layers_to_scan:
            def _hook(act, hook=None, _layer=li):
                residual[_layer] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_resid_post", _hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        for li in layers_to_scan:
            all_h[li].append(residual[li].float().cpu().numpy().flatten())

        # Generate
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(max_new_tokens - 1):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        answers_list.append(ans)
        is_correct = check_correct(ans, s["answers"], dataset=dataset)
        if is_correct:
            correct_count += 1
        labels.append(1 if is_correct else 0)

    for li in layers_to_scan:
        all_h[li] = np.stack(all_h[li], axis=0)

    elapsed = time.time() - t0
    return {
        "all_h": all_h,
        "labels": np.array(labels),
        "answers": answers_list,
        "correct_count": correct_count,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Direction A: Truth Subspace
# ═══════════════════════════════════════════════════════════════════════════════

def compute_orthogonal_directions(H, labels, k_max=20):
    """Compute k orthogonal discriminative directions via sequential mean diff.

    Algorithm:
      v₁ = μ_correct - μ_incorrect (normalized)
      For j = 2..k_max:
        Project H to remove v₁..v_{j-1} components
        Recompute means, take new difference → vⱼ

    Returns:
        directions: np.ndarray [d, k_max] — k_max orthogonal unit vectors
        scores: np.ndarray [N, k_max] — per-direction projection scores
        auroc_per_k: list of AUROC(k) for cumulative top-k
    """
    N, d = H.shape
    mask_c = labels == 1
    mask_i = labels == 0

    directions = np.zeros((d, k_max), dtype=np.float64)
    all_scores = np.zeros((N, k_max), dtype=np.float64)
    auroc_per_k = []

    H_work = H.copy()

    for k in range(k_max):
        # Mean difference on working data
        mu_c = H_work[mask_c].mean(axis=0)
        mu_i = H_work[mask_i].mean(axis=0)
        v = mu_c - mu_i
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            break
        v = v / v_norm

        # Store direction
        directions[:, k] = v

        # Project original data onto this direction → scores
        all_scores[:, k] = H @ v

        # Cumulative AUROC: combine top-(k+1) directions via mean of absolute scores
        # (use absolute because direction sign is arbitrary)
        scores_cumulative = np.abs(all_scores[:, :k+1]).mean(axis=1)

        valid = np.isfinite(scores_cumulative)
        if valid.sum() >= 10 and labels[valid].std() > 0:
            auroc = float(roc_auc_score(labels[valid], scores_cumulative[valid]))
            auroc = max(auroc, 1 - auroc)
            auroc_per_k.append(auroc)
        else:
            auroc_per_k.append(float("nan"))

        # Remove this direction from working data
        projections = H_work @ v  # [N]
        H_work = H_work - projections[:, None] * v[None, :]

    return directions, all_scores, auroc_per_k


def compute_per_class_pca(H, labels, q=10):
    """Compute top-q PCA directions for each class.

    Returns:
        pc_correct: [d, q] — top PCA directions for correct class
        pc_incorrect: [d, q] — top PCA directions for incorrect class
    """
    mask_c = labels == 1
    mask_i = labels == 0

    results = {}
    for name, mask in [("correct", mask_c), ("incorrect", mask_i)]:
        H_class = H[mask]
        H_centered = H_class - H_class.mean(axis=0)
        # Compute top-q PCA via SVD (avoids full covariance for memory)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        results[name] = Vt[:q].T  # [d, q] — top-q principal components

    return results["correct"], results["incorrect"]


# ═══════════════════════════════════════════════════════════════════════════════
# Direction D: Truth Direction Intervention
# ═══════════════════════════════════════════════════════════════════════════════

def intervene_and_generate(model, tokenizer, samples, dataset, device,
                           layer, v, alpha, mode="additive", max_new_tokens=20):
    """Generate with intervention at specified layer.

    Args:
        mode: "additive" → h' = h + α·v
              "amplify"  → h' = h + α·(v·h)·v
              "subtract" → h' = h - α·v

    Returns:
        dict with keys: correct_count, answers, per_sample_details
    """
    results = []
    correct_count = 0

    for s in tqdm(samples, desc=f"Intervene α={alpha:+0.1f} {mode}"):
        prompt = format_prompt(s["question"], s["context"], dataset=dataset)
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # ── Intervene during first forward pass ──
        def _intervene_hook(act, hook=None):
            h = act[:, -1, :]  # [1, d]
            if mode == "additive":
                h_modified = h + alpha * v
            elif mode == "amplify":
                proj = (h @ v)  # scalar
                h_modified = h + alpha * proj * v
            elif mode == "subtract":
                h_modified = h - alpha * v
            else:
                h_modified = h
            act[:, -1, :] = h_modified
            return act

        fwd_hooks = [(f"blocks.{layer}.hook_resid_post", _intervene_hook)]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]

        # Continue generation without intervention
        for _ in range(max_new_tokens - 1):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset=dataset)
        if is_correct:
            correct_count += 1

        results.append({
            "answer": ans,
            "is_correct": is_correct,
            "num_tokens": len(gids),
        })

    return {
        "correct_count": correct_count,
        "accuracy": correct_count / len(samples),
        "per_sample": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A+D: Truth Subspace + Intervention")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_subspace", action="store_true")
    parser.add_argument("--skip_intervention", action="store_true")
    parser.add_argument("--intervention_layer", type=int, default=20,
                        help="Layer for intervention (default: best from C2, L20)")
    parser.add_argument("--intervention_n", type=int, default=100,
                        help="Samples for intervention sweep (fewer = faster)")
    parser.add_argument("--k_max", type=int, default=20,
                        help="Max subspace dimensions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"A+D: Truth Subspace + Truth Direction Intervention")
    print(f"  Model: {args.model}  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # ── Load model + data ──
    print("Loading model...")
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    layers_to_scan = list(range(n_layers))
    print(f"  Loaded. {n_layers} layers")

    tq_samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  TriviaQA: {len(tq_samples)} samples")

    # ── Phase 1: Extract HS ──
    print(f"\n{'─'*40}")
    print(f"Phase 1: Extracting hidden states (all {n_layers} layers)")
    data = extract_hidden_states(
        model, tokenizer, tq_samples, "triviaqa",
        device, layers_to_scan,
    )
    labels = data["labels"]
    print(f"  Correct: {data['correct_count']}/{args.n_samples} "
          f"({data['correct_count']/args.n_samples:.1%})")
    print(f"  Time: {data['elapsed']:.0f}s")

    # ═══════════════════════════════════════════════════════════════════════════
    # Direction A: Truth Subspace
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.skip_subspace:
        print(f"\n{'='*60}")
        print(f"Direction A: Truth Subspace (1D → k-D)")
        print(f"{'='*60}")

        # ── A.1: Per-layer subspace scan ──
        print(f"\nA.1: Orthogonal Sequential Directions (k=1..{args.k_max})")
        print(f"  {'Layer':>5s}  {'k=1':>8s}  {'k=5':>8s}  {'k=10':>8s}  "
              f"{'k=20':>8s}  {'Best k':>8s}  {'Best':>8s}")
        print(f"  {'─'*60}")

        best_layer_1d = -1
        best_auroc_1d = 0.0
        best_layer_kd = -1
        best_auroc_kd = 0.0
        best_k_overall = -1
        layer_subspace_results = []

        for li in sorted(layers_to_scan):
            H = data["all_h"][li]
            mask_c = labels == 1
            mask_i = labels == 0
            if mask_c.sum() < 2 or mask_i.sum() < 2:
                continue

            _, _, auroc_per_k = compute_orthogonal_directions(
                H, labels, k_max=args.k_max)

            if len(auroc_per_k) == 0 or np.isnan(auroc_per_k[0]):
                continue

            # Track best
            k1_auroc = auroc_per_k[0]
            if k1_auroc > best_auroc_1d:
                best_auroc_1d = k1_auroc
                best_layer_1d = li

            best_local_k = int(np.argmax(auroc_per_k))
            best_local_auroc = auroc_per_k[best_local_k]
            if best_local_auroc > best_auroc_kd:
                best_auroc_kd = best_local_auroc
                best_layer_kd = li
                best_k_overall = best_local_k

            k5_idx = min(4, len(auroc_per_k) - 1)
            k10_idx = min(9, len(auroc_per_k) - 1)
            k20_idx = min(19, len(auroc_per_k) - 1)

            print(f"  {li:>5d}  {auroc_per_k[0]:>8.4f}  "
                  f"{auroc_per_k[k5_idx]:>8.4f}  "
                  f"{auroc_per_k[k10_idx]:>8.4f}  "
                  f"{auroc_per_k[k20_idx]:>8.4f}  "
                  f"{best_local_k:>8d}  {best_local_auroc:>8.4f}")

            layer_subspace_results.append({
                "layer": li,
                "auroc_per_k": [float(x) if not np.isnan(x) else None
                                for x in auroc_per_k],
                "best_k": best_local_k,
                "best_auroc": best_local_auroc,
            })

        print(f"\n  Best 1D:     L{best_layer_1d}  AUROC={best_auroc_1d:.4f}")
        print(f"  Best k-D:    L{best_layer_kd}  k={best_k_overall}  "
              f"AUROC={best_auroc_kd:.4f}")
        delta = best_auroc_kd - best_auroc_1d
        print(f"  Δ(k-D - 1D): {delta:+.4f} "
              f"({'✅ multidimensional helps' if delta > 0.005 else '≈ same as 1D'})")

        # ── A.2: Detailed k-sweep at best layer ──
        best_layer = best_layer_kd if best_auroc_kd > best_auroc_1d else best_layer_1d
        H_best = data["all_h"][best_layer]
        _, scores_best, auroc_best_k = compute_orthogonal_directions(
            H_best, labels, k_max=args.k_max)

        print(f"\nA.2: AUROC(k) curve at L{best_layer}")
        print(f"  k    AUROC    Δk")
        print(f"  {'─'*22}")
        prev = 0.0
        for k_idx, auroc in enumerate(auroc_best_k):
            if np.isnan(auroc):
                break
            delta_k = auroc - prev
            marker = " <--" if k_idx == 0 else (" ▲" if delta_k > 0.005 else "")
            print(f"  {k_idx+1:>3d}  {auroc:.4f}  {delta_k:+.4f}{marker}")
            prev = auroc

        # ── A.3: Per-class PCA analysis ──
        print(f"\nA.3: Per-class PCA structure (L{best_layer})")
        pc_correct, pc_incorrect = compute_per_class_pca(H_best, labels, q=10)

        # How much variance does each class's top PCA span?
        for name, pc in [("correct", pc_correct), ("incorrect", pc_incorrect)]:
            mask = (labels == 1) if name == "correct" else (labels == 0)
            H_class = H_best[mask]
            H_centered = H_class - H_class.mean(axis=0)
            total_var = (H_centered ** 2).sum()
            explained = []
            H_proj = H_centered.copy()
            for j in range(10):
                proj = H_proj @ pc[:, j]
                var_j = (proj ** 2).sum()
                explained.append(float(var_j / total_var))
                H_proj = H_proj - proj[:, None] * pc[:, j][None, :]
            cumsum = np.cumsum(explained)
            print(f"  {name:12s}  PC1={explained[0]:.3f}  "
                  f"PC5={cumsum[4]:.3f}  PC10={cumsum[9]:.3f}")

        # ── A.4: Direction analysis — which tokens do top directions encode? ──
        W_U = model.unembed.W_U.detach().float().cpu().numpy()  # [d, vocab]
        _, top_scores, _ = compute_orthogonal_directions(
            H_best, labels, k_max=min(5, args.k_max))
        # Get directions from the top scores (we redo compute_orthogonal for v access)
        directions_A4, _, _ = compute_orthogonal_directions(
            H_best, labels, k_max=5)

        print(f"\nA.4: Top tokens per direction (projected to vocab)")
        for d_idx in range(min(5, directions_A4.shape[1])):
            v_dir = directions_A4[:, d_idx]
            token_scores = v_dir @ W_U  # [vocab]
            top_indices = np.argsort(token_scores)[-15:][::-1]
            top_tokens = [tokenizer.decode([int(i)]).replace("\n", "\\n")
                          for i in top_indices]
            bottom_indices = np.argsort(token_scores)[:15]
            bottom_tokens = [tokenizer.decode([int(i)]).replace("\n", "\\n")
                             for i in bottom_indices]
            print(f"  v{d_idx+1} (+): {', '.join(top_tokens[:8])}")
            print(f"  v{d_idx+1} (-): {', '.join(bottom_tokens[:8])}")
            print()

        # Save A results
        a_save = output_dir / "A_truth_subspace.json"
        with open(a_save, "w") as f:
            json.dump({
                "best_layer_1d": best_layer_1d,
                "best_auroc_1d": best_auroc_1d,
                "best_layer_kd": best_layer_kd,
                "best_auroc_kd": best_auroc_kd,
                "best_k": best_k_overall,
                "delta": float(delta),
                "auroc_curve_best_layer": [float(x) if not np.isnan(x) else None
                                           for x in auroc_best_k],
                "per_layer": layer_subspace_results,
            }, f, indent=2)
        print(f"\n  Saved A: {a_save}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Direction D: Truth Direction Intervention
    # ═══════════════════════════════════════════════════════════════════════════
    if not args.skip_intervention:
        print(f"\n{'='*60}")
        print(f"Direction D: Truth Direction Intervention")
        print(f"{'='*60}")

        intervention_layer = args.intervention_layer
        intervention_n = min(args.intervention_n, args.n_samples)
        intervention_samples = load_triviaqa(
            n_samples=intervention_n, seed=args.seed + 99)

        # Compute v at intervention layer
        H_int = data["all_h"][intervention_layer]
        mask_c = labels == 1
        mask_i = labels == 0
        v = H_int[mask_c].mean(axis=0) - H_int[mask_i].mean(axis=0)
        v = v / np.linalg.norm(v)
        v = torch.from_numpy(v).float().to(device)
        print(f"  Truth direction computed at L{intervention_layer}")

        # ── Baseline (no intervention) ──
        print(f"\nD.0: Baseline (no intervention, {intervention_n} samples)")
        baseline = intervene_and_generate(
            model, tokenizer, intervention_samples, "triviaqa",
            device, intervention_layer, v, alpha=0.0, mode="additive",
        )
        base_acc = baseline["accuracy"]
        print(f"  Baseline accuracy: {base_acc:.3f} ({baseline['correct_count']}/{intervention_n})")

        # ── D.1-D.3: Sweep alpha ──
        alphas = [-2.0, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        modes = ["additive", "subtract", "amplify"]

        print(f"\nD.1-D.3: Intervention sweep")
        print(f"  {'Mode':>12s}  {'α':>6s}  {'Acc':>8s}  {'Δ base':>8s}  "
              f"{'Correct':>8s}")
        print(f"  {'─'*50}")

        all_intervention_results = []
        best_acc = base_acc
        best_config = ("none", 0.0)

        for mode in modes:
            for alpha in alphas:
                result = intervene_and_generate(
                    model, tokenizer, intervention_samples, "triviaqa",
                    device, intervention_layer, v, alpha=alpha, mode=mode,
                )
                acc = result["accuracy"]
                delta = acc - base_acc
                marker = " ✅" if delta > 0.02 else (" ⬆" if delta > 0 else "")

                print(f"  {mode:>12s}  {alpha:>+6.1f}  {acc:>8.3f}  "
                      f"{delta:>+8.3f}{marker}  {result['correct_count']:>8d}")

                all_intervention_results.append({
                    "mode": mode,
                    "alpha": alpha,
                    "accuracy": acc,
                    "delta": delta,
                    "correct_count": result["correct_count"],
                })

                if acc > best_acc:
                    best_acc = acc
                    best_config = (mode, alpha)

        print(f"\n  Baseline: {base_acc:.3f}")
        print(f"  Best:     {best_config[0]}, α={best_config[1]:+.1f} → {best_acc:.3f} "
              f"(Δ={best_acc-base_acc:+.3f})")

        if best_acc > base_acc + 0.02:
            print(f"  ✅ Intervention IMPROVES accuracy!")
        elif best_acc > base_acc - 0.02:
            print(f"  ≈ Intervention has no significant effect")
        else:
            print(f"  ❌ Intervention DEGRADES accuracy")

        # ── D.4: Multi-layer intervention ──
        print(f"\nD.4: Multi-layer intervention (L17-L22, additive)")
        ml_alphas = [-0.5, -0.2, 0.0, 0.2, 0.5]
        ml_layers = list(range(17, 23))  # L17-L22 sweet spot

        # Compute per-layer v
        ml_vs = {}
        for li in ml_layers:
            H_li = data["all_h"][li]
            v_li = H_li[mask_c].mean(axis=0) - H_li[mask_i].mean(axis=0)
            v_li = v_li / np.linalg.norm(v_li)
            ml_vs[li] = torch.from_numpy(v_li).float().to(device)

        print(f"  {'α':>6s}  {'Acc':>8s}  {'Δ base':>8s}  {'Correct':>8s}")
        print(f"  {'─'*38}")

        for alpha in ml_alphas:
            # Custom multi-layer intervention
            correct_ml = 0
            for s in tqdm(intervention_samples, desc=f"ML α={alpha:+0.1f}"):
                prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
                tokens = model.to_tokens(prompt, prepend_bos=True)
                if tokens.shape[1] > 1024:
                    tokens = tokens[:, :1024]

                # Hook all target layers
                fwd_hooks = []
                for li in ml_layers:
                    v_li = ml_vs[li]
                    def _make_ml_hook(layer_idx, v_vec):
                        def hook(act, hook=None):
                            h = act[:, -1, :]
                            act[:, -1, :] = h + alpha * v_vec
                            return act
                        return hook
                    fwd_hooks.append(
                        (f"blocks.{li}.hook_resid_post",
                         _make_ml_hook(li, v_li)))

                with torch.no_grad():
                    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                nid = int(logits[0, -1, :].argmax().item())
                gids = [nid]
                for _ in range(19):
                    if nid == tokenizer.eos_token_id:
                        break
                    tokens = torch.cat(
                        [tokens, torch.tensor([[nid]], device=device)], dim=1)
                    with torch.no_grad():
                        logits = model(tokens)
                    nid = int(logits[0, -1, :].argmax().item())
                    gids.append(nid)

                ans = tokenizer.decode(gids).strip()
                if check_correct(ans, s["answers"], dataset="triviaqa"):
                    correct_ml += 1

            acc_ml = correct_ml / len(intervention_samples)
            delta_ml = acc_ml - base_acc
            print(f"  {alpha:>+6.1f}  {acc_ml:>8.3f}  {delta_ml:>+8.3f}  "
                  f"{correct_ml:>8d}")

        # Save D results
        d_save = output_dir / "D_intervention.json"
        with open(d_save, "w") as f:
            json.dump({
                "intervention_layer": intervention_layer,
                "baseline_accuracy": base_acc,
                "baseline_correct": baseline["correct_count"],
                "best_mode": best_config[0],
                "best_alpha": best_config[1],
                "best_accuracy": best_acc,
                "best_delta": best_acc - base_acc,
                "single_layer_results": all_intervention_results,
            }, f, indent=2)
        print(f"\n  Saved D: {d_save}")

    print(f"\n{'='*60}")
    print(f"A+D complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
