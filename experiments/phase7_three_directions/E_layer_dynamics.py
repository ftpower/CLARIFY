"""Direction E: Layer-wise Truth Signal Dynamics.

Analyzes how the truth direction v evolves across layers, which components
contribute to it, and what it encodes.

E.1 — Inter-layer rotation: cos(v_L, v_{L+1}) across all adjacent layers
E.2 — Residual decomposition: attn_out vs mlp_out contribution to v
E.3 — Vocabulary projection: v @ W_U → semantic interpretation
E.4 — Knowledge layer comparison: v with known factual knowledge layers (ROME)
E.5 — Signal sparsity: dimension ablation on v

Usage:
    python E_layer_dynamics.py --n_samples 200
    python E_layer_dynamics.py --n_samples 100 --decomp_layers 10 20 27
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
# Phase 1: Multi-hook extraction (HS + attn_out + mlp_out + labels)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_hooks(model, tokenizer, samples, device, n_layers,
                      decomp_layers, max_new_tokens=20):
    """Single-pass extraction of resid_post, attn_out, mlp_out + generation labels.

    Returns:
        all_h: {layer: np.ndarray [N, d]} — resid_post at every layer
        all_attn: {layer: np.ndarray [N, d]} — attn_out at selected layers
        all_mlp: {layer: np.ndarray [N, d]} — mlp_out at selected layers
        labels: np.ndarray [N]
    """
    all_h = {li: [] for li in range(n_layers)}
    all_attn = {li: [] for li in decomp_layers}
    all_mlp = {li: [] for li in decomp_layers}
    labels = []
    correct_count = 0

    t0 = time.time()
    for s in tqdm(samples, desc="Extract all hooks"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # Build hooks for all resid_post + selected attn_out + mlp_out
        residual = {}
        fwd_hooks = []

        for li in range(n_layers):
            def _resid_hook(act, hook=None, _layer=li):
                residual[("resid", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_resid_post", _resid_hook))

        for li in decomp_layers:
            def _attn_hook(act, hook=None, _layer=li):
                residual[("attn", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_attn_out", _attn_hook))

            def _mlp_hook(act, hook=None, _layer=li):
                residual[("mlp", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_mlp_out", _mlp_hook))

        # ── Forward pass with all hooks ──
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Store resid_post
        for li in range(n_layers):
            all_h[li].append(residual[("resid", li)].float().cpu().numpy().flatten())

        # Store attn_out, mlp_out
        for li in decomp_layers:
            all_attn[li].append(residual[("attn", li)].float().cpu().numpy().flatten())
            all_mlp[li].append(residual[("mlp", li)].float().cpu().numpy().flatten())

        # ── Generate for correctness label ──
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
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")
        if is_correct:
            correct_count += 1
        labels.append(1 if is_correct else 0)

    # Stack
    for li in range(n_layers):
        all_h[li] = np.stack(all_h[li], axis=0)
    for li in decomp_layers:
        all_attn[li] = np.stack(all_attn[li], axis=0)
        all_mlp[li] = np.stack(all_mlp[li], axis=0)

    elapsed = time.time() - t0
    return {"all_h": all_h, "all_attn": all_attn, "all_mlp": all_mlp,
            "labels": np.array(labels), "correct_count": correct_count,
            "elapsed": elapsed}


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_v(H, labels):
    """Compute truth direction v = mean(correct) - mean(incorrect), normalized."""
    mask_c = labels == 1
    mask_i = labels == 0
    if mask_c.sum() < 2 or mask_i.sum() < 2:
        return None
    v = H[mask_c].mean(axis=0) - H[mask_i].mean(axis=0)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    return v


# ── E.1: Inter-layer rotation ──

def analyze_layer_rotation(all_h, labels):
    """Compute v at every layer and measure cosine between adjacent layers."""
    n_layers = len(all_h)
    vs = {}
    aurocs = {}

    for li in range(n_layers):
        v = compute_v(all_h[li], labels)
        if v is None:
            continue
        vs[li] = v
        scores = all_h[li] @ v
        valid = np.isfinite(scores)
        if valid.sum() >= 10 and labels[valid].std() > 0:
            a = float(roc_auc_score(labels[valid], scores[valid]))
            aurocs[li] = max(a, 1 - a)

    # Adjacent cosine similarities
    rotations = []
    sorted_layers = sorted(vs.keys())
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        cos_sim = float(np.dot(vs[l1], vs[l2]))
        angle = float(np.arccos(np.clip(abs(cos_sim), -1, 1)) * 180 / np.pi)
        rotations.append({
            "layer_from": l1, "layer_to": l2,
            "cosine_sim": cos_sim, "angle_deg": angle,
        })
    return vs, aurocs, rotations


# ── E.2: Residual decomposition ──

def analyze_residual_decomp(all_h, all_attn, all_mlp, labels, decomp_layers):
    """Decompose v into attention and MLP contributions.

    In the residual stream: h_L = h_{L-1} + attn_out_L + mlp_out_L
    So v_L should approximately align with v_attn + v_mlp (computed from
    attn_out and mlp_out separately).

    Also measures: how much of v_L's direction comes from attn vs mlp?
    """
    results = {}
    for li in decomp_layers:
        v_resid = compute_v(all_h[li], labels)
        v_attn = compute_v(all_attn[li], labels)
        v_mlp = compute_v(all_mlp[li], labels)

        if v_resid is None:
            results[li] = {"error": "insufficient samples"}
            continue

        entry = {"v_resid_norm": float(np.linalg.norm(v_resid))}

        # Alignment of v_attn, v_mlp with v_resid
        if v_attn is not None:
            entry["v_attn_cos"] = float(np.dot(v_resid, v_attn))
            entry["v_attn_angle"] = float(
                np.arccos(np.clip(abs(entry["v_attn_cos"]), -1, 1)) * 180 / np.pi)
        if v_mlp is not None:
            entry["v_mlp_cos"] = float(np.dot(v_resid, v_mlp))
            entry["v_mlp_angle"] = float(
                np.arccos(np.clip(abs(entry["v_mlp_cos"]), -1, 1)) * 180 / np.pi)

        # Which contributes more? Project v_resid onto v_attn and v_mlp
        if v_attn is not None and v_mlp is not None:
            # Decompose v_resid in the span of {v_attn, v_mlp}
            # v_resid ≈ a * v_attn + m * v_mlp (in the projection sense)
            attn_proj = float(np.dot(v_resid, v_attn))  # cosine if both unit
            mlp_proj = float(np.dot(v_resid, v_mlp))
            entry["attn_contribution"] = abs(attn_proj) / (abs(attn_proj) + abs(mlp_proj) + 1e-10)
            entry["mlp_contribution"] = abs(mlp_proj) / (abs(attn_proj) + abs(mlp_proj) + 1e-10)
            entry["dominant"] = "attn" if entry["attn_contribution"] > entry["mlp_contribution"] else "mlp"

        # Also compute: which component correlates more with correctness?
        if v_attn is not None:
            scores_attn = all_attn[li] @ v_attn
            valid = np.isfinite(scores_attn)
            if valid.sum() >= 10 and labels[valid].std() > 0:
                a = float(roc_auc_score(labels[valid], scores_attn[valid]))
                entry["auroc_attn"] = max(a, 1 - a)

        if v_mlp is not None:
            scores_mlp = all_mlp[li] @ v_mlp
            valid = np.isfinite(scores_mlp)
            if valid.sum() >= 10 and labels[valid].std() > 0:
                a = float(roc_auc_score(labels[valid], scores_mlp[valid]))
                entry["auroc_mlp"] = max(a, 1 - a)

        results[li] = entry
    return results


# ── E.3: Vocabulary projection ──

def analyze_v_vocab(v, W_U, tokenizer, top_k=20):
    """Project v onto unembedding matrix → top tokens."""
    # v: [d], W_U: [d, vocab]
    token_scores = v @ W_U  # [vocab]
    top_pos = np.argsort(token_scores)[-top_k:][::-1]
    top_neg = np.argsort(token_scores)[:top_k]

    def _decode(indices):
        return [tokenizer.decode([int(i)]).replace("\n", "\\n").replace("\t", "\\t")
                for i in indices]

    return {
        "top_positive": _decode(top_pos),
        "top_negative": _decode(top_neg),
        "top_pos_scores": [float(token_scores[i]) for i in top_pos],
        "top_neg_scores": [float(token_scores[i]) for i in top_neg],
    }


# ── E.5: Signal sparsity ──

def analyze_sparsity(H, labels, v, n_steps=20):
    """Dimension ablation: zero out top-k dimensions, measure AUROC decay.

    If top-200 dimensions capture 80% of the AUROC, v is sparse and interpretable.
    """
    d = v.shape[0]
    # Sort dimensions by absolute contribution to v
    dim_order = np.argsort(np.abs(v))[::-1]  # most important first

    n_kept_list = []
    auroc_list = []

    # Test progressively: keep only top-n dims, zero rest
    fractions = np.linspace(0.02, 1.0, n_steps)
    for frac in fractions:
        n_kept = max(1, int(d * frac))
        mask = np.zeros(d, dtype=bool)
        mask[dim_order[:n_kept]] = True
        v_sparse = v.copy()
        v_sparse[~mask] = 0
        v_sparse = v_sparse / (np.linalg.norm(v_sparse) + 1e-10)

        scores = H @ v_sparse
        valid = np.isfinite(scores)
        if valid.sum() < 10 or labels[valid].std() == 0:
            auroc_list.append(float("nan"))
        else:
            a = float(roc_auc_score(labels[valid], scores[valid]))
            auroc_list.append(max(a, 1 - a))
        n_kept_list.append(n_kept)

    # Find elbow: n at which AUROC reaches 95% of full
    full_auroc = auroc_list[-1]
    elbow_idx = None
    if not np.isnan(full_auroc):
        for i, a in enumerate(auroc_list):
            if not np.isnan(a) and a >= 0.95 * full_auroc:
                elbow_idx = i
                break

    return {
        "full_auroc": float(full_auroc) if not np.isnan(full_auroc) else None,
        "n_kept": [int(x) for x in n_kept_list],
        "auroc": [float(x) if not np.isnan(x) else None for x in auroc_list],
        "elbow_n_dims": int(n_kept_list[elbow_idx]) if elbow_idx is not None else None,
        "elbow_fraction": float(fractions[elbow_idx]) if elbow_idx is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E: Layer-wise Truth Dynamics")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decomp_layers", type=int, nargs="+",
                        default=[10, 20, 27],
                        help="Layers for attn/mlp decomposition (E.2)")
    parser.add_argument("--sparsity_layers", type=int, nargs="+",
                        default=[10, 20, 27],
                        help="Layers for sparsity analysis (E.5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Direction E: Layer-wise Truth Signal Dynamics")
    print(f"  Model: {args.model}  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # ── Load model + data ──
    print("Loading model...")
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  {n_layers} layers, d={d_model}")

    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)

    # ── Phase 1: Extract all hooks ──
    print(f"\nExtracting HS (all {n_layers}L) + attn/mlp ({args.decomp_layers})...")
    data = extract_all_hooks(
        model, tokenizer, samples, device, n_layers, args.decomp_layers)
    labels = data["labels"]
    print(f"  Correct: {data['correct_count']}/{args.n_samples} "
          f"({data['correct_count']/args.n_samples:.1%})")
    print(f"  Extraction time: {data['elapsed']:.0f}s")

    # ═══════════════════════════════════════════════════════════════════════════
    # E.1: Inter-layer rotation
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"E.1: Inter-layer Truth Direction Rotation")
    print(f"{'='*60}")

    vs, aurocs, rotations = analyze_layer_rotation(data["all_h"], labels)

    print(f"\n  {'From':>5s} → {'To':>5s}  {'cos(v₁,v₂)':>12s}  {'Angle°':>8s}  "
          f"{'AUROC₁':>8s}  {'AUROC₂':>8s}  {'Δ A':>8s}")
    print(f"  {'─'*62}")

    # Find the biggest rotation jumps
    big_jumps = []
    for r in rotations:
        l1, l2 = r["layer_from"], r["layer_to"]
        a1 = aurocs.get(l1, float("nan"))
        a2 = aurocs.get(l2, float("nan"))
        delta_a = (a2 - a1) if not np.isnan(a1) and not np.isnan(a2) else float("nan")

        print(f"  {l1:>5d} → {l2:>5d}  {r['cosine_sim']:>12.4f}  "
              f"{r['angle_deg']:>8.1f}  {a1:>8.4f}  {a2:>8.4f}  "
              f"{delta_a:>+8.4f}")

        if r["angle_deg"] > 30:
            big_jumps.append({**r, "auroc_from": a1, "auroc_to": a2,
                              "auroc_delta": delta_a})

    if big_jumps:
        print(f"\n  Large rotations (>30°):")
        for j in big_jumps:
            print(f"    L{j['layer_from']}→L{j['layer_to']}: "
                  f"{j['angle_deg']:.0f}°  "
                  f"(AUROC {j['auroc_from']:.3f}→{j['auroc_to']:.3f}, "
                  f"Δ={j['auroc_delta']:+.3f})")

    # ── Overall statistics ──
    angles = [r["angle_deg"] for r in rotations]
    cosines = [r["cosine_sim"] for r in rotations]
    print(f"\n  Rotation stats across {len(rotations)} adjacent pairs:")
    print(f"    Mean angle: {np.mean(angles):.1f}°  "
          f"Median: {np.median(angles):.1f}°  "
          f"Max: {np.max(angles):.1f}°")
    print(f"    Mean cos:  {np.mean(cosines):.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # E.2: Residual decomposition (attn vs mlp)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"E.2: Residual Stream Decomposition (Attn vs MLP)")
    print(f"{'='*60}")

    decomp_results = analyze_residual_decomp(
        data["all_h"], data["all_attn"], data["all_mlp"],
        labels, args.decomp_layers)

    print(f"\n  {'Layer':>5s}  {'v_attn cos':>10s}  {'v_mlp cos':>10s}  "
          f"{'Dominant':>10s}  {'AUROC_attn':>10s}  {'AUROC_mlp':>10s}")
    print(f"  {'─'*62}")

    for li in sorted(decomp_results.keys()):
        r = decomp_results[li]
        if "error" in r:
            print(f"  {li:>5d}  {r['error']}")
            continue
        attn_cos = f"{r.get('v_attn_cos', 0):.4f}" if 'v_attn_cos' in r else "N/A"
        mlp_cos = f"{r.get('v_mlp_cos', 0):.4f}" if 'v_mlp_cos' in r else "N/A"
        dominant = r.get("dominant", "?")
        auroc_a = f"{r.get('auroc_attn', 0):.4f}" if 'auroc_attn' in r else "N/A"
        auroc_m = f"{r.get('auroc_mlp', 0):.4f}" if 'auroc_mlp' in r else "N/A"
        print(f"  {li:>5d}  {attn_cos:>10s}  {mlp_cos:>10s}  "
              f"{dominant:>10s}  {auroc_a:>10s}  {auroc_m:>10s}")

    # ═══════════════════════════════════════════════════════════════════════════
    # E.3: Vocabulary projection
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"E.3: Truth Direction → Vocabulary Projection")
    print(f"{'='*60}")

    W_U = model.unembed.W_U.detach().float().cpu().numpy()  # [d, vocab]

    # Show v vocab for key layers: early, best, late
    vocab_layers = [0, 10, 20, 27]
    for li in vocab_layers:
        if li not in vs:
            continue
        v = vs[li]
        vocab_info = analyze_v_vocab(v, W_U, tokenizer, top_k=15)
        auroc_li = aurocs.get(li, float("nan"))

        print(f"\n  L{li} (AUROC={auroc_li:.4f}):")
        print(f"    v+ (top tokens):  {', '.join(vocab_info['top_positive'][:10])}")
        print(f"    v- (bottom tokens): {', '.join(vocab_info['top_negative'][:10])}")

    # ═══════════════════════════════════════════════════════════════════════════
    # E.4: Knowledge layer comparison
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"E.4: Comparison with Known Knowledge Storage Layers")
    print(f"{'='*60}")

    print(f"""
  Literature findings (ROME, Meng et al. 2022):
    - Factual knowledge is primarily stored in mid-upper MLP layers
    - For GPT-style models: layers ~30-70% of depth store most facts
    - Knowledge is localised: specific (subject, relation, object) triples
      can be edited by modifying a single MLP layer

  Our Truth Direction sweet spot: L17-L22 (61-79% of 28L)
    - This overlaps with the known "knowledge storage" region
    - Suggests: truth detection signal is strongest where facts are retrieved

  L27 collapse:
    - Last layer v is nearly orthogonal to mid-layer v
    - Last layer is dominated by next-token prediction, not fact encoding
    - This is consistent with the "early exit" hypothesis:
      factual decisions are made by mid-layers, final layers just decode
""")

    # Quantify: cosine between mid-layer v (avg L17-L22) and L27 v
    if all(li in vs for li in [17, 20, 22, 27]):
        v_mid = np.mean([vs[li] for li in range(17, 23)], axis=0)
        v_mid = v_mid / np.linalg.norm(v_mid)
        v_last = vs[27]
        cos_mid_last = float(np.dot(v_mid, v_last))
        angle_mid_last = float(np.arccos(np.clip(abs(cos_mid_last), -1, 1)) * 180 / np.pi)
        print(f"  Quantitative confirmation:")
        print(f"    cos(v_mid_L17-22, v_L27) = {cos_mid_last:.4f}")
        print(f"    Angle = {angle_mid_last:.1f}°")
        if angle_mid_last > 45:
            print(f"    → Mid-layer truth direction is NEARLY ORTHOGONAL to last layer")
            print(f"    → Last layer encodes something fundamentally different from truth")

    # ═══════════════════════════════════════════════════════════════════════════
    # E.5: Signal sparsity
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"E.5: Signal Sparsity — Dimension Ablation")
    print(f"{'='*60}")

    sparsity_results = {}
    for li in args.sparsity_layers:
        if li not in vs or li not in data["all_h"]:
            continue
        v = vs[li]
        H = data["all_h"][li]
        sp = analyze_sparsity(H, labels, v)
        sparsity_results[li] = sp

        full_a = sp["full_auroc"]
        elbow_n = sp["elbow_n_dims"]
        elbow_f = sp["elbow_fraction"]

        print(f"\n  L{li}:")
        if full_a is not None:
            print(f"    Full AUROC = {full_a:.4f}")
        if elbow_n is not None:
            print(f"    95% AUROC with {elbow_n} dims ({elbow_f:.1%} of {d_model})")
        else:
            print(f"    No clear elbow — signal is diffuse across dimensions")

        # Show first few data points
        for i in [0, 4, 9, 14, 19]:
            if i < len(sp["n_kept"]):
                nk = sp["n_kept"][i]
                a = sp["auroc"][i]
                a_str = f"{a:.4f}" if a is not None else "nan"
                print(f"    keep {nk:>5d} dims ({nk/d_model:.1%}): AUROC = {a_str}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════════════
    save_path = output_dir / "E_layer_dynamics.json"
    with open(save_path, "w") as f:
        json.dump({
            "E1_rotations": rotations,
            "E1_auroc_per_layer": {str(k): v for k, v in aurocs.items()},
            "E1_big_jumps": big_jumps,
            "E1_mean_angle": float(np.mean(angles)),
            "E1_max_angle": float(np.max(angles)),
            "E2_decomposition": {
                str(li): {k: v for k, v in r.items()}
                for li, r in decomp_results.items()
            },
            "E4_cos_mid_vs_last": cos_mid_last if 'cos_mid_last' in dir() else None,
            "E4_angle_mid_vs_last": angle_mid_last if 'angle_mid_last' in dir() else None,
            "E5_sparsity": {
                str(li): {k: v for k, v in sp.items() if k != "n_kept" and k != "auroc"}
                for li, sp in sparsity_results.items()
            },
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"Direction E complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
