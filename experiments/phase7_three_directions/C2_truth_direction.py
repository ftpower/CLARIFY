"""C2: Truth Direction — linear direction in residual stream for correctness.

Hypothesis: There exists a linear direction v in the residual stream such that
projecting the last-token hidden state onto v gives a scalar confidence score.

Algorithm:
  1. Compute truth direction v = mean(h_correct) - mean(h_incorrect) on training
  2. Project test hidden states: score = dot(h, v) / ||v||
  3. Higher projection = more "truth-like" = more likely correct

Inspired by: TruthPrInt (ICCV 2025) — truthful direction via PCA.
             CCS (Burns 2022) — contrast-consistent search.
             Linear probing theory (Alain & Bengio 2016).

Usage:
    python C2_truth_direction.py --n_samples 100
    python C2_truth_direction.py --n_samples 100 --layer 16
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

from shared import load_model_and_data, evaluate_auroc


# ═══════════════════════════════════════════════════════════════════════════════
# Core: extract hidden state + compute truth direction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hidden_at_layer(model, tokenizer, prompt: str, device: str,
                            layer: int) -> np.ndarray:
    """Extract residual stream at specified layer, last token position."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    residual = {}

    def _hook(act, hook=None, **kwargs):
        residual["h"] = act[:, -1, :].detach()
    return _hook

    fwd_hooks = [(f"blocks.{layer}.hook_resid_post", _hook)]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    return residual["h"].float().cpu().numpy().flatten()


def compute_truth_direction(
    h_correct: np.ndarray,  # [N_correct, d]
    h_incorrect: np.ndarray,  # [N_incorrect, d]
) -> np.ndarray:
    """Truth direction: v = mean(h_correct) - mean(h_incorrect), normalized."""
    v = h_correct.mean(axis=0) - h_incorrect.mean(axis=0)  # [d]
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    return v


def project_onto_direction(
    h: np.ndarray,  # [N, d]
    v: np.ndarray,  # [d]
) -> np.ndarray:
    """Project hidden states onto truth direction. Returns [N] scalar scores."""
    return h @ v  # [N]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="C2: Truth Direction Projection")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer to extract HS from (-1 = scan ALL layers)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"C2: Truth Direction Projection (layer={args.layer})")
    print(f"  Model: {args.model}  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    print("Loading model & data...")
    t0 = time.time()
    model, tokenizer, samples = load_model_and_data(
        n_samples=args.n_samples, seed=args.seed,
        device=device, model_id=args.model,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    from src.data_loader import format_prompt, check_correct
    n_layers = model.cfg.n_layers

    # Resolve layers to scan
    if args.layer >= 0:
        layers_to_scan = [args.layer]
    else:
        # --layer -1: scan ALL layers (efficient: one FP captures all)
        layers_to_scan = list(range(n_layers))
        print(f"  Full-layer scan: {len(layers_to_scan)} layers (L0-L{n_layers-1})")

    # ── Extract hidden states + labels ──
    # Key efficiency fix: hook ALL target layers on the FIRST generation forward pass.
    # Before: generation (up to 20 FPs) + N_layers separate FPs → up to 20+N FPs/sample
    # After:  generation with HS hooks on first FP → up to 20 FPs/sample (no extra cost)
    print(f"\nExtracting hidden states + generating labels...")
    print(f"  Layers: {len(layers_to_scan)}, Samples: {args.n_samples}")
    t0 = time.time()

    all_h = {li: [] for li in layers_to_scan}
    labels = []
    correct_count = 0

    # Pre-build hook list for all target layers (shared across samples, rebuilt per sample
    # since each hook captures into a per-sample residual dict)
    for s in tqdm(samples, desc="C2 extract HS"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # ── First forward pass: extract HS from ALL target layers + get first token ──
        residual = {}
        fwd_hooks = []
        for li in layers_to_scan:
            # Capture li by default-arg binding (not closure — avoids late-binding bug)
            def _hook(act, hook=None, _layer=li):
                residual[_layer] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_resid_post", _hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Store hidden states for all layers
        for li in layers_to_scan:
            all_h[li].append(residual[li].float().cpu().numpy().flatten())

        # Get first generated token
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]

        # ── Continue autoregressive generation (no hooks needed) ──
        for _ in range(19):  # max 19 more tokens (20 total)
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")
        if is_correct:
            correct_count += 1
        labels.append(1 if is_correct else 0)

    labels = np.array(labels)
    print(f"  Correct: {correct_count}/{len(samples)} ({correct_count/len(samples):.1%})")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Per-layer: compute truth direction + project ──
    print(f"\nAUROC per layer (truth direction projection):")
    print(f"  {'Layer':>6s}  {'AUROC':>8s}  {'v_norm':>8s}")
    print(f"  {'─'*28}")

    best_layer, best_auroc = -1, 0.0
    layer_results = []

    for li in sorted(all_h.keys()):
        H = np.stack(all_h[li], axis=0)  # [N, d]
        mask_correct = labels == 1
        mask_incorrect = labels == 0

        if mask_correct.sum() < 2 or mask_incorrect.sum() < 2:
            continue

        v = compute_truth_direction(H[mask_correct], H[mask_incorrect])
        scores = project_onto_direction(H, v)

        valid = np.isfinite(scores)
        if valid.sum() < 10 or labels[valid].std() == 0:
            continue

        auroc = float(roc_auc_score(labels[valid], scores[valid]))
        auroc = max(auroc, 1 - auroc)
        v_norm = np.linalg.norm(v)

        print(f"  {li:>6d}  {auroc:>8.4f}  {v_norm:>8.4f}")

        layer_results.append({"layer": li, "auroc": auroc, "v_norm": float(v_norm)})

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = li

    print(f"\n  Best: Layer {best_layer}, AUROC={best_auroc:.4f}")
    print(f"  Baseline max_p = 0.652 (1.7B)")

    # Save
    save_path = output_dir / "C2_truth_direction.json"
    with open(save_path, "w") as f:
        json.dump({
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "per_layer": sorted(layer_results, key=lambda x: x["auroc"], reverse=True),
            "n_samples": len(labels),
            "n_correct": int(labels.sum()),
        }, f, indent=2)
    print(f"  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"C2 complete — truth direction projection")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
