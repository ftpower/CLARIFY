"""8B Subspace Alignment Intervention — verify if 8B fixes the 89.7° orthogonal issue.

1.7B result: train/val PCA subspaces at L11 have max principal angle 89.7° —
essentially orthogonal, meaning direction-based intervention is unreliable.
Hypothesis: 8B's larger hidden space (4096 vs 2048) and stronger layer structure
produce more aligned subspaces, enabling effective direction intervention.

Usage (on AutoDL RTX 5090):
    python main_8b_subspace_intervention.py --n_dir 100 --n_eval 50 --device cuda
    python main_8b_subspace_intervention.py --n_dir 300 --n_eval 200 --device cuda

Key layers (1.7B → 8B proportional mapping):
    L11 → L14, L15 → L19, L17 → L22
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt

# 1.7B → 8B layer mapping (proportional: round(L_1.7B * 35/27))
LAYER_MAP = {11: 14, 15: 19, 17: 22}

# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _make_hook(storage, key):
    def hook(act, hook=None):
        storage[key] = act.detach()
        return act
    return hook


def load_train_hellaswag(n: int, seed: int) -> list[dict]:
    """Load from HellaSwag train split (for direction computation)."""
    ds = load_dataset("Rowan/hellaswag", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed)
    label_letters = ["A", "B", "C", "D"]
    samples = []
    for item in ds.select(range(min(n, len(ds)))):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        choices_text = "\n".join(f"{label_letters[i]}. {endings[i]}" for i in range(4))
        samples.append({
            "question": ctx,
            "answers": [endings[label], label_letters[label]],
            "context": choices_text,
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Hidden state collection
# ═══════════════════════════════════════════════════════════════════════════════


def collect_states(
    model,
    train_samples: list[dict],
    val_samples: list[dict],
    layers: list[int],
    letter_ids: dict[str, int],
) -> dict:
    """Collect last-token hidden states for train/val × correct/incorrect."""
    letters = ["A", "B", "C", "D"]

    accum = {L: {"train_correct": [], "train_incorrect": [],
                 "val_correct": [], "val_incorrect": []} for L in layers}

    def _collect(samples, split):
        storage = {}
        hooks = [(f"blocks.{L}.hook_resid_post", _make_hook(storage, str(L)))
                 for L in layers]

        for sample in tqdm(samples, desc=f"Collecting {split}", leave=False):
            prompt = format_prompt(sample["question"], sample["context"], dataset="hellaswag")
            correct = sample["answers"][1].upper()
            tokens = model.to_tokens(prompt, prepend_bos=True)
            last_pos = tokens.shape[1] - 1

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            logits_last = logits[0, last_pos, :]
            lid = torch.tensor([letter_ids[l] for l in letters], device=logits_last.device)
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            is_correct = letters[probs.argmax().item()] == correct

            for L in layers:
                h = storage[str(L)][0, last_pos, :].cpu()
                target = f"{split}_{'correct' if is_correct else 'incorrect'}"
                accum[L][target].append(h)

    _collect(train_samples, "train")
    _collect(val_samples, "val")

    result = {}
    for L in layers:
        result[L] = {}
        for key in accum[L]:
            states = accum[L][key]
            result[L][key] = torch.stack(states) if states else torch.zeros(0, model.cfg.d_model)
            print(f"  L{L} {key}: {len(states)} samples")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Subspace analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_subspace(
    train_states: torch.Tensor,
    val_states: torch.Tensor,
    k: int = 64,
) -> dict:
    """PCA bases, principal angles between train and val subspaces."""
    d_model = train_states.shape[1]
    k_actual = min(k, train_states.shape[0], val_states.shape[0], d_model)

    X_tr = train_states.numpy().astype(np.float64)
    X_val = val_states.numpy().astype(np.float64)

    pca_tr = PCA(n_components=k_actual).fit(X_tr)
    K_tr = pca_tr.components_.T  # [d_model, k]
    pca_val = PCA(n_components=k_actual).fit(X_val)
    K_val = pca_val.components_.T

    angles = subspace_angles(K_tr, K_val)
    max_deg = float(np.max(angles) * 180.0 / np.pi)
    mean_deg = float(np.mean(angles) * 180.0 / np.pi)

    # Alignment projector + aligned direction
    K_tr_t = torch.from_numpy(K_tr).float()
    P_tr = K_tr_t @ K_tr_t.T  # [d_model, d_model]

    return {
        "k_actual": k_actual,
        "principal_angles_deg": [float(a * 180.0 / np.pi) for a in angles],
        "max_angle_deg": max_deg,
        "mean_angle_deg": mean_deg,
        "alignment_matrix": P_tr,
        "train_pca_explained": float(pca_tr.explained_variance_ratio_.sum()),
        "val_pca_explained": float(pca_val.explained_variance_ratio_.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Intervention
# ═══════════════════════════════════════════════════════════════════════════════


def compute_mean_diff(accum: dict) -> torch.Tensor:
    """Mean-diff direction: h_incorrect - h_correct (normalized)."""
    c = accum["train_correct"].mean(dim=0)
    i = accum["train_incorrect"].mean(dim=0)
    diff = i - c
    return diff / (diff.norm() + 1e-8)


def make_intervention_hook(direction: torch.Tensor, lam: float, mode: str):
    sign = -1.0 if mode == "subtract" else 1.0
    def hook(act, hook=None):
        d = direction.to(act.dtype).to(act.device)
        proj = act @ d
        return act + sign * lam * proj.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
    return hook


def evaluate_intervention(
    model, val_samples, direction, layer, lam, mode, letter_ids,
) -> dict:
    """Evaluate one (layer, lam, mode, direction) config."""
    letters = ["A", "B", "C", "D"]
    hook_fn = make_intervention_hook(direction, lam, mode)
    hook_pt = f"blocks.{layer}.hook_resid_post"

    nc, nc_base = 0, 0
    per_sample = []

    for sample in tqdm(val_samples, desc=f"L{layer} {mode} λ={lam}", leave=False):
        prompt = format_prompt(sample["question"], sample["context"], dataset="hellaswag")
        correct = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)

        # Baseline
        with torch.no_grad():
            logits_base = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits_base[0, -1, :]
        lid = torch.tensor([letter_ids[l] for l in letters], device=logits_last.device)
        probs_base = F.softmax(logits_last[lid].float(), dim=-1)
        base_correct = letters[probs_base.argmax().item()] == correct

        # Intervention
        with torch.no_grad():
            logits_int = model.run_with_hooks(tokens, fwd_hooks=[(hook_pt, hook_fn)])
        logits_last = logits_int[0, -1, :]
        probs_int = F.softmax(logits_last[lid].float(), dim=-1)
        int_correct = letters[probs_int.argmax().item()] == correct

        nc += int(int_correct)
        nc_base += int(base_correct)
        per_sample.append({
            "base_correct": base_correct,
            "int_correct": int_correct,
            "p_correct": probs_base[letters.index(correct)].item(),
        })

    n = len(val_samples)
    return {
        "layer": layer,
        "lam": lam,
        "mode": mode,
        "baseline_acc": nc_base / n,
        "intervention_acc": nc / n,
        "delta": (nc - nc_base) / n,
        "n_total": n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="8B Subspace Alignment Intervention"
    )
    parser.add_argument("--n_dir", type=int, default=100,
                        help="Train samples for direction computation")
    parser.add_argument("--n_eval", type=int, default=50,
                        help="Validation samples for evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_pca", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[14, 19, 22],
                        help="8B layers (mapped from 1.7B L11,L15,L17)")
    parser.add_argument("--lams", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 1.0],
                        help="Intervention strengths")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Use cached states")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "subspace_8b_states.pt"

    # ── Load model ──
    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()
    print(f"  n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    letter_ids = {}
    for l in ["A", "B", "C", "D"]:
        toks = model.tokenizer.encode(f" {l}", add_special_tokens=False)
        letter_ids[l] = toks[-1] if len(toks) >= 1 else toks[0]

    # ── Load data ──
    print(f"\nLoading HellaSwag train (n={args.n_dir}) + val (n={args.n_eval})...")
    train_samples = load_train_hellaswag(args.n_dir, args.seed)
    val_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

    # ── Collect states ──
    if args.skip_extract and cache_path.exists():
        print(f"\nLoading cached states from {cache_path}")
        accum = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        print(f"\nCollecting hidden states at 8B layers {args.layers}")
        accum = collect_states(model, train_samples, val_samples, args.layers, letter_ids)
        torch.save(accum, cache_path)
        print(f"Saved to {cache_path}")

    # ── Subspace analysis ──
    print(f"\n{'=' * 60}")
    print("Subspace Alignment Analysis")
    print(f"{'=' * 60}\n")

    subspace_results = {}
    for L in args.layers:
        tr_corr = accum[L]["train_correct"]
        tr_incorr = accum[L]["train_incorrect"]
        val_all = torch.cat([accum[L]["val_correct"], accum[L]["val_incorrect"]])
        tr_all = torch.cat([tr_corr, tr_incorr])

        print(f"--- Layer {L} ---")
        a = analyze_subspace(tr_all, val_all, k=args.k_pca)
        subspace_results[str(L)] = a

        status = "✅" if a["max_angle_deg"] < 45 else ("⚠️" if a["max_angle_deg"] < 70 else "🔴")
        print(f"  Max principal angle: {a['max_angle_deg']:.2f}° {status}")
        print(f"  Mean principal angle: {a['mean_angle_deg']:.2f}°")
        print(f"  Train PCA explained: {a['train_pca_explained']:.3f}")
        print(f"  Val PCA explained: {a['val_pca_explained']:.3f}")

        # Check specific angles (top-5 and bottom-5)
        angles = a["principal_angles_deg"]
        print(f"  Top-5 angles (most aligned): {angles[:5]}")
        print(f"  Bottom-5 angles (most misaligned): {angles[-5:]}")

    # ── Intervention evaluation ──
    print(f"\n{'=' * 60}")
    print("Direction Intervention Evaluation")
    print(f"{'=' * 60}\n")

    all_results = []
    modes = ["subtract", "add"]

    # Compute baseline accuracy
    letters = ["A", "B", "C", "D"]
    nc_base = 0
    for sample in tqdm(val_samples, desc="Baseline eval", leave=False):
        prompt = format_prompt(sample["question"], sample["context"], dataset="hellaswag")
        correct = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        lid = torch.tensor([letter_ids[l] for l in letters], device=logits_last.device)
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        nc_base += int(letters[probs.argmax().item()] == correct)
    baseline_acc = nc_base / len(val_samples)
    print(f"Baseline accuracy: {baseline_acc:.4f} ({nc_base}/{len(val_samples)})")

    # Filtered baseline (P(correct) > 0.3)
    p_correct_vals = []
    filt_correct = 0
    filt_total = 0
    for sample in tqdm(val_samples, desc="Filtered baseline", leave=False):
        prompt = format_prompt(sample["question"], sample["context"], dataset="hellaswag")
        correct = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        lid = torch.tensor([letter_ids[l] for l in letters], device=logits_last.device)
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        p_c = probs[letters.index(correct)].item()
        p_correct_vals.append(p_c)
        if p_c > 0.3:
            filt_total += 1
            filt_correct += int(letters[probs.argmax().item()] == correct)

    if filt_total > 0:
        filt_base_acc = filt_correct / filt_total
        print(f"Filtered (P>0.3) baseline: {filt_base_acc:.4f} ({filt_correct}/{filt_total})")
    else:
        filt_base_acc = float("nan")

    # Evaluate interventions
    for L in args.layers:
        # Mean-diff direction
        d_raw = compute_mean_diff(accum[L])

        # PCA-aligned direction (align with train subspace projector)
        P = subspace_results[str(L)]["alignment_matrix"].to(device=d_raw.device, dtype=torch.float32)
        d_aligned = (P @ d_raw.float())
        d_aligned = d_aligned / (d_aligned.norm() + 1e-8)

        # Random direction (control)
        d_rand = torch.randn_like(d_raw)
        d_rand = d_rand / (d_rand.norm() + 1e-8)

        directions = {
            "mean_diff": d_raw,
            "pca_aligned": d_aligned,
            "random": d_rand,
        }

        for dname, dvec in directions.items():
            for lam in args.lams:
                for mode in modes:
                    r = evaluate_intervention(
                        model, val_samples, dvec, L, lam, mode, letter_ids,
                    )
                    r["direction"] = dname
                    r["baseline_acc"] = baseline_acc
                    if dname != "random":  # only print non-random results
                        sign = "✅" if r["delta"] > 0.01 else ("·" if abs(r["delta"]) < 0.01 else "🔴")
                        print(f"  L{L} {dname:>12s} {mode:>8s} λ={lam:.1f}: "
                              f"acc={r['intervention_acc']:.4f} Δ={r['delta']:+.4f} {sign}")
                    all_results.append(r)

    # ── Summary ──
    best = max(
        [r for r in all_results if r["direction"] != "random"],
        key=lambda r: r["delta"],
        default={"delta": 0.0},
    )

    rand_deltas = [r["delta"] for r in all_results if r["direction"] == "random"]
    rand_mean = np.mean(rand_deltas) if rand_deltas else 0.0
    rand_std = np.std(rand_deltas) if rand_deltas else 0.0

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Baseline accuracy:           {baseline_acc:.4f}")
    print(f"  Filtered baseline (P>0.3):   {filt_base_acc:.4f}")
    print(f"  Best intervention Δ:         {best.get('delta', 0):+.4f} "
          f"(L{best.get('layer', '?')} {best.get('direction', '?')} "
          f"λ={best.get('lam', 0):.1f} {best.get('mode', '?')})")
    print(f"  Random control Δ:            {rand_mean:+.4f} ± {rand_std:.4f}")
    print(f"  Random Δ range:              [{min(rand_deltas):+.4f}, {max(rand_deltas):+.4f}]")

    # Directionality check
    best_sub = max(
        [r for r in all_results if r["direction"] != "random" and r["mode"] == "subtract"],
        key=lambda r: r["delta"],
        default={"delta": 0.0},
    )
    best_add = max(
        [r for r in all_results if r["direction"] != "random" and r["mode"] == "add"],
        key=lambda r: r["delta"],
        default={"delta": 0.0},
    )
    directional = best_sub["delta"] > rand_mean + rand_std and best_add["delta"] < rand_mean
    print(f"  Subtract helps (+): {best_sub['delta']:+.4f}, "
          f"Add hurts (-): {best_add['delta']:+.4f}")
    print(f"  Directional signal:  {'✅ YES' if directional else '🔴 NO'}")

    # Verdict
    max_angles = [subspace_results[str(L)]["max_angle_deg"] for L in args.layers]
    worst_angle = max(max_angles)

    print(f"\n  Worst principal angle: {worst_angle:.2f}°")
    if worst_angle < 45:
        print("  ✅ Subspaces well-aligned — direction intervention likely effective.")
    elif worst_angle < 70:
        print("  ⚠️  Subspaces partially aligned — direction intervention marginal.")
    else:
        print("  🔴 Subspaces nearly orthogonal — direction intervention unreliable.")
        print("     (Same as 1.7B result of 89.7°)")

    # ── Save ──
    output = {
        "config": {
            "n_dir": args.n_dir,
            "n_eval": args.n_eval,
            "model": args.model,
            "layers": args.layers,
            "k_pca": args.k_pca,
            "lams": args.lams,
            "seed": args.seed,
            "d_model": model.cfg.d_model,
            "n_layers": model.cfg.n_layers,
        },
        "alignment": {k: {kk: vv for kk, vv in v.items()
                          if kk != "alignment_matrix"}
                      for k, v in subspace_results.items()},
        "baseline": {
            "full_acc": baseline_acc,
            "filtered_acc": filt_base_acc,
            "filtered_n": filt_total,
        },
        "best_intervention": best,
        "random_control": {
            "mean_delta": float(rand_mean),
            "std_delta": float(rand_std),
            "min_delta": float(min(rand_deltas)),
            "max_delta": float(max(rand_deltas)),
        },
        "directionality": {
            "subtract_helps": best_sub["delta"] > rand_mean + rand_std,
            "add_hurts": best_add["delta"] < rand_mean,
            "directional": directional,
        },
        "all_results": all_results,
    }

    results_path = output_dir / "subspace_intervention_8b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {results_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("8B subspace intervention complete. ✅")


if __name__ == "__main__":
    main()
