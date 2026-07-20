"""P1 Part B1: Subspace Alignment Intervention — ComnHallu-style PCA alignment.

Addresses the I1→S1 non-transfer problem: I1 L11 mean-diff direction gives +5.56pp
on train samples but -1.67pp on independent validation. Hypothesis: the train and
val hidden-state subspaces are misaligned, making the direction non-robust.

This experiment:
  1. Extracts L11 hidden states on both I1 train and S1 val sets
  2. Computes PCA(k=64) bases for each → K_train, K_test
  3. Computes principal angles between the two subspaces
  4. Regularizes the mean-diff direction: dir_aligned = K_train @ K_train^T @ dir_mean_diff
  5. Evaluates original vs aligned vs random directions on the val set

Key decision point: if max principal angle > 45°, the subspaces are fundamentally
different and direction-based intervention should be abandoned.

Usage:
    python main_subspace_intervention.py --n_dir 300 --n_eval 200 --k_pca 64
    python main_subspace_intervention.py --n_dir 300 --n_eval 200 --layers 11 15
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

sys.path.insert(0, str(Path(__file__).parent.parent / "phase2_entropy"))
from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Hidden state collection
# ═══════════════════════════════════════════════════════════════════════════════


def make_save_hook(storage: dict, key: str):
    def hook(activation, hook=None):
        storage[key] = activation.detach()
        return activation
    return hook


def collect_states_for_sets(
    model,
    train_samples: list[dict],
    val_samples: list[dict],
    layers: list[int],
    letter_ids: dict[str, int],
) -> dict:
    """Collect L-layer hidden states for train and val sets, labeled by correctness.

    Returns:
        {layer: {
            "train_correct": [N_c, d_model] tensor,
            "train_incorrect": [N_i, d_model] tensor,
            "val_correct": [N_c', d_model] tensor,
            "val_incorrect": [N_i', d_model] tensor,
        }}
    """
    n_layers = model.cfg.n_layers

    # Initialize accumulators
    accum = {
        L: {
            "train_correct": [],
            "train_incorrect": [],
            "val_correct": [],
            "val_incorrect": [],
        }
        for L in layers
    }

    def _process_split(samples, split_name):
        storage = {}
        hooks = []
        for L in layers:
            key = f"blocks.{L}.hook_resid_post"
            hooks.append((key, make_save_hook(storage, key)))

        letters = ["A", "B", "C", "D"]

        for sample in tqdm(samples, desc=f"Collecting {split_name}"):
            prompt = format_prompt(
                sample["question"], sample["context"], dataset="hellaswag"
            )
            correct_letter = sample["answers"][1].upper()
            tokens = model.to_tokens(prompt, prepend_bos=True)
            last_pos = tokens.shape[1] - 1

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            # Determine correctness
            logits_last = logits[0, last_pos, :]
            lid = torch.tensor(
                [letter_ids[l] for l in letters], device=logits_last.device
            )
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            pred_idx = probs.argmax().item()
            is_correct = letters[pred_idx] == correct_letter

            for L in layers:
                key = f"blocks.{L}.hook_resid_post"
                h = storage[key][0, last_pos, :].cpu()
                target = f"{split_name}_{'correct' if is_correct else 'incorrect'}"
                accum[L][target].append(h)

        return accum

    _process_split(train_samples, "train")
    _process_split(val_samples, "val")

    # Convert to tensors
    result = {}
    for L in layers:
        result[L] = {}
        for key in ["train_correct", "train_incorrect", "val_correct", "val_incorrect"]:
            states = accum[L][key]
            result[L][key] = (
                torch.stack(states) if states else torch.zeros(0, model.cfg.d_model)
            )
            print(f"  L{L} {key}: {len(states)} samples")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Subspace alignment analysis
# ═══════════════════════════════════════════════════════════════════════════════


def compute_subspace_analysis(
    train_states: torch.Tensor,
    val_states: torch.Tensor,
    k: int = 64,
) -> dict:
    """Compute PCA bases, principal angles, and alignment matrix.

    Args:
        train_states: [N_train, d_model] training hidden states.
        val_states: [N_val, d_model] validation hidden states.
        k: Number of PCA components to keep.

    Returns:
        dict with:
            K_train: [d_model, k] — orthonormal basis for train subspace
            K_val: [d_model, k] — orthonormal basis for val subspace
            principal_angles: [k] — angles (radians) between subspaces
            max_angle_deg: float
            alignment_matrix: [d_model, d_model] — K_train @ K_train^T projector
    """
    d_model = train_states.shape[1]
    k_actual = min(k, train_states.shape[0], val_states.shape[0], d_model)

    # PCA on train set
    X_train = train_states.numpy().astype(np.float64)
    X_val = val_states.numpy().astype(np.float64)

    pca_train = PCA(n_components=k_actual)
    pca_train.fit(X_train)
    K_train = pca_train.components_.T  # [d_model, k_actual]

    pca_val = PCA(n_components=k_actual)
    pca_val.fit(X_val)
    K_val = pca_val.components_.T  # [d_model, k_actual]

    # Principal angles between subspaces
    angles = subspace_angles(K_train, K_val)  # [k_actual], radians
    max_angle_deg = float(np.max(angles) * 180.0 / np.pi)
    mean_angle_deg = float(np.mean(angles) * 180.0 / np.pi)

    # Alignment projector: P_train = K_train @ K_train^T
    K_train_t = torch.from_numpy(K_train).float()
    alignment_matrix = K_train_t @ K_train_t.T  # [d_model, d_model]

    return {
        "k_actual": k_actual,
        "K_train": K_train,
        "K_val": K_val,
        "principal_angles_rad": [float(a) for a in angles],
        "principal_angles_deg": [float(a * 180.0 / np.pi) for a in angles],
        "max_angle_deg": max_angle_deg,
        "mean_angle_deg": mean_angle_deg,
        "alignment_matrix": alignment_matrix,
        "train_pca_explained": float(
            pca_train.explained_variance_ratio_.sum()
        ),
        "val_pca_explained": float(
            pca_val.explained_variance_ratio_.sum()
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Intervention evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_mean_diff_direction(accum: dict) -> torch.Tensor:
    """Mean-diff direction from train correct/incorrect states."""
    h_corr = accum["train_correct"].mean(dim=0)
    h_incorr = accum["train_incorrect"].mean(dim=0)
    diff = h_incorr - h_corr
    return diff / (diff.norm() + 1e-8)


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    """Return hook that projects hidden states onto/against a direction."""
    sign = -1.0 if mode == "subtract" else 1.0

    def hook(activation, hook=None):
        d = direction.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam * projection

    return hook


def evaluate_direction(
    model,
    val_samples: list[dict],
    direction: torch.Tensor,
    layer: int,
    lam: float,
    mode: str,
    letter_ids: dict[str, int],
) -> dict:
    """Evaluate a single intervention direction on val set."""
    letters = ["A", "B", "C", "D"]
    hook_fn = make_projection_hook(direction, lam, mode)
    hook_point = f"blocks.{layer}.hook_resid_post"

    n_correct = 0
    n_total = len(val_samples)
    n_correct_base = 0
    per_sample = []

    for sample in tqdm(val_samples, desc=f"Eval L{layer} {mode} λ={lam}", leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)

        # Baseline
        with torch.no_grad():
            logits_base = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last_base = logits_base[0, -1, :]
        lid = torch.tensor(
            [letter_ids[l] for l in letters], device=logits_last_base.device
        )
        probs_base = F.softmax(logits_last_base[lid].float(), dim=-1)
        pred_base = letters[probs_base.argmax().item()]
        is_base_correct = pred_base == correct_letter
        n_correct_base += int(is_base_correct)

        # Intervention
        with torch.no_grad():
            logits_int = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_point, hook_fn)]
            )
        logits_last_int = logits_int[0, -1, :]
        probs_int = F.softmax(logits_last_int[lid].float(), dim=-1)
        pred_int = letters[probs_int.argmax().item()]
        is_int_correct = pred_int == correct_letter
        n_correct += int(is_int_correct)
        p_correct = probs_int[letters.index(correct_letter)].item()

        per_sample.append(
            {
                "pred": pred_int,
                "correct": correct_letter,
                "is_correct": is_int_correct,
                "is_correct_base": is_base_correct,
                "p_correct": p_correct,
            }
        )

    acc = n_correct / n_total
    base_acc = n_correct_base / n_total
    delta = acc - base_acc

    # Knowledge-filtered metrics
    filtered = [s for s in per_sample if s["p_correct"] > 0.3]
    n_filt = len(filtered)
    filtered_acc = (
        sum(s["is_correct"] for s in filtered) / n_filt if n_filt >= 20 else None
    )
    base_filtered = [s for s in per_sample if s["p_correct"] > 0.3]
    base_filt_acc_base = (
        sum(s["is_correct_base"] for s in base_filtered) / n_filt
        if n_filt >= 20
        else None
    )
    delta_f = (
        filtered_acc - base_filt_acc_base
        if filtered_acc is not None and base_filt_acc_base is not None
        else None
    )

    return {
        "layer": layer,
        "lam": lam,
        "mode": mode,
        "accuracy": float(acc),
        "baseline_acc": float(base_acc),
        "delta": float(delta),
        "accuracy_filtered": float(filtered_acc) if filtered_acc is not None else None,
        "delta_filtered": float(delta_f) if delta_f is not None else None,
        "n_total": n_total,
        "n_filtered": n_filt,
        "n_correct": n_correct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="P1 Part B1: Subspace Alignment Intervention"
    )
    parser.add_argument("--n_dir", type=int, default=300)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[11])
    parser.add_argument("--k_pca", type=int, default=64)
    parser.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_collect", action="store_true",
                        help="Load cached states")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "subspace_states.pt"

    if args.skip_collect and cache_path.exists():
        print(f"Loading cached states from {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        all_states = cached["states"]
    else:
        # ── Load model ──
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(
                f" {letter}", add_special_tokens=False
            )
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
        print(f"Letter token IDs: {letter_ids}")

        # ── Load train (HellaSwag train split, like I1) ──
        print(f"\nLoading HellaSwag train (n={args.n_dir})...")
        ds_train = load_dataset(
            "Rowan/hellaswag", split="train", trust_remote_code=False
        )
        ds_train = ds_train.shuffle(seed=args.seed)
        label_letters = ["A", "B", "C", "D"]
        train_samples = []
        for item in ds_train.select(range(min(args.n_dir, len(ds_train)))):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            label_letter = label_letters[label]
            choices_text = "\n".join(
                f"{label_letters[i]}. {endings[i]}" for i in range(4)
            )
            train_samples.append(
                {
                    "question": ctx,
                    "answers": [endings[label], label_letter],
                    "context": choices_text,
                }
            )

        # ── Load val (HellaSwag validation split, like S1) ──
        print(f"Loading HellaSwag validation (n={args.n_eval})...")
        val_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

        # ── Collect hidden states ──
        print(f"\n{'=' * 60}")
        print("Phase 1: Collecting hidden states at candidate layers")
        print(f"{'=' * 60}")

        all_states = collect_states_for_sets(
            model, train_samples, val_samples, args.layers, letter_ids
        )

        # Save cache
        save_dict = {
            L: {k: v for k, v in layer_dict.items()}
            for L, layer_dict in all_states.items()
        }
        torch.save({"states": save_dict}, cache_path)
        print(f"Saved states to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phase 2: Subspace alignment analysis ──
    print(f"\n{'=' * 60}")
    print("Phase 2: Subspace Alignment Analysis")
    print(f"{'=' * 60}")

    alignment_results = {}
    aligned_directions = {}
    raw_directions = {}

    for L in args.layers:
        print(f"\n--- L{L} ---")
        states = all_states[L]

        if len(states["train_correct"]) == 0 or len(states["train_incorrect"]) == 0:
            print(f"  WARNING: Not enough train samples for L{L}, skipping")
            continue

        # Combine all train states for PCA
        train_all = torch.cat(
            [states["train_correct"], states["train_incorrect"]], dim=0
        )
        val_all = torch.cat(
            [states["val_correct"], states["val_incorrect"]], dim=0
        )

        print(f"  Train: {train_all.shape[0]} samples, Val: {val_all.shape[0]} samples")

        # Subspace analysis
        subspace = compute_subspace_analysis(train_all, val_all, k=args.k_pca)

        print(f"  PCA(k={subspace['k_actual']}) explained variance: "
              f"train={subspace['train_pca_explained']:.3f}, "
              f"val={subspace['val_pca_explained']:.3f}")
        print(f"  Principal angles: max={subspace['max_angle_deg']:.1f}°, "
              f"mean={subspace['mean_angle_deg']:.1f}°")
        print(f"  Top-5 angles: {[f'{a:.1f}°' for a in subspace['principal_angles_deg'][:5]]}")

        if subspace["max_angle_deg"] > 45.0:
            print(f"  ⚠ WARNING: max principal angle > 45°! "
                  f"Train/val subspaces fundamentally different.")
            print(f"  Direction-based intervention may be unreliable at this layer.")

        # Compute mean-diff direction
        raw_dir = compute_mean_diff_direction(states)
        raw_directions[L] = raw_dir

        # Align direction via PCA subspace projection
        K_train_t = torch.from_numpy(subspace["K_train"]).float()  # [d, k]
        # dir_aligned = K_train @ K_train^T @ dir_raw
        dir_raw_np = raw_dir.numpy()
        aligned_np = (
            subspace["K_train"] @ (subspace["K_train"].T @ dir_raw_np)
        )
        aligned_dir = torch.from_numpy(aligned_np).float()
        aligned_dir = aligned_dir / (aligned_dir.norm() + 1e-8)
        aligned_directions[L] = aligned_dir

        cos_sim = float((raw_dir * aligned_dir).sum())
        print(f"  cos(raw, aligned) = {cos_sim:+.4f} "
              f"({'nearly identical' if abs(cos_sim) > 0.95 else 'different'})")

        alignment_results[L] = subspace

    # ── Phase 3: Evaluate interventions ──
    print(f"\n{'=' * 60}")
    print("Phase 3: Intervention Evaluation")
    print(f"{'=' * 60}")

    print("\nLoading model for evaluation...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

    print(f"Loading HellaSwag validation (n={args.n_eval})...")
    val_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

    all_eval_results = []

    for L in args.layers:
        if L not in raw_directions:
            continue

        for lam in args.lam:
            for mode in ["subtract", "add"]:
                # --- Raw mean-diff ---
                print(f"\nEvaluating L{L} raw mean-diff, λ={lam}, {mode}")
                result_raw = evaluate_direction(
                    model, val_samples, raw_directions[L],
                    L, lam, mode, letter_ids,
                )
                result_raw["direction_type"] = "raw_mean_diff"
                result_raw["layer"] = L
                all_eval_results.append(result_raw)
                print(f"  Raw: acc={result_raw['accuracy']:.4f}, "
                      f"Δ={result_raw['delta']:+.4f}, "
                      f"Δf={result_raw['delta_filtered']:+.4f}"
                      if result_raw['delta_filtered'] is not None
                      else f"  Raw: acc={result_raw['accuracy']:.4f}, Δ={result_raw['delta']:+.4f}")

                # --- PCA-aligned ---
                print(f"Evaluating L{L} PCA-aligned, λ={lam}, {mode}")
                result_aligned = evaluate_direction(
                    model, val_samples, aligned_directions[L],
                    L, lam, mode, letter_ids,
                )
                result_aligned["direction_type"] = "pca_aligned"
                result_aligned["layer"] = L
                all_eval_results.append(result_aligned)
                print(f"  Aligned: acc={result_aligned['accuracy']:.4f}, "
                      f"Δ={result_aligned['delta']:+.4f}, "
                      f"Δf={result_aligned['delta_filtered']:+.4f}"
                      if result_aligned['delta_filtered'] is not None
                      else f"  Aligned: acc={result_aligned['accuracy']:.4f}, Δ={result_aligned['delta']:+.4f}")

                # --- Random control ---
                rand_dir = torch.randn(model.cfg.d_model)
                rand_dir = rand_dir / (rand_dir.norm() + 1e-8)
                print(f"Evaluating L{L} random, λ={lam}, {mode}")
                result_rand = evaluate_direction(
                    model, val_samples, rand_dir,
                    L, lam, mode, letter_ids,
                )
                result_rand["direction_type"] = "random"
                result_rand["layer"] = L
                all_eval_results.append(result_rand)

    # ── Report ──
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")

    print(f"\n{'Direction':<16} {'Layer':<6} {'λ':<6} {'Mode':<10} {'Acc':>8} {'Δ':>8} {'Δf':>8}")
    print("-" * 66)

    for r in all_eval_results:
        df_str = f"{r['delta_filtered']:>+8.4f}" if r['delta_filtered'] is not None else "     N/A"
        print(
            f"{r['direction_type']:<16} L{r['layer']:<5} {r['lam']:<6} {r['mode']:<10} "
            f"{r['accuracy']:>8.4f} {r['delta']:>+8.4f} {df_str}"
        )

    # Best per direction type
    print(f"\n--- Best per direction type (by Δf) ---")
    for dtype in ["raw_mean_diff", "pca_aligned", "random"]:
        dtype_results = [
            r for r in all_eval_results
            if r["direction_type"] == dtype and r["delta_filtered"] is not None
        ]
        if dtype_results:
            best = max(dtype_results, key=lambda r: r["delta_filtered"])
            print(
                f"  {dtype:<16}: L{best['layer']} λ={best['lam']} {best['mode']:<10} "
                f"Δf={best['delta_filtered']:+.4f}"
            )

    # Random control stats
    rand_deltas = [
        r["delta_filtered"]
        for r in all_eval_results
        if r["direction_type"] == "random" and r["delta_filtered"] is not None
    ]
    if rand_deltas:
        print(f"\nRandom control: mean Δf = {np.mean(rand_deltas):+.4f}, "
              f"std = {np.std(rand_deltas):+.4f}")

    # ── Save results ──
    # Convert alignment results for JSON (remove large matrices)
    alignment_json = {}
    for L, subspace in alignment_results.items():
        alignment_json[str(L)] = {
            "k_actual": subspace["k_actual"],
            "principal_angles_deg": subspace["principal_angles_deg"],
            "max_angle_deg": subspace["max_angle_deg"],
            "mean_angle_deg": subspace["mean_angle_deg"],
            "train_pca_explained": subspace["train_pca_explained"],
            "val_pca_explained": subspace["val_pca_explained"],
        }

    output = {
        "config": {
            "n_dir": args.n_dir,
            "n_eval": args.n_eval,
            "model": args.model,
            "layers": args.layers,
            "k_pca": args.k_pca,
            "lams": args.lam,
            "seed": args.seed,
        },
        "alignment": alignment_json,
        "evaluation": all_eval_results,
        "decision": {
            "any_max_angle_above_45": any(
                alignment_results[L]["max_angle_deg"] > 45.0
                for L in args.layers
                if L in alignment_results
            ),
            "recommendation": (
                "ABANDON direction intervention"
                if any(
                    alignment_results[L]["max_angle_deg"] > 45.0
                    for L in args.layers
                    if L in alignment_results
                )
                else "PROCEED with aligned direction"
            ),
        },
    }

    out_path = output_dir / "subspace_intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # ── Final recommendation ──
    print(f"\n{'=' * 60}")
    print("Decision")
    print(f"{'=' * 60}")
    for L in args.layers:
        if L in alignment_results:
            max_angle = alignment_results[L]["max_angle_deg"]
            status = "⚠ UNRELIABLE" if max_angle > 45.0 else "✓ acceptable"
            print(f"  L{L}: max principal angle = {max_angle:.1f}° {status}")

    if any(
        alignment_results[L]["max_angle_deg"] > 45.0
        for L in args.layers
        if L in alignment_results
    ):
        print("\nCONCLUSION: Train/val subspaces are fundamentally misaligned.")
        print("Direction-based intervention should be ABANDONED for this layer.")
        print("Consider alternative approaches: TruthPrInt backtracking, ")
        print("adaptive decoding, or model-agnostic statistical features.")
    else:
        print("\nCONCLUSION: Subspaces are sufficiently aligned.")
        print("PCA-regularized direction may fix I1's non-transfer problem.")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
