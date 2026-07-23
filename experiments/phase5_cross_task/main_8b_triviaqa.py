"""8B TriviaQA Hallucination Detection — baseline feature extraction + EigenScore.

Ports Phase 5 (per-token features) and Phase 5.5 (EigenScore fast) to Qwen3-8B
on AutoDL RTX 5090 32GB. Single script, single model load, comprehensive output.

Key comparisons against 1.7B baseline:
  - max_p/entropy/d2_js AUROC on full-vocab TriviaQA
  - EigenScore (fast, noise-perturbation) AUROC
  - Knowledge filtering effectiveness
  - Correct/incorrect group separation

Usage:
    python main_8b_triviaqa.py --n_samples 200              # full run
    python main_8b_triviaqa.py --n_samples 20 --skip_eigen  # quick feature check
    python main_8b_triviaqa.py --n_samples 200 --eigen_K 5  # fast EigenScore
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Path setup
_sys_parent = Path(__file__).parent
sys.path.insert(0, str(_sys_parent.parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent.parent / "phase4_generalization"))
sys.path.insert(0, str(_sys_parent))

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct
from phase5_utils.generation_features import (
    generate_with_per_token_features,
    compute_all_pair_js,
)
from phase4_utils.generalization_features import compute_eigenscore_fast


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Per-token feature extraction during generation (Phase 5.1 pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_per_token_features(
    model,
    samples: list[dict],
    device: str,
    max_new_tokens: int = 20,
    js_early: int = 11,
    js_late: int = 27,
) -> list[dict]:
    """Extract per-token max_p, entropy, d2_js during greedy generation.

    Mirrors Phase 5.1 pipeline exactly for comparability.
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)

    results = []
    correct_count = 0

    for sample in tqdm(samples, desc="Generating + features"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        result = generate_with_per_token_features(
            model=model,
            prompt=prompt,
            W_U=W_U,
            b_U=b_U,
            max_new_tokens=max_new_tokens,
            js_early_layer=js_early,
            js_late_layer=js_late,
        )

        generated_text = result["answer_text"]
        is_correct = check_correct(generated_text, sample["answers"], dataset="triviaqa")
        if is_correct:
            correct_count += 1

        # Compute all-pair JS from last-token full-vocab distributions
        last_js_pairs = {}
        if result.get("last_token_vocab_probs") is not None:
            last_js_pairs = compute_all_pair_js(
                result["last_token_vocab_probs"], n_layers, exclude_layer0=True
            )

        results.append({
            "sample_id": len(results),
            "question": sample["question"],
            "answers": sample["answers"],
            "generated_text": generated_text,
            "is_correct": is_correct,
            "n_generated_tokens": len(result["per_token"]),
            "per_token": result["per_token"],
            "last_token_js_all_pairs": last_js_pairs,
        })

    print(f"  Correct: {correct_count}/{len(samples)} ({correct_count/len(samples):.1%})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: EigenScore fast computation (Phase 5.5 P0-1 pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_eigenscores(
    model,
    samples: list[dict],
    layer_idx: int = 3,
    K: int = 10,
    noise_scale: float = 1e-3,
) -> list[dict]:
    """Compute fast EigenScore for each sample.

    Uses embedding-noise perturbation (no generation) for 10-15x speedup.
    Layer 3 was optimal in 1.7B multi-layer scan.
    """
    results = []
    failed = 0

    for i, sample in enumerate(tqdm(samples, desc="EigenScore fast")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        try:
            score = compute_eigenscore_fast(
                model=model,
                prompt=prompt,
                layer_idx=layer_idx,
                K=K,
                noise_scale=noise_scale,
            )
            if np.isnan(score):
                failed += 1
                results.append({"sample_id": i, "eigenscore": None})
            else:
                results.append({"sample_id": i, "eigenscore": float(score)})
        except Exception:
            failed += 1
            results.append({"sample_id": i, "eigenscore": None})

    if failed > 0:
        print(f"  EigenScore failed: {failed}/{len(samples)}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Analysis — AUROC computation + knowledge filtering
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_features(
    per_sample: list[dict],
    eigenscores: list[dict] | None = None,
) -> dict:
    """Compute AUROC for all features, with and without knowledge filtering.

    Returns comprehensive results dict.
    """
    n_layers = max(
        len(s["per_token"][0]["max_p"]) if s["per_token"] else 0
        for s in per_sample
    )
    labels = np.array([s["is_correct"] for s in per_sample], dtype=np.int32)
    n_samples = len(labels)
    n_correct = int(labels.sum())
    print(f"\nAnalysis: {n_samples} samples, {n_correct} correct ({n_correct/n_samples:.1%})")

    # ── Aggregate per-token features to per-sample ──
    # For each feature, try multiple aggregation strategies
    agg_methods = {
        "last": lambda arr: arr[-1] if len(arr) > 0 else np.nan,
        "mean": lambda arr: np.mean(arr) if len(arr) > 0 else np.nan,
        "early_mean": lambda arr: np.mean(arr[: max(1, len(arr)//3)]) if len(arr) > 0 else np.nan,
    }

    feature_results = {}

    for feat_name in ["max_p", "entropy", "d2_js"]:
        for agg_name, agg_fn in agg_methods.items():
            vals = []
            for s in per_sample:
                tokens = s["per_token"]
                if not tokens:
                    vals.append(np.nan)
                    continue
                # Collapse per-layer lists to per-token scalars, then aggregate
                if feat_name == "d2_js":
                    token_vals = [t.get("d2_js", np.nan) for t in tokens]
                else:
                    token_vals = [np.mean(t[feat_name]) if isinstance(t[feat_name], (list, np.ndarray))
                                  else t[feat_name] for t in tokens]
                arr = np.array(token_vals, dtype=np.float64)
                vals.append(agg_fn(arr))
            vals = np.array(vals, dtype=np.float64)

            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue

            key = f"{feat_name}_{agg_name}"
            l_val = labels[valid]
            v_val = vals[valid]
            try:
                auroc = max(
                    roc_auc_score(l_val, v_val),
                    roc_auc_score(l_val, -v_val),
                )
            except ValueError:
                auroc = 0.5

            feature_results[key] = {
                "auroc": float(auroc),
                "n_valid": int(valid.sum()),
                "mean_correct": float(v_val[l_val == 1].mean()) if l_val.sum() > 0 else None,
                "mean_incorrect": float(v_val[l_val == 0].mean()) if (1 - l_val).sum() > 0 else None,
            }

    # ── Per-layer scan for max_p and entropy ──
    layer_results = {}
    for feat_name in ["max_p", "entropy"]:
        for agg_name, agg_fn in agg_methods.items():
            if agg_name == "early_mean":
                continue  # skip for per-layer scan (same as mean for single values)
            best_auroc, best_layer = 0.0, -1
            for layer in range(n_layers):
                vals = []
                for s in per_sample:
                    tokens = s["per_token"]
                    if not tokens:
                        vals.append(np.nan)
                        continue
                    arr = np.array([t[feat_name][layer] if isinstance(t[feat_name], (list, np.ndarray))
                                    else t[feat_name] for t in tokens])
                    vals.append(agg_fn(arr))
                vals = np.array(vals, dtype=np.float64)
                valid = ~np.isnan(vals)
                if valid.sum() < 10:
                    continue
                l_val = labels[valid]
                v_val = vals[valid]
                try:
                    auroc = max(roc_auc_score(l_val, v_val), roc_auc_score(l_val, -v_val))
                except ValueError:
                    auroc = 0.5
                if auroc > best_auroc:
                    best_auroc, best_layer = auroc, layer
            layer_results[f"{feat_name}_{agg_name}"] = {
                "best_layer": best_layer,
                "best_auroc": float(best_auroc),
            }

    # ── Knowledge filtering ──
    # Use mean max_p as knowledge proxy (same pattern as Phase 2/4)
    p_correct_vals = []
    for s in per_sample:
        tokens = s["per_token"]
        if not tokens:
            p_correct_vals.append(np.nan)
            continue
        mps = [np.mean(t["max_p"]) if isinstance(t["max_p"], (list, np.ndarray)) else t["max_p"]
               for t in tokens]
        p_correct_vals.append(np.mean(mps))
    p_correct_vals = np.array(p_correct_vals, dtype=np.float64)

    knowledge_results = {}
    for threshold in [0.0, 0.35, 0.40, 0.45, 0.50]:
        kmask = p_correct_vals > threshold
        k_n = kmask.sum()
        if k_n < 20:
            knowledge_results[f"thr_{threshold:.2f}"] = {"n": int(k_n), "error": "too few samples"}
            continue
        k_labels = labels[kmask]
        k_acc = k_labels.sum() / k_n

        # Scan best max_p AUROC in this filtered subset
        best_k_auroc = 0.0
        best_k_feat = ""
        for feat_name in ["max_p", "entropy"]:
            for agg_name in ["last", "mean"]:
                vals = []
                fn = agg_methods[agg_name]
                for s_idx in np.where(kmask)[0]:
                    s = per_sample[s_idx]
                    tokens = s["per_token"]
                    if not tokens:
                        vals.append(np.nan)
                        continue
                    arr = np.array([np.mean(t[feat_name]) if isinstance(t[feat_name], (list, np.ndarray))
                                    else t[feat_name] for t in tokens], dtype=np.float64)
                    vals.append(fn(arr))
                vals = np.array(vals, dtype=np.float64)
                valid_mask = ~np.isnan(vals)
                if valid_mask.sum() < 5:
                    continue
                try:
                    auroc = max(
                        roc_auc_score(k_labels[valid_mask], vals[valid_mask]),
                        roc_auc_score(k_labels[valid_mask], -vals[valid_mask]),
                    )
                except ValueError:
                    auroc = 0.5
                if auroc > best_k_auroc:
                    best_k_auroc = auroc
                    best_k_feat = f"{feat_name}_{agg_name}"

        knowledge_results[f"thr_{threshold:.2f}"] = {
            "n": int(k_n),
            "n_correct": int(k_labels.sum()),
            "accuracy": float(k_acc),
            "best_auroc": float(best_k_auroc),
            "best_feature": best_k_feat,
        }

    # ── EigenScore analysis ──
    eigen_results = None
    if eigenscores is not None:
        eigen_valid = [(e["sample_id"], e["eigenscore"])
                       for e in eigenscores if e["eigenscore"] is not None]
        if len(eigen_valid) >= 10:
            e_idx = np.array([v[0] for v in eigen_valid])
            e_vals = np.array([v[1] for v in eigen_valid], dtype=np.float64)
            e_labels = labels[e_idx]

            try:
                auroc_neg = roc_auc_score(1 - e_labels, -e_vals)
                auroc_pos = roc_auc_score(1 - e_labels, e_vals)
            except ValueError:
                auroc_neg = auroc_pos = 0.5

            eigen_results = {
                "n_valid": len(eigen_valid),
                "auroc_neg": float(auroc_neg),  # low eigenscore = hallucination
                "auroc_pos": float(auroc_pos),  # high eigenscore = hallucination
                "mean": float(e_vals.mean()),
                "std": float(e_vals.std()),
                "correct_mean": float(e_vals[e_labels == 1].mean()) if e_labels.sum() > 0 else None,
                "incorrect_mean": float(e_vals[e_labels == 0].mean()) if (1 - e_labels).sum() > 0 else None,
            }
            print(f"\nEigenScore: n={eigen_results['n_valid']}, "
                  f"AUROC(neg)={auroc_neg:.4f}, AUROC(pos)={auroc_pos:.4f}")

    # ── Print summary ──
    print("\n=== Feature AUROC Summary ===")
    for key in sorted(feature_results.keys()):
        r = feature_results[key]
        print(f"  {key:<30}: AUROC={r['auroc']:.4f} (n={r['n_valid']})")

    if layer_results:
        print("\n=== Best Layer per Feature ===")
        for key, r in sorted(layer_results.items()):
            print(f"  {key:<30}: L{r['best_layer']} = {r['best_auroc']:.4f}")

    if knowledge_results:
        print("\n=== Knowledge Filtering ===")
        for key in sorted(knowledge_results.keys()):
            r = knowledge_results[key]
            if "error" in r:
                print(f"  {key}: {r['error']}")
            else:
                print(f"  {key}: n={r['n']} acc={r['accuracy']:.1%} "
                      f"best_AUROC={r.get('best_auroc', 'N/A')}")

    return {
        "n_total": n_samples,
        "n_correct": n_correct,
        "accuracy": float(n_correct / n_samples),
        "feature_results": feature_results,
        "layer_results": layer_results,
        "knowledge_results": knowledge_results,
        "eigen_results": eigen_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="8B TriviaQA Hallucination Detection Baseline"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="outputs_8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--js_early", type=int, default=11)
    parser.add_argument("--js_late", type=int, default=27)
    # EigenScore args
    parser.add_argument("--skip_eigen", action="store_true",
                        help="Skip EigenScore computation")
    parser.add_argument("--eigen_layer", type=int, default=3,
                        help="Layer for EigenScore (L3 best on 1.7B)")
    parser.add_argument("--eigen_K", type=int, default=10,
                        help="K perturbed passes for EigenScore")
    parser.add_argument("--eigen_noise", type=float, default=1e-3,
                        help="Embedding noise scale for fast EigenScore")
    parser.add_argument("--eigen_n_layers", type=int, nargs="+", default=None,
                        help="If set, scan these layers for EigenScore (overrides --eigen_layer)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"8B TriviaQA Hallucination Detection")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Output: {output_dir}")

    # ── Load model ──
    print(f"\nLoading {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  Loaded: {n_layers} layers, d_model={d_model}")

    # ── Load data ──
    print(f"Loading TriviaQA ({args.n_samples} samples)...")
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  Loaded {len(samples)} samples")

    # ── Phase 1: Per-token feature extraction ──
    print(f"\n{'='*60}")
    print("Phase 1: Per-token feature extraction (greedy generation)")
    print(f"{'='*60}")
    t1 = time.time()
    per_sample = extract_per_token_features(
        model=model,
        samples=samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        js_early=args.js_early,
        js_late=args.js_late,
    )
    print(f"  Time: {(time.time() - t1) / 60:.1f} min")

    # Save intermediate
    features_file = output_dir / "triviaqa_8b_features.json"
    with open(features_file, "w") as f:
        json.dump({
            "config": {
                "model_id": args.model,
                "n_samples": args.n_samples,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
            },
            "per_sample": per_sample,
        }, f, indent=2)
    print(f"  Features saved to {features_file}")

    # ── Phase 2: EigenScore ──
    eigen_results = None
    if not args.skip_eigen:
        print(f"\n{'='*60}")
        print("Phase 2: EigenScore (fast, embedding-noise)")
        print(f"{'='*60}")

        if args.eigen_n_layers is not None:
            # Multi-layer scan
            eigen_by_layer = {}
            for layer in args.eigen_n_layers:
                print(f"\n  Layer {layer}...")
                t_e = time.time()
                scores = compute_eigenscores(
                    model=model,
                    samples=samples,
                    layer_idx=layer,
                    K=args.eigen_K,
                    noise_scale=args.eigen_noise,
                )
                eigen_by_layer[f"L{layer}"] = {
                    "scores": scores,
                    "time_s": float(time.time() - t_e),
                }
            eigen_results = {"by_layer": eigen_by_layer}
        else:
            t2 = time.time()
            scores = compute_eigenscores(
                model=model,
                samples=samples,
                layer_idx=args.eigen_layer,
                K=args.eigen_K,
                noise_scale=args.eigen_noise,
            )
            eigen_results = {
                "layer": args.eigen_layer,
                "K": args.eigen_K,
                "noise_scale": args.eigen_noise,
                "scores": scores,
                "time_s": float(time.time() - t2),
            }
            print(f"  Time: {eigen_results['time_s'] / 60:.1f} min")

        # Save EigenScore results
        eigen_file = output_dir / "triviaqa_8b_eigenscore.json"
        with open(eigen_file, "w") as f:
            json.dump(eigen_results, f, indent=2)
        print(f"  EigenScore saved to {eigen_file}")

    # ── Phase 3: Analysis ──
    print(f"\n{'='*60}")
    print("Phase 3: Analysis")
    print(f"{'='*60}")

    eigen_scores_for_analysis = None
    if eigen_results is not None and "scores" in eigen_results:
        eigen_scores_for_analysis = eigen_results["scores"]

    analysis = analyze_features(
        per_sample=per_sample,
        eigenscores=eigen_scores_for_analysis,
    )

    # Final result
    final = {
        "config": {
            "model_id": args.model,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "n_layers": n_layers,
            "d_model": d_model,
            "eigen_layer": args.eigen_layer,
            "eigen_K": args.eigen_K,
        },
        "analysis": analysis,
    }
    final_file = output_dir / "triviaqa_8b_analysis.json"
    with open(final_file, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nAnalysis saved to {final_file}")

    # ── Cleanup ──
    del model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"8B TriviaQA complete in {elapsed / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
