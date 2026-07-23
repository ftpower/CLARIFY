"""A2: Top-K Probability Concentration — measure distribution concentration per token.

Hypothesis: When the model knows the answer, probability concentrates in fewer tokens.
When hallucinating, probability spreads thin across many unrelated tokens.

Measures per token:
  - topk_mass_{K}: fraction of total probability in top-K tokens (K=10, 50, 100)
  - effective_k_{pct}: K needed to reach X% cumulative mass (50%, 80%, 95%)

Based on: ShED-HD (entropy concentration) and Information Deficiency (layer-wise ID).

Usage:
    python A2_topk_mass.py --n_samples 200
    python A2_topk_mass.py --n_samples 50 --model Qwen/Qwen3-1.7B
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

from shared import load_model_and_data, evaluate_all, save_results, print_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Top-K concentration during greedy generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_topk_concentration(
    model, tokenizer, prompt: str, device: str,
    max_new_tokens: int = 20,
    K_values: list[int] = [10, 50, 100, 200, 500],
    mass_thresholds: list[float] = [0.5, 0.8, 0.95],
) -> dict:
    """Greedy decode extracting top-K concentration metrics at every step.

    At each step, computes the full-vocabulary softmax, then measures:
      - topk_mass_K: fraction of total probability in top-K tokens
      - effective_k_pct: minimum K to reach pct% cumulative probability

    Returns:
        dict with keys:
          - answer_text: str
          - per_token: list[dict] with "topk_mass_{K}" and "eff_k_{pct}" values
          - n_tokens: int
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    per_token_features = []
    generated_ids = []

    for step in range(max_new_tokens):
        # ── Forward pass (hook only final layer for efficiency) ──────────
        residuals = {}

        def _make_hook(name):
            def hook(act, hook=None, **kwargs):
                residuals[name] = act[:, -1, :].detach()
            return hook

        fwd_hooks = [
            (f"blocks.{i}.hook_resid_post", _make_hook(f"L{i}"))
            for i in range(n_layers)
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        next_id = int(logits[0, -1, :].argmax().item())
        generated_ids.append(next_id)

        # ── Per-layer top-K concentration ─────────────────────────────────
        step_features = {"step": step, "token_id": next_id,
                         "token_text": tokenizer.decode([next_id])}

        for i in range(n_layers):
            h = residuals[f"L{i}"].to(device)
            if i == n_layers - 1:
                h = model.ln_final(h)
            logits_l = (h @ W_U)
            if b_U is not None:
                logits_l = logits_l + b_U
            probs = F.softmax(logits_l.float(), dim=-1)  # [1, vocab_size]

            # top-K mass
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            for K in K_values:
                mass = float(sorted_probs[0, :K].sum().item())
                step_features.setdefault(f"topk_mass_{K}", []).append(mass)

            # effective K
            for pct in mass_thresholds:
                idx = int((cumsum >= pct).float().argmax().item())
                step_features.setdefault(f"eff_k_{int(pct*100)}", []).append(idx + 1)  # 1-indexed

        per_token_features.append(step_features)

        # ── Check EOS ──────────────────────────────────────────────────────
        if next_id == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)

    answer_text = tokenizer.decode(generated_ids).strip()
    return {
        "answer_text": answer_text,
        "per_token": per_token_features,
        "n_tokens": len(per_token_features),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_feature(per_token: list[dict], feat_key: str,
                      method: str = "early_mean") -> float:
    """Aggregate per-token per-layer feature to scalar.

    feat_key can be "topk_mass_50" (list per token) or "eff_k_50" (list per token).
    Each per-token value is a list of per-layer values; we mean across layers first.
    """
    if not per_token:
        return float("nan")

    token_vals = []
    for t in per_token:
        layer_vals = t.get(feat_key, [])
        if layer_vals:
            token_vals.append(float(np.mean([v for v in layer_vals
                                             if np.isfinite(v)])))

    if not token_vals:
        return float("nan")

    if method == "last":
        return token_vals[-1]
    elif method == "mean":
        return float(np.mean(token_vals))
    elif method == "early_mean":
        n = max(1, len(token_vals) // 3)
        return float(np.mean(token_vals[:n]))
    else:
        return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A2: Top-K Probability Concentration")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"A2: Top-K Probability Concentration")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # Load
    print("Loading model & data...")
    t0 = time.time()
    model, tokenizer, samples = load_model_and_data(
        n_samples=args.n_samples, seed=args.seed,
        device=device, model_id=args.model,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Feature extraction
    from src.data_loader import format_prompt, check_correct

    print(f"\nExtracting top-K concentration features...")
    t0 = time.time()
    results = []
    correct_count = 0

    for s in tqdm(samples, desc="A2 top-K mass"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        feats = generate_with_topk_concentration(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
        )
        is_correct = check_correct(
            feats["answer_text"], s["answers"], dataset="triviaqa"
        )
        if is_correct:
            correct_count += 1

        results.append({
            "sample_id": len(results),
            "question": s["question"],
            "answers": s["answers"],
            "generated_text": feats["answer_text"],
            "is_correct": is_correct,
            "n_tokens": feats["n_tokens"],
            "per_token": feats["per_token"],
        })

    elapsed = time.time() - t0
    print(f"  Correct: {correct_count}/{len(samples)} ({correct_count/len(samples):.1%})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")

    # Aggregation + AUROC
    labels = [1 if r["is_correct"] else 0 for r in results]

    # Build feature configs
    K_values = [10, 50, 100, 200, 500]
    mass_thresholds = [0.5, 0.8, 0.95]
    agg_methods = ["last", "mean", "early_mean"]

    feature_configs = []
    # top-K mass (higher = more concentrated = more correct)
    for K in K_values:
        for method in agg_methods:
            feature_configs.append({
                "key": f"topk_mass_{K}_{method}",
                "name": f"topk_mass_K={K}_{method}",
                "invert": False,
            })
    # effective K (lower = fewer tokens needed = more concentrated)
    for pct in mass_thresholds:
        for method in agg_methods:
            feature_configs.append({
                "key": f"eff_k_{int(pct*100)}_{method}",
                "name": f"eff_k_{int(pct*100)}pct_{method}",
                "invert": True,  # lower eff_k = better
            })

    # Compute aggregated features
    for r in results:
        for K in K_values:
            for method in agg_methods:
                r[f"topk_mass_{K}_{method}"] = aggregate_feature(
                    r["per_token"], f"topk_mass_{K}", method
                )
        for pct in mass_thresholds:
            for method in agg_methods:
                r[f"eff_k_{int(pct*100)}_{method}"] = aggregate_feature(
                    r["per_token"], f"eff_k_{int(pct*100)}", method
                )

    # Evaluate
    print(f"\nAUROC Results:")
    auroc_summary = evaluate_all(results, labels, feature_configs)

    # Best individual layer analysis
    n_layers = model.cfg.n_layers
    best_layer_results = {}
    for K in [50, 100]:  # Most interesting K values
        for layer in range(n_layers):
            key = f"topk_mass_{K}_L{layer}"
            scores = []
            for r in results:
                per_token = r["per_token"]
                vals = []
                for t in per_token:
                    layer_vals = t.get(f"topk_mass_{K}", [])
                    if len(layer_vals) > layer:
                        vals.append(layer_vals[layer])
                if vals:
                    scores.append(float(np.mean(vals[:max(1, len(vals)//3)])))
                else:
                    scores.append(float("nan"))
            labels_arr = np.array(labels)
            valid = np.isfinite(scores)
            if valid.sum() >= 10 and labels_arr[valid].std() > 0:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels_arr[valid],
                                    np.array(scores)[valid])
                best_layer_results[key] = {"layer": layer, "auroc": float(auc)}

    if best_layer_results:
        best = max(best_layer_results.values(), key=lambda x: x["auroc"])
        print(f"\n  Best per-layer: {max(best_layer_results, key=lambda k: best_layer_results[k]['auroc'])}")
        print(f"  Layer {best['layer']}: AUROC={best['auroc']:.4f}")

    # Save
    print_summary(auroc_summary)
    save_results(
        results, auroc_summary,
        output_path=str(output_dir / "A2_topk_mass.json"),
        extra={"best_layer_results": best_layer_results},
    )

    print(f"{'='*60}")
    print(f"A2 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
