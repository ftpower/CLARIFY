"""A1: Embedding-Weighted Confidence — semantic probability mass aggregation.

Hypothesis: When the model knows "Pulsar", nearby tokens ("pulsating", "pulsar",
"pulse") also have elevated probability. The total similarity-weighted probability
mass is a better confidence signal than max_p of any single token.

At each generation step:
  1. Get full-vocab softmax → top-200 token indices + probs
  2. Compute cosine similarity between each top-200 token's embedding and the
     ACTUALLY GENERATED token's embedding
  3. Weight each token's probability by similarity: score = sum(p_i * relu(cos_sim_i))
  4. Also compute "concept mass": total prob of tokens with cos_sim > 0.5

Inspired by: Semantic Entropy (Kuhn 2023) — clustering tokens by semantic
equivalence, but simplified to single-pass embedding similarity.

Usage:
    python A1_embed_weighted.py --n_samples 200
    python A1_embed_weighted.py --n_samples 50 --model Qwen/Qwen3-1.7B
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
# Core: embedding-weighted confidence during greedy generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_embed_weighted(
    model, tokenizer, prompt: str, device: str,
    max_new_tokens: int = 20,
    top_k: int = 200,
    cos_threshold: float = 0.5,
) -> dict:
    """Greedy decode extracting embedding-weighted confidence per token.

    Returns per_token features:
      - embed_weighted_confidence: sum(prob_i * max(0, cos_sim(emb_i, emb_gen)))
      - concept_mass: total prob of tokens with cos_sim > threshold
      - max_cos_sim: highest cosine similarity among top-K tokens
      - weighted_rank: rank of generated token after similarity weighting
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U  # keep on model's device, don't duplicate
    b_U = model.unembed.b_U

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    per_token_features = []
    generated_ids = []

    for step in range(max_new_tokens):
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

        # Fetch generated token's embedding from W_U (on CPU or GPU)
        gen_emb_raw = W_U[:, next_id].float().to(device)  # [d_model]
        gen_emb = F.normalize(gen_emb_raw, dim=0)  # unit-norm [d_model]

        step_features = {"step": step, "token_id": next_id,
                         "token_text": tokenizer.decode([next_id])}

        for i in range(n_layers):
            h = residuals[f"L{i}"].to(device)
            if i == n_layers - 1:
                h = model.ln_final(h)
            logits_l = (h @ W_U).float()
            if b_U is not None:
                logits_l = logits_l + b_U
            probs = F.softmax(logits_l, dim=-1).squeeze(0)  # [vocab_size]

            # Get top-K tokens (K=200, tiny)
            topk_probs, topk_ids = probs.topk(top_k, dim=-1)  # [K]

            # Fetch only K embedding vectors, normalize on the fly
            topk_embs = W_U[:, topk_ids].float().to(device)  # [d_model, K]
            topk_embs_norm = F.normalize(topk_embs, dim=0)  # unit-norm columns

            # Cosine similarity between each top-K embedding and generated token
            cos_sims = (gen_emb.unsqueeze(0) @ topk_embs_norm).squeeze(0)  # [K]
            cos_sims_clamped = torch.clamp(cos_sims, min=0)  # ReLU

            # Embedding-weighted confidence
            embed_weighted = float((topk_probs * cos_sims_clamped).sum().item())
            step_features.setdefault("embed_weighted", []).append(embed_weighted)

            # Concept mass
            concept_mask = cos_sims > cos_threshold
            concept_mass = float(topk_probs[concept_mask].sum().item())
            step_features.setdefault("concept_mass", []).append(concept_mass)

            # Max cosine similarity among top-K
            max_cos = float(cos_sims.max().item())
            step_features.setdefault("max_cos_sim", []).append(max_cos)

        per_token_features.append(step_features)

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
# Aggregation (same pattern as A2)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_feature(per_token: list[dict], feat_key: str,
                      method: str = "early_mean") -> float:
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
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A1: Embedding-Weighted Confidence")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"A1: Embedding-Weighted Confidence (top_k={args.top_k})")
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

    print(f"\nExtracting embedding-weighted features...")
    t0 = time.time()
    results = []
    correct_count = 0

    for s in tqdm(samples, desc="A1 embed-weighted"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        feats = generate_with_embed_weighted(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
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
    features = ["embed_weighted", "concept_mass", "max_cos_sim"]
    agg_methods = ["last", "mean", "early_mean"]

    feature_configs = []
    for feat in features:
        for method in agg_methods:
            feature_configs.append({
                "key": f"{feat}_{method}",
                "name": f"{feat}_{method}",
                "invert": False,
            })

    for r in results:
        for feat in features:
            for method in agg_methods:
                r[f"{feat}_{method}"] = aggregate_feature(
                    r["per_token"], feat, method
                )

    print(f"\nAUROC Results:")
    auroc_summary = evaluate_all(results, labels, feature_configs)
    print_summary(auroc_summary)

    # Save
    save_results(
        results, auroc_summary,
        output_path=str(output_dir / "A1_embed_weighted.json"),
    )

    print(f"{'='*60}")
    print(f"A1 complete — compare with baseline max_p 0.652")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
