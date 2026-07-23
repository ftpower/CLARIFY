"""A3: Embedding Dispersion — semantic diversity of top-K candidate tokens.

Hypothesis: When the model is uncertain (hallucinating), its top candidate tokens
are semantically diverse (e.g., "Paris", "London", "and", "the"). When certain,
they cluster around one concept.

At each generation step:
  1. Get top-50 token embeddings from the unembedding matrix
  2. Compute mean pairwise cosine distance
  3. High dispersion = tokens are diverse = uncertain = hallucination
  4. Also compute: distance from generated token to centroid of top-K

Inspired by: token embedding geometry analysis — measuring how "spread out" the
model's top candidates are in semantic space.

Usage:
    python A3_embedding_dispersion.py --n_samples 200
"""

import argparse
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
# Core
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_dispersion(
    model, tokenizer, prompt: str, device: str,
    max_new_tokens: int = 20,
    top_k: int = 50,
) -> dict:
    """Greedy decode extracting embedding dispersion per token.

    Features:
      - mean_pairwise_cos_dist: 1 - mean(cosine_sim(i, j)) across top-K emb pairs
        High = diverse = uncertain. Uses random subset for efficiency.
      - centroid_distance: distance from generated token to centroid of top-K
      - nearest_neighbor_sim: max cosine similarity IN the top-K set
        Low = no dominant cluster = uncertain
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U  # keep on model's device
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

        step_features = {"step": step, "token_id": next_id,
                         "token_text": tokenizer.decode([next_id])}

        for i in range(n_layers):
            h = residuals[f"L{i}"].to(device)
            if i == n_layers - 1:
                h = model.ln_final(h)
            logits_l = (h @ W_U).float()
            if b_U is not None:
                logits_l = logits_l + b_U
            probs = F.softmax(logits_l, dim=-1).squeeze(0)

            topk_ids = probs.topk(top_k, dim=-1).indices  # [K]
            # Fetch only K embeddings, normalize on the fly (avoid full W_U_norm OOM)
            topk_embs = F.normalize(W_U[:, topk_ids].float().to(device), dim=0)

            # ── Pairwise cosine distance (efficient: via pairwise_dot) ──
            # cos_sim_matrix = topk_embs^T @ topk_embs → [K, K]
            cos_matrix = topk_embs.T @ topk_embs  # [K, K]
            # Mean of upper triangle (excluding diagonal)
            mask = ~torch.eye(top_k, dtype=torch.bool, device=device)
            mean_pairwise_sim = float(cos_matrix[mask].mean().item())
            mean_pairwise_dist = 1.0 - mean_pairwise_sim  # 0=all identical, 2=max diverse
            step_features.setdefault("mean_pairwise_dist", []).append(mean_pairwise_dist)

            # ── Nearest neighbor similarity (max off-diagonal) ──
            cos_matrix_diag = cos_matrix.clone()
            cos_matrix_diag[torch.eye(top_k, dtype=torch.bool, device=device)] = -1.0
            nn_sim = float(cos_matrix_diag.max().item())
            step_features.setdefault("nearest_neighbor_sim", []).append(nn_sim)

            # ── Centroid distance to generated token ──
            centroid = topk_embs.mean(dim=1)  # [d_model]
            gen_emb_raw = W_U[:, next_id].float().to(device)  # [d_model]
            gen_emb = F.normalize(gen_emb_raw, dim=0)
            centroid_cos_sim = float((gen_emb @ centroid).item())
            centroid_dist = 1.0 - centroid_cos_sim
            step_features.setdefault("centroid_dist", []).append(centroid_dist)

            # ── Eigenvalue-based dispersion (ratio of top to sum) ──
            # The singular values of the embedding matrix capture spread
            # Quick: ratio of max eigenvalue to trace of covariance
            centered = topk_embs - topk_embs.mean(dim=1, keepdim=True)
            cov = centered @ centered.T  # [d_model, d_model] — but too large
            # Use KxK Gram matrix instead: G = centered^T @ centered [K, K]
            G = centered.T @ centered  # [K, K]
            eigvals = torch.linalg.eigvalsh(G.float())  # [K], ascending
            eigvals = eigvals[-min(top_k, 20):]  # top 20 eigenvalues
            # Effective rank: sum(eigvals)^2 / sum(eigvals^2)
            eig_sum = eigvals.sum()
            eig_sq_sum = (eigvals ** 2).sum()
            effective_rank = float((eig_sum ** 2 / eig_sq_sum).item()) if eig_sq_sum > 0 else 1.0
            step_features.setdefault("effective_rank", []).append(effective_rank)

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
# Aggregation
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
    parser = argparse.ArgumentParser(description="A3: Embedding Dispersion")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"A3: Embedding Dispersion (top_k={args.top_k})")
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

    print(f"\nExtracting embedding dispersion features...")
    t0 = time.time()
    results = []
    correct_count = 0

    for s in tqdm(samples, desc="A3 dispersion"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        feats = generate_with_dispersion(
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

    labels = [1 if r["is_correct"] else 0 for r in results]
    features = ["mean_pairwise_dist", "nearest_neighbor_sim",
                "centroid_dist", "effective_rank"]
    agg_methods = ["last", "mean", "early_mean"]

    feature_configs = []
    for feat in features:
        for method in agg_methods:
            # Higher dispersion = worse, so invert
            invert = feat not in ["nearest_neighbor_sim"]  # high nn_sim = good
            feature_configs.append({
                "key": f"{feat}_{method}",
                "name": f"{feat}_{method}",
                "invert": invert,
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

    save_results(
        results, auroc_summary,
        output_path=str(output_dir / "A3_embedding_dispersion.json"),
    )

    print(f"{'='*60}")
    print(f"A3 complete — compare with baseline max_p 0.652")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
