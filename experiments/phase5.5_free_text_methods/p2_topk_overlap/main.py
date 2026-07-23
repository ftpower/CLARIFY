"""P2-1: Top-K Token Set Overlap Ratio on TriviaQA.

Measures the overlap between L_early and L_late's top-K token ID sets
at each decode step. Unlike JS, this metric ignores probability values
and only checks WHICH tokens the layers focus on.

Metrics:
  - Jaccard: |S_early ∩ S_late| / |S_early ∪ S_late|
  - Overlap ratio: |S_early ∩ S_late| / K

Usage:
    python main.py --n_samples 200
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent / "phase5_cross_task"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import load_model_and_data, format_prompt
from phase5_utils.aggregation import aggregate_features, VALID_STRATEGIES


def compute_overlap(topk_early: torch.Tensor, topk_late: torch.Tensor) -> dict:
    """Compute token set overlap metrics between two top-K ID vectors."""
    set_e = set(topk_early.cpu().tolist())
    set_l = set(topk_late.cpu().tolist())
    inter = len(set_e & set_l)
    union = len(set_e | set_l)
    return {
        "jaccard": inter / union if union > 0 else 0.0,
        "overlap_ratio": inter / len(set_e) if len(set_e) > 0 else 0.0,
    }


def generate_with_overlap(model, prompt, W_U, b_U, n_layers, device,
                           early_layer=11, late_layer=27,
                           max_new_tokens=10, K=100):
    """Greedy decode, extracting per-token top-K overlap at each step."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    ln_final = model.ln_final
    per_token = []
    generated_ids = []

    for step in range(max_new_tokens):
        storage = {}
        def _hook(name):
            def hook(act, hook):
                storage[name] = act[:, -1, :].detach()
            return hook

        fwd_hooks = [
            (f"blocks.{early_layer}.hook_resid_post", _hook("h_early")),
            (f"blocks.{late_layer}.hook_resid_post", _hook("h_late")),
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Logit lens projection for target layers
        def get_topk(h_key, layer_idx):
            h = storage[h_key].to(device)
            if layer_idx == n_layers - 1:
                h = ln_final(h)
            logits_L = (h @ W_U).squeeze(0)
            if b_U is not None:
                logits_L = logits_L + b_U
            return torch.topk(logits_L, K).indices

        topk_e = get_topk("h_early", early_layer)
        topk_l = get_topk("h_late", late_layer)
        overlap = compute_overlap(topk_e, topk_l)
        per_token.append(overlap)

        next_id = logits[0, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_id)
        if next_id == model.tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=-1)

    return {
        "n_tokens": len(generated_ids),
        "per_token": per_token,
    }


def main(n_samples=200, device="cuda", model_id="Qwen/Qwen3-1.7B",
         output_dir="outputs", seed=42, K=100, early=11, late=27):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, samples, W_U, b_U = load_model_and_data(
        n_samples=n_samples, seed=seed, device=device, model_id=model_id
    )
    n_layers = model.cfg.n_layers

    phase5_json = (_sys_parent / "phase5_cross_task" / "outputs" / "triviaqa_features.json")
    with open(phase5_json) as f:
        p5_labels = [s["is_correct"] for s in json.load(f)["per_sample"]]

    print(f"\nExtracting top-K overlap (K={K}, L{early} vs L{late})...")
    per_sample = []
    for idx, sample in enumerate(tqdm(samples, desc="Overlap")):
        prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
        result = generate_with_overlap(
            model, prompt, W_U, b_U, n_layers, device,
            early_layer=early, late_layer=late, K=K,
        )
        for metric in ["jaccard", "overlap_ratio"]:
            agg = aggregate_features(
                result["per_token"],
                feature_keys=[metric],
                strategies=VALID_STRATEGIES,
            )
            if "aggregated" not in sample:
                result["aggregated"] = {}
            # Merge: flatten structure
            if "aggregated" not in result:
                result["aggregated"] = {}
            # Manually merge
        # Rebuild with both metrics
        all_agg = {}
        for metric in ["jaccard", "overlap_ratio"]:
            agg = aggregate_features(
                result["per_token"],
                feature_keys=[metric],
                strategies=VALID_STRATEGIES,
            )
            all_agg[metric] = agg[metric]
        per_sample.append({
            "is_correct": p5_labels[idx] if idx < len(p5_labels) else False,
            "per_token": result["per_token"],
            "aggregated": all_agg,
        })
        if (idx + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    labels = np.array([s["is_correct"] for s in per_sample], dtype=np.int32)

    # Evaluate all metric × strategy combos
    results_summary = {}
    for metric in ["jaccard", "overlap_ratio"]:
        print(f"\n--- {metric} ---")
        print(f"  {'Strategy':<14} {'AUROC':<10}")
        best_auc, best_s = 0.5, ""
        for strat in VALID_STRATEGIES:
            scores = np.array([s["aggregated"][metric][strat] for s in per_sample])
            try:
                auc = roc_auc_score(1 - labels, scores)
            except ValueError:
                auc = 0.5
            print(f"  {strat:<14} {auc:<10.4f}")
            if auc > best_auc:
                best_auc, best_s = auc, strat
        print(f"  Best: {best_s} = {best_auc:.4f}")
        results_summary[metric] = {"best_strategy": best_s, "best_auroc": best_auc}

    results = {"config": {"n_samples": n_samples, "K": K, "early": early, "late": late, "seed": seed},
               "results": results_summary}
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path / 'results.json'}")
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-1: Top-K overlap on TriviaQA")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--early", type=int, default=11)
    parser.add_argument("--late", type=int, default=27)
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device, model_id=args.model,
         output_dir=args.output_dir, seed=args.seed, K=args.K,
         early=args.early, late=args.late)
