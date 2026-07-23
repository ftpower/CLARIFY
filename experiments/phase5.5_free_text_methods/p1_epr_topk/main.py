"""P1-2: EPR Top-K truncated entropy on TriviaQA.

Computes Top-K (K=10) truncated entropy at each decode step. The entropy
is computed only over the K most probable tokens, renormalized — avoiding
the 152K-dimensional noise that washes out the signal.

EPR = -sum(p_i * log(p_i)) where p_i = softmax(top-K logits) renormalized.

Per-token EPR is aggregated (last, mean, min, max) and evaluated for AUROC.

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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent / "phase5_cross_task"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import load_model_and_data, format_prompt
from phase5_utils.aggregation import aggregate_features, VALID_STRATEGIES


def compute_epr(logits: torch.Tensor, K: int = 10) -> float:
    """Top-K truncated entropy.

    Correct implementation: compute full-softmax FIRST, then take top-K,
    renormalize over the top-K mass, compute entropy.
    This preserves the probability mass of non-top tokens in the normalization.
    """
    probs = torch.softmax(logits.float(), dim=-1)  # full vocab softmax
    topk_probs, _ = torch.topk(probs, K)
    Z = topk_probs.sum()  # mass in top-K
    if Z < 1e-10:
        return 0.0
    topk_renorm = topk_probs / Z  # renormalize within top-K
    log_p = torch.log(topk_renorm.clamp(min=1e-12))
    return float(-(topk_renorm * log_p).sum().detach())


def generate_with_epr(model, prompt, W_U, b_U, n_layers, device,
                       max_new_tokens=10, K=10):
    """Greedy decode, extracting per-token EPR at the final layer."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    ln_final = model.ln_final
    per_token_epr = []
    generated_ids = []

    for step in range(max_new_tokens):
        storage = {}
        def _hook(name):
            def hook(act, hook):
                storage[name] = act[:, -1, :].detach()
            return hook

        fwd_hooks = [(f"blocks.{n_layers - 1}.hook_resid_post", _hook("L_last"))]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # EPR at final layer
        h = ln_final(storage["L_last"].to(device))
        logits_L = (h @ W_U).squeeze(0)  # [vocab_size]
        if b_U is not None:
            logits_L = logits_L + b_U
        epr = compute_epr(logits_L, K=K)
        per_token_epr.append({"epr": epr})

        next_id = logits[0, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_id)
        if next_id == model.tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=-1)

    answer_text = model.tokenizer.decode(generated_ids).strip() if generated_ids else ""
    return {
        "answer_text": answer_text,
        "n_tokens": len(generated_ids),
        "per_token": per_token_epr,
    }


def main(n_samples=200, device="cuda", model_id="Qwen/Qwen3-1.7B",
         output_dir="outputs", seed=42, K=10):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, samples, W_U, b_U = load_model_and_data(
        n_samples=n_samples, seed=seed, device=device, model_id=model_id
    )
    n_layers = model.cfg.n_layers

    # Load labels
    phase5_json = (_sys_parent / "phase5_cross_task" / "outputs" / "triviaqa_features.json")
    with open(phase5_json) as f:
        p5_labels = [s["is_correct"] for s in json.load(f)["per_sample"]]

    # Per-sample EPR extraction
    print(f"\nExtracting per-token EPR (K={K})...")
    per_sample = []
    for idx, sample in enumerate(tqdm(samples, desc="EPR")):
        prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
        result = generate_with_epr(model, prompt, W_U, b_U, n_layers, device, K=K)
        agg = aggregate_features(result["per_token"], feature_keys=["epr"],
                                 strategies=VALID_STRATEGIES)
        per_sample.append({
            "is_correct": p5_labels[idx] if idx < len(p5_labels) else False,
            "per_token": result["per_token"],
            "aggregated": agg,
        })
        if (idx + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    # Evaluate AUROC per strategy
    labels = np.array([s["is_correct"] for s in per_sample], dtype=np.int32)
    print(f"\n{'Strategy':<14} {'AUROC':<10}")
    print("-" * 26)
    best_auroc, best_strat = 0.5, ""
    strat_results = {}
    for strat in VALID_STRATEGIES:
        scores = np.array([s["aggregated"]["epr"][strat] for s in per_sample])
        try:
            auc = roc_auc_score(1 - labels, scores)
        except ValueError:
            auc = 0.5
        print(f"  {strat:<14} {auc:<10.4f}")
        strat_results[strat] = float(auc)
        if auc > best_auroc:
            best_auroc, best_strat = auc, strat
    print(f"\n  Best: {best_strat} = {best_auroc:.4f}")

    results = {"config": {"n_samples": n_samples, "K": K, "seed": seed},
               "best_strategy": best_strat, "best_auroc": best_auroc,
               "per_strategy": strat_results}
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / 'results.json'}")
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1-2: EPR Top-K on TriviaQA")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=10)
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device, model_id=args.model,
         output_dir=args.output_dir, seed=args.seed, K=args.K)
