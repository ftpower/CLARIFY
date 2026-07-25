"""Phase 16.4: RepE-style Contrastive Prompt + All-Position Intervention.

Representation Engineering (Zou et al. 2023):
  - Extract "behavior direction" from contrastive prompt pairs
  - Inject at ALL token positions to create a global "mode"
  - Key: all-position injection prevents downstream compensation

Key difference from all previous phases:
  1. Direction comes from contrastive prompts, not mean(correct)-mean(wrong)
  2. Injection at ALL positions, not just the last token
  3. Creates a global behavioral mode shift

Usage:
  python phase16_repe.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract.json \
    --n_test 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════
# Contrastive Prompt Extraction
# ═══════════════════════════════════════════════════════════════════

NORMAL_PROMPT = "Answer the question with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:"

TRUTH_PROMPT = (
    "Answer the following question truthfully and accurately. "
    "Provide only the correct factual answer with a single word or short phrase.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

CONTRASTIVE_VARIANTS = {
    "accuracy": [
        "Answer the question with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:",
        "Answer the following question truthfully and accurately with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:",
    ],
    "confidence": [
        "Answer the question with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:",
        "Think carefully and answer the following question with high confidence only if you are certain.\n\nQuestion: {question}\n\nAnswer:",
    ],
    "instruction": [
        "Answer the question with a single word or short phrase.\n\nQuestion: {question}\n\nAnswer:",
        "You are a factually accurate AI assistant. Answer with only verified information.\n\nQuestion: {question}\n\nAnswer:",
    ],
}


def extract_contrastive_direction(
    model, records, layer: int, device: str, variant: str = "accuracy"
) -> np.ndarray:
    """Extract RepE direction from contrastive prompt pairs.

    For each question, run two forward passes (normal + truth prompt),
    collect residual stream at all positions, compute difference.
    """
    normal_template, truth_template = CONTRASTIVE_VARIANTS[variant]
    hook_name = f"blocks.{layer}.hook_resid_post"
    storage = {"diffs": []}

    def _capture_normal(act, hook=None):
        storage["normal"] = act[0].clone()  # [seq, d]
        return act

    def _capture_truth(act, hook=None):
        storage["truth"] = act[0].clone()  # [seq, d]
        return act

    for rec in tqdm(records, desc=f"  Extracting {variant} contrastive"):
        question = rec["question"]

        # Normal prompt
        normal_prompt = normal_template.format(question=question)
        tokens_n = model.to_tokens(normal_prompt, prepend_bos=True)
        if tokens_n.shape[1] > 1024:
            tokens_n = tokens_n[:, :1024]

        with torch.no_grad():
            model.run_with_hooks(tokens_n, fwd_hooks=[(hook_name, _capture_normal)])
        h_normal = storage["normal"]  # [seq, d]

        # Truth prompt
        truth_prompt = truth_template.format(question=question)
        tokens_t = model.to_tokens(truth_prompt, prepend_bos=True)
        if tokens_t.shape[1] > 1024:
            tokens_t = tokens_t[:, :1024]

        with torch.no_grad():
            model.run_with_hooks(tokens_t, fwd_hooks=[(hook_name, _capture_truth)])
        h_truth = storage["truth"]  # [seq, d]

        # Only compare at the LAST token position (answer prefix)
        # Normal and truth prompts have different lengths, so we compare
        # only the last-token representation where answer generation begins
        diff = h_truth[-1, :] - h_normal[-1, :]  # [d]
        storage["diffs"].append(diff.cpu().numpy())

    all_diffs = np.stack(storage["diffs"], axis=0)  # [n, d]
    v = all_diffs.mean(axis=0)  # [d]
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


# ═══════════════════════════════════════════════════════════════════
# RepE Intervention
# ═══════════════════════════════════════════════════════════════════


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20):
    """Core greedy generation with hooks."""
    gids = []
    for _step in range(max_new):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break
    return tokenizer.decode(gids).strip()


def make_repe_hook(
    layer: int,
    v_repe: np.ndarray,
    alpha: float,
    device: str,
    all_positions: bool = True,
):
    """Create RepE hook that adds α·v_repe to residual stream.

    Args:
      all_positions: if True, modify ALL positions (RepE style).
                     if False, only last position (our old style).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    mod_vec = torch.tensor(alpha * v_repe, dtype=torch.float32, device=device)

    def _hook(act, hook=None):
        if all_positions:
            act[0] = act[0] + mod_vec  # [seq, d] all positions
        else:
            act[0, -1, :] = act[0, -1, :] + mod_vec  # only last
        return act

    return hook_name, _hook


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 16 RepE: Contrastive Prompt + All-Position"
    )
    parser.add_argument("--load", required=True)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="*", default=[20])
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[-1.0, -0.5, 0.5, 1.0]
    )
    parser.add_argument(
        "--variants", nargs="*", default=["accuracy", "confidence", "instruction"]
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    with open(args.load) as f:
        data = json.load(f)
    all_records = data["records"]

    n_test = min(args.n_test, len(all_records) // 2)
    n_train = len(all_records) - n_test
    train_records = all_records[:n_train]
    test_records = all_records[n_train:]
    print(f"  Train: {n_train}, Test: {n_test}")

    # ── Load model ────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    tokenizer = model.tokenizer
    print(f"  Model: {model.cfg.model_name}")

    # ── Baseline ──────────────────────────────────────────────
    print("\n[3/4] Computing baseline...")
    baseline_correct = 0
    for rec in tqdm(test_records, desc="  Baseline"):
        question = rec["question"]
        context = rec.get("context", "")
        gt_answers = rec["gt_answers"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(model, tokenizer, tokens, device, [])
        if check_correct(generated, gt_answers, dataset="triviaqa"):
            baseline_correct += 1
    baseline_rate = baseline_correct / n_test
    print(f"  Baseline: {baseline_correct}/{n_test} = {baseline_rate:.1%}")

    # ── RepE Intervention ─────────────────────────────────────
    print("\n[4/4] RepE contrastive intervention...")
    results = {"baseline_rate": baseline_rate, "variants": {}}

    for variant in args.variants:
        print(f"\n  ── Variant: {variant} ──")
        variant_results = {}

        for layer in args.layers:
            # Extract direction from train set
            v_repe = extract_contrastive_direction(
                model, train_records[:50], layer, device, variant
            )
            print(f"    L{layer} ||v||={np.linalg.norm(v_repe):.4f}")

            for alpha in args.alphas:
                # All-position (RepE style)
                hook_name, hook_fn = make_repe_hook(
                    layer, v_repe, alpha, device, all_positions=True
                )
                correct = 0
                for rec in tqdm(
                    test_records, desc=f"    L{layer} α={alpha:+.1f}", leave=False
                ):
                    question = rec["question"]
                    context = rec.get("context", "")
                    gt_answers = rec["gt_answers"]
                    prompt = format_prompt(question, context, dataset="triviaqa")
                    tokens = model.to_tokens(prompt, prepend_bos=True)
                    if tokens.shape[1] > 1024:
                        tokens = tokens[:, :1024]
                    generated = _gen_greedy(
                        model, tokenizer, tokens, device, [(hook_name, hook_fn)]
                    )
                    if check_correct(generated, gt_answers, dataset="triviaqa"):
                        correct += 1

                rate = correct / n_test
                delta = rate - baseline_rate
                key = f"L{layer}_α{alpha:+.1f}"
                variant_results[key] = {"rate": rate, "delta": float(delta)}
                print(f"    {key}: {rate:.1%} (Δ={delta:+.1%})")

        # Best for this variant
        best_key = max(variant_results, key=lambda k: variant_results[k]["rate"])
        best_rate = variant_results[best_key]["rate"]
        print(f"    Best: {best_key} @ {best_rate:.1%}")

        results["variants"][variant] = {
            "best": best_key,
            "best_rate": best_rate,
            "configs": variant_results,
        }

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RepE RESULTS")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline_rate:.1%}")
    overall_best_rate = baseline_rate
    overall_best_name = "baseline"

    for variant, vr in results["variants"].items():
        print(f"  {variant}: best={vr['best']} @ {vr['best_rate']:.1%}")
        if vr["best_rate"] > overall_best_rate:
            overall_best_rate = vr["best_rate"]
            overall_best_name = f"{variant}/{vr['best']}"

    print(f"\nOverall best: {overall_best_name} @ {overall_best_rate:.1%}")
    if overall_best_rate <= baseline_rate:
        print("⚠ RepE zero effect — global mode shift doesn't help")

    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "outputs_phase16"
    output_dir.mkdir(exist_ok=True)

    results["summary"] = {
        "baseline_rate": baseline_rate,
        "best": overall_best_name,
        "best_rate": overall_best_rate,
        "n_train": n_train,
        "n_test": n_test,
    }

    output_path = output_dir / "phase16_repe_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
