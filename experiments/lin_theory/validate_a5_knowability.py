"""Diagnostic: check intervention effect on 'know-but-wrong' subset.

For each test sample, compute rank of y_true in model's top-K predictions.
Classify samples into:
  - know & correct: rank <= 50, generated correctly
  - know & wrong: rank <= 50, generated incorrectly  ← TARGET for intervention
  - don't know: rank > 50

Then run g-intervention on the 'know & wrong' subset specifically.
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    compute_g_L,
    extract_h_at_layer,
    greedy_generate,
)


def get_y_true_rank(logits, y_true_id):
    """Get rank of y_true in sorted logits (rank 0 = highest probability)."""
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


def generate_with_g_intervention(
    model, tokenizer, prompt, device, layer, g_unit, alpha
):
    d_f16 = g_unit.to(dtype=torch.float16)

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"
    return greedy_generate(
        model, tokenizer, prompt, device, fwd_hooks=[(hook_name, _intervene)]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[-3.0, -1.0, 1.0, 3.0, 5.0, 10.0]
    )
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(__file__).parent.parent / "outputs" / "lin_theory"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("Knowability-Stratified g-Intervention")
    print(
        f"Layer: L{args.layer}, n_test: {args.n_test}, "
        f"rank_threshold: {args.rank_threshold}"
    )
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load test samples
    print(f"\n[2/4] Loading {args.n_test} test samples...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    # Classify samples
    print(f"\n[3/4] Classifying samples by knowability...")
    classification = {"know_correct": [], "know_wrong": [], "dont_know": []}
    precomputed = {}

    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        h_L, logits, tokens, last_pos = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )

        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        # Compute g_L
        g_L = compute_g_L(h_L, y_true_id, W_U, b_U, ln_final)
        g_np = g_L.float().numpy()
        g_norm = float(np.linalg.norm(g_np))
        g_unit = g_np / (g_norm + 1e-10)
        g_unit_tensor = torch.from_numpy(g_unit).float().to(device)

        entry = {
            "sample_id": i,
            "rank": rank,
            "is_correct": is_correct,
            "g_unit": g_unit_tensor,
            "g_norm": g_norm,
            "y_true_id": y_true_id,
            "prompt": prompt,
            "tokens": tokens,
            "answers": sample["answers"],
            "question": sample["question"],
            "generated": gen_text,
        }

        if rank <= args.rank_threshold:
            if is_correct:
                classification["know_correct"].append(entry)
            else:
                classification["know_wrong"].append(entry)
        else:
            classification["dont_know"].append(entry)

    n_kc = len(classification["know_correct"])
    n_kw = len(classification["know_wrong"])
    n_dk = len(classification["dont_know"])
    n_total = n_kc + n_kw + n_dk

    print(f"\n  Classification (rank threshold ≤ {args.rank_threshold}):")
    print(f"  Know & Correct: {n_kc}/{n_total} ({n_kc / n_total:.1%})")
    print(f"  Know & Wrong:   {n_kw}/{n_total} ({n_kw / n_total:.1%})  ← TARGET")
    print(f"  Don't Know:     {n_dk}/{n_total} ({n_dk / n_total:.1%})")
    print(f"  Baseline acc (all): {(n_kc) / n_total:.1%}")

    # Run intervention on each subset
    print(f"\n[4/4] Running g-intervention on subsets...")

    all_results = {}
    for subset_name, entries in [
        ("know_wrong", classification["know_wrong"]),
        ("know_correct", classification["know_correct"]),
        ("dont_know", classification["dont_know"]),
        (
            "all",
            classification["know_correct"]
            + classification["know_wrong"]
            + classification["dont_know"],
        ),
    ]:
        if not entries:
            continue

        baseline_correct = sum(1 for e in entries if e["is_correct"])
        n = len(entries)

        subset_results = {
            "n": n,
            "baseline_correct": baseline_correct,
            "baseline_rate": baseline_correct / n,
            "alphas": {},
        }

        for alpha in args.alphas:
            correct = 0
            for e in tqdm(entries, desc=f"  {subset_name} α={alpha:+.1f}", leave=False):
                gen_text = generate_with_g_intervention(
                    model,
                    tokenizer,
                    e["prompt"],
                    device,
                    args.layer,
                    e["g_unit"],
                    alpha,
                )
                if check_correct(gen_text, e["answers"], dataset="triviaqa"):
                    correct += 1

            rate = correct / n
            delta = rate - subset_results["baseline_rate"]
            subset_results["alphas"][f"alpha={alpha:+.1f}"] = {
                "correct": correct,
                "rate": rate,
                "delta": float(delta),
            }
            print(
                f"  {subset_name:>15s} α={alpha:+5.1f}: {correct}/{n} = {rate:.1%} "
                f"(Δ={delta:+.1%})"
            )

        all_results[subset_name] = subset_results

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary: g-intervention effect by knowability")
    print(f"{'=' * 60}")
    print(f"{'Subset':>15s} {'n':>4s} {'Baseline':>10s}", end="")
    for alpha in args.alphas:
        print(f" {'α=' + str(alpha):>10s}", end="")
    print(f"\n{'-' * 15} {'-' * 4} {'-' * 10}", end="")
    for _ in args.alphas:
        print(f" {'-' * 10}", end="")
    print()

    for subset_name in ["know_wrong", "know_correct", "dont_know", "all"]:
        if subset_name not in all_results:
            continue
        r = all_results[subset_name]
        print(f"{subset_name:>15s} {r['n']:>4d} {r['baseline_rate']:>9.1%}", end="")
        for alpha in args.alphas:
            key = f"alpha={alpha:+.1f}"
            if key in r["alphas"]:
                print(f" {r['alphas'][key]['delta']:+9.1%}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    # Save
    output = {
        "config": {
            "n_test": args.n_test,
            "layer": args.layer,
            "alphas": args.alphas,
            "rank_threshold": args.rank_threshold,
        },
        "classification_counts": {
            "know_correct": n_kc,
            "know_wrong": n_kw,
            "dont_know": n_dk,
        },
        "results": all_results,
    }
    # Convert tensors to serializable
    for subset in ["know_correct", "know_wrong", "dont_know"]:
        for e in classification[subset]:
            e.pop("g_unit", None)
            e.pop("tokens", None)

    output["samples"] = classification
    out_path = output_dir / "a5_knowability_stratified.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
