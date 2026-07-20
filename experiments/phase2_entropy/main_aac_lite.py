"""AAC-lite: Hallucination direction projection suppression.

Zero-training: compute "hallucination direction" = normalized difference between
mean hidden states of incorrect vs correct samples. At inference, project out
the hallucination component from hidden states at a chosen layer.

Two modes:
- subtract: h -= lambda * proj_halluc(h)  (suppress hallucination)
- add:     h += lambda * proj_halluc(h)  (amplify, sanity check)

Usage:
    python main_aac_lite.py --n_dir 300 --n_eval 500
    python main_aac_lite.py --n_dir 300 --n_eval 500 --layers 11,15,20 --lam 0.1,0.5,1.0
"""

import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ── Direction computation ──────────────────────────────────────────────


def _make_save_hook(storage: dict, key: str):
    """Hook that saves last-position hidden state."""

    def hook(activation, hook=None):
        storage[key] = activation[0, -1, :].detach()
        return activation

    return hook


def compute_directions(
    model,
    samples: list[dict],
    candidate_layers: list[int],
    letter_ids: dict[str, int],
) -> dict[int, torch.Tensor]:
    """Compute hallucination direction per layer = normalized(mean_incorrect - mean_correct).

    Returns:
        {layer: direction [d_model]} unit-norm directions.
    """
    # Collect hidden states by correctness
    accum_correct = {L: [] for L in candidate_layers}
    accum_incorrect = {L: [] for L in candidate_layers}
    n_correct = 0
    n_incorrect = 0

    # Build hooks
    storage = {}
    hooks = []
    for L in candidate_layers:
        key = f"blocks.{L}.hook_resid_post"
        hooks.append((key, _make_save_hook(storage, key)))

    for sample in tqdm(samples, desc="Computing directions"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Check correctness
        logits_last = logits[0, -1, :]
        lid = torch.tensor(
            [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
        )
        pred_idx = logits_last[lid].argmax().item()
        is_correct = ["A", "B", "C", "D"][pred_idx] == correct_letter

        for L in candidate_layers:
            key = f"blocks.{L}.hook_resid_post"
            h = storage[key]  # [d_model]
            if is_correct:
                accum_correct[L].append(h)
            else:
                accum_incorrect[L].append(h)

        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1

    print(f"Direction samples: {n_correct} correct, {n_incorrect} incorrect")

    # Compute mean difference per layer
    directions = {}
    for L in candidate_layers:
        h_corr = torch.stack(accum_correct[L]).mean(dim=0)  # [d_model]
        h_incorr = torch.stack(accum_incorrect[L]).mean(dim=0)  # [d_model]
        diff = h_incorr - h_corr
        direction = diff / (diff.norm() + 1e-8)
        directions[L] = direction

        # Report cosine similarity between layers (should differ if layers matter)
        if L != candidate_layers[0]:
            cos = (direction * list(directions.values())[0]).sum().item()
            print(f"  L{L}: cos with L{candidate_layers[0]} = {cos:.4f}")

        # Magnitude of difference
        print(f"  L{L}: ||diff|| = {diff.norm().item():.4f}")

    return directions


# ── Projection hook ───────────────────────────────────────────────────


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    """Return a hook that projects hidden states onto/against the hallucination direction.

    Args:
        direction: [d_model] unit-norm hallucination direction.
        lam: intervention strength (0 = no effect).
        mode: "subtract" to suppress hallucination, "add" to amplify.
    """
    sign = -1.0 if mode == "subtract" else 1.0
    d = direction  # [d_model]

    def hook(activation, hook=None):
        # activation: [batch, seq, d_model]
        # proj_magnitude: [batch, seq] = activation @ d
        # projection: [batch, seq, d_model] = proj_magnitude * d
        proj_mag = activation @ d  # [batch, seq]
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)  # [b, s, d]
        return activation + sign * lam * projection

    return hook


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_aac(
    model,
    eval_samples: list[dict],
    directions: dict[int, torch.Tensor],
    letter_ids: dict[str, int],
    layers: list[int],
    lams: list[float],
    modes: list[str],
) -> dict:
    """Evaluate AAC-lite across layer x lambda x mode grid (sample-major for efficiency)."""
    # Pre-tokenize all samples
    print("Pre-tokenizing eval samples...")
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append(
            {
                "tokens": tokens,
                "correct_letter": correct_letter,
            }
        )

    # Initialize result accumulators
    configs = [(L, lam, mode) for L in layers for lam in lams for mode in modes]
    accum = {
        (L, lam, mode): {"n_correct": 0, "per_sample": []} for L, lam, mode in configs
    }

    # For each sample, run all configs
    lid_tensor = torch.tensor(
        [letter_ids[l] for l in ["A", "B", "C", "D"]]
    )  # will move to device

    for idx, item in enumerate(tqdm(tokenized, desc="Evaluating", leave=True)):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)

        for L, lam, mode in configs:
            direction = directions[L]
            hook_point = f"blocks.{L}.hook_resid_post"
            hook_fn = make_projection_hook(direction, lam, mode)

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_point, hook_fn)],
                )

            logits_last = logits[0, -1, :]
            pred_idx = logits_last[lid].argmax().item()
            pred = ["A", "B", "C", "D"][pred_idx]
            is_correct = pred == correct_letter

            probs = F.softmax(logits_last[lid].float(), dim=-1)
            p_correct = probs[["A", "B", "C", "D"].index(correct_letter)].item()

            accum[(L, lam, mode)]["n_correct"] += int(is_correct)
            accum[(L, lam, mode)]["per_sample"].append(
                {
                    "pred": pred,
                    "correct": correct_letter,
                    "is_correct": is_correct,
                    "p_correct": p_correct,
                }
            )

    # Build results list
    n_total = len(tokenized)
    results = []
    for L, lam, mode in configs:
        a = accum[(L, lam, mode)]
        acc = a["n_correct"] / n_total

        filtered = [s for s in a["per_sample"] if s["p_correct"] > 0.3]
        n_filt = len(filtered)
        acc_filt = (
            sum(s["is_correct"] for s in filtered) / n_filt if n_filt >= 20 else None
        )

        results.append(
            {
                "layer": L,
                "lambda": lam,
                "mode": mode,
                "accuracy": float(acc),
                "n_total": n_total,
                "n_correct": a["n_correct"],
                "accuracy_filtered": float(acc_filt) if n_filt >= 20 else None,
                "n_filtered": n_filt,
            }
        )

    return results


# ── Main ───────────────────────────────────────────────────────────────


def main(args):
    device = args.device

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = load_model(device=device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[0] if len(tok_ids) == 1 else tok_ids[-1]
    print(f"Letter token IDs: {letter_ids}")

    candidate_layers = args.layers

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_file = output_dir / "aac_lite_directions.pt"

    if args.eval_only:
        # ── Load pre-computed directions ─────────────────────────
        print(f"\nLoading directions from {dir_file}")
        ckpt = torch.load(dir_file, map_location=device)
        directions = {int(L): d.to(device) for L, d in ckpt.items()}
        print(f"Loaded directions for layers: {list(directions.keys())}")
    else:
        # ── Phase 1: Compute hallucination directions ────────────
        print(f"\n{'=' * 60}")
        print("Phase 1: Computing hallucination directions")
        print(f"{'=' * 60}")

        print("Loading HellaSwag train split...")
        ds_train = load_dataset(
            "Rowan/hellaswag", split="train", trust_remote_code=False
        )
        ds_train = ds_train.shuffle(seed=args.seed)
        label_letters = ["A", "B", "C", "D"]
        dir_samples = []
        for item in ds_train.select(range(min(args.n_dir, len(ds_train)))):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            correct_ending = endings[label]
            label_letter = label_letters[label]
            choices_text = "\n".join(
                f"{label_letters[i]}. {endings[i]}" for i in range(4)
            )
            dir_samples.append(
                {
                    "question": ctx,
                    "answers": [correct_ending, label_letter],
                    "context": choices_text,
                }
            )

        directions = compute_directions(
            model, dir_samples, candidate_layers, letter_ids
        )

        torch.save({str(L): d.cpu() for L, d in directions.items()}, dir_file)
        print(f"Saved directions to {dir_file}")

    # ── Phase 2: Evaluate ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 2: Evaluating AAC-lite intervention")
    print(f"{'=' * 60}")

    print("Loading HellaSwag validation for evaluation...")
    eval_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

    # Compute baseline accuracy (full + filtered)
    print("Computing baseline...")
    n_base_correct = 0
    base_per_sample = []
    for sample in tqdm(eval_samples, desc="Baseline"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        lid = torch.tensor(
            [letter_ids[l] for l in ["A", "B", "C", "D"]],
            device=logits_last.device,
        )
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        pred_idx = probs.argmax().item()
        is_correct = ["A", "B", "C", "D"][pred_idx] == correct_letter
        p_correct = probs[["A", "B", "C", "D"].index(correct_letter)].item()
        n_base_correct += int(is_correct)
        base_per_sample.append({"is_correct": is_correct, "p_correct": p_correct})

    baseline_acc = n_base_correct / len(eval_samples)
    base_filtered = [s for s in base_per_sample if s["p_correct"] > 0.3]
    n_base_filt = len(base_filtered)
    baseline_filt_acc = (
        sum(s["is_correct"] for s in base_filtered) / n_base_filt
        if n_base_filt >= 20
        else None
    )
    print(
        f"Baseline: full={baseline_acc:.4f} ({n_base_correct}/{len(eval_samples)}), "
        f"filtered(P>0.3)={baseline_filt_acc:.4f} (n={n_base_filt})"
        if baseline_filt_acc is not None
        else f"Baseline: full={baseline_acc:.4f} ({n_base_correct}/{len(eval_samples)})"
    )

    # Run evaluation sweep
    lams = args.lam
    modes = args.modes.split(",")

    eval_results = evaluate_aac(
        model=model,
        eval_samples=eval_samples,
        directions=directions,
        letter_ids=letter_ids,
        layers=candidate_layers,
        lams=lams,
        modes=modes,
    )

    # ── Report ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"AAC-lite Results — baseline={baseline_acc:.4f}")
    print(f"{'=' * 60}")
    print(
        f"{'Layer':<6} {'λ':<8} {'Mode':<10} {'Acc':>8} {'Δ':>8} {'FiltAcc':>8} {'Δf':>8}"
    )
    print(f"{'-' * 6} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    best = None
    best_delta = -999

    for r in eval_results:
        delta = r["accuracy"] - baseline_acc
        delta_f = (
            r["accuracy_filtered"] - baseline_filt_acc
            if (r["accuracy_filtered"] is not None and baseline_filt_acc is not None)
            else None
        )
        filt_str = (
            f"{r['accuracy_filtered']:.4f}"
            if r["accuracy_filtered"] is not None
            else "N/A"
        )
        df_str = f"{delta_f:+.4f}" if delta_f is not None else "N/A"
        print(
            f"{r['layer']:<6} {r['lambda']:<8} {r['mode']:<10} "
            f"{r['accuracy']:.4f} {delta:+.4f} {filt_str} {df_str}"
        )

        if delta > best_delta:
            best_delta = delta
            best = r

    print(
        f"\nBest: L{best['layer']} λ={best['lambda']} mode={best['mode']} "
        f"acc={best['accuracy']:.4f} Δ={best_delta:+.4f}"
    )

    # ── Save ────────────────────────────────────────────────────────
    results_file = output_dir / "aac_lite_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "baseline_acc": float(baseline_acc),
                "directions_computed_from": f"train_{args.n_dir}_samples",
                "eval_on": f"val_{args.n_eval}_samples",
                "best": best,
                "sweep": eval_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {results_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_dir",
        type=int,
        default=300,
        help="Samples for direction computation (train split)",
    )
    parser.add_argument(
        "--n_eval", type=int, default=500, help="Samples for evaluation (val split)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 7, 11, 15, 20])
    parser.add_argument(
        "--lam", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="subtract,add",
        help="Comma-separated: subtract,add",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip direction computation, load from disk",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
