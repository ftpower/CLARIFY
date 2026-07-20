"""EPR/WEPR: Token-level Entropy Production Rate as hallucination detector.

Zero-training: compute top-K truncated entropy H_K at the last prompt position.
Low H_K = sharp distribution = model is confident = likely correct.
High H_K = flat distribution = model is uncertain = likely hallucination.

Usage:
    python main_epr.py --n_samples 500
    python main_epr.py --n_samples 500 --knowledge_filter 0.3
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


def compute_epr(logits: torch.Tensor, K: int = 10) -> float:
    """Top-K truncated entropy H_K = -sum p_i * log(p_i), p_i renormalized over top-K.

    Args:
        logits: [vocab_size] float tensor at last position.
        K: number of top logits to keep.

    Returns:
        H_K: truncated entropy in nats (float).
    """
    topk_logits, _ = torch.topk(logits, K)
    topk_logprobs = F.log_softmax(topk_logits, dim=-1)
    topk_probs = torch.exp(topk_logprobs)
    topk_probs = topk_probs / topk_probs.sum()  # renormalize over top-K
    # Numerically stable: -sum(p * log(p)) using clamped log
    log_p = torch.log(topk_probs.clamp(min=1e-12))
    h_k = -(topk_probs * log_p).sum().item()
    return h_k


def compute_answer_entropy(logits: torch.Tensor, letter_ids: dict) -> float:
    """Entropy over {A,B,C,D} tokens."""
    lid = torch.tensor(
        [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits.device
    )
    letter_logits = logits[lid].float()
    probs = F.softmax(letter_logits, dim=-1)
    log_p = torch.log(probs.clamp(min=1e-12))
    return -(probs * log_p).sum().item()


def compute_p_correct(
    logits: torch.Tensor, letter_ids: dict, correct_letter: str
) -> float:
    """P(correct) = softmax over {A,B,C,D} at correct letter."""
    lid = torch.tensor(
        [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits.device
    )
    letter_logits = logits[lid].float()
    probs = F.softmax(letter_logits, dim=-1)
    idx = ["A", "B", "C", "D"].index(correct_letter.upper())
    return probs[idx].item()


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC. High score = predicted hallucination = label 0."""
    from sklearn.metrics import roc_auc_score

    # If all same label, AUROC undefined
    if len(np.unique(labels)) < 2:
        return 0.5
    # High EPR = uncertain = likely hallucination (label 0).
    # roc_auc_score(y_true=1 for positive class, y_score where higher=more positive)
    # y_true = 1 - labels  (incorrect=1=positive class)
    # y_score = scores      (higher EPR → higher uncertainty → more likely positive)
    try:
        return roc_auc_score(1 - labels, scores)
    except ValueError:
        return 0.5


def main(args):
    device = args.device

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = load_model(device=device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[0] if len(tok_ids) == 1 else tok_ids[-1]
    print(f"Letter token IDs: {letter_ids}")

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading HellaSwag ({args.n_samples} samples)...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── Run inference ──────────────────────────────────────────────────
    K_values = [5, 10, 20, 50, 100]
    results = []

    for sample in tqdm(samples, desc="Computing EPR"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])  # [1, seq, vocab]

        logits_last = logits[0, last_pos, :]  # [vocab_size]

        # EPR for various K
        epr_values = {}
        for k in K_values:
            epr_values[f"epr_K{k}"] = compute_epr(logits_last, K=k)

        # Answer-specific metrics
        ans_entropy = compute_answer_entropy(logits_last, letter_ids)
        p_correct = compute_p_correct(logits_last, letter_ids, correct_letter)

        # Check if model gets it right
        lid = torch.tensor(
            [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
        )
        pred = ["A", "B", "C", "D"][logits_last[lid].argmax().item()]
        is_correct = pred == correct_letter

        results.append(
            {
                **epr_values,
                "ans_entropy": ans_entropy,
                "p_correct": p_correct,
                "predicted": pred,
                "correct_letter": correct_letter,
                "is_correct": is_correct,
            }
        )

    # ── Evaluate AUROC ─────────────────────────────────────────────────
    labels = np.array([r["is_correct"] for r in results], dtype=int)
    p_correct_arr = np.array([r["p_correct"] for r in results])
    n_total = len(results)
    acc = labels.sum() / n_total

    print(f"\n{'=' * 60}")
    print(f"EPR Hallucination Detection — {n_total} samples, acc={acc:.4f}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<18} {'AUROC':>8}  Notes")
    print(f"{'-' * 18} {'-' * 8}  {'-' * 30}")

    header_printed = False
    for metric_name in [f"epr_K{k}" for k in K_values] + ["ans_entropy"]:
        scores = np.array([r[metric_name] for r in results])
        auroc = compute_auroc(scores, labels)
        print(f"{metric_name:<18} {auroc:>8.4f}")

    # baseline: P(correct) AUROC
    auroc_pc = compute_auroc(
        -p_correct_arr, labels
    )  # negative because high p_c → correct
    print(f"{'P(correct) (ref)':<18} {auroc_pc:>8.4f}  baseline")

    # ── Knowledge-filtered evaluation ──────────────────────────────────
    if args.knowledge_filter > 0:
        thr = args.knowledge_filter
        filtered = [r for r in results if r["p_correct"] > thr]
        if len(filtered) >= 20:
            filt_labels = np.array([r["is_correct"] for r in filtered], dtype=int)
            filt_acc = filt_labels.sum() / len(filtered)
            print(f"\n{'=' * 60}")
            print(
                f"Knowledge-filtered (P(correct) > {thr}): "
                f"{len(filtered)}/{n_total} samples, acc={filt_acc:.4f}"
            )
            print(f"{'=' * 60}")
            print(f"{'Metric':<18} {'AUROC':>8}")
            print(f"{'-' * 18} {'-' * 8}")

            for metric_name in [f"epr_K{k}" for k in K_values] + ["ans_entropy"]:
                scores = np.array([r[metric_name] for r in filtered])
                auroc = compute_auroc(scores, filt_labels)
                marker = " ★" if auroc > 0.72 else ""
                print(f"{metric_name:<18} {auroc:>8.4f}{marker}")

            auroc_pc_filt = compute_auroc(
                -np.array([r["p_correct"] for r in filtered]), filt_labels
            )
            print(f"{'P(correct) (ref)':<18} {auroc_pc_filt:>8.4f}  baseline")
        else:
            print(
                f"\nKnowledge filter threshold {thr} leaves only {len(filtered)} samples — skipping."
            )

    # ── Save results ───────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "epr_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "n_total": n_total,
                "accuracy": float(acc),
                "per_sample": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {output_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knowledge_filter", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
