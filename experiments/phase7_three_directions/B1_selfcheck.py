"""B1: SelfCheck Consistency — cross-sample generation consistency as confidence.

Hypothesis: Don't look at logits. Generate K=5 answers with temperature, measure
pairwise consistency. When the model knows: outputs converge. When hallucinating:
outputs diverge.

Three consistency metrics:
  - answer_agreement: fraction of pairs with exactly matching answer spans
  - rouge_l: mean pairwise ROUGE-L (longest common subsequence)
  - composite: agreement * 0.5 + rouge_l * 0.5

Inspired by: SelfCheckGPT (Manakul 2023) — surface-text consistency of
multiple sampled responses.

Usage:
    python B1_selfcheck.py --n_samples 100 --K 5
    python B1_selfcheck.py --n_samples 50 --K 3  # quick test
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
# Answer extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_answer_span(text: str) -> str:
    """Extract the actual answer from generated text.

    TriviaQA prompt ends with "Answer:", so take text after that.
    Strip punctuation and lowercase for comparison.
    """
    text = text.strip()
    # Take first non-empty line or sentence
    # Split on common delimiters
    for delim in ["\n", ". ", ", ", "; "]:
        if delim in text:
            text = text.split(delim)[0]
            break
    # Clean
    text = re.sub(r"[^\w\s]", "", text.lower().strip())
    # Remove common prefixes
    for prefix in ["the ", "a ", "an "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L (simple, no external dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _lcs_len(a: list, b: list) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 score between two strings."""
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return 0.0
    lcs = _lcs_len(a_tokens, b_tokens)
    prec = lcs / len(a_tokens) if a_tokens else 0
    rec = lcs / len(b_tokens) if b_tokens else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: multi-sample generation + consistency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_selfcheck_consistency(
    model, tokenizer, prompt: str, device: str,
    K: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 20,
) -> dict:
    """Generate K answers with temperature, compute pairwise consistency.

    Returns:
        dict with keys:
          - answers: list[str] — K generated answers
          - agreement: float — fraction of pairs that match exactly
          - rouge_l: float — mean pairwise ROUGE-L
          - composite: float — 0.5*agreement + 0.5*rouge_l
    """
    answers = []
    answer_spans = []

    for k in range(K):
        # Temperature-sampled generation
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        generated_ids = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(tokens)  # Uses model's default temperature
            # Apply temperature
            logits_t = logits[0, -1, :] / temperature
            probs = torch.softmax(logits_t, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], device=device)], dim=1
            )

        full_text = tokenizer.decode(generated_ids).strip()
        answers.append(full_text)
        answer_spans.append(extract_answer_span(full_text))

    # Pairwise agreement
    n_pairs = K * (K - 1) // 2
    agreements = 0
    rouge_scores = []
    for i in range(K):
        for j in range(i + 1, K):
            if answer_spans[i] == answer_spans[j] and len(answer_spans[i]) > 0:
                agreements += 1
            rouge_scores.append(rouge_l(answers[i], answers[j]))

    agreement = agreements / n_pairs if n_pairs > 0 else 0.0
    rouge_l_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    composite = 0.5 * agreement + 0.5 * rouge_l_mean

    return {
        "answers": answers,
        "answer_spans": answer_spans,
        "agreement": agreement,
        "rouge_l": rouge_l_mean,
        "composite": composite,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="B1: SelfCheck Consistency")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of temperature samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"B1: SelfCheck Consistency (K={args.K}, temp={args.temperature})")
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

    print(f"\nGenerating {args.K}× answers per sample...")
    t0 = time.time()
    results = []
    correct_count = 0
    total_generations = 0

    for s in tqdm(samples, desc="B1 selfcheck"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")

        consistency = compute_selfcheck_consistency(
            model, tokenizer, prompt, device,
            K=args.K,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        # Use the first (greedy-like but sampled) answer for correctness check
        # Actually, use majority: check if ANY generated answer is correct
        any_correct = any(
            check_correct(ans, s["answers"], dataset="triviaqa")
            for ans in consistency["answers"]
        )
        if any_correct:
            correct_count += 1
        total_generations += args.K

        results.append({
            "sample_id": len(results),
            "question": s["question"][:80],
            "answers": s["answers"][:3],
            "generated_answers": consistency["answers"],
            "answer_spans": consistency["answer_spans"],
            "any_correct": any_correct,
            "agreement": consistency["agreement"],
            "rouge_l": consistency["rouge_l"],
            "composite": consistency["composite"],
        })

    elapsed = time.time() - t0
    print(f"  Any-correct: {correct_count}/{len(samples)} ({correct_count/len(samples):.1%})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_generations:.2f}s/gen)")

    # AUROC: "any_correct" is a more generous label than greedy-only correct
    # Use it as the ground truth for self-consistency evaluation
    labels = [1 if r["any_correct"] else 0 for r in results]

    feature_configs = [
        {"key": "agreement", "name": "answer_agreement", "invert": False},
        {"key": "rouge_l", "name": "rouge_l_consistency", "invert": False},
        {"key": "composite", "name": "composite_score", "invert": False},
    ]

    print(f"\nAUROC Results (higher consistency = any-correct):")
    auroc_summary = evaluate_all(results, labels, feature_configs)

    # Also test with strict correctness (answer at rank 1 is correct)
    from src.data_loader import check_correct as cc
    labels_strict = [
        1 if cc(r["generated_answers"][0], samples[i]["answers"], dataset="triviaqa")
        else 0
        for i, r in enumerate(results)
    ]
    print(f"\nAUROC Results (higher consistency = first-answer-correct):")
    auroc_summary_strict = evaluate_all(results, labels_strict, feature_configs)

    print_summary(auroc_summary)

    save_results(
        results, auroc_summary,
        output_path=str(output_dir / "B1_selfcheck.json"),
        extra={
            "any_correct_rate": correct_count / len(samples),
            "K": args.K, "temperature": args.temperature,
            "auroc_strict": auroc_summary_strict,
        },
    )

    print(f"{'='*60}")
    print(f"B1 complete — consistency-based, no logit lens needed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
