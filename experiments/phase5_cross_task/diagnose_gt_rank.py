"""Diagnostic: where does the ground-truth answer token rank in the logit distribution?

Core question: Does the model "know" the right answer (high GT rank) but fail to
produce it (greedy decode picks wrong), OR does the model genuinely not know
(low GT rank)?

One forward pass per sample (no generation needed):
  1. Tokenize prompt
  2. Forward → logits at last position
  3. Tokenize first token of each GT answer alias
  4. For each alias, find rank in predicted distribution (full vocab ~152K)
  5. Best rank across aliases is the per-sample signal

If GT rank is systematically low for BOTH correct and incorrect samples, then the
full-vocab logit lens has no signal to extract — regardless of feature engineering.

Usage:
    python diagnose_gt_rank.py --n_samples 200
    python diagnose_gt_rank.py --n_samples 50  # quick check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent
sys.path.insert(0, str(_sys_parent.parent / "phase2_entropy"))

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct
from phase5_utils.generation_features import generate_with_per_token_features


# ═══════════════════════════════════════════════════════════════════════════════
# Core diagnostic: GT token rank at first prediction position
# ═══════════════════════════════════════════════════════════════════════════════

def get_first_answer_token_ids(tokenizer, answers: list[str]) -> torch.Tensor:
    """Tokenize the first token of each answer alias. Deduplicates."""
    ids = set()
    for ans in answers:
        tokens = tokenizer.encode(ans.lower().strip(), add_special_tokens=False)
        if tokens:
            ids.add(tokens[0])
    return torch.tensor(sorted(ids), dtype=torch.long)


def get_full_answer_token_ids(tokenizer, answer: str) -> list[int]:
    """Tokenize full answer (for exact-match check)."""
    return tokenizer.encode(answer.lower().strip(), add_special_tokens=False)


def diagnose_gt_rank(model, tokenizer, device: str,
                     samples: list[dict], max_prompt_len: int = 2048,
                     verbose: bool = True):
    """For each sample, check rank of GT first token in predicted distribution.

    Strategy: one forward pass on the PROMPT ONLY. The logits at the last prompt
    position are the model's prediction for the first generated token.

    Truncates prompts > max_prompt_len tokens to avoid OOM on long TriviaQA contexts.
    """
    results = []
    n_truncated = 0

    for i, s in enumerate(tqdm(samples, desc="GT rank diagnostic")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Truncate: some TriviaQA search contexts are 800K+ chars → OOM on 8GB
        if len(token_ids) > max_prompt_len:
            token_ids = token_ids[:max_prompt_len]
            n_truncated += 1

        prompt_tokens = torch.tensor([token_ids], dtype=torch.long,
                                     device=device)
        prompt_len = prompt_tokens.shape[1]

        # One forward pass
        with torch.no_grad():
            logits = model(prompt_tokens)

        # Logits at last prompt position → next-token distribution
        next_logits = logits[0, -1, :].float()  # [vocab_size]
        next_probs = torch.softmax(next_logits, dim=-1)

        # Get first-token IDs for all answer aliases
        gt_ids = get_first_answer_token_ids(tokenizer, s["answers"])
        gt_ids = gt_ids.to(device)
        if len(gt_ids) == 0:
            results.append({"sample_id": i, "gt_rank": None, "gt_prob": None,
                            "is_correct": False})
            continue

        # Best rank across aliases (lowest rank = highest probability)
        gt_probs = next_probs[gt_ids]
        best_idx = int(gt_probs.argmax().item())
        best_id = int(gt_ids[best_idx].item())
        best_prob = float(gt_probs.max().item())

        # Rank: sort all probs descending, find position of best_id
        # topk is much faster than argsort on 152K
        rank = int((next_probs > best_prob).sum().item())  # 0-indexed, 0 = top-1

        # Top token for reference
        top1_id = int(next_probs.argmax().item())
        top1_text = tokenizer.decode([top1_id])

        # Also check: if we greedily decode the first token, what do we get?
        generated_first_token = tokenizer.decode([top1_id])

        results.append({
            "sample_id": i,
            "question": s["question"],
            "answers": s["answers"],
            "gt_best_token": tokenizer.decode([best_id]),
            "gt_best_prob": best_prob,
            "gt_rank": rank,
            "top1_predicted": top1_text,
            "prompt_len": prompt_len,
        })

    if n_truncated > 0:
        print(f"  Truncated {n_truncated}/{len(samples)} prompts to {max_prompt_len} tokens")
    return results


def analyze_ranks(results: list[dict], samples: list[dict]):
    """Analyze: compare GT ranks for correct vs incorrect samples."""
    # Determine correctness using the same check_correct logic
    from src.data_loader import check_correct

    # Rerun generation to get correctness labels
    # Actually we can use the existing feature results if available
    # Or we can just use the results from a prior run
    # For standalone, we'll use the generated text from the prior 8B run

    # But for standalone diagnostic, let's use gpt-generated text path
    # For now just compute correctness from the model's greedy first token
    # This is an approximation

    correct_ranks = []
    incorrect_ranks = []
    all_ranks = []
    correct_probs = []
    incorrect_probs = []

    for r in results:
        if r["gt_rank"] is None:
            continue
        rank = r["gt_rank"]
        prob = r["gt_best_prob"]
        all_ranks.append(rank)

        # Check if top1 matches any answer (rough correctness proxy)
        top1 = r["top1_predicted"].strip().lower()
        answers = r["answers"]
        is_top1_correct = False
        for ans in answers:
            ans_lower = ans.lower().strip()
            if ans_lower in top1 or top1 in ans_lower:
                is_top1_correct = True
                break

        if is_top1_correct:
            correct_ranks.append(rank)
            correct_probs.append(prob)
        else:
            incorrect_ranks.append(rank)
            incorrect_probs.append(prob)

    return {
        "n_total": len(all_ranks),
        "n_correct_first_token": len(correct_ranks),
        "n_incorrect_first_token": len(incorrect_ranks),
        "correct": {
            "mean_rank": float(np.mean(correct_ranks)) if correct_ranks else None,
            "median_rank": float(np.median(correct_ranks)) if correct_ranks else None,
            "mean_prob": float(np.mean(correct_probs)) if correct_probs else None,
            "top1_frac": float(np.mean(np.array(correct_ranks) == 0)) if correct_ranks else None,
            "top10_frac": float(np.mean(np.array(correct_ranks) < 10)) if correct_ranks else None,
            "top100_frac": float(np.mean(np.array(correct_ranks) < 100)) if correct_ranks else None,
            "top1000_frac": float(np.mean(np.array(correct_ranks) < 1000)) if correct_ranks else None,
            "n": len(correct_ranks),
        },
        "incorrect": {
            "mean_rank": float(np.mean(incorrect_ranks)) if incorrect_ranks else None,
            "median_rank": float(np.median(incorrect_ranks)) if incorrect_ranks else None,
            "mean_prob": float(np.mean(incorrect_probs)) if incorrect_probs else None,
            "top1_frac": float(np.mean(np.array(incorrect_ranks) == 0)) if incorrect_ranks else None,
            "top10_frac": float(np.mean(np.array(incorrect_ranks) < 10)) if incorrect_ranks else None,
            "top100_frac": float(np.mean(np.array(incorrect_ranks) < 100)) if incorrect_ranks else None,
            "top1000_frac": float(np.mean(np.array(incorrect_ranks) < 1000)) if incorrect_ranks else None,
            "n": len(incorrect_ranks),
        },
        "all": {
            "mean_rank": float(np.mean(all_ranks)),
            "median_rank": float(np.median(all_ranks)),
        },
    }


def compute_auroc_from_rank(results: list[dict]) -> float:
    """Use rank as a detection score: lower rank → more confident → correct."""
    ranks = np.array([r["gt_rank"] if r["gt_rank"] is not None else 152064
                      for r in results], dtype=np.float64)
    scores = -ranks  # negate: higher score = lower rank = more correct
    labels = np.array([r.get("is_correct", False) for r in results], dtype=np.int32)
    mask = ~np.isnan(scores)
    if mask.sum() < 2 or labels[mask].std() == 0:
        return float("nan")
    return float(roc_auc_score(labels[mask], scores[mask]))


# ═══════════════════════════════════════════════════════════════════════════════
# Second diagnostic: Per-position GT rank during actual generation
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_per_position_ranks(model, tokenizer, device: str,
                                samples: list[dict], max_new: int = 5):
    """For each generate step, check where GT tokens rank in the distribution.

    Unlike the prompt-only diagnostic (which only checks the FIRST generated token),
    this runs greedy generation and at each step records the GT answer rank.
    """
    results = []

    for i, s in enumerate(tqdm(samples[:50], desc="Per-position rank (50 samples)")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long,
                                 device=device)
        prompt_len = input_ids.shape[1]

        # Get full answer tokenization for all aliases
        # We need to know what tokens the model SHOULD output
        answer_token_seqs = []
        for ans in s["answers"]:
            toks = tokenizer.encode(ans.lower().strip(), add_special_tokens=False)
            if toks:
                answer_token_seqs.append(toks)

        per_step = []
        current_ids = input_ids.clone()

        for step in range(max_new):
            with torch.no_grad():
                logits = model(current_ids)

            next_logits = logits[0, -1, :].float()
            next_probs = torch.softmax(next_logits, dim=-1)

            # Greedy next token
            next_id = int(next_logits.argmax().item())

            # Check: is this the correct next token for any answer sequence?
            is_correct_step = any(
                len(seq) > step and seq[step] == next_id
                for seq in answer_token_seqs
            )

            # Rank of the correct next token (for sequences that have a step-th token)
            correct_token_ranks = []
            for seq in answer_token_seqs:
                if len(seq) > step:
                    target_id = seq[step]
                    target_prob = float(next_probs[target_id].item())
                    rank = int((next_probs > target_prob).sum().item())
                    correct_token_ranks.append(rank)

            per_step.append({
                "step": step,
                "generated_id": next_id,
                "generated_text": tokenizer.decode([next_id]),
                "is_correct_step": is_correct_step,
                "correct_token_ranks": correct_token_ranks[:3],  # top 3 aliases
            })

            # Append and continue
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_id]], device=current_ids.device)
            ], dim=1)

        results.append({
            "sample_id": i,
            "question": s["question"][:80],
            "answers": s["answers"][:3],
            "prompt_len": prompt_len,
            "per_step": per_step,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GT token rank diagnostic")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs_8b")
    parser.add_argument("--per_position", action="store_true",
                        help="Also run per-position rank scan (50 samples)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GT Token Rank Diagnostic")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Load data
    print(f"Loading TriviaQA ({args.n_samples} samples)...")
    samples = load_triviaqa(n_samples=args.n_samples)

    # ── Diagnostic 1: Prompt-only GT rank ──
    print(f"\n{'─'*60}")
    print("Diagnostic 1: GT first-token rank (prompt-only, no generation)")
    print(f"{'─'*60}")
    t0 = time.time()
    results = diagnose_gt_rank(model, tokenizer, device, samples)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")

    # Analysis
    analysis = analyze_ranks(results, samples)
    print(f"\n  First-token correctness: {analysis['n_correct_first_token']}/{analysis['n_total']} "
          f"({analysis['n_correct_first_token']/analysis['n_total']:.1%})")

    print(f"\n  GT Token Rank Distribution:")
    print(f"  {'':25s} {'Correct':>10s} {'Incorrect':>10s}")
    print(f"  {'─'*45}")
    print(f"  {'N':25s} {analysis['correct']['n']:>10d} {analysis['incorrect']['n']:>10d}")
    print(f"  {'Mean rank (0=top-1)':25s} {analysis['correct']['mean_rank']:>10.1f} "
          f"{analysis['incorrect']['mean_rank']:>10.1f}" if analysis['correct']['mean_rank'] else "  N/A")
    print(f"  {'Median rank':25s} {analysis['correct']['median_rank']:>10.1f} "
          f"{analysis['incorrect']['median_rank']:>10.1f}" if analysis['correct']['median_rank'] else "  N/A")
    print(f"  {'Mean prob':25s} {analysis['correct']['mean_prob']:>10.6f} "
          f"{analysis['incorrect']['mean_prob']:>10.6f}" if analysis['correct']['mean_prob'] else "  N/A")
    print(f"  {'Fraction rank=0 (top-1)':25s} {analysis['correct']['top1_frac']:>10.3f} "
          f"{analysis['incorrect']['top1_frac']:>10.3f}" if analysis['correct']['top1_frac'] is not None else "  N/A")
    print(f"  {'Fraction rank<10':25s} {analysis['correct']['top10_frac']:>10.3f} "
          f"{analysis['incorrect']['top10_frac']:>10.3f}" if analysis['correct']['top10_frac'] is not None else "  N/A")
    print(f"  {'Fraction rank<100':25s} {analysis['correct']['top100_frac']:>10.3f} "
          f"{analysis['incorrect']['top100_frac']:>10.3f}" if analysis['correct']['top100_frac'] is not None else "  N/A")
    print(f"  {'Fraction rank<1000':25s} {analysis['correct']['top1000_frac']:>10.3f} "
          f"{analysis['incorrect']['top1000_frac']:>10.3f}" if analysis['correct']['top1000_frac'] is not None else "  N/A")

    # Sanity: show a few examples
    print(f"\n  Example predictions:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"    Q: {r['question'][:60]}...")
        print(f"      GT: {r['answers'][:2]} → best_token='{r['gt_best_token']}' rank={r['gt_rank']} prob={r['gt_best_prob']:.6f}")
        print(f"      Predicted top1: '{r['top1_predicted']}'")

    # Save
    diagnostic1_path = output_dir / "diagnostic_gt_rank.json"
    with open(diagnostic1_path, "w") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, default=str)
    print(f"\n  Saved: {diagnostic1_path}")

    # ── Diagnostic 2: Per-position ranks (optional) ──
    if args.per_position:
        print(f"\n{'─'*60}")
        print("Diagnostic 2: Per-position GT rank during greedy generation")
        print(f"{'─'*60}")
        pp_results = diagnose_per_position_ranks(model, tokenizer, device, samples)

        # Analyze: at step 0, what % have correct token in top-5?
        step0_correct_in_top5 = 0
        step0_total = 0
        for r in pp_results:
            if r["per_step"]:
                s0 = r["per_step"][0]
                if s0["correct_token_ranks"]:
                    if min(s0["correct_token_ranks"]) < 5:
                        step0_correct_in_top5 += 1
                    step0_total += 1
        print(f"\n  Step 0: GT token in top-5: {step0_correct_in_top5}/{step0_total} "
              f"({step0_correct_in_top5/max(step0_total,1):.1%})" if step0_total > 0 else "  N/A")

        diagnostic2_path = output_dir / "diagnostic_per_position.json"
        with open(diagnostic2_path, "w") as f:
            json.dump(pp_results, f, indent=2, default=str)
        print(f"  Saved: {diagnostic2_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}\n")

    # Key question: is there a separation between correct and incorrect?
    if analysis['correct']['mean_rank'] and analysis['incorrect']['mean_rank']:
        gap = analysis['incorrect']['mean_rank'] - analysis['correct']['mean_rank']
        print(f"  Correct first-token mean rank:  {analysis['correct']['mean_rank']:.0f}")
        print(f"  Incorrect first-token mean rank: {analysis['incorrect']['mean_rank']:.0f}")
        print(f"  Gap: {gap:.0f}")
        print()
        if gap > 1000:
            print("  ✅ GOOD separation — GT rank carries strong signal")
            print("     → Problem is aggregation method, not signal absence")
        elif gap > 100:
            print("  ⚠️ MODERATE separation — some signal, but noisy")
            print("     → Semantic clustering or cross-sample consistency may help")
        else:
            print("  ❌ POOR separation — model doesn't distinguish correct from incorrect")
            print("     → Full-vocab logit lens has NO signal at this model scale")
            print("     → Consider: cross-sample methods (SelfCheckGPT) or hidden-state probes")


if __name__ == "__main__":
    main()
