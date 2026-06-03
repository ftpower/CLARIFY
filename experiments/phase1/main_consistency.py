"""Consistency-based hallucination detection: sample N times, use answer agreement as confidence.

Usage:
    python main_consistency.py --n_samples 200 --dataset hellaswag --n_gen 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_triviaqa,
    load_squad,
    load_hellaswag,
    format_prompt,
    check_correct,
)
from src.model_utils import load_model, generate_token


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_consistency",
    dataset: str = "hellaswag",
    n_gen: int = 5,
    temperature: float = 0.7,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {dataset.upper()} ({n_samples} samples)...")
    if dataset == "squad":
        samples = load_squad(n_samples=n_samples)
    elif dataset == "hellaswag":
        samples = load_hellaswag(n_samples=n_samples)
    else:
        samples = load_triviaqa(n_samples=n_samples)

    # --- Load model ---
    print("Loading Qwen3-1.7B-Instruct...")
    model = load_model(device=device)

    # --- Generate N times per sample ---
    print(f"Phase: Generating {n_gen} samples per question (T={temperature})...")

    records = []
    correct_count = 0

    for sample in tqdm(samples, desc="Samples"):
        question = sample["question"]
        answers = sample["answers"]
        context = sample["context"]
        prompt = format_prompt(question, context, dataset=dataset)

        # Generate N tokens
        gen_results = []
        for _ in range(n_gen):
            token_id, token_text = generate_token(
                model, prompt, temperature=temperature
            )
            gen_results.append((token_id, token_text.strip()))

        # Majority vote
        token_ids = [g[0] for g in gen_results]
        id_counts = Counter(token_ids)
        most_common_id, majority_count = id_counts.most_common(1)[0]
        consistency = majority_count / n_gen

        # Majority answer text
        majority_text = next(g[1] for g in gen_results if g[0] == most_common_id)

        # Check correctness against ground truth
        is_correct = check_correct(majority_text, answers, dataset=dataset)
        if is_correct:
            correct_count += 1

        records.append(
            {
                "consistency": consistency,
                "is_correct": is_correct,
                "majority_answer": majority_text,
                "majority_count": majority_count,
                "n_gen": n_gen,
                "all_answers": [g[1] for g in gen_results],
                "all_ids": token_ids,
            }
        )

    accuracy = correct_count / n_samples
    print(f"\nAccuracy (majority vote): {correct_count}/{n_samples} = {accuracy:.1%}")

    # --- Distribution breakdown ---
    consistency_dist = Counter(r["consistency"] for r in records)
    print("\nConsistency score distribution:")
    for score in sorted(consistency_dist):
        n = consistency_dist[score]
        correct = sum(
            1 for r in records if r["consistency"] == score and r["is_correct"]
        )
        print(
            f"  {score:.1f} ({n_gen}x {int(score * n_gen)}/{n_gen} agree): "
            f"{n} samples, {correct} correct ({correct / n:.0%})"
        )

    # --- AUROC: consistency as confidence score ---
    scores = np.array([r["consistency"] for r in records])
    labels = np.array([int(r["is_correct"]) for r in records])

    auroc = roc_auc_score(labels, scores)
    print(f"\nConsistency AUROC: {auroc:.4f}")

    # Compare: AUROC using only full-consistency (1.0) vs not
    for threshold in [0.6, 0.8, 1.0]:
        pred_pos = scores >= threshold
        tp = np.sum(pred_pos & labels)
        fp = np.sum(pred_pos & ~labels)
        fn = np.sum(~pred_pos & labels)
        tn = np.sum(~pred_pos & ~labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(
            f"  Threshold >={threshold}: P={precision:.3f} R={recall:.3f} "
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )

    # --- Confident but wrong: high consistency yet incorrect ---
    cw_threshold = 0.8
    confident_wrong = [
        r for r in records if r["consistency"] >= cw_threshold and not r["is_correct"]
    ]
    print(
        f"\nConfident-but-wrong (consistency >= {cw_threshold}): "
        f"{len(confident_wrong)} / {n_samples} samples"
    )
    if confident_wrong:
        print(f"  Examples of high-confidence errors:")
        for r in confident_wrong[:3]:
            print(
                f"    consistency={r['consistency']:.1f} → "
                f"'{r['majority_answer']}' (true answers: not matched)"
            )

    # --- Save results ---
    output = {
        "n_samples": n_samples,
        "n_gen": n_gen,
        "temperature": temperature,
        "accuracy": accuracy,
        "n_correct": correct_count,
        "n_incorrect": n_samples - correct_count,
        "consistency_auroc": auroc,
        "records": [{k: v for k, v in r.items() if k != "all_ids"} for r in records],
    }
    results_file = output_path / "consistency_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_consistency")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hellaswag",
        choices=["triviaqa", "squad", "hellaswag"],
    )
    parser.add_argument(
        "--n_gen", type=int, default=5, help="Number of samples per question"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    main(
        args.n_samples,
        args.device,
        args.output_dir,
        args.dataset,
        args.n_gen,
        args.temperature,
    )
