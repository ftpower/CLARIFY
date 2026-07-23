"""P0-1: INSIDE EigenScore on TriviaQA.

EigenScore = (1/K) * log(det(Σ + αI)) where Σ is the covariance of K
temperature-sampled hidden states at a middle layer. Lower score = more
concentrated representations = potential overconfidence/hallucination.

INSIDE (Chen et al., ICLR 2024): AUROC 82.7% on LLaMA-7B TriviaQA.
Hidden-state-based, completely vocabulary-independent.

Cost: K=10 forward passes per sample × 200 samples = 2000 passes, ~30 min GPU.

Usage:
    python main.py --n_samples 200
    python main.py --n_samples 10   # quick test
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

# Path setup
_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent / "phase4_generalization"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for shared

from shared import load_model_and_data, format_prompt
from phase4_utils.generalization_features import compute_eigenscore


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    output_dir: str = "outputs",
    seed: int = 42,
    layer_idx: int = 17,
    K: int = 10,
    temperature: float = 0.5,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model + data ──────────────────────────────────────────────
    model, samples, _, _ = load_model_and_data(
        n_samples=n_samples, seed=seed, device=device, model_id=model_id
    )

    # ── Load Phase 5.1 labels first (match by sample index) ──────────
    phase5_json = (
        Path(__file__).parent.parent.parent
        / "phase5_cross_task"
        / "outputs"
        / "triviaqa_features.json"
    )
    if not phase5_json.exists():
        print(f"ERROR: Phase 5.1 labels not found at {phase5_json}")
        sys.exit(1)
    with open(phase5_json) as f:
        p5_data = json.load(f)
    all_labels = [s["is_correct"] for s in p5_data["per_sample"]]
    all_labels = all_labels[:n_samples]  # Truncate if n_samples < 200
    assert len(all_labels) == len(samples), (
        f"Label count mismatch: {len(all_labels)} vs {len(samples)} samples"
    )
    print(f"Loaded {len(all_labels)} labels from Phase 5.1 output")

    # ── Compute EigenScore per sample ──────────────────────────────────
    print(f"\nComputing EigenScore (layer={layer_idx}, K={K}, T={temperature})...")
    print(f"  Estimated time: ~{n_samples * K * 0.15:.0f}s for {n_samples} samples")

    raw_scores: list[tuple[int, float]] = []  # (sample_idx, eigenscore)

    for i, sample in enumerate(tqdm(samples, desc="EigenScore")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        try:
            score = compute_eigenscore(
                model=model,
                prompt=prompt,
                layer_idx=layer_idx,
                K=K,
                temperature=temperature,
            )
            if np.isnan(score):
                continue
            raw_scores.append((i, score))
        except Exception:
            continue

    failed = n_samples - len(raw_scores)

    # ── Align scores with labels by sample index ───────────────────────
    eigenscores = np.array([s for _, s in raw_scores], dtype=np.float64)
    labels = np.array(
        [1 if all_labels[i] else 0 for i, _ in raw_scores], dtype=np.int32
    )

    print(f"\n  Valid: {len(eigenscores)}/{n_samples} (failed: {failed})")
    print(f"  Correct: {labels.sum()}/{len(labels)}")
    print(f"  EigenScore: mean={eigenscores.mean():.2f}, std={eigenscores.std():.2f}")

    # ── AUROC ──────────────────────────────────────────────────────────
    if labels.sum() == 0 or labels.sum() == len(labels):
        print("  All samples same label — AUROC undefined")
        auroc = 0.5
    else:
        # Lower EigenScore = more concentrated = potential hallucination
        # So use negative EigenScore as detection score
        auroc = roc_auc_score(1 - labels, -eigenscores)
        print(f"  AUROC = {auroc:.4f}")

    # ── Compute correct/incorrect group stats ───────────────────────────
    if labels.sum() > 0 and (1 - labels).sum() > 0:
        correct_scores = eigenscores[labels == 1]
        incorrect_scores = eigenscores[labels == 0]
        print(
            f"  Correct:   mean={correct_scores.mean():.2f} ± {correct_scores.std():.2f}"
        )
        print(
            f"  Incorrect: mean={incorrect_scores.mean():.2f} ± {incorrect_scores.std():.2f}"
        )

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples,
            "n_valid": len(eigenscores),
            "n_failed": failed,
            "model_id": model_id,
            "layer_idx": layer_idx,
            "K": K,
            "temperature": temperature,
            "seed": seed,
        },
        "auroc": auroc,
        "eigenscore_mean": float(eigenscores.mean()),
        "eigenscore_std": float(eigenscores.std()),
        "correct_mean": (
            float(eigenscores[labels == 1].mean()) if labels.sum() > 0 else None
        ),
        "incorrect_mean": (
            float(eigenscores[labels == 0].mean())
            if (1 - labels).sum() > 0
            else None
        ),
    }

    output_file = output_path / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P0-1: INSIDE EigenScore on TriviaQA"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        model_id=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        layer_idx=args.layer,
        K=args.K,
        temperature=args.temperature,
    )
