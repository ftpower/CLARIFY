"""Direction B: Layer-wise Information Deficiency for Hallucination Detection.

Based on Kim et al. (ACL 2024): "Detecting LLM Hallucination Through
Layer-wise Information Deficiency."

Core idea: I_ℓ = H_ℓ(Q|∅) - H_ℓ(Q|C) — the per-layer entropy reduction
gained from context. Low LI → model didn't use context → higher
hallucination risk.

For HellaSwag: "context" = the 4 answer choices.
               "no context" = only the sentence stem.

Usage:
    python main_li_deficiency.py
    python main_li_deficiency.py --n_samples 200
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
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, check_correct
from src.hidden_state import extract_hidden_states
from src.entropy import compute_logit_lens_entropy, compute_per_layer_auroc


def format_prompt_no_ctx(sentence_stem: str) -> str:
    """HellaSwag prompt WITHOUT answer choices — model completes blindly."""
    return (
        f"Complete the sentence with the most natural ending.\n\n"
        f"Context: {sentence_stem}\n\n"
        f"Answer:"
    )


def format_prompt_with_ctx(sentence_stem: str, choices_text: str) -> str:
    """Standard HellaSwag prompt WITH 4 answer choices."""
    return (
        f"Complete the sentence with the most natural ending. "
        f"Answer with a single letter A, B, C, or D.\n\n"
        f"Context: {sentence_stem}\n"
        f"{choices_text}\n\n"
        f"Answer:"
    )


def main(
    n_samples: int = 500,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    output_dir: str = "outputs",
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    n_layers = model.cfg.n_layers
    n_total = n_layers + 1

    # ── Per-sample LI computation ──────────────────────────────────────
    print(f"Processing {n_samples} samples (2 forward passes each)...")
    per_sample = []
    correct_count = 0

    for sample in tqdm(samples, desc="Samples"):
        stem = sample["question"]
        choices = sample["context"]

        prompt_ctx = format_prompt_with_ctx(stem, choices)
        prompt_noctx = format_prompt_no_ctx(stem)

        # Forward pass WITH context
        hs_ctx, logits_ctx, gen_id, gen_text = extract_hidden_states(model, prompt_ctx)
        is_correct = check_correct(
            gen_text.strip(), sample["answers"], dataset="hellaswag"
        )
        if is_correct:
            correct_count += 1
        metrics_ctx = compute_logit_lens_entropy(hs_ctx, W_U, b_U, temperature=1.0)

        # Forward pass WITHOUT context
        hs_noctx, _, _, _ = extract_hidden_states(model, prompt_noctx)
        metrics_noctx = compute_logit_lens_entropy(hs_noctx, W_U, b_U, temperature=1.0)

        # LI per layer
        li_per_layer = [
            metrics_noctx["entropy"][i] - metrics_ctx["entropy"][i]
            for i in range(n_total)
        ]
        li_total = sum(li_per_layer)
        li_mean = np.mean(li_per_layer)

        per_sample.append(
            {
                "question": stem,
                "answers": sample["answers"],
                "generated_text": gen_text.strip(),
                "is_correct": is_correct,
                "entropy_ctx": metrics_ctx["entropy"],
                "entropy_noctx": metrics_noctx["entropy"],
                "li_per_layer": li_per_layer,
                "li_total": float(li_total),
                "li_mean": float(li_mean),
            }
        )

    accuracy = correct_count / n_samples
    print(f"\nAccuracy: {accuracy:.4f} ({correct_count}/{n_samples})")

    # ── Free GPU ───────────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── AUROC analysis ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Layer-wise Information Deficiency — AUROC Analysis")
    print(f"{'=' * 60}")

    # 1. Total LI as single detection score
    li_scores = np.array([s["li_total"] for s in per_sample])
    labels = np.array([1 if s["is_correct"] else 0 for s in per_sample])
    auroc_total = roc_auc_score(labels, li_scores)
    print(f"AUROC(LI_total): {auroc_total:.4f}")

    # 2. Mean LI
    li_mean = np.array([s["li_mean"] for s in per_sample])
    auroc_mean = roc_auc_score(labels, li_mean)
    print(f"AUROC(LI_mean):  {auroc_mean:.4f}")

    # 3. Per-layer LI AUROC (which layer's LI is most discriminative?)
    li_matrix = np.array([s["li_per_layer"] for s in per_sample])  # [N, n_total]
    print(f"\nPer-layer LI AUROC:")
    best_li_layer = -1
    best_li_auroc = 0.0
    for li in range(n_total):
        auroc = roc_auc_score(labels, li_matrix[:, li])
        marker = ""
        if auroc > best_li_auroc:
            best_li_auroc = auroc
            best_li_layer = li
            marker = " <-- best"
        if auroc > 0.55 or auroc < 0.45:
            print(f"  L{li:>2}: {auroc:.4f}{marker}")

    # 4. Group stats
    li_correct = li_scores[labels == 1]
    li_incorrect = li_scores[labels == 0]
    print(f"\nLI_total stats:")
    print(f"  Correct:   mean={li_correct.mean():.4f}  std={li_correct.std():.4f}")
    print(f"  Incorrect: mean={li_incorrect.mean():.4f}  std={li_incorrect.std():.4f}")
    d = abs(li_correct.mean() - li_incorrect.mean()) / np.sqrt(
        (li_correct.std() ** 2 + li_incorrect.std() ** 2) / 2
    )
    print(f"  Cohen's d: {d:.4f}")

    # Also check: is separate entropy (no context) alone a good detector?
    ent_noctx_matrix = np.array([s["entropy_noctx"] for s in per_sample])
    ent_ctx_matrix = np.array([s["entropy_ctx"] for s in per_sample])
    print(f"\nReference — AUROC(H_no_ctx) per layer (best 3):")
    noctx_aurocs = []
    for li in range(n_total):
        noctx_aurocs.append(roc_auc_score(labels, ent_noctx_matrix[:, li]))
    top3 = np.argsort(noctx_aurocs)[::-1][:3]
    for li in top3:
        print(f"  L{li}: {noctx_aurocs[li]:.4f}")

    # 5. Baseline comparison
    baseline = 0.68
    print(f"\n{'=' * 60}")
    print(f"Baseline (max_p at L28):     AUROC = {baseline:.4f}")
    print(f"LI_total:                    AUROC = {auroc_total:.4f}")
    print(f"Best per-layer LI (L{best_li_layer}):   AUROC = {best_li_auroc:.4f}")
    delta = auroc_total - baseline
    if delta > 0.02:
        print(f"✓ LI BEATS baseline by +{delta:.4f}")
    elif delta > -0.02:
        print(f"≈ LI TIED with baseline (Δ={delta:+.4f})")
    else:
        print(f"✗ LI below baseline (Δ={delta:+.4f})")

    # ── Save ───────────────────────────────────────────────────────────
    # Per-layer LI means for correct vs incorrect
    li_c_mean = li_matrix[labels == 1].mean(axis=0).tolist()
    li_i_mean = li_matrix[labels == 0].mean(axis=0).tolist()

    results = {
        "method": "Layer-wise Information Deficiency",
        "paper": "Kim et al. (ACL 2024)",
        "model": model_id,
        "dataset": "hellaswag",
        "n_samples": n_samples,
        "n_layers": n_layers,
        "n_total_layers": n_total,
        "accuracy": float(accuracy),
        "baseline_auroc": baseline,
        "li_total_auroc": float(auroc_total),
        "li_mean_auroc": float(auroc_mean),
        "best_per_layer_li_auroc": float(best_li_auroc),
        "best_per_layer_li_layer": int(best_li_layer),
        "li_correct_mean_curve": li_c_mean,
        "li_incorrect_mean_curve": li_i_mean,
    }

    results_file = output_path / "li_deficiency_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ── Quick plot ─────────────────────────────────────────────────────
    try:
        _plot_li(results, n_total, model_id, output_path)
    except Exception as e:
        print(f"Plot failed: {e}")


def _plot_li(results: dict, n_total: int, model_id: str, output_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_total))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) LI curve: correct vs incorrect
    ax = axes[0]
    ax.plot(
        layers, results["li_correct_mean_curve"], "b-", linewidth=2, label="Correct"
    )
    ax.plot(
        layers, results["li_incorrect_mean_curve"], "r-", linewidth=2, label="Incorrect"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("LI = H(no_ctx) - H(ctx)")
    ax.set_title("Layer-wise Information Deficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) LI text summary
    ax = axes[1]
    ax.axis("off")
    summary = (
        f"Model: {model_id}\n"
        f"Dataset: HellaSwag\n"
        f"Samples: {results['n_samples']}\n"
        f"Accuracy: {results['accuracy']:.4f}\n\n"
        f"LI_total AUROC: {results['li_total_auroc']:.4f}\n"
        f"LI_mean AUROC:  {results['li_mean_auroc']:.4f}\n"
        f"Best per-layer LI: L{results['best_per_layer_li_layer']} = "
        f"{results['best_per_layer_li_auroc']:.4f}\n\n"
        f"Baseline (max_p L28): 0.6800"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    fig.suptitle(
        f"Layer-wise Information Deficiency — {model_id} HellaSwag", fontsize=13
    )
    fig.tight_layout()

    plot_file = output_path / "li_deficiency.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_samples, args.device, args.model, args.output_dir, args.seed)
