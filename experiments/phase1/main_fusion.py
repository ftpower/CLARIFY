"""Signal fusion: combine logit_variance + residual_norm for hallucination detection.

Tests simple combinations (average, logistic regression) to see if fusion
beats individual signals.

Usage:
    python main_fusion.py --n_samples 200 --dataset hellaswag
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_triviaqa,
    load_squad,
    load_hellaswag,
    format_prompt,
    check_correct,
)
from src.model_utils import load_model, get_per_layer_hidden_states


def extract_signals(hidden_states, W_U, device):
    """Extract logit_variance and residual_norm for each layer."""
    n_layers = len(hidden_states)
    variances = []
    norms = []

    for h in hidden_states:
        h = h.to(device)
        # logit variance
        logits = h @ W_U  # [1, vocab_size]
        variances.append(float(logits.var(dim=-1).item()))
        # residual norm
        norms.append(float(torch.norm(h, dim=-1).item()))

    return variances, norms


def compute_auroc(scores, labels):
    """Compute AUROC, flipping if < 0.5."""
    try:
        auroc = roc_auc_score(labels, scores)
        if auroc < 0.5:
            auroc = roc_auc_score(labels, -scores)
        return max(auroc, 0.5)
    except ValueError:
        return np.nan


def main(
    n_samples: int = 200,
    device: str = "cuda",
    output_dir: str = "outputs_fusion",
    dataset: str = "hellaswag",
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
    W_U = model.unembed.W_U
    n_layers = model.cfg.n_layers
    n_total_layers = n_layers + 1

    # --- Collect signals ---
    print(f"Extracting signals across {n_total_layers} layers...")

    all_variances = [[] for _ in range(n_total_layers)]
    all_norms = [[] for _ in range(n_total_layers)]
    all_labels = []  # shared across layers

    for sample in tqdm(samples, desc="Samples"):
        question = sample["question"]
        answers = sample["answers"]
        context = sample["context"]
        prompt = format_prompt(question, context, dataset=dataset)

        hidden_states, _, gen_id, gen_text = get_per_layer_hidden_states(model, prompt)
        is_correct = check_correct(gen_text.strip(), answers)

        variances, norms = extract_signals(hidden_states, W_U, device)

        for li in range(n_total_layers):
            all_variances[li].append(variances[li])
            all_norms[li].append(norms[li])

        all_labels.append(int(is_correct))

    labels = np.array(all_labels)
    correct_count = int(labels.sum())
    print(f"Correct: {correct_count}, Incorrect: {n_samples - correct_count}")
    print(f"Accuracy: {correct_count / n_samples:.1%}")

    # --- Per-layer single-signal baselines ---
    best_var_auroc = 0
    best_norm_auroc = 0
    best_var_layer = -1
    best_norm_layer = -1

    for li in range(n_total_layers):
        auroc_var = compute_auroc(np.array(all_variances[li]), labels)
        auroc_norm = compute_auroc(np.array(all_norms[li]), labels)
        if auroc_var > best_var_auroc:
            best_var_auroc = auroc_var
            best_var_layer = li
        if auroc_norm > best_norm_auroc:
            best_norm_auroc = auroc_norm
            best_norm_layer = li

    print(f"\nSingle-signal baselines:")
    print(f"  logit_variance: AUROC = {best_var_auroc:.4f} @ L{best_var_layer}")
    print(f"  residual_norm:  AUROC = {best_norm_auroc:.4f} @ L{best_norm_layer}")

    # --- Fusion methods ---
    print("\n" + "=" * 80)
    print("Fusion Results: per-layer AUROC for each fusion method")
    print("=" * 80)

    best_overall = {"method": "", "layer": -1, "auroc": 0}

    # Method 1: z-score averaging (no training)
    print(
        f"\n{'Layer':>6} {'var_only':>10} {'norm_only':>10} {'zscore_avg':>12} {'logreg':>10} {'best_individual':>14}"
    )
    print("-" * 72)

    for li in range(n_total_layers):
        variance_scores = np.array(all_variances[li])
        norm_scores = np.array(all_norms[li])

        # Z-score normalize
        var_z = (variance_scores - variance_scores.mean()) / (
            variance_scores.std() + 1e-12
        )
        norm_z = (norm_scores - norm_scores.mean()) / (norm_scores.std() + 1e-12)

        # Simple average of z-scores
        fused_zscore_avg = var_z + norm_z  # equally weighted
        auroc_zscore = compute_auroc(fused_zscore_avg, labels)

        # Logistic regression fusion (5-fold cross-val style: split-half)
        # Fit on first half, evaluate on second half
        n_half = n_samples // 2
        X = np.stack([variance_scores, norm_scores], axis=1)
        X_scaled = StandardScaler().fit_transform(X)

        # Train on first half
        X_train, X_test = X_scaled[:n_half], X_scaled[n_half:]
        y_train, y_test = labels[:n_half], labels[n_half:]

        lr = LogisticRegression(max_iter=1000)
        try:
            lr.fit(X_train, y_train)
            y_pred = lr.predict_proba(X_test)[:, 1]
            auroc_logreg = compute_auroc(y_pred, y_test)
        except Exception:
            auroc_logreg = np.nan

        auroc_var = compute_auroc(variance_scores, labels)
        auroc_norm = compute_auroc(norm_scores, labels)
        best_individual = max(auroc_var, auroc_norm)

        print(
            f"{li:>6} {auroc_var:10.4f} {auroc_norm:10.4f} "
            f"{auroc_zscore:12.4f} {auroc_logreg:10.4f} {best_individual:14.4f}"
        )

        for method_name, auroc_val in [
            ("zscore_avg", auroc_zscore),
            ("logreg", auroc_logreg),
        ]:
            if not np.isnan(auroc_val) and auroc_val > best_overall["auroc"]:
                best_overall = {"method": method_name, "layer": li, "auroc": auroc_val}

    # --- Best layer summary ---
    print("\n" + "=" * 80)
    print("Best Results Summary")
    print("=" * 80)
    print(f"  logit_variance (single):     {best_var_auroc:.4f} @ L{best_var_layer}")
    print(f"  residual_norm (single):      {best_norm_auroc:.4f} @ L{best_norm_layer}")
    print(
        f"  z-score average (fusion):    {best_overall['auroc']:.4f} @ L{best_overall['layer']} [{best_overall['method']}]"
    )

    # --- Mid-layer average: practical approach that doesn't require layer selection ---
    mid_start, mid_end = 11, 23
    mid_var = np.array(
        [
            compute_auroc(np.array(all_variances[li]), labels)
            for li in range(mid_start, mid_end + 1)
        ]
    )
    mid_norm = np.array(
        [
            compute_auroc(np.array(all_norms[li]), labels)
            for li in range(mid_start, mid_end + 1)
        ]
    )
    print(f"\n  Mid-layer (L{mid_start}-L{mid_end}) mean AUROC:")
    print(f"    logit_variance: {mid_var.mean():.4f} ± {mid_var.std():.4f}")
    print(f"    residual_norm:  {mid_norm.mean():.4f} ± {mid_norm.std():.4f}")

    # --- Save ---
    output = {
        "n_samples": n_samples,
        "n_correct": correct_count,
        "n_incorrect": n_samples - correct_count,
        "accuracy": correct_count / n_samples,
        "best_single_var": {"auroc": best_var_auroc, "layer": best_var_layer},
        "best_single_norm": {"auroc": best_norm_auroc, "layer": best_norm_layer},
        "best_fusion": best_overall,
        "mid_layer_stats": {
            "range": f"L{mid_start}-L{mid_end}",
            "var_mean_auroc": float(mid_var.mean()),
            "var_std_auroc": float(mid_var.std()),
            "norm_mean_auroc": float(mid_norm.mean()),
            "norm_std_auroc": float(mid_norm.std()),
        },
    }
    results_file = output_path / "fusion_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_fusion")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hellaswag",
        choices=["triviaqa", "squad", "hellaswag"],
    )
    args = parser.parse_args()
    main(args.n_samples, args.device, args.output_dir, args.dataset)
