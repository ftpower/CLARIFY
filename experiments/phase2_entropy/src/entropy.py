"""Logit lens entropy and related metrics for hallucination diagnosis."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def compute_logit_lens_entropy(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    b_U: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict:
    """Compute per-layer entropy and confidence metrics via logit lens.

    For each layer's hidden state h_ℓ, projects onto the vocabulary:
        logits_ℓ = h_ℓ @ W_U + b_U
        p_ℓ = softmax(logits_ℓ / temperature)
        H(ℓ) = -Σ p_ℓ log p_ℓ

    Args:
        hidden_states: list of [1, d_model] tensors (on CPU or GPU).
        W_U: unembedding matrix [d_model, vocab_size].
        b_U: unembedding bias [vocab_size] or None.
        temperature: softmax temperature (default 1.0).

    Returns:
        dict with keys:
            entropy: list[float] — H(ℓ) per layer
            max_prob: list[float] — max softmax prob per layer
            top5_mass: list[float] — sum of top-5 softmax probs per layer
            top1_ids: list[int] — argmax token id per layer
            top5_ids: list[list[int]] — top-5 token ids per layer
    """
    device = W_U.device
    entropies = []
    max_probs = []
    top5_masses = []
    top1_ids = []
    top5_ids = []

    for h in hidden_states:
        h_dev = h.to(device)
        logits = h_dev @ W_U
        if b_U is not None:
            logits = logits + b_U

        probs = torch.softmax(logits / temperature, dim=-1)  # [1, vocab_size]

        # Entropy: H = -Σ p log p
        log_probs = torch.log_softmax(logits / temperature, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        entropies.append(entropy)

        # Max probability
        max_p = probs.max(dim=-1)
        max_probs.append(max_p.values.item())
        top1_ids.append(max_p.indices.item())

        # Top-5 mass
        top5_vals, top5_idx = torch.topk(probs, k=5, dim=-1)
        top5_masses.append(top5_vals.sum().item())
        top5_ids.append(top5_idx[0].tolist())

    return {
        "entropy": entropies,
        "max_prob": max_probs,
        "top5_mass": top5_masses,
        "top1_ids": top1_ids,
        "top5_ids": top5_ids,
    }


def compute_per_layer_auroc(
    per_sample_results: list[dict],
    n_layers: int,
    metric: str = "entropy",
) -> dict:
    """Compute AUROC for each layer using the given metric as detection score.

    Args:
        per_sample_results: list of dicts, each with:
            - "is_correct": bool
            - metric: list[float] — per-layer values of the metric
        n_layers: total number of layers.
        metric: which metric to use as score ("entropy", "max_prob", "top5_mass").

    Returns:
        dict with keys: aurocs (list[float]), best_layer (int), best_auroc (float).
    """
    aurocs = []
    best_auroc = 0.0
    best_layer = -1

    for li in range(n_layers):
        scores = []
        labels = []
        for r in per_sample_results:
            scores.append(r[metric][li])
            labels.append(1 if r["is_correct"] else 0)

        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")

        aurocs.append(auc)

        if not np.isnan(auc) and auc > best_auroc:
            best_auroc = auc
            best_layer = li

    return {"aurocs": aurocs, "best_layer": best_layer, "best_auroc": best_auroc}


def compute_collapse_stats(
    per_sample_results: list[dict],
    n_layers: int,
) -> dict:
    """Find the entropy collapse layer ℓ* = argmin H(ℓ) for each sample.

    Returns:
        dict with:
            collapse_layers_correct: list[int] — ℓ* for correct samples
            collapse_layers_incorrect: list[int] — ℓ* for incorrect samples
            mean_entropy_correct: list[float] — mean H(ℓ) per layer for correct
            std_entropy_correct: list[float]
            mean_entropy_incorrect: list[float]
            std_entropy_incorrect: list[float]
    """
    correct_entropies = [[] for _ in range(n_layers)]
    incorrect_entropies = [[] for _ in range(n_layers)]
    collapse_correct = []
    collapse_incorrect = []

    for r in per_sample_results:
        ent = np.array(r["entropy"])
        collapse_layer = int(np.argmin(ent))

        if r["is_correct"]:
            collapse_correct.append(collapse_layer)
            for li in range(n_layers):
                correct_entropies[li].append(ent[li])
        else:
            collapse_incorrect.append(collapse_layer)
            for li in range(n_layers):
                incorrect_entropies[li].append(ent[li])

    mean_c = [float(np.mean(v)) if v else np.nan for v in correct_entropies]
    std_c = [float(np.std(v)) if v else np.nan for v in correct_entropies]
    mean_i = [float(np.mean(v)) if v else np.nan for v in incorrect_entropies]
    std_i = [float(np.std(v)) if v else np.nan for v in incorrect_entropies]

    return {
        "collapse_layers_correct": collapse_correct,
        "collapse_layers_incorrect": collapse_incorrect,
        "mean_entropy_correct": mean_c,
        "std_entropy_correct": std_c,
        "mean_entropy_incorrect": mean_i,
        "std_entropy_incorrect": std_i,
    }
