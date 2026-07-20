"""Generalization feature computation for Phase 4 Plan 1 — Part A.

Seven detection features spanning three independent signal sources:
  1. EigenScore (INSIDE)        — cross-sample semantic consistency via K-shot sampling
  2. HaloScope ζ                — population-level outlier degree via cross-sample SVD
  3. Attn/FFN ratio             — sub-layer functional differentiation (InternalInspector)
  4. D2 JS top-K mean           — layer-wise prediction consistency (our D2 finding)
  5. max_p                      — per-layer max softmax probability (baseline)
  6. Entropy                    — per-layer logit-lens entropy
  7. Top-5 mass                 — per-layer top-5 probability concentration

All features are zero-training, single-or-few forward passes, no gradient computation.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1: EigenScore (INSIGHT / Chen et al.)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_eigenscore(
    model,
    prompt: str,
    layer_idx: int,
    K: int = 10,
    temperature: float = 0.5,
) -> float:
    """INSIDE EigenScore: K-shot hidden-state covariance log-determinant.

    For a single prompt, runs K temperature-sampled forward passes. At each pass,
    extracts the last-token hidden state at the given layer. The K hidden states
    form a d_model x K matrix Z. The covariance Σ = Z^T @ J_d @ Z (where J_d is
    the centering matrix) captures the semantic spread of the model's internal
    representations across different sampling paths.

    EigenScore = (1/K) * log(det(Σ + α*I))

    Lower EigenScore → more concentrated representations → potential
    overconfidence → higher hallucination risk.

    Args:
        model: HookedTransformer instance.
        prompt: Input prompt text.
        layer_idx: Which layer's residual stream to extract.
        K: Number of temperature-sampled forward passes (default 10).
        temperature: Sampling temperature (default 0.5).

    Returns:
        EigenScore value (float). Returns NaN if computation fails.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from phase4_utils.hidden_state_extended import generate_with_temperature

    hidden_states_list = []

    for _ in range(K):
        # Each call does a fresh temperature-sampled forward pass
        _, step_states = generate_with_temperature(
            model,
            prompt,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=20,
            return_hidden_states=True,
            hook_layer=layer_idx,
        )
        if step_states:
            # Use the hidden state at the LAST decode step
            hidden_states_list.append(step_states[-1])
        else:
            # Fallback: use a zero vector if generation produced nothing
            hidden_states_list.append(
                torch.zeros(model.cfg.d_model)
            )

    if len(hidden_states_list) < 2:
        return float("nan")

    # Build d × K matrix Z
    Z = torch.stack(hidden_states_list, dim=1)  # [d_model, K]
    d_model, K_actual = Z.shape

    # Center: Z_tilde = Z @ J_K where J_K = I - (1/K)*11^T
    J = torch.eye(K_actual) - (1.0 / K_actual) * torch.ones(K_actual, K_actual)
    Z_centered = Z @ J  # [d_model, K]

    # Covariance: Σ = Z_centered @ Z_centered^T  [d_model, d_model]
    # But d_model >> K, so compute via the K x K Gram matrix instead
    # det(Z_centered^T @ Z_centered + α*I) = det(Z_centered @ Z_centered^T + α*I)
    # (up to zero eigenvalues, which α handles)
    gram = Z_centered.T @ Z_centered  # [K, K]
    alpha = 0.001
    gram_reg = gram + alpha * torch.eye(K_actual)

    try:
        # Log-determinant via Cholesky on the K×K Gram matrix (much cheaper)
        L = torch.linalg.cholesky(gram_reg.float())
        log_det = 2.0 * torch.sum(torch.log(torch.diag(L))).item()
        eigenscore = log_det / K_actual
    except RuntimeError:
        # Cholesky failed (singular matrix), try eigendecomposition
        try:
            eigvals = torch.linalg.eigvalsh(gram_reg.float())
            eigvals = torch.clamp(eigvals, min=1e-10)
            log_det = torch.sum(torch.log(eigvals)).item()
            eigenscore = log_det / K_actual
        except Exception:
            return float("nan")

    return float(eigenscore)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 2: HaloScope ζ (HaloScope / Bohdal et al.)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_haloscope_zeta_batch(
    hidden_states_matrix: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """HaloScope ζ: cross-sample SVD weighted projection modulus.

    Given a batch of hidden states from the same layer across samples, performs
    SVD to find dominant directions of variation in the population. Each sample's
    ζ score is the weighted projection onto the top-k singular vectors:

        ζ_i = (1/k) * Σ_{j=1}^{k} σ_j * ⟨f̃_i, v_j⟩²

    where f̃_i is the centered hidden state, v_j is the j-th right singular vector,
    and σ_j is the corresponding singular value.

    High ζ → sample is an outlier in the population → higher hallucination risk.

    Args:
        hidden_states_matrix: [N, d_model] numpy array of hidden states at a
            specific layer, across samples.
        k: Number of top singular vectors to use (default 5).

    Returns:
        zeta_scores: [N] numpy array, per-sample HaloScope ζ scores.
    """
    N, d = hidden_states_matrix.shape
    if N < k + 2:
        # Not enough samples for meaningful SVD
        return np.zeros(N, dtype=np.float32)

    # Center the data
    mu = hidden_states_matrix.mean(axis=0, keepdims=True)  # [1, d]
    F_tilde = hidden_states_matrix - mu  # [N, d]

    # SVD: F_tilde = U @ Σ @ V^T
    # U: [N, N], Σ: [min(N,d)], V^T: [min(N,d), d]
    # We need V (right singular vectors, in d-space)
    # Since N << d (e.g., N=250, d=2048), use economy SVD
    try:
        U, S, Vt = np.linalg.svd(F_tilde.astype(np.float32), full_matrices=False)
        # U: [N, min(N,d)], S: [min(N,d)], Vt: [min(N,d), d]
    except np.linalg.LinAlgError:
        return np.zeros(N, dtype=np.float32)

    k_actual = min(k, len(S))
    if k_actual == 0:
        return np.zeros(N, dtype=np.float32)

    # Vt[:k_actual] has shape [k_actual, d]
    V_top = Vt[:k_actual, :]  # [k, d]
    S_top = S[:k_actual]  # [k]

    # Project each centered sample onto the top-k singular vectors
    # proj[i, j] = ⟨f̃_i, v_j⟩  →  F_tilde @ V_top^T  gives [N, k]
    projections = F_tilde @ V_top.T  # [N, k]
    squared_projections = projections ** 2  # [N, k]

    # Weight by singular values
    weighted = squared_projections * S_top[None, :]  # [N, k]
    zeta = weighted.sum(axis=1) / k_actual  # [N]

    return zeta.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3: Attn/FFN Ratio (InternalInspector)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_attn_ffn_ratio(
    attn_output: torch.Tensor,
    ffn_output: torch.Tensor,
) -> float:
    """InternalInspector-style Attn/FFN functional differentiation ratio.

    r = ||attn_output||_2 / (||ffn_output||_2 + ε)

    Hypothesis (from InternalInspector + ROME/Knowledge Neurons literature):
      - FFN-dominated → parametric knowledge retrieval → factual tasks
      - Attention-dominated → context integration → reading comprehension
      - Imbalance in either direction can signal hallucination risk.

    Args:
        attn_output: [d_model] tensor, attention sub-layer output.
        ffn_output: [d_model] tensor, MLP sub-layer output.

    Returns:
        Ratio (float).
    """
    attn_norm = attn_output.norm(p=2).item()
    ffn_norm = ffn_output.norm(p=2).item()
    eps = 1e-8
    return attn_norm / (ffn_norm + eps)


def compute_attn_ffn_ratio_batch(
    attn_states_per_sample: list[torch.Tensor],
    ffn_states_per_sample: list[torch.Tensor],
) -> np.ndarray:
    """Batch version: compute Attn/FFN ratio for a list of samples at one layer.

    Args:
        attn_states_per_sample: list of [d_model] tensors, one per sample.
        ffn_states_per_sample: list of [d_model] tensors, one per sample.

    Returns:
        [N] numpy array of ratio scores.
    """
    ratios = []
    for a, f in zip(attn_states_per_sample, ffn_states_per_sample):
        ratios.append(compute_attn_ffn_ratio(a, f))
    return np.array(ratios, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 4: D2 JS Top-K Mean (our Phase 3 finding)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_d2_js_topk(
    choice_probs: np.ndarray,
    top_k: int = 5,
    exclude_layer0: bool = True,
) -> dict:
    """D2 layer consistency: mean JS divergence of top-K most-divergent layer pairs.

    Scans all layer pairs (early, late) where early < late, computes per-sample
    JS divergence on 4-choice softmax distributions, and returns the mean JS
    across the top-K pairs (ranked by per-pair JS variance, indicating the pairs
    most sensitive to hallucination).

    Args:
        choice_probs: [N, n_layers, 4] numpy array of per-layer 4-choice softmax.
        top_k: Number of top layer pairs to average (default 5).
        exclude_layer0: If True, exclude L0 from being an early layer (DoLa
            showed L0 is always uniformly distributed in 1.7B, providing no
            meaningful contrast).

    Returns:
        dict with:
            "js_scores": [N] array — mean JS across top-K pairs per sample
            "best_pair": (early, late) — pair with highest JS AUROC
            "best_auroc": float — AUROC of best single pair (for reference)
            "top_pairs": list of (early, late, mean_js, auroc) tuples
    """
    from sklearn.metrics import roc_auc_score

    N, n_layers, n_choices = choice_probs.shape
    assert n_choices == 4, f"Expected 4-choice softmax, got {n_choices}"

    eps = 1e-10

    # We compute JS for all pairs but don't know labels here. Instead, just
    # compute JS scores for all samples across all pairs.
    all_pair_js = {}  # (early, late) -> [N] js array

    start_early = 1 if exclude_layer0 else 0
    for early in range(start_early, n_layers):
        for late in range(early + 1, n_layers):
            p_early = np.clip(choice_probs[:, early, :], eps, 1.0)
            p_late = np.clip(choice_probs[:, late, :], eps, 1.0)
            m = 0.5 * (p_early + p_late)

            kl_early = np.sum(p_early * np.log(p_early / m), axis=1)
            kl_late = np.sum(p_late * np.log(p_late / m), axis=1)
            js = 0.5 * (kl_early + kl_late)

            all_pair_js[(early, late)] = js

    # For AUROC, we need labels. If labels are provided, we compute per-pair
    # AUROC to select top-K. Otherwise, select by JS variance (higher variance
    # means the pair is more discriminative).
    # We'll return all pair info; the caller provides labels for AUROC.

    return {
        "all_pair_js": all_pair_js,
        "n_pairs": len(all_pair_js),
    }


def compute_d2_js_score(
    choice_probs: np.ndarray,
    early_layer: int,
    late_layer: int,
) -> np.ndarray:
    """Compute JS divergence for a specific (early, late) layer pair.

    Args:
        choice_probs: [N, n_layers, 4] array of 4-choice softmax probabilities.
        early_layer: Index of the early (shallower) layer.
        late_layer: Index of the late (deeper) layer.

    Returns:
        js_scores: [N] array of per-sample JS divergence values.
    """
    eps = 1e-10
    p_early = np.clip(choice_probs[:, early_layer, :], eps, 1.0)
    p_late = np.clip(choice_probs[:, late_layer, :], eps, 1.0)
    m = 0.5 * (p_early + p_late)

    kl_early = np.sum(p_early * np.log(p_early / m), axis=1)
    kl_late = np.sum(p_late * np.log(p_late / m), axis=1)
    js = 0.5 * (kl_early + kl_late)

    return js.astype(np.float32)


def select_top_js_pairs(
    all_pair_js: dict,
    labels: np.ndarray,
    top_k: int = 5,
) -> dict:
    """Select top-K layer pairs by their individual JS AUROC.

    Args:
        all_pair_js: {(early, late): [N] js_scores} dict from compute_d2_js_topk().
        labels: [N] binary array (1=correct, 0=incorrect).
        top_k: Number of pairs to keep.

    Returns:
        dict with "js_scores", "best_pair", "best_auroc", "top_pairs".
    """
    from sklearn.metrics import roc_auc_score

    pair_aurocs = []
    for (early, late), js_scores in all_pair_js.items():
        try:
            auc = roc_auc_score(labels, js_scores)
        except ValueError:
            auc = 0.5
        pair_aurocs.append((early, late, float(js_scores.mean()), auc))

    pair_aurocs.sort(key=lambda x: x[3], reverse=True)

    top_pairs = pair_aurocs[:top_k]
    N = len(labels)

    # Mean JS across top-K pairs per sample
    js_scores_sum = np.zeros(N, dtype=np.float64)
    for early, late, _mean_js, _auc in top_pairs:
        js_scores_sum += all_pair_js[(early, late)]

    js_scores = (js_scores_sum / top_k).astype(np.float32)

    best_early, best_late, best_mean_js, best_auc = top_pairs[0]

    return {
        "js_scores": js_scores,
        "best_pair": (best_early, best_late),
        "best_auroc": float(best_auc),
        "top_pairs": [
            {"early": int(e), "late": int(l), "mean_js": float(mj), "auroc": float(a)}
            for e, l, mj, a in top_pairs
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Features 5-7: max_p, Entropy, Top-5 mass (wrappers around existing entropy.py)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_max_prob_per_layer(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    b_U: torch.Tensor | None = None,
) -> np.ndarray:
    """Per-layer max softmax probability via logit lens.

    Args:
        hidden_states: list of [1, d_model] tensors, one per layer.
        W_U: unembedding matrix [d_model, vocab_size].
        b_U: optional unembedding bias [vocab_size].

    Returns:
        max_probs: [n_layers] numpy array of max probabilities.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_entropy"))
    from src.entropy import compute_logit_lens_entropy

    result = compute_logit_lens_entropy(hidden_states, W_U, b_U)
    return np.array(result["max_prob"], dtype=np.float32)


def compute_entropy_per_layer(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    b_U: torch.Tensor | None = None,
) -> np.ndarray:
    """Per-layer logit-lens entropy.

    Args:
        hidden_states: list of [1, d_model] tensors, one per layer.
        W_U: unembedding matrix [d_model, vocab_size].
        b_U: optional unembedding bias [vocab_size].

    Returns:
        entropies: [n_layers] numpy array of entropy values.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_entropy"))
    from src.entropy import compute_logit_lens_entropy

    result = compute_logit_lens_entropy(hidden_states, W_U, b_U)
    return np.array(result["entropy"], dtype=np.float32)


def compute_top5_mass_per_layer(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    b_U: torch.Tensor | None = None,
) -> np.ndarray:
    """Per-layer top-5 probability mass via logit lens.

    Args:
        hidden_states: list of [1, d_model] tensors, one per layer.
        W_U: unembedding matrix [d_model, vocab_size].
        b_U: optional unembedding bias [vocab_size].

    Returns:
        top5_masses: [n_layers] numpy array of top-5 probability sums.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_entropy"))
    from src.entropy import compute_logit_lens_entropy

    result = compute_logit_lens_entropy(hidden_states, W_U, b_U)
    return np.array(result["top5_mass"], dtype=np.float32)
