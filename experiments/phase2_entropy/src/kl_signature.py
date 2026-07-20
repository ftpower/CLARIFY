"""KL Signature Matrix for hallucination detection.

Based on "Between the Layers Lies the Truth" (Badash et al., ICML 2026).
Converts per-layer post-MLP activations into an L×L KL-divergence signature
matrix that captures cross-layer agreement patterns.

Core insight: hallucination manifests as inconsistency BETWEEN layers,
not as absolute values at individual layers.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _stable_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled softmax with numerical stability."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits / temperature
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    return torch.softmax(scaled, dim=-1)


def _pairwise_kl(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute pairwise KL divergence matrix.

    Args:
        probs: [L, D] tensor where each row is a probability distribution.
        eps: small constant for numerical stability.

    Returns:
        kl_matrix: [L, L] tensor where kl_matrix[i, j] = D_KL(p_i || p_j).
    """
    probs = probs.clamp(min=eps, max=1.0 - eps)
    log_probs = torch.log(probs)  # [L, D]

    # D_KL(p_i || p_j) = Σ_k p_i_k * (log p_i_k - log p_j_k)
    # = Σ_k p_i_k * log p_i_k - Σ_k p_i_k * log p_j_k
    # Broadcast: [L, 1, D] vs [1, L, D] -> [L, L, D]
    p_i = probs.unsqueeze(1)  # [L, 1, D]
    log_p_i = log_probs.unsqueeze(1)  # [L, 1, D]
    log_p_j = log_probs.unsqueeze(0)  # [1, L, D]

    # entropies = -Σ p_i * log p_i (only for debugging)
    cross_entropy = -(p_i * log_p_j).sum(dim=-1)  # [L, L]
    entropy = -(p_i * log_p_i).sum(dim=-1)  # [L, 1]

    kl = cross_entropy - entropy  # [L, L]

    # Clamp negative values (numerical noise)
    kl = torch.clamp(kl, min=0.0)
    return kl


def contrast_correction(kl_matrix: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply contrast correction: S' = 1 - exp(-alpha * S).

    Maps KL values from [0, ∞) to [0, 1), improving dynamic range.
    """
    return 1.0 - torch.exp(-alpha * kl_matrix)


def compute_kl_signature(
    post_mlp_states: list[torch.Tensor],
    temperature: float = 1.0,
    eps: float = 1e-12,
    alpha: float | None = None,
    return_raw: bool = False,
) -> np.ndarray:
    """Compute the KL signature matrix from per-layer post-MLP activations.

    Args:
        post_mlp_states: list of [1, d_model] tensors, one per layer.
        temperature: softmax temperature (lower = sharper distributions).
        eps: numerical stability epsilon for KL.
        alpha: contrast correction strength. None = skip correction.
        return_raw: if True, return (features, kl_matrix) for analysis.

    Returns:
        feature_vector: 1D numpy array of shape [L*L,] containing the
          flattened KL signature (upper triangle if symmetric measure used).
    """
    # Stack into [L, D]
    states = torch.cat([h for h in post_mlp_states], dim=0)  # [L, d_model]

    # Step 1: Temperature-scaled softmax
    probs = _stable_softmax(states, temperature)  # [L, d_model]

    # Step 2: Pairwise KL divergence
    kl = _pairwise_kl(probs, eps=eps)  # [L, L]

    # Step 3: Optional contrast correction
    if alpha is not None and alpha > 0:
        kl = contrast_correction(kl, alpha)

    features = kl.flatten().cpu().numpy().astype(np.float64)

    if return_raw:
        return features, kl.cpu().numpy()
    return features


def compute_kl_signatures_batch(
    post_mlp_states_list: list[list[torch.Tensor]],
    temperature: float = 1.0,
    eps: float = 1e-12,
    alpha: float | None = None,
) -> np.ndarray:
    """Compute KL signatures for a batch of samples.

    Args:
        post_mlp_states_list: list of per-sample post_mlp_states.
        temperature, eps, alpha: passed to compute_kl_signature.

    Returns:
        X: [n_samples, L*L] feature matrix.
    """
    features = []
    for states in post_mlp_states_list:
        feat = compute_kl_signature(states, temperature, eps, alpha)
        features.append(feat)
    return np.array(features)
