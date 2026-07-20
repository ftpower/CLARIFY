"""Trajectory shape feature extraction for hallucination detection.

Extracts ~20 geometric/speed/curvature features from per-layer entropy,
max_prob, and top5_mass trajectories. Designed to capture trajectory
SHAPE beyond single-layer point values.
"""

import numpy as np
from scipy import stats


def _safe_entropy(entropy: np.ndarray) -> np.ndarray:
    """Clip entropy to non-negative for numerical stability."""
    return np.clip(entropy, 0, None)


def _half_life_layer(values: np.ndarray) -> float:
    """Find the (interpolated) layer where values first drop to half of max."""
    v = np.asarray(values, dtype=np.float64)
    half = v[0] / 2.0
    if half <= 0:
        return 0.0
    below = np.where(v <= half)[0]
    if len(below) == 0:
        return float(len(v) - 1)
    idx = below[0]
    if idx == 0:
        return 0.0
    # Linear interpolation
    frac = (half - v[idx]) / (v[idx - 1] - v[idx]) if v[idx - 1] != v[idx] else 0.0
    return float(idx - frac)


def _saturation_layer(values: np.ndarray, threshold: float = 0.9) -> int:
    """First layer where value exceeds threshold."""
    above = np.where(np.asarray(values) >= threshold)[0]
    return int(above[0]) if len(above) > 0 else len(values) - 1


def _segmented_slope(values: np.ndarray, start: int, end: int) -> float:
    """Average per-layer slope in [start, end] via linear regression."""
    x = np.arange(start, min(end, len(values)), dtype=np.float64)
    y = np.asarray(values[start : min(end, len(values))], dtype=np.float64)
    if len(x) < 2:
        return 0.0
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)


def _curvature_features(values: np.ndarray) -> dict:
    """Second-order finite differences to characterize curvature."""
    v = np.asarray(values, dtype=np.float64)
    d2 = np.diff(v, n=2)  # second derivative
    if len(d2) == 0:
        return {
            "curvature_mean": 0.0,
            "curvature_max": 0.0,
            "curvature_max_layer": -1,
            "inflection_count": 0,
        }
    abs_d2 = np.abs(d2)
    # Count sign changes in d2 (inflection points)
    signs = np.sign(d2)
    sign_changes = (
        np.sum(np.abs(np.diff(signs[signs != 0])) > 0) if np.any(signs != 0) else 0
    )
    return {
        "curvature_mean": float(np.mean(abs_d2)),
        "curvature_max": float(np.max(abs_d2)),
        "curvature_max_layer": int(np.argmax(abs_d2)) + 1,
        "inflection_count": int(sign_changes),
    }


def extract_trajectory_features(
    entropy: list[float],
    max_prob: list[float],
    top5_mass: list[float],
) -> dict:
    """Extract shape features from per-layer trajectories.

    Args:
        entropy: per-layer logit lens entropy, length n_layers+1 (incl. embedding).
        max_prob: per-layer max softmax probability.
        top5_mass: per-layer top-5 probability mass.

    Returns:
        dict of ~20 scalar features.
    """
    ent = _safe_entropy(np.asarray(entropy, dtype=np.float64))
    mp = np.asarray(max_prob, dtype=np.float64)
    t5 = np.asarray(top5_mass, dtype=np.float64)
    n = len(ent)

    features = {}

    # ── Amplitude ────────────────────────────────────────────────────────
    features["entropy_auc"] = float(np.trapz(ent))
    features["entropy_max"] = float(np.max(ent))
    features["entropy_min"] = float(np.min(ent))
    features["entropy_range"] = float(np.max(ent) - np.min(ent))
    features["entropy_final"] = float(ent[-1])
    features["entropy_initial"] = float(ent[0])

    # ── Convergence speed ────────────────────────────────────────────────
    features["half_life"] = _half_life_layer(ent)
    features["quarter_life"] = (
        _half_life_layer(ent / ent[0] * 0.25) if ent[0] > 0 else 0.0
    )
    features["max_p_saturation"] = float(_saturation_layer(mp, 0.9))
    features["max_p_slope_to_sat"] = _segmented_slope(
        mp, 0, _saturation_layer(mp, 0.9) + 1
    )

    # ── Segmented slopes ─────────────────────────────────────────────────
    features["slope_early"] = _segmented_slope(ent, 0, 6)  # L0→L5
    features["slope_mid"] = _segmented_slope(ent, 8, 16)  # L8→L15
    features["slope_late"] = _segmented_slope(ent, 20, n)  # L20→L28
    # Slope ratios
    features["slope_ratio_early_mid"] = (
        features["slope_early"] / features["slope_mid"]
        if features["slope_mid"] != 0
        else 0.0
    )
    features["slope_ratio_mid_late"] = (
        features["slope_mid"] / features["slope_late"]
        if features["slope_late"] != 0
        else 0.0
    )

    # ── Curvature ────────────────────────────────────────────────────────
    curv = _curvature_features(ent)
    features.update(curv)

    # ── Layer-wise differences ───────────────────────────────────────────
    deltas = -np.diff(ent)  # H(ℓ) - H(ℓ+1), positive = entropy drop
    features["delta_max"] = float(np.max(deltas)) if len(deltas) > 0 else 0.0
    features["delta_max_layer"] = int(np.argmax(deltas)) if len(deltas) > 0 else -1
    features["delta_std"] = float(np.std(deltas)) if len(deltas) > 0 else 0.0
    features["delta_mean"] = float(np.mean(deltas)) if len(deltas) > 0 else 0.0

    # ── Cross-metric ─────────────────────────────────────────────────────
    ent_std = float(np.std(ent))
    features["entropy_cv"] = (
        ent_std / float(np.mean(ent)) if float(np.mean(ent)) > 0 else 0.0
    )
    features["entropy_max_p_corr"] = (
        float(np.corrcoef(ent, mp)[0, 1]) if ent_std > 0 else 0.0
    )
    features["top5_entropy_ratio"] = (
        float(np.mean(t5)) / float(np.mean(ent)) if float(np.mean(ent)) > 0 else 0.0
    )
    features["top5_min"] = float(np.min(t5))
    features["top5_final"] = float(t5[-1])
    features["max_prob_final"] = float(mp[-1])

    # ── Sanitize ─────────────────────────────────────────────────────────
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features
