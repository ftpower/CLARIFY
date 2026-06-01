"""Four estimators of hallucination channel width q^(ℓ)."""

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score


def _safe_kde(data: np.ndarray):
    """Fit a gaussian_kde, returning None if data variance is too low."""
    if np.std(data) < 1e-12:
        return None
    try:
        return gaussian_kde(data, bw_method='scott')
    except np.linalg.LinAlgError:
        return None


def estimate_q_overlap(conf_correct: np.ndarray, conf_incorrect: np.ndarray,
                       n_points: int = 1000) -> float:
    """Method B (primary): KDE overlap area.

    q = integral of min(p_K(p), p_N(p)) dp

    This is the most robust estimator. Range [0, 1].
    q → 0: distributions are well-separated (good)
    q → 1: distributions completely overlap (bad)
    """
    if len(conf_correct) < 3 or len(conf_incorrect) < 3:
        return np.nan

    kde_K = _safe_kde(conf_correct)
    kde_N = _safe_kde(conf_incorrect)
    if kde_K is None or kde_N is None:
        return np.nan

    lo = min(conf_correct.min(), conf_incorrect.min()) - 1e-8
    hi = max(conf_correct.max(), conf_incorrect.max()) + 1e-8
    lo, hi = max(0, lo), min(1, hi)
    grid = np.linspace(lo, hi, n_points)

    p_K = kde_K(grid)
    p_N = kde_N(grid)
    p_K /= p_K.sum()
    p_N /= p_N.sum()

    overlap = np.trapz(np.minimum(p_K, p_N), grid)
    return float(overlap)


def estimate_q_kl(conf_correct: np.ndarray, conf_incorrect: np.ndarray) -> float:
    """Method A: q = exp(-KL(μ_K || μ_N)) via KDE-based KL divergence.

    q → 0: large KL divergence between distributions (good separation)
    q → 1: distributions are nearly identical (same as overlap method)
    """
    if len(conf_correct) < 3 or len(conf_incorrect) < 3:
        return np.nan

    lo = min(conf_correct.min(), conf_incorrect.min()) - 1e-8
    hi = max(conf_correct.max(), conf_incorrect.max()) + 1e-8
    lo, hi = max(0, lo), min(1, hi)
    grid = np.linspace(lo, hi, 1000)

    kde_K = _safe_kde(conf_correct)
    kde_N = _safe_kde(conf_incorrect)
    if kde_K is None or kde_N is None:
        return np.nan
    p_K = kde_K(grid) + 1e-12
    p_N = kde_N(grid) + 1e-12
    p_K /= p_K.sum()
    p_N /= p_N.sum()

    kl = np.sum(p_K * np.log(p_K / p_N))
    q = np.exp(-kl)
    return float(min(q, 1.0))


def estimate_q_bhattacharyya(conf_correct: np.ndarray, conf_incorrect: np.ndarray,
                             n_points: int = 1000) -> float:
    """Method C: Bhattacharyya coefficient.

    BC = integral sqrt(p_K(p) * p_N(p)) dp
    Range [0, 1]. BC → 1 = distributions overlap completely.
    """
    if len(conf_correct) < 3 or len(conf_incorrect) < 3:
        return np.nan

    lo = min(conf_correct.min(), conf_incorrect.min()) - 1e-8
    hi = max(conf_correct.max(), conf_incorrect.max()) + 1e-8
    lo, hi = max(0, lo), min(1, hi)
    grid = np.linspace(lo, hi, n_points)

    kde_K = _safe_kde(conf_correct)
    kde_N = _safe_kde(conf_incorrect)
    if kde_K is None or kde_N is None:
        return np.nan
    p_K = kde_K(grid)
    p_N = kde_N(grid)
    p_K /= p_K.sum()
    p_N /= p_N.sum()

    bc = np.trapz(np.sqrt(p_K * p_N), grid)
    return float(bc)


def estimate_q_auroc(conf_correct: np.ndarray, conf_incorrect: np.ndarray) -> float:
    """Method D: q = 1 - AUROC.

    Uses confidence as detection score for classifying correct vs incorrect.
    q → 0: confidence perfectly separates correct/incorrect (good)
    q → 0.5: random performance
    """
    if len(conf_correct) < 2 or len(conf_incorrect) < 2:
        return np.nan

    scores = np.concatenate([conf_correct, conf_incorrect])
    labels = np.concatenate([np.ones(len(conf_correct)), np.zeros(len(conf_incorrect))])

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        return np.nan

    q = 1.0 - auroc
    return float(max(0.0, min(1.0, q)))


def compute_all_estimators(conf_correct: np.ndarray, conf_incorrect: np.ndarray) -> dict:
    """Compute all four q^(ℓ) estimators.

    Returns dict with keys: overlap, kl, bhattacharyya, auroc.
    """
    return {
        "overlap": estimate_q_overlap(conf_correct, conf_incorrect),
        "kl": estimate_q_kl(conf_correct, conf_incorrect),
        "bhattacharyya": estimate_q_bhattacharyya(conf_correct, conf_incorrect),
        "auroc": estimate_q_auroc(conf_correct, conf_incorrect),
    }
