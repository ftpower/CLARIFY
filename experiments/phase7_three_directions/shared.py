"""Shared infrastructure for Phase 7 three-direction experiments.

Reuses Phase 2 model/data loaders and Phase 5 generation utilities.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Path setup
_sys_parent = Path(__file__).parent
for _p in [
    str(_sys_parent.parent / "phase2_entropy"),
    str(_sys_parent.parent / "phase4_generalization"),
    str(_sys_parent.parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model as _load_model
from src.data_loader import load_triviaqa as _load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Model & data
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_data(
    n_samples: int = 200,
    seed: int = 42,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
):
    """Load model + TriviaQA samples. Returns (model, tokenizer, samples)."""
    model = _load_model(device=device, model_id=model_id)
    tokenizer = model.tokenizer
    samples = _load_triviaqa(n_samples=n_samples, seed=seed)
    return model, tokenizer, samples


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_auroc(
    scores: list[float],
    labels: list[int],
    name: str = "",
    invert: bool = False,
    verbose: bool = True,
) -> dict:
    """Compute AUROC with optional negation (higher score = correct by default).

    Args:
        scores: Per-sample scalar scores.
        labels: Per-sample binary labels (1=correct, 0=incorrect).
        name: Feature name for logging.
        invert: If True, negate scores before computing AUROC.
        verbose: Print result to stdout.

    Returns:
        dict with keys: name, auroc, n_valid, n_correct, correct_mean, incorrect_mean.
    """
    arr = np.array(scores, dtype=np.float64)
    lab = np.array(labels, dtype=np.int32)

    # Filter NaN/Inf
    valid = np.isfinite(arr)
    arr = arr[valid]
    lab = lab[valid]
    n_valid = int(valid.sum())

    if n_valid < 2 or lab.std() == 0:
        result = {"name": name, "auroc": float("nan"), "n_valid": n_valid,
                  "n_correct": int(lab.sum()),
                  "correct_mean": float("nan"), "incorrect_mean": float("nan")}
        if verbose:
            print(f"  {name:35s}: AUROC=nan (n={n_valid})")
        return result

    if invert:
        arr = -arr

    # AUROC: flip if < 0.5 for consistent reporting
    auroc_raw = float(roc_auc_score(lab, arr))
    auroc = max(auroc_raw, 1 - auroc_raw)

    correct_mean = float(arr[lab == 1].mean()) if lab.sum() > 0 else float("nan")
    incorrect_mean = float(arr[lab == 0].mean()) if (lab == 0).sum() > 0 else float("nan")

    result = {
        "name": name,
        "auroc": auroc,
        "n_valid": n_valid,
        "n_correct": int(lab.sum()),
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
    }
    if verbose:
        print(f"  {name:35s}: AUROC={auroc:.4f} (n={n_valid})")
    return result


def evaluate_all(
    results: list[dict],
    labels: list[int],
    feature_configs: list[dict],
) -> list[dict]:
    """Evaluate multiple features at once.

    Args:
        results: List of per-sample dicts containing feature values.
        labels: Per-sample binary labels.
        feature_configs: List of {"key": str, "name": str, "invert": bool}.

    Returns:
        List of AUROC result dicts, sorted by AUROC descending.
    """
    all_results = []
    for cfg in feature_configs:
        scores = [r.get(cfg["key"], float("nan")) for r in results]
        r = evaluate_auroc(
            scores, labels,
            name=cfg.get("name", cfg["key"]),
            invert=cfg.get("invert", False),
        )
        all_results.append(r)

    all_results.sort(key=lambda x: (x["auroc"] if not np.isnan(x["auroc"]) else 0), reverse=True)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Per-token aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_per_token(
    per_token_values: list[list[float]],
    method: str = "early_mean",
) -> float:
    """Aggregate per-token scalar list into a single scalar.

    Args:
        per_token_values: List of lists, outer=token, inner=layers or scalar per token.
        method: One of "last", "mean", "early_mean", "max", "min".

    Returns:
        Aggregated scalar, or nan if input is empty.
    """
    if not per_token_values:
        return float("nan")

    # Flatten if nested (per-token per-layer arrays)
    if per_token_values and isinstance(per_token_values[0], list):
        # Per-token list of lists → aggregate layers first, then tokens
        token_scalars = [np.mean(v) if v else float("nan") for v in per_token_values]
    else:
        token_scalars = [float(v) if v is not None else float("nan") for v in per_token_values]

    token_scalars = [v for v in token_scalars if np.isfinite(v)]
    if not token_scalars:
        return float("nan")

    if method == "last":
        return token_scalars[-1]
    elif method == "mean":
        return float(np.mean(token_scalars))
    elif method == "early_mean":
        n_early = max(1, len(token_scalars) // 3)
        return float(np.mean(token_scalars[:n_early]))
    elif method == "max":
        return float(np.max(token_scalars))
    elif method == "min":
        return float(np.min(token_scalars))
    else:
        raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    results: list[dict],
    auroc_summary: list[dict],
    output_path: str,
    extra: dict | None = None,
):
    """Save per-sample results + AUROC summary to JSON."""
    out = {
        "n_samples": len(results),
        "auroc_summary": auroc_summary,
        "per_sample": results,
    }
    if extra:
        out["extra"] = extra
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {output_path}")


def print_summary(auroc_summary: list[dict]):
    """Print formatted AUROC summary table."""
    print(f"\n  {'Feature':35s} {'AUROC':>8s}  {'N':>5s}")
    print(f"  {'─'*50}")
    for r in auroc_summary:
        auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "nan"
        print(f"  {r['name']:35s} {auroc_str:>8s}  {r['n_valid']:>5d}")
    print()
