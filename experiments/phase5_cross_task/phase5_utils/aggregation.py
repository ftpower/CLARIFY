"""Multi-token aggregation strategies for per-token feature sequences.

In HellaSwag, features are extracted from a single token position (the last input
token's 4-choice answer distribution). In TriviaQA, features are extracted at
EACH decode step. This module provides aggregation strategies to reduce a sequence
of per-token features to a single scalar (or per-layer vector) for evaluation.

Strategies:
  - last:       use the LAST generated token's features (analog to HellaSwag)
  - mean:       arithmetic mean across all tokens
  - min:        minimum value across tokens
  - max:        maximum value across tokens
  - var:        variance across tokens (measures generation instability)
  - early_mean: mean of the FIRST HALF of tokens
  - late_mean:  mean of the SECOND HALF of tokens
"""

import numpy as np

VALID_STRATEGIES = ["last", "mean", "min", "max", "var", "early_mean", "late_mean"]


def aggregate_features(
    per_token: list[dict],
    feature_keys: list[str] | None = None,
    strategies: list[str] | None = None,
) -> dict:
    """Aggregate per-token features across the generation sequence.

    Handles two types of features:
      - Array-valued: max_p[28], entropy[28] → aggregation per layer → [n_layers] per strategy
      - Scalar-valued: d2_js → aggregation → scalar per strategy

    Args:
        per_token: list of per-step feature dicts from generate_with_per_token_features().
            Each dict has keys: "step", "token_id", "token_text", "max_p", "entropy", "d2_js".
        feature_keys: which keys to aggregate (default: ["max_p", "entropy", "d2_js"]).
        strategies: which strategies to apply (default: all VALID_STRATEGIES).

    Returns:
        Nested dict: {feature_key: {strategy: aggregated_value}}
        Array-valued features: aggregated_value is list[float] of length n_layers.
        Scalar-valued features: aggregated_value is float.
        Returns NaN for all entries if per_token is empty.

    Examples:
        >>> per_token = [{"max_p": [0.1, 0.5], "d2_js": 0.04},
        ...              {"max_p": [0.2, 0.6], "d2_js": 0.06}]
        >>> result = aggregate_features(per_token, feature_keys=["max_p", "d2_js"],
        ...                             strategies=["last", "mean"])
        >>> result["max_p"]["last"]
        [0.2, 0.6]
        >>> result["d2_js"]["mean"]
        0.05
    """
    if feature_keys is None:
        feature_keys = ["max_p", "entropy", "d2_js"]
    if strategies is None:
        strategies = VALID_STRATEGIES

    n_tokens = len(per_token)

    # Determine which keys are array-valued vs scalar-valued from the first token
    if n_tokens == 0:
        return {
            key: {s: float("nan") for s in strategies} for key in feature_keys
        }

    result = {}
    for key in feature_keys:
        sample_val = per_token[0][key]
        is_array = isinstance(sample_val, (list, np.ndarray))

        if is_array:
            # Array-valued: [n_tokens, n_layers]
            values = np.array([t[key] for t in per_token], dtype=np.float64)  # [T, L]
            n_layers_val = values.shape[1]

            half = max(1, n_tokens // 2)
            agg = {}
            for s in strategies:
                if s == "last":
                    agg[s] = values[-1].tolist()
                elif s == "mean":
                    agg[s] = values.mean(axis=0).tolist()
                elif s == "min":
                    agg[s] = values.min(axis=0).tolist()
                elif s == "max":
                    agg[s] = values.max(axis=0).tolist()
                elif s == "var":
                    if n_tokens > 1:
                        agg[s] = values.var(axis=0, ddof=1).tolist()
                    else:
                        agg[s] = [0.0] * n_layers_val
                elif s == "early_mean":
                    agg[s] = values[:half].mean(axis=0).tolist()
                elif s == "late_mean":
                    agg[s] = values[half:].mean(axis=0).tolist()
            result[key] = agg
        else:
            # Scalar-valued: [n_tokens]
            values = np.array([t[key] for t in per_token], dtype=np.float64)  # [T]

            half = max(1, n_tokens // 2)
            agg = {}
            for s in strategies:
                if s == "last":
                    agg[s] = float(values[-1])
                elif s == "mean":
                    agg[s] = float(values.mean())
                elif s == "min":
                    agg[s] = float(values.min())
                elif s == "max":
                    agg[s] = float(values.max())
                elif s == "var":
                    if n_tokens > 1:
                        agg[s] = float(values.var(ddof=1))
                    else:
                        agg[s] = 0.0
                elif s == "early_mean":
                    agg[s] = float(values[:half].mean())
                elif s == "late_mean":
                    agg[s] = float(values[half:].mean())
            result[key] = agg

    return result


def build_feature_vectors(
    per_sample_results: list[dict],
    strategy: str,
    feature_spec: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix from per-sample aggregated features.

    Helper to extract a feature matrix X and label vector y from a list of
    per-sample results, using a specific aggregation strategy.

    Args:
        per_sample_results: list of per-sample dicts from the 5.1 pipeline.
            Each must have "is_correct" and "aggregated" keys (from aggregate_features).
        strategy: which aggregation strategy to use (e.g., "last", "mean").
        feature_spec: dict specifying which features and layers to extract.
            Example: {"max_p": [27], "entropy": [27], "d2_js": None}
            None means scalar feature.
            Default: {"max_p": [27], "entropy": [27], "d2_js": None}.

    Returns:
        X: [N, n_features] feature matrix.
        y: [N] binary labels (1=correct, 0=incorrect).
        feature_names: list of str feature names matching columns of X.
    """
    if feature_spec is None:
        feature_spec = {"max_p": [27], "entropy": [27], "d2_js": None}

    y = np.array([s["is_correct"] for s in per_sample_results], dtype=np.float64)

    # Build feature list
    feature_cols = []
    feature_names = []

    for feat_key, layer_indices in feature_spec.items():
        vals = []
        for sample in per_sample_results:
            agg = sample["aggregated"]
            if layer_indices is None:
                # Scalar feature
                vals.append(agg[feat_key][strategy])
                if len(feature_names) < 1:
                    feature_names.append(f"{feat_key}_{strategy}")
            else:
                # Array-valued: pick specific layers
                for li in layer_indices:
                    v = agg[feat_key][strategy]
                    vals.append(v[li] if isinstance(v, (list, np.ndarray)) else v)
                    if len(feature_names) < len(layer_indices):
                        feature_names.append(f"{feat_key}_L{li}_{strategy}")

        # Reshape: each sample should be one row
        n_feat_for_key = 1 if layer_indices is None else len(layer_indices)
        if len(vals) == n_feat_for_key:
            feature_cols.append(np.array(vals).reshape(-1, 1))
        else:
            feature_cols.append(
                np.array(vals).reshape(len(per_sample_results), n_feat_for_key)
            )

    X = np.concatenate(feature_cols, axis=1)
    return X.astype(np.float64), y, feature_names
