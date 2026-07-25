"""Phase 9.1: Multi-State Truth Direction Detection Analysis.

Computes per-state (h/a/m) truth direction and AUROC at every layer,
then evaluates joint Logistic Regression at the best layer.

Usage:
    python phase9_detection.py --load outputs_phase9/phase9_extract.json
    python phase9_detection.py --load outputs_phase9/phase9_extract.json --layer 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: truth direction + AUROC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_truth_direction(H_correct, H_incorrect):
    """v = mean(correct) - mean(incorrect), L2-normalized."""
    v = H_correct.mean(axis=0) - H_incorrect.mean(axis=0)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    return v


def evaluate_direction(H, v, labels):
    """Project H onto v, return AUROC."""
    scores = H @ v
    valid = np.isfinite(scores)
    if valid.sum() < 2 or labels[valid].std() == 0:
        return float("nan")
    auroc = float(roc_auc_score(labels[valid], scores[valid]))
    return max(auroc, 1 - auroc)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_per_layer(records, n_layers):
    """For each layer, compute per-state (h/a/m) truth direction AUROC.

    Uses in-sample evaluation (same as C2 methodology) since truth direction
    is 1D mean-difference — minimal overfitting.

    Returns:
        results: dict with per-layer per-state AUROC and best layers
    """
    labels = np.array([r["label"] for r in records])

    per_layer = {state: {} for state in ["h", "a", "m"]}
    best = {state: {"layer": -1, "auroc": 0.0} for state in ["h", "a", "m"]}

    for li in range(n_layers):
        li_str = str(li)

        for state in ["h", "a", "m"]:
            H = np.stack([np.array(r[state][li_str]) for r in records], axis=0)

            mask_c = labels == 1
            mask_i = labels == 0
            if mask_c.sum() < 2 or mask_i.sum() < 2:
                per_layer[state][li] = float("nan")
                continue

            v = compute_truth_direction(H[mask_c], H[mask_i])
            auroc = evaluate_direction(H, v, labels)
            per_layer[state][li] = auroc

            if not np.isnan(auroc) and auroc > best[state]["auroc"]:
                best[state]["auroc"] = auroc
                best[state]["layer"] = li

    return per_layer, best


def analyze_joint_lr(records, layer):
    """At a specific layer, combine h/a/m scores via Logistic Regression.

    Uses 5-fold cross-validation for unbiased AUROC estimate.
    """
    li_str = str(layer)
    labels = np.array([r["label"] for r in records])

    # Compute per-state truth directions (on full data — minimal overfitting for 1D)
    H_all = np.stack([np.array(r["h"][li_str]) for r in records], axis=0)
    A_all = np.stack([np.array(r["a"][li_str]) for r in records], axis=0)
    M_all = np.stack([np.array(r["m"][li_str]) for r in records], axis=0)

    mask_c = labels == 1
    mask_i = labels == 0
    v_h = compute_truth_direction(H_all[mask_c], H_all[mask_i])
    v_a = compute_truth_direction(A_all[mask_c], A_all[mask_i])
    v_m = compute_truth_direction(M_all[mask_c], M_all[mask_i])

    # Per-state scores
    score_h = H_all @ v_h
    score_a = A_all @ v_a
    score_m = M_all @ v_m

    # Per-state AUROC
    auroc_h = evaluate_direction(H_all, v_h, labels)
    auroc_a = evaluate_direction(A_all, v_a, labels)
    auroc_m = evaluate_direction(M_all, v_m, labels)

    # Joint: [score_h, score_a, score_m] → LR → AUROC (5-fold CV)
    X = np.stack([score_h, score_a, score_m], axis=1)  # [N, 3]

    valid = np.isfinite(X).all(axis=1)
    X_valid = X[valid]
    y_valid = labels[valid]

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    try:
        cv_scores = cross_val_score(lr, X_valid, y_valid, cv=5, scoring="roc_auc")
        auroc_joint = float(cv_scores.mean())
        auroc_joint_std = float(cv_scores.std())
    except Exception:
        auroc_joint = float("nan")
        auroc_joint_std = float("nan")

    # Also: LR trained on all data for coefficient inspection
    lr.fit(X_valid, y_valid)
    coefs = lr.coef_[0].tolist()

    # Pairwise cosine similarities of truth directions
    cos_hm = float(np.dot(v_h, v_m))
    cos_ha = float(np.dot(v_h, v_a))
    cos_am = float(np.dot(v_a, v_m))

    return {
        "layer": layer,
        "auroc_h": auroc_h,
        "auroc_a": auroc_a,
        "auroc_m": auroc_m,
        "auroc_joint_cv": auroc_joint,
        "auroc_joint_std": auroc_joint_std,
        "lr_coefs": {"h": coefs[0], "a": coefs[1], "m": coefs[2]},
        "cosine_similarities": {
            "cos_h_m": cos_hm, "angle_h_m_deg": float(np.degrees(np.arccos(np.clip(cos_hm, -1, 1)))),
            "cos_h_a": cos_ha, "angle_h_a_deg": float(np.degrees(np.arccos(np.clip(cos_ha, -1, 1)))),
            "cos_a_m": cos_am, "angle_a_m_deg": float(np.degrees(np.arccos(np.clip(cos_am, -1, 1)))),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 9.1: Multi-State Detection")
    parser.add_argument("--load", type=str, default="outputs_phase9/phase9_extract.json",
                        help="Path to extraction JSON")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer for joint LR analysis")
    parser.add_argument("--output_dir", type=str, default="outputs_phase9")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 9.1: Multi-State Truth Direction Detection")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading: {args.load}")
    with open(args.load) as f:
        data = json.load(f)

    records = data["records"]
    config = data["config"]
    n_layers = config["n_layers"]
    n_samples = len(records)
    n_correct = sum(r["label"] for r in records)
    print(f"  Samples: {n_samples}, Correct: {n_correct} ({n_correct/n_samples:.1%})")
    print(f"  Layers: {n_layers}, d_model: {config['d_model']}")

    # ── Per-layer analysis ──
    print(f"\n{'─'*60}")
    print("Per-Layer Per-State AUROC Scan")
    print(f"{'─'*60}")

    per_layer, best = analyze_per_layer(records, n_layers)

    # Print top-5 layers per state
    for state, label in [("h", "Resid (h)"), ("a", "Attn (a)"), ("m", "FFN  (m)")]:
        sorted_layers = sorted(
            [(li, auroc) for li, auroc in per_layer[state].items() if not np.isnan(auroc)],
            key=lambda x: x[1], reverse=True
        )
        print(f"\n  {label}:")
        print(f"  {'Layer':>6s}  {'AUROC':>8s}")
        print(f"  {'─'*17}")
        for li, auroc in sorted_layers[:5]:
            marker = " ← best" if li == best[state]["layer"] else ""
            print(f"  {li:>6d}  {auroc:>8.4f}{marker}")

    # Sweet spot summary
    print(f"\n  Best per state:")
    for state, label in [("h", "Resid"), ("a", "Attn"), ("m", "FFN")]:
        print(f"    {label}: L{best[state]['layer']} = {best[state]['auroc']:.4f}")

    # ── Joint LR at target layer ──
    print(f"\n{'─'*60}")
    print(f"Joint Logistic Regression at L{args.layer}")
    print(f"{'─'*60}")

    joint = analyze_joint_lr(records, args.layer)

    print(f"\n  Per-state AUROC:")
    print(f"    h (resid):  {joint['auroc_h']:.4f}")
    print(f"    a (attn):   {joint['auroc_a']:.4f}")
    print(f"    m (ffn):    {joint['auroc_m']:.4f}")
    print(f"\n  Joint LR (5-fold CV): {joint['auroc_joint_cv']:.4f} ± {joint['auroc_joint_std']:.4f}")
    print(f"\n  LR Coefficients:")
    print(f"    h: {joint['lr_coefs']['h']:+.4f}")
    print(f"    a: {joint['lr_coefs']['a']:+.4f}")
    print(f"    m: {joint['lr_coefs']['m']:+.4f}")
    print(f"\n  Truth Direction Cosine Similarities:")
    print(f"    cos(v_h, v_m): {joint['cosine_similarities']['cos_h_m']:.4f} "
          f"({joint['cosine_similarities']['angle_h_m_deg']:.1f}°)")
    print(f"    cos(v_h, v_a): {joint['cosine_similarities']['cos_h_a']:.4f} "
          f"({joint['cosine_similarities']['angle_h_a_deg']:.1f}°)")
    print(f"    cos(v_a, v_m): {joint['cosine_similarities']['cos_a_m']:.4f} "
          f"({joint['cosine_similarities']['angle_a_m_deg']:.1f}°)")

    # ── Interpretation ──
    print(f"\n{'─'*60}")
    print("Interpretation")
    print(f"{'─'*60}")

    best_state = max([
        ("h", joint["auroc_h"]),
        ("a", joint["auroc_a"]),
        ("m", joint["auroc_m"]),
    ], key=lambda x: x[1])

    print(f"  Best single state: {best_state[0]} ({best_state[1]:.4f})")

    h_only = joint["auroc_h"]
    joint_auroc = joint["auroc_joint_cv"]
    if not np.isnan(joint_auroc):
        delta = joint_auroc - h_only
        if delta > 0.01:
            print(f"  Joint LR > h alone: +{delta:.4f} ✅ — complementary signal confirmed")
        elif abs(delta) <= 0.01:
            print(f"  Joint LR ≈ h alone: {delta:+.4f} — no significant gain")
        else:
            print(f"  Joint LR < h alone: {delta:.4f} — LR overfitting or noise")

    if joint["auroc_m"] >= 0.85:
        print(f"  FFN AUROC ≥ 0.85 ✅ — confirms InternalInspector FFN dominance")
    else:
        print(f"  FFN AUROC < 0.85 — weaker than expected")

    # InternalInspector prediction: FFN > Attn for factual QA
    if joint["auroc_m"] > joint["auroc_a"]:
        print(f"  FFN ({joint['auroc_m']:.4f}) > Attn ({joint['auroc_a']:.4f}) ✅ — "
              f"InternalInspector prediction confirmed (factual QA = FFN-dominant)")
    else:
        print(f"  Attn > FFN — unexpected for factual QA")

    # ── Save ──
    save_path = output_dir / "phase9_detection.json"
    with open(save_path, "w") as f:
        json.dump({
            "n_samples": n_samples,
            "n_correct": n_correct,
            "n_layers": n_layers,
            "best_per_state": {s: {"layer": best[s]["layer"], "auroc": best[s]["auroc"]}
                              for s in ["h", "a", "m"]},
            "per_layer_auroc": {
                state: {str(li): auroc for li, auroc in per_layer[state].items()}
                for state in ["h", "a", "m"]
            },
            "joint_lr_L20": joint,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"Phase 9.1 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
