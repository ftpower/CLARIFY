"""Phase 3 Detection Methods D2/D3/D4: layer consistency, fragility, residual scan.

D2: Layer-wise consistency score — JS divergence between early/late layer 4-choice softmax.
D3: Fragility score — shallow-layer activation norm anomaly detection.
D4: Residual signal AUROC scan — per-layer statistics (L2 dist, cosine, norm) discrimination.

All three share one model-load + forward-pass phase, then pure CPU analysis.

Usage:
    python main_phase3_detection.py                    # default: 500 samples
    python main_phase3_detection.py --n_samples 200    # quick test
    python main_phase3_detection.py --skip_extract     # use cached states
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Data extraction — residual stream + 4-choice softmax per layer
# ═══════════════════════════════════════════════════════════════════════════


def extract_residual_states(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    cache_path: Path,
) -> dict:
    """Extract per-layer residual stream hidden states and 4-choice softmax.

    Returns dict with keys:
        labels: [N] int32 — 1=correct, 0=incorrect
        p_correct: [N] float32 — P(correct) from final logits
        max_prob: [N, n_layers] float32 — per-layer max probability (logit lens)
        choice_probs: [N, n_layers, 4] float32 — per-layer 4-choice softmax
        norms: [N, n_layers] float32 — per-layer L2 norm of residual state
        states: [N, n_layers, d_model] float16 — residual stream states
    """
    n_samples = len(samples)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    labels = np.zeros(n_samples, dtype=np.int32)
    p_correct_arr = np.zeros(n_samples, dtype=np.float32)
    # We store 29 entries: embed + 27 block outputs + ln_final(L27) = 29 "layers"
    n_total = n_layers + 1
    max_prob = np.zeros((n_samples, n_total), dtype=np.float32)
    choice_probs = np.zeros((n_samples, n_total, 4), dtype=np.float32)
    norms = np.zeros((n_samples, n_total), dtype=np.float32)
    states = np.zeros((n_samples, n_layers, d_model), dtype=np.float16)

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]

    # Build hooks for all resid_post locations
    storage = {}
    hooks = []
    for i in range(n_layers):
        key = f"blocks.{i}.hook_resid_post"
        hooks.append((key, _make_collect_hook(storage, key, last_only=True)))

    for idx, sample in enumerate(tqdm(samples, desc="Extracting states")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # ── Per-layer metrics via logit lens ──
        for li in range(n_layers):
            h = storage[f"blocks.{li}.hook_resid_post"][0, last_pos, :]  # [d_model]
            states[idx, li, :] = h.detach().cpu().to(torch.float16)
            norms[idx, li] = h.norm(p=2).item()

            # Logit lens: project to vocab
            logits_L = h.to(W_U.device) @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)
            probs_L = torch.softmax(logits_L, dim=-1)
            max_prob[idx, li] = probs_L.max().item()

            # 4-choice softmax
            choice_logits = logits_L[letter_tok_ids]
            choice_probs_L = torch.softmax(choice_logits, dim=-1)
            choice_probs[idx, li, :] = choice_probs_L.detach().cpu().to(torch.float32)

        # ── Embedding layer (idx = n_layers, the "layer 0" in old convention) ──
        # We don't hook embed separately; set norms/max_prob/choice for it
        # Actually for the residual states we stored 28 layers (0-27), the analysis
        # treats these as transformer layer outputs. The embedding is not
        # needed for D2/D3/D4 since those are about transformer layer behavior.

        # ── Final prediction ──
        logits_last = logits[0, last_pos, :]
        choice_logits_final = logits_last[letter_tok_ids]
        probs_final = torch.softmax(choice_logits_final.float(), dim=-1)
        pred_idx = probs_final.argmax().item()
        is_correct = letters[pred_idx] == correct_letter
        labels[idx] = int(is_correct)
        p_correct_arr[idx] = probs_final[letters.index(correct_letter)].item()

    return {
        "labels": labels,
        "p_correct": p_correct_arr,
        "max_prob": max_prob,
        "choice_probs": choice_probs,
        "norms": norms,
        "states": states,
    }


def _make_collect_hook(storage: dict, key: str, last_only: bool = True):
    def hook(activation, hook=None):
        if last_only:
            storage[key] = activation.detach()
        else:
            storage[key] = activation.detach()
        return activation

    return hook


# ═══════════════════════════════════════════════════════════════════════════
# D2: Layer-wise consistency score
# ═══════════════════════════════════════════════════════════════════════════


def run_d2_layer_consistency(data: dict, output_dir: Path):
    """JS divergence between early/late layer 4-choice softmax distributions."""
    print("\n" + "=" * 60)
    print("D2: Layer-wise Consistency Score")
    print("=" * 60)

    labels = data["labels"]
    p_correct = data["p_correct"]
    choice_probs = data["choice_probs"]  # [N, n_layers, 4]
    max_prob = data["max_prob"]
    n_layers = choice_probs.shape[1]

    # Knowledge filter
    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_choice = choice_probs[filt_mask]
    filt_maxp = max_prob[filt_mask]
    n_filt = filt_mask.sum()
    print(
        f"Filtered (P>0.3): {n_filt} / {len(labels)} samples, acc={filt_labels.mean():.4f}"
    )

    # ── Part 1: Consistency between specific early/late pairs ──
    # Test L11 vs each late layer, and scan all pairs
    early_candidates = list(range(n_layers))  # 0..27
    late_candidates = list(range(n_layers))  # 0..27

    results = []

    for early in tqdm(early_candidates, desc="D2 scanning pairs"):
        for late in late_candidates:
            if early >= late:
                continue

            p_early = filt_choice[:, early, :]  # [N_filt, 4]
            p_late = filt_choice[:, late, :]  # [N_filt, 4]

            # JS divergence per sample
            m = 0.5 * (p_early + p_late)
            # KL(p||m) = sum(p * log(p/m))
            # Handle zeros by clamping
            eps = 1e-10
            p_early_safe = np.clip(p_early, eps, 1.0)
            p_late_safe = np.clip(p_late, eps, 1.0)
            m_safe = np.clip(m, eps, 1.0)

            kl_early = np.sum(p_early_safe * np.log(p_early_safe / m_safe), axis=1)
            kl_late = np.sum(p_late_safe * np.log(p_late_safe / m_safe), axis=1)
            js = 0.5 * (kl_early + kl_late)

            consistency = 1.0 - js  # high = layers agree
            js_score = js  # high = layers disagree → hallucination risk

            # AUROC of JS as detector
            try:
                auroc_js = roc_auc_score(filt_labels, js_score)
                auroc_consistency = roc_auc_score(filt_labels, consistency)
            except ValueError:
                auroc_js = 0.5
                auroc_consistency = 0.5

            # Joint with max_p at late layer (use max_p across all vocab, not just choices)
            mp_late = filt_maxp[:, late]

            # Simple logistic regression would be ideal, but for speed use
            # the average rank (mean of percentile ranks) — rough joint AUROC
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            X_joint = np.stack([mp_late, js_score], axis=1)
            try:
                lr = LogisticRegression(max_iter=1000)
                joint_auroc = cross_val_score(
                    lr, X_joint, filt_labels, cv=5, scoring="roc_auc"
                ).mean()
            except Exception:
                joint_auroc = float("nan")

            results.append(
                {
                    "early": early,
                    "late": late,
                    "auroc_js": float(auroc_js),
                    "auroc_consistency": float(auroc_consistency),
                    "auroc_joint": float(joint_auroc),
                    "mean_js": float(js.mean()),
                }
            )

    # Best pair
    results.sort(key=lambda r: r["auroc_js"], reverse=True)
    best = results[0]
    print(f"\nBest pair: L{best['early']} vs L{best['late']}")
    print(f"  AUROC(JS)       = {best['auroc_js']:.4f}")
    print(f"  AUROC(consist)  = {best['auroc_consistency']:.4f}")
    print(f"  AUROC(joint mp) = {best['auroc_joint']:.4f}")
    print(f"  Mean JS         = {best['mean_js']:.6f}")

    # Top-10 pairs
    print(f"\nTop-10 layer pairs by JS AUROC:")
    for r in results[:10]:
        print(
            f"  L{r['early']:>2} vs L{r['late']:>2}: JS={r['auroc_js']:.4f}, "
            f"joint={r['auroc_joint']:.4f}, mean_js={r['mean_js']:.6f}"
        )

    # ── Part 2: JS monotonicity check (does JS always increase with depth gap?) ──
    # Check if JS(L0, L) monotonically increases with L
    l0_pairs = [r for r in results if r["early"] == 0]
    l0_pairs.sort(key=lambda r: r["late"])
    mono = all(
        l0_pairs[i]["mean_js"] <= l0_pairs[i + 1]["mean_js"]
        for i in range(len(l0_pairs) - 1)
    )
    print(f"\nJS(L0, *) monotonic with depth: {mono}")

    # ── Part 3: Does max_p alone beat the best JS? ──
    mp_late_all = np.array(
        [
            filt_maxp[:, late]
            for _, late, _, _, _, _ in [
                (r["early"], r["late"], 0, 0, 0, 0) for r in results[:1]
            ]
        ]
    ).squeeze()
    # Actually recalculate: best max_p AUROC in filtered set
    best_mp_auroc = 0.0
    best_mp_layer = -1
    for li in range(n_layers):
        auc = roc_auc_score(filt_labels, filt_maxp[:, li])
        if auc > best_mp_auroc:
            best_mp_auroc = auc
            best_mp_layer = li
    print(f"Best max_p in filtered set: L{best_mp_layer} AUROC={best_mp_auroc:.4f}")

    # ── Part 4: Full-set evaluation ──
    full_choice = choice_probs
    full_maxp = max_prob
    full_labels = labels

    # Best JS pair on full set
    best_early, best_late = best["early"], best["late"]
    p_e_full = full_choice[:, best_early, :]
    p_l_full = full_choice[:, best_late, :]
    eps = 1e-10
    m_full = 0.5 * (p_e_full + p_l_full)
    kl_e_full = np.sum(
        np.clip(p_e_full, eps, 1.0)
        * np.log(np.clip(p_e_full, eps, 1.0) / np.clip(m_full, eps, 1.0)),
        axis=1,
    )
    kl_l_full = np.sum(
        np.clip(p_l_full, eps, 1.0)
        * np.log(np.clip(p_l_full, eps, 1.0) / np.clip(m_full, eps, 1.0)),
        axis=1,
    )
    js_full = 0.5 * (kl_e_full + kl_l_full)

    auroc_js_full = roc_auc_score(full_labels, js_full)
    best_mp_full = 0.0
    for li in range(n_layers):
        best_mp_full = max(best_mp_full, roc_auc_score(full_labels, full_maxp[:, li]))
    print(
        f"\nFull set: best JS AUROC = {auroc_js_full:.4f}, best max_p AUROC = {best_mp_full:.4f}"
    )

    # Save
    out = {
        "best_pair": {"early": best["early"], "late": best["late"]},
        "best_auroc_js": best["auroc_js"],
        "best_auroc_joint": best["auroc_joint"],
        "best_maxp_auroc_filtered": best_mp_auroc,
        "best_maxp_layer_filtered": best_mp_layer,
        "full_set_js_auroc": float(auroc_js_full),
        "full_set_maxp_auroc": float(best_mp_full),
        "js_monotonic_with_depth": mono,
        "top_pairs": results[:20],
    }
    with open(output_dir / "d2_consistency_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {output_dir / 'd2_consistency_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# D3: Fragility score — shallow-layer activation norm anomaly
# ═══════════════════════════════════════════════════════════════════════════


def run_d3_fragility(data: dict, output_dir: Path):
    """Shallow-layer L2 norm anomalies as hallucination detector."""
    print("\n" + "=" * 60)
    print("D3: Fragility Score (Activation Norm Anomaly)")
    print("=" * 60)

    labels = data["labels"]
    p_correct = data["p_correct"]
    norms = data["norms"]  # [N, n_layers]
    n_layers = norms.shape[1]

    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_norms = norms[filt_mask]
    n_filt = filt_mask.sum()

    # ── Per-layer norm AUROC ──
    print("\nPer-layer L2 norm AUROC (filtered):")
    best_norm_auroc = 0.0
    best_norm_layer = -1
    norm_aurocs = []
    for li in range(n_layers):
        try:
            auc = roc_auc_score(filt_labels, filt_norms[:, li])
        except ValueError:
            auc = float("nan")
        norm_aurocs.append(auc)
        if not np.isnan(auc) and auc > best_norm_auroc:
            best_norm_auroc = auc
            best_norm_layer = li
        if li <= 3 or li >= n_layers - 4:
            print(f"  L{li}: {auc:.4f}")

    print(f"  ... (best: L{best_norm_layer} = {best_norm_auroc:.4f})")

    # ── Fragility score: shallow layers (L0-L3) norm anomaly ──
    shallow_layers = [0, 1, 2, 3]
    medians = [np.median(filt_norms[:, li]) for li in shallow_layers]
    fragility = np.zeros(n_filt)
    for li in shallow_layers:
        fragility += filt_norms[:, li] / medians[shallow_layers.index(li)]

    try:
        auroc_fragility = roc_auc_score(filt_labels, fragility)
    except ValueError:
        auroc_fragility = 0.5
    print(f"\nFragility (L0-L3 norm/median sum): AUROC = {auroc_fragility:.4f}")

    # Joint with max_p
    filt_maxp = data["max_prob"][filt_mask]
    best_mp = np.max(
        [roc_auc_score(filt_labels, filt_maxp[:, li]) for li in range(n_layers)]
    )
    print(f"Best max_p (filtered): {best_mp:.4f}")

    # Try joint LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.stack([filt_maxp[:, best_norm_layer], fragility], axis=1)
    try:
        joint_auc = cross_val_score(
            LogisticRegression(max_iter=1000), X, filt_labels, cv=5, scoring="roc_auc"
        ).mean()
        print(f"Joint (max_p + fragility): AUROC = {joint_auc:.4f}")
    except Exception as e:
        joint_auc = float("nan")
        print(f"Joint LR failed: {e}")

    # Full set
    full_labels = labels
    full_norms = norms
    full_maxp = data["max_prob"]
    fragility_full = np.zeros(len(labels))
    for li in shallow_layers:
        m = np.median(full_norms[:, li])
        fragility_full += full_norms[:, li] / m
    auroc_frag_full = roc_auc_score(full_labels, fragility_full)
    best_mp_full = np.max(
        [roc_auc_score(full_labels, full_maxp[:, li]) for li in range(n_layers)]
    )
    print(
        f"\nFull set: fragility AUROC = {auroc_frag_full:.4f}, best max_p = {best_mp_full:.4f}"
    )

    out = {
        "per_layer_norm_auroc": [
            float(a) if not np.isnan(a) else None for a in norm_aurocs
        ],
        "best_norm_layer": best_norm_layer,
        "best_norm_auroc": float(best_norm_auroc),
        "fragility_auroc_filtered": float(auroc_fragility),
        "fragility_auroc_full": float(auroc_frag_full),
        "joint_auroc_filtered": float(joint_auc) if not np.isnan(joint_auc) else None,
        "best_maxp_filtered": float(best_mp),
        "best_maxp_full": float(best_mp_full),
    }
    with open(output_dir / "d3_fragility_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {output_dir / 'd3_fragility_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# D4: Residual signal AUROC scan
# ═══════════════════════════════════════════════════════════════════════════


def run_d4_residual_scan(data: dict, output_dir: Path):
    """Per-layer scan: which statistic at which layer best discriminates correct/incorrect."""
    print("\n" + "=" * 60)
    print("D4: Residual Signal AUROC Scan")
    print("=" * 60)

    labels = data["labels"]
    p_correct = data["p_correct"]
    states = data["states"]  # [N, n_layers, d_model] float16
    n_layers = states.shape[1]

    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    filt_states = states[filt_mask].astype(np.float32)
    n_filt = filt_mask.sum()

    print(
        f"Filtered (P>0.3): {n_filt} / {len(labels)} samples, acc={filt_labels.mean():.4f}"
    )

    # ── Per-layer per-statistic AUROC ──
    stats_names = ["l2_to_centroid", "cosine_to_centroid", "l2_norm"]
    results = {name: np.zeros(n_layers) for name in stats_names}

    # Precompute correct centroid per layer
    correct_mask = filt_labels == 1
    n_correct = correct_mask.sum()
    print(f"  Correct in filtered: {n_correct}, Incorrect: {n_filt - n_correct}")
    centroids = np.zeros((n_layers, states.shape[2]), dtype=np.float32)
    for li in range(n_layers):
        if n_correct > 0:
            centroids[li] = filt_states[correct_mask, li, :].mean(axis=0)

    for li in tqdm(range(n_layers), desc="D4 scanning layers"):
        h = filt_states[:, li, :]  # [N_filt, d_model]

        # L2 distance to correct centroid
        diff_l2 = h - centroids[li]  # [N_filt, d_model]
        l2_dist = np.sqrt(np.maximum((diff_l2**2).sum(axis=1), 0.0))
        try:
            results["l2_to_centroid"][li] = roc_auc_score(filt_labels, l2_dist)
        except ValueError:
            results["l2_to_centroid"][li] = 0.5

        # Cosine distance to correct centroid
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        c_norm = centroids[li] / (np.linalg.norm(centroids[li]) + 1e-8)
        cos_sim = np.clip(h_norm @ c_norm, -1.0, 1.0)  # [N_filt]
        cos_dist = 1.0 - cos_sim
        try:
            results["cosine_to_centroid"][li] = roc_auc_score(filt_labels, cos_dist)
        except ValueError:
            results["cosine_to_centroid"][li] = 0.5

        # L2 norm
        l2_norm = np.linalg.norm(h, axis=1)
        try:
            results["l2_norm"][li] = roc_auc_score(filt_labels, l2_norm)
        except ValueError:
            results["l2_norm"][li] = 0.5

    # ── Report ──
    print(f"\n{'Layer':<6} {'L2→cent':>10} {'Cos→cent':>10} {'L2 norm':>10}")
    print("-" * 40)
    best_overall = {"stat": "", "layer": -1, "auroc": 0.0}
    for li in range(n_layers):
        vals = {k: results[k][li] for k in stats_names}
        print(
            f"{li:<6} {vals['l2_to_centroid']:>10.4f} {vals['cosine_to_centroid']:>10.4f} {vals['l2_norm']:>10.4f}"
        )
        for stat in stats_names:
            if vals[stat] > best_overall["auroc"]:
                best_overall = {"stat": stat, "layer": li, "auroc": vals[stat]}

    # Best max_p for reference
    filt_maxp = data["max_prob"][filt_mask]
    best_mp_auroc = 0.0
    best_mp_layer = -1
    for li in range(n_layers):
        auc = roc_auc_score(filt_labels, filt_maxp[:, li])
        if auc > best_mp_auroc:
            best_mp_auroc = auc
            best_mp_layer = li

    print(
        f"\nBest overall: {best_overall['stat']} at L{best_overall['layer']} = {best_overall['auroc']:.4f}"
    )
    print(f"Best max_p:   L{best_mp_layer} = {best_mp_auroc:.4f}")
    print(f"Delta: {best_overall['auroc'] - best_mp_auroc:+.4f}")

    # Top layers per stat
    for stat in stats_names:
        sorted_layers = sorted(
            [(li, results[stat][li]) for li in range(n_layers)],
            key=lambda x: x[1],
            reverse=True,
        )
        print(
            f"\n{stat} — top 5 layers: {[(f'L{li}', f'{v:.4f}') for li, v in sorted_layers[:5]]}"
        )

    # Variance ratio analysis (correct vs incorrect group)
    print("\n--- Variance ratio (correct/incorrect) per layer ---")
    for li in range(n_layers):
        h_corr = filt_states[correct_mask, li, :]
        h_incorr = filt_states[~correct_mask, li, :]
        var_corr = h_corr.var(axis=0).mean()  # mean variance across dims
        var_incorr = h_incorr.var(axis=0).mean()
        ratio = var_corr / (var_incorr + 1e-8)
        if li <= 3 or li >= n_layers - 4 or li == 11 or li == 15:
            print(
                f"  L{li}: var_corr={var_corr:.6f}, var_incorr={var_incorr:.6f}, ratio={ratio:.4f}"
            )

    out = {
        "filtered_n": n_filt,
        "filtered_acc": float(filt_labels.mean()),
        "per_layer": {
            "l2_to_centroid": [float(v) for v in results["l2_to_centroid"]],
            "cosine_to_centroid": [float(v) for v in results["cosine_to_centroid"]],
            "l2_norm": [float(v) for v in results["l2_norm"]],
        },
        "best_overall": best_overall,
        "best_maxp": {"layer": best_mp_layer, "auroc": float(best_mp_auroc)},
        "delta": float(best_overall["auroc"] - best_mp_auroc),
    }
    with open(output_dir / "d4_residual_scan_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'd4_residual_scan_results.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Loading HellaSwag ({args.n_samples} samples)...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── Load model & extract ──
    state_cache = output_dir / "phase3_residual_states.npz"

    if args.skip_extract and state_cache.exists():
        print(f"Loading cached states from {state_cache}")
        cached = np.load(state_cache, allow_pickle=True)
        data = {
            "labels": cached["labels"],
            "p_correct": cached["p_correct"],
            "max_prob": cached["max_prob"],
            "choice_probs": cached["choice_probs"],
            "norms": cached["norms"],
            "states": cached["states"],
        }
    else:
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
        print(f"Letter token IDs: {letter_ids}")

        data = extract_residual_states(model, samples, letter_ids, state_cache)

        # Save cache
        np.savez_compressed(
            state_cache,
            labels=data["labels"],
            p_correct=data["p_correct"],
            max_prob=data["max_prob"],
            choice_probs=data["choice_probs"],
            norms=data["norms"],
            states=data["states"],
        )
        print(f"Cached states to {state_cache}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Quick stats ──
    acc = data["labels"].mean()
    n_correct = data["labels"].sum()
    n_total = len(data["labels"])
    n_filt = (data["p_correct"] > 0.3).sum()
    acc_filt = data["labels"][data["p_correct"] > 0.3].mean()
    print(f"\nFull: {n_correct}/{n_total} = {acc:.4f}")
    print(f"Filtered (P>0.3): {n_filt} samples, acc={acc_filt:.4f}")

    # ── Run D2, D3, D4 ──
    run_d2_layer_consistency(data, output_dir)
    run_d3_fragility(data, output_dir)
    run_d4_residual_scan(data, output_dir)

    print("\n" + "=" * 60)
    print("Phase 3 Detection (D2/D3/D4) — Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
