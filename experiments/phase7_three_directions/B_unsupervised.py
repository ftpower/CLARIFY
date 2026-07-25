"""Direction B: Unsupervised Truth Discovery (B.1 + B.2).

B.1 — Hidden State Statistical Features (no labels, no GPU generation):
  - L2 norm of last-token hidden state
  - Cosine to prompt-mean hidden state
  - Cosine to global-mean hidden state
  - Neighborhood density (k-NN mean distance)
  - Local intrinsic dimension (TwoNN estimator)

B.2 — Attention Pattern Features (one GPU forward pass, no generation):
  - Attention entropy per head (mean/max across heads)
  - Max attention concentration
  - Evidence-token mass (attention on last-N prompt tokens)

Usage:
    python B_unsupervised.py --n_samples 200 --layer 20
    python B_unsupervised.py --n_samples 100 --layer 20 --skip_attn
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_sys_parent = Path(__file__).parent
for _p in [
    str(_sys_parent.parent / "phase2_entropy"),
    str(_sys_parent.parent / "phase4_generalization"),
    str(_sys_parent.parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction: one forward pass → HS (last + all tokens) + attention pattern
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hs_and_attention(model, tokenizer, samples, device, layer,
                              max_new_tokens=20):
    """Extract hidden states and attention patterns at target layer.

    One forward pass per sample (no extra passes for generation — we generate
    greedily after the hooked pass).

    Returns:
        records: list of dicts with keys:
            h_last: [d] last-token hidden state
            h_prompt_mean: [d] mean over all prompt tokens
            attn_pattern: [n_heads, seq_len] attention pattern at last query pos (or None)
            label: 0/1
            prompt_len: int
    """
    records = []
    correct_count = 0

    for s in tqdm(samples, desc=f"Extract L{layer} HS+Attn"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        prompt_len = tokens.shape[1]

        # ── Build hooks: resid_post (all positions) + attention pattern ──
        residual = {}

        def _resid_hook(act, hook=None):
            # Store ALL token positions (not just last)
            residual["resid_all"] = act[0, :, :].detach()  # [seq, d]
            return act

        def _attn_hook(act, hook=None):
            # Attention pattern: [batch, n_heads, seq_q, seq_k]
            # Store last query position
            residual["attn_pattern"] = act[0, :, -1, :].detach()  # [n_heads, seq_k]
            return act

        fwd_hooks = [
            (f"blocks.{layer}.hook_resid_post", _resid_hook),
            (f"blocks.{layer}.attn.hook_pattern", _attn_hook),
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # ── Store HS ──
        h_all = residual["resid_all"].float().cpu().numpy()  # [seq, d]
        h_last = h_all[-1, :].copy()
        h_prompt_mean = h_all[:prompt_len, :].mean(axis=0)  # mean over prompt tokens

        # ── Store attention pattern ──
        attn_pattern = residual["attn_pattern"].float().cpu().numpy()  # [n_heads, seq_k]

        # ── Generate for label ──
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(max_new_tokens - 1):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")
        if is_correct:
            correct_count += 1

        records.append({
            "h_last": h_last,
            "h_prompt_mean": h_prompt_mean,
            "attn_pattern": attn_pattern,
            "label": 1 if is_correct else 0,
            "prompt_len": prompt_len,
        })

    return records, correct_count


# ═══════════════════════════════════════════════════════════════════════════════
# B.1 Features: Hidden State Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_b1_features(records):
    """Compute B.1 unsupervised features from hidden states.

    All features are computed WITHOUT using labels.
    """
    N = len(records)
    H = np.stack([r["h_last"] for r in records], axis=0)  # [N, d]

    # ── L2 norm ──
    l2_norms = np.linalg.norm(H, axis=1)  # [N]

    # ── Cosine to prompt mean ──
    H_prompt_mean = np.stack([r["h_prompt_mean"] for r in records], axis=0)  # [N, d]
    cos_to_prompt = np.sum(H * H_prompt_mean, axis=1) / (
        np.linalg.norm(H, axis=1) * np.linalg.norm(H_prompt_mean, axis=1) + 1e-10
    )

    # ── Cosine to global mean ──
    global_mean = H.mean(axis=0)  # [d]
    cos_to_global = (H @ global_mean) / (
        np.linalg.norm(H, axis=1) * np.linalg.norm(global_mean) + 1e-10
    )

    # ── Neighborhood density: mean distance to k=5 nearest neighbors ──
    k = min(5, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(H)
    distances, _ = nbrs.kneighbors(H)
    # distances[:, 0] is self-distance (0), use [:, 1:]
    neighborhood_density = distances[:, 1:].mean(axis=1)  # [N], lower = denser

    # ── Local intrinsic dimension (TwoNN estimator) ──
    # TwoNN: for each point, compute μ = r2/r1 where r1,r2 are distances to 1st and 2nd NN.
    # LID = -1 / (log μ). We use the median of LID across points for stability,
    # but we report per-point LID estimates.
    # Need at least 3 points for 2 neighbors
    if N >= 3:
        nbrs2 = NearestNeighbors(n_neighbors=3, metric="cosine").fit(H)
        dists2, _ = nbrs2.kneighbors(H)
        r1 = dists2[:, 1]  # distance to 1st neighbor
        r2 = dists2[:, 2]  # distance to 2nd neighbor
        # Avoid division by zero
        mu = np.maximum(r2, 1e-10) / np.maximum(r1, 1e-10)
        # Filter unreasonable mu values
        valid_mu = (mu > 1.0 + 1e-8) & np.isfinite(mu)
        local_id = np.full(N, np.nan)
        # LID = log(2) / log(mu) — but the standard formula is LID ≈ -1 / mean(log μ)
        # For per-point: -ln(2) / ln(r1/r2)? Let's use the standard: LID = -1 / log2(μ) per point
        # Actually: LID ≈ -log(2) / log(r1/r2) for each point. Or simpler: 1 / (log(r2) - log(r1))
        # Using the canonical formulation: LID_i ≈ -1 / (log(r2_i) - log(r1_i))
        # but this is unstable per-point. Use the distribution median.
        # For per-point we use a smoothed estimate:
        log_mu = np.log(mu)
        local_id[valid_mu] = -1.0 / log_mu[valid_mu]
    else:
        local_id = np.full(N, np.nan)

    return {
        "l2_norm": l2_norms,
        "cos_to_prompt_mean": cos_to_prompt,
        "cos_to_global_mean": cos_to_global,
        "neighborhood_density": neighborhood_density,
        "local_intrinsic_dim": local_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# B.2 Features: Attention Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def compute_b2_features(records):
    """Compute B.2 unsupervised features from attention patterns.

    All features are computed WITHOUT using labels.
    """
    N = len(records)

    attn_entropy_mean = np.zeros(N)
    attn_entropy_max = np.zeros(N)
    attn_concentration = np.zeros(N)
    evidence_mass_last_5 = np.zeros(N)
    evidence_mass_last_10 = np.zeros(N)
    evidence_mass_last_20 = np.zeros(N)

    for i, r in enumerate(records):
        attn = r["attn_pattern"]  # [n_heads, seq_len]
        prompt_len = r["prompt_len"]

        # Focus on prompt tokens only (exclude generated tokens)
        attn_prompt = attn[:, :prompt_len]  # [n_heads, prompt_len]

        # ── Attention entropy per head ──
        # H(head) = -sum(p * log(p)), higher = more uniform = less focused
        eps = 1e-10
        entropies = -np.sum(attn_prompt * np.log(attn_prompt + eps), axis=1)
        # Normalize by max entropy (log(prompt_len)) for comparability
        max_entropy = np.log(prompt_len)
        entropies_norm = entropies / max_entropy
        attn_entropy_mean[i] = entropies_norm.mean()
        attn_entropy_max[i] = entropies_norm.max()

        # ── Max attention concentration ──
        # How peaked is the attention? max over all heads and positions
        attn_concentration[i] = attn_prompt.max()

        # ── Evidence mass on last-N prompt tokens ──
        # Attention mass on the most informative part of the prompt
        # (Usually the question/context is at the end)
        def _mass_last_n(n):
            if prompt_len <= n:
                return attn_prompt[:, :].sum(axis=1).mean()
            return attn_prompt[:, -n:].sum(axis=1).mean()

        evidence_mass_last_5[i] = _mass_last_n(5)
        evidence_mass_last_10[i] = _mass_last_n(10)
        evidence_mass_last_20[i] = _mass_last_n(20)

    return {
        "attn_entropy_mean": attn_entropy_mean,
        "attn_entropy_max": attn_entropy_max,
        "attn_concentration": attn_concentration,
        "evidence_mass_last_5": evidence_mass_last_5,
        "evidence_mass_last_10": evidence_mass_last_10,
        "evidence_mass_last_20": evidence_mass_last_20,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_feature(scores, labels, name):
    """Compute AUROC for a single feature."""
    arr = np.array(scores, dtype=np.float64)
    lab = np.array(labels, dtype=np.int32)
    valid = np.isfinite(arr)
    arr = arr[valid]
    lab = lab[valid]
    n_valid = int(valid.sum())

    if n_valid < 2 or lab.std() == 0:
        return {"name": name, "auroc": float("nan"), "n_valid": n_valid}

    auroc_raw = float(roc_auc_score(lab, arr))
    auroc = max(auroc_raw, 1 - auroc_raw)

    correct_mean = float(arr[lab == 1].mean()) if lab.sum() > 0 else float("nan")
    incorrect_mean = float(arr[lab == 0].mean()) if (lab == 0).sum() > 0 else float("nan")

    return {
        "name": name,
        "auroc": auroc,
        "n_valid": n_valid,
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
    }


def evaluate_all_features(b1_features, b2_features, labels):
    """Evaluate all B.1 and B.2 features."""
    results = []

    # B.1 features
    b1_configs = [
        ("l2_norm", "B.1 L2 norm"),
        ("cos_to_prompt_mean", "B.1 cos to prompt mean"),
        ("cos_to_global_mean", "B.1 cos to global mean"),
        ("neighborhood_density", "B.1 neighborhood density"),
        ("local_intrinsic_dim", "B.1 local intrinsic dim"),
    ]
    for key, name in b1_configs:
        if key in b1_features:
            r = evaluate_feature(b1_features[key], labels, name)
            results.append(r)

    # B.2 features
    if b2_features:
        b2_configs = [
            ("attn_entropy_mean", "B.2 attn entropy mean"),
            ("attn_entropy_max", "B.2 attn entropy max"),
            ("attn_concentration", "B.2 attn concentration"),
            ("evidence_mass_last_5", "B.2 evidence mass (last 5)"),
            ("evidence_mass_last_10", "B.2 evidence mass (last 10)"),
            ("evidence_mass_last_20", "B.2 evidence mass (last 20)"),
        ]
        for key, name in b2_configs:
            if key in b2_features:
                r = evaluate_feature(b2_features[key], labels, name)
                results.append(r)

    results.sort(key=lambda x: (x["auroc"] if not np.isnan(x["auroc"]) else 0), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="B.1 + B.2: Unsupervised Truth Discovery")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to extract HS and attention from")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_attn", action="store_true",
                        help="Skip attention pattern extraction (B.1 only)")
    parser.add_argument("--load_cache", type=str, default=None,
                        help="Load cached extraction from JSON instead of re-extracting")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"B: Unsupervised Truth Discovery (L{args.layer})")
    print(f"  Samples: {args.n_samples}  Skip attn: {args.skip_attn}")
    print(f"{'='*60}\n")

    # ── Load or extract ──
    if args.load_cache:
        print(f"Loading cached extraction from: {args.load_cache}")
        with open(args.load_cache) as f:
            cache = json.load(f)
        records = cache["records"]
        correct_count = cache["correct_count"]
        # Reconstruct numpy arrays
        for r in records:
            r["h_last"] = np.array(r["h_last"])
            r["h_prompt_mean"] = np.array(r["h_prompt_mean"])
            if "attn_pattern" in r and r["attn_pattern"] is not None:
                r["attn_pattern"] = np.array(r["attn_pattern"])
    else:
        print("Loading model & data...")
        t0 = time.time()
        model = load_model(device=device, model_id=args.model)
        tokenizer = model.tokenizer
        samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
        print(f"  Loaded in {time.time()-t0:.0f}s")

        print(f"\nExtracting hidden states + attention at layer {args.layer}...")
        t0 = time.time()
        records, correct_count = extract_hs_and_attention(
            model, tokenizer, samples, device, args.layer,
        )
        print(f"  Extracted in {time.time()-t0:.0f}s")
        print(f"  Correct: {correct_count}/{len(records)} ({correct_count/len(records):.1%})")

    labels = [r["label"] for r in records]

    # ── B.1: Hidden state statistics ──
    print(f"\n{'─'*50}")
    print("B.1: Hidden State Statistical Features")
    print(f"{'─'*50}")
    t0 = time.time()
    b1_features = compute_b1_features(records)
    print(f"  Computed in {time.time()-t0:.1f}s")

    # Print summary stats for each feature
    for key, arr in b1_features.items():
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            print(f"  {key:30s}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                  f"min={valid.min():.4f}, max={valid.max():.4f}")

    # ── B.2: Attention patterns ──
    has_attn = all(r.get("attn_pattern") is not None for r in records)
    b2_features = None
    if has_attn and not args.skip_attn:
        print(f"\n{'─'*50}")
        print("B.2: Attention Pattern Features")
        print(f"{'─'*50}")
        t0 = time.time()
        b2_features = compute_b2_features(records)
        print(f"  Computed in {time.time()-t0:.1f}s")

        for key, arr in b2_features.items():
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                print(f"  {key:30s}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                      f"min={valid.min():.4f}, max={valid.max():.4f}")

    # ── Evaluate ──
    print(f"\n{'─'*50}")
    print("AUROC Results")
    print(f"{'─'*50}")
    all_results = evaluate_all_features(b1_features, b2_features or {}, labels)

    print(f"\n  {'Feature':35s} {'AUROC':>8s}  {'N':>5s}  {'Correct':>10s}  {'Incorrect':>10s}")
    print(f"  {'─'*75}")
    for r in all_results:
        auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "nan"
        cm = f"{r['correct_mean']:10.4f}" if not np.isnan(r.get('correct_mean', float('nan'))) else "       nan"
        im = f"{r['incorrect_mean']:10.4f}" if not np.isnan(r.get('incorrect_mean', float('nan'))) else "       nan"
        print(f"  {r['name']:35s} {auroc_str:>8s}  {r['n_valid']:>5d}  {cm}  {im}")

    # ── Classification of features ──
    max_auroc = max(r["auroc"] for r in all_results if not np.isnan(r["auroc"]))
    best_feature = [r for r in all_results if r["auroc"] == max_auroc][0]
    print(f"\n  Best: {best_feature['name']} = {best_feature['auroc']:.4f}")

    if max_auroc > 0.70:
        print(f"  ✅ SUCCESS: At least one unsupervised feature exceeds 0.70!")
    else:
        print(f"  ❌ All B.1/B.2 features < 0.70 — no pure-unsupervised signal found")

    # ── Save ──
    # Prepare serializable records (convert numpy arrays to lists)
    serializable_records = []
    for r in records:
        sr = {
            "h_last": r["h_last"].tolist(),
            "h_prompt_mean": r["h_prompt_mean"].tolist(),
            "label": r["label"],
            "prompt_len": r["prompt_len"],
        }
        if r.get("attn_pattern") is not None:
            sr["attn_pattern"] = r["attn_pattern"].tolist()
        else:
            sr["attn_pattern"] = None
        serializable_records.append(sr)

    save_path = output_dir / "B_unsupervised.json"
    with open(save_path, "w") as f:
        json.dump({
            "n_samples": len(records),
            "n_correct": correct_count,
            "layer": args.layer,
            "auroc_summary": all_results,
            "best_feature": best_feature,
            "records": serializable_records,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    # ── Also save a compact feature summary (no HS data) ──
    summary_path = output_dir / "B_unsupervised_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_samples": len(records),
            "n_correct": correct_count,
            "layer": args.layer,
            "auroc_summary": all_results,
            "best_feature": best_feature,
            "b1_feature_stats": {
                key: {
                    "mean": float(arr[np.isfinite(arr)].mean()),
                    "std": float(arr[np.isfinite(arr)].std()),
                }
                for key, arr in b1_features.items()
                if np.isfinite(arr).sum() > 0
            },
        }, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"B.1 + B.2 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
