"""Subspace/geometry-aware direction intervention — CLEAN RE-RUN (protocol review).

Paradigm under review (old P1 Part B1, phase4): fit a truth direction at a middle
layer from calibration hidden states (mean(h_wrong) - mean(h_correct)), optionally
regularize it with the calibration PCA subspace projector (K @ K^T @ d), and steer
`blocks.{L}.hook_resid_post` during the FIRST forward pass of greedy generation.
The old script also measured principal angles between train/val PCA subspaces as a
geometric diagnostic — kept here (cal vs val, both fit-side).

Why a clean re-run: the old script
  (a) selected λ/mode/best-direction-type on the very set it reported (eval = test),
  (b) used HellaSwag letter-argmax labels plus a p(correct)>0.3 ad-hoc knowledge
      filter instead of rank<=50 + check_correct_exact,
  (c) never enforced a 1024-token window.
Per docs/protocol/evaluation-protocol.md §3 (8-point checklist) the old numbers are
pre-screened as contaminated and must not enter the thesis.

Split design (protocol rule 5 — no test-side parameter choice):
  cal  — --n_dir samples, seed --seed (default 42):  fit direction + PCA basis.
  val  — --n_eval samples, seed --seed_val (default seed+1): select (mode, λ).
  test — --n_test samples, seed --seed_test (default 123): REPORT ONLY.
All splits come from load_triviaqa(n_samples=..., seed=...) with different shuffle
seeds (disjoint slices); same batch 口径 as the TLDC line.

Knowledge 口径 (identical to experiments/lin_theory/validate_s14_tldc.py):
  rank = 1-indexed rank of the first true-answer token in the model's FINAL logits
  at the last prompt position (get_y_true_rank, +1 fix); rank <= --rank_threshold
  (50) → know; correct = check_correct_exact (word-boundary, ≤3-char alias skip).
  --layer_early (20) is used ONLY for an early-exit diagnostic rank; the early-exit
  logits are lens-style (ln_final @ W_U) and are never used for predictions
  (protocol rule 5: lens 仅参考).

Theory: docs/theory/theory-intervention-failure.md §1.2.1 (operational proxy) +
docs/protocol/evaluation-protocol.md §3-4.

Usage (1.7B local, first seed):
    python experiments/phase4_generalization/main_subspace_intervention.py \
        --n_dir 300 --n_eval 200 --model Qwen/Qwen3-1.7B --layers 11 \
        --k_pca 64 --lam 0.3 0.5 1.0 --seed 42 --n_test 300 --seed_test 123
Second seed for the double-seed verdict (reuses cached cal/val states):
    python experiments/phase4_generalization/main_subspace_intervention.py \
        --n_dir 300 --n_eval 200 --model Qwen/Qwen3-1.7B --layers 11 \
        --k_pca 64 --lam 0.3 0.5 1.0 --seed 42 --n_test 300 --seed_test 456 \
        --skip_collect
"""

import argparse
import gc
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct_exact

MAX_PROMPT_TOKENS = 1024  # unified window (protocol rule 4); keep TAIL (Question)
MIN_VAL_KW = 10  # val KW samples required to select λ by KW Δ (else All Δ)

SUBSET_ORDER = ["know_wrong", "know_correct", "dont_know", "all"]


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol helpers (identical semantics to experiments/lin_theory/validate_s14_tldc.py
# and experiments/lin_theory/common.py — copied here so this script has no runtime
# dependency on files that are being concurrently edited in lin_theory/).
# ═══════════════════════════════════════════════════════════════════════════════


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% CI for a binomial proportion (Clopper-Pearson), pure Python.

    Same implementation as validate_s14_tldc.py. Returns (lo, hi).
    """
    from math import comb

    if n == 0:
        return (0.0, 0.0)
    if k == 0:
        return (0.0, 1 - (alpha / 2) ** (1.0 / n))
    if k == n:
        return ((alpha / 2) ** (1.0 / n), 1.0)

    def _cdf_le_k(p):
        # P(X <= k) for X ~ Binomial(n, p)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    def _sf_ge_k(p):
        # P(X >= k) = 1 - P(X <= k-1); increasing in p
        return 1.0 - sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k))

    # lower bound: solve P(X >= k) = alpha/2 (increasing fn)
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        if _sf_ge_k(m) < alpha / 2:
            a = m
        else:
            b = m
    lo = (a + b) / 2
    # upper bound: solve P(X <= k) = alpha/2 (decreasing fn)
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        if _cdf_le_k(m) > alpha / 2:
            a = m
        else:
            b = m
    hi = (a + b) / 2
    return (lo, hi)


def _pct(x, signed=False):
    """None-safe percentage formatting（2026-09-21：空子集 rate/delta 为 None，
    旧报告段直接 `:+.1%` → TypeError 崩溃；CPU 冒烟实测）。"""
    if x is None:
        return "N/A"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def get_first_answer_token_id(tokenizer, answers):
    """First token ID of the first non-empty answer alias (leading-space).

    Same as common.py: the model generates answer tokens with a leading space
    (prompt ends with "Answer:"), so encode " " + answer.
    """
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


def get_y_true_rank(logits, y_true_id):
    """Rank 1 = highest probability (1-indexed; top-50 = ranks 1..50).

    Same as validate_s14_tldc.py: flattens to 1-D so the nonzero()[0] index is
    the rank position (historical 2-D bug made every rank collapse to 1).
    """
    # 2026-09-21 修复：early-exit logits 形状为 [1, vocab]（dim==2），旧判定 dim()>1
    # 会走 [0,-1,:] → IndexError（冒烟测试实测）。统一先降到 1-D。
    row = logits[0, -1, :].float() if logits.dim() > 2 else logits.float()
    row = row.reshape(-1)
    sorted_ids = row.argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item() + 1
    return rank


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Lens-style early-exit logits (ln_final + W_U) — DIAGNOSTIC RANK ONLY.

    Never used for predictions (protocol rule 5: lens 仅参考). Stays in float16
    to avoid OOM from W_U.float().
    """
    dtype = next(ln_final.parameters()).dtype
    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def greedy_generate(model, tokenizer, prompt, device, fwd_hooks=None, max_new=20):
    """Greedy generation with optional forward hooks on the FIRST step only.

    Same semantics as common.greedy_generate: prompt truncated to keep the TAIL
    (Question); hooks apply to the first forward pass; continuation steps run the
    clean model. The intervention paradigm (single-step direction steering at
    layer L before the first answer token) matches the old HellaSwag script,
    which ran exactly one hooked forward pass.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > MAX_PROMPT_TOKENS:
        tokens = tokens[:, -MAX_PROMPT_TOKENS:]  # keep TAIL (Question) — Critical 1

    hooks = fwd_hooks if fwd_hooks else []
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]

    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    return tokenizer.decode(gids).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════════════════


def make_save_hook(storage: dict, key: str):
    """Capture hook: stores a detached copy (protocol rule 8 — no grad leak)."""

    def hook(act, hook=None):
        storage[key] = act.detach()
        return act

    return hook


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    """Project the residual onto/against `direction` with strength lam.

    direction has requires_grad=False by construction (built from detached
    states), so the intervention path carries no gradient.
    """
    sign = -1.0 if mode == "subtract" else 1.0

    def hook(act, hook=None):
        d = direction.to(act.dtype).to(act.device)
        proj = act @ d  # [1, T]
        return act + sign * lam * proj.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)

    return hook


# ═══════════════════════════════════════════════════════════════════════════════
# Split collection + labeling
# ═══════════════════════════════════════════════════════════════════════════════


def collect_split(model, tokenizer, samples, layers, layer_early, device, split_name):
    """Collect per-sample hidden states + knowability features for one split.

    For every sample: truncated prompt (tail), resid_post at each `layer` (last
    position, detached), FINAL logits (real model output) → rank_final,
    lens-style early-exit rank at layer_early (diagnostic only), greedy baseline
    generation, and exact-match label.

    Returns (entries, n_skipped). Entries carry "h" only for the requested
    layers (pass layers=[] for the test split, which is report-only).
    """
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    ln_final = model.ln_final

    entries = []
    n_skipped = 0
    hook_layers = sorted(set(layers) | {layer_early})

    for i, s in enumerate(tqdm(samples, desc=f"Collect {split_name}", leave=False)):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            n_skipped += 1
            continue

        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > MAX_PROMPT_TOKENS:
            tokens = tokens[:, -MAX_PROMPT_TOKENS:]  # keep TAIL (Question)
        last_pos = tokens.shape[1] - 1

        storage = {}
        hooks = [
            (f"blocks.{L}.hook_resid_post", make_save_hook(storage, str(L)))
            for L in hook_layers
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        rank_final = get_y_true_rank(logits, y_true_id)  # 1-indexed, real logits
        l_early = compute_early_exit_logits(
            storage[str(layer_early)][0, -1:, :], ln_final, W_U, b_U
        )
        rank_early = get_y_true_rank(l_early, y_true_id)  # diagnostic only

        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct_exact(gen_text, s["answers"])  # exact label

        entries.append(
            {
                "sample_id": i,
                "prompt": prompt,
                "answers": s["answers"],
                "question": s["question"][:80],
                "y_true_id": y_true_id,
                "rank_final": rank_final,
                "rank_early": rank_early,
                "is_correct": bool(is_correct),
                "pred_base": gen_text[:80],
                "h": {L: storage[str(L)][0, last_pos, :].cpu() for L in layers},
            }
        )

    return entries, n_skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Direction + PCA fitting (CAL ONLY — test is never touched here)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_layer_models(cal_entries, val_entries, layer, k_pca):
    """Fit mean-diff direction + PCA bases on CAL states; PCA on VAL is diagnostic.

    Returns None when the layer cannot be fit (no correct/incorrect cal samples
    or no val states for the diagnostic).
    """
    h_corr = [e["h"][layer] for e in cal_entries if e["is_correct"]]
    h_inc = [e["h"][layer] for e in cal_entries if not e["is_correct"]]
    if len(h_corr) == 0 or len(h_inc) == 0:
        return None

    h_corr = torch.stack(h_corr).float()
    h_inc = torch.stack(h_inc).float()
    d_raw = h_inc.mean(dim=0) - h_corr.mean(dim=0)
    d_raw = d_raw / (d_raw.norm() + 1e-8)

    cal_all = torch.cat([h_corr, h_inc], dim=0)
    val_h = [e["h"][layer] for e in val_entries]
    if not val_h:
        return None
    val_all = torch.stack(val_h).float()

    d_model = cal_all.shape[1]
    k_actual = min(k_pca, cal_all.shape[0], val_all.shape[0], d_model)
    if k_actual < 1:
        return None

    X_cal = cal_all.numpy().astype(np.float64)
    X_val = val_all.numpy().astype(np.float64)
    pca_cal = PCA(n_components=k_actual).fit(X_cal)
    pca_val = PCA(n_components=k_actual).fit(X_val)
    K_cal = pca_cal.components_.T  # [d_model, k_actual]
    K_val = pca_val.components_.T
    angles = subspace_angles(K_cal, K_val)  # [k_actual], radians

    # Aligned direction: project raw direction into the CAL PCA subspace
    K_cal_t = torch.from_numpy(K_cal).float()
    d_aligned = K_cal_t @ (K_cal_t.T @ d_raw)
    d_aligned = d_aligned / (d_aligned.norm() + 1e-8)

    return {
        "d_raw": d_raw,
        "d_aligned": d_aligned,
        "cos_raw_aligned": float((d_raw * d_aligned).sum()),
        "k_actual": k_actual,
        "principal_angles_deg": [float(a * 180.0 / np.pi) for a in angles],
        "max_angle_deg": float(np.max(angles) * 180.0 / np.pi),
        "mean_angle_deg": float(np.mean(angles) * 180.0 / np.pi),
        "cal_pca_explained": float(pca_cal.explained_variance_ratio_.sum()),
        "val_pca_explained": float(pca_val.explained_variance_ratio_.sum()),
        "n_cal_correct": int(len(h_corr)),
        "n_cal_incorrect": int(len(h_inc)),
        "n_val_states": int(val_all.shape[0]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# KW/KC/DK classification + results assembly
# ═══════════════════════════════════════════════════════════════════════════════


def assign_subsets(entries, rank_threshold):
    """In-place: know = rank_final <= threshold; correct = exact (TLDC 口径)."""
    for e in entries:
        if e["rank_final"] <= rank_threshold:
            e["subset"] = "know_correct" if e["is_correct"] else "know_wrong"
        else:
            e["subset"] = "dont_know"
    return entries


def baseline_subset_stats(entries):
    """Per-subset baseline stats from the no-intervention greedy labels."""
    stats = {}
    for s in SUBSET_ORDER:
        sel = entries if s == "all" else [e for e in entries if e["subset"] == s]
        n = len(sel)
        c = sum(1 for e in sel if e["is_correct"])
        stats[s] = {
            "correct": c,
            "total": n,
            "rate": (c / n) if n else None,
        }
    return stats


def evaluate_intervention(
    model, tokenizer, entries, direction, layer, lam, mode, device, desc=""
):
    """Run the intervention generation for every entry; return subset counts.

    counts: {subset: [n_correct, n_total]}; per_sample: {sample_id: {pred, is_correct}}.
    """
    hook_fn = make_projection_hook(direction, lam, mode)
    hook_pt = f"blocks.{layer}.hook_resid_post"

    counts = {s: [0, 0] for s in SUBSET_ORDER}
    per_sample = {}

    for e in tqdm(entries, desc=desc, leave=False):
        gen_text = greedy_generate(
            model, tokenizer, e["prompt"], device, fwd_hooks=[(hook_pt, hook_fn)]
        )
        ok = check_correct_exact(gen_text, e["answers"])
        counts[e["subset"]][1] += 1
        counts["all"][1] += 1
        if ok:
            counts[e["subset"]][0] += 1
            counts["all"][0] += 1
        per_sample[e["sample_id"]] = {"pred": gen_text[:80], "is_correct": bool(ok)}

    return counts, per_sample


def build_subset_results(counts, baseline_stats):
    """Per-subset {correct, total, rate, delta, ci95} vs the baseline stats.

    CI is Clopper-Pearson 95% on the post-intervention rate (template 口径).
    Δ is defined-value subtraction (baseline is the no-intervention greedy run,
    not a re-sampled proportion).
    """
    out = {}
    for s in SUBSET_ORDER:
        c, t = counts[s]
        if t == 0:
            out[s] = {"correct": 0, "total": 0, "rate": None, "delta": None, "ci95": None}
            continue
        rate = c / t
        bl = baseline_stats[s]["rate"]
        delta = rate - bl if bl is not None else None
        lo, hi = clopper_pearson(c, t)
        out[s] = {
            "correct": c,
            "total": t,
            "rate": float(rate),
            "delta": float(delta) if delta is not None else None,
            "ci95": [float(lo), float(hi)],
        }
    return out


def select_config(sweep, val_kw_n):
    """Select (mode, λ) on VAL only: KW Δ if val KW >= MIN_VAL_KW, else All Δ.

    Sweep entries are ordered (mode, ascending λ); max() keeps the first on ties
    → prefers smaller λ. Returns (selected_dict, criterion).
    """
    if val_kw_n >= MIN_VAL_KW:

        def key(r):
            d = r["results"]["know_wrong"]["delta"]
            return d if d is not None else float("-inf")

        criterion = "val_kw_delta"
    else:

        def key(r):
            d = r["results"]["all"]["delta"]
            return d if d is not None else float("-inf")

        criterion = "val_all_delta"

    return max(sweep, key=key), criterion


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Subspace alignment intervention — clean protocol re-run"
    )
    parser.add_argument("--n_dir", type=int, default=300,
                        help="Calibration samples (direction + PCA basis)")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Validation samples ((mode, λ) selection)")
    parser.add_argument("--n_test", type=int, default=300,
                        help="Test samples (report only)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' falls back to CPU when unavailable; 'auto' = autodetect")
    parser.add_argument("--layers", type=int, nargs="+", default=[11])
    parser.add_argument("--k_pca", type=int, default=64)
    parser.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.5, 1.0])
    parser.add_argument("--modes", type=str, nargs="+", default=["subtract", "add"],
                        choices=["subtract", "add"],
                        help="扫描的干预模式（默认两种；只跑 subtract 可把 val 扫描时间减半）")
    parser.add_argument("--layer_early", type=int, default=20,
                        help="Early-exit layer for the DIAGNOSTIC rank only (TLDC parity)")
    parser.add_argument("--rank_threshold", type=int, default=50,
                        help="rank <= threshold → know (1-indexed, TLDC 口径)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: experiments/outputs/phase4_subspace_review/")
    parser.add_argument("--seed", type=int, default=42, help="Calibration seed")
    parser.add_argument("--seed_val", type=int, default=None,
                        help="Val seed (default: seed + 1, disjoint shuffle slice)")
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--skip_collect", action="store_true",
                        help="Load cached cal/val hidden states")
    args = parser.parse_args()

    if args.seed_val is None:
        args.seed_val = args.seed + 1

    device = args.device
    if device == "auto" or (device == "cuda" and not torch.cuda.is_available()):
        if device == "cuda":
            print("WARNING: CUDA not available → falling back to CPU")
        device = "cpu"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (_SCRIPT_DIR.parent / "outputs" / "phase4_subspace_review")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Subspace/geometry intervention — clean protocol re-run")
    print(f"  Model: {args.model} | layers: {args.layers} | k_pca={args.k_pca}")
    print(f"  Splits: cal(n={args.n_dir}, seed={args.seed}) | "
          f"val(n={args.n_eval}, seed={args.seed_val}) | "
          f"test(n={args.n_test}, seed={args.seed_test})")
    print(f"  λ candidates: {args.lam} | modes: {'/'.join(args.modes)} | "
          f"know: rank<= {args.rank_threshold} (1-indexed) + check_correct_exact")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/6] Loading model...")
    model = load_model(device=device, model_id=args.model)
    model.eval()
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    print(f"  {n_layers} layers (final L{final_layer}), d_model={model.cfg.d_model}")

    # ── Collect (or load cached) cal/val hidden states ──
    cache_path = (
        output_dir
        / f"subspace_states_cal{args.seed}_n{args.n_dir}_val{args.seed_val}_n{args.n_eval}.pt"
    )
    cache_meta = {
        "model": args.model,
        "layers": args.layers,
        "layer_early": args.layer_early,
        "n_dir": args.n_dir,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "seed_val": args.seed_val,
    }

    cal_entries = None
    val_entries = None
    n_skip_cal = n_skip_val = 0
    if args.skip_collect and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("meta") == cache_meta:
            cal_entries, val_entries = cached["cal"], cached["val"]
            n_skip_cal = n_skip_val = None  # unknown from cache; recorded as null
            print(f"Loaded cached cal/val states from {cache_path}")
    if cal_entries is None:
        print(f"\n[2/6] Collecting cal/val hidden states...")
        cal_samples = load_triviaqa(n_samples=args.n_dir, seed=args.seed)
        val_samples = load_triviaqa(n_samples=args.n_eval, seed=args.seed_val)
        cal_entries, n_skip_cal = collect_split(
            model, model.tokenizer, cal_samples, args.layers,
            args.layer_early, device, "cal",
        )
        val_entries, n_skip_val = collect_split(
            model, model.tokenizer, val_samples, args.layers,
            args.layer_early, device, "val",
        )
        torch.save({"meta": cache_meta, "cal": cal_entries, "val": val_entries}, cache_path)
        print(f"  Saved cache to {cache_path}")
    skip_msg = (
        f"cal: {len(cal_entries)} usable (skipped {n_skip_cal}), "
        f"val: {len(val_entries)} usable (skipped {n_skip_val})"
    )
    if n_skip_cal is None:
        skip_msg = (
            f"cal: {len(cal_entries)} usable, val: {len(val_entries)} usable "
            f"(from cache — skip counts unknown)"
        )
    print(f"  {skip_msg}")

    # ── Fit directions + PCA on cal (diagnostic angles vs val) ──
    print(f"\n[3/6] Fitting directions + PCA on CAL (test untouched)...")
    layer_models = {}
    for L in args.layers:
        fit = fit_layer_models(cal_entries, val_entries, L, args.k_pca)
        if fit is None:
            print(f"  WARNING: L{L} skipped (need ≥1 correct AND ≥1 incorrect cal "
                  f"sample, and ≥1 val state)")
            continue
        layer_models[L] = fit
        print(f"  L{L}: cal correct/incorrect = {fit['n_cal_correct']}/{fit['n_cal_incorrect']}, "
              f"PCA k={fit['k_actual']}, max angle={fit['max_angle_deg']:.1f}°, "
              f"cos(raw, aligned)={fit['cos_raw_aligned']:+.4f}")

    if not layer_models:
        raise RuntimeError("No layer could be fit — aborting")

    # ── Val: classify + λ/mode sweep + selection ──
    print(f"\n[4/6] Val sweep: selecting (mode, λ) on val...")
    assign_subsets(val_entries, args.rank_threshold)
    val_baseline = baseline_subset_stats(val_entries)
    val_kw_n = val_baseline["know_wrong"]["total"]
    print(f"  val: KW={val_kw_n}, KC={val_baseline['know_correct']['total']}, "
          f"DK={val_baseline['dont_know']['total']}, All={val_baseline['all']['total']}")

    val_selection = {}
    lambda0_ok = True
    for L, fit in layer_models.items():
        val_selection[L] = {}
        for dtype, dvec in [("raw_mean_diff", fit["d_raw"]),
                            ("pca_aligned", fit["d_aligned"])]:
            dtype_sweep = []
            for mode in args.modes:
                for lam in sorted(args.lam):
                    counts, _ = evaluate_intervention(
                        model, model.tokenizer, val_entries, dvec, L, lam, mode,
                        device, desc=f"val L{L} {dtype} {mode} λ={lam}",
                    )
                    dtype_sweep.append(
                        {"mode": mode, "lam": lam,
                         "results": build_subset_results(counts, val_baseline)}
                    )
            selected, criterion = select_config(dtype_sweep, val_kw_n)
            val_selection[L][dtype] = {
                "selected_mode": selected["mode"],
                "selected_lam": selected["lam"],
                "criterion": criterion,
                "sweep": dtype_sweep,
            }
            print(f"  L{L} {dtype}: selected mode={selected['mode']} λ={selected['lam']} "
                  f"(by {criterion})")

        # λ=0 degeneracy check (protocol rule 5: λ=0 must strictly equal baseline)
        _counts0, flags0 = evaluate_intervention(
            model, model.tokenizer, val_entries, fit["d_raw"], L, 0.0, "subtract",
            device, desc=f"val L{L} λ=0 check",
        )
        flags0 = [flags0[e["sample_id"]]["is_correct"] for e in val_entries]
        ok0 = flags0 == [e["is_correct"] for e in val_entries]
        lambda0_ok = lambda0_ok and ok0
        print(f"  L{L}: λ=0 degenerates to baseline: {'✅' if ok0 else '❌'}")

        # Random control (deterministic seed): val at raw's selection, test same
        rng = torch.Generator().manual_seed(args.seed * 1000 + L)
        d_rand = torch.randn(model.cfg.d_model, generator=rng)
        d_rand = d_rand / (d_rand.norm() + 1e-8)
        fit["d_rand"] = d_rand
        sel_raw = val_selection[L]["raw_mean_diff"]
        counts_r, _ = evaluate_intervention(
            model, model.tokenizer, val_entries, d_rand, L,
            sel_raw["selected_lam"], sel_raw["selected_mode"], device,
            desc=f"val L{L} random",
        )
        fit["val_random_results"] = build_subset_results(counts_r, val_baseline)

    # ── Test: classify + report only ──
    print(f"\n[5/6] Test (seed={args.seed_test}) — report only...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_entries, n_skip_test = collect_split(
        model, model.tokenizer, test_samples, [], args.layer_early, device, "test"
    )
    assign_subsets(test_entries, args.rank_threshold)
    test_baseline = baseline_subset_stats(test_entries)
    print(f"  test: n={len(test_entries)} (skipped {n_skip_test}); "
          f"KW={test_baseline['know_wrong']['total']}, "
          f"KC={test_baseline['know_correct']['total']}, "
          f"DK={test_baseline['dont_know']['total']}")
    print(f"  baseline: All={_pct(test_baseline['all']['rate'])}, "
          f"KW={_pct(test_baseline['know_wrong']['rate'])}, "
          f"KC={_pct(test_baseline['know_correct']['rate'])}, "
          f"DK={_pct(test_baseline['dont_know']['rate'])}")

    results = {}
    test_per_sample = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "rank_threshold": args.rank_threshold,
            "layer_early": args.layer_early,
        },
        "samples": {},
    }
    for e in test_entries:
        test_per_sample["samples"][str(e["sample_id"])] = {
            "subset": e["subset"],
            "rank_final": e["rank_final"],
            "rank_early": e["rank_early"],
            "question": e["question"],
            "answers": e["answers"],
            "baseline_correct": bool(e["is_correct"]),
            "pred_base": e["pred_base"],
            "interventions": {},
        }

    for L, fit in layer_models.items():
        results[L] = {}
        for dtype, dvec in [
            ("raw_mean_diff", fit["d_raw"]),
            ("pca_aligned", fit["d_aligned"]),
            ("random", fit["d_rand"]),
        ]:
            sel = val_selection[L][dtype if dtype != "random" else "raw_mean_diff"]
            counts, per_sample = evaluate_intervention(
                model, model.tokenizer, test_entries, dvec, L,
                sel["selected_lam"], sel["selected_mode"], device,
                desc=f"test L{L} {dtype}",
            )
            res = build_subset_results(counts, test_baseline)
            results[L][dtype] = {
                "selected_mode": sel["selected_mode"],
                "selected_lam": sel["selected_lam"],
                "selected_on_val_by": sel["criterion"],
                "test": res,
            }
            for e in test_entries:
                test_per_sample["samples"][str(e["sample_id"])]["interventions"][
                    f"L{L}|{dtype}"
                ] = per_sample[e["sample_id"]]

    # ── Report ──
    print(f"\n[6/6] Results (Δ vs baseline; KW 95% CI on post-intervention rate)")
    for L, fit in layer_models.items():
        for dtype in ["raw_mean_diff", "pca_aligned", "random"]:
            r = results[L][dtype]
            sel = val_selection[L][dtype if dtype != "random" else "raw_mean_diff"]
            print(f"\n  ── L{L} {dtype} (mode={r['selected_mode']}, λ={r['selected_lam']}, "
                  f"selected on val by {r['selected_on_val_by']}) ──")
            print(f"  {'subset':<12} {'corr/total':>10} {'rate':>8} {'Δ':>9} {'95% CI':>18}")
            for s in SUBSET_ORDER:
                x = r["test"][s]
                if x["total"] == 0:
                    continue
                delta = f"{x['delta']:+.1%}" if x["delta"] is not None else "N/A"
                ci = f"[{x['ci95'][0]:.1%}, {x['ci95'][1]:.1%}]"
                print(f"  {s:<12} {x['correct']:>4}/{x['total']:<5} {x['rate']:>8.1%} "
                      f"{delta:>9} {ci:>18}")
            if dtype != "random":
                rnd = fit["val_random_results"]
                kwr = rnd["know_wrong"]
                print(f"  random control (same mode/λ): KW {kwr['correct']}/{kwr['total']} "
                      f"(Δ={_pct(kwr['delta'], signed=True)})")

    # ── Statistical note (same structure as TLDC, 2026-08-25) ──
    stat_note = (
        "KW baseline rate is a DEFINED value (0: KW = know AND baseline-greedy-wrong), "
        "not a sampled proportion. Under H0 'intervention has zero effect' (p=0), "
        "P(any rescue)=0 → any k>0 rescue on KW rejects 'absolutely no effect'; the "
        "effect size is given by the Clopper-Pearson CI on the post-intervention rate. "
        "Verdict rule: the selected config's KW CI lower bound stays > 0 across two "
        "test seeds (123/456) → real (if small) KW effect; decide practical value by "
        "the point estimate vs the 5% bar."
    )
    print(f"\n  ── Statistical note ──\n  {stat_note}")

    # ── Protocol self-check (8-point checklist) ──
    protocol_checks = {
        "1_truncation_keep_tail": {
            "applied": True,
            "evidence": f"tokens[:, -{MAX_PROMPT_TOKENS}:] on every forward; "
                        "format_prompt(triviaqa) head-truncates context",
        },
        "2_exact_labels": {
            "applied": True,
            "evidence": "check_correct_exact for direction labels AND KW/KC/DK",
        },
        "3_heldout_cv": {
            "applied": True,
            "evidence": f"direction/PCA fit on cal only (n={len(cal_entries)}); "
                        "val selects (mode, λ); test reports only",
        },
        "4_rank_1_indexed": {
            "applied": True,
            "evidence": f"get_y_true_rank returns 1-indexed position; "
                        f"know = rank_final <= {args.rank_threshold}",
        },
        "5_real_logits": {
            "applied": True,
            "evidence": "predictions from model.run_with_hooks/model() FINAL logits; "
                        "lens (ln_final @ W_U) only for layer_early diagnostic rank",
        },
        "6_param_selection_off_test": {
            "applied": True,
            "evidence": f"(mode, λ) selected on val (n={len(val_entries)}) by "
                        f"KW Δ (val KW={val_kw_n}); test never enters any fit/selection",
        },
        "7_detach": {
            "applied": True,
            "evidence": "all hidden captures .detach() under torch.no_grad()",
        },
        "8_no_sign_flip": {
            "applied": True,
            "evidence": "no AUROC/sign selection anywhere in this script",
        },
        "lambda0_degenerate_baseline": bool(lambda0_ok),
    }

    # ── Save ──
    output = {
        "protocol": "docs/protocol/evaluation-protocol.md §3 (8-point checklist)",
        "config": {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "device_used": device,
            "n_layers": n_layers,
            "final_layer": final_layer,
            "n_skipped_cal": n_skip_cal,
            "n_skipped_val": n_skip_val,
            "n_skipped_test": n_skip_test,
        },
        "calibration": {
            L: {
                "n_cal_correct": fit["n_cal_correct"],
                "n_cal_incorrect": fit["n_cal_incorrect"],
                "n_val_states": fit["n_val_states"],
            }
            for L, fit in layer_models.items()
        },
        "subspace_analysis": {
            str(L): {
                "k_actual": fit["k_actual"],
                "max_angle_deg": fit["max_angle_deg"],
                "mean_angle_deg": fit["mean_angle_deg"],
                "principal_angles_deg": fit["principal_angles_deg"],
                "cal_pca_explained": fit["cal_pca_explained"],
                "val_pca_explained": fit["val_pca_explained"],
                "cos_raw_aligned": fit["cos_raw_aligned"],
                "note": "45° threshold is a geometric DIAGNOSTIC only; the review "
                        "verdict comes from the test Δ/CI below",
            }
            for L, fit in layer_models.items()
        },
        "val_selection": {str(L): v for L, v in val_selection.items()},
        "val_random_control": {
            str(L): fit["val_random_results"] for L, fit in layer_models.items()
        },
        "test": {
            "n_total": test_baseline["all"]["total"],
            "n_know_correct": test_baseline["know_correct"]["total"],
            "n_know_wrong": test_baseline["know_wrong"]["total"],
            "n_dont_know": test_baseline["dont_know"]["total"],
            "baseline_rate": test_baseline["all"]["rate"],
            "kw_baseline_rate": test_baseline["know_wrong"]["rate"],
            "kc_baseline_rate": test_baseline["know_correct"]["rate"],
            "dk_baseline_rate": test_baseline["dont_know"]["rate"],
        },
        "results": {str(L): v for L, v in results.items()},
        "protocol_checks": protocol_checks,
        "statistical_note": stat_note,
    }

    out_path = output_dir / f"subspace_review_s{args.seed_test}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved summary to {out_path}")

    samples_path = output_dir / f"subspace_review_s{args.seed_test}_samples.json"
    with open(samples_path, "w") as f:
        json.dump(test_per_sample, f, indent=2, ensure_ascii=False)
    print(f"Saved per-sample archive to {samples_path}")

    # ── Geometric diagnostic (paradigm-preserving, non-verdict) ──
    print(f"\n{'=' * 72}\nGeometric diagnostic (cal vs val PCA subspaces)")
    for L, fit in layer_models.items():
        max_angle = fit["max_angle_deg"]
        status = "⚠ UNRELIABLE" if max_angle > 45.0 else "✓ acceptable"
        print(f"  L{L}: max principal angle = {max_angle:.1f}° {status}")
    print("NOTE: this diagnostic decides nothing — the review verdict is the test Δ/CI.")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
