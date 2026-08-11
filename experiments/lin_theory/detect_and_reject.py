"""Path C: Detect-and-Reject — oracle upper bound for output-level hallucination mitigation.

Uses the g_vec classifier (from A.1) to detect likely-wrong answers and reject them
at inference time. Since g_vec requires y_true_id (the ground-truth answer token),
this is an ORACLE experiment — it establishes the information-theoretic ceiling for
what detect-and-reject can achieve.

If even the oracle cannot improve effective accuracy by >5%, then zero-shot
detection methods (without ground-truth answers) cannot either.

Gate C.3: Effective Accuracy > Baseline Accuracy + 5%

Usage:
  # 1.7B (default)
  python detect_and_reject.py --n_samples 500

  # 8B
  python detect_and_reject.py \
      --model_path /root/autodl-tmp/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
      --n_samples 500
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_offline = os.environ.get("HF_ALLOW_ONLINE", "") != "1"
if _offline:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Path setup (same as analyze_g_vec) ─────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import format_prompt, check_correct  # noqa: E402

# ── Import reusable modules from analyze_g_vec ──────────────────────────────────
from analyze_g_vec import (  # noqa: E402
    _get_delta_layers,
    _compute_auroc,
    load_triviaqa_train,
    get_first_answer_token_id,
    classify_sample,
    generate_answer,
    compute_g_vec,
    compute_scalar_delta,
    LogisticRegression,
    loocv_auroc,
    NUM_DELTA_LAYERS,
    RANK_THRESHOLD,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"


# ── New functions for Detect-and-Reject ─────────────────────────────────────────


def _find_model_path() -> str:
    """Resolve default model path (Qwen3-1.7B) from cache."""
    MODEL_ID = "Qwen/Qwen3-1.7B"
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if os.path.isdir(local):
            if os.path.isfile(os.path.join(local, "config.json")):
                return local
            snaps = os.path.join(local, "snapshots")
            if os.path.isdir(snaps):
                for s in sorted(os.listdir(snaps)):
                    sp = os.path.join(snaps, s)
                    if os.path.isfile(os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID


def train_classifier_on_all(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0
) -> tuple[LogisticRegression, np.ndarray]:
    """Train a single LogisticRegression on all data; return fitted clf + P(override) scores.

    Returns:
        clf: fitted LogisticRegression
        scores: shape [n], P(override) for each sample in [0, 1]
    """
    clf = LogisticRegression(l2=l2)
    clf.fit(X, y)
    scores = clf.predict_proba(X)
    return clf, scores


def scan_rejection_thresholds(
    scores: np.ndarray, correctness: np.ndarray, n_steps: int = 101
) -> list[dict]:
    """Scan rejection thresholds τ ∈ [0, 1], producing the reject curve.

    Rejection rule: reject if P(override) > τ  (i.e. keep if scores <= τ).

    Args:
        scores: shape [n], P(override) in [0, 1]
        correctness: shape [n], bool — whether the generated answer is correct
        n_steps: number of threshold values to scan

    Returns:
        list of dicts with keys:
          threshold, reject_rate, n_kept, n_rejected, accuracy_on_kept, effective_accuracy
    """
    n = len(scores)
    thresholds = np.linspace(0.0, 1.0, n_steps)
    results = []

    for tau in thresholds:
        keep_mask = scores <= tau  # keep = NOT predicted as override
        n_kept = int(keep_mask.sum())
        n_rejected = n - n_kept
        reject_rate = n_rejected / n

        if n_kept > 0:
            acc_kept = float(correctness[keep_mask].mean())
            eff_acc = acc_kept * (1.0 - reject_rate)
        else:
            acc_kept = None
            eff_acc = 0.0

        results.append(
            {
                "threshold": round(float(tau), 4),
                "reject_rate": round(float(reject_rate), 6),
                "n_kept": n_kept,
                "n_rejected": n_rejected,
                "accuracy_on_kept": round(acc_kept, 6)
                if acc_kept is not None
                else None,
                "effective_accuracy": round(float(eff_acc), 6),
            }
        )

    return results


def find_best_threshold(
    scan_results: list[dict],
    baseline_accuracy: float,
    gate_margin: float = 0.05,
) -> tuple[dict | None, dict]:
    """Find the threshold that maximises effective accuracy; evaluate gate.

    Returns:
        best: dict with best threshold entry (or None if no valid entry)
        gate: dict with keys effective_accuracy, baseline_accuracy, delta, target, passed
    """
    valid = [
        r for r in scan_results if r["accuracy_on_kept"] is not None and r["n_kept"] > 0
    ]
    if not valid:
        gate = {
            "effective_accuracy": 0.0,
            "baseline_accuracy": round(baseline_accuracy, 6),
            "delta": 0.0,
            "target": gate_margin,
            "passed": False,
        }
        return None, gate

    # Sort: max effective_accuracy, then min reject_rate (prefer answering more)
    valid.sort(key=lambda r: (-r["effective_accuracy"], r["reject_rate"]))
    best = dict(valid[0])

    delta = best["effective_accuracy"] - baseline_accuracy
    gate = {
        "effective_accuracy": best["effective_accuracy"],
        "baseline_accuracy": round(baseline_accuracy, 6),
        "delta": round(float(delta), 6),
        "target": gate_margin,
        "passed": delta > gate_margin,
    }
    return best, gate


def rejection_stats_by_category(
    scores: np.ndarray,
    categories: list[str],
    correctness: np.ndarray,
    best_tau: float | None,
) -> dict:
    """Break down rejection impact per sample category (KC, KW, DK).

    Args:
        scores: P(override) per sample
        categories: "KC" | "KW" | "DK" per sample
        correctness: bool per sample
        best_tau: best threshold (None → no rejection)

    Returns:
        dict keyed by category, each with n, n_rejected, accuracy_before, accuracy_after
    """
    result = {}
    for cat in ["KC", "KW", "DK"]:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        idx_arr = np.array(idx)
        cat_correct = correctness[idx_arr]
        cat_scores = scores[idx_arr]
        acc_before = float(cat_correct.mean())

        if best_tau is not None:
            keep_mask = cat_scores <= best_tau
            n_rejected = int((~keep_mask).sum())
            if keep_mask.any():
                acc_after = float(cat_correct[keep_mask].mean())
            else:
                acc_after = None
        else:
            n_rejected = 0
            acc_after = acc_before

        result[cat] = {
            "n": len(idx),
            "n_rejected": n_rejected,
            "accuracy_before": round(acc_before, 6),
            "accuracy_after": round(acc_after, 6) if acc_after is not None else None,
        }
    return result


# ── Main pipeline ───────────────────────────────────────────────────────────────


def run_detect_and_reject(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = args.model_path or _find_model_path()
    k = args.top_k
    print(f"Path C: Detect-and-Reject | n={args.n_samples} | k={k} | oracle=True")

    # ── 1. Load model + tokenizer ──────────────────────────────────────────
    print("\n[1/6] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    ref_layer, last_layer = _get_delta_layers(model)
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(
        f"  Model: {model.config.num_hidden_layers} layers, d_model={d_model}, "
        f"vocab={vocab_size}, ref=L{ref_layer}, last=L{last_layer}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load data ───────────────────────────────────────────────────────
    print(f"\n[2/6] Loading {args.n_samples} TriviaQA train samples...")
    samples = load_triviaqa_train(n_samples=args.n_samples, seed=args.seed)

    # ── 3. Register hooks (same pattern as analyze_g_vec) ──────────────────
    try:
        layers = model.model.layers
        norm_fn = model.model.norm
        lm_head = model.lm_head
    except AttributeError:
        try:
            layers = model.base_model.model.model.layers
            norm_fn = model.base_model.model.model.norm
            lm_head = model.base_model.model.lm_head
        except AttributeError:
            layers = model.model.model.layers
            norm_fn = model.model.model.norm
            lm_head = model.model.lm_head

    h_ref_cache = {}
    h_last_cache = {}

    def _hook_ref(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        h_ref_cache["h"] = hs[:, -1, :].detach()

    def _hook_last(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        h_last_cache["h"] = hs[:, -1, :].detach()

    handle_ref = layers[ref_layer].register_forward_hook(_hook_ref)
    handle_last = layers[last_layer].register_forward_hook(_hook_last)

    # ── 4. Extract g_vec + generate + evaluate per sample ──────────────────
    print(f"\n[3/6] Processing {len(samples)} samples (extract g_vec + generate)...")
    X_all = []  # g_vec features [n, k+1]
    y_all = []  # 1=KW, 0=KC+DK (for classifier training)
    delta_all = []  # scalar deltas
    categories = []  # "KC" | "KW" | "DK"
    correctness = []  # bool: is generated answer correct?
    metadata = []  # per-sample info for JSON

    for s in tqdm(samples):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        h_ref_cache.clear()
        h_last_cache.clear()

        with torch.no_grad():
            _ = model(**tokens)

        h_ref = h_ref_cache.get("h")
        h_last = h_last_cache.get("h")
        if h_ref is None or h_last is None:
            continue

        # Channel gains g(t) = logits_last(t) - logits_ref(t)
        h_ref_norm = norm_fn(h_ref.to(dtype=norm_fn.weight.dtype))
        logits_ref = lm_head(h_ref_norm).float().detach().cpu().numpy().flatten()

        h_last_norm = norm_fn(h_last.to(dtype=norm_fn.weight.dtype))
        logits_last = lm_head(h_last_norm).float().detach().cpu().numpy().flatten()

        g = logits_last - logits_ref  # [vocab]

        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        logits_first = torch.from_numpy(logits_last)
        generated = generate_answer(model, tokenizer, prompt, device)
        category = classify_sample(logits_first, y_true_id, generated, s["answers"])
        is_correct = check_correct(generated, s["answers"], dataset="triviaqa")

        if y_true_id is not None and y_true_id < len(g):
            g_vec = compute_g_vec(g, y_true_id, k=k)
            delta_val = compute_scalar_delta(g, y_true_id)
        else:
            g_vec = np.zeros(k + 1, dtype=np.float64)
            delta_val = 0.0

        X_all.append(g_vec)
        y_all.append(1 if category == "KW" else 0)
        delta_all.append(delta_val)
        categories.append(category)
        correctness.append(is_correct)
        metadata.append(
            {
                "question": s["question"][:80],
                "category": category,
                "generated": generated[:120],
                "is_correct": is_correct,
            }
        )

    handle_ref.remove()
    handle_last.remove()

    X = np.array(X_all, dtype=np.float64)  # [n, k+1]
    y = np.array(y_all, dtype=np.int64)  # [n]
    deltas = np.array(delta_all, dtype=np.float64)  # [n]
    correct_arr = np.array(correctness, dtype=bool)  # [n]

    n_kw = int(y.sum())
    n_total = len(X)
    baseline_accuracy = float(correct_arr.mean())
    baseline_correct = int(correct_arr.sum())
    baseline_incorrect = n_total - baseline_correct

    # Count per category
    cat_counts = defaultdict(int)
    for c in categories:
        cat_counts[c] += 1

    print(
        f"\n  Samples: {n_total} total, {cat_counts.get('KW', 0)} KW, "
        f"{cat_counts.get('KC', 0)} KC, {cat_counts.get('DK', 0)} DK"
    )
    print(
        f"  Baseline accuracy: {baseline_accuracy:.4f} ({baseline_correct}/{n_total})"
    )

    # ── 5. Train classifier + threshold scan ───────────────────────────────
    print(f"\n[4/6] Training classifier + scanning rejection thresholds...")

    # AUROC evaluation (same as A.1, for reporting)
    if n_kw >= 3:
        gvec_auroc = loocv_auroc(X, y, l2=args.l2)
    else:
        gvec_auroc = 0.5
        print("  WARNING: <3 KW samples, LOOCV AUROC set to 0.5")

    delta_auroc = _compute_auroc(deltas, y)

    print(f"  g_vec AUROC (LOOCV): {gvec_auroc:.4f}")
    print(f"  scalar δ AUROC:       {delta_auroc:.4f}")

    # Train classifier on all data for scoring
    clf, scores = train_classifier_on_all(X, y, l2=args.l2)

    # Scan thresholds
    print(f"\n[5/6] Scanning {args.n_thresholds} thresholds...")
    scan_results = scan_rejection_thresholds(
        scores, correct_arr, n_steps=args.n_thresholds
    )

    # Find best threshold
    best, gate = find_best_threshold(
        scan_results, baseline_accuracy, gate_margin=args.gate_margin
    )

    # Per-category rejection stats
    best_tau = best["threshold"] if best else None
    per_cat = rejection_stats_by_category(scores, categories, correct_arr, best_tau)

    # ── 6. Report + save ────────────────────────────────────────────────────
    print(f"\n[6/6] Results\n{'=' * 60}")

    if best:
        print(f"\n  Best threshold τ = {best['threshold']:.4f}:")
        print(
            f"    Reject rate:        {best['reject_rate']:.4f} "
            f"({best['n_rejected']}/{n_total} samples)"
        )
        print(f"    Accuracy on kept:   {best['accuracy_on_kept']:.4f}")
        print(f"    Effective accuracy: {best['effective_accuracy']:.4f}")
    else:
        print("\n  No valid threshold found (all reject all or keep all).")

    print(f"\n  Baseline accuracy:    {baseline_accuracy:.4f}")
    print(f"  Δ effective vs base:  {gate['delta']:+.4f}")
    print(
        f"  Gate C.3 (Δ > {args.gate_margin}): "
        f"{'✅ PASS' if gate['passed'] else '❌ FAIL'}"
    )

    # Per-category breakdown
    print(f"\n  Per-category rejection:")
    for cat in ["KC", "KW", "DK"]:
        if cat in per_cat:
            s = per_cat[cat]
            acc_str = (
                f"acc={s['accuracy_before']:.4f}→{s['accuracy_after']:.4f}"
                if s["accuracy_after"] is not None
                else f"acc={s['accuracy_before']:.4f}→all_rejected"
            )
            print(f"    {cat}: {s['n_rejected']}/{s['n']} rejected, {acc_str}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUT_DIR / "detect_and_reject.json"

    output = {
        "config": {
            "n_samples": args.n_samples,
            "k": k,
            "seed": args.seed,
            "l2": args.l2,
            "ref_layer": ref_layer,
            "last_layer": last_layer,
            "model": MODEL_PATH,
            "gate_margin": args.gate_margin,
            "n_thresholds": args.n_thresholds,
            "oracle": True,
        },
        "sample_distribution": {
            "total": n_total,
            "KW": cat_counts.get("KW", 0),
            "KC": cat_counts.get("KC", 0),
            "DK": cat_counts.get("DK", 0),
        },
        "auroc": {
            "g_vec_loocv": float(gvec_auroc),
            "scalar_delta": float(delta_auroc),
        },
        "baseline": {
            "accuracy": round(baseline_accuracy, 6),
            "n_total": n_total,
            "n_correct": baseline_correct,
            "n_incorrect": baseline_incorrect,
        },
        "threshold_scan": scan_results,
        "best": best,
        "gate": gate,
        "per_category": per_cat,
        "per_sample": metadata,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Path C: Detect-and-Reject (oracle upper bound)"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path override (required for 8B).",
    )
    parser.add_argument(
        "--n_thresholds",
        type=int,
        default=101,
        help="Number of threshold values to scan.",
    )
    parser.add_argument(
        "--gate_margin",
        type=float,
        default=0.05,
        help="Required effective accuracy improvement over baseline.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_detect_and_reject(args)


if __name__ == "__main__":
    main()
