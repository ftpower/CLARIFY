"""Phase 16.1+16.2: ITI-style Multi-Head Truth Probe + Attention Shift.

Inference-Time Intervention (Li et al. 2023):
  1. Train linear probes on each attention head's output to predict truthfulness
  2. Select top-K "truth-related heads"
  3. During inference, shift those heads' outputs along probe direction

Key difference from all previous phases: modifies attention head output (hook_z),
NOT residual stream. This changes information routing rather than adding a vector
that downstream layers can compensate for.

Usage:
  python phase16_iti.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract.json \
    --n_test 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════
# Phase 16.1: Train Multi-Head Probes
# ═══════════════════════════════════════════════════════════════════


def extract_head_outputs(model, records, layer: int, device: str):
    """Extract per-head z vectors at last token position for all records.

    Returns:
      X: np.ndarray [n_samples, n_heads, d_head] — per-head output vectors
      y: np.ndarray [n_samples] — correctness labels
    """
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    n = len(records)

    X = np.zeros((n, n_heads, d_head), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    hook_name = f"blocks.{layer}.attn.hook_z"
    storage = {}

    def _capture(act, hook=None):
        storage["z"] = act[0, -1, :, :].clone()  # [n_heads, d_head]
        return act

    for i, rec in enumerate(tqdm(records, desc=f"  Extracting L{layer} head outputs")):
        question = rec["question"]
        context = rec.get("context", "")
        label = rec["label"]

        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])

        X[i] = storage["z"].cpu().numpy()
        y[i] = label

    return X, y


def train_head_probes(X_train, y_train, X_val, y_val):
    """Train logistic regression probe per head. Returns list of (auroc, coef, intercept)."""
    n_heads = X_train.shape[1]
    probes = []

    for h in range(n_heads):
        # Logistic regression on 128-dim head output
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=1000, random_state=42
        )
        clf.fit(X_train[:, h, :], y_train)

        # Evaluate
        proba = clf.predict_proba(X_val[:, h, :])[:, 1]
        auroc = roc_auc_score(y_val, proba)

        # Direction = weight vector pointing toward "truthful"
        coef = clf.coef_[0].astype(np.float32)  # [d_head]
        coef = coef / (np.linalg.norm(coef) + 1e-8)

        probes.append(
            {
                "head": h,
                "auroc": float(auroc),
                "coef": coef,
                "intercept": float(clf.intercept_[0]),
            }
        )

    # Sort by AUROC descending
    probes.sort(key=lambda p: p["auroc"], reverse=True)
    return probes


# ═══════════════════════════════════════════════════════════════════
# Phase 16.2: Attention Shift Intervention
# ═══════════════════════════════════════════════════════════════════


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20):
    """Core greedy generation with hooks."""
    input_len = tokens.shape[1]
    gids = []

    for _step in range(max_new):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break

    return tokenizer.decode(gids).strip()


def make_iti_hook(layer: int, top_k_probes: list, alpha: float, device: str):
    """Create a hook that shifts top-K attention head outputs by α·v_head.

    Modifies blocks.{layer}.attn.hook_z which has shape [batch, seq, n_heads, d_head].
    Only the top-K heads are shifted; other heads are left unchanged.

    Args:
      top_k_probes: list of dicts with keys 'head' and 'coef'
      alpha: scaling factor
      device: torch device
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    # Build per-head shift vectors [K, d_head]
    shifts = {}
    for probe in top_k_probes:
        h = probe["head"]
        vec = torch.tensor(alpha * probe["coef"], dtype=torch.float32, device=device)
        shifts[h] = vec

    def _hook(act, hook=None):
        # act: [batch, seq, n_heads, d_head]
        # Only modify last token position
        for h, vec in shifts.items():
            act[0, -1, h, :] = act[0, -1, h, :] + vec
        return act

    return hook_name, _hook


def evaluate_iti(model, tokenizer, test_records, device, layer, top_k_probes, alpha):
    """Evaluate ITI intervention on test set."""
    hook_name, hook_fn = make_iti_hook(layer, top_k_probes, alpha, device)

    correct = 0
    total = len(test_records)

    for rec in tqdm(
        test_records, desc=f"    ITI K={len(top_k_probes)} α={alpha:+.1f}", leave=False
    ):
        question = rec["question"]
        context = rec.get("context", "")
        gt_answers = rec["gt_answers"]

        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        generated = _gen_greedy(
            model, tokenizer, tokens, device, [(hook_name, hook_fn)]
        )
        if check_correct(generated, gt_answers, dataset="triviaqa"):
            correct += 1

    return correct / total


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 16 ITI: Multi-Head Truth Probe + Attention Shift"
    )
    parser.add_argument("--load", required=True, help="phase9 extraction JSON")
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[20],
        help="Layers to probe (default: L20 only)",
    )
    parser.add_argument("--top_k_values", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[-1.0, -0.5, 0.5, 1.0]
    )
    parser.add_argument("--skip_probe_training", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    with open(args.load) as f:
        data = json.load(f)
    all_records = data["records"]

    n_test = min(args.n_test, len(all_records) // 2)
    n_train = len(all_records) - n_test
    train_records = all_records[:n_train]
    test_records = all_records[n_train:]
    print(f"  Train: {n_train}, Test: {n_test}")
    print(
        f"  Test correct: {sum(1 for r in test_records if r['label'] == 1)} / {n_test}"
    )

    # ── Load model ────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    tokenizer = model.tokenizer
    n_head = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"  Model: {model.cfg.model_name}, heads={n_head}, d_head={d_head}")

    # ── Phase 16.1: Train probes ──────────────────────────────
    print(f"\n[3/4] Phase 16.1: Training head probes (layers {args.layers})...")

    # Split train into train/val for probe evaluation
    n_val = min(30, n_train // 3)
    probe_train = train_records[:-n_val]
    probe_val = train_records[-n_val:]

    all_layer_probes = {}  # layer -> sorted list of probes
    all_layer_aurocs = {}  # layer -> best AUROC

    for layer in args.layers:
        print(f"\n  ── Layer {layer} ──")
        X_tr, y_tr = extract_head_outputs(model, probe_train, layer, device)
        X_v, y_v = extract_head_outputs(model, probe_val, layer, device)

        # Check class balance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        print(
            f"  Train: {n_pos} pos / {n_neg} neg (balanced: {min(n_pos, n_neg) / max(n_pos, n_neg):.2f})"
        )

        probes = train_head_probes(X_tr, y_tr, X_v, y_v)
        all_layer_probes[layer] = probes
        all_layer_aurocs[layer] = probes[0]["auroc"]

        print(f"  Top-5 heads by AUROC:")
        for p in probes[:5]:
            print(f"    Head {p['head']:2d}: AUROC={p['auroc']:.4f}")

    # ── Baseline generation ───────────────────────────────────
    print(f"\n[4/4] Phase 16.2: ITI attention shift...")

    # Real baseline (no intervention)
    print("  Getting baseline (no intervention)...")
    baseline_correct = 0
    for rec in tqdm(test_records, desc="  Baseline"):
        question = rec["question"]
        context = rec.get("context", "")
        gt_answers = rec["gt_answers"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(model, tokenizer, tokens, device, [])
        if check_correct(generated, gt_answers, dataset="triviaqa"):
            baseline_correct += 1
    baseline_rate = baseline_correct / n_test
    print(f"  Baseline: {baseline_correct}/{n_test} = {baseline_rate:.1%}")

    # ── ITI evaluation grid ───────────────────────────────────
    results = {"baseline_rate": baseline_rate, "layers": {}}

    for layer in args.layers:
        probes = all_layer_probes[layer]
        layer_results = {}
        best_rate = baseline_rate
        best_config = "baseline"

        for K in args.top_k_values:
            # Clamp K to available heads
            K_actual = min(K, len(probes))
            top_k = probes[:K_actual]

            for alpha in args.alphas:
                t0 = time.time()
                rate = evaluate_iti(
                    model, tokenizer, test_records, device, layer, top_k, alpha
                )
                elapsed = time.time() - t0

                key = f"K{K_actual}_α{alpha:+.1f}"
                layer_results[key] = {"rate": rate, "time_s": elapsed}
                delta = rate - baseline_rate
                print(f"  L{layer} {key}: {rate:.1%} (Δ={delta:+.1%}) [{elapsed:.0f}s]")

                if rate > best_rate:
                    best_rate = rate
                    best_config = key

        results["layers"][str(layer)] = {
            "best_rate": best_rate,
            "best_config": best_config,
            "configs": layer_results,
            "head_aurocs": {str(p["head"]): p["auroc"] for p in probes},
        }
        print(
            f"  L{layer} best: {best_config} @ {best_rate:.1%} (Δ={best_rate - baseline_rate:+.1%})"
        )

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ITI RESULTS")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline_rate:.1%}")
    overall_best_rate = baseline_rate
    overall_best = "baseline"

    for layer in args.layers:
        lr = results["layers"][str(layer)]
        print(f"  L{layer}: best={lr['best_config']} @ {lr['best_rate']:.1%}")
        if lr["best_rate"] > overall_best_rate:
            overall_best_rate = lr["best_rate"]
            overall_best = f"L{layer}_{lr['best_config']}"

    print(f"\nOverall best: {overall_best} @ {overall_best_rate:.1%}")

    if overall_best_rate <= baseline_rate:
        print("⚠ ITI zero effect — probe direction can detect but not control")
    else:
        print(f"✓ ITI works! Δ = {overall_best_rate - baseline_rate:+.1%}")

    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "outputs_phase16"
    output_dir.mkdir(exist_ok=True)

    results["summary"] = {
        "baseline_rate": baseline_rate,
        "overall_best": overall_best,
        "overall_best_rate": overall_best_rate,
        "n_train": n_train,
        "n_test": n_test,
        "layers_probed": args.layers,
        "n_heads": n_head,
    }

    output_path = output_dir / "phase16_iti_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
