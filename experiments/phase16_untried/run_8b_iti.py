"""Phase 16 ITI 8B: Multi-Head Truth Probe + Attention Shift on Qwen3-8B.

ITI (Li et al. 2023) shifts attention head outputs along probe direction.
This is the most promising alternative intervention paradigm — modifies information
routing rather than adding a vector to the residual stream.

Usage (AutoDL RTX 5090 32GB):
  python run_8b_iti.py --n_samples 200 --n_test 50
  python run_8b_iti.py --n_samples 50 --layers 22 23 24  --quick  # fast test
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
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════
# Phase 16.1: Train Multi-Head Probes
# ═══════════════════════════════════════════════════════════════════


def extract_head_outputs(model, records, layer, device):
    """Extract per-head z vectors at last token position.

    Returns X: [n, n_heads, d_head], y: [n].
    """
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    n = len(records)
    hook_name = f"blocks.{layer}.attn.hook_z"

    X = np.zeros((n, n_heads, d_head), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for i, rec in enumerate(tqdm(records, desc=f"  Extracting L{layer} heads")):
        question = rec["question"]
        context = rec.get("context", "")
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        storage = {}

        def _capture(act, hook=None):
            storage["z"] = act[0, -1, :, :].clone()
            return act

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])

        X[i] = storage["z"].cpu().numpy()
        y[i] = rec["label"]

    return X, y


def train_head_probes(X_train, y_train, X_val, y_val):
    """Train logistic regression probe per head. Returns probes sorted by AUROC."""
    n_heads = X_train.shape[1]
    probes = []

    for h in range(n_heads):
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=1000, random_state=42
        )
        clf.fit(X_train[:, h, :], y_train)
        proba = clf.predict_proba(X_val[:, h, :])[:, 1]
        auroc = roc_auc_score(y_val, proba)
        coef = clf.coef_[0].astype(np.float32)  # [d_head]
        coef = coef / (np.linalg.norm(coef) + 1e-8)
        probes.append(
            {
                "head": h,
                "auroc": float(auroc),
                "coef": coef.tolist(),
                "intercept": float(clf.intercept_[0]),
            }
        )

    probes.sort(key=lambda p: p["auroc"], reverse=True)
    return probes


# ═══════════════════════════════════════════════════════════════════
# Phase 16.2: ITI Attention Shift Intervention
# ═══════════════════════════════════════════════════════════════════


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20):
    """Greedy generation with hooks."""
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


def make_iti_hook(layer, top_k_probes, alpha, device):
    """Create hook that shifts top-K attention heads by α·v_head.

    Modifies blocks.{layer}.attn.hook_z: [batch, seq, n_heads, d_head].
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    shifts = {}
    for probe in top_k_probes:
        h = probe["head"]
        vec = torch.tensor(
            alpha * np.array(probe["coef"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        shifts[h] = vec

    def _hook(act, hook=None):
        for h, vec in shifts.items():
            act[0, -1, h, :] = act[0, -1, h, :] + vec
        return act

    return hook_name, _hook


def evaluate_iti(model, tokenizer, test_records, device, layer, top_k_probes, alpha):
    """Evaluate ITI intervention on test set."""
    hook_name, hook_fn = make_iti_hook(layer, top_k_probes, alpha, device)

    correct = 0
    for rec in test_records:
        question = rec["question"]
        context = rec.get("context", "")
        answers = rec.get("gt_answers") or rec.get("answers") or rec["gt_answer"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(
            model, tokenizer, tokens, device, [(hook_name, hook_fn)]
        )
        if check_correct(generated, answers, dataset="triviaqa"):
            correct += 1

    return correct / len(test_records)


def probe_all_layers(model, train_records, probe_train, probe_val, layers, device):
    """Train probes for all specified layers. Returns dict of layer→probes."""
    all_probes = {}
    all_aurocs = {}

    for layer in layers:
        print(f"\n  ── Layer {layer} ──")
        X_tr, y_tr = extract_head_outputs(model, probe_train, layer, device)
        X_v, y_v = extract_head_outputs(model, probe_val, layer, device)

        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        print(
            f"    Train: {n_pos} pos / {n_neg} neg (ratio: {min(n_pos, n_neg) / max(n_pos, n_neg):.2f})"
        )

        probes = train_head_probes(X_tr, y_tr, X_v, y_v)
        all_probes[layer] = probes
        all_aurocs[layer] = probes[0]["auroc"]

        print(
            f"    Top-5 heads: "
            + ", ".join(f"H{p['head']}={p['auroc']:.3f}" for p in probes[:5])
        )

    return all_probes, all_aurocs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 16 ITI 8B: Multi-Head Truth Probe + Attention Shift"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to probe (default: auto-select from sweet spot)",
    )
    parser.add_argument("--top_k_values", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[-1.0, -0.5, 0.5, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--use_extraction",
        default=None,
        help="Path to pre-extracted 8B data JSON (skip extraction)",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "outputs_phase16_8b"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/4] Loading data...")

    if args.use_extraction:
        print(f"  Loading pre-extracted data from {args.use_extraction}")
        with open(args.use_extraction) as f:
            data = json.load(f)
        all_records = data["records"]
        n_test = min(args.n_test, len(all_records) // 2)
        n_train = len(all_records) - n_test
        train_records = all_records[:n_train]
        test_records = all_records[n_train:]
    else:
        samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
        n_test = min(args.n_test, len(samples) // 2)
        n_train = len(samples) - n_test
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]

        # We need to generate + label the data first
        # For simplicity, we generate and extract heads in a combined pass
        print(f"  Loaded {len(samples)} samples")

    print(f"  Train: {n_train}, Test: {n_test}")

    # ── 2. Load model ─────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"  Loaded: {n_layers} layers, {n_heads} heads, d_head={d_head}")

    # Select layers to probe
    if args.layers is None:
        # 8B sweet spot: L20-L28 (proportional to 1.7B L17-L22)
        probe_layers = [20, 22, 24, 26, 28]
        print(f"  Auto-selected probe layers: {probe_layers}")
    else:
        probe_layers = args.layers

    # ── Get labeled records ───────────────────────────────────
    if not args.use_extraction:
        # Quick generation pass to get labels
        print("\n  Generating answers for labels...")
        train_records = []
        for sample in tqdm(train_samples, desc="  Train labels"):
            question = sample["question"]
            context = sample.get("context", "")
            answers = sample["answers"]
            prompt = format_prompt(question, context, dataset="triviaqa")
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            generated = _gen_greedy(model, tokenizer, tokens, device, [])
            is_correct = check_correct(generated, answers, dataset="triviaqa")
            train_records.append(
                {
                    "question": question,
                    "context": context,
                    "answers": answers,
                    "label": int(is_correct),
                }
            )

        test_records = []
        for sample in tqdm(test_samples, desc="  Test labels"):
            question = sample["question"]
            context = sample.get("context", "")
            answers = sample["answers"]
            prompt = format_prompt(question, context, dataset="triviaqa")
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            generated = _gen_greedy(model, tokenizer, tokens, device, [])
            is_correct = check_correct(generated, answers, dataset="triviaqa")
            test_records.append(
                {
                    "question": question,
                    "context": context,
                    "answers": answers,
                    "label": int(is_correct),
                    "generated": generated,
                }
            )

        train_correct = sum(1 for r in train_records if r["label"] == 1)
        test_correct = sum(1 for r in test_records if r["label"] == 1)
        print(
            f"  Train correct: {train_correct}/{n_train} = {train_correct / n_train:.1%}"
        )
        print(f"  Test correct:  {test_correct}/{n_test} = {test_correct / n_test:.1%}")

    # ── 3. Train probes ───────────────────────────────────────
    print("\n[3/4] Training head probes...")

    # Split train for probe validation
    n_val = min(30, n_train // 3)
    probe_train = train_records[:-n_val]
    probe_val = train_records[-n_val:]

    all_probes, all_aurocs = probe_all_layers(
        model, train_records, probe_train, probe_val, probe_layers, device
    )

    # ── 4. ITI Intervention ───────────────────────────────────
    print("\n[4/4] ITI attention shift intervention...")

    # Baseline (no intervention)
    baseline_correct = 0
    for rec in tqdm(test_records, desc="  Baseline"):
        question = rec["question"]
        context = rec.get("context", "")
        answers = rec.get("gt_answers") or rec.get("answers") or rec["gt_answer"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(model, tokenizer, tokens, device, [])
        if check_correct(generated, answers, dataset="triviaqa"):
            baseline_correct += 1
    baseline_rate = baseline_correct / n_test
    print(f"  Baseline: {baseline_correct}/{n_test} = {baseline_rate:.1%}")

    # ITI grid search
    iti_results = {}
    best_rate = baseline_rate
    best_config = "baseline"

    for layer in probe_layers:
        probes = all_probes[layer]
        n_heads_avail = len(probes)
        print(f"\n  L{layer} (best head AUROC={probes[0]['auroc']:.4f}):")

        for K in args.top_k_values:
            K_actual = min(K, n_heads_avail)
            top_k = probes[:K_actual]

            for alpha in args.alphas:
                t0 = time.time()
                rate = evaluate_iti(
                    model, tokenizer, test_records, device, layer, top_k, alpha
                )
                elapsed = time.time() - t0

                key = f"L{layer}_K{K_actual}_α{alpha:+.1f}"
                delta = rate - baseline_rate
                iti_results[key] = {
                    "rate": rate,
                    "delta": float(delta),
                    "time_s": elapsed,
                }
                print(f"    {key}: {rate:.1%} (Δ={delta:+.1%}) [{elapsed:.0f}s]")

                if rate > best_rate:
                    best_rate = rate
                    best_config = key

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ITI 8B RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline: {baseline_rate:.1%}")
    print(
        f"  Best probe AUROC: "
        + ", ".join(f"L{l}={all_aurocs[l]:.3f}" for l in probe_layers)
    )
    print(
        f"  Best ITI: {best_config} @ {best_rate:.1%} (Δ={best_rate - baseline_rate:+.1%})"
    )

    if best_rate <= baseline_rate:
        print("  ⚠ ITI 8B zero effect — probe direction can detect but not control")

    # ── Save ──────────────────────────────────────────────────
    results = {
        "config": {
            "model": args.model,
            "n_samples": args.n_samples
            if not args.use_extraction
            else "from_extraction",
            "n_test": n_test,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "seed": args.seed,
            "probe_layers": probe_layers,
            "alphas": args.alphas,
            "top_k_values": args.top_k_values,
        },
        "probes": {
            str(l): {
                "best_auroc": all_aurocs[l],
                "head_aurocs": {str(p["head"]): p["auroc"] for p in all_probes[l]},
            }
            for l in probe_layers
        },
        "intervention": {
            "baseline_rate": baseline_rate,
            "best_config": best_config,
            "best_rate": best_rate,
            "results": iti_results,
        },
    }

    results_path = output_dir / "phase16_8b_iti_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
