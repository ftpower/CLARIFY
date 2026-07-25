"""Phase 7+9+10 8B: Truth Direction Detection + Multi-Layer Cascade Intervention.

On Qwen3-8B (36 layers, L0-L35). Single script, single model load.

Phase 7: Truth direction detection — AUROC per layer
Phase 9: Multi-state (h/a/m) detection + single-layer additive intervention
Phase 10: Multi-layer cascade intervention
  - Sweet spots for 8B: L17-L24 (proportional to 1.7B's L17-L22 / 28 * 36 ≈ L22-L28)
  - Actually test multiple layer groups

Usage (AutoDL RTX 5090 32GB):
  python run_8b_detection_cascade.py --n_samples 200 --n_test 50
  python run_8b_detection_cascade.py --n_samples 50 --skip_cascade  # quick detection only
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
# Phase 7+9: Detection — Extract states + compute AUROC
# ═══════════════════════════════════════════════════════════════════


def extract_all_states(model, samples, device, layers=None):
    """Extract h/a/m at last token for all specified layers + generate answer.

    Returns list of records with per-layer states and correctness labels.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    n_layers = model.cfg.n_layers
    records = []

    for sample in tqdm(samples, desc="Extracting states"):
        question = sample["question"]
        context = sample.get("context", "")
        answers = sample["answers"]
        prompt = format_prompt(question, context, dataset="triviaqa")

        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        # Capture h/a/m at all requested layers + generate
        storage = {}

        def make_capture(lyr, key):
            hook_name = f"blocks.{lyr}.hook_resid_post"

            def _capture(act, hook=None):
                storage[(lyr, key)] = act[0, input_len - 1, :].clone()
                return act

            return hook_name, _capture

        hooks = []
        for lyr in layers:
            hooks.append(make_capture(lyr, "h"))

        # Generate with hooks to get states at each step
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Greedy generation
        gids = []
        current_tokens = tokens.clone()
        for _step in range(20):
            nid = int(logits[0, -1, :].argmax().item())
            if nid == model.tokenizer.eos_token_id:
                break
            gids.append(nid)
            current_tokens = torch.cat(
                [current_tokens, torch.tensor([[nid]], device=device)], dim=1
            )
            if current_tokens.shape[1] > 1024:
                break
            with torch.no_grad():
                logits = model(current_tokens)

        generated = model.tokenizer.decode(gids).strip()
        is_correct = check_correct(generated, answers, dataset="triviaqa")

        # Also capture a, m states (need separate forward passes for each layer group)
        # For efficiency, capture h only in main pass; a, m in batched passes
        # Simplified: capture h/a/m by doing 3 passes (h, a, m)
        # We do this lazily — only for the best layers after AUROC scan

        rec = {
            "question": question,
            "answers": answers,
            "generated": generated,
            "label": int(is_correct),
            "input_len": int(input_len),
            "h": {},
        }
        for lyr in layers:
            rec["h"][str(lyr)] = storage[(lyr, "h")].cpu().numpy().tolist()

        records.append(rec)

    return records


def extract_attention_mlp_states(model, records, layers, device):
    """Augment existing records with attention (a) and MLP (m) states.

    Captures all requested layers in a SINGLE forward pass per record
    to avoid O(n_layers) passes.
    """
    print("  Extracting attention (a) and MLP (m) states...")

    for i, rec in enumerate(tqdm(records, desc="  Extracting a/m")):
        question = rec["question"]
        prompt = format_prompt(question, "", dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        rec["a"] = {}
        rec["m"] = {}
        storage = {}

        # Build all hooks for all layers in one pass
        hooks = []
        for lyr in layers:
            # Need per-layer closure; capture lyr in default arg
            def make_capture_a(l=lyr):
                def _h(act, hook=None):
                    storage[(l, "a")] = act[0, input_len - 1, :].clone()
                    return act

                return _h

            def make_capture_m(l=lyr):
                def _h(act, hook=None):
                    storage[(l, "m")] = act[0, input_len - 1, :].clone()
                    return act

                return _h

            hooks.append((f"blocks.{lyr}.hook_attn_out", make_capture_a()))
            hooks.append((f"blocks.{lyr}.hook_mlp_out", make_capture_m()))

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=hooks)

        for lyr in layers:
            rec["a"][str(lyr)] = storage[(lyr, "a")].cpu().numpy().tolist()
            rec["m"][str(lyr)] = storage[(lyr, "m")].cpu().numpy().tolist()

    return records


def compute_truth_auroc(records, layers, state_key="h"):
    """Compute AUROC using truth direction dot product.

    Uses leave-one-out-ish approach: for each layer, compute v from all records,
    then dot with each record's vector.
    """
    aurocs = {}
    directions = {}

    for lyr in tqdm(layers, desc=f"  Truth AUROC {state_key}"):
        correct_vecs = []
        wrong_vecs = []
        for rec in records:
            vec = np.array(rec[state_key][str(lyr)], dtype=np.float32)
            if rec["label"] == 1:
                correct_vecs.append(vec)
            else:
                wrong_vecs.append(vec)

        if not correct_vecs or not wrong_vecs:
            aurocs[lyr] = 0.5
            directions[lyr] = np.zeros(
                len(correct_vecs[0]) if correct_vecs else len(wrong_vecs[0])
            )
            continue

        mu_c = np.mean(correct_vecs, axis=0)
        mu_w = np.mean(wrong_vecs, axis=0)
        v = mu_c - mu_w
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[lyr] = v

        scores = []
        labels = []
        for rec in records:
            vec = np.array(rec[state_key][str(lyr)], dtype=np.float32)
            scores.append(float(np.dot(vec, v)))
            labels.append(rec["label"])

        try:
            aurocs[lyr] = float(roc_auc_score(labels, scores))
        except ValueError:
            aurocs[lyr] = 0.5

    return aurocs, directions


# ═══════════════════════════════════════════════════════════════════
# Phase 10: Cascade Intervention
# ═══════════════════════════════════════════════════════════════════

# 8B sweet spot: proportional mapping from 1.7B
# 1.7B sweet spot L17-L22 → 8B equivalent: L22-L28 (17/28*36≈22, 22/28*36≈28)
# Also test original mapping


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


def make_cascade_hooks(layers, directions, alpha, device, norm_balanced=False):
    """Create hooks that add α·v at each specified layer's resid_post.

    If norm_balanced, use α/n_layers to keep total perturbation comparable.
    """
    n_active = max(len(layers), 1)
    effective_alpha = alpha / n_active if norm_balanced else alpha
    hooks = []

    for lyr in layers:
        hook_name = f"blocks.{lyr}.hook_resid_post"
        v = directions.get(lyr)
        if v is None:
            continue
        mod_vec = torch.tensor(effective_alpha * v, dtype=torch.float32, device=device)

        def make_hook(vec):
            def _hook(act, hook=None):
                act[0, -1, :] = act[0, -1, :] + vec
                return act

            return _hook

        hooks.append((hook_name, make_hook(mod_vec)))

    return hooks


# Cascade intervention sets for 8B (36 layers)
CASCADE_SETS_8B = {
    "L22_only": [22],
    "L20_only": [20],
    "Sweet-2": [22, 23],
    "Sweet-3": [22, 23, 24],
    "Sweet-6": [22, 23, 24, 25, 26, 27],
    "All-36": list(range(36)),
    "Late-3": [33, 34, 35],
    # Also test 1.7B proportional mapping
    "L17-22": [17, 18, 19, 20, 21, 22],  # same as 1.7B sweet spot
}


def evaluate_cascade(
    model,
    tokenizer,
    test_records,
    device,
    cascade_name,
    cascade_layers,
    directions,
    alpha,
    norm_balanced=False,
):
    """Evaluate one cascade configuration."""
    hooks = make_cascade_hooks(cascade_layers, directions, alpha, device, norm_balanced)

    correct = 0
    for rec in test_records:
        question = rec["question"]
        context = rec.get("context", "")
        answers = rec["answers"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(model, tokenizer, tokens, device, hooks)
        if check_correct(generated, answers, dataset="triviaqa"):
            correct += 1

    return correct / len(test_records)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7+9+10 8B: Truth Direction Detection + Cascade Intervention"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=[-1.0, -0.5, 0.5, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_cascade",
        action="store_true",
        help="Skip cascade intervention (detection only)",
    )
    parser.add_argument(
        "--skip_am",
        action="store_true",
        help="Skip attention/MLP state extraction (h-only)",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "outputs_8b"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.n_samples}, Test: {args.n_test}")

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/5] Loading data...")
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  Loaded {len(samples)} TriviaQA samples")

    n_test = min(args.n_test, len(samples) // 2)
    n_train = len(samples) - n_test
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]
    print(f"  Train: {n_train}, Test: {n_test}")

    # ── 2. Load model ─────────────────────────────────────────
    print("\n[2/5] Loading model...")
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  Loaded: {n_layers} layers, d_model={d_model}")

    all_layers = list(range(n_layers))

    # ── 3. Extract states + generate ──────────────────────────
    print("\n[3/5] Extracting states + generating...")
    t0 = time.time()

    # Full extraction on train+test combined
    all_records = extract_all_states(model, samples, device, layers=all_layers)

    # Split into train/test records
    train_records = all_records[:n_train]
    test_records = all_records[n_train:]

    train_correct = sum(1 for r in train_records if r["label"] == 1)
    test_correct = sum(1 for r in test_records if r["label"] == 1)
    print(f"  Train correct: {train_correct}/{n_train} = {train_correct / n_train:.1%}")
    print(f"  Test correct:  {test_correct}/{n_test} = {test_correct / n_test:.1%}")
    print(f"  Extraction time: {time.time() - t0:.0f}s")

    # Save extraction for reuse
    extract_path = output_dir / "phase9_8b_extract.json"
    extraction_data = {
        "summary": {
            "n_samples": len(all_records),
            "n_correct": sum(1 for r in all_records if r["label"] == 1),
            "correct_rate": sum(1 for r in all_records if r["label"] == 1)
            / len(all_records),
            "extraction_time_s": time.time() - t0,
        },
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "n_layers": n_layers,
            "d_model": d_model,
            "seed": args.seed,
        },
        "records": all_records,
    }
    with open(extract_path, "w") as f:
        json.dump(extraction_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved extraction to {extract_path}")

    # ── 4. Detection AUROC ────────────────────────────────────
    print("\n[4/5] Computing detection AUROC...")

    # h-state AUROC (Phase 7)
    h_aurocs, h_directions = compute_truth_auroc(train_records, all_layers, "h")
    best_h_layer = max(h_aurocs, key=h_aurocs.get)
    print(f"  h-state AUROC: best L{best_h_layer} = {h_aurocs[best_h_layer]:.4f}")
    print(f"  Top-5 h layers: {sorted(h_aurocs, key=h_aurocs.get, reverse=True)[:5]}")

    # Attention and MLP states (Phase 9)
    if not args.skip_am:
        # Extract a/m for a subset of layers (top h layers + neighbors)
        top_h_layers = sorted(h_aurocs, key=h_aurocs.get, reverse=True)[:10]
        print(f"  Extracting a/m for top {len(top_h_layers)} layers...")

        # Reuse train_records already have h; add a/m
        train_records = extract_attention_mlp_states(
            model, train_records, top_h_layers, device
        )

        a_aurocs, a_directions = compute_truth_auroc(train_records, top_h_layers, "a")
        m_aurocs, m_directions = compute_truth_auroc(train_records, top_h_layers, "m")

        if a_aurocs:
            best_a = max(a_aurocs, key=a_aurocs.get)
            print(f"  a-state AUROC: best L{best_a} = {a_aurocs[best_a]:.4f}")
        if m_aurocs:
            best_m = max(m_aurocs, key=m_aurocs.get)
            print(f"  m-state AUROC: best L{best_m} = {m_aurocs[best_m]:.4f}")
    else:
        a_aurocs, a_directions = {}, {}
        m_aurocs, m_directions = {}, {}

    # ── 5. Cascade Intervention ───────────────────────────────
    if args.skip_cascade:
        print("\n[5/5] Skipping cascade intervention (--skip_cascade)")
    else:
        print("\n[5/5] Cascade intervention...")

        # Baseline generation (no intervention)
        print("  Getting baseline...")
        baseline_correct = 0
        for rec in tqdm(test_records, desc="  Baseline"):
            question = rec["question"]
            context = rec.get("context", "")
            answers = rec["answers"]
            prompt = format_prompt(question, context, dataset="triviaqa")
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            generated = _gen_greedy(model, tokenizer, tokens, device, [])
            if check_correct(generated, answers, dataset="triviaqa"):
                baseline_correct += 1
        baseline_rate = baseline_correct / n_test
        print(f"  Baseline: {baseline_correct}/{n_test} = {baseline_rate:.1%}")

        cascade_results = {}
        best_overall_rate = baseline_rate
        best_overall_config = "baseline"

        for cascade_name, cascade_layers in CASCADE_SETS_8B.items():
            for alpha in args.alphas:
                for norm_balanced in [False, True]:
                    if norm_balanced and len(cascade_layers) <= 1:
                        continue  # norm_balanced is trivial for single layer

                    t_start = time.time()
                    rate = evaluate_cascade(
                        model,
                        tokenizer,
                        test_records,
                        device,
                        cascade_name,
                        cascade_layers,
                        h_directions,
                        alpha,
                        norm_balanced,
                    )
                    elapsed = time.time() - t_start

                    nb_str = "_norm" if norm_balanced else ""
                    key = f"{cascade_name}_α{alpha:+.1f}{nb_str}"
                    delta = rate - baseline_rate
                    cascade_results[key] = {
                        "rate": rate,
                        "delta": float(delta),
                        "time_s": elapsed,
                        "layers": cascade_layers,
                        "alpha": alpha,
                        "norm_balanced": norm_balanced,
                    }
                    print(f"    {key}: {rate:.1%} (Δ={delta:+.1%}) [{elapsed:.0f}s]")

                    if rate > best_overall_rate:
                        best_overall_rate = rate
                        best_overall_config = key

        print(
            f"\n  Best cascade: {best_overall_config} @ {best_overall_rate:.1%} "
            f"(Δ={best_overall_rate - baseline_rate:+.1%})"
        )

    # ── Save results ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8B RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Detection (h): L{best_h_layer} AUROC={h_aurocs[best_h_layer]:.4f}")
    if not args.skip_cascade:
        print(f"  Cascade baseline: {baseline_rate:.1%}")
        print(f"  Cascade best: {best_overall_config} @ {best_overall_rate:.1%}")

    results = {
        "config": {
            "model": args.model,
            "n_samples": args.n_samples,
            "n_test": n_test,
            "n_layers": n_layers,
            "d_model": d_model,
            "seed": args.seed,
            "alphas": args.alphas,
        },
        "detection": {
            "h_best_layer": best_h_layer,
            "h_best_auroc": h_aurocs[best_h_layer],
            "h_aurocs": {str(k): v for k, v in h_aurocs.items()},
            "a_aurocs": {str(k): v for k, v in a_aurocs.items()} if a_aurocs else {},
            "m_aurocs": {str(k): v for k, v in m_aurocs.items()} if m_aurocs else {},
        },
    }

    if not args.skip_cascade:
        results["cascade"] = {
            "baseline_rate": baseline_rate,
            "best_config": best_overall_config,
            "best_rate": best_overall_rate,
            "results": {
                k: {"rate": v["rate"], "delta": v["delta"], "time_s": v["time_s"]}
                for k, v in cascade_results.items()
            },
        }

    results_path = output_dir / "phase7_9_10_8b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")
    print(f"Extraction data saved to {extract_path}")


if __name__ == "__main__":
    main()
