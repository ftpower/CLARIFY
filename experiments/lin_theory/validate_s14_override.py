"""Phase 14a: Override Direction Construction & Diagnosis.

Theory: docs/theory-intervention-failure.md Section 14.3

v_override = mean(h[know_wrong]) - mean(h[know_correct])
  - Both subsets have rank(y_true) <= 50 (model "knows" the answer)
  - Difference is purely expression/override signal, not knowledge quantity

Gates:
  O1: ||v_override|| > 0.01 * ||v_classic||  (must be non-zero)
  O2: 0.3 < cos(v_override, v_classic) < 0.9  (distinct from classic direction)
  O7: ||v_override|| late layers > early layers  (override is a late-layer phenomenon)

If O1 or O2 fails → override hypothesis falsified → skip Phase 14b, go to 14c directly.

Usage:
    python validate_s14_override.py --n_calibrate 300 --rank_threshold 50
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
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

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import load_model_and_unembed, get_first_answer_token_id, greedy_generate


# ═════════════════════════════════════════════════════════════════════════════
# Core: extract hidden states at ALL layers in one forward pass
# ═════════════════════════════════════════════════════════════════════════════


def extract_all_layers(model, tokenizer, prompt, device):
    """Forward pass, capturing resid_post at every layer (last token position).

    Returns:
        h_dict: dict[layer_idx] -> np.ndarray [d_model] float32
        logits: tensor [1, 1, vocab_size] on device
        tokens: token tensor
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    n_layers = model.cfg.n_layers
    residual = {}

    hooks = []
    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_post"

        def _make_hook(l):
            def _hook(act, hook=None):
                residual[l] = act[:, -1, :].detach().float().cpu().numpy().flatten()
                return act

            return _hook

        hooks.append((hook_name, _make_hook(layer)))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    return residual, logits, tokens


def get_y_true_rank(logits, y_true_id):
    """Get rank of y_true in sorted logits (rank 0 = highest probability)."""
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 14a: Override direction construction & diagnosis"
    )
    parser.add_argument(
        "--n_calibrate",
        type=int,
        default=300,
        help="Number of TriviaQA calibration samples",
    )
    parser.add_argument(
        "--rank_threshold",
        type=int,
        default=50,
        help="Rank threshold for 'know' classification",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 64)
    print("Phase 14a: Override Direction Construction & Diagnosis")
    print(f"n_calibrate={args.n_calibrate}, rank_threshold={args.rank_threshold}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  Qwen3-1.7B: {n_layers} layers, d_model={d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Load calibration samples ──
    print(
        f"\n[2/5] Loading {args.n_calibrate} calibration samples (seed={args.seed})..."
    )
    samples = load_triviaqa(n_samples=args.n_calibrate, seed=args.seed)
    samples = samples[: args.n_calibrate]
    print(f"  Loaded {len(samples)} samples")

    # ── Forward pass all samples, capture all layers ──
    print(f"\n[3/5] Forward pass + classification...")
    print(f"  Extracting h_ℓ for all {n_layers} layers per sample")

    # Storage: per-layer lists for classification
    classification = {
        "know_correct": [],  # rank <= threshold AND generated correctly
        "know_wrong": [],  # rank <= threshold AND generated incorrectly
        "dont_know": [],  # rank > threshold
    }

    # Also store per-sample metadata for v_classic computation
    all_h = {l: {"correct": [], "incorrect": []} for l in range(n_layers)}

    skipped = 0
    for i, sample in enumerate(tqdm(samples, desc="  Forward + classify")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            skipped += 1
            continue

        # Extract hidden states at ALL layers
        h_dict, logits, tokens = extract_all_layers(model, tokenizer, prompt, device)

        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        entry = {
            "sample_id": i,
            "rank": rank,
            "is_correct": is_correct,
            "h_layers": h_dict,  # dict[layer] -> np.ndarray[d_model]
            "y_true_id": y_true_id,
            "prompt": prompt,
            "answers": sample["answers"],
            "question": sample["question"][:80],
            "generated": gen_text,
        }

        # Classify by knowability
        if rank <= args.rank_threshold:
            if is_correct:
                classification["know_correct"].append(entry)
            else:
                classification["know_wrong"].append(entry)
        else:
            classification["dont_know"].append(entry)

        # Accumulate for v_classic
        for l in range(n_layers):
            if is_correct:
                all_h[l]["correct"].append(h_dict[l])
            else:
                all_h[l]["incorrect"].append(h_dict[l])

    n_kc = len(classification["know_correct"])
    n_kw = len(classification["know_wrong"])
    n_dk = len(classification["dont_know"])
    n_valid = n_kc + n_kw + n_dk

    print(f"\n  Classification (rank threshold ≤ {args.rank_threshold}):")
    print(
        f"  Know & Correct: {n_kc}/{n_valid} ({n_kc / n_valid:.1%})"
        if n_valid
        else "  No valid samples"
    )
    print(f"  Know & Wrong:   {n_kw}/{n_valid} ({n_kw / n_valid:.1%})  ← TARGET")
    print(
        f"  Don't Know:     {n_dk}/{n_valid} ({n_dk / n_valid:.1%})" if n_valid else ""
    )
    print(f"  Skipped:        {skipped} (no valid y_true_id)")

    if n_kw == 0:
        print("\n  ❌ FATAL: No know-wrong samples! Cannot compute v_override.")
        print("     Try increasing n_calibrate or lowering rank_threshold.")
        return

    if n_kc == 0:
        print("\n  ❌ FATAL: No know-correct samples! Cannot compute v_override.")
        return

    # ── Compute v_override(ℓ) and v_classic(ℓ) per layer ──
    print(f"\n[4/5] Computing v_override and v_classic per layer...")

    results_per_layer = {}

    for l in range(n_layers):
        # v_override = mean(h[know_wrong]) - mean(h[know_correct])
        h_kw = np.stack(
            [e["h_layers"][l] for e in classification["know_wrong"]], axis=0
        )
        h_kc = np.stack(
            [e["h_layers"][l] for e in classification["know_correct"]], axis=0
        )

        v_override_raw = h_kw.mean(axis=0) - h_kc.mean(axis=0)
        v_override_norm = float(np.linalg.norm(v_override_raw))
        v_override = v_override_raw / (v_override_norm + 1e-10)

        # v_classic = mean(h[correct]) - mean(h[incorrect])
        h_corr = np.stack(all_h[l]["correct"], axis=0)
        h_incorr = np.stack(all_h[l]["incorrect"], axis=0)

        v_classic_raw = h_corr.mean(axis=0) - h_incorr.mean(axis=0)
        v_classic_norm = float(np.linalg.norm(v_classic_raw))
        v_classic = v_classic_raw / (v_classic_norm + 1e-10)

        # Cosine similarity
        cos_sim = float(np.dot(v_override, v_classic))

        results_per_layer[l] = {
            "v_override_norm": v_override_norm,
            "v_classic_norm": v_classic_norm,
            "norm_ratio": v_override_norm / (v_classic_norm + 1e-10),
            "cos_sim": cos_sim,
            "n_kw": n_kw,
            "n_kc": n_kc,
            "n_correct": len(all_h[l]["correct"]),
            "n_incorrect": len(all_h[l]["incorrect"]),
        }

    # ── Gate verification ──
    print(f"\n[5/5] Gate verification")
    print(
        f"\n{'Layer':>6s}  {'||v_over||':>10s}  {'||v_clas||':>10s}  "
        f"{'Ratio':>8s}  {'cos(v_o,v_c)':>13s}  {'O1':>5s}  {'O2':>5s}"
    )
    print(
        f"{'=' * 6}  {'=' * 10}  {'=' * 10}  {'=' * 8}  {'=' * 13}  {'=' * 5}  {'=' * 5}"
    )

    # O1: ||v_override|| > 0.01 * ||v_classic||
    # O2: 0.3 < cos(v_override, v_classic) < 0.9
    o1_threshold = 0.01

    o1_pass = []
    o2_pass = []
    for l in range(n_layers):
        r = results_per_layer[l]
        o1 = r["norm_ratio"] > o1_threshold
        # O2: |cos| in (0.3, 0.9) — anti-alignment also valid
        # (v_override ≈ -v_classic means override reverses truth direction)
        o2 = 0.3 < abs(r["cos_sim"]) < 0.9
        o1_pass.append(o1)
        o2_pass.append(o2)
        o1_flag = "✅" if o1 else "❌"
        o2_flag = "✅" if o2 else "❌"
        print(
            f"  L{l:>4d}  {r['v_override_norm']:>10.4f}  {r['v_classic_norm']:>10.4f}  "
            f"{r['norm_ratio']:>8.4f}  {r['cos_sim']:>+13.4f}  {o1_flag:>5s}  {o2_flag:>5s}"
        )

    # O7: v_override norm should increase with layer depth (late-layer phenomenon)
    # Test: Spearman rank correlation between layer index and norm
    from scipy.stats import spearmanr

    layers_arr = np.arange(n_layers)
    norms_arr = np.array(
        [results_per_layer[l]["v_override_norm"] for l in range(n_layers)]
    )
    spearman_r, spearman_p = spearmanr(layers_arr, norms_arr)
    o7_pass = bool(spearman_r > 0.3 and spearman_p < 0.05)
    o7_flag = "✅" if o7_pass else "❌"

    # Summary
    n_layers_o1 = sum(o1_pass)
    n_layers_o2 = sum(o2_pass)
    best_layer = max(
        range(n_layers), key=lambda l: results_per_layer[l]["v_override_norm"]
    )

    print(f"\n  Gate summary:")
    print(
        f"  O1 (||v_override|| > {o1_threshold}·||v_classic||): "
        f"{'✅' if n_layers_o1 > n_layers // 2 else '❌'} "
        f"({n_layers_o1}/{n_layers} layers pass, best ratio L{best_layer}={results_per_layer[best_layer]['norm_ratio']:.4f})"
    )
    best_cos_layer = max(
        range(n_layers), key=lambda l: abs(results_per_layer[l]["cos_sim"] - 0.6)
    )
    print(
        f"  O2 (0.3 < |cos| < 0.9): "
        f"{'✅' if n_layers_o2 > n_layers // 2 else '❌'} "
        f"({n_layers_o2}/{n_layers} layers pass, "
        f"best |cos| L{best_cos_layer}={abs(results_per_layer[best_cos_layer]['cos_sim']):.4f})"
    )
    print(
        f"  O7 (norm ↑ with layer): "
        f"{o7_flag} "
        f"(Spearman r={spearman_r:+.3f}, p={spearman_p:.4f})"
    )

    # Overall gate decision
    o1_overall = bool(n_layers_o1 > n_layers // 2)
    o2_overall = bool(n_layers_o2 > n_layers // 2)

    if o1_overall and o2_overall:
        print(f"\n  ✅ ALL GATES PASS — override direction EXISTS and is DISTINCT")
        print(f"     → Proceed to Phase 14b (anti-override intervention)")
    elif not o1_overall:
        print(f"\n  ❌ O1 FAILED — v_override is near-zero")
        print(f"     Override hypothesis FALSIFIED in hidden space")
        print(f"     → Skip 14b, go to Phase 14c (TLDC) only")
        print(f"     → Alternative: check attention patterns for override mechanism")
    elif not o2_overall:
        abs_cos_values = [abs(r["cos_sim"]) for r in results_per_layer.values()]
        if all(ac > 0.9 for ac in abs_cos_values):
            print(f"\n  ❌ O2 FAILED — |cos(v_override, v_classic)| ≈ 1")
            print(f"     v_override ≈ ±v_classic — same direction (just sign-flipped)")
            print(
                f"     Override = classic direction → override hypothesis needs revision"
            )
            print(f"     → Skip 14b, go to Phase 14c (TLDC) only")
        elif all(ac < 0.3 for ac in abs_cos_values):
            print(f"\n  ❌ O2 FAILED — |cos| too low (v_override ⊥ v_classic)")
            print(f"     v_override is orthogonal to v_classic")
            print(f"     → Unexpected pattern, investigate further")
        else:
            print(f"\n  ⚠️  O2 MIXED — |cos| varies across layers")

    # ── Save results ──
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "rank_threshold": args.rank_threshold,
            "n_layers": n_layers,
            "d_model": d_model,
            "seed": args.seed,
        },
        "classification_counts": {
            "know_correct": n_kc,
            "know_wrong": n_kw,
            "dont_know": n_dk,
            "skipped": skipped,
        },
        "gates": {
            "O1": {
                "pass": o1_overall,
                "n_layers_pass": n_layers_o1,
                "threshold": o1_threshold,
                "best_ratio": float(results_per_layer[best_layer]["norm_ratio"]),
                "best_layer": best_layer,
            },
            "O2": {
                "pass": o2_overall,
                "n_layers_pass": n_layers_o2,
                "cos_range": [
                    float(min(r["cos_sim"] for r in results_per_layer.values())),
                    float(max(r["cos_sim"] for r in results_per_layer.values())),
                ],
            },
            "O7": {
                "pass": o7_pass,
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            },
            "overall_pass": bool(o1_overall and o2_overall),
        },
        "per_layer": {str(l): results_per_layer[l] for l in range(n_layers)},
        "best_override_layer": best_layer,
    }

    # Keep sample info but strip large h_layers arrays
    light_samples = {}
    for cat in ["know_correct", "know_wrong", "dont_know"]:
        light_samples[cat] = []
        for e in classification[cat]:
            light_samples[cat].append(
                {
                    "sample_id": e["sample_id"],
                    "rank": e["rank"],
                    "is_correct": e["is_correct"],
                    "question": e["question"],
                    "generated": e["generated"],
                    "answers": e["answers"],
                }
            )
    output["samples"] = light_samples

    out_path = output_dir / "s14_override_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # ── Layer-wise norm plot data ──
    print(f"\n  v_override norm per layer (for O7):")
    for l in range(n_layers):
        bar = "█" * int(results_per_layer[l]["v_override_norm"] / max(norms_arr) * 40)
        print(f"  L{l:>4d}: {results_per_layer[l]['v_override_norm']:>8.4f}  {bar}")


if __name__ == "__main__":
    main()
