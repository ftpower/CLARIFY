"""Section 12: Logit-space truth direction intervention.

Theory: docs/theory-intervention-failure.md Section 12.
Core claim: Detection (hidden space) and intervention (logit space)
should use DIFFERENT representations. v_logit bypasses the RMSNorm
bottleneck that attenuates hidden-space interventions by ~45x.

Experiment:
  1. Compute v_logit = mean(logits | correct) - mean(logits | wrong)
  2. On test samples: logits += alpha * v_logit at first generation step
  3. Compare with hidden-space v intervention (same calibration set)
  4. Stratify by knowability (rank of y_true in top-K)

Predictions (Section 12.5):
  P1: logit intervention Δ_accuracy > 0 for some alpha
  P2: logit effect > hidden v effect
  P3: v_logit top tokens semantically related to correct answers
  P4: stronger effect on "know but wrong" subset
  P5: Δ_accuracy negatively correlated with detection score

Usage:
    python validate_s12_logit_intervention.py --n_calibrate 200 --n_test 50
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
from common import load_model_and_unembed, greedy_generate, get_first_answer_token_id


# ═════════════════════════════════════════════════════════════════════════════
# 1. Compute v_logit from calibration samples
# ═════════════════════════════════════════════════════════════════════════════


def compute_v_logit(model, tokenizer, samples, device, seed=42):
    """Compute v_logit = mean(logits | correct) - mean(logits | wrong).

    Logits are collected at the last prompt token position (before generation).
    """
    logits_correct = []
    logits_wrong = []

    for s in tqdm(samples, desc="  Calibrate v_logit"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        with torch.no_grad():
            logits = model(tokens)

        # Generate answer greedily
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(19):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits_step = model(tokens)
            nid = int(logits_step[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")

        # Collect logits at last prompt position (float32 on CPU)
        logit_vec = logits[0, -1, :].float().cpu().numpy()

        if is_correct:
            logits_correct.append(logit_vec)
        else:
            logits_wrong.append(logit_vec)

    n_correct = len(logits_correct)
    n_wrong = len(logits_wrong)

    if n_correct == 0 or n_wrong == 0:
        raise RuntimeError(
            f"Need at least 1 correct and 1 wrong sample, got {n_correct}/{n_wrong}"
        )

    logits_correct = np.stack(logits_correct, axis=0)
    logits_wrong = np.stack(logits_wrong, axis=0)

    v_logit = logits_correct.mean(axis=0) - logits_wrong.mean(axis=0)
    v_logit_norm = float(np.linalg.norm(v_logit))

    stats = {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "v_logit_norm": v_logit_norm,
    }

    return torch.from_numpy(v_logit).float().to(device), stats


# ═════════════════════════════════════════════════════════════════════════════
# 2. Generate with logit-space intervention
# ═════════════════════════════════════════════════════════════════════════════


def generate_with_logit_intervention(
    model, tokenizer, prompt, device, v_logit, alpha, max_new=20
):
    """Generate with logit intervention at first step only.

    First forward pass: logits += alpha * v_logit, then argmax.
    Subsequent steps: unmodified autoregressive generation.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits = model(tokens)

    # Intervene on logits at last position
    logits[0, -1, :] += alpha * v_logit

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


# ═════════════════════════════════════════════════════════════════════════════
# 3. Hidden-space v intervention (for comparison)
# ═════════════════════════════════════════════════════════════════════════════


def generate_with_hidden_intervention(
    model, tokenizer, prompt, device, layer, v_hidden, alpha
):
    """Hidden-space intervention at specified layer, first step only."""
    d_f16 = v_hidden.to(dtype=torch.float16)

    def _intervene(act, hook=None):
        act[:, -1, :] += alpha * d_f16.unsqueeze(0)
        return act

    hook_name = f"blocks.{layer}.hook_resid_post"
    return greedy_generate(
        model, tokenizer, prompt, device, fwd_hooks=[(hook_name, _intervene)]
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Knowability classification
# ═════════════════════════════════════════════════════════════════════════════


def classify_knowability(logits, y_true_id, rank_threshold=50):
    """Return rank of y_true in sorted logits."""
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="S12: Logit-space truth direction intervention"
    )
    parser.add_argument(
        "--n_calibrate",
        type=int,
        default=200,
        help="Samples for computing v_logit and v_hidden",
    )
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument(
        "--layer", type=int, default=27, help="Layer for hidden-space v (comparison)"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--skip_hidden", action="store_true", help="Skip hidden-space comparison"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("S12: Logit-Space Truth Direction Intervention")
    print(f"Calibrate: {args.n_calibrate}, Test: {args.n_test}")
    print(f"Alphas: {args.alphas}")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/6] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    vocab_size = W_U.shape[1]
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, vocab={vocab_size}, loaded in {time.time() - t0:.1f}s")

    # ── Compute v_logit ──
    print(f"\n[2/6] Computing v_logit from {args.n_calibrate} samples...")
    calib_samples = load_triviaqa(n_samples=args.n_calibrate, seed=42)
    calib_samples = calib_samples[: args.n_calibrate]

    v_logit, vlogit_stats = compute_v_logit(model, tokenizer, calib_samples, device)
    print(
        f"  correct={vlogit_stats['n_correct']}, "
        f"wrong={vlogit_stats['n_wrong']}, "
        f"||v_logit||={vlogit_stats['v_logit_norm']:.1f}"
    )

    # Inspect top/bottom tokens of v_logit (P3)
    v_logit_np = v_logit.float().cpu().numpy()
    top_k = 20
    top_idx = np.argsort(v_logit_np)[-top_k:][::-1]
    bottom_idx = np.argsort(v_logit_np)[:top_k]
    print(f"\n  Top-{top_k} tokens (boosted by v_logit):")
    for i, idx in enumerate(top_idx):
        tok = tokenizer.decode([int(idx)])
        print(f"    {i + 1:>2d}. {tok!r:20s} ({v_logit_np[idx]:+.4f})")
    print(f"\n  Bottom-{top_k} tokens (suppressed by v_logit):")
    for i, idx in enumerate(bottom_idx):
        tok = tokenizer.decode([int(idx)])
        print(f"    {i + 1:>2d}. {tok!r:20s} ({v_logit_np[idx]:+.4f})")

    # ── Compute v_hidden for comparison ──
    if not args.skip_hidden:
        print(f"\n[3/6] Computing v_hidden at L{args.layer} for comparison...")
        from common import compute_v

        v_hidden, vh_stats = compute_v(
            model, tokenizer, args.n_calibrate, device, args.layer, seed=42
        )
        print(
            f"  correct={vh_stats['n_correct']}, "
            f"wrong={vh_stats['n_incorrect']}, "
            f"||v_hidden||=1.0 (normalized)"
        )
    else:
        print("\n[3/6] Skipping hidden-space v (--skip_hidden)")
        v_hidden = None

    # ── Load test samples and classify ──
    print(f"\n[4/6] Loading {args.n_test} test samples + knowability...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    test_data = []
    classification = {"know_correct": [], "know_wrong": [], "dont_know": []}

    for i, s in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        with torch.no_grad():
            logits = model(tokens)

        # Baseline generation
        baseline_gen = greedy_generate(model, tokenizer, prompt, device)
        baseline_correct = check_correct(baseline_gen, s["answers"], dataset="triviaqa")

        # Knowability
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        rank = classify_knowability(logits, y_true_id) if y_true_id else 99999

        entry = {
            "sample_id": i,
            "prompt": prompt,
            "rank": rank,
            "baseline_correct": baseline_correct,
            "baseline_gen": baseline_gen,
            "y_true_id": y_true_id,
            "answers": s["answers"],
        }

        if y_true_id and rank <= args.rank_threshold:
            if baseline_correct:
                classification["know_correct"].append(i)
            else:
                classification["know_wrong"].append(i)
        else:
            classification["dont_know"].append(i)

        test_data.append(entry)

    n_kc = len(classification["know_correct"])
    n_kw = len(classification["know_wrong"])
    n_dk = len(classification["dont_know"])
    baseline_total = sum(1 for e in test_data if e["baseline_correct"])

    print(f"\n  Know & Correct: {n_kc}/{args.n_test} ({n_kc / args.n_test:.1%})")
    print(
        f"  Know & Wrong:   {n_kw}/{args.n_test} ({n_kw / args.n_test:.1%})  ← TARGET"
    )
    print(f"  Don't Know:     {n_dk}/{args.n_test} ({n_dk / args.n_test:.1%})")
    print(
        f"  Baseline acc:   {baseline_total}/{args.n_test} ({baseline_total / args.n_test:.1%})"
    )

    # ── Run interventions ──
    print(f"\n[5/6] Running interventions...")

    all_results = {}

    # Logit-space intervention
    print("\n  ── Logit-space v_logit ──")
    for alpha in args.alphas:
        correct = 0
        subset_correct = {"know_wrong": 0, "know_correct": 0, "dont_know": 0}

        for entry in tqdm(test_data, desc=f"    α={alpha:+5.1f}", leave=False):
            gen = generate_with_logit_intervention(
                model, tokenizer, entry["prompt"], device, v_logit, alpha
            )
            is_correct = check_correct(gen, entry["answers"], dataset="triviaqa")
            if is_correct:
                correct += 1
                i = entry["sample_id"]
                for subset in ["know_correct", "know_wrong", "dont_know"]:
                    if i in classification[subset]:
                        subset_correct[subset] += 1

        rate = correct / args.n_test
        delta = rate - baseline_total / args.n_test
        all_results[f"logit_α={alpha:+.1f}"] = {
            "correct": correct,
            "total": args.n_test,
            "rate": rate,
            "delta": float(delta),
            "subset_correct": subset_correct,
        }
        kw_delta = subset_correct["know_wrong"] / max(n_kw, 1) if n_kw > 0 else 0.0
        print(
            f"    α={alpha:+5.1f}: {correct}/{args.n_test} = {rate:.1%} "
            f"(Δ={delta:+.1%})  know_wrong: {subset_correct['know_wrong']}/{n_kw} "
            f"({kw_delta:.1%})"
        )

    # Hidden-space intervention (comparison)
    if not args.skip_hidden and v_hidden is not None:
        print("\n  ── Hidden-space v_hidden ──")
        for alpha in args.alphas:
            correct = 0
            subset_correct = {"know_wrong": 0, "know_correct": 0, "dont_know": 0}

            for entry in tqdm(test_data, desc=f"    α={alpha:+5.1f}", leave=False):
                gen = generate_with_hidden_intervention(
                    model,
                    tokenizer,
                    entry["prompt"],
                    device,
                    args.layer,
                    v_hidden,
                    alpha,
                )
                is_correct = check_correct(gen, entry["answers"], dataset="triviaqa")
                if is_correct:
                    correct += 1
                    i = entry["sample_id"]
                    for subset in ["know_correct", "know_wrong", "dont_know"]:
                        if i in classification[subset]:
                            subset_correct[subset] += 1

            rate = correct / args.n_test
            delta = rate - baseline_total / args.n_test
            all_results[f"hidden_α={alpha:+.1f}"] = {
                "correct": correct,
                "total": args.n_test,
                "rate": rate,
                "delta": float(delta),
                "subset_correct": subset_correct,
            }
            kw_delta = subset_correct["know_wrong"] / max(n_kw, 1) if n_kw > 0 else 0.0
            print(
                f"    α={alpha:+5.1f}: {correct}/{args.n_test} = {rate:.1%} "
                f"(Δ={delta:+.1%})  know_wrong: {subset_correct['know_wrong']}/{n_kw} "
                f"({kw_delta:.1%})"
            )

    # ── Summary ──
    print(f"\n[6/6] Summary")
    print(f"\n  {'Method':>20s} {'Best Δ':>10s} {'Best α':>10s} {'KW Δ':>10s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")

    logit_best = max(
        ((k, v) for k, v in all_results.items() if k.startswith("logit_")),
        key=lambda x: x[1]["delta"],
    )
    print(
        f"  {'logit v_logit':>20s} {logit_best[1]['delta']:+9.1%} "
        f"{logit_best[0].split('=')[1]:>10s} "
        f"{logit_best[1]['subset_correct']['know_wrong'] / max(n_kw, 1):+9.1%}"
        if n_kw > 0
        else "N/A"
    )

    if not args.skip_hidden:
        hidden_best = max(
            ((k, v) for k, v in all_results.items() if k.startswith("hidden_")),
            key=lambda x: x[1]["delta"],
        )
        print(
            f"  {'hidden v_hidden':>20s} {hidden_best[1]['delta']:+9.1%} "
            f"{hidden_best[0].split('=')[1]:>10s} "
            f"{hidden_best[1]['subset_correct']['know_wrong'] / max(n_kw, 1):+9.1%}"
            if n_kw > 0
            else "N/A"
        )

    # Gate checks
    p1_passes = logit_best[1]["delta"] > 0.05
    p2_passes = (
        not args.skip_hidden and logit_best[1]["delta"] > hidden_best[1]["delta"]
    )
    p4_passes = (
        n_kw > 0 and (logit_best[1]["subset_correct"]["know_wrong"] / n_kw) > 0.0
    )

    print(f"\n  P1 (logit Δ > 5%): {'PASS' if p1_passes else 'FAIL'}")
    if not args.skip_hidden:
        print(f"  P2 (logit > hidden): {'PASS' if p2_passes else 'FAIL'}")
    print(f"  P4 (know_wrong Δ > 0): {'PASS' if p4_passes else 'FAIL'}")

    # ── Save ──
    output = {
        "config": {
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "layer": args.layer,
            "alphas": args.alphas,
            "rank_threshold": args.rank_threshold,
            "seed": args.seed,
        },
        "v_logit_stats": vlogit_stats,
        "v_logit_top_tokens": [
            {
                "token": tokenizer.decode([int(idx)]),
                "id": int(idx),
                "score": float(v_logit_np[idx]),
            }
            for idx in top_idx
        ],
        "v_logit_bottom_tokens": [
            {
                "token": tokenizer.decode([int(idx)]),
                "id": int(idx),
                "score": float(v_logit_np[idx]),
            }
            for idx in bottom_idx
        ],
        "classification": {
            "know_correct": n_kc,
            "know_wrong": n_kw,
            "dont_know": n_dk,
        },
        "baseline": {
            "correct": baseline_total,
            "total": args.n_test,
            "rate": baseline_total / args.n_test,
        },
        "results": all_results,
        "gates": {"p1": p1_passes, "p2": p2_passes, "p4": p4_passes},
    }

    out_path = output_dir / "s12_logit_intervention.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
