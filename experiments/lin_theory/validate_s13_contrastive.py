"""Phase 13.1: Contrastive Prompt Decoding for Hallucination Intervention.

Theory: docs/theory-intervention-failure.md Section 13.2.
Core claim: l_truth - l_std captures the model's "truthful mode" bias,
preserving question-conditional information that global v loses.

Experiment:
  1. For each test sample, run forward pass with two prompts:
     - Standard: "... Answer:"
     - Truthful: "... Answer truthfully and accurately:"
  2. Combined logits: l_final = l_std + alpha * (l_truth - l_std)
  3. Greedy decode from combined logits
  4. Compare accuracy vs baseline (standard prompt, no intervention)

Predictions (Section 13.2.3):
  C1: ||l_truth - l_std|| > 0 (prompt change actually affects logits)
  C2: Δ accuracy > 0 for optimal alpha (gate: Δ > 5%)
  C3: Know-wrong subset Δ > All Δ
  C4: truthful prompt P(y_true) > standard P(y_true) on know-wrong subset

Usage:
    python validate_s13_contrastive.py --n_test 100 --n_calibrate 50
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
from common import (
    load_model_and_unembed,
    greedy_generate,
    get_first_answer_token_id,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Prompt construction
# ═════════════════════════════════════════════════════════════════════════════


def make_contrastive_prompts(question, context=""):
    """Build standard and truthful prompt variants for the same question.

    The only difference is the final instruction:
      - std:    "... Answer:"
      - truth:  "... Answer truthfully and accurately:"

    Returns (prompt_std, prompt_truth).
    """
    prompt_std = format_prompt(question, context, dataset="triviaqa")

    # Build truthful variant by replacing the trailing "Answer:" with
    # "Answer truthfully and accurately:" — this is the only difference.
    # format_prompt always ends with "\n\nAnswer:", so we replace that suffix.
    if prompt_std.endswith("Answer:"):
        prompt_truth = (
            prompt_std[: -len("Answer:")] + "Answer truthfully and accurately:"
        )
    else:
        # Fallback: just in case the format changes
        prompt_truth = prompt_std.replace(
            "Answer:", "Answer truthfully and accurately:", 1
        )

    return prompt_std, prompt_truth


# ═════════════════════════════════════════════════════════════════════════════
# 2. Get logits from a prompt (single forward pass)
# ═════════════════════════════════════════════════════════════════════════════


def get_logits(model, prompt, device):
    """Run forward pass and return logits at last prompt token position.

    Returns:
        logits: [vocab_size] float32 on CPU
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        out = model(tokens)

    return out[0, -1, :].float().cpu()


# ═════════════════════════════════════════════════════════════════════════════
# 3. Generate with contrastive logit intervention
# ═════════════════════════════════════════════════════════════════════════════


def generate_contrastive(
    model, tokenizer, prompt_std, prompt_truth, device, alpha, max_new=20
):
    """Generate with contrastive logit combination.

    1. Forward both prompts to get l_std and l_truth
    2. l_final = l_std + alpha * (l_truth - l_std)
    3. Greedy decode from l_final

    The combined logits are used ONLY for the first token.
    Subsequent tokens use standard autoregressive generation from prompt_std.
    """
    l_std = get_logits(model, prompt_std, device)
    l_truth = get_logits(model, prompt_truth, device)

    # Combined logits for first token
    l_combined = l_std + alpha * (l_truth - l_std)

    # Now generate: first token from l_combined, rest autoregressive
    tokens = model.to_tokens(prompt_std, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    # Pick first token from combined logits
    nid = int(l_combined.argmax().item())
    gids = [nid]

    # Continue autoregressively
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
# 4. Knowability classification (reuse from S12)
# ═════════════════════════════════════════════════════════════════════════════


def classify_knowability(model, tokenizer, prompt_std, device, answers):
    """Classify sample as know-correct / know-wrong / don't-know.

    Knowability: rank of y_true first token in the model's logit distribution.
    rank <= 50 → "know", else "don't know".
    """
    l_std = get_logits(model, prompt_std, device)
    y_true_id = get_first_answer_token_id(tokenizer, answers)

    if y_true_id is None:
        return None, None, None

    sorted_ids = l_std.argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()

    # Generate baseline answer
    nid = int(l_std.argmax().item())
    gids = [nid]
    tokens = model.to_tokens(prompt_std, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    for _ in range(19):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    is_correct = check_correct(ans, answers, dataset="triviaqa")

    # P(y_true) under standard prompt
    l_std_f32 = l_std.float()
    probs = torch.softmax(l_std_f32, dim=-1)
    p_true_std = float(probs[y_true_id].item())

    # P(y_true) under truthful prompt
    prompt_truth = make_contrastive_prompts(*(_extract_qc(prompt_std)))[1]
    l_truth = get_logits(model, prompt_truth, device)
    probs_truth = torch.softmax(l_truth.float(), dim=-1)
    p_true_truth = float(probs_truth[y_true_id].item())

    knowability_info = {
        "rank": rank,
        "y_true_id": y_true_id,
        "p_true_std": p_true_std,
        "p_true_truth": p_true_truth,
        "baseline_correct": is_correct,
        "baseline_gen": ans,
    }

    if rank <= 50:
        if is_correct:
            return "know_correct", knowability_info, l_std
        else:
            return "know_wrong", knowability_info, l_std
    else:
        return "dont_know", knowability_info, l_std


def _extract_qc(prompt_std):
    """Extract (question, context) from a standard prompt string. Hacky but works."""
    # The prompt format is:
    # Based on the provided context, ...\n\nContext: {c}\n\nQuestion: {q}\n\nAnswer:
    question = ""
    context = ""
    for line in prompt_std.split("\n"):
        if line.startswith("Question:"):
            question = line[len("Question:") :].strip()
        elif line.startswith("Context:"):
            context = line[len("Context:") :].strip()
    return question, context


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="S13.1: Contrastive Prompt Decoding")
    parser.add_argument(
        "--n_test", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--n_calibrate",
        type=int,
        default=50,
        help="Samples for quick calibration/analysis (not used for v)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("Phase 13.1: Contrastive Prompt Decoding")
    print(f"Test samples: {args.n_test}")
    print(f"Alphas: {args.alphas}")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    vocab_size = W_U.shape[1]
    d_model = model.cfg.d_model
    print(f"  d_model={d_model}, vocab={vocab_size}, loaded in {time.time() - t0:.1f}s")

    # ── Load test samples ──
    print(f"\n[2/5] Loading {args.n_test} test samples...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    test_samples = test_samples[: args.n_test]

    # ── Classify samples ──
    print(
        f"\n[3/5] Classifying samples (knowability + P(y_true) under both prompts)..."
    )
    test_data = []
    classification = {"know_correct": [], "know_wrong": [], "dont_know": []}

    for i, s in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt_std, prompt_truth = make_contrastive_prompts(
            s["question"], s.get("context", "")
        )

        category, kinfo, _ = classify_knowability(
            model, tokenizer, prompt_std, device, s["answers"]
        )

        if category is None:
            continue  # skip samples where y_true_id is None

        entry = {
            "sample_id": i,
            "prompt_std": prompt_std,
            "prompt_truth": prompt_truth,
            "answers": s["answers"],
            "question": s["question"],
            **kinfo,
        }
        test_data.append(entry)
        classification[category].append(i)

    n_kc = len(classification["know_correct"])
    n_kw = len(classification["know_wrong"])
    n_dk = len(classification["dont_know"])
    n_total = len(test_data)
    baseline_total = sum(1 for e in test_data if e["baseline_correct"])

    print(f"\n  Know & Correct: {n_kc}/{n_total} ({n_kc / n_total:.1%})")
    print(f"  Know & Wrong:   {n_kw}/{n_total} ({n_kw / n_total:.1%})  ← TARGET")
    print(f"  Don't Know:     {n_dk}/{n_total} ({n_dk / n_total:.1%})")
    print(
        f"  Baseline acc:   {baseline_total}/{n_total} ({baseline_total / n_total:.1%})"
    )

    # ── Check C1: does truthful prompt actually change logits? ──
    l_diff_norms = []
    p_true_diffs = []
    for entry in test_data:
        l_std = get_logits(model, entry["prompt_std"], device)
        l_truth = get_logits(model, entry["prompt_truth"], device)
        diff = (l_truth - l_std).float()
        l_diff_norms.append(float(torch.norm(diff).item()))
        p_true_diffs.append(entry["p_true_truth"] - entry["p_true_std"])

    mean_diff_norm = np.mean(l_diff_norms)
    mean_p_true_diff = np.mean(p_true_diffs)

    print(
        f"\n  C1: ||l_truth - l_std|| = {mean_diff_norm:.2f} (mean over {n_total} samples)"
    )
    print(f"  C4: mean ΔP(y_true) = {mean_p_true_diff:+.4f} (truthful - std)")
    c1_passes = mean_diff_norm > 1.0  # non-trivial change
    print(f"  C1 gate (||diff|| > 1.0): {'PASS' if c1_passes else 'FAIL'}")

    # C4: check on know-wrong subset
    kw_p_true_diffs = [
        p_true_diffs[list(test_data).index(e)]
        for e in test_data
        if e["sample_id"] in classification["know_wrong"]
    ]
    if kw_p_true_diffs:
        mean_kw_p_diff = np.mean(kw_p_true_diffs)
        print(f"  C4: know-wrong mean ΔP(y_true) = {mean_kw_p_diff:+.4f}")
    else:
        mean_kw_p_diff = 0.0

    # ── Run contrastive interventions ──
    print(f"\n[4/5] Running contrastive interventions...")

    all_results = {}

    for alpha in args.alphas:
        correct = 0
        subset_correct = {"know_wrong": 0, "know_correct": 0, "dont_know": 0}

        for entry in tqdm(test_data, desc=f"    α={alpha:+5.1f}", leave=False):
            gen = generate_contrastive(
                model,
                tokenizer,
                entry["prompt_std"],
                entry["prompt_truth"],
                device,
                alpha,
            )
            is_correct = check_correct(gen, entry["answers"], dataset="triviaqa")
            if is_correct:
                correct += 1
                sid = entry["sample_id"]
                for subset in ["know_correct", "know_wrong", "dont_know"]:
                    if sid in classification[subset]:
                        subset_correct[subset] += 1

        rate = correct / n_total
        delta = rate - baseline_total / n_total
        all_results[f"contrastive_α={alpha:+.1f}"] = {
            "correct": correct,
            "total": n_total,
            "rate": rate,
            "delta": float(delta),
            "subset_correct": subset_correct,
        }

        kw_rate = subset_correct["know_wrong"] / max(n_kw, 1) if n_kw > 0 else 0.0
        print(
            f"    α={alpha:+5.1f}: {correct}/{n_total} = {rate:.1%} "
            f"(Δ={delta:+.1%})  know_wrong: {subset_correct['know_wrong']}/{n_kw} "
            f"({kw_rate:.1%})"
        )

    # ── Also run a "truthful-only" baseline (α → ∞, i.e. use l_truth directly) ──
    print(f"\n  ── Truthful-only baseline (l_truth, no combination) ──")
    correct = 0
    subset_correct = {"know_wrong": 0, "know_correct": 0, "dont_know": 0}
    for entry in tqdm(test_data, desc="    truthful-only", leave=False):
        l_truth = get_logits(model, entry["prompt_truth"], device)
        nid = int(l_truth.argmax().item())
        gids = [nid]
        tokens = model.to_tokens(entry["prompt_std"], prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        for _ in range(19):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)
        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, entry["answers"], dataset="triviaqa")
        if is_correct:
            correct += 1
            sid = entry["sample_id"]
            for subset in ["know_correct", "know_wrong", "dont_know"]:
                if sid in classification[subset]:
                    subset_correct[subset] += 1

    truthful_rate = correct / n_total
    truthful_delta = truthful_rate - baseline_total / n_total
    all_results["truthful_only"] = {
        "correct": correct,
        "total": n_total,
        "rate": truthful_rate,
        "delta": float(truthful_delta),
        "subset_correct": subset_correct,
    }
    kw_rate = subset_correct["know_wrong"] / max(n_kw, 1) if n_kw > 0 else 0.0
    print(
        f"    truthful-only: {correct}/{n_total} = {truthful_rate:.1%} "
        f"(Δ={truthful_delta:+.1%})  know_wrong: {subset_correct['know_wrong']}/{n_kw} "
        f"({kw_rate:.1%})"
    )

    # ── Summary ──
    print(f"\n[5/5] Summary")
    print(f"\n  {'Method':>25s} {'Best Δ':>10s} {'Best α':>10s} {'KW Δ':>10s}")

    contrastive_results = {
        k: v for k, v in all_results.items() if k.startswith("contrastive_")
    }
    if contrastive_results:
        best = max(contrastive_results.items(), key=lambda x: x[1]["delta"])
        best_kw = (
            best[1]["subset_correct"]["know_wrong"] / max(n_kw, 1) if n_kw > 0 else 0.0
        )
        print(
            f"  {'contrastive':>25s} {best[1]['delta']:+9.1%} "
            f"{best[0].split('=')[1]:>10s} "
            f"{best_kw:+9.1%}"
        )

    truthful_kw = (
        all_results["truthful_only"]["subset_correct"]["know_wrong"] / max(n_kw, 1)
        if n_kw > 0
        else 0.0
    )
    print(
        f"  {'truthful-only':>25s} {all_results['truthful_only']['delta']:+9.1%} "
        f"{'N/A':>10s} "
        f"{truthful_kw:+9.1%}"
    )

    # Gate checks
    c2_passes = best[1]["delta"] > 0.05 if contrastive_results else False
    c3_passes = (
        n_kw > 0
        and contrastive_results
        and (best[1]["subset_correct"]["know_wrong"] / n_kw) > baseline_total / n_total
    )

    print(f"\n  C1 (||l_truth - l_std|| > 1): {'PASS' if c1_passes else 'FAIL'}")
    print(f"  C2 (contrastive Δ > 5%): {'PASS' if c2_passes else 'FAIL'}")
    print(f"  C3 (know_wrong > all): {'PASS' if c3_passes else 'FAIL'}")

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "n_calibrate": args.n_calibrate,
            "alphas": args.alphas,
            "rank_threshold": args.rank_threshold,
            "seed": args.seed,
        },
        "classification": {
            "know_correct": n_kc,
            "know_wrong": n_kw,
            "dont_know": n_dk,
        },
        "baseline": {
            "correct": baseline_total,
            "total": n_total,
            "rate": baseline_total / n_total,
        },
        "c1_stats": {
            "mean_l_diff_norm": float(mean_diff_norm),
            "l_diff_norms": l_diff_norms,
            "mean_p_true_diff": float(mean_p_true_diff),
            "kw_mean_p_true_diff": float(mean_kw_p_diff) if kw_p_true_diffs else 0.0,
        },
        "results": all_results,
        "gates": {
            "c1": bool(c1_passes),
            "c2": bool(c2_passes),
            "c3": bool(c3_passes),
        },
    }

    out_path = output_dir / "s13_1_contrastive.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
