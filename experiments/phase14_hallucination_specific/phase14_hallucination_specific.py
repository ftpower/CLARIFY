"""Phase 14: Hallucination-Specific Feature Suppression.

Key insight: Multiple samples per question → paired comparison
  Δ_q = mean(h_hallucinated) - mean(h_correct)  [within same question]

This isolates hallucination-specific signal by removing question variance.

Usage:
    python phase14_hallucination_specific.py --n_questions 30 --k_samples 5 --n_test 30 --quick
    python phase14_hallucination_specific.py --n_questions 50 --k_samples 10 --n_test 50
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
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
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════
# Multi-sample extraction with paired comparison
# ═══════════════════════════════════════════════


def extract_multi_sample(
    model,
    tokenizer,
    samples,
    device,
    n_layers,
    intervention_layer,
    k=5,
    temperature=0.8,
    max_new=20,
):
    """Sample K times per question. Extract hidden states + labels.

    Returns per-question records with lists of hidden states for correct
    and hallucinated answers.
    """
    results = []
    t0 = time.time()

    for si, sample in enumerate(tqdm(samples, desc="Multi-sample")):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        gt_answers = sample.get("answers", [sample.get("gt_answer", "")])
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        correct_hs = []
        hallucinated_hs = []

        for _ in range(k):
            gids = []
            tk = tokens.clone()

            with torch.no_grad():
                # First forward with temperature sampling
                logits = model(tk)
                # Apply temperature to logits
                logits_scaled = logits / temperature
                probs = torch.softmax(logits_scaled[0, -1, :], dim=-1)
                nid = torch.multinomial(probs, 1).item()
            gids.append(nid)

            new_tokens = torch.cat([tk, torch.tensor([[nid]], device=device)], dim=1)

            for _ in range(max_new - 1):
                if nid == tokenizer.eos_token_id:
                    break
                if new_tokens.shape[1] > 1024:
                    break
                with torch.no_grad():
                    logits = model(new_tokens)
                logits_scaled = logits / temperature
                probs = torch.softmax(logits_scaled[0, -1, :], dim=-1)
                nid = torch.multinomial(probs, 1).item()
                gids.append(nid)
                new_tokens = torch.cat(
                    [new_tokens, torch.tensor([[nid]], device=device)], dim=1
                )

            ans = tokenizer.decode(gids).strip()
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")

            # Extract hidden state at intervention layer (last input token)
            stored = {}

            def _capture(act, hook=None):
                stored["h"] = act[0, input_len - 1, :].detach().cpu().clone()
                return act

            hook_name = f"blocks.{intervention_layer}.hook_resid_post"

            with torch.no_grad():
                model.run_with_hooks(tk, fwd_hooks=[(hook_name, _capture)])

            if is_correct:
                correct_hs.append(stored["h"].numpy())
            else:
                hallucinated_hs.append(stored["h"].numpy())

        results.append(
            {
                "question": sample["question"][:100],
                "n_correct": len(correct_hs),
                "n_hallucinated": len(hallucinated_hs),
                "correct_hs": correct_hs,
                "hallucinated_hs": hallucinated_hs,
                "gt_answers": gt_answers,
            }
        )

    elapsed = time.time() - t0
    n_paired = sum(1 for r in results if r["n_correct"] > 0 and r["n_hallucinated"] > 0)
    n_any_correct = sum(1 for r in results if r["n_correct"] > 0)
    n_any_halluc = sum(1 for r in results if r["n_hallucinated"] > 0)

    print(f"  Extracted {len(results)} questions in {elapsed:.0f}s")
    print(f"  Questions with both types: {n_paired}/{len(results)}")
    print(f"  Questions with correct: {n_any_correct}, hallucinated: {n_any_halluc}")

    return results


# ═══════════════════════════════════════════════
# Compute hallucination-specific direction
# ═══════════════════════════════════════════════


def compute_hallucination_direction(multi_results):
    """Compute v_halluc from within-question paired differences.

    For each question q with both correct and hallucinated samples:
      Δ_q = mean(h_hallucinated_q) - mean(h_correct_q)

    v_halluc = mean(Δ_q across all paired questions), normalized.

    Also compute:
      v_global (old method): mean(all correct) - mean(all incorrect)
      cos(v_halluc, v_global): how different are they?
    """
    deltas = []
    all_correct = []
    all_halluc = []

    for r in multi_results:
        if r["n_correct"] > 0 and r["n_hallucinated"] > 0:
            h_c_mean = np.mean(r["correct_hs"], axis=0)
            h_h_mean = np.mean(r["hallucinated_hs"], axis=0)
            delta = h_h_mean - h_c_mean
            deltas.append(delta)

        all_correct.extend(r["correct_hs"])
        all_halluc.extend(r["hallucinated_hs"])

    if len(deltas) < 3:
        print("  ⚠️  Too few paired questions for reliable direction")
        return None

    v_halluc = np.mean(deltas, axis=0)
    v_norm = np.linalg.norm(v_halluc)
    print(f"  Mean delta norm before normalize: {v_norm:.6f}")
    if v_norm < 1e-6:
        print(
            "  ⚠️  v_halluc is degenerate (norm ≈ 0) — no consistent hallucination direction exists!"
        )
        return None
    v_halluc /= v_norm

    # Old method for comparison
    if all_correct and all_halluc:
        v_global = np.mean(all_correct, axis=0) - np.mean(all_halluc, axis=0)
        v_global /= np.linalg.norm(v_global) + 1e-10
        cos_sim = np.dot(v_halluc, v_global)
    else:
        v_global = None
        cos_sim = float("nan")

    print(f"  Paired questions used: {len(deltas)}")
    print(f"  v_halluc norm: {np.linalg.norm(v_halluc):.4f}")
    if v_global is not None:
        print(
            f"  cos(v_halluc, v_global): {cos_sim:.4f} "
            f"({np.degrees(np.arccos(np.clip(abs(cos_sim), -1, 1))):.1f}°)"
        )

    return {
        "v_halluc": v_halluc,
        "v_global": v_global,
        "cos_sim": float(cos_sim),
        "n_paired": len(deltas),
    }


# ═══════════════════════════════════════════════
# Intervention methods
# ═══════════════════════════════════════════════


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20):
    gids = []
    for _ in range(max_new):
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


def baseline_gen(model, tokenizer, prompt, device, max_new=20):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    gids = []
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break
    return tokenizer.decode(gids).strip()


def suppress_gen(
    model, tokenizer, prompt, device, layer, v_suppress, alpha, max_new=20
):
    """Intervention: SUBTRACT hallucination direction."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]
    mod = torch.tensor(-alpha * v_suppress, dtype=torch.float32, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def hook(act, hook=None):
        act[0, input_len - 1, :] += mod
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, hook)], max_new)


def project_out_gen(
    model, tokenizer, prompt, device, layer, v_proj_out, alpha, max_new=20
):
    """Intervention: PROJECT OUT hallucination component."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]
    v = torch.tensor(v_proj_out, dtype=torch.float32, device=device)

    def hook(act, hook=None):
        h = act[0, input_len - 1, :]
        proj = torch.dot(h, v) * v
        act[0, input_len - 1, :] = h - alpha * proj
        return act

    return _gen_greedy(
        model,
        tokenizer,
        tokens,
        device,
        [(f"blocks.{layer}.hook_resid_post", hook)],
        max_new,
    )


def evaluate(gen_model, tokenizer, test_samples, device, gen_fn, **kwargs):
    correct = 0
    for s in tqdm(test_samples, desc="eval", leave=False):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        ans = gen_fn(gen_model, tokenizer, prompt, device, **kwargs)
        if check_correct(
            ans, s.get("answers", [s.get("gt_answer", "")]), dataset="triviaqa"
        ):
            correct += 1
    return correct / len(s)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 14: Hallucination-Specific")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_questions",
        type=int,
        default=40,
        help="Number of questions for multi-sampling",
    )
    parser.add_argument("--k_samples", type=int, default=5, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_questions = min(args.n_questions, 20)
        args.k_samples = min(args.k_samples, 5)
        args.n_test = min(args.n_test, 30)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 65}")
    print(f"Phase 14: Hallucination-Specific Feature Suppression")
    print(f"  Questions: {args.n_questions} × {args.k_samples} samples")
    print(f"  Test: {args.n_test}  Layer: {args.layer}")
    print(f"{'=' * 65}\n")

    # ── Load model + data ──
    print("Loading model...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # ── Phase 1: Multi-sample extraction ──
    print(f"\n{'─' * 50}")
    print("Phase 1: Multi-sample extraction")
    print(f"{'─' * 50}")

    all_samples = load_triviaqa(
        n_samples=args.n_questions + args.n_test, seed=args.seed
    )
    multi_samples = all_samples[: args.n_questions]
    test_samples = all_samples[args.n_questions : args.n_questions + args.n_test]

    multi_results = extract_multi_sample(
        model,
        tokenizer,
        multi_samples,
        device,
        n_layers,
        args.layer,
        k=args.k_samples,
        temperature=args.temperature,
    )

    # ── Phase 2: Compute hallucination direction ──
    print(f"\n{'─' * 50}")
    print("Phase 2: Computing hallucination-specific direction")
    print(f"{'─' * 50}")

    dir_info = compute_hallucination_direction(multi_results)
    if dir_info is None:
        print("FATAL: Cannot compute direction. Need more paired questions.")
        return

    # ── Phase 3: Evaluate interventions ──
    print(f"\n{'─' * 50}")
    print("Phase 3: Intervention Evaluation")
    print(f"{'─' * 50}")

    # Also compute old v for comparison
    train_labels = []
    train_hs = []
    for r in multi_results:
        for h in r["correct_hs"]:
            train_labels.append(1)
            train_hs.append(h)
        for h in r["hallucinated_hs"]:
            train_labels.append(0)
            train_hs.append(h)
    train_hs = np.array(train_hs)
    train_labels = np.array(train_labels)
    mc, mi = train_labels == 1, train_labels == 0
    v_old = train_hs[mc].mean(0) - train_hs[mi].mean(0)
    v_old /= np.linalg.norm(v_old) + 1e-10

    # Baseline
    print("\n[Baseline]")
    bl = evaluate(model, tokenizer, test_samples, device, baseline_gen)
    print(f"  Baseline: {bl:.2%}")

    # M0: Old method (subtract v_global)
    print("\n[M0] Subtract v_global (old method)...")
    m0_best = bl
    for alpha in [0.5, 1.0]:
        rate = evaluate(
            model,
            tokenizer,
            test_samples,
            device,
            suppress_gen,
            layer=args.layer,
            v_suppress=v_old,
            alpha=alpha,
        )
        m0_best = max(m0_best, rate)
        d = rate - bl
        print(f"    α={alpha}: {rate:.2%} (Δ={d:+.1%})")
    print(f"  M0 best: {m0_best:.2%}")

    # M1: Subtract v_halluc
    print("\n[M1] Subtract v_halluc (hallucination-specific)...")
    m1_best = bl
    for alpha in [0.5, 1.0, 2.0]:
        rate = evaluate(
            model,
            tokenizer,
            test_samples,
            device,
            suppress_gen,
            layer=args.layer,
            v_suppress=dir_info["v_halluc"],
            alpha=alpha,
        )
        m1_best = max(m1_best, rate)
        d = rate - bl
        marker = " ↑" if d > 0.05 else ""
        print(f"    α={alpha}: {rate:.2%} (Δ={d:+.1%}){marker}")
    print(f"  M1 best: {m1_best:.2%}")

    # M2: Project out v_halluc
    print("\n[M2] Project out v_halluc...")
    m2_best = bl
    for alpha in [0.3, 0.5, 1.0]:
        rate = evaluate(
            model,
            tokenizer,
            test_samples,
            device,
            project_out_gen,
            layer=args.layer,
            v_proj_out=dir_info["v_halluc"],
            alpha=alpha,
        )
        m2_best = max(m2_best, rate)
        d = rate - bl
        marker = " ↑" if d > 0.05 else ""
        print(f"    α={alpha}: {rate:.2%} (Δ={d:+.1%}){marker}")
    print(f"  M2 best: {m2_best:.2%}")

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"  Paired questions: {dir_info['n_paired']}/{args.n_questions}")
    print(f"  cos(v_halluc, v_global): {dir_info['cos_sim']:.4f}")
    print(f"  Baseline:   {bl:.2%}")
    print(f"  M0 (old):   {m0_best:.2%}  Δ={m0_best - bl:+.1%}")
    print(f"  M1 (sub):   {m1_best:.2%}  Δ={m1_best - bl:+.1%}")
    print(f"  M2 (proj):  {m2_best:.2%}  Δ={m2_best - bl:+.1%}")

    # Save
    save_path = output_dir / "phase14_results.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "n_questions": args.n_questions,
                "k_samples": args.k_samples,
                "n_test": args.n_test,
                "n_paired": dir_info["n_paired"],
                "cos_v_halluc_v_global": dir_info["cos_sim"],
                "baseline": bl,
                "m0_best": m0_best,
                "m1_best": m1_best,
                "m2_best": m2_best,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {save_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
