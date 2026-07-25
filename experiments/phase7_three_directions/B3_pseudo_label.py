"""B.3: Pseudo-Label Bootstrapping via Self-Consistency.

Core idea: without human labels, use the DIVERSITY of K sampled answers as a signal.
  - High self-consistency (similar answers) → model is confident → likely correct
  - Low self-consistency (divergent answers) → model is uncertain → likely hallucinating

Algorithm:
  1. Generate K=5 answers per question with temperature > 0
  2. Compute mean pairwise ROUGE-L → "self-consistency" score
  3. Evaluate AUROC of self-consistency directly (no training needed)
  4. Binarize by median split → pseudo-labels → truth direction
  5. Compare pseudo-direction with supervised upper bound

Key metrics:
  - Self-consistency AUROC (direct, no training)
  - Pseudo-truth-direction AUROC (with median-split labels)
  - Cosine similarity with supervised truth direction

Usage:
    python B3_pseudo_label.py --n_samples 100 --k_generations 5
    python B3_pseudo_label.py --n_samples 100 --k_generations 5 --load_hs <path>
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
# ROUGE-L (simple, no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _lcs_len(x, y):
    """Longest common subsequence length (1D DP)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l_f1(ref: str, hyp: str) -> float:
    """ROUGE-L F1 between two strings."""
    ref_tokens = ref.lower().split()
    hyp_tokens = hyp.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def self_consistency(answers: list[str]) -> float:
    """Mean pairwise ROUGE-L F1 across all answer pairs.

    High = answers agree → model is confident → likely correct.
    Low = answers diverge → model is uncertain → likely hallucinating.
    """
    K = len(answers)
    if K < 2:
        return 1.0
    sims = []
    for i in range(K):
        for j in range(i + 1, K):
            sims.append(rouge_l_f1(answers[i], answers[j]))
    return float(np.mean(sims))


# ═══════════════════════════════════════════════════════════════════════════════
# Generation with temperature
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_temperature(model, tokenizer, prompt, device,
                               temperature=0.7, max_new_tokens=30):
    """Generate one answer with temperature sampling. Returns decoded string."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits = model(tokens)
        logits_t = logits[0, -1, :] / temperature
        probs = torch.softmax(logits_t, dim=-1)
        nid = int(torch.multinomial(probs, num_samples=1).item())
        gids = [nid]

        for _ in range(max_new_tokens - 1):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            if tokens.shape[1] > 1024:
                break
            with torch.no_grad():
                logits = model(tokens)
            logits_t = logits[0, -1, :] / temperature
            probs = torch.softmax(logits_t, dim=-1)
            nid = int(torch.multinomial(probs, num_samples=1).item())
            gids.append(nid)

    return tokenizer.decode(gids).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Truth direction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_truth_direction(h_correct, h_incorrect):
    """v = mean(correct) - mean(incorrect), normalized."""
    v = h_correct.mean(axis=0) - h_incorrect.mean(axis=0)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-10:
        v = v / v_norm
    return v


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="B.3: Self-Consistency Pseudo-Labels")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--k_generations", type=int, default=5,
                        help="Number of diverse generations per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/phase7_three_directions/outputs_phase7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_hs", type=str, default=None,
                        help="Load cached HS from JSON (skip HS extraction)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"B.3: Self-Consistency Pseudo-Labels (K={args.k_generations}, T={args.temperature})")
    print(f"  Samples: {args.n_samples}  Layer: {args.layer}")
    print(f"{'='*60}\n")

    # ── Load model & data ──
    print("Loading model & data...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # ── Phase 1: Extract or load hidden states + true labels ──
    if args.load_hs:
        print(f"Loading cached HS from: {args.load_hs}")
        with open(args.load_hs) as f:
            cache = json.load(f)
        cached_records = cache["records"]
        h_all = np.stack([np.array(r["h_last"]) for r in cached_records[:args.n_samples]], axis=0)
        true_labels = np.array([r["label"] for r in cached_records[:args.n_samples]])
        print(f"  Loaded {len(h_all)} hidden states, correct={true_labels.sum()}/{len(true_labels)}")
    else:
        print(f"Extracting hidden states at L{args.layer}...")
        t0 = time.time()
        h_list, tl_list = [], []
        correct_count = 0

        for s in tqdm(samples, desc="Extract HS"):
            prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]

            residual = {}
            def _hook(act, hook=None):
                residual["h"] = act[:, -1, :].detach()
                return act

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{args.layer}.hook_resid_post", _hook)],
                )
            h_list.append(residual["h"].float().cpu().numpy().flatten())

            # Greedy decode for true label
            nid = int(logits[0, -1, :].argmax().item())
            gids = [nid]
            for _ in range(29):
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
            tl_list.append(1 if is_correct else 0)

        h_all = np.stack(h_list, axis=0)
        true_labels = np.array(tl_list)
        print(f"  Extracted in {time.time()-t0:.0f}s, correct={correct_count}/{len(samples)}")

    N = len(samples)

    # ── Phase 2: Generate K diverse answers per question ──
    print(f"\nGenerating K={args.k_generations} answers per question (T={args.temperature})...")
    t0 = time.time()

    all_generations = []   # [N, K]
    consistency_scores = []  # [N]

    for s in tqdm(samples, desc="Generate K answers"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        answers = []
        for _ in range(args.k_generations):
            ans = generate_with_temperature(
                model, tokenizer, prompt, device,
                temperature=args.temperature,
            )
            answers.append(ans)
        all_generations.append(answers)
        consistency_scores.append(self_consistency(answers))

    consistency_scores = np.array(consistency_scores)
    elapsed_gen = time.time() - t0
    print(f"  Generated in {elapsed_gen:.0f}s ({elapsed_gen/N:.1f}s/sample)")

    # ── Phase 3: Evaluate self-consistency as direct detection score ──
    print(f"\n{'─'*50}")
    print("Self-Consistency as Direct Score")
    print(f"{'─'*50}")

    valid_c = np.isfinite(consistency_scores)
    auroc_consistency = float(roc_auc_score(true_labels[valid_c], consistency_scores[valid_c]))
    auroc_consistency = max(auroc_consistency, 1 - auroc_consistency)

    cons_correct = consistency_scores[true_labels == 1].mean()
    cons_incorrect = consistency_scores[true_labels == 0].mean()

    print(f"  Mean consistency (correct):   {cons_correct:.4f}")
    print(f"  Mean consistency (incorrect): {cons_incorrect:.4f}")
    print(f"  Self-consistency AUROC:       {auroc_consistency:.4f}")

    # ── Phase 4: Median-split binarization → truth direction ──
    print(f"\n{'─'*50}")
    print("Median-Split Pseudo-Labels → Truth Direction")
    print(f"{'─'*50}")

    median_cons = np.median(consistency_scores)
    pseudo_labels = (consistency_scores > median_cons).astype(int)  # above median → pseudo-correct
    n_pseudo_correct = pseudo_labels.sum()
    n_pseudo_incorrect = N - n_pseudo_correct

    print(f"  Median consistency: {median_cons:.4f}")
    print(f"  Pseudo-correct: {n_pseudo_correct}  Pseudo-incorrect: {n_pseudo_incorrect}")

    # Pseudo-label accuracy vs true
    pseudo_acc = (pseudo_labels == true_labels).mean()
    print(f"  Pseudo-label accuracy: {pseudo_acc:.4f}")

    # Compute truth direction from pseudo-labels
    mask_pc = pseudo_labels == 1
    mask_pi = pseudo_labels == 0

    if mask_pc.sum() >= 2 and mask_pi.sum() >= 2:
        v_pseudo = compute_truth_direction(h_all[mask_pc], h_all[mask_pi])
        scores_pseudo = h_all @ v_pseudo
        valid = np.isfinite(scores_pseudo)
        auroc_pseudo = float(roc_auc_score(true_labels[valid], scores_pseudo[valid]))
        auroc_pseudo = max(auroc_pseudo, 1 - auroc_pseudo)
    else:
        v_pseudo = None
        auroc_pseudo = float("nan")

    # Supervised upper bound (on same data)
    mask_tc = true_labels == 1
    mask_ti = true_labels == 0
    v_true = compute_truth_direction(h_all[mask_tc], h_all[mask_ti])
    scores_true = h_all @ v_true
    auroc_supervised = float(roc_auc_score(true_labels, scores_true))
    auroc_supervised = max(auroc_supervised, 1 - auroc_supervised)

    cos_sim = float(np.dot(v_pseudo, v_true)) if v_pseudo is not None else float("nan")

    print(f"\n  Pseudo-label TD AUROC:    {auroc_pseudo:.4f}")
    print(f"  Supervised TD AUROC:      {auroc_supervised:.4f}")
    print(f"  AUROC gap:                {auroc_supervised - auroc_pseudo:.4f}")
    print(f"  cos(v_pseudo, v_true):    {cos_sim:.4f}")

    # ── Also try: consistency-weighted truth direction ──
    # Weight samples by their consistency when computing v
    print(f"\n  Consistency-weighted truth direction:")
    # Weighted mean: higher consistency → more weight in "correct" direction
    weights = consistency_scores - consistency_scores.min()
    weights = weights / (weights.max() + 1e-10)  # normalize to [0, 1]
    v_weighted = (h_all * weights[:, None]).mean(axis=0) - (h_all * (1 - weights)[:, None]).mean(axis=0)
    v_weighted = v_weighted / (np.linalg.norm(v_weighted) + 1e-10)
    scores_weighted = h_all @ v_weighted
    valid = np.isfinite(scores_weighted)
    auroc_weighted = float(roc_auc_score(true_labels[valid], scores_weighted[valid]))
    auroc_weighted = max(auroc_weighted, 1 - auroc_weighted)
    cos_weighted = float(np.dot(v_weighted, v_true))
    print(f"  Weighted TD AUROC:        {auroc_weighted:.4f}")
    print(f"  cos(v_weighted, v_true):  {cos_weighted:.4f}")

    # ── Show examples ──
    print(f"\n  Example generations:")
    for i in range(min(4, N)):
        q = samples[i]["question"]
        print(f"\n  Q{i}: {q[:80]}...")
        print(f"    True: {'correct' if true_labels[i] else 'incorrect'} | "
              f"Consistency: {consistency_scores[i]:.3f} | "
              f"Pseudo: {'correct' if pseudo_labels[i] else 'incorrect'}")
        for k, ans in enumerate(all_generations[i]):
            print(f"    #{k}: {ans[:80]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Self-consistency AUROC:         {auroc_consistency:.4f}")
    print(f"  Pseudo-label TD AUROC:          {auroc_pseudo:.4f}")
    print(f"  Consistency-weighted TD AUROC:  {auroc_weighted:.4f}")
    print(f"  Supervised TD AUROC (upper):    {auroc_supervised:.4f}")

    best_unsupervised = max(auroc_consistency, auroc_pseudo, auroc_weighted)
    if best_unsupervised > 0.75:
        print(f"\n  ✅ SUCCESS: Best unsupervised AUROC = {best_unsupervised:.4f} > 0.75")
    elif best_unsupervised > 0.70:
        print(f"\n  ⚠️  MARGINAL: Best unsupervised AUROC = {best_unsupervised:.4f} > 0.70")
    else:
        print(f"\n  ❌ Best unsupervised AUROC = {best_unsupervised:.4f} < 0.70")

    # ── Save ──
    save_path = output_dir / "B3_pseudo_label.json"
    with open(save_path, "w") as f:
        json.dump({
            "n_samples": N,
            "k_generations": args.k_generations,
            "temperature": args.temperature,
            "layer": args.layer,
            "auroc_self_consistency": auroc_consistency,
            "auroc_pseudo_label_td": auroc_pseudo,
            "auroc_weighted_td": auroc_weighted,
            "auroc_supervised": auroc_supervised,
            "cos_sim_pseudo_vs_true": cos_sim,
            "cos_sim_weighted_vs_true": cos_weighted,
            "pseudo_label_accuracy": float(pseudo_acc),
            "median_consistency": float(median_cons),
            "mean_consistency_correct": float(cons_correct),
            "mean_consistency_incorrect": float(cons_incorrect),
            "per_sample": [
                {
                    "idx": i,
                    "question": samples[i]["question"][:120],
                    "true_label": int(true_labels[i]),
                    "pseudo_label": int(pseudo_labels[i]),
                    "consistency": float(consistency_scores[i]),
                    "generations": all_generations[i],
                }
                for i in range(N)
            ],
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"B.3 complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
