"""Phase 3 S1: Detect-then-Intervene Full Pipeline.

Combines the best detector (D2: JS divergence L17 vs L26 + max_p, joint AUROC 0.936)
with the best intervention (I1: L11 mean-diff direction projection, λ=0.5, +5.6pp).

The pipeline:
1. Knowledge filter: only operate on samples with P(correct)>0.3
2. Detection: compute risk score = logistic regression on (JS_divergence, max_p)
3. Intervention: apply L11 direction projection ONLY to high-risk samples
4. Compare: baseline vs blind-intervention vs detect-then-intervene

Strategy: vary the risk threshold to trace the precision-recall tradeoff of
"intervene only when needed."

Usage:
    python main_s1_pipeline.py --n_eval 500 --skip_extract
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    sign = -1.0 if mode == "subtract" else 1.0

    def hook(activation, hook=None):
        d = direction.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam * projection

    return hook


def compute_js_divergence(p_early, p_late, eps=1e-10):
    """JS divergence between two probability distributions."""
    m = 0.5 * (p_early + p_late)
    p_early_s = np.clip(p_early, eps, 1.0)
    p_late_s = np.clip(p_late, eps, 1.0)
    m_s = np.clip(m, eps, 1.0)
    kl_early = np.sum(p_early_s * np.log(p_early_s / m_s), axis=-1)
    kl_late = np.sum(p_late_s * np.log(p_late_s / m_s), axis=-1)
    return 0.5 * (kl_early + kl_late)


def run_s1(
    model,
    eval_samples: list[dict],
    letter_ids: dict[str, int],
    direction: torch.Tensor,
    output_dir: Path,
    intervention_layer: int = 11,
    lam: float = 0.5,
    js_early_layer: int = 17,
    js_late_layer: int = 26,
):
    """Run S1 Detect-then-Intervene pipeline."""
    print("\n" + "=" * 60)
    print("S1: Detect-then-Intervene Full Pipeline")
    print("=" * 60)

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers

    # ── Step 1: Extract baseline data (knowledge score + JS features) ──
    print("\nStep 1: Extracting baseline features...")
    storage = {}
    hooks = []
    for i in range(n_layers):
        key = f"blocks.{i}.hook_resid_post"
        hooks.append((key, _make_collect_hook(storage, key)))

    baseline_results = []
    # batch tokenize
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append({"tokens": tokens, "correct_letter": correct_letter})

    for idx, item in enumerate(tqdm(tokenized, desc="Forward pass")):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # JS divergence L17 vs L26
        h_early = storage[f"blocks.{js_early_layer}.hook_resid_post"][0, last_pos, :]
        h_late = storage[f"blocks.{js_late_layer}.hook_resid_post"][0, last_pos, :]

        W_U = model.unembed.W_U
        logits_early = h_early.to(W_U.device) @ W_U
        logits_late = h_late.to(W_U.device) @ W_U
        probs_early = (
            F.softmax(logits_early[letter_tok_ids].float(), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        probs_late = (
            F.softmax(logits_late[letter_tok_ids].float(), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        js = compute_js_divergence(probs_early[None, :], probs_late[None, :])[0]

        # max_p from final logit lens (L27)
        h_final = storage[f"blocks.{n_layers - 1}.hook_resid_post"][0, last_pos, :]
        logits_final_all = h_final.to(W_U.device) @ W_U
        max_p = F.softmax(logits_final_all.float(), dim=-1).max().item()

        # Baseline prediction
        logits_last = logits[0, last_pos, :]
        choice_probs = F.softmax(logits_last[letter_tok_ids].float(), dim=-1)
        p_correct = choice_probs[letters.index(correct_letter)].item()
        pred_idx = choice_probs.argmax().item()
        is_correct = letters[pred_idx] == correct_letter

        baseline_results.append(
            {
                "js": float(js),
                "max_p": float(max_p),
                "p_correct": float(p_correct),
                "is_correct": is_correct,
                "correct_letter": correct_letter,
                "tokens": tokens,
            }
        )

    n_total = len(baseline_results)
    baseline_acc = sum(r["is_correct"] for r in baseline_results) / n_total
    print(f"\nBaseline accuracy: {baseline_acc:.4f} ({n_total} samples)")

    # Knowledge filter
    knowledge_mask = np.array([r["p_correct"] > 0.3 for r in baseline_results])
    n_knowledge = knowledge_mask.sum()
    knowledge_acc = (
        sum(r["is_correct"] for r, m in zip(baseline_results, knowledge_mask) if m)
        / n_knowledge
    )
    print(f"Knowledge-filtered: {n_knowledge} samples, acc={knowledge_acc:.4f}")

    # ── Step 2: Train detector LR on knowledge-filtered samples ──
    print("\nStep 2: Training risk detector (LR on JS + max_p)...")
    filt_js = np.array([r["js"] for r, m in zip(baseline_results, knowledge_mask) if m])
    filt_mp = np.array(
        [r["max_p"] for r, m in zip(baseline_results, knowledge_mask) if m]
    )
    filt_labels = np.array(
        [r["is_correct"] for r, m in zip(baseline_results, knowledge_mask) if m],
        dtype=int,
    )

    # Cross-validated risk model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    risk_scores = np.zeros(n_knowledge)

    for tr_idx, te_idx in cv.split(filt_js, filt_labels):
        X_tr = np.stack([filt_js[tr_idx], filt_mp[tr_idx]], axis=1)
        X_te = np.stack([filt_js[te_idx], filt_mp[te_idx]], axis=1)
        y_tr = filt_labels[tr_idx]

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_tr, y_tr)
        risk_scores[te_idx] = lr.predict_proba(X_te)[:, 1]  # P(incorrect)

    # AUROC of detector
    from sklearn.metrics import roc_auc_score

    detector_auroc = roc_auc_score(filt_labels, risk_scores)
    print(f"Detector AUROC (CV risk score): {detector_auroc:.4f}")

    # ── Step 3: Intervention at various risk thresholds ──
    print("\nStep 3: Detect-then-Intervene at varying thresholds...")
    risk_thresholds = np.linspace(0.1, 0.9, 9)

    # Blind intervention baseline (intervene on ALL knowledge-filtered samples)
    print("Evaluating blind intervention (all knowledge-filtered)...")
    blind_correct = 0
    blind_total = 0
    filt_indices = np.where(knowledge_mask)[0]
    for fi, idx in enumerate(tqdm(filt_indices, desc="Blind intervene")):
        r = baseline_results[idx]
        tokens = r["tokens"]

        intervene_hooks = [
            (
                f"blocks.{intervention_layer}.hook_resid_post",
                make_projection_hook(direction, lam, "subtract"),
            ),
        ]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=intervene_hooks)

        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last[letter_tok_ids].float(), dim=-1)
        pred_idx = probs.argmax().item()
        blind_correct += int(letters[pred_idx] == r["correct_letter"])
        blind_total += 1

    blind_acc = blind_correct / blind_total
    blind_delta = blind_acc - knowledge_acc
    print(
        f"Blind intervention: {blind_acc:.4f} (Δ={blind_delta:+.4f} vs knowledge baseline)"
    )

    # Per-threshold detect-then-intervene
    print("\nEvaluating per-threshold detect-then-intervene...")
    results_by_threshold = []

    for thresh in risk_thresholds:
        intervene_correct = 0
        intervene_total = 0
        skip_correct = 0
        skip_total = 0

        for fi, idx in enumerate(filt_indices):
            r = baseline_results[idx]
            tokens = r["tokens"]
            risk = risk_scores[fi]

            if risk >= thresh:
                # High risk → intervene
                intervene_hooks = [
                    (
                        f"blocks.{intervention_layer}.hook_resid_post",
                        make_projection_hook(direction, lam, "subtract"),
                    ),
                ]
                with torch.no_grad():
                    logits = model.run_with_hooks(tokens, fwd_hooks=intervene_hooks)
                logits_last = logits[0, -1, :]
                probs = F.softmax(logits_last[letter_tok_ids].float(), dim=-1)
                pred_idx = probs.argmax().item()
                is_c = letters[pred_idx] == r["correct_letter"]
                intervene_correct += int(is_c)
                intervene_total += 1
            else:
                # Low risk → skip (use baseline result)
                skip_correct += int(r["is_correct"])
                skip_total += 1

        total_correct = intervene_correct + skip_correct
        total_combined = intervene_total + skip_total
        combined_acc = total_correct / total_combined if total_combined > 0 else 0.0
        delta = combined_acc - knowledge_acc

        intervene_acc = (
            intervene_correct / intervene_total if intervene_total > 0 else float("nan")
        )

        results_by_threshold.append(
            {
                "risk_threshold": float(thresh),
                "n_intervened": intervene_total,
                "n_skipped": skip_total,
                "intervene_acc": float(intervene_acc)
                if not np.isnan(intervene_acc)
                else None,
                "skip_acc": float(skip_correct / skip_total)
                if skip_total > 0
                else None,
                "combined_acc": float(combined_acc),
                "delta": float(delta),
            }
        )

    # ── Report ──
    print(
        f"\n{'Thr':<6} {'N_int':>6} {'N_skip':>6} {'IntAcc':>8} {'SkipAcc':>8} {'CombAcc':>8} {'Δ':>8}"
    )
    print("-" * 58)
    best_r = None
    for r in results_by_threshold:
        int_acc_str = f"{r['intervene_acc']:.4f}" if r["intervene_acc"] else "N/A"
        skip_acc_str = f"{r['skip_acc']:.4f}" if r["skip_acc"] else "N/A"
        print(
            f"{r['risk_threshold']:<6.2f} {r['n_intervened']:>6} {r['n_skipped']:>6} "
            f"{int_acc_str:>8} {skip_acc_str:>8} {r['combined_acc']:>8.4f} {r['delta']:>+8.4f}"
        )
        if best_r is None or r["delta"] > best_r["delta"]:
            best_r = r

    if best_r:
        print(
            f"\nBest: thr={best_r['risk_threshold']:.2f}, n_intervened={best_r['n_intervened']}, "
            f"Δ={best_r['delta']:+.4f}"
        )

    # Comparison summary
    print(f"\n{'=' * 40}")
    print(f"Summary:")
    print(f"  Baseline (full):        {baseline_acc:.4f}")
    print(f"  Baseline (knowledge):   {knowledge_acc:.4f}")
    print(f"  Blind intervene:        {blind_acc:.4f}  (Δ={blind_delta:+.4f})")
    if best_r:
        print(
            f"  Detect-then-intervene:  {best_r['combined_acc']:.4f}  (Δ={best_r['delta']:+.4f})"
        )
        print(f"    → n_intervened={best_r['n_intervened']}/{n_knowledge}")
        selectivity = best_r["delta"] - blind_delta
        print(f"    → selectivity gain: {selectivity:+.4f} pp over blind")

    out = {
        "n_total": n_total,
        "baseline_acc": float(baseline_acc),
        "n_knowledge_filtered": int(n_knowledge),
        "knowledge_baseline_acc": float(knowledge_acc),
        "blind_intervene_acc": float(blind_acc),
        "blind_delta": float(blind_delta),
        "detector_auroc": float(detector_auroc),
        "intervention_layer": intervention_layer,
        "lambda": lam,
        "js_early_layer": js_early_layer,
        "js_late_layer": js_late_layer,
        "by_threshold": results_by_threshold,
        "best_threshold": best_r,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    fpath = output_dir / "s1_pipeline_results.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {fpath}")


def _make_collect_hook(storage: dict, key: str):
    def hook(activation, hook=None):
        storage[key] = activation.detach()
        return activation

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
    print(f"Letter token IDs: {letter_ids}")

    # Load direction from I1
    direction_path = output_dir / "i1_directions.pt"
    if direction_path.exists():
        directions = torch.load(direction_path, map_location="cpu")
        direction = directions["mean_diff"]["11"]  # L11 mean-diff direction
        print(f"Loaded L11 mean-diff direction from {direction_path}")
    else:
        print(f"WARNING: {direction_path} not found, using random direction")
        d_model = model.cfg.d_model
        direction = torch.randn(d_model)
        direction = direction / direction.norm()

    print(f"Loading HellaSwag validation ({args.n_eval} samples)...")
    eval_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed)

    run_s1(
        model,
        eval_samples,
        letter_ids,
        direction,
        output_dir,
        intervention_layer=11,
        lam=0.5,
        js_early_layer=17,
        js_late_layer=26,
    )

    print("\nS1 — Done")


if __name__ == "__main__":
    main()
