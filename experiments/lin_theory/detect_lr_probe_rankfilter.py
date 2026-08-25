"""Detection rerun on TriviaQA with knowability (rank) filtering.

Motivation (2026-08-25):
  - HellaSwag knowledge filtering lifted max_p AUROC 0.68 -> 0.87
    (main_knowledge_filtered.py: 4-way softmax P(correct) > 0.3).
  - The TriviaQA detection pillar (LR probe 0.7708 @L26, truth direction
    0.7564 @L18) was evaluated on ALL samples — NO knowledge filter.
  - Question: is 0.77 a task ceiling, or is it dragged down by "ignorance"
    samples (y_true rank > 50 — the model does not know the answer at all)?

Protocol (same clean CV as detect_lr_probe_cv.py):
  - Fixed prompts (format_prompt truncates context, Question intact)
  - Exact labels via greedy generation + check_correct_exact
  - Per-layer LR probe: Pipeline(StandardScaler, LogisticRegression) inside
    5-fold StratifiedKFold CV (scaler never sees test folds)
  - Knowability filter: y_true rank in final-layer logits <= threshold
    (same 1-indexed top-50 convention as validate_s14_tldc.py); AUROC is
    computed on the knowable subset in addition to the full set.
  - No hyperparameter tuning on the evaluation labels

Usage:
    python detect_lr_probe_rankfilter.py --n_samples 200 --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct_exact
from common import get_first_answer_token_id
from validate_s14_tldc import get_y_true_rank


def extract_hidden_rank_label(model, tokenizer, sample, device):
    """One hooked forward: last-token resid_post at ALL layers + y_true rank.

    Returns (list_of_h_np [n_layers][d_model], label, rank or None).
    """
    prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]  # keep TAIL (Question) — see code review Critical 1
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers

    storage = {}

    def _make_hook(li):
        def hook(act, hook=None):
            storage[li] = act[:, last_pos, :].detach()
            return act
        return hook

    fwd_hooks = [
        (f"blocks.{li}.hook_resid_post", _make_hook(li)) for li in range(n_layers)
    ]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    # y_true rank in FINAL-layer logits (knowability filter, TLDC convention)
    y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
    rank = get_y_true_rank(logits, y_true_id) if y_true_id is not None else None

    # Greedy generation for the label
    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(19):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    label = int(check_correct_exact(ans, sample["answers"]))

    return [storage[li].float().cpu().numpy().flatten() for li in range(n_layers)], label, rank


def main():
    parser = argparse.ArgumentParser(
        description="LR probe detection on TriviaQA, full vs knowability-filtered"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rank_thresholds",
        type=str,
        default="20,50,100",
        help="y_true rank cutoffs for the knowable subset (comma list). "
        "50 matches validate_s14_tldc.py.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    thresholds = [int(t) for t in args.rank_thresholds.split(",") if t.strip()]

    print("=" * 64)
    print("LR probe detection, TriviaQA: full set vs knowability-filtered")
    print(f"  model={args.model} n={args.n_samples} seed={args.seed}")
    print(f"  rank thresholds: {thresholds}")
    print("=" * 64)

    # ── Load model + data ──
    print("\n[1/3] Loading model & data...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    model.eval()
    tokenizer = model.tokenizer
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    n_layers = model.cfg.n_layers
    print(f"  Loaded in {time.time()-t0:.0f}s | {len(samples)} samples | {n_layers} layers")

    # ── Extract hidden states + labels + ranks ──
    print(f"\n[2/3] Extracting hidden states + labels + y_true ranks...")
    t0 = time.time()
    H = [[] for _ in range(n_layers)]
    y, ranks = [], []
    n_correct = 0
    for s in tqdm(samples, desc="  Extract"):
        hs, label, rank = extract_hidden_rank_label(model, tokenizer, s, device)
        for li in range(n_layers):
            H[li].append(hs[li])
        y.append(label)
        ranks.append(rank)
        n_correct += label

    y = np.array(y, dtype=np.int32)
    ranks = np.array([r if r is not None else 10**9 for r in ranks], dtype=np.int64)
    H = [np.stack(h, axis=0) for h in H]
    n_know50 = int((ranks <= 50).sum())
    print(f"  Correct: {n_correct}/{len(samples)} ({n_correct/len(samples):.1%})")
    print(f"  Knowable (rank<=50): {n_know50}/{len(samples)}")
    print(f"  Extraction time: {time.time()-t0:.0f}s")

    # ── Evaluation ──
    print(f"\n[3/3] LR probe evaluation ({args.n_folds}-fold CV)")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)

    def _probe_cv(X, y_sub):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        )
        return cross_val_score(pipe, X, y_sub, cv=skf, scoring="roc_auc")

    results = {"config": vars(args), "protocol": "5-fold stratified CV, exact labels, fixed prompts (tail-truncation), knowability = final-layer y_true rank", "groups": {}}

    def _eval_group(tag, y_sub, H_sub):
        per_layer = []
        for li in range(n_layers):
            fold_scores = _probe_cv(H_sub[li], y_sub)
            per_layer.append(
                {
                    "layer": li,
                    "auroc": float(fold_scores.mean()),
                    "auroc_std": float(fold_scores.std()),
                    "folds": [float(v) for v in fold_scores],
                }
            )
        best = max(per_layer, key=lambda d: d["auroc"])
        X_joint = np.concatenate(H_sub, axis=1)
        joint = _probe_cv(X_joint, y_sub)
        print(
            f"    {tag:<34} n={len(y_sub):>4} acc={y_sub.mean():.3f}  "
            f"best L{best['layer']:>2} = {best['auroc']:.4f}±{best['auroc_std']:.4f}  "
            f"joint_all = {joint.mean():.4f}±{joint.std():.4f}"
        )
        return {
            "n": int(len(y_sub)),
            "accuracy": float(y_sub.mean()),
            "best_layer": best["layer"],
            "best_auroc": best["auroc"],
            "best_auroc_std": best["auroc_std"],
            "per_layer": per_layer,
            "joint_all_auroc": float(joint.mean()),
            "joint_all_auroc_std": float(joint.std()),
        }

    # Full set (no filter) — the 0.7708 pillar baseline
    results["groups"]["full"] = _eval_group("full (no filter)", y, H)

    # Knowability-filtered subsets
    for thr in thresholds:
        mask = ranks <= thr
        if mask.sum() < 40:
            print(f"    rank<={thr}: only {mask.sum()} samples — skip")
            results["groups"][f"rank_le_{thr}"] = {"n": int(mask.sum()), "skipped": "too few"}
            continue
        y_sub = y[mask]
        if len(np.unique(y_sub)) < 2 or min(np.bincount(y_sub)) < 10:
            print(f"    rank<={thr}: n={mask.sum()} class imbalance — skip")
            results["groups"][f"rank_le_{thr}"] = {"n": int(mask.sum()), "skipped": "class imbalance"}
            continue
        H_sub = [h[mask] for h in H]
        results["groups"][f"rank_le_{thr}"] = _eval_group(f"rank <= {thr}", y_sub, H_sub)

    out_path = output_dir / "detect_lr_probe_rankfilter.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nReading guide:")
    print(f"  filtered best AUROC > full (0.77) + 0.03  → 0.77 is partly 'ignorance' pollution,")
    print(f"  detection narrative needs revision")
    print(f"  filtered best AUROC ≈ full (0.77)        → task ceiling confirmed;")
    print(f"  the knowability filter does not help detection on TriviaQA")


if __name__ == "__main__":
    main()
