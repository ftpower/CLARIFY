"""Detection rerun on TriviaQA: logistic-regression probe on hidden states, 5-fold CV.

Code-review P0 follow-up, step A of the detection-pillar decision. The C2
rerun used the mean-difference direction v (a weak classifier); this script
adds the standard strong baseline — a linear probe (logistic regression)
trained on last-token hidden states.

Protocol (clean):
  - Fixed prompts (format_prompt truncates context, Question intact;
    tail-truncation safety net)
  - Exact labels via greedy generation + check_correct_exact
  - Per-layer LR probe: Pipeline(StandardScaler, LogisticRegression) inside
    5-fold StratifiedKFold CV (scaler never sees test folds)
  - Joint probes: (a) all layers concatenated, (b) L_peak + last layer
  - No hyperparameter tuning on the evaluation labels

Usage:
    python detect_lr_probe_cv.py --n_samples 200 --seed 42
    # 8B on server: --model Qwen/Qwen3-8B
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

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct_exact


def extract_hidden(model, tokenizer, sample, device):
    """One hooked forward: capture last-token resid_post at ALL layers.

    Also generates the greedy answer and scores it with exact match.
    Returns (list_of_h_np [n_layers][d_model], label).
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

    # Greedy generation for the label
    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(19):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat(
            [tokens, torch.tensor([[nid]], device=device)], dim=1
        )
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    label = int(check_correct_exact(ans, sample["answers"]))

    return [storage[li].float().cpu().numpy().flatten() for li in range(n_layers)], label


def main():
    parser = argparse.ArgumentParser(description="LR probe detection rerun (TriviaQA)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layer_peak", type=int, default=18,
                        help="Detection-peak layer for the L_peak+last joint probe (1.7B: L18)")
    parser.add_argument("--seed", type=int, default=42)
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

    print("=" * 64)
    print("LR probe detection rerun on TriviaQA (5-fold CV)")
    print(f"  model={args.model} n={args.n_samples} seed={args.seed}")
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

    # ── Extract hidden states + labels ──
    print(f"\n[2/3] Extracting hidden states + generating labels...")
    t0 = time.time()
    H = [[] for _ in range(n_layers)]
    y = []
    n_correct = 0
    for s in tqdm(samples, desc="  Extract"):
        hs, label = extract_hidden(model, tokenizer, s, device)
        for li in range(n_layers):
            H[li].append(hs[li])
        y.append(label)
        n_correct += label

    y = np.array(y, dtype=np.int32)
    H = [np.stack(h, axis=0) for h in H]  # each [N, d_model]
    print(f"  Correct: {n_correct}/{len(samples)} ({n_correct/len(samples):.1%})")
    print(f"  Extraction time: {time.time()-t0:.0f}s")

    if len(np.unique(y)) < 2:
        print("  ERROR: single-class labels, AUROC undefined")
        return

    # ── Evaluation ──
    print(f"\n[3/3] LR probe evaluation ({args.n_folds}-fold CV)")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)

    def _probe_cv(X):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        )
        return cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")

    # Per-layer probes
    print(f"\n  Per-layer LR probe:")
    print(f"  {'Layer':>6s}  {'AUROC±std':>14s}")
    per_layer = []
    for li in range(n_layers):
        fold_scores = _probe_cv(H[li])
        per_layer.append({
            "layer": li,
            "auroc": float(fold_scores.mean()),
            "auroc_std": float(fold_scores.std()),
            "folds": [float(v) for v in fold_scores],
        })
        print(f"  {li:>6d}  {fold_scores.mean():>8.4f}±{fold_scores.std():.4f}")

    best = max(per_layer, key=lambda d: d["auroc"])
    print(f"\n  Best single-layer probe: L{best['layer']} = {best['auroc']:.4f}±{best['auroc_std']:.4f}")

    # Joint probes
    print(f"\n  Joint probes:")
    joint = {}
    joint_specs = {
        "joint_all_layers": list(range(n_layers)),
        "joint_peak_last": [args.layer_peak, n_layers - 1],
    }
    for tag, layers in joint_specs.items():
        X = np.concatenate([H[li] for li in layers], axis=1)
        fold_scores = _probe_cv(X)
        joint[tag] = {
            "layers": layers,
            "auroc": float(fold_scores.mean()),
            "auroc_std": float(fold_scores.std()),
            "folds": [float(v) for v in fold_scores],
        }
        print(f"    {tag:<16} (L{layers[0]}..L{layers[-1]}): {fold_scores.mean():.4f}±{fold_scores.std():.4f}")

    # ── Save ──
    np.savez_compressed(
        output_dir / "detect_lr_probe_hidden.npz",
        **{f"h_L{li}": H[li] for li in range(n_layers)},
        y=y,
    )
    results = {
        "config": {
            "model": args.model,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "layer_peak": args.layer_peak,
            "n_folds": args.n_folds,
            "protocol": (
                "5-fold stratified CV; Pipeline(StandardScaler, LogisticRegression "
                "max_iter=3000 class_weight=balanced) fit inside folds; exact labels "
                "via greedy gen + check_correct_exact; fixed prompts with tail-truncation "
                "safety net; no hyperparameter tuning on evaluation labels"
            ),
        },
        "n_valid": int(len(y)),
        "n_correct": int(y.sum()),
        "correct_rate": float(y.mean()),
        "per_layer": per_layer,
        "best_layer": best["layer"],
        "best_auroc": best["auroc"],
        "joint": joint,
    }
    out_path = output_dir / "detect_lr_probe_cv.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved hidden states: {output_dir / 'detect_lr_probe_hidden.npz'}")


if __name__ == "__main__":
    main()
