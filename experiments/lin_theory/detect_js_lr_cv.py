"""Detection rerun on TriviaQA: JS/LR feature set with clean 5-fold CV protocol.

Code-review P0 follow-up. The old phase4 pipeline (HellaSwag, JS/joint 0.936) has
two protocol flaws we must NOT inherit:
  1. select_top_js_pairs() picks the top-5 layer pairs by AUROC on the SAME
     labels used for evaluation (in-sample feature construction).
  2. StandardScaler is fit on the FULL set before cross_val_score (scaler leakage).

This script is protocol-clean:
  - All features are label-free at construction time (no pair selection by
    AUROC, no best-layer selection on labels, no batch SVD).
  - Per-feature AUROC is computed on the full set WITHOUT any fitting step
    (features are fixed scorers) → not in-sample in the fitting sense.
  - Joint logistic regression uses a Pipeline(scaler, LR) INSIDE 5-fold
    StratifiedKFold CV → scaler fit on train folds only.

Features (logit lens at the LAST prompt token, all layers):
  1. max_p_last      max prob, final layer (logit lens of final residual)
  2. entropy_last    entropy of final-layer distribution
  3. top5_mass_last  top-5 probability mass, final layer
  4. max_p_peak      max prob at L_peak (FIXED layer from the C2 CV rerun
                     peak L18 — pre-registered, not selected on this data)
  5. js_union_top10  mean JS divergence over ALL layer pairs; support =
                     union of each layer's top-10 tokens (renormalized)
  6. js_final_top10  mean JS over ALL layer pairs; support = final layer's
                     top-10 tokens (renormalized)
  7. attn_ffn_peak   L2-norm ratio ||attn||/||ffn|| at L_peak

Labels: greedy generation (fixed prompt, tail-truncation safety net) scored
with check_correct_exact.

Usage:
    python detect_js_lr_cv.py --n_samples 200 --seed 42
    # 8B on server: --model Qwen/Qwen3-8B --layer_peak 23
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
from sklearn.metrics import roc_auc_score
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

FEATURE_NAMES = [
    "max_p_last",
    "entropy_last",
    "top5_mass_last",
    "max_p_peak",
    "js_union_top10",
    "js_final_top10",
    "attn_ffn_peak",
]
# Column indices used for ablations
IDX_JS = [4, 5]
IDX_NOJS = [0, 1, 2, 3, 6]
EPS = 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════════════


def _pair_js_mean(probs_list, top10_by_layer, mode):
    """Mean JS divergence over all layer pairs (early < late, early >= 1).

    Args:
        probs_list: list of [vocab] float32 arrays, one per layer.
        top10_by_layer: list of [10] token-index arrays, one per layer.
        mode: "union_top10" — support = union of the two layers' top-10;
              "final_top10" — support = final layer's top-10.

    Returns float (mean JS over pairs).
    """
    n_layers = len(probs_list)
    final_top10 = top10_by_layer[-1]
    total = 0.0
    n_pairs = 0
    for early in range(1, n_layers):
        for late in range(early + 1, n_layers):
            if mode == "union_top10":
                ids = np.union1d(top10_by_layer[early], top10_by_layer[late])
            else:
                ids = final_top10
            p_e = probs_list[early][ids]
            p_l = probs_list[late][ids]
            se, sl = float(p_e.sum()), float(p_l.sum())
            if se <= EPS or sl <= EPS:
                continue  # underflowed support mass (very early layers) — skip pair
            p_e = p_e / se
            p_l = p_l / sl
            m = 0.5 * (p_e + p_l)
            js = 0.5 * (
                np.sum(p_e * np.log((p_e + EPS) / (m + EPS)))
                + np.sum(p_l * np.log((p_l + EPS) / (m + EPS)))
            )
            total += float(js)
            n_pairs += 1
    return total / max(n_pairs, 1)


def extract_sample_features(model, tokenizer, sample, device, layer_peak):
    """Extract per-layer logit-lens probs + sub-layer states for one sample.

    Also generates the greedy answer and scores it with exact match.
    Returns (features_row_or_None, label, generated_text).
    """
    prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]  # keep TAIL (Question) — see code review Critical 1
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers

    storage = {}
    fwd_hooks = []

    def _make_hook(key):
        def hook(act, hook=None):
            storage[key] = act[:, last_pos, :].detach()
            return act
        return hook

    for i in range(n_layers):
        fwd_hooks.append((f"blocks.{i}.hook_resid_post", _make_hook(f"h{i}")))
    fwd_hooks.append((f"blocks.{layer_peak}.hook_attn_out", _make_hook("attn_p")))
    fwd_hooks.append((f"blocks.{layer_peak}.mlp.hook_post", _make_hook("ffn_p")))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    # ── Logit lens at every layer ──
    W_U = model.unembed.W_U  # [d_model, vocab] f16 on device
    b_U = model.unembed.b_U
    probs_list = []
    top10_by_layer = []
    with torch.no_grad():
        for i in range(n_layers):
            h = storage[f"h{i}"].to(W_U.device).to(W_U.dtype)  # [1, d_model]
            l = h @ W_U  # [1, vocab]
            if b_U is not None:
                l = l + b_U
            p = torch.softmax(l.float(), dim=-1)[0].cpu().numpy().astype(np.float32)
            probs_list.append(p)
            top10_by_layer.append(np.argpartition(p, -10)[-10:])

    # ── Features ──
    p_last = probs_list[-1]
    max_p_last = float(p_last.max())
    log_p = np.log(p_last + EPS)
    entropy_last = float(-np.sum(p_last * log_p))
    top5_mass_last = float(np.sum(np.sort(p_last)[-5:]))
    max_p_peak = float(probs_list[layer_peak].max())

    attn_norm = float(storage["attn_p"].norm(p=2).item())
    ffn_norm = float(storage["ffn_p"].norm(p=2).item())
    attn_ffn_peak = attn_norm / (ffn_norm + EPS)

    js_union = _pair_js_mean(probs_list, top10_by_layer, "union_top10")
    js_final = _pair_js_mean(probs_list, top10_by_layer, "final_top10")

    features = np.array(
        [
            max_p_last,
            entropy_last,
            top5_mass_last,
            max_p_peak,
            js_union,
            js_final,
            attn_ffn_peak,
        ],
        dtype=np.float32,
    )

    # ── Greedy generation for the label (no hooks needed after first step) ──
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

    return features, label, ans


def main():
    parser = argparse.ArgumentParser(description="JS/LR detection rerun (TriviaQA)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layer_peak", type=int, default=18,
                        help="Fixed detection-peak layer (1.7B: L18 from C2 CV rerun; 8B: ~L23)")
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
    print("JS/LR detection rerun on TriviaQA (clean 5-fold CV)")
    print(f"  model={args.model} n={args.n_samples} seed={args.seed} peak=L{args.layer_peak}")
    print("=" * 64)

    # ── Load model + data ──
    print("\n[1/3] Loading model & data...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    model.eval()
    tokenizer = model.tokenizer
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  Loaded in {time.time()-t0:.0f}s | {len(samples)} samples")

    # ── Extract features + labels ──
    print(f"\n[2/3] Extracting features + generating labels...")
    t0 = time.time()
    X_rows, y_rows = [], []
    n_correct = 0
    for s in tqdm(samples, desc="  Extract"):
        feats, label, _ans = extract_sample_features(
            model, tokenizer, s, device, args.layer_peak
        )
        X_rows.append(feats)
        y_rows.append(label)
        n_correct += label

    X = np.stack(X_rows, axis=0)  # [N, 7]
    y = np.array(y_rows, dtype=np.int32)
    print(f"  Correct: {n_correct}/{len(samples)} ({n_correct/len(samples):.1%})")
    print(f"  Extraction time: {time.time()-t0:.0f}s")

    # Drop NaN/Inf rows
    valid = np.isfinite(X).all(axis=1)
    if (~valid).any():
        print(f"  Dropping {(~valid).sum()} samples with NaN/Inf features")
        X, y = X[valid], y[valid]

    if len(np.unique(y)) < 2:
        print("  ERROR: single-class labels, AUROC undefined")
        return

    # ── Evaluation ──
    print(f"\n[3/3] Evaluation")
    print(f"\n  Per-feature AUROC (full set, no fitting → protocol-clean):")
    print(f"  {'Feature':<16} {'AUROC':>8}")
    per_feature = {}
    for ci, name in enumerate(FEATURE_NAMES):
        try:
            auc = float(roc_auc_score(y, X[:, ci]))
        except ValueError:
            auc = float("nan")
        per_feature[name] = auc
        print(f"  {name:<16} {auc:>8.4f}")

    # Joint LR, scaler inside the folds
    print(f"\n  Joint logistic regression ({args.n_folds}-fold CV, Pipeline scaler+LR):")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    ablations = {
        "joint_all": list(range(len(FEATURE_NAMES))),
        "joint_no_js": IDX_NOJS,
        "joint_js_only": IDX_JS,
    }
    joint_results = {}
    for tag, cols in ablations.items():
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        try:
            fold_scores = cross_val_score(
                pipe, X[:, cols], y, cv=skf, scoring="roc_auc"
            )
            joint_results[tag] = {
                "mean": float(fold_scores.mean()),
                "std": float(fold_scores.std()),
                "folds": [float(v) for v in fold_scores],
                "features": [FEATURE_NAMES[c] for c in cols],
            }
            print(f"    {tag:<14}: {fold_scores.mean():.4f} ± {fold_scores.std():.4f} "
                  f"folds={[round(v, 3) for v in fold_scores]}")
        except Exception as e:
            joint_results[tag] = {"error": str(e)}
            print(f"    {tag:<14}: FAILED ({e})")

    # ── Save ──
    np.savez_compressed(
        output_dir / "detect_js_lr_features.npz",
        X=X, y=y, feature_names=np.array(FEATURE_NAMES),
    )
    results = {
        "config": {
            "model": args.model,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "layer_peak": args.layer_peak,
            "n_folds": args.n_folds,
            "protocol": (
                "5-fold stratified CV; label-free features (no pair/layer selection "
                "on labels); scaler fit inside folds; exact labels via greedy gen + "
                "check_correct_exact; fixed prompts with tail-truncation safety net"
            ),
        },
        "n_valid": int(len(y)),
        "n_correct": int(y.sum()),
        "correct_rate": float(y.mean()),
        "per_feature_auroc": per_feature,
        "joint": joint_results,
    }
    out_path = output_dir / "detect_js_lr_cv.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved features: {output_dir / 'detect_js_lr_features.npz'}")


if __name__ == "__main__":
    main()
