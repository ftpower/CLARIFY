"""Phase 4.3 (Plan 2): Three-Zone Two-Stage Hallucination Detection.

Formalizes the preemptive+post-hoc detection into a three-zone framework inspired
by program analysis (static analysis → dynamic verification):

  Stage 1 (Preemptive): MLP on last-input-token hidden state → score₁ ∈ [0,1]
    - score₁ < τ_low  → CLEAN (trust model, skip Stage 2)
    - score₁ > τ_high → FLAG  (high risk, skip Stage 2 — already caught)
    - otherwise       → GRAY  (ambiguous, escalate to Stage 2)

  Stage 2 (Post-hoc): Multi-signal LR detector → score₂ ∈ [0,1]
    - score₂ > 0.5 → FLAG
    - score₂ ≤ 0.5 → CLEAN

Key metrics beyond standard AUROC:
  - Skip Rate: fraction bypassing Stage 2 (efficiency)
  - False Skip Rate: hallucinations missed in CLEAN zone (safety)
  - Efficiency Gain: compute saved by skipping Stage 2

Usage:
    python main_two_stage_detection.py --n_samples 500 --device cuda
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase4_generalization"))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt
from phase4_utils.hidden_state_extended import extract_all_sub_layer_states
from phase4_utils.generalization_features import (
    compute_d2_js_topk, select_top_js_pairs, compute_d2_js_score,
    compute_haloscope_zeta_batch, compute_attn_ffn_ratio,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Preemptive MLP
# ═══════════════════════════════════════════════════════════════════════════════


class PreemptiveMLP(torch.nn.Module):
    def __init__(self, d_model, hidden1=128, hidden2=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_stage1_mlp(X, y, n_splits=5, epochs=100, lr=1e-3, seed=42):
    """Train Stage 1 MLP with stratified CV, return out-of-fold probabilities."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = X.shape[1]

    all_probs = np.zeros(len(y), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        X_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)

        model = PreemptiveMLP(d_model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        pos_w = torch.tensor([n_neg / max(n_pos, 1)], device=device)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X_tr_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_t, pos_weight=pos_w)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            all_probs[val_idx] = torch.sigmoid(model(X_va_t)).cpu().numpy()

    return all_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Two-stage evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_two_stage(
    stage1_scores: np.ndarray,
    stage2_scores: np.ndarray,
    labels: np.ndarray,
    tau_low: float = 0.2,
    tau_high: float = 0.8,
    tau_stage2: float = 0.5,
) -> dict:
    """Evaluate the three-zone two-stage detection framework.

    Args:
        stage1_scores: [N] Stage 1 risk scores (higher = more likely hallucination).
        stage2_scores: [N] Stage 2 risk scores.
        labels: [N] ground truth (1=correct, 0=hallucination).
        tau_low: Below this → CLEAN (skip Stage 2).
        tau_high: Above this → FLAG (skip Stage 2, already flagged).
        tau_stage2: Stage 2 decision threshold.

    Returns:
        dict with all evaluation metrics.
    """
    n_total = len(labels)
    n_hallucinations = int((1 - labels).sum())

    # Zone assignment
    clean_mask = stage1_scores < tau_low
    flag_mask = stage1_scores > tau_high
    gray_mask = ~(clean_mask | flag_mask)

    n_clean = int(clean_mask.sum())
    n_flag = int(flag_mask.sum())
    n_gray = int(gray_mask.sum())

    # Stage 2 decisions (only for GRAY zone)
    stage2_decisions = stage2_scores > tau_stage2
    final_flag = np.zeros(n_total, dtype=bool)
    final_flag[flag_mask] = True  # Stage 1 direct flag
    final_flag[gray_mask] = stage2_decisions[gray_mask]  # Stage 2 decision

    # Metric 1: AUROC(joint)
    # Combine: CLEAN → score=0, FLAG → score=1, GRAY → stage2_score
    joint_scores = np.zeros(n_total, dtype=np.float32)
    joint_scores[flag_mask] = 1.0
    joint_scores[gray_mask] = stage2_scores[gray_mask]
    joint_scores[clean_mask] = 0.0

    try:
        joint_auroc = roc_auc_score(1 - labels, joint_scores)  # hallucination=positive
    except ValueError:
        joint_auroc = float("nan")

    # Metric 2: Skip Rate (bypass Stage 2)
    skip_rate = (n_clean + n_flag) / n_total

    # Metric 3: False Skip Rate (hallucinations in CLEAN zone)
    clean_hallucinations = int(((1 - labels) & clean_mask).sum())
    false_skip_rate = clean_hallucinations / max(n_clean, 1)

    # Metric 4: Efficiency Gain (assume Stage 2 costs 10× more than Stage 1)
    stage1_cost = 1.0
    stage2_cost = 10.0
    total_cost = n_total * stage1_cost + n_gray * stage2_cost
    baseline_cost = n_total * (stage1_cost + stage2_cost)
    efficiency_gain = 1.0 - (total_cost / baseline_cost)

    # Metric 5: Recall at τ_low (fraction of hallucinations NOT in CLEAN)
    if n_hallucinations > 0:
        recall_tau_low = 1.0 - clean_hallucinations / n_hallucinations
    else:
        recall_tau_low = float("nan")

    # Final accuracy of the two-stage system
    # (correct predictions that are NOT flagged)
    correct_and_clean = int(((labels == 1) & ~final_flag).sum())
    correct_and_flagged = int(((labels == 1) & final_flag).sum())
    incorrect_and_flagged = int(((labels == 0) & final_flag).sum())
    incorrect_and_clean = int(((labels == 0) & ~final_flag).sum())

    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "n_total": n_total,
        "n_clean": n_clean,
        "n_gray": n_gray,
        "n_flag": n_flag,
        "skip_rate": float(skip_rate),
        "false_skip_rate": float(false_skip_rate),
        "clean_hallucinations": clean_hallucinations,
        "recall_at_tau_low": float(recall_tau_low),
        "efficiency_gain": float(efficiency_gain),
        "joint_auroc": float(joint_auroc) if not np.isnan(joint_auroc) else None,
        "outcomes": {
            "correct_clean": correct_and_clean,
            "correct_flagged": correct_and_flagged,
            "incorrect_flagged": incorrect_and_flagged,
            "incorrect_clean": incorrect_and_clean,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.3: Two-Stage Detection Framework"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preemptive_layer", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "two_stage_features.npz"

    if args.skip_extract and cache_path.exists():
        print(f"Loading cached from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data = {k: cached[k] for k in cached.files}
    else:
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        W_U = model.unembed.W_U
        b_U = model.unembed.b_U

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

        samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

        N = len(samples)
        letters = ["A", "B", "C", "D"]
        letter_tok_ids = [letter_ids[l] for l in letters]

        labels = np.zeros(N, dtype=np.int32)
        p_correct_arr = np.zeros(N, dtype=np.float32)
        preemptive_states = np.zeros((N, d_model), dtype=np.float32)
        all_choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
        all_maxp = np.zeros((N, n_layers), dtype=np.float32)

        hidden_at_halo = []
        attn_at_ratio = []
        ffn_at_ratio = []

        for idx, sample in enumerate(tqdm(samples, desc="Extracting")):
            prompt = format_prompt(
                sample["question"], sample["context"], dataset="hellaswag"
            )
            correct_letter = sample["answers"][1].upper()
            sub = extract_all_sub_layer_states(model, prompt)

            preemptive_states[idx] = (
                sub["hidden"][args.preemptive_layer][0, :].detach().cpu().to(torch.float32)
            )
            hidden_at_halo.append(sub["hidden"][17][0, :].cpu())
            attn_at_ratio.append(sub["attn"][17][0, :].cpu())
            ffn_at_ratio.append(sub["ffn"][17][0, :].cpu())

            for li in range(n_layers):
                h = sub["hidden"][li].to(W_U.device)
                logits_L = h @ W_U
                if b_U is not None:
                    logits_L = logits_L + b_U.to(W_U.device)
                all_maxp[idx, li] = torch.softmax(logits_L.float(), dim=-1).max().item()
                choice_logits = logits_L[0, letter_tok_ids]
                all_choice_probs[idx, li, :] = torch.softmax(
                    choice_logits.float(), dim=-1
                ).detach().cpu().to(torch.float32)

            logits_last = sub["logits"]
            cf = logits_last[letter_tok_ids]
            pf = torch.softmax(cf.float(), dim=-1)
            pred_idx = pf.argmax().item()
            labels[idx] = int(letters[pred_idx] == correct_letter)
            p_correct_arr[idx] = pf[letters.index(correct_letter)].item()

        # Post-hoc features
        js_scores = compute_d2_js_score(all_choice_probs, 17, 26)
        hidden_matrix = torch.stack(hidden_at_halo).numpy()
        halo_zeta = compute_haloscope_zeta_batch(hidden_matrix, k=5)
        ratios = np.array(
            [compute_attn_ffn_ratio(a, f) for a, f in zip(attn_at_ratio, ffn_at_ratio)],
            dtype=np.float32,
        )

        # Best max_p
        best_mp_auroc = 0.0
        best_mp_layer = 0
        for li in range(n_layers):
            try:
                auc = roc_auc_score(labels, all_maxp[:, li])
            except ValueError:
                auc = 0.5
            if auc > best_mp_auroc:
                best_mp_auroc = auc
                best_mp_layer = li

        data = {
            "labels": labels,
            "p_correct": p_correct_arr,
            "preemptive_states": preemptive_states,
            "js_scores": js_scores,
            "halo_zeta": halo_zeta,
            "attn_ffn_ratio": ratios,
            "max_p": all_maxp[:, best_mp_layer],
            "best_mp_layer": best_mp_layer,
            "best_mp_auroc": best_mp_auroc,
        }

        np.savez_compressed(cache_path, **data)
        print(f"Cached to {cache_path}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Knowledge filter ──
    labels = data["labels"]
    p_correct = data["p_correct"]
    filt_mask = p_correct > 0.3
    filt_labels = labels[filt_mask]
    n_filt = filt_mask.sum()
    print(f"\nFiltered: {n_filt}/{len(labels)}, acc={filt_labels.mean():.4f}")

    # ── Stage 1: Preemptive MLP ──
    print(f"\n{'=' * 60}")
    print("Stage 1: Preemptive MLP Training")
    print(f"{'=' * 60}")

    X1 = data["preemptive_states"][filt_mask]
    y1 = filt_labels
    stage1_probs = train_stage1_mlp(X1, y1, n_splits=5, epochs=args.epochs, seed=args.seed)
    try:
        stage1_auroc = roc_auc_score(y1, stage1_probs)
    except ValueError:
        stage1_auroc = float("nan")
    print(f"Stage 1 AUROC (OOF): {stage1_auroc:.4f}")

    # ── Stage 2: Post-hoc LR ──
    print(f"\n{'=' * 60}")
    print("Stage 2: Post-hoc Multi-Signal LR")
    print(f"{'=' * 60}")

    X2 = np.stack([
        data["js_scores"][filt_mask],
        data["halo_zeta"][filt_mask],
        data["attn_ffn_ratio"][filt_mask],
        data["max_p"][filt_mask],
    ], axis=1)

    # Handle NaN
    valid = ~np.isnan(X2).any(axis=1)
    X2_clean = X2[valid]
    y2_clean = y1[valid]
    nan_dropped = (~valid).sum()

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2_clean)

    lr2 = LogisticRegression(max_iter=2000)
    try:
        stage2_auroc = cross_val_score(
            lr2, X2_scaled, y2_clean, cv=5, scoring="roc_auc"
        ).mean()
    except Exception:
        stage2_auroc = float("nan")
    lr2.fit(X2_scaled, y2_clean)
    stage2_probs = lr2.predict_proba(X2_scaled)[:, 1]
    print(f"Stage 2 AUROC (5-fold CV): {stage2_auroc:.4f}")
    if nan_dropped > 0:
        print(f"  (dropped {nan_dropped} NaN samples)")

    # ── Two-stage evaluation ──
    print(f"\n{'=' * 60}")
    print("Two-Stage Framework Evaluation")
    print(f"{'=' * 60}")
    print(f"{'τ_low':<8} {'τ_high':<8} {'Skip%':<8} {'FalseSkip%':<12} "
          f"{'Recall':<8} {'Efficiency':<12} {'JointAUROC':<10}")
    print("-" * 70)

    all_two_stage_results = []
    best_efficiency = 0.0
    best_config_2s = None

    for tau_low in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for tau_high in [0.7, 0.75, 0.8, 0.85, 0.9]:
            result = evaluate_two_stage(
                stage1_probs, stage2_probs, y2_clean,
                tau_low=tau_low, tau_high=tau_high,
            )
            all_two_stage_results.append(result)

            print(
                f"{tau_low:<8.2f} {tau_high:<8.2f} "
                f"{result['skip_rate']:<8.1%} {result['false_skip_rate']:<12.1%} "
                f"{result['recall_at_tau_low']:<8.3f} "
                f"{result['efficiency_gain']:<12.1%} "
                f"{result['joint_auroc']:<10.4f}"
                if result["joint_auroc"] else
                f"{tau_low:<8.2f} {tau_high:<8.2f} "
                f"{result['skip_rate']:<8.1%} {result['false_skip_rate']:<12.1%} "
                f"{result['recall_at_tau_low']:<8.3f} "
                f"{result['efficiency_gain']:<12.1%} N/A"
            )

            if result["skip_rate"] > 0.5 and result["false_skip_rate"] < 0.05:
                if result["efficiency_gain"] > best_efficiency:
                    best_efficiency = result["efficiency_gain"]
                    best_config_2s = result

    # ── Check criteria ──
    print(f"\n{'=' * 60}")
    print("Criteria Check")
    print(f"{'=' * 60}")

    if best_config_2s:
        print(f"Best config: τ_low={best_config_2s['tau_low']}, "
              f"τ_high={best_config_2s['tau_high']}")
        print(f"  Skip Rate:     {best_config_2s['skip_rate']:.1%} (target > 50%)")
        print(f"  False Skip:    {best_config_2s['false_skip_rate']:.1%} (target < 5%)")
        print(f"  Efficiency:    {best_config_2s['efficiency_gain']:.1%} (target > 30%)")
        print(f"  Joint AUROC:   {best_config_2s['joint_auroc']:.4f} "
              f"(target > {stage2_auroc:.4f} + 0.01)")

        joint_ok = (
            best_config_2s["joint_auroc"] is not None
            and best_config_2s["joint_auroc"] > stage2_auroc + 0.01
        )
        skip_ok = best_config_2s["skip_rate"] > 0.5
        false_ok = best_config_2s["false_skip_rate"] < 0.05

        if joint_ok and skip_ok and false_ok:
            print("\n✅ Two-stage framework MEETS all criteria!")
        else:
            print(f"\n⚠ Criteria NOT fully met: joint={joint_ok}, "
                  f"skip={skip_ok}, false_skip={false_ok}")
    else:
        print("⚠ No configuration meets Skip > 50% AND False Skip < 5%")
        print("   Two-stage framework may not be viable at this model scale.")

    # Save
    output = {
        "config": {"n_samples": args.n_samples, "model": args.model,
                   "preemptive_layer": args.preemptive_layer,
                   "n_filtered": int(n_filt), "filtered_acc": float(filt_labels.mean())},
        "stage1": {"auroc_oof": float(stage1_auroc)},
        "stage2": {"auroc_cv": float(stage2_auroc), "n_features": 4,
                   "features": ["js", "halo_zeta", "attn_ffn", "max_p"]},
        "two_stage_sweep": all_two_stage_results,
        "best_config": best_config_2s,
        "criteria_met": {
            "joint_auroc_above_stage2": joint_ok if best_config_2s else False,
            "skip_rate_above_50": skip_ok if best_config_2s else False,
            "false_skip_below_5": false_ok if best_config_2s else False,
        } if best_config_2s else None,
    }

    with open(output_dir / "two_stage_detection_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'two_stage_detection_results.json'}")


if __name__ == "__main__":
    main()
