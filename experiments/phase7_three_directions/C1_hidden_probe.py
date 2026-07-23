"""C1: Hidden State Probe — train classifier on hidden states to predict correctness.

Hypothesis: The hidden state at the last input token position encodes whether the
model "knows" the answer, independent of what the logit lens reads.

Two training modes:
  - in_domain: train/test split on TriviaQA (80/20, 5-fold CV)
  - cross_task: train on HellaSwag hidden states, test on TriviaQA

Classifier: Logistic Regression (simple, fast, interpretable)
Probe features: hidden states from layers [4, 8, 12, 16, 20, 24] concatenated

Based on: FactCheckmate (NeurIPS 2025) — pre-decode MLP on last input token.
          SAPLMA (Azaria & Mitchell 2023) — internal awareness probes.

Usage:
    python C1_hidden_probe.py --n_samples 200 --mode in_domain
    python C1_hidden_probe.py --n_samples 200 --mode cross_task
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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

from shared import load_model_and_data, evaluate_auroc


# ═══════════════════════════════════════════════════════════════════════════════
# Hidden state extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_last_token_hidden_states(
    model, tokenizer, prompt: str, device: str,
    layers: list[int] | None = None,
) -> np.ndarray:
    """Extract hidden states at last input token position for specified layers.

    Single forward pass — no generation. Returns concatenated hidden states.

    Args:
        layers: list of layer indices to extract. None = use [4,8,12,16,20,24].
    Returns:
        np.ndarray of shape [n_layers * d_model], concatenated hidden states.
    """
    if layers is None:
        n = model.cfg.n_layers
        layers = list(range(0, n, max(1, n // 6)))[:6]  # 6 evenly spaced layers

    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    # Hook into specified layers
    residuals = {}

    def _make_hook(name):
        def hook(act, hook=None, **kwargs):
            residuals[name] = act[:, -1, :].detach()
        return hook

    fwd_hooks = [(f"blocks.{i}.hook_resid_post", _make_hook(f"L{i}"))
                 for i in layers]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    # Concatenate
    h_list = []
    for i in layers:
        h = residuals[f"L{i}"].float().cpu().numpy().flatten()
        h_list.append(h)

    return np.concatenate(h_list)


# ═══════════════════════════════════════════════════════════════════════════════
# HellaSwag data for cross-task training
# ═══════════════════════════════════════════════════════════════════════════════

def load_hellaswag_labels(n_samples: int = 500) -> list[dict]:
    """Load HellaSwag data with correctness labels from greedy generation.

    We need to generate answers and label them for training the probe.
    """
    from src.data_loader import load_hellaswag, format_prompt

    samples = load_hellaswag(n_samples=n_samples)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Probe training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_probe_cv(
    X: np.ndarray,  # [N, D]
    y: np.ndarray,  # [N]
    n_folds: int = 5,
) -> dict:
    """Train Logistic Regression with stratified K-fold CV.

    Returns:
        dict with mean_auroc, std_auroc, fold_aurocs, coef_norm.
    """
    # Filter NaN
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]

    if len(y) < 10 or y.std() == 0:
        return {"mean_auroc": float("nan"), "std_auroc": float("nan"),
                "n_valid": len(y)}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                             max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=min(n_folds, min((y == 0).sum(), (y == 1).sum())),
                         shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    return {
        "mean_auroc": float(scores.mean()),
        "std_auroc": float(scores.std()),
        "fold_aurocs": [float(s) for s in scores],
        "n_valid": len(y),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
    }


def evaluate_probe_transfer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """Train on one dataset, test on another (cross-task transfer)."""
    valid_train = np.isfinite(X_train).all(axis=1)
    valid_test = np.isfinite(X_test).all(axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    if len(y_train) < 10 or y_train.std() == 0:
        return {"auroc": float("nan"), "n_train": len(y_train), "n_test": len(y_test)}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                             max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)[:, 1]

    auroc = float(roc_auc_score(y_test, probs))
    auroc = max(auroc, 1 - auroc)  # handle direction

    return {
        "auroc": auroc,
        "n_train": len(y_train),
        "n_train_correct": int(y_train.sum()),
        "n_test": len(y_test),
        "n_test_correct": int(y_test.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="C1: Hidden State Probe")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase7")
    parser.add_argument("--mode", type=str, default="in_domain",
                        choices=["in_domain", "cross_task", "both"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"C1: Hidden State Probe (mode={args.mode})")
    print(f"  Model: {args.model}  Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    print("Loading model & data...")
    t0 = time.time()
    model, tokenizer, samples = load_model_and_data(
        n_samples=args.n_samples, seed=args.seed,
        device=device, model_id=args.model,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    from src.data_loader import format_prompt, check_correct

    if args.mode in ("in_domain", "both"):
        # ── In-domain: TriviaQA train/test ──
        print(f"\n{'─'*40}")
        print("In-domain probe (TriviaQA → TriviaQA)")
        print(f"{'─'*40}")

        # Generate answers for labels
        print("Generating answers for labels...")
        t0 = time.time()
        X_list, y_list = [], []
        correct_count = 0

        for s in tqdm(samples, desc="C1 extract HS"):
            prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")

            # Extract hidden states
            h = extract_last_token_hidden_states(model, tokenizer, prompt, device)

            # Generate answer for label
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            generated_ids = []
            for _ in range(20):
                with torch.no_grad():
                    logits = model(tokens)
                next_id = int(logits[0, -1, :].argmax().item())
                generated_ids.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    break
                tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
            answer_text = tokenizer.decode(generated_ids).strip()
            is_correct = check_correct(answer_text, s["answers"], dataset="triviaqa")
            if is_correct:
                correct_count += 1

            X_list.append(h)
            y_list.append(1 if is_correct else 0)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)
        print(f"  Correct: {correct_count}/{len(samples)} ({correct_count/len(samples):.1%})")
        print(f"  Time: {time.time()-t0:.1f}s")

        # 5-fold CV
        result_cv = train_probe_cv(X, y, n_folds=5)
        print(f"\n  5-fold CV AUROC: {result_cv['mean_auroc']:.4f} ± {result_cv['std_auroc']:.4f}")
        print(f"  N={result_cv['n_valid']}, correct={result_cv['n_correct']}, "
              f"incorrect={result_cv['n_incorrect']}")
        for i, s in enumerate(result_cv.get("fold_aurocs", [])):
            print(f"    Fold {i}: {s:.4f}")

    if args.mode in ("cross_task", "both"):
        # ── Cross-task: HellaSwag → TriviaQA ──
        print(f"\n{'─'*40}")
        print("Cross-task probe (HellaSwag → TriviaQA)")
        print(f"{'─'*40}")

        from src.data_loader import load_hellaswag

        n_hs = min(args.n_samples, 500)
        print(f"Loading {n_hs} HellaSwag samples for training...")
        hs_samples = load_hellaswag(n_samples=n_hs)

        # Extract HellaSwag hidden states + labels
        print("Extracting HellaSwag hidden states...")
        t0 = time.time()
        X_train_list, y_train_list = [], []
        hs_correct = 0

        for s in tqdm(hs_samples, desc="C1 HellaSwag HS"):
            prompt = format_prompt(s["question"], s["context"], dataset="hellaswag")
            h = extract_last_token_hidden_states(model, tokenizer, prompt, device)

            # Check if model picks correct answer letter
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]

            with torch.no_grad():
                logits = model(tokens)

            # Check which of A, B, C, D has highest probability
            letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0]
                         for c in ["A", "B", "C", "D"]]
            letter_probs = torch.softmax(logits[0, -1, letter_ids].float(), dim=-1)
            predicted_letter = ["A", "B", "C", "D"][int(letter_probs.argmax().item())]
            correct_letter = s["answers"][1]  # label_letter

            is_correct = (predicted_letter.upper() == correct_letter.upper())
            if is_correct:
                hs_correct += 1

            X_train_list.append(h)
            y_train_list.append(1 if is_correct else 0)

        X_train = np.stack(X_train_list, axis=0)
        y_train = np.array(y_train_list)
        print(f"  HellaSwag accuracy: {hs_correct}/{n_hs} ({hs_correct/n_hs:.1%})")
        print(f"  Time: {time.time()-t0:.1f}s")

        # Extract TriviaQA hidden states (reuse from in_domain if available)
        if "X" not in dir():
            print("Extracting TriviaQA hidden states...")
            X_test_list, y_test_list = [], []
            t_correct = 0
            for s in tqdm(samples, desc="C1 TriviaQA HS"):
                prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
                h = extract_last_token_hidden_states(model, tokenizer, prompt, device)
                tokens = model.to_tokens(prompt, prepend_bos=True)
                if tokens.shape[1] > 1024:
                    tokens = tokens[:, :1024]
                gids = []
                for _ in range(20):
                    with torch.no_grad():
                        logits = model(tokens)
                    nid = int(logits[0, -1, :].argmax().item())
                    gids.append(nid)
                    if nid == tokenizer.eos_token_id:
                        break
                    tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
                ans = tokenizer.decode(gids).strip()
                ic = check_correct(ans, s["answers"], dataset="triviaqa")
                if ic:
                    t_correct += 1
                X_test_list.append(h)
                y_test_list.append(1 if ic else 0)
            X_test = np.stack(X_test_list, axis=0)
            y_test = np.array(y_test_list)
            print(f"  TriviaQA accuracy: {t_correct}/{len(samples)}")

        # Transfer evaluation
        result_transfer = evaluate_probe_transfer(X_train, y_train, X_test, y_test)
        print(f"\n  Cross-task transfer AUROC: {result_transfer['auroc']:.4f}")
        print(f"  Train: {result_transfer['n_train']} (correct={result_transfer['n_train_correct']})")
        print(f"  Test:  {result_transfer['n_test']} (correct={result_transfer['n_test_correct']})")

    # Save
    output = {}
    if args.mode in ("in_domain", "both") and "result_cv" in dir():
        output["in_domain"] = result_cv
    if args.mode in ("cross_task", "both") and "result_transfer" in dir():
        output["cross_task"] = result_transfer

    save_path = output_dir / "C1_hidden_probe.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"C1 complete — hidden states probes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
