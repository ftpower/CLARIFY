"""Knowledge-filtered hallucination detection.

Hypothesis: detection methods fail because accuracy is too low (52%),
meaning most "incorrect" answers are random noise, not true hallucinations.
If we first filter by P(correct) to select high-knowledge samples, the
correct/incorrect internal state differences should be clearer.

Usage:
    python main_knowledge_filtered.py
    python main_knowledge_filtered.py --n_samples 500 --thresholds 0.1,0.3,0.5,0.7
"""

import gc
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.hidden_state import extract_hidden_states
from src.entropy import compute_logit_lens_entropy, compute_per_layer_auroc
from src.trajectory_features import extract_trajectory_features


def compute_p_correct_hellaswag(
    logits_final: torch.Tensor,
    correct_letter: str,
    model,
) -> float:
    """Compute P(correct_letter) via 4-way softmax over A/B/C/D.

    Args:
        logits_final: [vocab_size] raw logits at last token position.
        correct_letter: 'A', 'B', 'C', or 'D'.
        model: HookedTransformer for tokenizer access.

    Returns:
        float in [0, 1] — probability assigned to the correct letter.
    """
    letters = ["A", "B", "C", "D"]
    letter_ids = [
        model.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters
    ]
    letter_logits = logits_final[letter_ids]  # [4]
    probs = torch.softmax(letter_logits, dim=-1)
    idx = letters.index(correct_letter.upper())
    return float(probs[idx].item())


def main(
    n_samples: int = 500,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    thresholds: list[float] | None = None,
    output_dir: str = "outputs",
    seed: int = 42,
):
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / "knowledge_filtered_data.json"

    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    n_layers = model.cfg.n_layers
    n_total = n_layers + 1

    # ── Extract per-sample data ───────────────────────────────────────
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file) as f:
            per_sample = json.load(f)
    else:
        print(f"Extracting {n_samples} samples...")
        per_sample = []
        for sample in tqdm(samples, desc="Extracting"):
            prompt = format_prompt(
                sample["question"], sample["context"], dataset="hellaswag"
            )
            hs, logits_f, gen_id, gen_text = extract_hidden_states(model, prompt)
            is_correct = check_correct(
                gen_text.strip(), sample["answers"], dataset="hellaswag"
            )
            p_correct = compute_p_correct_hellaswag(
                logits_f, sample["answers"][1], model
            )
            metrics = compute_logit_lens_entropy(hs, W_U, b_U, temperature=1.0)

            per_sample.append(
                {
                    "question": sample["question"],
                    "answers": sample["answers"],
                    "generated_text": gen_text.strip(),
                    "is_correct": is_correct,
                    "p_correct": p_correct,
                    "entropy": metrics["entropy"],
                    "max_prob": metrics["max_prob"],
                    "top5_mass": metrics["top5_mass"],
                }
            )

        with open(cache_file, "w") as f:
            json.dump(per_sample, f, indent=2)
        print(f"Saved to {cache_file}")

    # ── Free GPU ──────────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Full-set stats ────────────────────────────────────────────────
    full_labels = np.array([1 if s["is_correct"] else 0 for s in per_sample])
    full_acc = full_labels.mean()
    print(f"\nFull set: n={len(per_sample)}, acc={full_acc:.4f}")

    # ── Threshold sweep ───────────────────────────────────────────────
    all_results = []

    for thr in thresholds:
        # Filter
        filtered = [s for s in per_sample if s["p_correct"] > thr]
        n_filt = len(filtered)
        if n_filt < 40:
            print(f"\nthr={thr}: {n_filt} samples — too few, skip")
            continue

        filt_labels = np.array([1 if s["is_correct"] else 0 for s in filtered])
        filt_acc = filt_labels.mean()
        n_correct = int(filt_labels.sum())
        n_incorrect = n_filt - n_correct
        min_class = min(n_correct, n_incorrect)

        if min_class < 10:
            print(
                f"\nthr={thr}: acc={filt_acc:.4f} ({n_filt} samples) — class imbalance, skip"
            )
            continue

        print(
            f"\nthr={thr}: acc={filt_acc:.4f} ({n_filt} samples, "
            f"{n_correct}C/{n_incorrect}I)"
        )

        row = {"threshold": thr, "n_samples": n_filt, "accuracy": float(filt_acc)}

        # ── Method 1: max_p at L28 ──────────────────────────────────
        mp_l28 = np.array([s["max_prob"][n_total - 1] for s in filtered])
        auroc_mp = roc_auc_score(filt_labels, mp_l28)
        print(f"  max_p L28:    AUROC={auroc_mp:.4f}")
        row["max_p_l28_auroc"] = float(auroc_mp)

        # ── Method 2: Trajectory shape + LR (5-fold CV) ──────────────
        features = []
        for s in filtered:
            feats = extract_trajectory_features(
                s["entropy"], s["max_prob"], s["top5_mass"]
            )
            features.append(feats)
        feat_names = sorted(features[0].keys())
        X = np.array([[f[name] for name in feat_names] for f in features])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            lr = LogisticRegression(max_iter=2000, random_state=seed)
            y_proba_lr = cross_val_predict(
                lr, X_scaled, filt_labels, cv=cv, method="predict_proba"
            )[:, 1]
            auroc_shape = roc_auc_score(filt_labels, y_proba_lr)
            print(f"  trajectory LR: AUROC={auroc_shape:.4f}")
        except Exception as e:
            auroc_shape = float("nan")
            print(f"  trajectory LR: failed ({e})")
        row["trajectory_lr_auroc"] = (
            float(auroc_shape) if not np.isnan(auroc_shape) else None
        )

        # ── Method 3: Per-layer entropy AUROC (best layer) ───────────
        ent_matrix = np.array([s["entropy"] for s in filtered])  # [N, n_total]
        best_ent_layer = -1
        best_ent_auroc = 0.0
        for li in range(n_total):
            auroc = roc_auc_score(filt_labels, ent_matrix[:, li])
            if auroc > best_ent_auroc:
                best_ent_auroc = auroc
                best_ent_layer = li
        print(f"  best entropy L{best_ent_layer}: AUROC={best_ent_auroc:.4f}")
        row["best_entropy_layer"] = int(best_ent_layer)
        row["best_entropy_auroc"] = float(best_ent_auroc)

        all_results.append(row)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'thr':>6} {'n':>5} {'acc':>8} {'max_p':>8} {'traj_LR':>9} {'best_ent':>9}")
    print("-" * 70)
    baseline_mp = 0.68
    for r in all_results:
        mp_str = f"{r['max_p_l28_auroc']:.4f}"
        lr_str = (
            f"{r['trajectory_lr_auroc']:.4f}"
            if r["trajectory_lr_auroc"]
            else "  failed"
        )
        ent_str = f"{r['best_entropy_auroc']:.4f}"
        print(
            f"{r['threshold']:>6.1f} {r['n_samples']:>5} {r['accuracy']:>8.4f} "
            f"{mp_str:>8} {lr_str:>9} {ent_str:>9}"
        )

    print(f"\nBaseline (no filter, max_p L28): AUROC = {baseline_mp:.4f}")

    # Save
    results = {
        "model": model_id,
        "dataset": "hellaswag",
        "n_samples_full": n_samples,
        "full_accuracy": float(full_acc),
        "baseline_auroc": baseline_mp,
        "threshold_results": all_results,
    }
    results_file = output_path / "knowledge_filtered_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--thresholds", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        args.n_samples,
        args.device,
        args.model,
        [float(t) for t in args.thresholds.split(",")],
        args.output_dir,
        args.seed,
    )
