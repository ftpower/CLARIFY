"""Direction A: KL Signature Matrix for Hallucination Detection.

Based on "Between the Layers Lies the Truth" (Badash et al., ICML 2026).

Usage:
    python main_kl_signature.py
    python main_kl_signature.py --n_samples 200 --sweep
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.hidden_state import extract_post_mlp_states, generate_answer
from src.kl_signature import compute_kl_signatures_batch


def main(
    n_samples: int = 500,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    temperatures: list[float] | None = None,
    alphas: list[float] | None = None,
    output_dir: str = "outputs",
    seed: int = 42,
    save_states: bool = True,
):
    if temperatures is None:
        temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
    if alphas is None:
        alphas = [None, 0.5, 1.0, 2.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    states_cache = output_path / "post_mlp_states.npz"

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    n_layers = model.cfg.n_layers

    # ── Extract post-MLP states ─────────────────────────────────────────
    if states_cache.exists():
        print(f"Loading cached states from {states_cache}")
        cached = np.load(states_cache, allow_pickle=True)
        all_states = [torch.from_numpy(cached[f"s{i}"]) for i in range(n_samples)]
        labels = cached["labels"]
        correct_count = int(labels.sum())
    else:
        print(f"Extracting post-MLP states ({n_samples} samples)...")
        all_states = []
        all_labels = []
        correct_count = 0

        for i, sample in enumerate(samples):
            prompt = format_prompt(
                sample["question"], sample["context"], dataset="hellaswag"
            )
            mlp_states, gen_id = extract_post_mlp_states(model, prompt)
            all_states.append(mlp_states)

            gen_text = model.tokenizer.decode(gen_id).strip()
            is_corr = check_correct(gen_text, sample["answers"], dataset="hellaswag")
            all_labels.append(1 if is_corr else 0)
            if is_corr:
                correct_count += 1

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n_samples} samples processed")

        labels = np.array(all_labels, dtype=np.int32)

        if save_states:
            print(f"Saving states cache to {states_cache}")
            save_dict = {"labels": labels}
            for i, states in enumerate(all_states):
                save_dict[f"s{i}"] = torch.cat([s for s in states], dim=0).numpy()
            np.savez_compressed(states_cache, **save_dict)

    accuracy = correct_count / n_samples

    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{n_samples})")
    print(f"L×L signature dimension: {n_layers}×{n_layers} = {n_layers**2}")

    # ── Free GPU memory ──────────────────────────────────────────────────
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Sweep hyperparameters ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(
        f"Sweeping {len(temperatures)} τ × {len(alphas)} α = "
        f"{len(temperatures) * len(alphas)} configs"
    )
    print(f"{'=' * 60}")

    results = []
    best_auroc = 0.0
    best_config = None

    for tau in temperatures:
        for alpha in alphas:
            alpha_label = f"α={alpha}" if alpha else "α=None"
            print(f"\nτ={tau}, {alpha_label}")

            # Compute KL signatures
            X = compute_kl_signatures_batch(all_states, temperature=tau, alpha=alpha)

            # 5-fold CV LightGBM
            try:
                from lightgbm import LGBMClassifier

                model_cls = LGBMClassifier(
                    n_estimators=200,
                    max_depth=5,
                    num_leaves=31,
                    min_child_samples=20,
                    random_state=seed,
                    verbose=-1,
                )
            except ImportError:
                print("  lightgbm not available, using RandomForest instead")
                from sklearn.ensemble import RandomForestClassifier

                model_cls = RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1
                )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            y_proba = cross_val_predict(
                model_cls, X, labels, cv=cv, method="predict_proba"
            )[:, 1]
            y_pred = cross_val_predict(model_cls, X, labels, cv=cv, method="predict")

            auroc = roc_auc_score(labels, y_proba)
            acc = accuracy_score(labels, y_pred)
            f1 = f1_score(labels, y_pred)

            print(f"  AUROC: {auroc:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")

            results.append(
                {
                    "temperature": tau,
                    "alpha": alpha,
                    "auroc": float(auroc),
                    "accuracy": float(acc),
                    "f1": float(f1),
                }
            )

            if auroc > best_auroc:
                best_auroc = auroc
                best_config = {"tau": tau, "alpha": alpha}

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    baseline = 0.68
    print(f"Baseline (single-point max_p at L28): AUROC = {baseline:.4f}")
    print(f"Best KL signature:                    AUROC = {best_auroc:.4f}")
    print(f"  Config: τ={best_config['tau']}, α={best_config['alpha']}")

    delta = best_auroc - baseline
    if delta > 0.02:
        print(f"✓ KL signature BEATS baseline by +{delta:.4f}")
    elif delta > -0.02:
        print(f"≈ KL signature TIED with baseline (Δ={delta:+.4f})")
    else:
        print(f"✗ KL signature below baseline (Δ={delta:+.4f})")

    # All configs sorted
    print(f"\nAll configs (sorted by AUROC):")
    results_sorted = sorted(results, key=lambda x: -x["auroc"])
    for r in results_sorted:
        a = f"α={r['alpha']}" if r["alpha"] else "None"
        print(f"  τ={r['temperature']:<5} α={a:<8} AUROC={r['auroc']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "method": "KL_Signature",
        "paper": "Between the Layers Lies the Truth (ICML 2026)",
        "model": model_id,
        "dataset": "hellaswag",
        "n_samples": n_samples,
        "n_layers": n_layers,
        "signature_dim": n_layers**2,
        "accuracy": float(accuracy),
        "baseline_auroc": baseline,
        "baseline_description": "single-point max_p at L28",
        "best_auroc": float(best_auroc),
        "best_config": best_config,
        "all_results": results,
    }

    results_file = output_path / "kl_signature_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--no-sweep", action="store_true", help="Use single τ=1.0 α=None only"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_sweep:
        main(
            args.n_samples,
            args.device,
            args.model,
            temperatures=[1.0],
            alphas=[None],
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        main(
            args.n_samples,
            args.device,
            args.model,
            output_dir=args.output_dir,
            seed=args.seed,
        )
