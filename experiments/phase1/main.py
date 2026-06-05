"""Main pipeline: measure per-layer hallucination channel width q^(ℓ).

Usage:
    python main.py --n_samples 100
    python main.py --n_samples 200 --dataset squad
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_triviaqa,
    load_squad,
    load_hellaswag,
    format_prompt,
    check_correct,
)
from src.model_utils import (
    load_model,
    get_per_layer_hidden_states,
    get_confidence_dot,
    get_confidence_cosine,
    calibrate_temperatures,
    compute_p_correct,
)
from src.estimators import compute_all_estimators
from src.visualization import (
    plot_q_curve,
    plot_distribution_overlap,
    plot_estimator_consistency,
)


def compute_q_results(confidences, n_total_layers):
    """Compute q^(ℓ) estimators for all layers. Returns list of result dicts."""
    results = []
    for layer_idx in range(n_total_layers):
        conf_c = np.array(confidences[layer_idx]["correct"])
        conf_i = np.array(confidences[layer_idx]["incorrect"])

        if len(conf_c) < 3 or len(conf_i) < 3:
            results.append(
                {
                    "layer": layer_idx,
                    "q_overlap": np.nan,
                    "q_kl": np.nan,
                    "q_bhattacharyya": np.nan,
                    "q_auroc": np.nan,
                    "n_correct": len(conf_c),
                    "n_incorrect": len(conf_i),
                    "mean_correct_conf": np.nan,
                    "mean_incorrect_conf": np.nan,
                }
            )
            continue

        qs = compute_all_estimators(conf_c, conf_i)
        results.append(
            {
                "layer": layer_idx,
                "q_overlap": qs["overlap"],
                "q_kl": qs["kl"],
                "q_bhattacharyya": qs["bhattacharyya"],
                "q_auroc": qs["auroc"],
                "n_correct": len(conf_c),
                "n_incorrect": len(conf_i),
                "mean_correct_conf": float(np.mean(conf_c)),
                "mean_incorrect_conf": float(np.mean(conf_i)),
            }
        )
    return results


def print_results_table(results, title="q^(ℓ) Results"):
    """Print a formatted results table."""
    print(f"\n{'=' * 60} {title} {'=' * 60}")
    header = (
        f"{'Layer':>6} {'q_overlap':>10} {'q_kl':>10} {'q_BC':>10} "
        f"{'q_AUROC':>10} {'mean_c':>12} {'mean_i':>12}"
    )
    print(header)
    print("-" * len(header))

    def fmt(v, spec=".4f"):
        try:
            return f"{float(v):{spec}}"
        except (ValueError, TypeError):
            return "       nan"

    for r in results:
        print(
            f"{r['layer']:>6} {fmt(r.get('q_overlap')):>10} {fmt(r.get('q_kl')):>10} "
            f"{fmt(r.get('q_bhattacharyya')):>10} {fmt(r.get('q_auroc')):>10} "
            f"{fmt(r.get('mean_correct_conf'), '.6f'):>12} {fmt(r.get('mean_incorrect_conf'), '.6f'):>12}"
        )


def main(
    n_samples: int = 100,
    device: str = "cuda",
    output_dir: str = "outputs",
    dataset: str = "triviaqa",
    model_id: str = "Qwen/Qwen3-1.7B",
):
    torch.manual_seed(42)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {dataset.upper()} ({n_samples} samples)...")
    if dataset == "squad":
        samples = load_squad(n_samples=n_samples)
    elif dataset == "hellaswag":
        samples = load_hellaswag(n_samples=n_samples)
    else:
        samples = load_triviaqa(n_samples=n_samples)

    # --- Load model ---
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = model.cfg.n_layers
    n_total_layers = n_layers + 1  # embed + 12 blocks

    # --- Phase 1: Collect hidden states and labels ---
    print(f"Phase 1: Processing {n_samples} samples...")

    # dot_confidences[layer_idx] = {correct: [...], incorrect: [...]}
    dot_confidences = [{"correct": [], "incorrect": []} for _ in range(n_total_layers)]

    # Store raw hidden states per layer for calibration
    per_layer_h = [[] for _ in range(n_total_layers)]
    per_layer_targets = [[] for _ in range(n_total_layers)]
    per_layer_labels = [[] for _ in range(n_total_layers)]

    correct_count = 0
    incorrect_count = 0
    p_correct_values = []  # per-sample knowledge proxy for wAUROC
    generated_texts = []  # per-sample generated text

    for sample in tqdm(samples, desc="Samples"):
        question = sample["question"]
        answers = sample["answers"]
        context = sample["context"]
        prompt = format_prompt(question, context, dataset=dataset)

        hidden_states, logits_final, gen_id, gen_text = get_per_layer_hidden_states(
            model, prompt
        )
        is_correct = check_correct(gen_text.strip(), answers, dataset=dataset)

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        target_id = gen_id

        # Compute P(correct | final_logits) as knowledge proxy for wAUROC
        p_c = compute_p_correct(logits_final, answers, model.tokenizer)
        p_correct_values.append(p_c)
        generated_texts.append(gen_text.strip())

        # Dot-product confidence (baseline, same as before)
        confs_dot = get_confidence_dot(hidden_states, W_U, b_U, target_id)
        for layer_idx, c in enumerate(confs_dot):
            bucket = "correct" if is_correct else "incorrect"
            dot_confidences[layer_idx][bucket].append(c)

        # Store hidden states for calibration
        for layer_idx, h in enumerate(hidden_states):
            per_layer_h[layer_idx].append(h.cpu())
            per_layer_targets[layer_idx].append(target_id)
            per_layer_labels[layer_idx].append(int(is_correct))

    print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
    print(f"Class balance: {correct_count / n_samples * 100:.1f}% correct")

    # --- Phase 2: Calibrate per-layer temperatures (split-half) ---
    print("\nPhase 2: Calibrating per-layer temperatures (split-half)...")
    n_cal = n_samples // 2
    n_eval = n_samples - n_cal

    # Build calibration set from first half
    per_layer_h_cal = [layer_list[:n_cal] for layer_list in per_layer_h]
    per_layer_targets_cal = [layer_list[:n_cal] for layer_list in per_layer_targets]
    per_layer_labels_cal = [layer_list[:n_cal] for layer_list in per_layer_labels]

    temperatures = calibrate_temperatures(
        per_layer_h_cal, W_U, per_layer_targets_cal, per_layer_labels_cal, n_steps=50
    )
    print("Per-layer calibrated temperatures:")
    for i, t in enumerate(temperatures):
        print(f"  Layer {i:>2}: T = {t:.6f}")

    # --- Phase 3: Evaluate on held-out half ---
    print(f"\nPhase 3: Evaluating on held-out half ({n_eval} samples)...")
    dot_confidences_eval = [
        {"correct": [], "incorrect": []} for _ in range(n_total_layers)
    ]
    cos_confidences_eval = [
        {"correct": [], "incorrect": []} for _ in range(n_total_layers)
    ]

    correct_eval = 0
    per_sample_eval = []  # per-sample data for offline wAUROC / logit lens analysis

    for sample_idx in tqdm(range(n_cal, n_samples), desc="Eval"):
        sample = samples[sample_idx]
        is_correct = per_layer_labels[0][sample_idx]
        target_id = per_layer_targets[0][sample_idx]
        bucket = "correct" if is_correct else "incorrect"
        if is_correct:
            correct_eval += 1

        # Reconstruct hidden states for this sample
        h_list = [
            per_layer_h[li][sample_idx].to(device) for li in range(n_total_layers)
        ]

        # Dot-product (baseline) on eval set
        confs_dot = get_confidence_dot(h_list, W_U, b_U, target_id)
        for layer_idx, c in enumerate(confs_dot):
            dot_confidences_eval[layer_idx][bucket].append(c)

        # Cosine + calibrated on eval set
        confs_cos = get_confidence_cosine(
            h_list, W_U, target_id, temperatures=temperatures
        )
        for layer_idx, c in enumerate(confs_cos):
            cos_confidences_eval[layer_idx][bucket].append(c)

        # Collect per-sample data for offline analysis (wAUROC, logit lens)
        per_sample_eval.append(
            {
                "question": sample["question"],
                "answers": sample["answers"],
                "context": sample.get("context", ""),
                "generated_text": generated_texts[sample_idx],
                "is_correct": bool(is_correct),
                "p_correct": p_correct_values[sample_idx],
                "dot_confidences": confs_dot,
                "cos_confidences": confs_cos,
            }
        )

    print(f"Eval set: {correct_eval} correct, {n_eval - correct_eval} incorrect")

    # --- Results: dot-product (baseline) ---
    dot_results = compute_q_results(dot_confidences_eval, n_total_layers)
    print_results_table(dot_results, title="Dot-Product Confidence (Baseline)")

    # --- Results: cosine + calibrated ---
    cos_results = compute_q_results(cos_confidences_eval, n_total_layers)
    print_results_table(cos_results, title="Cosine + Calibrated Confidence")

    # --- Save results ---
    output = {
        "n_samples": n_samples,
        "n_cal": n_cal,
        "n_eval": n_eval,
        "n_correct": correct_count,
        "n_incorrect": incorrect_count,
        "eval_correct": correct_eval,
        "eval_incorrect": n_eval - correct_eval,
        "temperatures": temperatures,
        "dot_product": dot_results,
        "cosine_calibrated": cos_results,
    }
    results_file = output_path / "q_curve_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save per-sample data for offline analysis (wAUROC, logit lens)
    per_sample_file = output_path / "per_sample.json"
    with open(per_sample_file, "w") as f:
        json.dump(per_sample_eval, f, indent=2, ensure_ascii=False)
    print(f"Per-sample data saved to {per_sample_file}")

    # --- Visualize (cosine + calibrated as primary) ---
    plot_q_curve(
        cos_results,
        output_path / "q_curve.png",
        model_name=model_id,
        dataset_name=dataset.upper(),
    )
    plot_distribution_overlap(
        cos_confidences_eval, output_path / "distribution_overlap.png"
    )
    plot_estimator_consistency(cos_results, output_path / "estimator_consistency.png")
    print("Plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
        choices=["triviaqa", "squad", "hellaswag"],
    )
    args = parser.parse_args()
    main(args.n_samples, args.device, args.output_dir, args.dataset, args.model)
