"""Direction 2: Adaptive Noise Intervention — max_p triggered, multi-strategy.

Usage:
    # Single strategy, specific config
    python main_adaptive_intervention.py --strategy A --threshold 0.5 --sigma_base 0.1

    # Full sweep across all strategies and hyperparameters
    python main_adaptive_intervention.py --sweep --n_samples 200

    # Phase 5: fixed-sigma full-layer scan
    python main_adaptive_intervention.py --phase5 --n_samples 200
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.hidden_state import extract_hidden_states, generate_answer
from src.adaptive_noise import (
    detect_per_layer_max_p,
    generate_with_noise_hook,
    generate_with_per_layer_noise,
    generate_with_hooks,
    compute_adaptive_sigma,
    run_strategy_A,
    run_strategy_B,
    run_strategy_C,
    run_fixed_sigma,
)


def build_hyperparameter_grid():
    """Return list of (threshold, sigma_base, alpha) combos."""
    thresholds = [0.3, 0.5, 0.7, 0.9]
    sigma_bases = [0.01, 0.05, 0.1, 0.5]
    alphas = [0.5, 1.0, 2.0]
    grid = []
    for t in thresholds:
        for s in sigma_bases:
            for a in alphas:
                grid.append((t, s, a))
    return grid


def collect_baseline(model, samples, dataset, W_U, b_U, n_layers, device):
    """Run baseline: extract per-layer max_p and record generation results."""
    results = []
    correct_count = 0

    for sample in tqdm(samples, desc="Baseline"):
        prompt = format_prompt(sample["question"], sample["context"], dataset=dataset)
        answers = sample["answers"]

        # Extract per-layer hidden states + single-token output
        hidden_states, _, gen_id, gen_text_single = extract_hidden_states(model, prompt)

        # Compute per-layer max_p
        max_probs = []
        for h in hidden_states:
            h_dev = h.to(device)
            logits = h_dev @ W_U
            if b_U is not None:
                logits = logits + b_U
            probs = torch.softmax(logits, dim=-1)
            max_probs.append(probs.max().item())

        # Multi-token answer for QA; single token enough for HellaSwag
        if dataset in ("triviaqa", "squad"):
            gen_text = generate_answer(model, prompt, max_new_tokens=20)
        else:
            gen_text = gen_text_single.strip()

        is_correct = check_correct(gen_text, answers, dataset=dataset)
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": sample["question"],
                "context": sample["context"],
                "answers": answers,
                "baseline_text": gen_text,
                "baseline_correct": is_correct,
                "max_p_per_layer": max_probs,
            }
        )

    accuracy = correct_count / len(samples)
    print(f"Baseline accuracy: {accuracy:.4f} ({correct_count}/{len(samples)})")
    return results, accuracy


def sweep_strategies(
    model,
    samples,
    dataset,
    baseline_results,
    W_U,
    b_U,
    n_layers,
    strategy_names=("A", "B", "C"),
):
    """Run hyperparameter sweep across strategies A/B/C.

    For efficiency, reuses per-sample max_p from baseline_results.
    """
    grid = build_hyperparameter_grid()
    all_sweep_results = {}

    for strat_name in strategy_names:
        print(f"\n{'=' * 60}")
        print(f"Strategy {strat_name}")
        print(f"{'=' * 60}")

        strategy_results = []
        total_combos = len(grid)

        for ci, (threshold, sigma_base, alpha) in enumerate(grid):
            correct = 0
            n_triggered = 0
            n_correct_when_triggered = 0
            triggered_sigmas = []

            for br in baseline_results:
                max_p_detect = br["max_p_per_layer"][-1]  # L28 for A/B

                if max_p_detect > threshold:
                    n_triggered += 1
                    sigma = compute_adaptive_sigma(
                        max_p_detect, threshold, sigma_base, alpha
                    )

                    prompt = format_prompt(
                        br["question"], br["context"], dataset=dataset
                    )

                    if strat_name == "A":
                        gen_text = generate_with_noise_hook(
                            model,
                            prompt,
                            inject_idx=n_layers,
                            sigma=sigma,
                            n_layers=n_layers,
                            max_new_tokens=1,
                        )
                    elif strat_name == "B":
                        gen_text = generate_with_noise_hook(
                            model,
                            prompt,
                            inject_idx=11,
                            sigma=sigma,
                            n_layers=n_layers,
                            max_new_tokens=1,
                        )
                    elif strat_name == "C":
                        # Per-layer: compute sigma for each layer exceeding threshold
                        per_layer_sigmas = {}
                        for idx, mp in enumerate(br["max_p_per_layer"]):
                            if mp > threshold:
                                s = compute_adaptive_sigma(
                                    mp, threshold, sigma_base, alpha
                                )
                                if s > 1e-8:
                                    per_layer_sigmas[idx] = s
                        gen_text = generate_with_per_layer_noise(
                            model,
                            prompt,
                            per_layer_sigmas,
                            n_layers,
                            max_new_tokens=1,
                        )

                    is_correct = check_correct(gen_text, br["answers"], dataset=dataset)
                    if is_correct:
                        n_correct_when_triggered += 1
                    triggered_sigmas.append(sigma)
                    correct += is_correct
                else:
                    # Not triggered — use baseline result
                    correct += br["baseline_correct"]

            accuracy = correct / len(baseline_results)
            triggered_acc = (
                n_correct_when_triggered / n_triggered
                if n_triggered > 0
                else float("nan")
            )
            mean_sigma = float(np.mean(triggered_sigmas)) if triggered_sigmas else 0.0

            strategy_results.append(
                {
                    "threshold": threshold,
                    "sigma_base": sigma_base,
                    "alpha": alpha,
                    "accuracy": accuracy,
                    "n_triggered": n_triggered,
                    "triggered_accuracy": triggered_acc,
                    "mean_sigma": mean_sigma,
                    "n_total": len(baseline_results),
                }
            )

            if (ci + 1) % 12 == 0:
                print(
                    f"  [{ci + 1}/{total_combos}] thr={threshold} σb={sigma_base} α={alpha}: "
                    f"acc={accuracy:.4f} trig={n_triggered}/{len(baseline_results)}"
                )

        all_sweep_results[strat_name] = strategy_results

        # Find best config
        best = max(strategy_results, key=lambda r: r["accuracy"])
        print(
            f"  Best: thr={best['threshold']} σb={best['sigma_base']} α={best['alpha']} "
            f"acc={best['accuracy']:.4f}"
        )

    return all_sweep_results


def sweep_fixed_sigma(
    model,
    samples,
    dataset,
    n_layers,
    baseline_accuracy,
    sparse: bool = True,
):
    """Phase 5: fixed σ noise scan across layers.

    Returns {sigma: {layer: accuracy}} and per-layer dose-response data.
    """
    sigmas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    if sparse:
        # 8 representative layers to keep runtime ~30 min
        layers = [0, 3, 7, 11, 15, 20, 24, n_layers]
    else:
        layers = list(range(n_layers + 1))  # 0..28

    # Heatmap: sigmas × layers
    heatmap = {sigma: {} for sigma in sigmas}

    for sigma in sigmas:
        print(f"\n--- Fixed σ = {sigma} ---")
        for layer_idx in tqdm(layers, desc=f"σ={sigma}"):
            correct = 0
            for sample in samples:
                prompt = format_prompt(
                    sample["question"], sample["context"], dataset=dataset
                )
                result = run_fixed_sigma(
                    model,
                    prompt,
                    sample["answers"],
                    dataset,
                    inject_idx=layer_idx,
                    sigma=sigma,
                    n_layers=n_layers,
                )
                if result["is_correct"]:
                    correct += 1

            accuracy = correct / len(samples)
            heatmap[sigma][layer_idx] = {
                "accuracy": accuracy,
                "delta_accuracy": accuracy - baseline_accuracy,
                "n_correct": correct,
            }
            print(
                f"  L{layer_idx}: acc={accuracy:.4f} (Δ={accuracy - baseline_accuracy:+.4f})"
            )

        gc.collect()
        torch.cuda.empty_cache()

    return heatmap


def plot_results(
    sweep_results, fixed_heatmap, baseline_accuracy, n_layers, output_path
):
    """Generate summary plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Strategy A: accuracy vs threshold (best sigma_base, alpha)
    ax = axes[0, 0]
    if "A" in sweep_results:
        for sa in sorted(set(r["sigma_base"] for r in sweep_results["A"])):
            pts = [
                (r["threshold"], r["accuracy"])
                for r in sweep_results["A"]
                if r["sigma_base"] == sa and r["alpha"] == 1.0
            ]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, "o-", label=f"σb={sa}")
        ax.axhline(y=baseline_accuracy, color="gray", linestyle="--", label="baseline")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")
        ax.set_title("Strategy A: Accuracy vs Threshold (α=1.0)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # (b) Strategy B: same
    ax = axes[0, 1]
    if "B" in sweep_results:
        for sa in sorted(set(r["sigma_base"] for r in sweep_results["B"])):
            pts = [
                (r["threshold"], r["accuracy"])
                for r in sweep_results["B"]
                if r["sigma_base"] == sa and r["alpha"] == 1.0
            ]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, "o-", label=f"σb={sa}")
        ax.axhline(y=baseline_accuracy, color="gray", linestyle="--", label="baseline")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")
        ax.set_title("Strategy B: Accuracy vs Threshold (α=1.0)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # (c) Strategy C: same
    ax = axes[0, 2]
    if "C" in sweep_results:
        for sa in sorted(set(r["sigma_base"] for r in sweep_results["C"])):
            pts = [
                (r["threshold"], r["accuracy"])
                for r in sweep_results["C"]
                if r["sigma_base"] == sa and r["alpha"] == 1.0
            ]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, "o-", label=f"σb={sa}")
        ax.axhline(y=baseline_accuracy, color="gray", linestyle="--", label="baseline")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Accuracy")
        ax.set_title("Strategy C: Accuracy vs Threshold (α=1.0)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # (d) Strategy comparison: best config per strategy
    ax = axes[1, 0]
    strategies = []
    accs = []
    for s_name in ["A", "B", "C"]:
        if s_name in sweep_results:
            best = max(sweep_results[s_name], key=lambda r: r["accuracy"])
            strategies.append(s_name)
            accs.append(best["accuracy"])
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(strategies, accs, color=bar_colors[: len(strategies)])
    ax.axhline(y=baseline_accuracy, color="gray", linestyle="--", label="baseline")
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Accuracy per Strategy")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.4f}",
            ha="center",
            fontsize=10,
        )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (e) Phase 5: heatmap σ × layer → Δaccuracy
    ax = axes[1, 1]
    if fixed_heatmap:
        sigmas = sorted(fixed_heatmap.keys())
        layers_list = sorted(fixed_heatmap[sigmas[0]].keys())
        data = np.zeros((len(sigmas), len(layers_list)))
        for si, sigma in enumerate(sigmas):
            for li, layer in enumerate(layers_list):
                data[si, li] = fixed_heatmap[sigma][layer]["delta_accuracy"]
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="RdBu_r",
            origin="lower",
            vmin=-0.15,
            vmax=0.15,
            extent=[
                min(layers_list) - 0.5,
                max(layers_list) + 0.5,
                -0.5,
                len(sigmas) - 0.5,
            ],
        )
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(s) for s in sigmas])
        ax.set_ylabel("σ")
        ax.set_xlabel("Layer")
        ax.set_title("Fixed σ: ΔAccuracy Heatmap")
        plt.colorbar(im, ax=ax, label="ΔAccuracy")

    # (f) Summary text
    ax = axes[1, 2]
    ax.axis("off")
    lines = [f"Baseline accuracy: {baseline_accuracy:.4f}"]
    for s_name in ["A", "B", "C"]:
        if s_name in sweep_results:
            best = max(sweep_results[s_name], key=lambda r: r["accuracy"])
            lines.append(
                f"Strategy {s_name} best: thr={best['threshold']} "
                f"σb={best['sigma_base']} α={best['alpha']} "
                f"acc={best['accuracy']:.4f}"
            )
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    fig.suptitle("Adaptive Noise Intervention — max_p Triggered", fontsize=14)
    fig.tight_layout()

    plot_file = output_path / "adaptive_intervention.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_file}")


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    dataset: str = "hellaswag",
    output_dir: str = "outputs",
    seed: int = 42,
    strategy: str | None = None,
    threshold: float | None = None,
    sigma_base: float | None = None,
    alpha: float | None = None,
    do_sweep: bool = False,
    do_phase5: bool = False,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)

    # ── Load model ──
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    n_layers = model.cfg.n_layers
    n_total = n_layers + 1  # embed + blocks

    # ── Phase 1: Baseline ──
    print("\n" + "=" * 60)
    print("Phase 1: Baseline")
    print("=" * 60)
    baseline_results, baseline_accuracy = collect_baseline(
        model,
        samples,
        dataset,
        W_U,
        b_U,
        n_layers,
        device,
    )

    # ── Phase 2-4: Strategy sweep ──
    sweep_results = {}

    if do_sweep:
        print("\n" + "=" * 60)
        print("Phase 2-4: Strategy Sweep")
        print("=" * 60)
        sweep_results = sweep_strategies(
            model,
            samples,
            dataset,
            baseline_results,
            W_U,
            b_U,
            n_layers,
        )

    elif strategy is not None and threshold is not None and sigma_base is not None:
        print(
            f"\n--- Single config: Strategy {strategy} "
            f"thr={threshold} σb={sigma_base} α={alpha} ---"
        )
        correct = 0
        n_triggered = 0
        for br in tqdm(baseline_results, desc=f"Strategy {strategy}"):
            prompt = format_prompt(
                br["question"], br.get("context", ""), dataset=dataset
            )
            if strategy == "A":
                result = run_strategy_A(
                    model,
                    prompt,
                    br["answers"],
                    dataset,
                    W_U,
                    b_U,
                    n_layers,
                    threshold,
                    sigma_base,
                    alpha,
                )
            elif strategy == "B":
                result = run_strategy_B(
                    model,
                    prompt,
                    br["answers"],
                    dataset,
                    W_U,
                    b_U,
                    n_layers,
                    threshold,
                    sigma_base,
                    alpha,
                )
            elif strategy == "C":
                result = run_strategy_C(
                    model,
                    prompt,
                    br["answers"],
                    dataset,
                    W_U,
                    b_U,
                    n_layers,
                    threshold,
                    sigma_base,
                    alpha,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if result["triggered"]:
                n_triggered += 1
                correct += result["is_correct"]
            else:
                correct += br["baseline_correct"]

        accuracy = correct / len(baseline_results)
        print(f"Accuracy: {accuracy:.4f} (triggered: {n_triggered}/{n_samples})")
        sweep_results[strategy] = [
            {
                "threshold": threshold,
                "sigma_base": sigma_base,
                "alpha": alpha,
                "accuracy": accuracy,
                "n_triggered": n_triggered,
                "n_total": n_samples,
            }
        ]

    # ── Phase 5: Fixed σ scan ──
    fixed_heatmap = {}
    if do_phase5:
        print("\n" + "=" * 60)
        print("Phase 5: Fixed σ Full-Layer Scan")
        print("=" * 60)
        fixed_heatmap = sweep_fixed_sigma(
            model,
            samples,
            dataset,
            n_layers,
            baseline_accuracy,
        )

    # ── Save results ──
    output = {
        "config": {
            "n_samples": n_samples,
            "model_id": model_id,
            "dataset": dataset,
            "n_layers": n_layers,
            "n_total_layers": n_total,
            "baseline_accuracy": baseline_accuracy,
        },
        "baseline": [
            {k: v for k, v in br.items() if k != "max_p_per_layer"}
            for br in baseline_results
        ],
        "strategy_sweep": sweep_results,
        "fixed_sigma_heatmap": {
            str(sigma): {str(layer): v for layer, v in layers.items()}
            for sigma, layers in fixed_heatmap.items()
        },
    }

    results_file = output_path / "adaptive_intervention_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # ── Plots ──
    try:
        plot_results(
            sweep_results, fixed_heatmap, baseline_accuracy, n_layers, output_path
        )
    except Exception as e:
        print(f"Plot failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default=None, choices=["A", "B", "C"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sigma_base", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--sweep", action="store_true", help="Full hyperparameter sweep"
    )
    parser.add_argument(
        "--phase5", action="store_true", help="Run fixed-σ full-layer scan"
    )
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        model_id=args.model,
        dataset="hellaswag",
        output_dir=args.output_dir,
        seed=args.seed,
        strategy=args.strategy,
        threshold=args.threshold,
        sigma_base=args.sigma_base,
        alpha=args.alpha,
        do_sweep=args.sweep,
        do_phase5=args.phase5,
    )
