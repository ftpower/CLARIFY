"""Hierarchical intervention experiment: noise & scaling at detection-optimal layers.

Usage:
    python main_intervention.py --n_samples 50
    python main_intervention.py --n_samples 100 --target_layer 11 --control_layers 0,14
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

from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.model_utils import (
    load_model,
    get_per_layer_hidden_states,
    get_confidence_dot,
    make_noise_hook,
    make_scale_hook,
    make_store_hook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hidden_state_index_to_hook(idx: int, n_layers: int) -> str:
    """Map hidden_states index to hook point name.

    hidden_states layout (len = n_layers + 1):
      idx 0         = embedding              → blocks.0.hook_resid_pre
      idx 1..n_layers-1 = after block i-1    → blocks.{i-1}.hook_resid_post
      idx n_layers  = ln_final(block_{n_layers-1}.resid_post) → blocks.{n_layers-1}.hook_resid_post
    """
    if idx == 0:
        return "blocks.0.hook_resid_pre"
    if idx <= n_layers - 1:
        return f"blocks.{idx - 1}.hook_resid_post"
    return f"blocks.{n_layers - 1}.hook_resid_post"


def _build_detection_hooks(n_layers: int):
    """Return (hook_points, store_hooks) for collecting per-layer hidden states.

    Matches the layout of get_per_layer_hidden_states().
    """
    stored = {}
    hook_specs = []
    # Embedding
    hook_specs.append(("blocks.0.hook_resid_pre", make_store_hook(stored, 0)))
    # Raw residual after each block
    for i in range(n_layers - 1):
        hook_specs.append((f"blocks.{i}.hook_resid_post", make_store_hook(stored, i + 1)))
    # Last block + ln_final will be applied later
    hook_specs.append(
        (f"blocks.{n_layers - 1}.hook_resid_post", make_store_hook(stored, n_layers))
    )
    return stored, hook_specs


def _extract_confidences(stored: dict, W_U, b_U, target_id: int, n_layers: int, device,
                        ln_final=None):
    """Extract per-layer dot-product confidence from stored hidden states.

    stored layout matches get_per_layer_hidden_states():
      idx 0         = embed (raw)
      idx 1..n_layers-1 = after block i-1 (raw)
      idx n_layers  = after last block (raw) — ln_final applied if provided
    """
    confs = []
    for idx in range(n_layers + 1):
        h_full = stored[idx]               # [1, seq_len, d_model] on CPU
        h_last = h_full[:, -1, :].to(device)  # [1, d_model] on GPU
        if idx == n_layers and ln_final is not None:
            h_last = ln_final(h_last)
        logits = h_last @ W_U               # [1, vocab_size]
        if b_U is not None:
            logits = logits + b_U
        probs = torch.softmax(logits, dim=-1)
        confs.append(probs[0, target_id].item())
    return confs


def run_intervention_forward(
    model, prompt: str, intervention_hooks: list, detection_hook_specs: list,
) -> tuple[int, str]:
    """Single forward pass with intervention + detection hooks. Returns (token_id, token_text)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    last_pos = tokens.shape[1] - 1

    all_hooks = intervention_hooks + detection_hook_specs

    with model.hooks(fwd_hooks=all_hooks):
        with torch.no_grad():
            logits = model(tokens)

    gen_id = logits[0, last_pos, :].argmax(dim=-1).item()
    gen_text = model.tokenizer.decode(gen_id).strip()
    return gen_id, gen_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    n_samples: int = 100,
    device: str = "cuda",
    output_dir: str = "outputs_intervention",
    model_id: str = "Qwen/Qwen3-1.7B",
    target_layer: int | None = None,
    control_layers: list[int] | None = None,
    noise_levels: list[float] | None = None,
    scale_levels: list[float] | None = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading HellaSwag ({n_samples} samples)...")
    samples = load_hellaswag(n_samples=n_samples, seed=seed)
    prompts = [
        format_prompt(s["question"], s["context"], dataset="hellaswag")
        for s in samples
    ]

    # --- Load model ---
    print(f"Loading {model_id}...")
    model = load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    n_layers = model.cfg.n_layers  # number of transformer blocks

    # --- Set up layer ↔ hook mapping ---
    if target_layer is None:
        # Default: wAUROC-optimal layer for HellaSwag
        target_layer = 11 if "1.7B" in model_id else n_layers
    if control_layers is None:
        mid = n_layers // 2
        control_layers = [0, mid]

    target_hook = _hidden_state_index_to_hook(target_layer, n_layers)
    control_hooks = [_hidden_state_index_to_hook(l, n_layers) for l in control_layers]

    print(f"Model: {model_id}, n_layers={n_layers}")
    print(f"Target layer: L{target_layer} → {target_hook}")
    for li, hook in zip(control_layers, control_hooks):
        print(f"Control layer: L{li} → {hook}")

    if noise_levels is None:
        noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    if scale_levels is None:
        scale_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    # =====================================================================
    # Phase 1: Baseline (no intervention)
    # =====================================================================
    print(f"\n{'=' * 60} Phase 1: Baseline {'=' * 60}")

    baseline_correct = []
    baseline_per_layer_conf = {"correct": [[] for _ in range(n_layers + 1)],
                               "incorrect": [[] for _ in range(n_layers + 1)]}

    for i, (sample, prompt) in enumerate(tqdm(
        zip(samples, prompts), total=n_samples, desc="Baseline"
    )):
        answers = sample["answers"]
        stored, det_specs = _build_detection_hooks(n_layers)

        gen_id, gen_text = run_intervention_forward(
            model, prompt, intervention_hooks=[], detection_hook_specs=det_specs,
        )
        # stored dict already populated by detection hooks
        is_correct = check_correct(gen_text, answers, dataset="hellaswag")
        baseline_correct.append(is_correct)

        # Per-layer confidence from stored hidden states
        confs = _extract_confidences(stored, W_U, b_U, gen_id, n_layers, device,
                                      ln_final=model.ln_final)
        bucket = "correct" if is_correct else "incorrect"
        for li, c in enumerate(confs):
            baseline_per_layer_conf[bucket][li].append(c)

    baseline_acc = sum(baseline_correct) / n_samples
    print(f"Baseline accuracy: {baseline_acc:.4f} ({sum(baseline_correct)}/{n_samples})")

    # Compute baseline per-layer AUROC
    from sklearn.metrics import roc_auc_score

    baseline_aurocs = []
    for li in range(n_layers + 1):
        scores = (baseline_per_layer_conf["correct"][li] +
                  baseline_per_layer_conf["incorrect"][li])
        labels = ([1] * len(baseline_per_layer_conf["correct"][li]) +
                  [0] * len(baseline_per_layer_conf["incorrect"][li]))
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")
        baseline_aurocs.append(auc)
        if li == target_layer:
            print(f"  Baseline AUROC @ L{li}: {auc:.4f}")

    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # Phase 2: Noise sweep
    # =====================================================================
    print(f"\n{'=' * 60} Phase 2: Noise sweep {'=' * 60}")

    noise_results = {}  # noise_results[layer_label][sigma_idx] = {accuracy, auroc, per_sample}

    all_layer_hooks = [("target", target_hook)] + [
        (f"L{control_layers[i]}", control_hooks[i]) for i in range(len(control_layers))
    ]

    for label, hook_point in all_layer_hooks:
        print(f"\n--- Noise @ {label} ({hook_point}) ---")
        layer_results = []
        for sigma in noise_levels:
            noise_hook = make_noise_hook(sigma)
            intervention_hooks = [(hook_point, noise_hook)]
            correct_list = []
            per_layer_conf = {"correct": [[] for _ in range(n_layers + 1)],
                              "incorrect": [[] for _ in range(n_layers + 1)]}

            for i, (sample, prompt) in enumerate(
                tqdm(zip(samples, prompts), total=n_samples, desc=f"σ={sigma}")
            ):
                answers = sample["answers"]
                stored, det_specs = _build_detection_hooks(n_layers)

                gen_id, gen_text = run_intervention_forward(
                    model, prompt,
                    intervention_hooks=intervention_hooks,
                    detection_hook_specs=det_specs,
                )
                is_correct = check_correct(gen_text, answers, dataset="hellaswag")
                correct_list.append(is_correct)

                confs = _extract_confidences(stored, W_U, b_U, gen_id, n_layers, device,
                                      ln_final=model.ln_final)
                bucket = "correct" if is_correct else "incorrect"
                for li, c in enumerate(confs):
                    per_layer_conf[bucket][li].append(c)

            acc = sum(correct_list) / n_samples
            delta = baseline_acc - acc

            # AUROC at target layer
            scores = (per_layer_conf["correct"][target_layer] +
                      per_layer_conf["incorrect"][target_layer])
            labels_ = ([1] * len(per_layer_conf["correct"][target_layer]) +
                       [0] * len(per_layer_conf["incorrect"][target_layer]))
            try:
                auroc = roc_auc_score(labels_, scores)
            except ValueError:
                auroc = float("nan")

            layer_results.append({
                "sigma": sigma,
                "accuracy": acc,
                "delta_accuracy": delta,
                "auroc_at_target": auroc,
                "n_correct": sum(correct_list),
                "n_incorrect": n_samples - sum(correct_list),
            })
            print(f"  σ={sigma:.4f}: acc={acc:.4f} (Δ={delta:+.4f}), AUROC={auroc:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        noise_results[label] = layer_results

    # =====================================================================
    # Phase 3: Scale sweep
    # =====================================================================
    print(f"\n{'=' * 60} Phase 3: Scale sweep {'=' * 60}")

    scale_results = {}

    for label, hook_point in all_layer_hooks:
        print(f"\n--- Scale @ {label} ({hook_point}) ---")
        layer_results = []
        for alpha in scale_levels:
            scale_hook = make_scale_hook(alpha)
            intervention_hooks = [(hook_point, scale_hook)]
            correct_list = []

            for i, (sample, prompt) in enumerate(
                tqdm(zip(samples, prompts), total=n_samples, desc=f"α={alpha}")
            ):
                answers = sample["answers"]
                stored, det_specs = _build_detection_hooks(n_layers)

                gen_id, gen_text = run_intervention_forward(
                    model, prompt,
                    intervention_hooks=intervention_hooks,
                    detection_hook_specs=det_specs,
                )
                is_correct = check_correct(gen_text, answers, dataset="hellaswag")
                correct_list.append(is_correct)

            acc = sum(correct_list) / n_samples
            delta = baseline_acc - acc

            layer_results.append({
                "alpha": alpha,
                "accuracy": acc,
                "delta_accuracy": delta,
                "n_correct": sum(correct_list),
                "n_incorrect": n_samples - sum(correct_list),
            })
            print(f"  α={alpha:.2f}: acc={acc:.4f} (Δ={delta:+.4f})")

            gc.collect()
            torch.cuda.empty_cache()

        scale_results[label] = layer_results

    # =====================================================================
    # Phase 4: Save results
    # =====================================================================
    print(f"\n{'=' * 60} Phase 4: Results {'=' * 60}")

    # Summary table
    print(f"\n{'Layer':>12} {'Method':>8} {'Param':>8} {'Acc':>8} {'Δ Acc':>8} {'AUROC':>8}")
    print("-" * 62)
    for label in noise_results:
        r0 = noise_results[label][0]
        print(f"{label:>12} {'noise':>8} {'σ=0':>8} {r0['accuracy']:>8.4f} {'—':>8} {r0['auroc_at_target']:>8.4f}")
        r_last = noise_results[label][-1]
        sigma_str = f"σ={r_last['sigma']:.1f}"
        print(f"{label:>12} {'noise':>8} {sigma_str:>8} {r_last['accuracy']:>8.4f} {r_last['delta_accuracy']:>+8.4f} {r_last['auroc_at_target']:>8.4f}")
    for label in scale_results:
        r_last = scale_results[label][-1]
        alpha_str = f"α={r_last['alpha']:.1f}"
        print(f"{label:>12} {'scale':>8} {alpha_str:>8} {r_last['accuracy']:>8.4f} {r_last['delta_accuracy']:>+8.4f} {'—':>8}")

    # --- Save JSON ---
    output = {
        "config": {
            "n_samples": n_samples,
            "model_id": model_id,
            "n_layers": n_layers,
            "target_layer": target_layer,
            "target_hook": target_hook,
            "control_layers": control_layers,
            "control_hooks": control_hooks,
            "baseline_accuracy": baseline_acc,
            "baseline_aurocs": baseline_aurocs,
            "noise_levels": noise_levels,
            "scale_levels": scale_levels,
        },
        "noise_results": noise_results,
        "scale_results": scale_results,
    }

    results_file = output_path / "intervention_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # --- Dose-response plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Noise dose-response
        for label, results in noise_results.items():
            sigmas = [r["sigma"] for r in results]
            accs = [r["accuracy"] for r in results]
            ax1.plot(sigmas, accs, "o-", label=label)
        ax1.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax1.set_xlabel("Gaussian noise std (σ)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Noise Injection Dose-Response")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scale dose-response
        for label, results in scale_results.items():
            alphas = [r["alpha"] for r in results]
            accs = [r["accuracy"] for r in results]
            ax2.plot(alphas, accs, "o-", label=label)
        ax2.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax2.set_xlabel("Activation scale (α)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Activation Scaling Dose-Response")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{model_id} — HellaSwag Intervention", fontsize=13)
        fig.tight_layout()

        plot_file = output_path / "dose_response.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_intervention")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--control_layers", type=str, default=None)
    parser.add_argument("--noise_levels", type=str, default="0,0.01,0.05,0.1,0.5,1.0,2.0,5.0")
    parser.add_argument("--scale_levels", type=str, default="0,0.1,0.3,0.5,0.7,0.9,1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    control_layers = None
    if args.control_layers is not None:
        control_layers = [int(x) for x in args.control_layers.split(",")]

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    scale_levels = [float(x) for x in args.scale_levels.split(",")]

    main(
        n_samples=args.n_samples,
        device=args.device,
        output_dir=args.output_dir,
        model_id=args.model,
        target_layer=args.target_layer,
        control_layers=control_layers,
        noise_levels=noise_levels,
        scale_levels=scale_levels,
        seed=args.seed,
    )
