"""P1-1: HaloScope zeta on TriviaQA.

Computes population-level outlier degree via cross-sample SVD at each layer.
zeta_i = (1/k) * sum_{j=1..k} sigma_j * <f_i, v_j>^2

High zeta = sample is an outlier = higher hallucination risk.

HaloScope (Bohdal et al., NeurIPS 2024): AUROC 77.4% on LLaMA-2-7B TriviaQA.
Hidden-state-based, completely vocabulary-independent.

WARNING: HaloScope assumes pi < 0.5. TriviaQA 1.7B accuracy is 46.5% (pi ~ 0.535).
Expected some degradation.

Usage:
    python main.py --n_samples 200
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Path setup
_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent / "phase4_generalization"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import load_model_and_data, format_prompt
from phase4_utils.generalization_features import compute_haloscope_zeta_batch


def extract_all_hidden_states(model, prompt, device, n_layers):
    """Extract hidden states at ALL layers for the last input token.

    Single forward pass — hooks all layers simultaneously.
    HaloScope operates on input representations, not generated tokens.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    storage = {}
    def _make_hook(name):
        def hook(act, hook):
            storage[name] = act[:, -1, :].detach()
        return hook

    fwd_hooks = [(f"blocks.{i}.hook_resid_post", _make_hook(f"L{i}")) for i in range(n_layers)]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    return [storage[f"L{i}"].squeeze(0).cpu() for i in range(n_layers)]  # list of [d_model]


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    output_dir: str = "outputs",
    seed: int = 42,
    k: int = 5,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model + data ──────────────────────────────────────────────
    model, samples, _, _ = load_model_and_data(
        n_samples=n_samples, seed=seed, device=device, model_id=model_id
    )

    # ── Load labels from Phase 5.1 ─────────────────────────────────────
    phase5_json = (
        Path(__file__).parent.parent.parent
        / "phase5_cross_task"
        / "outputs"
        / "triviaqa_features.json"
    )
    if phase5_json.exists():
        with open(phase5_json) as f:
            p5_data = json.load(f)
        labels = np.array(
            [s["is_correct"] for s in p5_data["per_sample"]], dtype=np.int32
        )
    else:
        labels = None

    n_layers = model.cfg.n_layers

    # ── Extract hidden states at all layers ────────────────────────────
    print(f"\nExtracting hidden states for {n_layers} layers...")
    all_hidden = {li: [] for li in range(n_layers)}
    valid_labels = []

    for idx, sample in enumerate(tqdm(samples, desc="Hidden states")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        try:
            hs_list = extract_all_hidden_states(
                model, prompt, device, n_layers
            )
            for li in range(n_layers):
                all_hidden[li].append(hs_list[li])
            if labels is not None and idx < len(labels):
                valid_labels.append(labels[idx])
        except Exception as e:
            print(f"  Sample {idx} failed: {e}")
            continue

    valid_labels = np.array(valid_labels, dtype=np.int32)
    n_valid = len(valid_labels)
    print(f"  Valid: {n_valid}")

    # ── HaloScope zeta per layer ──────────────────────────────────────
    print(f"\n{'Layer':<6} {'zeta mean':<12} {'zeta std':<12} {'AUROC':<10}")
    print("-" * 42)

    best_auroc = 0.5
    best_layer = -1
    all_results = {}

    for li in range(n_layers):
        if len(all_hidden[li]) < k + 2:
            continue

        # Stack hidden states
        h_matrix = torch.stack(all_hidden[li], dim=0).numpy()  # [N, d_model]
        zeta = compute_haloscope_zeta_batch(h_matrix, k=k)

        try:
            auroc = roc_auc_score(1 - valid_labels, zeta)
        except ValueError:
            auroc = 0.5

        print(f"  L{li:<4} {zeta.mean():<12.4f} {zeta.std():<12.4f} {auroc:<10.4f}")

        all_results[f"L{li}"] = {
            "zeta_mean": float(zeta.mean()),
            "zeta_std": float(zeta.std()),
            "auroc": float(auroc),
        }

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = li

    print(f"\n  Best: L{best_layer} AUROC = {best_auroc:.4f}")

    # Check pi violation
    pi = 1 - valid_labels.sum() / len(valid_labels)
    print(f"  Hallucination rate pi = {pi:.3f} ({'OK < 0.5' if pi < 0.5 else 'VIOLATES < 0.5'})")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples,
            "n_valid": n_valid,
            "model_id": model_id,
            "k": k,
            "seed": seed,
            "pi_hallucination_rate": float(pi),
            "pi_warning": bool(pi >= 0.5),
        },
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "per_layer": all_results,
    }

    output_file = output_path / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1-1: HaloScope zeta on TriviaQA")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(
        n_samples=args.n_samples, device=args.device,
        model_id=args.model, output_dir=args.output_dir,
        seed=args.seed, k=args.k,
    )
