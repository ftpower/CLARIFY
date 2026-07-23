"""P0-2: InternalInspector Attn/FFN Ratio on TriviaQA.

Computes r = ||attn_output||2 / ||ffn_output||2 for each layer at the last
generated token position. TriviaQA (factual QA) should be FFN-dominated:
low ratio = FFN dominates = knowledge retrieval = less hallucination.

InternalInspector (EMNLP 2024): FFN-only ACC 70.3% on LLaMA-2-7B TriviaQA.
Hidden-state-based, completely vocabulary-independent.

Usage:
    python main.py --n_samples 200
    python main.py --n_samples 10   # quick test
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
sys.path.insert(0, str(Path(__file__).parent.parent))  # for shared

from shared import load_model_and_data, format_prompt
from phase4_utils.generalization_features import (
    compute_attn_ffn_ratio_batch,
)


def extract_attn_ffn_at_last_token(model, prompt, device):
    """Extract attn/ffn sub-layer outputs at the last generated token.

    Runs greedy generation (max_new_tokens tokens), hooks attn_out and
    mlp_post at each layer for the FINAL token position.

    Returns:
        attn_outputs: list of [d_model] tensors, length n_layers
        ffn_outputs: list of [d_model] tensors, length n_layers
        generated_text: str (decoded answer)
    """
    n_layers = model.cfg.n_layers
    max_new_tokens = 10
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    # Greedy decode loop
    for step in range(max_new_tokens):
        # Hook attn_out and mlp_post at all layers
        attn_storage = {}
        ffn_storage = {}

        def _make_attn_hook(name):
            def hook(act, hook):
                attn_storage[name] = act[:, -1, :].detach()

            return hook

        def _make_ffn_hook(name):
            def hook(act, hook):
                ffn_storage[name] = act[:, -1, :].detach()

            return hook

        fwd_hooks = []
        for i in range(n_layers):
            fwd_hooks.append(
                (f"blocks.{i}.hook_attn_out", _make_attn_hook(f"L{i}"))
            )
            fwd_hooks.append(
                (f"blocks.{i}.mlp.hook_post", _make_ffn_hook(f"L{i}"))
            )

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        next_id = logits[0, -1, :].argmax(dim=-1).item()
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=device)], dim=-1
        )

        if next_id == model.tokenizer.eos_token_id:
            break

    # Extract attn/ffn outputs at the LAST token
    attn_outputs = [attn_storage[f"L{i}"].squeeze(0).cpu() for i in range(n_layers)]
    ffn_outputs = [ffn_storage[f"L{i}"].squeeze(0).cpu() for i in range(n_layers)]

    return attn_outputs, ffn_outputs


def main(
    n_samples: int = 200,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
    output_dir: str = "outputs",
    seed: int = 42,
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
        p5_labels = [s["is_correct"] for s in p5_data["per_sample"]]
    else:
        p5_labels = None

    # ── Extract attn/ffn per sample ────────────────────────────────────
    print(f"\nExtracting Attn/FFN outputs at last generated token...")
    n_layers = model.cfg.n_layers
    all_attn = {li: [] for li in range(n_layers)}
    all_ffn = {li: [] for li in range(n_layers)}
    labels = []

    for idx, sample in enumerate(tqdm(samples, desc="Attn/FFN")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        try:
            attn_outputs, ffn_outputs = extract_attn_ffn_at_last_token(
                model, prompt, device
            )
            for li in range(n_layers):
                all_attn[li].append(attn_outputs[li])
                all_ffn[li].append(ffn_outputs[li])
            if p5_labels and idx < len(p5_labels):
                labels.append(p5_labels[idx])
            else:
                labels.append(sample.get("is_correct", False))
        except Exception as e:
            print(f"  Sample {idx} failed: {e}")
            continue

        if (idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    labels = np.array(labels, dtype=np.int32)
    n_valid = len(labels)
    print(f"  Valid samples: {n_valid}")

    # ── Compute per-layer ratio + AUROC ────────────────────────────────
    print(f"\n{'Layer':<6} {'Ratio mean':<12} {'Ratio std':<12} {'AUROC':<10}")
    print("-" * 42)

    best_auroc = 0.5
    best_layer = -1
    all_results = {}

    for li in range(n_layers):
        if len(all_attn[li]) < 2:
            continue

        ratios = compute_attn_ffn_ratio_batch(all_attn[li], all_ffn[li])
        ratio_mean = ratios.mean()
        ratio_std = ratios.std()

        try:
            # TriviaQA = factual → FFN-dominated → high ratio? Actually
            # r = ||attn|| / ||ffn||. Lower r = more FFN-dominated = better
            # So higher r (attention-dominated) = potential hallucination
            auroc = roc_auc_score(1 - labels, ratios)
        except ValueError:
            auroc = 0.5

        print(f"  L{li:<4} {ratio_mean:<12.4f} {ratio_std:<12.4f} {auroc:<10.4f}")

        all_results[f"L{li}"] = {
            "ratio_mean": float(ratio_mean),
            "ratio_std": float(ratio_std),
            "auroc": float(auroc),
        }

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = li

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n  Best: L{best_layer} AUROC = {best_auroc:.4f}")

    if labels.sum() > 0 and (1 - labels).sum() > 0:
        # Check FFN-vs-Attn dominance on correct vs incorrect
        best_ratios = compute_attn_ffn_ratio_batch(
            all_attn[best_layer], all_ffn[best_layer]
        )
        corr_r = best_ratios[labels == 1].mean() if labels.sum() > 0 else 0
        incorr_r = (
            best_ratios[labels == 0].mean() if (1 - labels).sum() > 0 else 0
        )
        print(
            f"  Best layer ratio: correct={corr_r:.4f}, incorrect={incorr_r:.4f}"
        )

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples,
            "n_valid": n_valid,
            "model_id": model_id,
            "n_layers": n_layers,
            "seed": seed,
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
    parser = argparse.ArgumentParser(
        description="P0-2: Attn/FFN ratio on TriviaQA"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        device=args.device,
        model_id=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
    )
