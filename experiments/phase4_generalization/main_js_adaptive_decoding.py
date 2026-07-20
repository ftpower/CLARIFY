"""P0 Part B2: Adaptive JS Divergence Decoding.

Generalizes Self-Correcting Decoding from multimodal (T2I-dependent) to fully
self-contained layer-pair decoding. Uses JS divergence between L17 (mid) and
L26 (late) logit-lens distributions as an adaptive switch:

  JS < τ → Complementary: final = logits_L26 + α1 * logits_L17
  JS ≥ τ → Contrastive:  final = (1+α2) * logits_L26 − α2 * logits_L17

Grid search over τ, α1, α2 against greedy baseline. The key innovation is
replacing the external T2I model with layer-pair disagreement as the confidence
signal.

Usage:
    python main_js_adaptive_decoding.py --n_samples 200 --device cuda
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "phase2_entropy"))
from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct


def compute_4choice_softmax(
    logits: torch.Tensor,
    letter_ids: list[int],
) -> torch.Tensor:
    """Compute 4-choice softmax from full-vocab logits."""
    choice_logits = logits[letter_ids]
    return torch.softmax(choice_logits.float(), dim=-1)


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JS divergence between two probability vectors. Returns scalar."""
    eps = 1e-10
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_p = torch.sum(p * torch.log(p / m))
    kl_q = torch.sum(q * torch.log(q / m))
    return 0.5 * (kl_p + kl_q)


def decode_with_js_adaptive(
    model,
    prompt: str,
    letter_ids: dict[str, int],
    early_layer: int = 17,
    late_layer: int = 26,
    tau: float = 0.1,
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    max_new_tokens: int = 5,
) -> str:
    """Generate answer with layer-pair adaptive decoding.

    At each decode step:
      1. Extract logit-lens softmax at early_layer and late_layer
      2. Compute JS divergence between the two
      3. JS < τ: complementary (add) → confidence high, reinforce
      4. JS ≥ τ: contrastive (subtract) → disagreement, sharpen

    Args:
        model: HookedTransformer.
        prompt: Input text.
        letter_ids: Dict mapping A/B/C/D to token IDs.
        early_layer: Index of the early (mid) layer.
        late_layer: Index of the late (deep) layer.
        tau: JS threshold for switching between complementary/contrastive.
        alpha1: Weight for complementary logits (add).
        alpha2: Weight for contrastive logits (subtract).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated answer text (prompt excluded).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    prompt_len = tokens.shape[1]

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Storage for hooks
    storage = {}

    def _make_hook(key):
        def hook(act, hook=None):
            storage[key] = act.detach()
            return act
        return hook

    fwd_hooks = [
        (f"blocks.{early_layer}.hook_resid_post", _make_hook("early")),
        (f"blocks.{late_layer}.hook_resid_post", _make_hook("late")),
    ]

    for _step in range(max_new_tokens):
        storage.clear()

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Extract layer-specific logit lens
        h_early = storage["early"][0, -1, :]  # [d_model]
        h_late = storage["late"][0, -1, :]  # [d_model]

        logits_early = h_early @ W_U
        logits_late = h_late @ W_U
        if b_U is not None:
            logits_early = logits_early + b_U.to(h_early.device)
            logits_late = logits_late + b_U.to(h_late.device)

        # 4-choice softmax at each layer
        p_early = compute_4choice_softmax(logits_early, letter_tok_ids)
        p_late = compute_4choice_softmax(logits_late, letter_tok_ids)

        # JS divergence
        js = compute_js_divergence(p_early, p_late)

        # Adaptive decoding
        logits_final = logits[0, -1, :]  # [vocab_size]
        choice_logits_final = logits_final[letter_tok_ids]  # [4]

        if js.item() < tau:
            # Complementary: layers agree → reinforce confidence
            choice_logits_early = logits_early[letter_tok_ids]
            adjusted = choice_logits_final + alpha1 * choice_logits_early
        else:
            # Contrastive: layers disagree → sharpen by subtraction
            choice_logits_early = logits_early[letter_tok_ids]
            adjusted = (
                (1.0 + alpha2) * choice_logits_final
                - alpha2 * choice_logits_early
            )

        # Pick argmax among 4 choices for HellaSwag
        # For generation: use the letter token corresponding to argmax
        choice_idx = adjusted.argmax().item()
        next_id = torch.tensor(
            [[letter_tok_ids[choice_idx]]],
            device=tokens.device,
        )

        tokens = torch.cat([tokens, next_id], dim=-1)

        # Stop if EOS or we've generated an answer letter
        if next_id.item() == model.tokenizer.eos_token_id:
            break

    new_ids = tokens[0, prompt_len:]
    answer = model.tokenizer.decode(new_ids).strip()
    return answer


def evaluate_config(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
    early_layer: int,
    late_layer: int,
    tau: float,
    alpha1: float,
    alpha2: float,
    max_new_tokens: int = 5,
) -> dict:
    """Evaluate a single (τ, α1, α2) config on a set of samples."""
    n_correct = 0
    n_total = len(samples)
    per_sample = []

    for sample in tqdm(samples, desc=f"τ={tau:.2f} α1={alpha1} α2={alpha2}", leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        answer = decode_with_js_adaptive(
            model,
            prompt,
            letter_ids,
            early_layer=early_layer,
            late_layer=late_layer,
            tau=tau,
            alpha1=alpha1,
            alpha2=alpha2,
            max_new_tokens=max_new_tokens,
        )

        is_correct = check_correct(answer, sample["answers"], dataset="hellaswag")
        n_correct += int(is_correct)
        per_sample.append(
            {"pred": answer, "correct": correct_letter, "is_correct": is_correct}
        )

    acc = n_correct / n_total
    return {
        "tau": tau,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "accuracy": float(acc),
        "n_correct": n_correct,
        "n_total": n_total,
        "per_sample": per_sample,
    }


def evaluate_greedy_baseline(
    model,
    samples: list[dict],
    max_new_tokens: int = 5,
) -> dict:
    """Standard greedy decoding baseline."""
    n_correct = 0
    n_total = len(samples)

    for sample in tqdm(samples, desc="Greedy baseline", leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )

        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        prompt_len = tokens.shape[1]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(tokens)
                next_id = logits[0, -1, :].argmax(dim=-1)
                tokens = torch.cat(
                    [tokens, next_id.unsqueeze(0).unsqueeze(0)], dim=-1
                )
                if next_id.item() == model.tokenizer.eos_token_id:
                    break

        new_ids = tokens[0, prompt_len:]
        answer = model.tokenizer.decode(new_ids).strip()
        is_correct = check_correct(answer, sample["answers"], dataset="hellaswag")
        n_correct += int(is_correct)

    acc = n_correct / n_total
    return {"accuracy": float(acc), "n_correct": n_correct, "n_total": n_total}


def main():
    parser = argparse.ArgumentParser(
        description="P0 Part B2: Adaptive JS Decoding"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_layer", type=int, default=17)
    parser.add_argument("--late_layer", type=int, default=26)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    # Encode letter token IDs
    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
    print(f"Letter token IDs: {letter_ids}")

    # ── Load data ──
    print(f"\nLoading HellaSwag ({args.n_samples} samples)...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── Greedy baseline ──
    print(f"\n{'=' * 60}")
    print("Greedy Baseline")
    print(f"{'=' * 60}")
    baseline = evaluate_greedy_baseline(
        model, samples, max_new_tokens=args.max_new_tokens
    )
    baseline_acc = baseline["accuracy"]
    print(f"Greedy accuracy: {baseline_acc:.4f} ({baseline['n_correct']}/{baseline['n_total']})")

    # ── Grid search ──
    print(f"\n{'=' * 60}")
    print("Grid Search: Adaptive JS Decoding")
    print(f"{'=' * 60}")

    tau_values = [0.1, 0.2]
    alpha1_values = [0.5, 1.0]
    alpha2_values = [0.5, 1.0]

    all_results = []
    best_config = None
    best_delta = -float("inf")

    total_configs = len(tau_values) * len(alpha1_values) * len(alpha2_values)
    print(f"Grid: {len(tau_values)} τ × {len(alpha1_values)} α1 × "
          f"{len(alpha2_values)} α2 = {total_configs} configs")

    for tau in tau_values:
        for alpha1 in alpha1_values:
            for alpha2 in alpha2_values:
                result = evaluate_config(
                    model,
                    samples,
                    letter_ids,
                    early_layer=args.early_layer,
                    late_layer=args.late_layer,
                    tau=tau,
                    alpha1=alpha1,
                    alpha2=alpha2,
                    max_new_tokens=args.max_new_tokens,
                )
                delta = result["accuracy"] - baseline_acc
                result["delta"] = float(delta)
                all_results.append(result)

                status = "✓" if delta > 0 else "✗"
                print(
                    f"  τ={tau:.2f} α1={alpha1} α2={alpha2}: "
                    f"acc={result['accuracy']:.4f} Δ={delta:+.4f} {status}"
                )

                if delta > best_delta:
                    best_delta = delta
                    best_config = result

    # ── Best config ──
    print(f"\n{'=' * 60}")
    print("Best Configuration")
    print(f"{'=' * 60}")
    if best_config:
        print(
            f"τ={best_config['tau']:.2f}, α1={best_config['alpha1']}, "
            f"α2={best_config['alpha2']}"
        )
        print(f"Accuracy: {best_config['accuracy']:.4f} "
              f"({best_config['n_correct']}/{best_config['n_total']})")
        print(f"Δ over greedy: {best_config['delta']:+.4f}")

        # Success/failure analysis
        if best_config["delta"] <= 0.01:
            print("\nWARNING: No config improves over greedy by >1pp.")
            print("JS-guided adaptive decoding may not help at this model scale.")
            print("This is consistent with the DoLa failure on 1.7B — ")
            print("layer-wise differentiation may be too weak for effective contrast.")

    # ── Compare with pure strategies ──
    print(f"\n{'=' * 60}")
    print("Ablation: Pure Strategies")
    print(f"{'=' * 60}")

    # Pure complementary (always add, tau=∞)
    pure_comp = evaluate_config(
        model, samples, letter_ids,
        early_layer=args.early_layer, late_layer=args.late_layer,
        tau=float("inf"), alpha1=1.0, alpha2=0.0,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Pure complementary (τ=∞, α1=1): "
          f"acc={pure_comp['accuracy']:.4f} Δ={pure_comp['accuracy'] - baseline_acc:+.4f}")

    # Pure contrastive (always subtract, tau=0)
    pure_cont = evaluate_config(
        model, samples, letter_ids,
        early_layer=args.early_layer, late_layer=args.late_layer,
        tau=0.0, alpha1=0.0, alpha2=1.0,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Pure contrastive (τ=0, α2=1):  "
          f"acc={pure_cont['accuracy']:.4f} Δ={pure_cont['accuracy'] - baseline_acc:+.4f}")

    # ── Save results ──
    output = {
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "early_layer": args.early_layer,
            "late_layer": args.late_layer,
            "max_new_tokens": args.max_new_tokens,
            "tau_values": tau_values,
            "alpha1_values": alpha1_values,
            "alpha2_values": alpha2_values,
            "seed": args.seed,
        },
        "baseline": baseline,
        "best_config": {
            "tau": best_config["tau"],
            "alpha1": best_config["alpha1"],
            "alpha2": best_config["alpha2"],
            "accuracy": best_config["accuracy"],
            "delta": best_config["delta"],
        } if best_config else None,
        "ablation": {
            "pure_complementary": {
                "accuracy": pure_comp["accuracy"],
                "delta": pure_comp["accuracy"] - baseline_acc,
            },
            "pure_contrastive": {
                "accuracy": pure_cont["accuracy"],
                "delta": pure_cont["accuracy"] - baseline_acc,
            },
        },
        "full_sweep": [
            {
                "tau": r["tau"],
                "alpha1": r["alpha1"],
                "alpha2": r["alpha2"],
                "accuracy": r["accuracy"],
                "delta": r["delta"],
            }
            for r in all_results
        ],
    }

    out_path = output_dir / "js_adaptive_decoding_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
