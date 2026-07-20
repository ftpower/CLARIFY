"""Phase 4.2 (Plan 2): Enhanced Self-Contained Layer-Pair Adaptive Decoding.

Key improvements over Plan 1's basic JS decoding:
  1. Top-K layer pairs (K=3) — smooths pair-selection noise vs single pair
  2. Explicit L0 exclusion — avoids DoLa's "always picks L0" failure mode
  3. Richer baselines: greedy, DoLa-standard, pure-complementary, pure-contrastive
  4. Per-token JS trajectory logging for diagnostic analysis

The core insight: DoLa always selects L0 as the "premature" layer in 1.7B because
JS divergence monotonically decreases with depth. By excluding L0 and using
top-3 mid-layer pairs, we get genuinely informative layer disagreement.

Usage:
    python main_enhanced_js_decoding.py --n_samples 200 --device cuda
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

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# JS computation utilities
# ═══════════════════════════════════════════════════════════════════════════════


def compute_4choice_softmax(logits, letter_ids):
    choice_logits = logits[letter_ids]
    return torch.softmax(choice_logits.float(), dim=-1)


def compute_js_divergence(p, q):
    eps = 1e-10
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced JS-adaptive decoder
# ═══════════════════════════════════════════════════════════════════════════════


def decode_enhanced_js(
    model,
    prompt: str,
    letter_ids: dict[str, int],
    layer_pairs: list[tuple[int, int]],
    tau: float = 0.1,
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    top_k: int = 3,
    max_new_tokens: int = 5,
) -> tuple[str, list[dict]]:
    """Enhanced adaptive decoding with top-K layer pairs.

    At each decode step:
      1. Extract logit lens at all candidate layer pairs
      2. Compute JS divergence for each pair
      3. Select top-K pairs by JS (excluding any pair with L0)
      4. JS_mean < τ → complementary (add across top-K pairs)
      5. JS_mean ≥ τ → contrastive (subtract across top-K pairs)

    Args:
        model: HookedTransformer.
        prompt: Input text.
        letter_ids: Dict mapping A/B/C/D to token IDs.
        layer_pairs: List of (early, late) candidate pairs.
        tau: JS threshold for complementary/contrastive switch.
        alpha1: Complementary weight.
        alpha2: Contrastive weight.
        top_k: Number of top-JS pairs to use.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        (answer_text, per_step_diagnostics)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    prompt_len = tokens.shape[1]

    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Build hooks for all unique layers in pairs
    unique_layers = set()
    for e, l in layer_pairs:
        unique_layers.add(e)
        unique_layers.add(l)
    unique_layers = sorted(unique_layers)

    storage = {}

    def _make_hook(key):
        def hook(act, hook=None):
            storage[key] = act.detach()
            return act
        return hook

    fwd_hooks = [
        (f"blocks.{li}.hook_resid_post", _make_hook(f"L{li}"))
        for li in unique_layers
    ]

    per_step_diag = []

    for step in range(max_new_tokens):
        storage.clear()

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Compute JS for each candidate pair (exclude L0)
        pair_js = {}
        for early, late in layer_pairs:
            if early == 0:
                continue  # Exclude L0 — always uniform in 1.7B
            if f"L{early}" not in storage or f"L{late}" not in storage:
                continue

            h_early = storage[f"L{early}"][0, -1, :]
            h_late = storage[f"L{late}"][0, -1, :]

            logits_early = h_early @ W_U
            logits_late = h_late @ W_U
            if b_U is not None:
                logits_early = logits_early + b_U.to(h_early.device)
                logits_late = logits_late + b_U.to(h_late.device)

            p_early = compute_4choice_softmax(logits_early, letter_tok_ids)
            p_late = compute_4choice_softmax(logits_late, letter_tok_ids)
            js = compute_js_divergence(p_early, p_late)
            pair_js[(early, late)] = js.item()

        if not pair_js:
            # Fallback: greedy on final logits
            next_id = logits[0, -1, :].argmax(dim=-1)
            tokens = torch.cat(
                [tokens, next_id.unsqueeze(0).unsqueeze(0)], dim=-1
            )
            per_step_diag.append({"step": step, "js_mean": 0.0, "mode": "fallback"})
            if next_id.item() == model.tokenizer.eos_token_id:
                break
            continue

        # Select top-K pairs by JS
        sorted_pairs = sorted(pair_js.items(), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_pairs[: min(top_k, len(sorted_pairs))]
        js_mean = np.mean([js for _, js in top_pairs])

        # Adaptive decoding
        logits_final = logits[0, -1, :]
        choice_final = logits_final[letter_tok_ids]

        if js_mean < tau:
            # Complementary: layers agree → reinforce
            adjusted = choice_final.clone()
            for (early, late), _js in top_pairs:
                h_e = storage[f"L{early}"][0, -1, :]
                logits_e = h_e @ W_U
                if b_U is not None:
                    logits_e = logits_e + b_U.to(h_e.device)
                adjusted = adjusted + alpha1 * logits_e[letter_tok_ids]
            mode = "complementary"
        else:
            # Contrastive: layers disagree → sharpen
            adjusted = (1.0 + alpha2) * choice_final.clone()
            for (early, late), _js in top_pairs:
                h_e = storage[f"L{early}"][0, -1, :]
                logits_e = h_e @ W_U
                if b_U is not None:
                    logits_e = logits_e + b_U.to(h_e.device)
                adjusted = adjusted - alpha2 * logits_e[letter_tok_ids]
            mode = "contrastive"

        choice_idx = adjusted.argmax().item()
        next_id = torch.tensor(
            [[letter_tok_ids[choice_idx]]], device=tokens.device
        )

        per_step_diag.append(
            {
                "step": step,
                "js_mean": float(js_mean),
                "mode": mode,
                "top_pairs": [(int(e), int(l)) for (e, l), _ in top_pairs],
                "pair_js_values": [float(js) for _, js in top_pairs],
            }
        )

        tokens = torch.cat([tokens, next_id], dim=-1)
        if next_id.item() == model.tokenizer.eos_token_id:
            break

    new_ids = tokens[0, prompt_len:]
    answer = model.tokenizer.decode(new_ids).strip()
    return answer, per_step_diag


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_config(
    model, samples, letter_ids, layer_pairs, tau, alpha1, alpha2, top_k,
    max_new_tokens=5,
):
    n_correct = 0
    for sample in tqdm(
        samples, desc=f"τ={tau:.2f} α1={alpha1} α2={alpha2}", leave=False
    ):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        answer, _diag = decode_enhanced_js(
            model, prompt, letter_ids, layer_pairs,
            tau=tau, alpha1=alpha1, alpha2=alpha2, top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        if check_correct(answer, sample["answers"], dataset="hellaswag"):
            n_correct += 1

    acc = n_correct / len(samples)
    return {"accuracy": float(acc), "n_correct": n_correct, "n_total": len(samples)}


def evaluate_greedy_baseline(model, samples, max_new_tokens=5):
    n_correct = 0
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
        if check_correct(answer, sample["answers"], dataset="hellaswag"):
            n_correct += 1
    return {"accuracy": float(n_correct / len(samples)), "n_correct": n_correct,
            "n_total": len(samples)}


def evaluate_dola_baseline(model, samples, letter_ids, max_new_tokens=5):
    """Standard DoLa: contrast all layers vs final, pick JS-max early layer."""
    n_layers = model.cfg.n_layers
    letters = ["A", "B", "C", "D"]
    letter_tok_ids = [letter_ids[l] for l in letters]
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    n_correct = 0
    for sample in tqdm(samples, desc="DoLa baseline", leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        prompt_len = tokens.shape[1]

        storage = {}

        def _hook_factory(key):
            def h(act, hook=None):
                storage[key] = act.detach()
                return act
            return h

        hooks = [
            (f"blocks.{i}.hook_resid_post", _hook_factory(f"L{i}"))
            for i in range(n_layers)
        ]

        for _ in range(max_new_tokens):
            storage.clear()
            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

            # Find premature layer with max JS from mature (L{n_layers-1})
            h_mature = storage[f"L{n_layers - 1}"][0, -1, :]
            logits_m = h_mature @ W_U
            if b_U is not None:
                logits_m = logits_m + b_U.to(h_mature.device)
            p_mature = compute_4choice_softmax(logits_m, letter_tok_ids)

            max_js = -1.0
            best_premature = 0
            for li in range(n_layers - 1):
                h_pre = storage[f"L{li}"][0, -1, :]
                logits_p = h_pre @ W_U
                if b_U is not None:
                    logits_p = logits_p + b_U.to(h_pre.device)
                p_pre = compute_4choice_softmax(logits_p, letter_tok_ids)
                js = compute_js_divergence(p_pre, p_mature).item()
                if js > max_js:
                    max_js = js
                    best_premature = li

            # Contrastive: mature − premature
            h_pre = storage[f"L{best_premature}"][0, -1, :]
            logits_p = h_pre @ W_U
            if b_U is not None:
                logits_p = logits_p + b_U.to(h_pre.device)

            logits_final = logits[0, -1, :]
            adjusted = F.log_softmax(logits_final[letter_tok_ids].float(), dim=-1) - \
                       F.log_softmax(logits_p[letter_tok_ids].float(), dim=-1)

            choice_idx = adjusted.argmax().item()
            next_id = torch.tensor(
                [[letter_tok_ids[choice_idx]]], device=tokens.device
            )
            tokens = torch.cat([tokens, next_id], dim=-1)
            if next_id.item() == model.tokenizer.eos_token_id:
                break

        new_ids = tokens[0, prompt_len:]
        answer = model.tokenizer.decode(new_ids).strip()
        if check_correct(answer, sample["answers"], dataset="hellaswag"):
            n_correct += 1

    return {"accuracy": float(n_correct / len(samples)), "n_correct": n_correct,
            "n_total": len(samples)}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.2: Enhanced JS Layer-Pair Decoding"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
    print(f"Letter token IDs: {letter_ids}")

    n_layers = model.cfg.n_layers
    print(f"Model has {n_layers} layers")

    # Build candidate layer pairs: exclude L0 (always uniform), use mid-deep pairs
    # Candidate early layers: L5-L20 (skip very shallow and very deep)
    # Candidate late layers: early+3 to n_layers-1
    layer_pairs = []
    for early in range(5, min(21, n_layers - 3)):
        for late in range(early + 3, n_layers):
            layer_pairs.append((early, late))
    print(f"Candidate layer pairs (excluding L0): {len(layer_pairs)}")

    print(f"\nLoading HellaSwag (n={args.n_samples})...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── Baselines ──
    print(f"\n{'=' * 60}")
    print("Baselines")
    print(f"{'=' * 60}")

    greedy = evaluate_greedy_baseline(model, samples, args.max_new_tokens)
    print(f"Greedy baseline:           acc={greedy['accuracy']:.4f}")

    dola = evaluate_dola_baseline(model, samples, letter_ids, args.max_new_tokens)
    print(f"DoLa standard:             acc={dola['accuracy']:.4f} "
          f"Δ={dola['accuracy'] - greedy['accuracy']:+.4f}")

    # Pure complementary (tau=inf, all complementary)
    pure_comp = evaluate_config(
        model, samples, letter_ids, layer_pairs,
        tau=float("inf"), alpha1=1.0, alpha2=0.0, top_k=3,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Pure complementary (τ=∞): acc={pure_comp['accuracy']:.4f} "
          f"Δ={pure_comp['accuracy'] - greedy['accuracy']:+.4f}")

    # Pure contrastive (tau=0, all contrastive)
    pure_cont = evaluate_config(
        model, samples, letter_ids, layer_pairs,
        tau=0.0, alpha1=0.0, alpha2=1.0, top_k=3,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Pure contrastive (τ=0):    acc={pure_cont['accuracy']:.4f} "
          f"Δ={pure_cont['accuracy'] - greedy['accuracy']:+.4f}")

    # ── Grid search ──
    print(f"\n{'=' * 60}")
    print("Grid Search: Enhanced JS Adaptive Decoding")
    print(f"{'=' * 60}")

    tau_values = [0.05, 0.1, 0.15, 0.2]
    alpha1_values = [0.5, 1.0, 2.0]
    alpha2_values = [0.5, 1.0, 2.0]

    all_results = []
    best_delta = -float("inf")
    best_config = None

    for tau in tau_values:
        for alpha1 in alpha1_values:
            for alpha2 in alpha2_values:
                result = evaluate_config(
                    model, samples, letter_ids, layer_pairs,
                    tau=tau, alpha1=alpha1, alpha2=alpha2, top_k=3,
                    max_new_tokens=args.max_new_tokens,
                )
                delta = result["accuracy"] - greedy["accuracy"]
                result["delta"] = float(delta)
                result["tau"] = tau
                result["alpha1"] = alpha1
                result["alpha2"] = alpha2
                all_results.append(result)

                status = "✓" if delta > 0 else "✗"
                print(f"  τ={tau:.2f} α1={alpha1} α2={alpha2}: "
                      f"acc={result['accuracy']:.4f} Δ={delta:+.4f} {status}")

                if delta > best_delta:
                    best_delta = delta
                    best_config = result

    # ── Report ──
    print(f"\n{'=' * 60}")
    print("Best Configuration")
    print(f"{'=' * 60}")
    if best_config:
        print(f"τ={best_config['tau']:.2f}, α1={best_config['alpha1']}, "
              f"α2={best_config['alpha2']}")
        print(f"Accuracy: {best_config['accuracy']:.4f} "
              f"({best_config['n_correct']}/{best_config['n_total']})")
        print(f"Δ over greedy: {best_config['delta']:+.4f}")

    if best_delta < 1.0:
        print("\n⚠ All τ/α configurations Δ < +1pp — abandon criteria triggered.")
        print("Layer-pair disagreement insufficient for effective decoding at 1.7B scale.")

    # Save
    output = {
        "config": {"n_samples": args.n_samples, "model": args.model,
                   "n_layers": n_layers, "n_candidate_pairs": len(layer_pairs)},
        "baselines": {
            "greedy": greedy,
            "dola_standard": dola,
            "pure_complementary": pure_comp,
            "pure_contrastive": pure_cont,
        },
        "best_config": best_config,
        "full_sweep": all_results,
    }
    with open(output_dir / "enhanced_js_decoding_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_dir / 'enhanced_js_decoding_results.json'}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
