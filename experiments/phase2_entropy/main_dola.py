"""DoLa: Decoding by Contrasting Layers for HellaSwag 4-choice.

Pure inference-time intervention: subtract premature-layer logits from mature-layer
logits to suppress generic/hallucinatory patterns and amplify knowledge-focused signals.

For each sample:
1. Forward pass with caching → hidden states at all layers
2. Project candidate premature layers + mature layer through ln_final + unembed → logits
3. Compute JS divergence between each premature and mature softmax distribution
4. Select premature layer L* with max JS divergence
5. DoLa score for letter = mature_logprob[letter] - premature_logprob[letter]
6. Apply log_softmax to contrastive difference, pick highest-scoring letter

Reference: Chuang et al., ICLR 2024. Code: reference_code/DoLa/

Usage:
    python main_dola.py --n_samples 500
    python main_dola.py --n_samples 500 --mode dola-static --premature_layer 14
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


def _make_save_last_hook(storage: dict, key: str):
    """Return a hook that saves only the last-position hidden state."""

    def hook(activation, hook=None):
        storage[key] = activation[0, -1, :].detach()  # [d_model], detach from graph
        return activation  # pass-through, no modification

    return hook


def extract_dola_states(
    model,
    tokens: torch.Tensor,
    mature_layer: int,
    candidate_premature_layers: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
    """Forward pass with hooks to extract last-position hidden states only.

    Returns:
        mature_logits: [vocab_size] logits from mature layer.
        premature_states: {layer_idx: hidden_state [d_model]}.
        full_logits: [1, seq, vocab_size] final model logits.
    """
    storage = {}
    hooks = []

    # Hook for mature layer
    mature_key = f"blocks.{mature_layer}.hook_resid_post"
    hooks.append((mature_key, _make_save_last_hook(storage, mature_key)))

    # Hooks for candidate premature layers
    for L in candidate_premature_layers:
        key = f"blocks.{L}.hook_resid_post"
        hooks.append((key, _make_save_last_hook(storage, key)))

    with torch.no_grad():
        full_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Project mature hidden state through ln_final + unembed
    ln_final = model.ln_final
    # Pre-compute contiguous unembedding weights once to avoid repeated
    # .contiguous() calls (each allocates ~620MB for W_U.T).
    W_U_T = model.unembed.W_U.T.contiguous()
    b_U = model.unembed.b_U

    # Project mature hidden state
    h_mature = storage[mature_key]
    mature_logits = F.linear(ln_final(h_mature), W_U_T, b_U)

    # Extract premature hidden states (raw, no projection yet)
    premature_states = {}
    for L in candidate_premature_layers:
        key = f"blocks.{L}.hook_resid_post"
        premature_states[L] = storage[key]

    return mature_logits, premature_states, full_logits, W_U_T, b_U


def _project(
    h: torch.Tensor, ln_final, W_U_T: torch.Tensor, b_U: torch.Tensor
) -> torch.Tensor:
    """Project hidden state through final LayerNorm + unembedding matrix."""
    return F.linear(ln_final(h), W_U_T, b_U)


def compute_dola_scores(
    mature_logits: torch.Tensor,
    premature_states: dict[int, torch.Tensor],
    candidate_premature_layers: list[int],
    model,
    letter_ids: dict[str, int],
    W_U_T: torch.Tensor,
    b_U: torch.Tensor,
    mode: str = "dola",
    premature_layer: int | None = None,
) -> dict:
    """Compute DoLa contrastive scores using pre-extracted hidden states."""
    ln_final = model.ln_final

    if mode == "dola-static":
        assert premature_layer is not None
        h_pre = premature_states[premature_layer]
        premature_logits = _project(h_pre, ln_final, W_U_T, b_U)
        js_divs = None
        selected_layer = premature_layer

    elif mode == "dola":
        mature_logprobs = F.log_softmax(mature_logits.float(), dim=-1)
        mature_probs = torch.exp(mature_logprobs)

        best_js = -1.0
        best_layer = candidate_premature_layers[0]
        js_divs = []
        all_premature_logits = {}

        for L in candidate_premature_layers:
            h_pre = premature_states[L]
            pre_logits = _project(h_pre, ln_final, W_U_T, b_U)
            all_premature_logits[L] = pre_logits

            pre_logprobs = F.log_softmax(pre_logits.float(), dim=-1)
            pre_probs = torch.exp(pre_logprobs)

            M = 0.5 * (mature_probs + pre_probs)
            kl1 = F.kl_div(mature_logprobs, M, reduction="sum")
            kl2 = F.kl_div(pre_logprobs, M, reduction="sum")
            js = 0.5 * (kl1 + kl2).item()

            js_divs.append({"layer": L, "js": js})

            if js > best_js:
                best_js = js
                best_layer = L

        selected_layer = best_layer
        premature_logits = all_premature_logits[selected_layer]

    # ── Compute DoLa contrastive scores ─────────────────────────────
    mature_lp = F.log_softmax(mature_logits.float(), dim=-1)
    premature_lp = F.log_softmax(premature_logits.float(), dim=-1)
    diff = mature_lp - premature_lp
    dola_logprobs = F.log_softmax(diff, dim=-1)

    scores = {}
    for letter in ["A", "B", "C", "D"]:
        scores[letter] = dola_logprobs[letter_ids[letter]].item()

    return {
        "scores": scores,
        "premature_layer": selected_layer,
        "js_divs": js_divs,
    }


def main(args):
    device = args.device

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = load_model(device=device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[0] if len(tok_ids) == 1 else tok_ids[-1]
    print(f"Letter token IDs: {letter_ids}")

    n_layers = model.cfg.n_layers
    mature_layer = n_layers - 1  # last layer (27 for 1.7B, 35 for 8B)
    candidate_premature_layers = list(
        range(0, n_layers, 2)
    )  # even layers: 0,2,4,...,26
    # Remove mature from candidates (if it happens to be even)
    candidate_premature_layers = [
        l for l in candidate_premature_layers if l < mature_layer
    ]

    print(
        f"Model: {n_layers} layers, mature={mature_layer}, "
        f"candidates={candidate_premature_layers}"
    )

    # ── Load data ────────────────────────────────────────────────────
    print(f"Loading HellaSwag ({args.n_samples} samples)...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── Run evaluation ───────────────────────────────────────────────
    results = []
    layer_counts = {}

    for sample in tqdm(samples, desc="DoLa scoring"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()

        tokens = model.to_tokens(prompt, prepend_bos=True)

        # Extract hidden states (memory-efficient: only last position)
        mature_logits, premature_states, logits_full, W_U_T, b_U = extract_dola_states(
            model=model,
            tokens=tokens,
            mature_layer=mature_layer,
            candidate_premature_layers=candidate_premature_layers,
        )

        # Baseline prediction (no DoLa) from full logits
        logits_last = logits_full[0, -1, :]
        lid = torch.tensor(
            [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
        )
        baseline_probs = F.softmax(logits_last[lid].float(), dim=-1)
        baseline_pred = ["A", "B", "C", "D"][baseline_probs.argmax().item()]

        # P(correct) for knowledge filtering
        p_correct = baseline_probs[["A", "B", "C", "D"].index(correct_letter)].item()

        # DoLa scoring
        dola_result = compute_dola_scores(
            mature_logits=mature_logits,
            premature_states=premature_states,
            candidate_premature_layers=candidate_premature_layers,
            model=model,
            letter_ids=letter_ids,
            W_U_T=W_U_T,
            b_U=b_U,
            mode=args.mode,
            premature_layer=args.premature_layer,
        )

        dola_scores = dola_result["scores"]
        dola_pred = max(dola_scores, key=dola_scores.get)
        prem_layer = dola_result["premature_layer"]
        layer_counts[prem_layer] = layer_counts.get(prem_layer, 0) + 1

        results.append(
            {
                "baseline_pred": baseline_pred,
                "dola_pred": dola_pred,
                "correct_letter": correct_letter,
                "baseline_correct": baseline_pred == correct_letter,
                "dola_correct": dola_pred == correct_letter,
                "p_correct": p_correct,
                "premature_layer": prem_layer,
                "dola_scores": dola_scores,
            }
        )

    # ── Evaluate ─────────────────────────────────────────────────────
    n_total = len(results)
    baseline_acc = sum(r["baseline_correct"] for r in results) / n_total
    dola_acc = sum(r["dola_correct"] for r in results) / n_total
    delta = dola_acc - baseline_acc

    # Count flips
    corrected = sum(
        1 for r in results if not r["baseline_correct"] and r["dola_correct"]
    )
    broken = sum(1 for r in results if r["baseline_correct"] and not r["dola_correct"])

    print(f"\n{'=' * 60}")
    print(f"DoLa Contrastive Decoding — {n_total} samples")
    print(f"{'=' * 60}")
    print(f"  Baseline acc:  {baseline_acc:.4f}")
    print(f"  DoLa acc:      {dola_acc:.4f}")
    print(f"  Delta:         {delta:+.4f}")
    print(f"  Corrected: {corrected}, Broken: {broken}")

    # Premature layer distribution
    print(f"\nPremature layer selection (top-5):")
    sorted_layers = sorted(layer_counts.items(), key=lambda x: -x[1])
    for L, count in sorted_layers[:5]:
        pct = 100 * count / n_total
        print(f"  L{L:2d}: {count:4d} ({pct:5.1f}%)")

    # ── Knowledge-filtered evaluation ────────────────────────────────
    if args.knowledge_filter > 0:
        thr = args.knowledge_filter
        filtered = [r for r in results if r["p_correct"] > thr]
        if len(filtered) >= 20:
            n_f = len(filtered)
            base_f = sum(r["baseline_correct"] for r in filtered) / n_f
            dola_f = sum(r["dola_correct"] for r in filtered) / n_f
            corr_f = sum(
                1 for r in filtered if not r["baseline_correct"] and r["dola_correct"]
            )
            brok_f = sum(
                1 for r in filtered if r["baseline_correct"] and not r["dola_correct"]
            )
            print(f"\nKnowledge-filtered (P(correct) > {thr}): {n_f}/{n_total} samples")
            print(f"  Baseline acc:  {base_f:.4f}")
            print(f"  DoLa acc:      {dola_f:.4f}")
            print(f"  Delta:         {dola_f - base_f:+.4f}")
            print(f"  Corrected: {corr_f}, Broken: {brok_f}")
        else:
            print(
                f"\nKnowledge filter threshold {thr} leaves only "
                f"{len(filtered)} samples — skipping."
            )

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dola_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "n_total": n_total,
                "baseline_acc": float(baseline_acc),
                "dola_acc": float(dola_acc),
                "delta": float(delta),
                "corrected": corrected,
                "broken": broken,
                "layer_counts": {str(k): v for k, v in sorted_layers},
                "per_sample": [
                    {
                        "baseline_pred": r["baseline_pred"],
                        "dola_pred": r["dola_pred"],
                        "correct_letter": r["correct_letter"],
                        "baseline_correct": r["baseline_correct"],
                        "dola_correct": r["dola_correct"],
                        "p_correct": r["p_correct"],
                        "premature_layer": r["premature_layer"],
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {output_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mode",
        type=str,
        default="dola",
        choices=["dola", "dola-static"],
        help="dola=dynamic premature selection, dola-static=fixed premature layer",
    )
    parser.add_argument(
        "--premature_layer",
        type=int,
        default=None,
        help="Fixed premature layer for dola-static mode",
    )
    parser.add_argument("--knowledge_filter", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
