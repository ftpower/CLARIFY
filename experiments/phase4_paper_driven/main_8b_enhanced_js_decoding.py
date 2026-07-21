"""8B Enhanced JS Decoding — test if 8B's stronger layer differentiation
enables effective contrastive decoding via layer-pair JS divergence.

On 1.7B, all JS decoding configs produced ≤ greedy baseline accuracy because:
  (a) JS monotonically decreases with depth (L0 always "most different")
  (b) Layer-pair contrast signal is too weak in 28-layer model

Hypothesis: 8B's 36 layers provide stronger differentiation, making JS-based
contrastive decoding effective. Specific predictions:
  - L0 will NOT dominate all pairs (unlike 1.7B)
  - Top pairs will involve mid-vs-deep layers (L15-L25 range)
  - Contrastive decoding will improve over greedy baseline

Usage (on AutoDL RTX 5090):
    python main_8b_enhanced_js_decoding.py --n_samples 200 --device cuda
    python main_8b_enhanced_js_decoding.py --n_samples 200 --skip_scan  # use cache
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# JS divergence utilities
# ═══════════════════════════════════════════════════════════════════════════════


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    return float(0.5 * (kl_pm + kl_qm))


# ═══════════════════════════════════════════════════════════════════════════════
# D2 scan: find best layer pairs for JS
# ═══════════════════════════════════════════════════════════════════════════════


def extract_choice_probs_8b(
    model, samples: list[dict], letter_ids: dict[str, int],
) -> dict:
    """Extract 4-choice softmax at every layer for all samples.

    Single forward pass per sample. Returns [N, n_layers, 4] array.
    """
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    letters = ["A", "B", "C", "D"]
    letter_toks = [letter_ids[l] for l in letters]

    N = len(samples)
    labels = np.zeros(N, dtype=np.int32)
    p_correct_arr = np.zeros(N, dtype=np.float32)
    choice_probs = np.zeros((N, n_layers, 4), dtype=np.float32)
    all_maxp = np.zeros((N, n_layers), dtype=np.float32)

    for idx, sample in enumerate(tqdm(samples, desc="Extracting choice probs")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        # Hook all layers' residual post
        storage = {}
        hooks = []
        for li in range(n_layers):
            def _hook_factory(key):
                def h(act, hook=None):
                    storage[key] = act.detach()
                    return act
                return h
            hooks.append((f"blocks.{li}.hook_resid_post", _hook_factory(li)))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        for li in range(n_layers):
            h_last = storage[li][0, last_pos, :].to(W_U.device).float()
            logits_L = h_last @ W_U.float()
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)

            probs = torch.softmax(logits_L, dim=-1)
            all_maxp[idx, li] = probs.max().item()

            choice_logits = logits_L[letter_toks]
            choice_probs[idx, li, :] = torch.softmax(choice_logits, dim=-1).cpu().numpy()

        # Correctness
        logits_last = logits[0, last_pos, :]
        lid = torch.tensor(letter_toks, device=logits_last.device)
        pf = torch.softmax(logits_last[lid].float(), dim=-1)
        pred_idx = pf.argmax().item()
        labels[idx] = int(letters[pred_idx] == correct_letter)
        p_correct_arr[idx] = pf[letters.index(correct_letter)].item()

    return {
        "labels": labels,
        "p_correct": p_correct_arr,
        "choice_probs": choice_probs,
        "all_maxp": all_maxp,
    }


def scan_layer_pairs(
    choice_probs: np.ndarray,
    labels: np.ndarray,
    p_correct: np.ndarray,
    exclude_layer0: bool = True,
) -> list[dict]:
    """Scan all (early, late) layer pairs for JS divergence AUROC.

    Returns top-20 pairs sorted by filtered AUROC.
    """
    N, n_layers, _ = choice_probs.shape
    start = 1 if exclude_layer0 else 0

    # Knowledge filter
    filt_mask = p_correct > 0.3
    y = labels[filt_mask]

    print(f"  Scanning {n_layers} layers, "
          f"{(n_layers - start) * (n_layers - start - 1) // 2} pairs...")

    pairs = []
    for early in tqdm(range(start, n_layers - 1), desc="D2 scan", leave=False):
        for late in range(early + 1, n_layers):
            # Compute JS for all samples
            js_all = np.array([
                js_divergence(choice_probs[i, early, :], choice_probs[i, late, :])
                for i in range(N)
            ])
            js_filt = js_all[filt_mask]

            try:
                auc = float(roc_auc_score(y, js_filt))
            except ValueError:
                auc = 0.5

            pairs.append({
                "early": early,
                "late": late,
                "auroc": auc,
                "js_mean": float(js_filt.mean()),
                "js_std": float(js_filt.std()),
            })

    pairs.sort(key=lambda p: p["auroc"], reverse=True)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Contrastive decoding evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_contrastive_decoding(
    model,
    samples: list[dict],
    early_layer: int,
    late_layer: int,
    alpha: float,
    letter_ids: dict[str, int],
) -> dict:
    """Evaluate contrastive decoding with a given (early, late, alpha) config.

    final_logprobs = log_softmax(logits_late) - alpha * log_softmax(logits_early)
    """
    letters = ["A", "B", "C", "D"]
    letter_toks = [letter_ids[l] for l in letters]
    lid_tensor = torch.tensor(letter_toks)

    storage = {}

    def _hook_factory(key):
        def h(act, hook=None):
            storage[key] = act.detach()
            return act
        return h

    hooks = [
        (f"blocks.{early_layer}.hook_resid_post", _hook_factory("early")),
        (f"blocks.{late_layer}.hook_resid_post", _hook_factory("late")),
    ]

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    n_correct = 0
    n_total = len(samples)

    for sample in tqdm(samples, desc=f"Contrastive L{early_layer}-L{late_layer} α={alpha}",
                       leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=hooks)

        lid = lid_tensor.to(W_U.device)

        # Logit lens: hidden → logits per layer
        h_early = storage["early"][0, last_pos, :].float()
        logits_early = h_early @ W_U.float()
        if b_U is not None:
            logits_early = logits_early + b_U.to(W_U.device)

        h_late = storage["late"][0, last_pos, :].float()
        logits_late = h_late @ W_U.float()
        if b_U is not None:
            logits_late = logits_late + b_U.to(W_U.device)

        # Contrastive: logp_late - alpha * logp_early  (on 4-choice only)
        logp_early = F.log_softmax(logits_early[lid].float(), dim=-1)
        logp_late = F.log_softmax(logits_late[lid].float(), dim=-1)
        contrastive = logp_late - alpha * logp_early
        pred_idx = contrastive.argmax().item()
        n_correct += int(letters[pred_idx] == correct_letter)

    acc = n_correct / n_total
    return {"accuracy": acc, "n_correct": n_correct, "n_total": n_total}


def evaluate_greedy_baseline(
    model,
    samples: list[dict],
    letter_ids: dict[str, int],
) -> dict:
    """Greedy baseline accuracy (no intervention)."""
    letters = ["A", "B", "C", "D"]
    letter_toks = [letter_ids[l] for l in letters]

    n_correct = 0
    for sample in tqdm(samples, desc="Greedy baseline", leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, last_pos, :]
        lid = torch.tensor(letter_toks, device=logits_last.device)
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        pred_idx = probs.argmax().item()
        n_correct += int(letters[pred_idx] == correct_letter)

    return {"accuracy": n_correct / len(samples),
            "n_correct": n_correct, "n_total": len(samples)}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="8B Enhanced JS Contrastive Decoding"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_pairs", type=int, default=3,
                        help="Number of top pairs to evaluate")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 1.0, 2.0])
    parser.add_argument("--skip_scan", action="store_true",
                        help="Use cached D2 scan results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_cache = output_dir / "d2_scan_8b.npz"
    scan_json = output_dir / "d2_scan_8b_pairs.json"

    # ── Load model ──
    print(f"Loading model {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"  n_layers={n_layers}, d_model={model.cfg.d_model}")

    letter_ids = {}
    for l in ["A", "B", "C", "D"]:
        toks = model.tokenizer.encode(f" {l}", add_special_tokens=False)
        letter_ids[l] = toks[-1] if len(toks) >= 1 else toks[0]

    print(f"\nLoading HellaSwag (n={args.n_samples})...")
    samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)

    # ── D2 scan (or load cache) ──
    if args.skip_scan and scan_json.exists():
        print(f"\nLoading cached D2 scan from {scan_json}")
        with open(scan_json) as f:
            all_pairs = json.load(f)
        top_pairs = all_pairs[:args.top_k_pairs]
    else:
        print(f"\n{'=' * 60}")
        print(f"D2 Layer-Pair Scan ({n_layers} layers, {args.n_samples} samples)")
        print(f"{'=' * 60}")

        data = extract_choice_probs_8b(model, samples, letter_ids)

        # Save scan data
        np.savez_compressed(
            scan_cache,
            labels=data["labels"],
            p_correct=data["p_correct"],
            choice_probs=data["choice_probs"],
            all_maxp=data["all_maxp"],
        )
        print(f"  Cached choice probs to {scan_cache}")

        all_pairs = scan_layer_pairs(
            data["choice_probs"], data["labels"], data["p_correct"],
            exclude_layer0=True,
        )

        # Save pairs
        with open(scan_json, "w") as f:
            json.dump(all_pairs, f, indent=2)
        print(f"  Saved {len(all_pairs)} pairs to {scan_json}")

        # Identify L0 regression fix
        top_pairs = all_pairs[:args.top_k_pairs]

    # ── Report scan results ──
    print(f"\nTop-{args.top_k_pairs} Layer Pairs by Filtered JS AUROC:")
    print(f"  {'Rank':<6s} {'Early':<8s} {'Late':<8s} {'AUROC':<10s} {'JS Mean':<10s}")
    print(f"  {'-'*42}")
    for i, p in enumerate(top_pairs):
        print(f"  {i+1:<6d} L{p['early']:<7d} L{p['late']:<7d} "
              f"{p['auroc']:<10.4f} {p['js_mean']:<10.6f}")

    # Check the L0 hypothesis
    l0_pairs = [p for p in all_pairs if p["early"] == 0]
    if l0_pairs:
        best_l0 = l0_pairs[0]  # already sorted
        print(f"\n  Best L0-involving pair: L0-L{best_l0['late']}, "
              f"AUROC={best_l0['auroc']:.4f}")
        print(f"  Rank of best L0 pair: {all_pairs.index(best_l0) + 1}/{len(all_pairs)}")
        if best_l0["auroc"] >= top_pairs[0]["auroc"]:
            print("  🔴 L0 STILL dominates (same as 1.7B)")
        else:
            print(f"  ✅ L0 NOT dominant — 8B fixes the DoLa failure mode!")

    # ── Contrastive decoding evaluation ──
    print(f"\n{'=' * 60}")
    print("Contrastive Decoding Evaluation")
    print(f"{'=' * 60}")

    # Greedy baseline
    print("\n  Computing greedy baseline...")
    baseline = evaluate_greedy_baseline(model, samples, letter_ids)
    print(f"  Greedy baseline: {baseline['accuracy']:.4f} "
          f"({baseline['n_correct']}/{baseline['n_total']})")

    # Evaluate all (pair, alpha) combos
    results = []
    for i, pair in enumerate(top_pairs):
        early, late = pair["early"], pair["late"]
        for alpha in args.alphas:
            r = evaluate_contrastive_decoding(
                model, samples, early, late, alpha, letter_ids,
            )
            r["pair_rank"] = i + 1
            r["early_layer"] = early
            r["late_layer"] = late
            r["alpha"] = alpha
            r["delta_vs_greedy"] = r["accuracy"] - baseline["accuracy"]
            sign = "✅" if r["delta_vs_greedy"] > 0.01 else \
                   ("·" if abs(r["delta_vs_greedy"]) < 0.01 else "🔴")
            print(f"  L{early}-L{late} α={alpha:.1f}: "
                  f"acc={r['accuracy']:.4f} Δ={r['delta_vs_greedy']:+.4f} {sign}")
            results.append(r)

    # ── Also test L0 pairs as control ──
    print("\n  Testing L0 pairs (DoLa baseline, expected to fail)...")
    for late in range(10, n_layers, 5):  # sample every 5th layer
        for alpha in [0.5, 1.0]:
            r = evaluate_contrastive_decoding(
                model, samples, 0, late, alpha, letter_ids,
            )
            r["pair_rank"] = -1
            r["early_layer"] = 0
            r["late_layer"] = late
            r["alpha"] = alpha
            r["delta_vs_greedy"] = r["accuracy"] - baseline["accuracy"]
            results.append(r)

    # ── Summary ──
    best = max(results, key=lambda r: r["delta_vs_greedy"])
    top3_positive = sum(1 for r in results if r["delta_vs_greedy"] > 0.01 and r["pair_rank"] > 0)
    top3_total = len([r for r in results if r["pair_rank"] > 0])

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Greedy baseline:            {baseline['accuracy']:.4f}")
    print(f"  Best contrastive:           {best['accuracy']:.4f} "
          f"(L{best['early_layer']}-L{best['late_layer']} α={best['alpha']:.1f})")
    print(f"  Best Δ over greedy:         {best['delta_vs_greedy']:+.4f}")
    print(f"  Top-3 positive configs:     {top3_positive}/{top3_total}")

    if best["delta_vs_greedy"] > 0.02:
        print("  ✅ Contrastive decoding IMPROVES over greedy on 8B!")
    elif best["delta_vs_greedy"] > -0.01:
        print("  ⚠️  Contrastive decoding ~= greedy — marginal improvement.")
    else:
        print("  🔴 All contrastive configs ≤ greedy — same as 1.7B result.")

    # ── Save ──
    output = {
        "config": {
            "n_samples": args.n_samples,
            "model": args.model,
            "n_layers": n_layers,
            "top_k_pairs": args.top_k_pairs,
            "alphas": args.alphas,
            "seed": args.seed,
        },
        "d2_scan": {
            "top_pairs": top_pairs,
            "n_total_pairs": len(all_pairs),
        },
        "baseline": baseline,
        "best": best,
        "top3_positive_rate": top3_positive / max(top3_total, 1),
        "all_results": results,
    }

    results_path = output_dir / "enhanced_js_8b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("8B enhanced JS decoding complete. ✅")


if __name__ == "__main__":
    main()
