"""Phase 18.2: Multi-reference layer TLDC — diagnostic experiment.

Theory: docs/theory-intervention-failure.md §5
Plan:   ~/.claude/plans/CLARIFY/phase18-tldc-improvements.md

Tests whether combining δ from multiple reference layers (not just L20)
improves detection (AUROC) or can be used for EGC/MRC combining.

Diagnostic gates:
  P18.2.1: min pairwise ρ(δ_ℓa, δ_ℓb) < 0.8 — layers have complementary info
  P18.2.2: AUROC(δ_multi, KW vs rest) > AUROC(δ_L20, KW vs rest)

Usage:
    python diagnose_multiref_corr.py --n_test 100

Output:
    experiments/outputs/lin_theory/s18_2_multiref.json
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_lin_dir = str(Path(__file__).parent)
if _lin_dir not in sys.path:
    sys.path.insert(0, _lin_dir)

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    greedy_generate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-layer extraction in a single forward pass
# ═════════════════════════════════════════════════════════════════════════════


def extract_multilayer_logits(
    model, tokenizer, prompt, device, layers, final_layer, ln_final, W_U, b_U
):
    """Extract early-exit logits at multiple layers in a single forward pass.

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        prompt: string
        device: "cuda" or "cpu"
        layers: list of int — layers to extract from
        final_layer: int — last layer index
        ln_final: RMSNorm
        W_U: [d_model, vocab]
        b_U: [vocab] or None

    Returns:
        logits_dict: {layer: [vocab] float32} — early-exit logits per layer
        l_final: [vocab] float32 — final-layer logits
        rank: int — y_true rank from final layer (or -1)
        y_true_id: int or None
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    captured = {}

    # Register hooks for all requested layers + final
    hooks = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Closure-safe factory
        def _make_hook(l):
            def _hook(act, hook=None):
                captured[l] = act[:, -1:, :].detach()
                return act

            return _hook

        hooks.append((hook_name, _make_hook(layer)))

    # Also hook final layer
    hook_final = f"blocks.{final_layer}.hook_resid_post"

    def _hook_final(act, hook=None):
        captured["final"] = act[:, -1:, :].detach()
        return act

    hooks.append((hook_final, _hook_final))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Compute early-exit logits for each layer
    dtype = next(ln_final.parameters()).dtype

    def compute_logits(h):
        h_f16 = h.to(dtype=dtype)
        h_norm = ln_final(h_f16)
        logits = h_norm @ W_U.to(dtype)
        if b_U is not None:
            logits = logits + b_U.to(dtype)
        return logits.float().squeeze()  # [vocab]

    logits_dict = {}
    for layer in layers:
        logits_dict[layer] = compute_logits(captured[layer])

    l_final = compute_logits(captured["final"])

    return logits_dict, l_final


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 18.2: Multi-reference layer diagnosis"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--ref_layers",
        type=int,
        nargs="*",
        default=[10, 15, 18, 20, 22, 24],
        help="Reference layers for multi-ref TLDC (default: L10 L15 L18 L20 L22 L24)",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.05, 0.08, 0.10, 0.12, 0.15],
        help="Beta sweep for multi-ref first-token accuracy",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    ref_layers = args.ref_layers

    print("=" * 72)
    print("Phase 18.2: Multi-Reference Layer TLDC — Diagnosis")
    print(f"  Reference layers: {ref_layers}")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1  # L27
    print(f"  Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Extract multi-layer logits ──
    print(f"\n[2/4] Extracting multi-layer logits ({args.n_test} samples)...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Multi-layer extract")):
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )

        logits_dict, l_final = extract_multilayer_logits(
            model,
            tokenizer,
            prompt,
            device,
            ref_layers,
            final_layer,
            ln_final,
            W_U,
            b_U,
        )

        # y_true rank (final layer)
        sorted_ids = l_final.float().argsort(descending=True)
        rank = int((sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item())

        # Baseline generation for KC/KW/DK classification
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        # Compute δ_ℓ = l_ℓ - l_L for each reference layer
        deltas = {}
        for layer in ref_layers:
            deltas[f"L{layer}"] = (logits_dict[layer] - l_final).cpu()

        # Also store logits for first-token checks
        entry = {
            "sample_id": i,
            "rank": rank,
            "subset": subset,
            "prompt": prompt,
            "answers": sample["answers"],
            "question": sample["question"][:100],
            "y_true_id": y_true_id,
            "l_final": l_final.float().cpu(),  # [vocab]
            "gen_correct": is_correct,
        }
        for layer in ref_layers:
            entry[f"l_L{layer}"] = logits_dict[layer].float().cpu()
            entry[f"delta_L{layer}"] = deltas[f"L{layer}"]
        entries.append(entry)

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]

    n_kw, n_kc, n_dk = len(kw), len(kc), len(dk)
    print(f"  KC={n_kc}, KW={n_kw}, DK={n_dk}, Total={len(entries)}")

    # ── Analysis ──
    print(f"\n[3/4] Analysis: δ correlation + AUROC scan...")

    # P18.2.1: Pairwise δ vector correlation
    print(f"\n  ── P18.2.1: Pairwise δ correlation matrix ──")

    # Stack delta vectors per layer across samples (only on KW+KC for signal relevance)
    # δ per sample: [vocab], but we need a single vector per sample.
    # Use the δ values at y_true_id as the key signal (most relevant for KW correction)
    # Also compute full-vocabulary correlation as robustness check

    K = len(ref_layers)
    # Method 1: correlation of δ[tokens_of_interest] across samples
    # Use y_true_ids + top predictions as the token subspace
    # For simplicity, use the δ vector restricted to a relevant token subspace

    # Method A: Use δ at y_true_id across samples
    delta_at_ytrue = {}  # {layer: [n_samples]}
    for layer in ref_layers:
        vals = []
        for e in entries:
            yt = e["y_true_id"]
            delta = e[f"delta_L{layer}"]
            vals.append(float(delta[yt].item()))
        delta_at_ytrue[f"L{layer}"] = np.array(vals)

    print(f"  ── δ at y_true_id (n={len(entries)} samples) ──")
    corr_yt = np.zeros((K, K))
    layer_names = [f"L{ℓ}" for ℓ in ref_layers]
    for i in range(K):
        for j in range(K):
            corr_yt[i, j] = np.corrcoef(
                delta_at_ytrue[layer_names[i]], delta_at_ytrue[layer_names[j]]
            )[0, 1]

    # Print correlation matrix
    header = "       " + "  ".join(f"{n:>7}" for n in layer_names)
    print(f"  {header}")
    for i, name in enumerate(layer_names):
        row = "  ".join(f"{corr_yt[i, j]:>7.4f}" for j in range(K))
        print(f"  {name:>4}: {row}")

    # Method B: Mean absolute δ per sample (scalar summary)
    delta_mean_abs = {}
    for layer in ref_layers:
        vals = []
        for e in entries:
            delta = e[f"delta_L{layer}"]
            vals.append(float(delta.abs().mean().item()))
        delta_mean_abs[f"L{layer}"] = np.array(vals)

    print(f"\n  ── Mean |δ| (scalar summary, n={len(entries)} samples) ──")
    header2 = "       " + "  ".join(f"{n:>7}" for n in layer_names)
    print(f"  {header2}")
    corr_ma = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            corr_ma[i, j] = np.corrcoef(
                delta_mean_abs[layer_names[i]], delta_mean_abs[layer_names[j]]
            )[0, 1]
    for i, name in enumerate(layer_names):
        row = "  ".join(f"{corr_ma[i, j]:>7.4f}" for j in range(K))
        print(f"  {name:>4}: {row}")

    # P18.2.1: min pairwise correlation (using y_true_id method)
    # Exclude diagonal
    off_diag = []
    for i in range(K):
        for j in range(i + 1, K):
            off_diag.append(abs(corr_yt[i, j]))

    min_pairwise_corr = min(off_diag) if off_diag else 1.0
    p1821_pass = min_pairwise_corr < 0.8
    print(f"\n  Min pairwise |ρ| (δ at y_true): {min_pairwise_corr:.4f}")
    print(f"  P18.2.1 (min pairwise ρ < 0.8): {'✅' if p1821_pass else '❌'}")

    # Also show the y_true correlation range
    print(
        f"  Correlation range (δ at y_true): {min(off_diag):.4f} - {max(off_diag):.4f}"
    )

    # P18.2.2: AUROC comparison
    print(f"\n  ── P18.2.2: AUROC(δ, KW vs KC+DK) per layer ──")

    # Binary label: KW=1, KC+DK=0
    labels = np.array([1 if e["subset"] == "know_wrong" else 0 for e in entries])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    print(f"  KW (positive) = {n_pos}, KC+DK (negative) = {n_neg}")

    auroc_per_layer = {}
    for layer in ref_layers:
        # Use mean(|δ|) across top-100 tokens as signal (sparsity: see 17.1a)
        # Alternative: use δ at y_true_id as signal (simpler, more targeted)
        scores = []
        for e in entries:
            yt = e["y_true_id"]
            delta = e[f"delta_L{layer}"]
            # Signal: signed delta at y_true (model over-hypes when δ < 0)
            scores.append(float(delta[yt].item()))
        scores = np.array(scores)

        if n_pos >= 2 and n_neg >= 2:
            # AUROC: higher score = more over-hyped (more negative δ)
            # FLIP sign so that higher = more likely KW
            auroc = roc_auc_score(labels, -scores)
        else:
            auroc = 0.5
        auroc_per_layer[f"L{layer}"] = float(auroc)
        print(f"  L{layer}: AUROC(δ[y_true], KW vs KC+DK) = {auroc:.4f}")

    auroc_l20 = auroc_per_layer["L20"]
    print(f"\n  L20 AUROC baseline: {auroc_l20:.4f}")

    # EGC: equal gain combining: score = mean(δ_ℓ)
    # MRC: max-ratio combining: weighted by AUROC
    # Test K=2, K=3, K=all combinations
    print(f"\n  ── Multi-ref combination AUROC ──")
    print(f"  {'Method':<20} {'Layers':<25} {'AUROC':>8} {'Δ vs L20':>10}")
    print(f"  {'─' * 20} {'─' * 25} {'─' * 8} {'─' * 10}")

    combination_results = {}

    for k in [2, 3, len(ref_layers)]:
        for combo in combinations(ref_layers, k):
            combo_names = [f"L{ℓ}" for ℓ in combo]
            combo_key = "+".join(combo_names)

            # EGC: mean of delta at y_true
            egc_scores = np.zeros(len(entries))
            for e_idx, e in enumerate(entries):
                yt = e["y_true_id"]
                avg_delta = np.mean([float(e[f"delta_L{ℓ}"][yt].item()) for ℓ in combo])
                egc_scores[e_idx] = avg_delta

            if n_pos >= 2 and n_neg >= 2:
                egc_auroc = roc_auc_score(labels, -egc_scores)
            else:
                egc_auroc = 0.5

            egc_delta = egc_auroc - auroc_l20

            # MRC: weighted by auroc_per_layer
            weights = np.array([auroc_per_layer[f"L{ℓ}"] for ℓ in combo])
            weights = weights / weights.sum()  # normalize

            mrc_scores = np.zeros(len(entries))
            for e_idx, e in enumerate(entries):
                yt = e["y_true_id"]
                w_avg = np.sum(
                    [
                        weights[j] * float(e[f"delta_L{ℓ}"][yt].item())
                        for j, ℓ in enumerate(combo)
                    ]
                )
                mrc_scores[e_idx] = w_avg

            if n_pos >= 2 and n_neg >= 2:
                mrc_auroc = roc_auc_score(labels, -mrc_scores)
            else:
                mrc_auroc = 0.5

            mrc_delta = mrc_auroc - auroc_l20

            print(
                f"  {'EGC':<20} {combo_key:<25} {egc_auroc:>8.4f} {egc_delta:>+10.4f}"
            )
            print(
                f"  {'MRC':<20} {combo_key:<25} {mrc_auroc:>8.4f} {mrc_delta:>+10.4f}"
            )

            combination_results[f"EGC_{combo_key}"] = {
                "method": "EGC",
                "layers": combo_names,
                "auroc": float(egc_auroc),
                "delta_vs_l20": float(egc_delta),
            }
            combination_results[f"MRC_{combo_key}"] = {
                "method": "MRC",
                "layers": combo_names,
                "auroc": float(mrc_auroc),
                "delta_vs_l20": float(mrc_delta),
            }

    # P18.2.2: best multi-ref AUROC > L20 AUROC
    best_multi_auroc = (
        max(r["auroc"] for r in combination_results.values())
        if combination_results
        else auroc_l20
    )

    p1822_pass = best_multi_auroc > auroc_l20
    best_multi_name = (
        [k for k, r in combination_results.items() if r["auroc"] == best_multi_auroc][0]
        if combination_results
        else "none"
    )

    print(f"\n  Best multi-ref AUROC: {best_multi_name} = {best_multi_auroc:.4f}")
    print(
        f"  P18.2.2 (multi-ref AUROC > L20 AUROC): "
        f"{'✅' if p1822_pass else '❌'} "
        f"({best_multi_auroc:.4f} vs {auroc_l20:.4f})"
    )

    # ── Multi-ref TLDC first-token sweep ──
    print(f"\n  ── Multi-ref first-token TLDC sweep (EGC β={args.betas}) ──")
    print(f"  y_egc = l_L + β/K · Σ(l_ℓ - l_L)")

    # Baseline first-token
    bl_kw_ft = sum(
        1 for e in kw if int(e["l_final"].argmax().item()) == e["y_true_id"]
    ) / max(1, n_kw)
    bl_kc_ft = sum(
        1 for e in kc if int(e["l_final"].argmax().item()) == e["y_true_id"]
    ) / max(1, n_kc)
    bl_all_ft = sum(
        1 for e in entries if int(e["l_final"].argmax().item()) == e["y_true_id"]
    ) / max(1, len(entries))

    print(
        f"  Baseline first-token: KW={bl_kw_ft:.1%}, KC={bl_kc_ft:.1%}, All={bl_all_ft:.1%}"
    )

    # Standard TLDC (L20 only) for comparison
    print(f"\n  ── Standard TLDC (L20 only) ──")
    for beta in args.betas:
        correct_kw = sum(
            1
            for e in kw
            if int((e["l_final"] + beta * e[f"delta_L20"]).argmax().item())
            == e["y_true_id"]
        )
        correct_kc = sum(
            1
            for e in kc
            if int((e["l_final"] + beta * e[f"delta_L20"]).argmax().item())
            == e["y_true_id"]
        )
        kw_acc = correct_kw / max(1, n_kw)
        kc_acc = correct_kc / max(1, n_kc)
        print(f"  β={beta:.2f}: KW={kw_acc:.1%}, KC={kc_acc:.1%}")

    # Multi-ref EGC sweep for best K=3 and K=all
    print(f"\n  ── Multi-ref EGC TLDC ──")
    for k in [2, 3, len(ref_layers)]:
        # Pick the best-AUROC combo of size k (from combination_results)
        combos_of_k = [
            (key, r)
            for key, r in combination_results.items()
            if r["method"] == "EGC" and len(r["layers"]) == k
        ]
        if not combos_of_k:
            continue
        best_combo_k = max(combos_of_k, key=lambda x: x[1]["auroc"])
        combo_layers = best_combo_k[1]["layers"]
        combo_key = best_combo_k[0].replace("EGC_", "")

        print(f"\n  EGC K={k}: {combo_key}")
        for beta in args.betas:
            correct_kw = 0
            correct_kc = 0
            correct_all = 0
            for e in entries:
                # EGC: l_L + β/K · Σ δ_ℓ
                delta_sum = sum(e[f"delta_{name}"] for name in combo_layers)
                l_combined = e["l_final"] + (beta / k) * delta_sum
                nid = int(l_combined.argmax().item())
                if e["subset"] == "know_wrong":
                    correct_kw += int(nid == e["y_true_id"])
                elif e["subset"] == "know_correct":
                    correct_kc += int(nid == e["y_true_id"])
                correct_all += int(nid == e["y_true_id"])

            kw_acc = correct_kw / max(1, n_kw)
            kc_acc = correct_kc / max(1, n_kc)
            all_acc = correct_all / max(1, len(entries))
            marker = ""
            if kw_acc > bl_kw_ft:
                marker = " ← KW↑"
            print(
                f"  β={beta:.2f}: KW={kw_acc:.1%}, KC={kc_acc:.1%}, All={all_acc:.1%}{marker}"
            )

    # ── Gate summary ──
    print(f"\n[4/4] Gate Summary")
    print(f"  {'=' * 60}")
    gates = {
        "P18.2.1": {
            "pass": p1821_pass,
            "desc": f"min pairwise |ρ(δ)| < 0.8 (actual: {min_pairwise_corr:.4f})",
        },
        "P18.2.2": {
            "pass": p1822_pass,
            "desc": f"AUROC(δ_multi) > AUROC(δ_L20) (best: {best_multi_auroc:.4f} vs {auroc_l20:.4f})",
        },
    }
    for gname, ginfo in gates.items():
        status = "✅ PASS" if ginfo["pass"] else "❌ FAIL"
        print(f"  {gname}: {status} — {ginfo['desc']}")

    n_pass = sum(1 for g in gates.values() if g["pass"])
    print(f"\n  {n_pass}/{len(gates)} gates passed")
    if n_pass >= 1:
        print(f"  ✅ At least one gate passed → evaluate full-generation eligibility")
    else:
        print(f"  ❌ All gates failed → skip full-generation")

    # ── Save ──
    # Build serializable correlation matrices
    corr_yt_serializable = {
        "layers": layer_names,
        "matrix": [[float(corr_yt[i, j]) for j in range(K)] for i in range(K)],
    }

    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "ref_layers": ref_layers,
            "final_layer": final_layer,
            "rank_threshold": args.rank_threshold,
        },
        "sample_counts": {"KC": n_kc, "KW": n_kw, "DK": n_dk, "total": len(entries)},
        "correlation_yt": corr_yt_serializable,
        "min_pairwise_corr": float(min_pairwise_corr),
        "auroc_per_layer": auroc_per_layer,
        "combination_auroc": combination_results,
        "gates": {
            "P18.2.1": {
                "pass": bool(p1821_pass),
                "min_pairwise_corr": float(min_pairwise_corr),
            },
            "P18.2.2": {
                "pass": bool(p1822_pass),
                "best_multi_auroc": float(best_multi_auroc),
                "auroc_l20": float(auroc_l20),
            },
        },
        "per_sample": [
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank": e["rank"],
                "gen_correct": e["gen_correct"],
                "question": e["question"],
            }
            for e in entries
        ],
    }

    out_path = output_dir / "s18_2_multiref.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
