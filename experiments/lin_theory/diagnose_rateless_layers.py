"""Phase 19.3: Rateless adaptive layers — diagnostic experiment.

Theory: docs/llm-coding-theory.md §12
Plan:   ~/.claude/plans/CLARIFY/phase19-beyond-tldc.md

Tests whether samples need different numbers of reference layers — KW samples
should need more layers (higher K(x)) before argmax flips, because their
override is stronger and requires more diversity branches to overcome.

Diagnostic gate:
  P19.3.1: E_KW[K(x)] > E_KC[K(x)], Mann-Whitney U, α=0.05

Usage:
    python diagnose_rateless_layers.py --n_test 100

Output:
    experiments/outputs/lin_theory/s19_3_rateless.json
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import mannwhitneyu

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
# Multi-layer logit extraction (same as diagnose_multiref_corr.py)
# ═════════════════════════════════════════════════════════════════════════════


def extract_multilayer_logits(
    model, tokenizer, prompt, device, layers, final_layer, ln_final, W_U, b_U
):
    """Extract early-exit logits at multiple layers in a single forward pass."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    captured = {}

    def _make_hook(l):
        def _hook(act, hook=None):
            captured[l] = act[:, -1:, :].detach()
            return act

        return _hook

    hooks = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        hooks.append((hook_name, _make_hook(layer)))

    hook_final = f"blocks.{final_layer}.hook_resid_post"

    def _hook_final(act, hook=None):
        captured["final"] = act[:, -1:, :].detach()
        return act

    hooks.append((hook_final, _hook_final))

    with torch.no_grad():
        _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

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
        description="Phase 19.3: Rateless adaptive layer diagnosis"
    )
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--ref_layers",
        type=int,
        nargs="*",
        default=[10, 15, 18, 20, 22, 24],
        help="Candidate reference layers",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.05, 0.08, 0.10, 0.15],
        help="Beta sweep for K(x) sensitivity analysis",
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

    print("=" * 72)
    print("Phase 19.3: Rateless Adaptive Layers — Diagnosis")
    print(f"  Reference layers: {args.ref_layers}")
    print(f"  n_test={args.n_test}, seed={args.seed_test}")
    print("=" * 72)

    # ── Load model ──
    print("\n[1/3] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    final_layer = model.cfg.n_layers - 1  # L27
    print(f"  Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Extract multi-layer logits ──
    print(f"\n[2/3] Extracting multi-layer logits ({args.n_test} samples)...")
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
            args.ref_layers,
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

        # Store logits
        entry = {
            "sample_id": i,
            "rank": rank,
            "subset": subset,
            "y_true_id": y_true_id,
            "l_final": l_final.float().cpu(),
            "gen_correct": is_correct,
            "question": sample["question"][:100],
        }
        for layer in args.ref_layers:
            entry[f"l_L{layer}"] = logits_dict[layer].float().cpu()
            entry[f"delta_L{layer}"] = (logits_dict[layer] - l_final).float().cpu()
        entries.append(entry)

    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]

    n_kw, n_kc, n_dk = len(kw), len(kc), len(dk)
    print(f"  KC={n_kc}, KW={n_kw}, DK={n_dk}, Total={len(entries)}")

    # ── Analysis: K(x) distribution ──
    print(f"\n[3/3] Rateless K(x) analysis...")

    # Compute per-layer AUROC for layer ordering
    from sklearn.metrics import roc_auc_score

    labels = np.array([1 if e["subset"] == "know_wrong" else 0 for e in entries])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    auroc_per_layer = {}
    for layer in args.ref_layers:
        scores = np.array(
            [float(e[f"delta_L{layer}"][e["y_true_id"]].item()) for e in entries]
        )
        if n_pos >= 2 and n_neg >= 2:
            auroc = roc_auc_score(labels, -scores)  # negate: more negative δ = more KW
        else:
            auroc = 0.5
        auroc_per_layer[layer] = float(auroc)

    # Order layers by AUROC descending (best detection first)
    ordered_layers = sorted(
        auroc_per_layer.keys(), key=lambda l: auroc_per_layer[l], reverse=True
    )
    print(f"\n  Layer ordering (by AUROC desc): {[f'L{l}' for l in ordered_layers]}")
    for l in ordered_layers:
        print(f"    L{l}: AUROC = {auroc_per_layer[l]:.4f}")

    # Rateless K(x) computation
    # For each sample, sequentially add layers. Stop when argmax changes from baseline.
    # K(x) = 0 if no layer flips argmax; otherwise K(x) = ordinal of first flipping layer.

    results_by_beta = {}

    for beta in args.betas:
        print(f"\n  ── β = {beta:.2f} ──")

        k_values = {}  # {subset: [K(x)]}
        flip_details = {}  # {subset: [layer that caused flip or None]}

        for subset_name, subset_entries in [("KC", kc), ("KW", kw), ("DK", dk)]:
            k_vals = []
            flip_layers = []

            for e in subset_entries:
                l_final = e["l_final"]
                baseline_argmax = int(l_final.argmax().item())
                yt = e["y_true_id"]

                k = 0  # K(x) = number of layers before flip (0 = no flip)
                flipped = False
                flip_layer = None

                # Sequential accumulation
                y_accum = l_final.clone()
                for ordinal, layer in enumerate(ordered_layers):
                    delta = e[f"delta_L{layer}"]
                    y_accum = y_accum + beta * delta
                    new_argmax = int(y_accum.argmax().item())

                    if new_argmax != baseline_argmax:
                        k = ordinal + 1
                        flip_layer = f"L{layer}"
                        flipped = True
                        break

                # Also check: did the flip help? (argmax → y_true)
                flipped_to_correct = False
                if flipped:
                    flipped_to_correct = new_argmax == yt

                k_vals.append(k)
                flip_layers.append(
                    {
                        "k": k,
                        "flipped": flipped,
                        "flip_layer": flip_layer,
                        "baseline_argmax": baseline_argmax,
                        "y_true_id": yt,
                        "baseline_correct": baseline_argmax == yt,
                        "flipped_to_correct": flipped_to_correct,
                    }
                )

            k_values[subset_name] = np.array(k_vals)
            flip_details[subset_name] = flip_layers

        # Statistics
        print(
            f"  {'Subset':<8} {'n':>4} {'mean K':>8} {'std K':>8} "
            f"{'median K':>8} {'flip%':>8} {'flip→correct%':>14}"
        )

        for subset_name, subset_entries in [("KC", kc), ("KW", kw), ("DK", dk)]:
            kv = k_values[subset_name]
            fd = flip_details[subset_name]
            n = len(kv)
            mean_k = kv.mean()
            std_k = kv.std()
            median_k = np.median(kv)
            flip_pct = sum(1 for f in fd if f["flipped"]) / max(1, n)
            flip_correct_pct = sum(1 for f in fd if f["flipped_to_correct"]) / max(1, n)

            print(
                f"  {subset_name:<8} {n:>4} {mean_k:>8.2f} {std_k:>8.2f} "
                f"{median_k:>8.1f} {flip_pct:>7.1%} {flip_correct_pct:>13.1%}"
            )

        # P19.3.1: E_KW[K(x)] > E_KC[K(x)], Mann-Whitney U
        kw_k = k_values["KW"]
        kc_k = k_values["KC"]

        if len(kw_k) > 0 and len(kc_k) > 0:
            # One-sided: KW > KC
            stat, p_value = mannwhitneyu(kw_k, kc_k, alternative="greater")
            p1931_pass = p_value < 0.05
            print(
                f"\n  P19.3.1 (E_KW[K] > E_KC[K]): "
                f"KW mean={kw_k.mean():.2f}, KC mean={kc_k.mean():.2f}, "
                f"U={stat:.0f}, p={p_value:.4f} "
                f"{'✅' if p1931_pass else '❌'}"
            )
        else:
            p_value = 1.0
            p1931_pass = False
            print(f"\n  P19.3.1: Insufficient data (KW={len(kw_k)}, KC={len(kc_k)})")

        # Also check KW vs DK
        if len(kw_k) > 0 and len(dk) > 0:
            dk_k = k_values["DK"]
            stat_kw_dk, p_kw_dk = mannwhitneyu(kw_k, dk_k, alternative="greater")
            print(
                f"  KW vs DK: KW mean={kw_k.mean():.2f}, DK mean={dk_k.mean():.2f}, "
                f"U={stat_kw_dk:.0f}, p={p_kw_dk:.4f}"
            )

        results_by_beta[str(beta)] = {
            "beta": beta,
            "layer_ordering": [f"L{l}" for l in ordered_layers],
            "auroc_per_layer": {f"L{l}": auroc_per_layer[l] for l in ordered_layers},
            "k_stats": {
                sn: {
                    "n": int(len(k_values[sn])),
                    "mean": float(k_values[sn].mean()),
                    "std": float(k_values[sn].std()),
                    "median": float(np.median(k_values[sn])),
                    "flip_pct": float(
                        sum(1 for f in flip_details[sn] if f["flipped"])
                        / max(1, len(k_values[sn]))
                    ),
                    "flip_to_correct_pct": float(
                        sum(1 for f in flip_details[sn] if f["flipped_to_correct"])
                        / max(1, len(k_values[sn]))
                    ),
                }
                for sn in ["KC", "KW", "DK"]
            },
            "gate_p1931": {
                "pass": bool(p1931_pass),
                "kw_mean": float(kw_k.mean()),
                "kc_mean": float(kc_k.mean()),
                "mannwhitney_u": float(stat)
                if len(kw_k) > 0 and len(kc_k) > 0
                else None,
                "p_value": float(p_value),
            },
        }

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Gate Summary")
    print(f"{'=' * 60}")
    for beta_key, rb in results_by_beta.items():
        g = rb["gate_p1931"]
        status = "✅ PASS" if g["pass"] else "❌ FAIL"
        print(f"  β={beta_key}: {status} — p={g['p_value']:.4f}")

    n_pass = sum(1 for rb in results_by_beta.values() if rb["gate_p1931"]["pass"])
    print(f"\n  {n_pass}/{len(results_by_beta)} β values passed P19.3.1")

    # ── Save ──
    output = {
        "config": {
            "n_test": args.n_test,
            "seed_test": args.seed_test,
            "ref_layers": args.ref_layers,
            "final_layer": final_layer,
            "rank_threshold": args.rank_threshold,
            "betas": args.betas,
        },
        "sample_counts": {"KC": n_kc, "KW": n_kw, "DK": n_dk, "total": len(entries)},
        "results_by_beta": results_by_beta,
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

    out_path = output_dir / "s19_3_rateless.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
