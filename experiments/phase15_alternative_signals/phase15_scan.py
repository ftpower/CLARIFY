"""Phase 15: Scan alternative signals for consistent hallucination markers.

Extracts 7 low-dimensional signals from multi-sample paired data,
checks within-question consistency (Δ_q ≠ 0 consistently across questions).

Usage:
    python phase15_scan.py --n_questions 30 --k_samples 5 --temperature 0.8
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct


def extract_signals_for_sample(model, tokens, input_len, device):
    """Extract 7 signals from a single forward pass.

    Returns dict of scalars/vectors.
    """
    n_layers = model.cfg.n_layers
    stored = {}

    # Hook: resid_post at ALL layers and last-token logits
    hooks = []
    for li in range(n_layers):

        def _h(act, hook=None, _li=li):
            stored[f"h_{_li}"] = act[:, :, :].detach()
            return act

        hooks.append((f"blocks.{li}.hook_resid_post", _h))

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    last_logits = logits[0, -1, :]  # [vocab]
    probs = torch.softmax(last_logits, dim=-1)

    # Signal 1: max_prob
    max_prob = probs.max().item()

    # Signal 2: logprob entropy
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    # Signal 3: residual L2 norm at last token (layer 20)
    h_l20 = stored["h_20"][0, input_len - 1, :]
    h_norm = torch.norm(h_l20).item()

    # Signal 4: FFN neuron activation sparsity (from mlp_out at L20)
    # Proxy: L1/L2 ratio of resid (lower = sparser)
    h_l1 = torch.norm(h_l20, p=1).item()
    h_l2 = torch.norm(h_l20, p=2).item()
    sparsity_proxy = h_l1 / (h_l2 + 1e-10)  # higher = denser

    # Signal 5: attention pattern entropy (last token → all prefix)
    # Use resid norm ratio across layers as proxy for attention focus
    # Early layers focus on local, late layers on global
    h_early = stored["h_5"][0, input_len - 1, :]
    h_mid = stored["h_15"][0, input_len - 1, :]
    attn_focus = (torch.norm(h_mid) / (torch.norm(h_early) + 1e-10)).item()

    # Signal 6: cross-layer JS divergence (L5 vs L27)
    probs_early = torch.softmax(
        model.unembed(stored["h_5"][0, input_len - 1, :].unsqueeze(0)), dim=-1
    )
    probs_late = torch.softmax(
        model.unembed(stored["h_27"][0, input_len - 1, :].unsqueeze(0)), dim=-1
    )
    m = 0.5 * (probs_early + probs_late)
    js_div = (
        0.5
        * (
            (
                probs_early * (torch.log(probs_early + 1e-10) - torch.log(m + 1e-10))
            ).sum()
            + (
                probs_late * (torch.log(probs_late + 1e-10) - torch.log(m + 1e-10))
            ).sum()
        ).item()
    )

    # Signal 7: confidence gap (top1 - top2 prob)
    top2 = probs.topk(2).values
    conf_gap = (top2[0] - top2[1]).item()

    return {
        "max_prob": max_prob,
        "entropy": entropy,
        "h_norm": h_norm,
        "sparsity": sparsity_proxy,
        "attn_focus": attn_focus,
        "js_div": js_div,
        "conf_gap": conf_gap,
    }


def multi_sample_extract(model, tokenizer, samples, device, ns=5, temp=0.8):
    """Multi-sample extraction with per-sample signal computation."""
    results = []
    t0 = time.time()

    for sample in tqdm(samples, desc="Multi-sample+signals"):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        gt_answers = sample.get("answers", [sample.get("gt_answer", "")])
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        correct_signals = {
            k: []
            for k in [
                "max_prob",
                "entropy",
                "h_norm",
                "sparsity",
                "attn_focus",
                "js_div",
                "conf_gap",
            ]
        }
        halluc_signals = {sk: [] for sk in correct_signals}

        # Extract pre-generation signals (from input only)
        pre_signals = extract_signals_for_sample(model, tokens, input_len, device)

        for _ in range(ns):
            gids = []
            tk = tokens.clone()
            with torch.no_grad():
                logits = model(tk) / temp
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                nid = torch.multinomial(probs, 1).item()
            gids.append(nid)
            new_tk = torch.cat([tk, torch.tensor([[nid]], device=device)], dim=1)
            for _ in range(19):
                if nid == tokenizer.eos_token_id:
                    break
                if new_tk.shape[1] > 1024:
                    break
                with torch.no_grad():
                    logits = model(new_tk) / temp
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                nid = torch.multinomial(probs, 1).item()
                gids.append(nid)
                new_tk = torch.cat(
                    [new_tk, torch.tensor([[nid]], device=device)], dim=1
                )

            ans = tokenizer.decode(gids).strip()
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")

            target = correct_signals if is_correct else halluc_signals
            for k, v in pre_signals.items():
                target[k].append(v)

        results.append(
            {
                "question": sample["question"][:80],
                "correct_signals": correct_signals,
                "halluc_signals": halluc_signals,
                "n_correct": sum(1 for _ in correct_signals["max_prob"]),
                "n_halluc": sum(1 for _ in halluc_signals["max_prob"]),
            }
        )

    print(f"  Extracted {len(results)} questions in {time.time() - t0:.0f}s")
    return results


def analyze_signals(multi_results):
    """Compute within-question Δ_q for each signal.

    Returns:
        per_signal_stats: dict with 'mean_delta', 't_stat', 'frac_same_sign'
        interpretation: which signals are consistent
    """
    signal_names = [
        "max_prob",
        "entropy",
        "h_norm",
        "sparsity",
        "attn_focus",
        "js_div",
        "conf_gap",
    ]
    stats = {}

    for sig in signal_names:
        deltas = []
        for r in multi_results:
            if r["n_correct"] > 0 and r["n_halluc"] > 0:
                c_mean = np.mean(r["correct_signals"][sig])
                h_mean = np.mean(r["halluc_signals"][sig])
                deltas.append(h_mean - c_mean)

        if len(deltas) < 3:
            stats[sig] = None
            continue

        deltas = np.array(deltas)
        mean_d = np.mean(deltas)
        std_d = np.std(deltas, ddof=1)
        t_stat = mean_d / (std_d / np.sqrt(len(deltas)) + 1e-10)
        # Fraction of questions where Δ has the same sign as mean
        if mean_d != 0:
            frac_same = (deltas * np.sign(mean_d) > 0).mean()
        else:
            frac_same = 0.5

        stats[sig] = {
            "mean_delta": float(mean_d),
            "std_delta": float(std_d),
            "t_stat": float(t_stat),
            "frac_same_sign": float(frac_same),
            "n_paired": len(deltas),
        }

    return stats


def print_analysis(stats, multi_results):
    """Print signal analysis results."""
    n_total = len(multi_results)
    n_paired = sum(1 for r in multi_results if r["n_correct"] > 0 and r["n_halluc"] > 0)

    print(f"\n  Paired questions: {n_paired}/{n_total}")
    print(
        f"\n  {'Signal':15s} {'Δ_mean':>10s} {'t_stat':>8s} "
        f"{'frac_same':>10s} {'Consistent?':>12s}"
    )
    print(f"  {'─' * 58}")

    for sig, s in sorted(
        stats.items(), key=lambda x: abs(x[1]["t_stat"]) if x[1] else 0, reverse=True
    ):
        if s is None:
            continue
        consistent = (
            "✅" if (abs(s["t_stat"]) > 2.0 and s["frac_same_sign"] > 0.6) else ""
        )
        print(
            f"  {sig:15s} {s['mean_delta']:>+10.4f} {s['t_stat']:>+8.2f} "
            f"{s['frac_same_sign']:>10.2f} {consistent:>12s}"
        )


def main():
    parser = argparse.ArgumentParser(description="Phase 15: Signal Scan")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase15")
    parser.add_argument("--n_questions", type=int, default=30)
    parser.add_argument("--k_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 60}")
    print(f"Phase 15: Alternative Signal Scan")
    print(f"  Questions: {args.n_questions} × {args.k_samples} samples")
    print(f"{'=' * 60}\n")

    # Load
    print("Loading model...")
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    samples = load_triviaqa(n_samples=args.n_questions, seed=args.seed)

    # Extract
    print(f"\n[Phase 1] Multi-sample extraction with signal computation...")
    results = multi_sample_extract(
        model, tokenizer, samples, device, ns=args.k_samples, temp=args.temperature
    )

    # Analyze
    print(f"\n[Phase 2] Within-question signal consistency analysis...")
    stats = analyze_signals(results)
    print_analysis(stats, results)

    # Summary
    consistent = [
        (sig, s)
        for sig, s in stats.items()
        if s and abs(s["t_stat"]) > 2.0 and s["frac_same_sign"] > 0.6
    ]
    print(f"\n{'=' * 60}")
    if consistent:
        print("CONSISTENT signals found:")
        for sig, s in sorted(consistent, key=lambda x: -abs(x[1]["t_stat"])):
            print(f"  {sig}: t={s['t_stat']:.2f}, frac_same={s['frac_same_sign']:.2f}")
    else:
        print("NO consistent signal found across questions.")
        print(
            "All signals show question-dependent variation — no unified hallucination marker."
        )
    print(f"{'=' * 60}\n")

    # Save
    save = Path(args.output_dir) / "phase15_signals.json"
    with open(save, "w") as f:
        json.dump(
            {
                "n_questions": args.n_questions,
                "k_samples": args.k_samples,
                "signal_stats": {k: v for k, v in stats.items() if v},
                "has_consistent": len(consistent) > 0,
            },
            f,
            indent=2,
        )
    print(f"Saved: {save}")


if __name__ == "__main__":
    main()
