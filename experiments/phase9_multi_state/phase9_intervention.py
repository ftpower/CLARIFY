"""Phase 9.2: FFN-Targeted Preemptive Intervention.

Tests whether modifying FFN output (vs resid_post) can change generation correctness.

Intervention conditions:
  A. FFN fixed-direction: m' = m + α·v_m    (α ∈ scan range)
  B. Answer-aware global: m' = m + β·v_m*   (v_m* = mean(m*) - mean(m))
  C. Learned mapping:    m' = m + g_φ(m)    (low-rank MLP, trained on m→m*)
  D. Resid baseline:     h' = h + α·v_h    (expect zero effect, replicating Phase 8D)

Usage:
    python phase9_intervention.py --load outputs_phase9/phase9_extract.json --n_test 50
    python phase9_intervention.py --load outputs_phase9/phase9_extract.json --n_test 30 --skip_learned
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from src.data_loader import format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Low-Rank Correction Network
# ═══════════════════════════════════════════════════════════════════════════════

class LowRankCorrection(nn.Module):
    """g_φ(m) = U @ (V @ m) + b  — rank-r linear correction.

    U ∈ R^{d×r}, V ∈ R^{r×d}, b ∈ R^d
    Total params: 2·d·r + d
    """
    def __init__(self, d_model, rank=4):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.U = nn.Linear(rank, d_model, bias=False)   # r → d
        self.V = nn.Linear(d_model, rank, bias=False)   # d → r
        self.bias = nn.Parameter(torch.zeros(d_model))
        # Initialize small
        nn.init.normal_(self.U.weight, std=0.01 / rank)
        nn.init.normal_(self.V.weight, std=0.01 / d_model)

    def forward(self, m):
        """m: [batch, d] or [d] → correction: same shape."""
        z = self.V(m)       # [..., r]
        out = self.U(z)     # [..., d]
        return out + self.bias


def train_correction_network(train_m, train_m_star, rank=4,
                              lr=1e-3, weight_decay=1e-2, epochs=500,
                              verbose=True):
    """Train g_φ to map m → (m* - m).

    Args:
        train_m: [N, d] normal FFN outputs
        train_m_star: [N, d] answer-augmented FFN outputs
    Returns:
        trained LowRankCorrection module (on CPU, eval mode)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = train_m.shape[1]

    net = LowRankCorrection(d_model, rank=rank).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    m_tensor = torch.tensor(train_m, dtype=torch.float32).to(device)
    m_star_tensor = torch.tensor(train_m_star, dtype=torch.float32).to(device)
    target = m_star_tensor - m_tensor  # correction target

    # Simple train/val split within training set
    n_train = int(len(m_tensor) * 0.8)
    idx = torch.randperm(len(m_tensor))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    best_val_loss = float("inf")
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        pred = net(m_tensor[train_idx])
        loss = nn.functional.mse_loss(m_tensor[train_idx] + pred, m_star_tensor[train_idx])
        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_pred = net(m_tensor[val_idx])
            val_loss = nn.functional.mse_loss(
                m_tensor[val_idx] + val_pred, m_star_tensor[val_idx]
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        net.load_state_dict(best_state)

    # Move to CPU and eval mode
    net.eval()
    net.cpu()

    if verbose:
        # Compute metrics on CPU
        m_cpu = m_tensor.cpu()
        m_star_cpu = m_star_tensor.cpu()
        target_cpu = target.cpu()
        with torch.no_grad():
            all_pred = net(m_cpu)
            mse = nn.functional.mse_loss(m_cpu + all_pred, m_star_cpu).item()
            corrections = all_pred.numpy()
            actual_diffs = target_cpu.numpy()
            cos_sim = np.mean([
                np.dot(corrections[i], actual_diffs[i]) /
                (np.linalg.norm(corrections[i]) * np.linalg.norm(actual_diffs[i]) + 1e-10)
                for i in range(len(corrections))
            ])
        print(f"  g_φ trained: MSE={mse:.6f}, val_MSE={best_val_loss:.6f}, "
              f"mean cos(correction, actual)={cos_sim:.4f}")

    return net


# ═══════════════════════════════════════════════════════════════════════════════
# Generation with intervention
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_intervention(model, tokenizer, prompt, device, layer,
                                intervention_type, intervention_value,
                                max_new_tokens=20):
    """Generate with an intervention hook at specified layer.

    Args:
        intervention_type: 'mlp' (hook_mlp_out) or 'resid' (hook_resid_post)
        intervention_value: vector [d] to SET at last input token position

    Returns:
        generated answer string
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    # Hook name
    if intervention_type == "mlp":
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif intervention_type == "resid":
        hook_name = f"blocks.{layer}.hook_resid_post"
    else:
        raise ValueError(f"Unknown intervention_type: {intervention_type}")

    m_mod = torch.tensor(intervention_value, dtype=torch.float32, device=device)

    def _intervention_hook(act, hook=None):
        # Only modify last INPUT token position (NOT generated tokens)
        act[0, input_len - 1, :] = m_mod
        return act

    # First forward: encode prompt with intervention
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _intervention_hook)])

    # Generate
    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(max_new_tokens - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _intervention_hook)])
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    return tokenizer.decode(gids).strip()


def baseline_generate(model, tokenizer, prompt, device, max_new_tokens=20):
    """Normal generation without any intervention."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits = model(tokens)

    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(max_new_tokens - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    return tokenizer.decode(gids).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 9.2: FFN Intervention")
    parser.add_argument("--load", type=str, default="outputs_phase9/phase9_extract.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=50,
                        help="Number of test samples for intervention eval")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--alphas", type=str, default="-1.0,-0.5,0.5,1.0",
                        help="Comma-separated α values for fixed-direction scan")
    parser.add_argument("--skip_learned", action="store_true",
                        help="Skip learned g_φ training + evaluation")
    parser.add_argument("--learned_rank", type=int, default=4,
                        help="Rank for low-rank correction network")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"\n{'='*60}")
    print(f"Phase 9.2: FFN-Targeted Preemptive Intervention")
    print(f"  Layer: {args.layer}  Test samples: {args.n_test}")
    print(f"  Alphas: {alphas}")
    print(f"{'='*60}\n")

    # ── Load data ──
    print(f"Loading: {args.load}")
    with open(args.load) as f:
        data = json.load(f)

    records = data["records"]
    n_total = len(records)

    # Split: last n_test samples as test, rest as train
    np.random.seed(args.seed)
    indices = np.random.permutation(n_total)
    test_idx = indices[:args.n_test]
    train_idx = indices[args.n_test:]

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    labels_all = np.array([r["label"] for r in records])

    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(f"  Test correct: {labels_all[test_idx].sum()}/{len(test_idx)} "
          f"({labels_all[test_idx].mean():.1%})")

    li_str = str(args.layer)

    # ── Compute truth directions (on train set) ──
    train_H = np.stack([np.array(r["h"][li_str]) for r in train_records], axis=0)
    train_A = np.stack([np.array(r["a"][li_str]) for r in train_records], axis=0)
    train_M = np.stack([np.array(r["m"][li_str]) for r in train_records], axis=0)
    train_M_star = np.stack([np.array(r["m_star"]) for r in train_records], axis=0)
    train_labels = np.array([r["label"] for r in train_records])

    # v_h = mean(correct) - mean(incorrect) for resid
    mc, mi = train_labels == 1, train_labels == 0
    v_h = train_H[mc].mean(0) - train_H[mi].mean(0)
    v_h /= np.linalg.norm(v_h) + 1e-10

    # v_m = same for FFN
    v_m = train_M[mc].mean(0) - train_M[mi].mean(0)
    v_m /= np.linalg.norm(v_m) + 1e-10

    # v_m* = global answer-aware direction
    v_m_star = train_M_star.mean(0) - train_M.mean(0)
    v_m_star /= np.linalg.norm(v_m_star) + 1e-10

    print(f"\n  cos(v_h, v_m):   {np.dot(v_h, v_m):.4f}")
    print(f"  cos(v_h, v_m*):  {np.dot(v_h, v_m_star):.4f}")
    print(f"  cos(v_m, v_m*):  {np.dot(v_m, v_m_star):.4f}")

    # ── Train learned correction ──
    g_phi = None
    if not args.skip_learned:
        print(f"\n{'─'*50}")
        print(f"Training g_φ (rank={args.learned_rank})...")
        g_phi = train_correction_network(
            train_M, train_M_star,
            rank=args.learned_rank,
        )

    # ── Load model for generation ──
    print(f"\nLoading model for generation...")
    t0 = time.time()
    gen_model = load_model(device=device, model_id=args.model)
    gen_tokenizer = gen_model.tokenizer
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # ── Run interventions on test set ──
    print(f"\n{'─'*60}")
    print(f"Running interventions on {len(test_idx)} test samples")
    print(f"{'─'*60}")

    all_results = []

    for i_test, rec in enumerate(tqdm(test_records, desc="Intervention test")):
        prompt = format_prompt(rec["question"], rec.get("context", ""), dataset="triviaqa")

        # Get the initial states from the extraction data
        m_initial = np.array(rec["m"][li_str])
        h_initial = np.array(rec["h"][li_str])
        true_label = rec["label"]
        # Use all acceptable answers for correctness checking
        gt_answers = rec.get("gt_answers", [rec["gt_answer"]])

        sample_result = {
            "idx": int(test_idx[i_test]),
            "question": rec["question"][:100],
            "true_label": true_label,
            "conditions": {},
        }

        # ── Baseline (no intervention) ──
        ans_base = baseline_generate(gen_model, gen_tokenizer, prompt, device)
        is_correct_base = check_correct(ans_base, gt_answers, dataset="triviaqa")
        sample_result["conditions"]["baseline"] = {
            "type": "baseline", "correct": is_correct_base, "answer": ans_base[:150],
        }

        # ── A: FFN fixed-direction ──
        for alpha in alphas:
            m_mod = m_initial + alpha * v_m
            ans = generate_with_intervention(
                gen_model, gen_tokenizer, prompt, device,
                args.layer, "mlp", m_mod,
            )
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
            sample_result["conditions"][f"ffn_fixed_a{alpha}"] = {
                "type": "ffn_fixed", "alpha": alpha, "correct": is_correct,
                "answer": ans[:150],
            }

        # ── B: Answer-aware global direction ──
        for alpha in alphas:
            m_mod = m_initial + alpha * v_m_star
            ans = generate_with_intervention(
                gen_model, gen_tokenizer, prompt, device,
                args.layer, "mlp", m_mod,
            )
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
            sample_result["conditions"][f"ffn_aware_a{alpha}"] = {
                "type": "ffn_aware", "alpha": alpha, "correct": is_correct,
                "answer": ans[:150],
            }

        # ── C: Learned mapping ──
        if g_phi is not None:
            with torch.no_grad():
                m_tensor = torch.tensor(m_initial, dtype=torch.float32)
                correction = g_phi(m_tensor).numpy()
            m_mod = m_initial + correction
            ans = generate_with_intervention(
                gen_model, gen_tokenizer, prompt, device,
                args.layer, "mlp", m_mod,
            )
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
            sample_result["conditions"]["ffn_learned"] = {
                "type": "ffn_learned", "correct": is_correct, "answer": ans[:150],
                "correction_norm": float(np.linalg.norm(correction)),
            }

        # ── D: Resid baseline ──
        for alpha in alphas:
            h_mod = h_initial + alpha * v_h
            ans = generate_with_intervention(
                gen_model, gen_tokenizer, prompt, device,
                args.layer, "resid", h_mod,
            )
            is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
            sample_result["conditions"][f"resid_baseline_a{alpha}"] = {
                "type": "resid_baseline", "alpha": alpha, "correct": is_correct,
                "answer": ans[:150],
            }

        all_results.append(sample_result)

    # ── Aggregate results ──
    print(f"\n{'─'*60}")
    print("Intervention Results")
    print(f"{'─'*60}")

    # Baseline
    baseline_correct = sum(
        r["conditions"]["baseline"]["correct"] for r in all_results
    )
    n_test = len(all_results)

    print(f"\n  Baseline:            {baseline_correct}/{n_test} "
          f"({baseline_correct/n_test:.1%})")

    # Check baseline vs extraction label consistency
    extraction_labels = np.array([r["true_label"] for r in all_results])
    baseline_matches = np.array([r["conditions"]["baseline"]["correct"] for r in all_results])
    label_agreement = (extraction_labels == baseline_matches).mean()
    if label_agreement < 0.9:
        print(f"  ⚠️  Baseline-label agreement: {label_agreement:.1%} — "
              f"regenerated baseline differs from extraction labels!")
        print(f"     Extraction correct rate: {extraction_labels.mean():.1%}")
        print(f"     Baseline correct rate:   {baseline_matches.mean():.1%}")
    else:
        print(f"  Baseline-label agreement: {label_agreement:.1%} ✓")

    # Group conditions by type
    condition_types = {}

    for r in all_results:
        for cond_name, cond in r["conditions"].items():
            ctype = cond["type"]
            if ctype not in condition_types:
                condition_types[ctype] = {"correct": 0, "total": 0}
            condition_types[ctype]["correct"] += cond["correct"]
            condition_types[ctype]["total"] += 1

            # Also track per-alpha for fixed direction types
            if "alpha" in cond:
                alpha = cond["alpha"]
                alpha_key = f"{ctype}_a{alpha}"
                if alpha_key not in condition_types:
                    condition_types[alpha_key] = {"correct": 0, "total": 0}
                condition_types[alpha_key]["correct"] += cond["correct"]
                condition_types[alpha_key]["total"] += 1

    # Print per-condition-type summary
    print(f"\n  {'Condition':30s} {'Correct':>8s}  {'Rate':>8s}  {'Δ':>8s}")
    print(f"  {'─'*58}")
    for ctype in sorted(condition_types.keys()):
        ct = condition_types[ctype]
        rate = ct["correct"] / ct["total"]
        delta = rate - baseline_correct / n_test

        if ctype == "baseline":
            continue

        marker = ""
        if delta > 0.05:
            marker = " ↑"
        elif delta < -0.05:
            marker = " ↓"

        print(f"  {ctype:30s} {ct['correct']:>4d}/{ct['total']:<3d} "
              f"{rate:>8.1%}  {delta:>+8.1%}{marker}")

    # ── Per-sample change analysis ──
    print(f"\n{'─'*60}")
    print("Per-Sample Change Analysis")
    print(f"{'─'*60}")

    # For each intervention type, count: improved, worsened, unchanged
    for ctype in ["ffn_fixed", "ffn_aware", "resid_baseline"]:
        improved, worsened, unchanged = 0, 0, 0
        for r in all_results:
            base_correct = r["conditions"]["baseline"]["correct"]
            # Aggregate across alphas for this type
            for cond_name, cond in r["conditions"].items():
                if cond["type"] == ctype:
                    if cond["correct"] and not base_correct:
                        improved += 1
                    elif not cond["correct"] and base_correct:
                        worsened += 1
                    else:
                        unchanged += 1
        total = improved + worsened + unchanged
        if total > 0:
            print(f"  {ctype:20s}: improved={improved}, worsened={worsened}, "
                  f"unchanged={unchanged} (total checks={total})")

    if g_phi is not None:
        improved, worsened, unchanged = 0, 0, 0
        for r in all_results:
            base_correct = r["conditions"]["baseline"]["correct"]
            lc = r["conditions"]["ffn_learned"]["correct"]
            if lc and not base_correct:
                improved += 1
            elif not lc and base_correct:
                worsened += 1
            else:
                unchanged += 1
        print(f"  {'ffn_learned':20s}: improved={improved}, worsened={worsened}, "
              f"unchanged={unchanged}")

    # ── Show examples: samples where FFN intervention changed the answer ──
    print(f"\n{'─'*60}")
    print("Examples: samples where FFN intervention changed answer")
    print(f"{'─'*60}")

    shown = 0
    for r in all_results:
        if shown >= 5:
            break
        base = r["conditions"]["baseline"]
        # Check if any FFN intervention changed correctness
        for cond_name, cond in r["conditions"].items():
            if cond["type"] in ("ffn_fixed", "ffn_aware") and \
               cond["correct"] != base["correct"]:
                status = "FIXED" if cond["correct"] else "BROKE"
                print(f"\n  [{status}] {r['question'][:80]}...")
                print(f"    True: {r['true_label']}, "
                      f"Base: {'✓' if base['correct'] else '✗'} ({base['answer'][:60]}...)")
                print(f"    {cond_name}: {'✓' if cond['correct'] else '✗'} "
                      f"({cond['answer'][:60]}...)")
                shown += 1
                break

    if shown == 0:
        print("  No samples changed by FFN intervention.")

    # ── Save ──
    save_path = output_dir / "phase9_intervention.json"
    with open(save_path, "w") as f:
        json.dump({
            "n_test": n_test,
            "n_train": len(train_idx),
            "layer": args.layer,
            "alphas": alphas,
            "baseline_correct": baseline_correct,
            "baseline_rate": baseline_correct / n_test,
            "condition_summary": {
                ctype: {"correct": ct["correct"], "total": ct["total"],
                        "rate": ct["correct"] / ct["total"]}
                for ctype, ct in condition_types.items()
            },
            "per_sample": all_results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    # ── Final verdict ──
    print(f"\n{'='*60}")
    best_ffn_rate = baseline_correct / n_test
    best_ffn_name = "baseline"
    for ctype, ct in condition_types.items():
        if "ffn" in ctype and ct["total"] == n_test:
            rate = ct["correct"] / ct["total"]
            if rate > best_ffn_rate:
                best_ffn_rate = rate
                best_ffn_name = ctype

    delta_best = best_ffn_rate - baseline_correct / n_test
    if delta_best > 0:
        print(f"  Best FFN intervention: {best_ffn_name} ({best_ffn_rate:.1%}, Δ={delta_best:+.1%})")
        print(f"  ✅ FFN intervention shows improvement!")
    else:
        print(f"  All FFN interventions ≤ baseline ({baseline_correct/n_test:.1%})")
        print(f"  ❌ No FFN intervention effect detected")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
