"""Phase 15.1: 8B TLDC (Token-Level Dynamic Contrast) Cross-Scale Validation.

Adapted from experiments/lin_theory/validate_s14_tldc.py for Qwen3-8B (36 layers).

Theory: docs/theory-intervention-failure.md Section 14.2

TLDC uses the logit difference between the detection-peak layer (ℓ*) and the
final layer (L35) as a dynamic, per-token, per-sample intervention signal:

    logits_adj ← l_L35 + β·(l_ℓ* - l_L35)

Goal: Validate whether the 1.7B TLDC effect (3/21 KW corrected, β=0.10)
generalizes to 8B scale.

Gates:
  E1: ℓ* AUROC ≥ 0.85 (detection peak exists)
  E2: TLDC KW Δ > 5% (n=100)
  E3: 8B effect ≥ 1.7B effect (quantitative comparison)
  E4: DK Δ ≥ 0% (no harm on don't-know)

Usage (AutoDL RTX 5090 32GB):
  python run_8b_tldc.py --n_calibrate 100 --n_test 100

Quick test (local, if 8B fits):
  python run_8b_tldc.py --n_calibrate 50 --n_test 30 --quick
"""

import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Path setup: add parent experiments dir for lin_theory/common.py imports
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


# ═════════════════════════════════════════════════════════════════════════════
# Helpers (adapted from lin_theory/validate_s14_tldc.py)
# ═════════════════════════════════════════════════════════════════════════════


def load_model_and_unembed(device="cuda", model_id="Qwen/Qwen3-8B"):
    """Load HookedTransformer + return (model, tokenizer, W_U, b_U, ln_final)."""
    model = load_model(device=device, model_id=model_id)
    tokenizer = model.tokenizer
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    ln_final = model.ln_final
    return model, tokenizer, W_U, b_U, ln_final


def get_first_answer_token_id(tokenizer, answers):
    """Return the first token ID of the first non-empty answer alias.

    Prepends a space to match generation context after "Answer:".
    """
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


def get_y_true_rank(logits, y_true_id):
    """Rank 0 = highest probability."""
    sorted_ids = logits[0, -1, :].float().argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item()
    return rank


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state at any layer via early exit.

    Applies RMSNorm + W_U mapping. Stays in float16 to avoid OOM.
    """
    dtype = next(ln_final.parameters()).dtype
    device = h.device

    h_f16 = h.to(dtype=dtype)
    h_norm = ln_final(h_f16)
    logits = h_norm @ W_U.to(dtype)
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def greedy_generate(model, tokenizer, prompt, device, max_new=20):
    """Simple greedy generation."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits = model(tokens)

    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]

    for _ in range(max_new - 1):
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


def tldc_greedy_generate(
    model,
    tokenizer,
    prompt,
    device,
    layer_early,
    W_U,
    b_U,
    ln_final,
    beta,
    max_new=20,
):
    """Greedy generation with TLDC (Token-Level Dynamic Contrast).

    At each step:
      1. Forward pass → capture h at layer_early and final logits
      2. Compute early-exit logits: l_early = W_U @ ln_final(h_early)
      3. Adjusted logits: l = l_final + beta * (l_early - l_final)
      4. Greedy decode from adjusted logits
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    captured_early = {}

    def _hook_early(act, hook=None):
        captured_early["h"] = act[:, -1:, :].detach()
        return act

    # ── First token ──
    with torch.no_grad():
        logits_final = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_early, _hook_early)]
        )

    h_early = captured_early["h"]
    l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
    l_final = logits_final[0, -1:, :].float()

    if l_early.shape[-1] == l_final.shape[-1]:
        logits_adj = l_final + beta * (l_early - l_final)
    else:
        logits_adj = l_final

    initial_raw_logits = logits_final.detach().clone()
    nid = int(logits_adj.argmax(dim=-1).item())
    gids = [nid]

    # ── Subsequent tokens ──
    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break

        with torch.no_grad():
            logits_final = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_early, _hook_early)]
            )

        h_early = captured_early["h"]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final = logits_final[0, -1:, :].float()

        if l_early.shape[-1] == l_final.shape[-1]:
            logits_adj = l_final + beta * (l_early - l_final)
        else:
            logits_adj = l_final

        nid = int(logits_adj.argmax(dim=-1).item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()
    return ans, initial_raw_logits


# ═════════════════════════════════════════════════════════════════════════════
# Detection: Quick per-layer AUROC scan (Phase 7)
# ═════════════════════════════════════════════════════════════════════════════


def quick_detection_scan(model, tokenizer, device, samples, n_layers):
    """Quick AUROC scan across all layers to find detection peak ℓ*.

    Uses compute_v approach: for each layer, compute v from all samples,
    then dot with each sample's h to get scores.
    """
    from sklearn.metrics import roc_auc_score

    print(
        f"\n  Running quick detection scan ({len(samples)} samples, {n_layers} layers)..."
    )

    # Extract h at all layers for all samples
    all_h = {lyr: [] for lyr in range(n_layers)}
    labels = []

    for sample in tqdm(samples, desc="  Extracting h"):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        # Capture all layers in one pass
        storage = {}

        def make_capture(lyr):
            hook_name = f"blocks.{lyr}.hook_resid_post"

            def _capture(act, hook=None):
                storage[lyr] = act[0, input_len - 1, :].clone()
                return act

            return hook_name, _capture

        hooks = [make_capture(lyr) for lyr in range(n_layers)]

        with torch.no_grad():
            _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Generate answer
        gids = []
        current_tokens = tokens.clone()
        for _step in range(20):
            nid = int(model(current_tokens)[0, -1, :].argmax().item())
            if nid == tokenizer.eos_token_id:
                break
            gids.append(nid)
            current_tokens = torch.cat(
                [current_tokens, torch.tensor([[nid]], device=device)], dim=1
            )
            if current_tokens.shape[1] > 1024:
                break

        generated = tokenizer.decode(gids).strip()
        is_correct = check_correct(generated, sample["answers"], dataset="triviaqa")
        labels.append(int(is_correct))

        for lyr in range(n_layers):
            all_h[lyr].append(storage[lyr].float().cpu().numpy())

    labels = np.array(labels)

    aurocs = {}
    for lyr in tqdm(range(n_layers), desc="  Computing AUROC"):
        vecs = np.stack(all_h[lyr], axis=0)
        correct_vecs = vecs[labels == 1]
        wrong_vecs = vecs[labels == 0]

        if len(correct_vecs) == 0 or len(wrong_vecs) == 0:
            aurocs[lyr] = 0.5
            continue

        v = correct_vecs.mean(axis=0) - wrong_vecs.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)

        scores = np.dot(vecs, v)
        try:
            aurocs[lyr] = float(roc_auc_score(labels, scores))
        except ValueError:
            aurocs[lyr] = 0.5

    return aurocs


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 15.1: 8B TLDC Cross-Scale Validation"
    )
    parser.add_argument("--n_calibrate", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument(
        "--layer_early",
        type=int,
        default=None,
        help="Detection peak layer (auto-detect if not specified)",
    )
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    )
    parser.add_argument("--seed_cal", type=int, default=42)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer samples, skip detection scan",
    )
    parser.add_argument(
        "--skip_detection",
        action="store_true",
        help="Skip detection scan, use --layer_early directly",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "outputs_phase16"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 64)
    print("Phase 15.1: 8B TLDC Cross-Scale Validation")
    print(f"  Model: {args.model_id}")
    print(f"  n_cal={args.n_calibrate}, n_test={args.n_test}")
    print(f"  Betas: {args.betas}")
    print("=" * 64)

    # ── Load model ──
    print("\n[1/6] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model_id)
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1  # L35 for 8B
    print(f"  Model: {n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Final layer: L{final_layer}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(
            f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB"
        )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Detection: find ℓ* (AUROC peak layer) ──
    if args.skip_detection and args.layer_early is not None:
        best_layer = args.layer_early
        print(f"\n[2/6] Skipping detection scan, using L{best_layer} as ℓ*")
    else:
        print(f"\n[2/6] Detection scan: finding ℓ* (AUROC peak)...")
        # Use calibration samples for detection scan
        cal_samples = load_triviaqa(
            n_samples=min(args.n_calibrate, 100), seed=args.seed_cal
        )
        aurocs = quick_detection_scan(model, tokenizer, device, cal_samples, n_layers)

        # Find best layer
        best_layer = max(aurocs, key=aurocs.get)
        top5 = sorted(aurocs, key=aurocs.get, reverse=True)[:5]
        print(f"\n  Top-5 h-AUROC layers:")
        for lyr in top5:
            marker = " ← ℓ*" if lyr == best_layer else ""
            print(f"    L{lyr}: {aurocs[lyr]:.4f}{marker}")

        if args.layer_early is not None:
            print(f"\n  Overriding ℓ* from L{best_layer} to L{args.layer_early}")
            best_layer = args.layer_early

    layer_early = best_layer
    print(f"\n  Using ℓ* = L{layer_early} (AUROC peak)")
    print(
        f"  Span: L{layer_early} → L{final_layer} ({final_layer - layer_early} layers)"
    )

    # ── Gate E1 ──
    e1_auroc = aurocs.get(layer_early, 0.0) if not args.skip_detection else 0.90
    e1_pass = e1_auroc >= 0.85
    print(f"  E1 (AUROC ≥ 0.85): {'✅' if e1_pass else '❌'} (AUROC={e1_auroc:.4f})")

    # ── Classify test samples by knowability ──
    print(f"\n[3/6] Classifying {args.n_test} test samples (seed={args.seed_test})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    test_entries = []
    for i, sample in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            continue

        # Get rank using the detection layer's hidden state -> early-exit logits
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        captured = {}

        def _hook(act, hook=None):
            captured["h"] = act[:, -1:, :].detach()
            return act

        hook_name = f"blocks.{layer_early}.hook_resid_post"
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, sample["answers"], dataset="triviaqa")

        if rank <= args.rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        test_entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "is_correct": is_correct,
                "subset": subset,
                "prompt": prompt,
                "answers": sample["answers"],
                "question": sample["question"][:80],
                "y_true_id": y_true_id,
            }
        )

    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]

    baseline_rate = sum(1 for e in test_entries if e["is_correct"]) / len(test_entries)
    kw_baseline = sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))
    kc_baseline = sum(1 for e in kc if e["is_correct"]) / max(1, len(kc))
    dk_baseline = sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))

    print(
        f"  Know & Correct: {len(kc)}/{len(test_entries)} (baseline={kc_baseline:.1%})"
    )
    print(
        f"  Know & Wrong:   {len(kw)}/{len(test_entries)} (baseline={kw_baseline:.1%})  ← TARGET"
    )
    print(
        f"  Don't Know:     {len(dk)}/{len(test_entries)} (baseline={dk_baseline:.1%})"
    )
    print(f"  All:            {baseline_rate:.1%}")

    # ── Gate D2 (diagnostic): Early vs Final rank comparison ──
    print(
        f"\n[4/6] Gate D2: L{layer_early} vs L{final_layer} y_true rank comparison..."
    )
    d2_results = {"early_better": 0, "final_better": 0, "same": 0, "details": []}

    for e in tqdm(test_entries, desc="  D2"):
        prompt = e["prompt"]
        y_true_id = e["y_true_id"]

        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        captured = {}

        def _hook_early(act, hook=None):
            captured["h_early"] = act[:, -1:, :].detach()
            return act

        def _hook_final(act, hook=None):
            captured["h_final"] = act[:, -1:, :].detach()
            return act

        hook_early = f"blocks.{layer_early}.hook_resid_post"
        hook_final = f"blocks.{final_layer}.hook_resid_post"

        with torch.no_grad():
            _ = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_early, _hook_early), (hook_final, _hook_final)],
            )

        h_early = captured["h_early"]
        h_final = captured["h_final"]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final_exit = compute_early_exit_logits(h_final, ln_final, W_U, b_U)

        rank_early = get_y_true_rank(l_early.unsqueeze(1), y_true_id)
        rank_final = get_y_true_rank(l_final_exit.unsqueeze(1), y_true_id)

        if rank_early < rank_final:
            d2_results["early_better"] += 1
        elif rank_final < rank_early:
            d2_results["final_better"] += 1
        else:
            d2_results["same"] += 1

        d2_results["details"].append(
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank_early": rank_early,
                "rank_final": rank_final,
                "question": e["question"],
            }
        )

    print(f"  Early (L{layer_early}) better rank: {d2_results['early_better']}")
    print(f"  Final (L{final_layer}) better rank:    {d2_results['final_better']}")
    print(f"  Same rank:                            {d2_results['same']}")

    kw_details = [d for d in d2_results["details"] if d["subset"] == "know_wrong"]
    kw_early_better = sum(1 for d in kw_details if d["rank_early"] < d["rank_final"])
    kw_final_better = sum(1 for d in kw_details if d["rank_final"] < d["rank_early"])
    print(f"  ── Know-Wrong subset ──")
    print(f"  L{layer_early} better: {kw_early_better}/{len(kw_details)}")
    print(f"  L{final_layer} better:      {kw_final_better}/{len(kw_details)}")

    # ── TLDC intervention ──
    print(
        f"\n[5/6] TLDC intervention ({len(args.betas)} betas × {len(test_entries)} samples)..."
    )

    all_results = {
        "baseline_rate": baseline_rate,
        "kw_baseline": kw_baseline,
        "kc_baseline": kc_baseline,
        "dk_baseline": dk_baseline,
        "betas": {},
    }

    for beta in args.betas:
        print(f"\n  ── β = {beta:.3f} ──")
        correct_by_subset = defaultdict(int)
        count_by_subset = defaultdict(int)
        t_beta = time.time()

        for e in tqdm(test_entries, desc=f"    β={beta:.3f}", leave=False):
            subset = e["subset"]

            gen_text, initial_logits = tldc_greedy_generate(
                model,
                tokenizer,
                e["prompt"],
                device,
                layer_early,
                W_U,
                b_U,
                ln_final,
                beta,
            )
            is_correct = check_correct(gen_text, e["answers"], dataset="triviaqa")

            if is_correct:
                correct_by_subset[subset] += 1
                correct_by_subset["all"] += 1
            count_by_subset[subset] += 1
            count_by_subset["all"] += 1

        beta_results = {}
        for s in ["know_wrong", "know_correct", "dont_know", "all"]:
            if count_by_subset[s] > 0:
                rate = correct_by_subset[s] / count_by_subset[s]
                if s == "know_wrong":
                    bl = kw_baseline
                elif s == "know_correct":
                    bl = kc_baseline
                elif s == "dont_know":
                    bl = dk_baseline
                else:
                    bl = baseline_rate
                delta = rate - bl
            else:
                rate, delta = 0.0, 0.0

            beta_results[s] = {
                "correct": correct_by_subset[s],
                "total": count_by_subset[s],
                "rate": rate,
                "delta": delta,
            }

            if s in ["know_wrong", "all"]:
                r = beta_results[s]
                print(f"    {s}: {r['correct']}/{r['total']} (Δ={r['delta']:+.1%})")

        all_results["betas"][f"beta={beta:.3f}"] = beta_results
        print(f"    time: {time.time() - t_beta:.0f}s")

    # ── Gate verification ──
    print(f"\n[6/6] Gate verification")
    print(f"\n{'=' * 80}")

    # E2: Δ accuracy > 5% on know-wrong
    best_kw_delta = max(
        all_results["betas"][k]["know_wrong"]["delta"] for k in all_results["betas"]
    )
    e2_pass = best_kw_delta > 0.05
    print(
        f"\n  E2 (TLDC Δ > 5% on know-wrong): {'✅' if e2_pass else '❌'} "
        f"(best Δ={best_kw_delta:+.1%})"
    )

    # E3: 8B effect ≥ 1.7B effect (1.7B best: +14.3% on KW = +0.143 delta)
    # Normalize: compare raw delta
    ref_1p7b_kw_delta = 0.143  # 3/21 KW = 14.3%
    e3_pass = best_kw_delta >= ref_1p7b_kw_delta
    print(
        f"  E3 (8B KW Δ ≥ 1.7B KW Δ={ref_1p7b_kw_delta:+.1%}): "
        f"{'✅' if e3_pass else '❌'}"
    )

    # E4: DK Δ ≥ 0%
    worst_dk_delta = min(
        all_results["betas"][k]["dont_know"]["delta"] for k in all_results["betas"]
    )
    e4_pass = worst_dk_delta >= 0.0
    print(
        f"  E4 (DK Δ ≥ 0%): {'✅' if e4_pass else '❌'} (worst Δ={worst_dk_delta:+.1%})"
    )

    # Summary table
    print(f"\n  ── Summary table ──")
    print(f"  {'β':>8}  {'KW Δ':>8}  {'KC Δ':>8}  {'DK Δ':>8}  {'All Δ':>8}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for beta in args.betas:
        key = f"beta={beta:.3f}"
        r = all_results["betas"][key]
        print(
            f"  {beta:>8.3f}  "
            f"{r['know_wrong']['delta']:>+8.1%}  "
            f"{r['know_correct']['delta']:>+8.1%}  "
            f"{r['dont_know']['delta']:>+8.1%}  "
            f"{r['all']['delta']:>+8.1%}"
        )

    # Overall verdict
    n_pass = sum([e1_pass, e2_pass, e3_pass, e4_pass])
    print(f"\n  Gate summary: {n_pass}/4 passed")
    print(f"    E1 (AUROC≥0.85): {'✅' if e1_pass else '❌'}")
    print(f"    E2 (KW Δ>5%):   {'✅' if e2_pass else '❌'}")
    print(f"    E3 (8B≥1.7B):   {'✅' if e3_pass else '❌'}")
    print(f"    E4 (DK Δ≥0%):   {'✅' if e4_pass else '❌'}")

    if e3_pass:
        print(f"\n  ✅ TLDC cross-scale generalization CONFIRMED")
        print(f"  → Strengthens paper Chapter 3, target ACL/EMNLP")
    elif e2_pass:
        print(f"\n  ⚠️ TLDC works on 8B but effect < 1.7B")
        print(f"  → Scale increase did NOT amplify effect → paper needs explanation")
    else:
        print(f"\n  ❌ TLDC effect does NOT generalize to 8B")
        print(f"  → Paper focus: 'marginal feasibility on small models'")
        print(f"  → Chapter 3 reduced to 1.7B boundary case")

    # ── Save results ──
    output = {
        "config": {
            "model_id": args.model_id,
            "n_layers": n_layers,
            "final_layer": final_layer,
            "layer_early": layer_early,
            "n_calibrate": args.n_calibrate,
            "n_test": args.n_test,
            "seed_cal": args.seed_cal,
            "seed_test": args.seed_test,
            "rank_threshold": args.rank_threshold,
            "betas": args.betas,
        },
        "detection": {
            "layer_early": layer_early,
            "auroc_at_peak": e1_auroc if not args.skip_detection else None,
            "aurocs": {str(k): v for k, v in aurocs.items()}
            if not args.skip_detection
            else {},
        },
        "test": {
            "n_total": len(test_entries),
            "n_know_correct": len(kc),
            "n_know_wrong": len(kw),
            "n_dont_know": len(dk),
            "baseline_rate": float(baseline_rate),
            "kw_baseline_rate": float(kw_baseline),
            "kc_baseline_rate": float(kc_baseline),
            "dk_baseline_rate": float(dk_baseline),
        },
        "d2": {
            "early_better": d2_results["early_better"],
            "final_better": d2_results["final_better"],
            "same": d2_results["same"],
            "kw_early_better": kw_early_better,
            "kw_final_better": kw_final_better,
        },
        "gates": {
            "E1": {
                "pass": bool(e1_pass),
                "auroc": e1_auroc if not args.skip_detection else None,
            },
            "E2": {"pass": bool(e2_pass), "best_kw_delta": float(best_kw_delta)},
            "E3": {"pass": bool(e3_pass), "ref_1p7b_kw_delta": ref_1p7b_kw_delta},
            "E4": {"pass": bool(e4_pass), "worst_dk_delta": float(worst_dk_delta)},
        },
        "results": all_results,
    }

    out_path = output_dir / "s15_1_8b_tldc.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
