"""Phase 16.3: ROME-style Rank-1 FFN Weight Edit.

ROME (Meng et al. 2022) edits factual knowledge via rank-1 weight update.
We adapt: push FFN output weights to amplify the "truth direction" signal.

Key difference: modifies WEIGHTS (permanent change to computation),
not transient activation perturbations that can be compensated.

Mechanism:
  W_out += λ · outer(k, c) / ||k||^2
  where c = truth direction (output, d_model=2048)
        k = mean MLP intermediate activation (d_mlp=6144, from mlp.hook_post)

Effect: for any input x, output shifts along c proportionally to k·x.

Usage:
  python phase16_rome.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract.json \
    --n_test 50
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


# ═══════════════════════════════════════════════════════════════════
# Core
# ═══════════════════════════════════════════════════════════════════


def get_truth_direction(records, layer: int) -> np.ndarray:
    """Truth direction from stored h states: c = mean(correct) - mean(wrong)."""
    correct, wrong = [], []
    for r in records:
        vec = np.array(r["h"][str(layer)], dtype=np.float32)
        if r["label"] == 1:
            correct.append(vec)
        else:
            wrong.append(vec)
    v = np.mean(correct, axis=0) - np.mean(wrong, axis=0)
    return v / (np.linalg.norm(v) + 1e-8)


def get_mlp_intermediate_mean(
    model, records, layer: int, device: str, max_samples: int = 50
) -> torch.Tensor:
    """Extract mean MLP intermediate (d_mlp=6144) at last token position.

    Uses mlp.hook_post which fires after activation (SiLU(gate)*up),
    before the W_out projection.
    """
    hook_name = f"blocks.{layer}.mlp.hook_post"
    all_k = []

    for rec in tqdm(records[:max_samples], desc=f"  Extracting MLP intern L{layer}"):
        question = rec["question"]
        context = rec.get("context", "")
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # Use a local list captured by the hook
        result = []

        def _capture(act, hook):
            result.append(act[0, -1, :].clone())
            return act

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])

        if result:
            all_k.append(result[0])

    return torch.stack(all_k, dim=0).mean(dim=0)  # [d_mlp]


@contextmanager
def temporary_W_out_edit(model, layer: int, delta_W: torch.Tensor):
    """Temporarily edit blocks.{layer}.mlp.W_out, restoring after exit."""
    weight = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    original = weight.data.clone()
    weight.data = weight.data + delta_W.to(weight.device).to(weight.dtype)
    try:
        yield
    finally:
        weight.data = original


def _gen_greedy(model, tokenizer, tokens, device, max_new=20):
    """Simple greedy generation."""
    gids = []
    for _step in range(max_new):
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break
    return tokenizer.decode(gids).strip()


def evaluate_generation(model, tokenizer, test_records, device):
    """Evaluate generation accuracy on test set."""
    correct = 0
    for rec in test_records:
        question = rec["question"]
        context = rec.get("context", "")
        gt_answers = rec["gt_answers"]
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        generated = _gen_greedy(model, tokenizer, tokens, device)
        if check_correct(generated, gt_answers, dataset="triviaqa"):
            correct += 1
    return correct / len(test_records)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 16 ROME: Rank-1 FFN Weight Edit"
    )
    parser.add_argument("--load", required=True)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, nargs="*", default=[20])
    parser.add_argument(
        "--lambdas", type=float, nargs="*", default=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    with open(args.load) as f:
        data = json.load(f)
    all_records = data["records"]

    n_test = min(args.n_test, len(all_records) // 2)
    n_train = len(all_records) - n_test
    train_records = all_records[:n_train]
    test_records = all_records[n_train:]
    print(f"  Train: {n_train}, Test: {n_test}")

    # ── Load model ────────────────────────────────────────────
    print("\n[2/3] Loading model...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    tokenizer = model.tokenizer
    d_model = model.cfg.d_model
    d_mlp = model.blocks[0].mlp.W_out.shape[0]
    print(f"  Model: {model.cfg.model_name}, d_model={d_model}, d_mlp={d_mlp}")

    # ── Baseline ──────────────────────────────────────────────
    print("\n[3/3] ROME rank-1 weight edit...")
    print("  Computing baseline...")
    baseline_rate = evaluate_generation(model, tokenizer, test_records, device)
    print(f"  Baseline: {baseline_rate:.1%}")

    results = {"baseline_rate": baseline_rate, "layers": {}}

    for layer in args.layers:
        print(f"\n  ── Layer {layer} ──")

        # c: truth direction (d_model=2048)
        c_np = get_truth_direction(train_records, layer)
        c = torch.tensor(c_np, dtype=torch.float32, device=device)
        print(f"    ||c|| = {torch.norm(c):.4f}")

        # k: mean MLP intermediate (d_mlp=6144) from train subset
        k = get_mlp_intermediate_mean(
            model, train_records, layer, device, max_samples=50
        )
        k_norm_sq = torch.dot(k, k)
        print(f"    ||k|| = {torch.sqrt(k_norm_sq):.1f}")

        layer_results = {}
        best_rate = baseline_rate
        best_lambda = 0.0

        for lam in args.lambdas:
            # ΔW = λ · outer(k, c) / ||k||²
            # outer(k, c): [d_mlp] ⊗ [d_model] = [d_mlp, d_model]
            delta_W = lam * torch.outer(k, c) / k_norm_sq

            with temporary_W_out_edit(model, layer, delta_W):
                rate = evaluate_generation(model, tokenizer, test_records, device)

            delta = rate - baseline_rate
            key = f"λ{lam:+.1f}"
            layer_results[key] = {"rate": rate, "delta": float(delta)}
            print(f"    {key}: {rate:.1%} (Δ={delta:+.1%})")

            if rate > best_rate:
                best_rate = rate
                best_lambda = lam

        results["layers"][str(layer)] = {
            "best_rate": best_rate,
            "best_lambda": best_lambda,
            "configs": layer_results,
            "c_norm": float(torch.norm(c)),
            "k_norm": float(torch.sqrt(k_norm_sq)),
        }
        print(
            f"    L{layer} best: λ={best_lambda:+.1f} @ {best_rate:.1%} (Δ={best_rate - baseline_rate:+.1%})"
        )

    # ── Summary ───────────────────────────────────────────────
    overall_best = max(
        (lr["best_rate"], int(layer), lr["best_lambda"])
        for layer, lr in results["layers"].items()
    )
    print(f"\n{'=' * 60}")
    print("ROME RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline: {baseline_rate:.1%}")
    print(
        f"  Best: L{overall_best[1]} λ={overall_best[2]:+.1f} @ {overall_best[0]:.1%}"
    )
    print(f"  Δ = {overall_best[0] - baseline_rate:+.1%}")

    if overall_best[0] <= baseline_rate:
        print("  ⚠ ROME zero effect — even weight editing doesn't help")

    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "outputs_phase16"
    output_dir.mkdir(exist_ok=True)

    results["summary"] = {
        "baseline_rate": baseline_rate,
        "best_rate": overall_best[0],
        "best_layer": overall_best[1],
        "best_lambda": overall_best[2],
    }

    output_path = output_dir / "phase16_rome_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
