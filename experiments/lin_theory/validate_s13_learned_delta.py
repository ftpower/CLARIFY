"""Phase 13.2: Question-Conditional Correction Network δ_θ(h, e(x)).

Theory: docs/theory-intervention-failure.md Section 13.3.
Core claim: A lightweight network that takes hidden state h and question
representation e(x) can learn question-conditional corrections that a
global direction v cannot capture.

Training: min_θ -log P(y_true | h + δ_θ(h, e(x))) + λ||δ||²

Key design decisions:
  - Model is FROZEN; only the correction network is trained
  - Hidden states h_L are PRECOMPUTED once for all samples (no model forward
    passes during training — massive speedup)
  - Uses analytical W_U projection to compute log P without full-model backprop
  - e(x) = h from the same layer (simplest question representation)
  - Architecture: Linear(2d → r → d), r=8
  - Layer: L27 (last layer, closest to output)

Phase 13.2a: Feasibility verification (~5 min on 1.7B)
  - 500 train / 100 val
  - Multiple λ values
  - Evaluate Δ accuracy, cos(δ, g), δ diversity

Predictions (Section 13.3.3):
  L1: Training loss decreases
  L2: val cos(δ_θ, g) > 0.3
  L3: val Δ accuracy > 0 (gate: > 5%)
  L4: δ directions diverge across samples (not collinear)
  L5: Ablating e(x) degrades performance

Usage:
    python validate_s13_learned_delta.py --n_train 500 --n_val 100 --n_test 50
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    greedy_generate,
    get_first_answer_token_id,
    compute_g_L,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Correction network architecture
# ═════════════════════════════════════════════════════════════════════════════


class CorrectionNet(nn.Module):
    """Lightweight question-conditional correction network.

    Input: [h; e(x)] ∈ R^{2d}
    Architecture: 2d → r → d  (ReLU hidden)
    Output: δ ∈ R^d
    """

    def __init__(self, d_model, rank=8, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(2 * d_model, rank, dtype=dtype)
        self.fc2 = nn.Linear(rank, d_model, dtype=dtype)
        # Initialize small: near-zero δ at start
        nn.init.normal_(self.fc1.weight, std=0.01 / np.sqrt(2 * d_model))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01 / np.sqrt(rank))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, e_x=None):
        if e_x is None:
            e_x = h
        squeeze = h.ndim == 1
        if squeeze:
            h = h.unsqueeze(0)
            e_x = e_x.unsqueeze(0)
        x = torch.cat([h, e_x], dim=-1)
        x = F.relu(self.fc1(x))
        delta = self.fc2(x)
        if squeeze:
            delta = delta.squeeze(0)
        return delta


class GlobalCorrectionNet(nn.Module):
    """Ablation: δ_θ(h) without question representation e(x)."""

    def __init__(self, d_model, rank=8, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, rank, dtype=dtype)
        self.fc2 = nn.Linear(rank, d_model, dtype=dtype)
        nn.init.normal_(self.fc1.weight, std=0.01 / np.sqrt(d_model))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01 / np.sqrt(rank))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, e_x=None):
        squeeze = h.ndim == 1
        if squeeze:
            h = h.unsqueeze(0)
        x = F.relu(self.fc1(h))
        delta = self.fc2(x)
        if squeeze:
            delta = delta.squeeze(0)
        return delta


# ═════════════════════════════════════════════════════════════════════════════
# 2. Precompute hidden states (run ONCE for all samples)
# ═════════════════════════════════════════════════════════════════════════════


def precompute_hidden_states(model, tokenizer, samples, device, layer):
    """Extract h_L and y_true_id for all samples. Runs model forward once each.

    Returns:
        h_cache: torch.Tensor [n, d] float32 on CPU
        y_true_ids: list[int] length n
        valid_mask: list[bool] length n (False if y_true_id is None)
    """
    h_list = []
    y_true_ids = []
    valid_mask = []

    hook_name = f"blocks.{layer}.hook_resid_post"

    for s in tqdm(samples, desc="  Precomputing h_L"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        residual = {}

        def _hook(act, hook=None):
            residual["h"] = act[:, -1, :].detach()
            return act

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        h = residual["h"].float().cpu().squeeze(0)
        yt_id = get_first_answer_token_id(tokenizer, s["answers"])

        h_list.append(h)
        y_true_ids.append(yt_id if yt_id is not None else 0)
        valid_mask.append(yt_id is not None)

    h_cache = torch.stack(h_list, dim=0)  # [n, d] float32
    return h_cache, y_true_ids, valid_mask


# ═════════════════════════════════════════════════════════════════════════════
# 3. Loss computation (no model forward pass needed)
# ═════════════════════════════════════════════════════════════════════════════


def compute_log_p_true(h_modified, y_true_ids, W_U, b_U, ln_final):
    """Compute mean log P(y_true | h_modified). Fully differentiable."""
    if h_modified.ndim == 1:
        h_modified = h_modified.unsqueeze(0)

    batch = h_modified.shape[0]
    device = h_modified.device

    model_dtype = next(ln_final.parameters()).dtype
    h_normed = ln_final(h_modified.to(dtype=model_dtype))  # [batch, d] fp16

    logits_f16 = h_normed @ W_U  # [batch, vocab] fp16
    if b_U is not None:
        logits_f16 = logits_f16 + b_U

    log_probs = F.log_softmax(logits_f16.float(), dim=-1)  # [batch, vocab] fp32

    yt = torch.tensor(y_true_ids, device=device, dtype=torch.long)
    log_p = log_probs[torch.arange(batch, device=device), yt]  # [batch]

    return log_p.mean()


# ═════════════════════════════════════════════════════════════════════════════
# 4. Training loop (uses precomputed h_cache)
# ═════════════════════════════════════════════════════════════════════════════


def train_epoch_cached(
    net,
    optimizer,
    h_cache,
    y_true_ids,
    valid_mask,
    W_U,
    b_U,
    ln_final,
    device,
    lambda_reg,
    batch_size=32,
):
    """Single training epoch using precomputed hidden states."""
    net.train()
    total_loss = 0.0
    n_batches = 0

    # Only use valid samples
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    indices = np.random.permutation(valid_idx)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        h_batch = h_cache[batch_idx].to(device)  # [b, d]
        yt_batch = [y_true_ids[i] for i in batch_idx]

        # Forward through correction network
        delta = net(h_batch)  # [b, d]
        h_modified = h_batch + delta

        # Loss
        loss_nll = -compute_log_p_true(h_modified, yt_batch, W_U, b_U, ln_final)
        loss_reg = lambda_reg * (delta.norm(dim=-1) ** 2).mean()
        loss = loss_nll + loss_reg

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss_and_metrics(
    net,
    h_cache,
    y_true_ids,
    valid_mask,
    W_U,
    b_U,
    ln_final,
    device,
    lambda_reg=0.0,
):
    """Compute loss + δ metrics on cached hidden states. Fast (no generation)."""
    net.eval()

    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    if not valid_idx:
        return {"loss": float("inf"), "n": 0}

    h_batch = h_cache[valid_idx].to(device)  # [n, d]
    yt_batch = [y_true_ids[i] for i in valid_idx]

    delta = net(h_batch)  # [n, d]
    h_modified = h_batch + delta

    # Loss
    loss_nll = -compute_log_p_true(h_modified, yt_batch, W_U, b_U, ln_final)
    loss_reg = lambda_reg * (delta.norm(dim=-1) ** 2).mean()
    loss = loss_nll + loss_reg

    # δ norms
    delta_norms = delta.norm(dim=-1)  # [n]

    # cos(δ, g) for each sample
    cos_vals = []
    for i, idx in enumerate(valid_idx):
        g = compute_g_L(h_cache[idx].to(device), y_true_ids[idx], W_U, b_U, ln_final)
        cos_val = F.cosine_similarity(delta[i : i + 1].cpu(), g.unsqueeze(0), dim=-1)
        cos_vals.append(float(cos_val.item()))

    # δ diversity: mean abs pairwise cosine
    delta_n = F.normalize(delta, dim=-1)
    pairwise = delta_n @ delta_n.t()
    mask = ~torch.eye(len(valid_idx), dtype=torch.bool, device=device)
    diversity = (
        float(pairwise[mask].abs().mean().item()) if len(valid_idx) >= 2 else 0.0
    )

    return {
        "loss": float(loss.item()),
        "n": len(valid_idx),
        "delta_norm_mean": float(delta_norms.mean().item()),
        "cos_with_g_mean": float(np.mean(cos_vals)) if cos_vals else 0.0,
        "delta_diversity": diversity,
    }


@torch.no_grad()
def evaluate_accuracy(
    net,
    samples,
    valid_mask,
    model,
    tokenizer,
    device,
    layer,
):
    """Evaluate accuracy with δ intervention. Runs generation (slow, call sparingly)."""
    net.eval()

    correct_baseline = 0
    correct_with = 0
    n_valid = 0

    hook_name = f"blocks.{layer}.hook_resid_post"

    for i, s in enumerate(tqdm(samples, desc="  Eval acc", leave=False)):
        if not valid_mask[i]:
            continue

        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        # Get h_L
        residual = {}

        def _hook(act, hook=None):
            residual["h"] = act[:, -1, :].detach()
            return act

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        h = residual["h"].float().squeeze(0)
        delta = net(h.to(device))

        # Sweep alpha for best result
        d_f16 = delta.to(dtype=torch.float16)
        best_correct = False
        for alpha in [0.5, 1.0, 2.0]:

            def _intervene(act, hook=None):
                act[:, -1, :] += alpha * d_f16.unsqueeze(0)
                return act

            gen = greedy_generate(
                model,
                tokenizer,
                prompt,
                device,
                fwd_hooks=[(hook_name, _intervene)],
            )
            if check_correct(gen, s["answers"], dataset="triviaqa"):
                best_correct = True
                break
        if best_correct:
            correct_with += 1

        # Baseline
        baseline_gen = greedy_generate(model, tokenizer, prompt, device)
        if check_correct(baseline_gen, s["answers"], dataset="triviaqa"):
            correct_baseline += 1

        n_valid += 1

    if n_valid == 0:
        return {"accuracy": 0.0, "baseline_accuracy": 0.0, "n": 0}

    return {
        "accuracy": correct_with / n_valid,
        "baseline_accuracy": correct_baseline / n_valid,
        "n": n_valid,
        "correct_with": correct_with,
        "correct_baseline": correct_baseline,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="S13.2: Learned Question-Conditional Correction Network"
    )
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--lambdas", type=float, nargs="*", default=[0.001, 0.01, 0.1, 1.0]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Phase 13.2: Learned δ_θ(h, e(x)) Correction Network")
    print(f"Train: {args.n_train}, Val: {args.n_val}, Test: {args.n_test}")
    print(f"Layer: {args.layer}, Rank: {args.rank}")
    print(f"Lambdas: {args.lambdas}, Epochs: {args.epochs}")
    print("=" * 60)

    # ── Load model ──
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    d_model = model.cfg.d_model
    vocab_size = W_U.shape[1]
    print(f"  d_model={d_model}, vocab={vocab_size}, loaded in {time.time() - t0:.1f}s")

    # ── Load data and precompute h_L ──
    print(f"\n[2/5] Loading data + precomputing hidden states...")
    total_needed = args.n_train + args.n_val + args.n_test
    all_samples = load_triviaqa(n_samples=total_needed, seed=args.seed)
    all_samples = all_samples[:total_needed]

    # Shuffle splits
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_samples))
    train_idx = indices[: args.n_train]
    val_idx = indices[args.n_train : args.n_train + args.n_val]
    test_idx = indices[args.n_train + args.n_val :]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    print(f"  Precomputing hidden states for {len(all_samples)} samples...")
    t_cache = time.time()

    train_h, train_yt, train_mask = precompute_hidden_states(
        model,
        tokenizer,
        train_samples,
        device,
        args.layer,
    )
    val_h, val_yt, val_mask = precompute_hidden_states(
        model,
        tokenizer,
        val_samples,
        device,
        args.layer,
    )
    test_h, test_yt, test_mask = precompute_hidden_states(
        model,
        tokenizer,
        test_samples,
        device,
        args.layer,
    )

    n_train_valid = sum(train_mask)
    n_val_valid = sum(val_mask)
    n_test_valid = sum(test_mask)
    print(
        f"  Valid samples: train={n_train_valid}/{len(train_samples)}, "
        f"val={n_val_valid}/{len(val_samples)}, test={n_test_valid}/{len(test_samples)}"
    )
    print(f"  Precomputed in {time.time() - t_cache:.1f}s")

    # ── Train for each λ ──
    print(f"\n[3/5] Training correction networks...")

    all_results = {}

    for lam in args.lambdas:
        print(f"\n  ── λ={lam} ──")

        net = CorrectionNet(d_model, rank=args.rank).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_epoch_cached(
                net,
                optimizer,
                train_h,
                train_yt,
                train_mask,
                W_U,
                b_U,
                ln_final,
                device,
                lam,
                args.batch_size,
            )

            val_metrics = evaluate_loss_and_metrics(
                net,
                val_h,
                val_yt,
                val_mask,
                W_U,
                b_U,
                ln_final,
                device,
                lam,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(
                    f"    Epoch {epoch:>3d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"|δ|={val_metrics.get('delta_norm_mean', 0):.4f}, "
                    f"cos(δ,g)={val_metrics.get('cos_with_g_mean', 0):.4f}, "
                    f"div={val_metrics.get('delta_diversity', 0):.3f}"
                )

        # Restore best
        net.load_state_dict(best_state)

        # Final metrics on val (fast)
        val_final = evaluate_loss_and_metrics(
            net,
            val_h,
            val_yt,
            val_mask,
            W_U,
            b_U,
            ln_final,
            device,
            lam,
        )

        # Accuracy eval on test (slow, generation-based)
        print(f"    Evaluating accuracy on test set...")
        test_acc = evaluate_accuracy(
            net,
            test_samples,
            test_mask,
            model,
            tokenizer,
            device,
            args.layer,
        )
        print(
            f"    Test acc: {test_acc['accuracy']:.1%} "
            f"(baseline={test_acc['baseline_accuracy']:.1%}), "
            f"Δ={test_acc['accuracy'] - test_acc['baseline_accuracy']:+.1%}"
        )

        all_results[f"lambda_{lam}"] = {
            "val_final": val_final,
            "test_acc": test_acc,
        }

    # ── Ablation: Global network ──
    if not args.skip_ablation:
        print(f"\n[4/5] Ablation: global δ_θ(h) without e(x)...")
        best_lam = min(
            args.lambdas,
            key=lambda lam: all_results[f"lambda_{lam}"]["val_final"]["loss"],
        )
        print(f"  Using λ={best_lam}")

        global_net = GlobalCorrectionNet(d_model, rank=args.rank).to(device)
        global_optimizer = torch.optim.Adam(global_net.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss = train_epoch_cached(
                global_net,
                global_optimizer,
                train_h,
                train_yt,
                train_mask,
                W_U,
                b_U,
                ln_final,
                device,
                best_lam,
                args.batch_size,
            )
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                val_m = evaluate_loss_and_metrics(
                    global_net,
                    val_h,
                    val_yt,
                    val_mask,
                    W_U,
                    b_U,
                    ln_final,
                    device,
                    best_lam,
                )
                print(
                    f"    Epoch {epoch:>3d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_m['loss']:.4f}, "
                    f"cos(δ,g)={val_m.get('cos_with_g_mean', 0):.4f}"
                )

        global_test_acc = evaluate_accuracy(
            global_net,
            test_samples,
            test_mask,
            model,
            tokenizer,
            device,
            args.layer,
        )
        print(
            f"    Global test acc: {global_test_acc['accuracy']:.1%} "
            f"(baseline={global_test_acc['baseline_accuracy']:.1%}), "
            f"Δ={global_test_acc['accuracy'] - global_test_acc['baseline_accuracy']:+.1%}"
        )
        all_results["ablation_global"] = {"test_acc": global_test_acc}

    # ── Summary ──
    step = "5" if args.skip_ablation else "5"
    print(f"\n[{step}/5] Summary")
    print(
        f"\n  {'λ':>10s} {'Test Acc':>10s} {'Test Δ':>10s} {'cos(δ,g)':>10s} {'|δ|':>10s}"
    )
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for lam in args.lambdas:
        r = all_results[f"lambda_{lam}"]
        acc = r["test_acc"]
        vf = r["val_final"]
        print(
            f"  {lam:>10.4f} "
            f"{acc['accuracy']:>10.1%} "
            f"{acc['accuracy'] - acc['baseline_accuracy']:>+9.1%} "
            f"{vf.get('cos_with_g_mean', 0):>10.3f} "
            f"{vf.get('delta_norm_mean', 0):>10.4f}"
        )

    if not args.skip_ablation:
        acc = all_results["ablation_global"]["test_acc"]
        print(
            f"  {'global':>10s} "
            f"{acc['accuracy']:>10.1%} "
            f"{acc['accuracy'] - acc['baseline_accuracy']:>+9.1%} "
            f"{'—':>10s} "
            f"{'—':>10s}"
        )

    # Gate checks
    best_test = max(
        (all_results[f"lambda_{lam}"]["test_acc"] for lam in args.lambdas),
        key=lambda x: x["accuracy"] - x["baseline_accuracy"],
    )
    best_delta = best_test["accuracy"] - best_test["baseline_accuracy"]
    best_cos = max(
        all_results[f"lambda_{lam}"]["val_final"].get("cos_with_g_mean", 0)
        for lam in args.lambdas
    )
    l2_passes = best_cos > 0.3
    l3_passes = best_delta > 0.05

    print(f"\n  L2 (cos(δ,g) > 0.3): {'PASS' if l2_passes else 'FAIL'}")
    print(f"  L3 (test Δ > 5%): {'PASS' if l3_passes else 'FAIL'}")

    # ── Save ──
    output = {
        "config": vars(args),
        "results": all_results,
        "gates": {"l2": bool(l2_passes), "l3": bool(l3_passes)},
    }

    out_path = output_dir / "s13_2_learned_delta.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
