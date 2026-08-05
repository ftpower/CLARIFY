"""Phase 20.3: Adapter Bottleneck Injection — bypass L21-L27 override computation.

Theory: docs/theory-intervention-failure.md
Plan:   ~/.claude/plans/CLARIFY/phase20-training-intervention.md §20.3

Core idea:
  h_L20 already contains enough information for correct answers (AUROC=0.75-0.81).
  The override that causes hallucinations happens in L21-L27.
  Instead of fixing the override mechanism, bypass it entirely:
    h_adapted = h_L20 + Adapter(h_L20)
    logits    = ln_final(h_adapted) @ W_U + b_U

  Adapter(h) = W_up @ SiLU(W_down @ LayerNorm(h))
    W_down: [2048, 64], W_up: [64, 2048]  (~262K params, < 1% of 1.7B model)

Training:  CE(y_true | adapted_logits) + λ ||Adapter(h_L20)||₂²
Inference: L0-L20 → Adapter → ln_final → W_U → argmax  (L21-L27 skipped)

Gates (see plan §20.3.6):
  P20.3.1 (primary): Adapter KW first-token accuracy > L27 baseline
  P20.3.2:           Adapter KC accuracy ≥ L27 baseline × 0.8

Usage:
  python train_adapter.py --mode train --n_train 500 --n_test 100
  python train_adapter.py --mode eval
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Environment ────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Path setup ─────────────────────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import load_model_and_unembed, get_first_answer_token_id

# ── Constants ──────────────────────────────────────────────────────────────────
LAYER = 20  # Extract h at blocks.20.hook_resid_post (before L21-L27 override)
BOTTLENECK = 64
D_MODEL = 2048
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"
ADAPTER_PATH = OUTPUT_DIR / "s20_3_adapter.pt"
RESULTS_PATH = OUTPUT_DIR / "s20_3_adapter.json"

# KC/KW/DK classification thresholds (from plan §20 math conventions)
RANK_THRESHOLD = 50


# ═══════════════════════════════════════════════════════════════════════════════
# Adapter module
# ═══════════════════════════════════════════════════════════════════════════════


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: LayerNorm → Linear(d→r) → SiLU → Linear(r→d).

    Zero-initialized W_up so that at init, Adapter(h) ≈ 0 (safe residual start).
    """

    def __init__(self, d_model: int = D_MODEL, bottleneck: int = BOTTLENECK):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        # Small init for down-projection, zero init for up (residual-safe)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., d_model] → residual correction [..., d_model]."""
        return self.up(F.silu(self.down(self.ln(h))))


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation: extract h_L20 and y_true_id for training samples
# ═══════════════════════════════════════════════════════════════════════════════


def load_triviaqa_train(n_samples: int, seed: int = 42) -> list[dict]:
    """Load TriviaQA TRAIN split samples (unlike load_triviaqa which uses validation).

    Returns list of dicts with keys: question, answers (list of aliases), context (str).
    """
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        question = item["question"]
        answers = item["answer"]["aliases"]
        search_contexts = item["search_results"]["search_context"]
        context = "\n\n".join(ctx for ctx in search_contexts if ctx)
        samples.append({"question": question, "answers": answers, "context": context})
    return samples


def extract_h_L20_batch(
    model, tokenizer, samples: list[dict], device: str
) -> tuple[torch.Tensor, list[int], list[dict]]:
    """Forward each sample, extract h_L20 at last prompt token position.

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        samples: list of {"question", "answers", "context"}
        device: "cuda" or "cpu"

    Returns:
        h_stack: [n, d_model] float32 on CPU — hidden states at L20
        y_true_ids: list of int — first token ID of correct answer (None if invalid)
        valid_samples: list of dict — samples with valid y_true_id
    """
    hook_name = f"blocks.{LAYER}.hook_resid_post"
    h_list = []
    y_true_ids = []
    valid_samples = []

    for s in tqdm(samples, desc=f"  Extracting h_L{LAYER}"):
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

        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            continue  # skip samples whose answer can't be tokenized

        h_list.append(residual["h"].float().cpu())
        y_true_ids.append(y_true_id)
        valid_samples.append(s)

    if not h_list:
        raise RuntimeError("No valid samples after filtering!")

    h_stack = torch.cat(h_list, dim=0)  # [n, d_model]
    return h_stack, y_true_ids, valid_samples


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_adapter(args):
    """Main training routine: extract h_L20, train adapter, save."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Phase 20.3: Adapter Training | n_train={args.n_train} lr={args.lr}")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    print(f"  Loaded in {time.time() - t0:.1f}s | d_model={model.cfg.d_model}")

    # ── 2. Load training data & extract h_L20 ───────────────────────────────
    print(f"\n[2/5] Loading {args.n_train} TriviaQA train samples...")
    train_samples = load_triviaqa_train(n_samples=args.n_train, seed=args.seed)
    h_train, y_true_ids, valid_samples = extract_h_L20_batch(
        model, tokenizer, train_samples, device
    )
    n_valid = len(y_true_ids)
    print(
        f"  Valid samples: {n_valid}/{args.n_train} "
        f"(filtered {args.n_train - n_valid} with un-tokenizable answers)"
    )

    y_true_tensor = torch.tensor(y_true_ids, dtype=torch.long)  # [n_valid]

    # ── 3. Create adapter ──────────────────────────────────────────────────
    print(f"\n[3/5] Creating adapter (bottleneck={BOTTLENECK})...")
    adapter = BottleneckAdapter(d_model=D_MODEL, bottleneck=BOTTLENECK).to(device)
    adapter.train()
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Trainable params: {n_params:,}")

    # ── 3.5 Free model, keep only ln_final + unembedding components ──────
    # Model itself (~3.4 GB float16) is no longer needed after h_L20 extraction.
    # Clone the small components needed for adapter training before freeing.
    print(f"\n[3.5/5] Freeing model, keeping only unembedding components...")
    W_U_weight = W_U.detach().clone()  # [d_model, vocab_size] float16
    b_U_weight = (
        b_U.detach().clone() if b_U is not None else None
    )  # [vocab_size] float16
    # TransformerLens RMSNorm uses 'w' attribute (not 'weight')
    ln_w_attr = "w" if hasattr(ln_final, "w") else "weight"
    ln_weight = getattr(ln_final, ln_w_attr).detach().clone()  # [d_model] float16
    ln_eps = getattr(ln_final, "eps", 1e-6)
    del model  # free ~3.4 GB GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model freed, GPU memory available")

    # ── 4. Train ────────────────────────────────────────────────────────────
    print(
        f"\n[4/5] Training | lr={args.lr} batch_size={args.batch_size} "
        f"epochs={args.epochs} lambda={args.lambda_l2}"
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Move training data to device once
    h_train_dev = h_train.to(device)  # [n, d_model] float32
    y_true_dev = y_true_tensor.to(device)

    dataset = TensorDataset(h_train_dev, y_true_dev)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_losses = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{args.epochs}")

        for h_batch, y_batch in pbar:
            # h_batch: [B, d_model] float32
            # y_batch: [B] int64

            # Forward through adapter
            delta = adapter(h_batch)  # [B, d_model]
            h_adapted = h_batch + delta  # residual connection [B, d_model]

            # RMSNorm: h_norm = h * rsqrt(mean(h²) + eps) * weight
            # Compute in float32 for stability, then cast back for W_U matmul
            h_f32 = h_adapted.float()
            rms = torch.sqrt(torch.mean(h_f32**2, dim=-1, keepdim=True) + ln_eps)
            h_norm_f16 = (h_f32 / rms * ln_weight.float()).to(torch.float16)

            # Compute logits in float16 (saves 600+ MB vs float32)
            logits_f16 = h_norm_f16 @ W_U_weight  # [B, vocab_size] float16
            if b_U_weight is not None:
                logits_f16 = logits_f16 + b_U_weight

            # CE loss: cast logits to float32 for softmax stability
            ce_loss = F.cross_entropy(logits_f16.float(), y_batch)
            l2_loss = delta.pow(2).mean()
            loss = ce_loss + args.lambda_l2 * l2_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "l2": f"{l2_loss.item():.6f}",
                }
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()
        print(f"    Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(adapter.state_dict(), ADAPTER_PATH)
            print(f"    -> Saved best adapter to {ADAPTER_PATH}")

    # ── 5. Save training metadata ──────────────────────────────────────────
    print(f"\n[5/5] Saving results...")
    results = {
        "config": {
            "phase": "20.3",
            "n_train": args.n_train,
            "n_valid": n_valid,
            "seed": args.seed,
            "layer": LAYER,
            "bottleneck": BOTTLENECK,
            "d_model": D_MODEL,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lambda_l2": args.lambda_l2,
        },
        "train_losses": train_losses,
        "best_loss": best_loss,
        "n_params": n_params,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {RESULTS_PATH}")

    # Cleanup (model already freed in step 3.5)
    del W_U_weight
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def classify_sample(
    logits_L27: torch.Tensor, y_true_id: int, generated: str, answers: list[str]
) -> str:
    """Classify a sample as KC, KW, or DK based on model knowledge and output.

    KC (Known-Correct):   generated answer matches ground truth
    KW (Known-Wrong):     rank(y_true) ≤ RANK_THRESHOLD but generated ≠ correct
    DK (Don't-Know):      rank(y_true) > RANK_THRESHOLD
    """
    is_correct = check_correct(generated, answers, dataset="triviaqa")
    if is_correct:
        return "KC"

    # Compute rank of y_true in L27 logits
    sorted_indices = torch.argsort(logits_L27, descending=True)
    rank = (sorted_indices == y_true_id).nonzero(as_tuple=True)[0].item() + 1

    if rank <= RANK_THRESHOLD:
        return "KW"
    return "DK"


@torch.no_grad()
def compute_first_token_logits(
    model,
    tokenizer,
    prompt: str,
    device: str,
    ln_final,
    W_U,
    b_U,
    adapter: BottleneckAdapter | None = None,
):
    """Compute first-token logits at last prompt position.

    Args:
        adapter: if provided, uses adapter(h_L20) + ln_final → W_U (bypass L21-L27).
                 if None, uses normal L27 path.

    Returns:
        logits: [vocab_size] float32 on CPU
        first_token_id: int — argmax token
        first_token_text: str — decoded argmax token
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    if adapter is not None:
        # Adapter path: extract h_L20 → adapter → ln_final → W_U
        hook_name = f"blocks.{LAYER}.hook_resid_post"
        residual = {}

        def _hook(act, hook=None):
            residual["h"] = act[:, -1, :].detach()
            return act

        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        h_L20 = residual["h"].float()  # [1, d_model]
        adapter.eval()
        delta = adapter(h_L20.to(device))
        h_adapted = h_L20.to(device) + delta

        model_dtype = next(ln_final.parameters()).dtype
        h_norm = ln_final(h_adapted.to(dtype=model_dtype))

        logits = (h_norm.float() @ W_U.float()).squeeze(0)  # [vocab_size]
        if b_U is not None:
            logits = logits + b_U.float()
    else:
        # Normal L27 path
        with torch.no_grad():
            raw_logits = model(tokens)
        logits = raw_logits[0, -1, :].float().cpu()

    logits_cpu = logits.cpu()
    first_token_id = int(logits_cpu.argmax().item())
    first_token_text = tokenizer.decode([first_token_id])
    return logits_cpu, first_token_id, first_token_text


@torch.no_grad()
def generate_full_answer(
    model, tokenizer, prompt: str, device: str, first_token_id: int, max_new: int = 20
) -> str:
    """Generate full answer given the forced first token.

    Runs: first_token → normal L0-L27 autoregressive completion.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    # Append forced first token
    first_tok = torch.tensor([[first_token_id]], device=device)
    tokens = torch.cat([tokens, first_tok], dim=1)
    gids = [first_token_id]

    # Continue with normal greedy generation
    for _ in range(max_new - 1):
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    return tokenizer.decode(gids).strip()


def evaluate(args):
    """Evaluate adapter vs baseline on test set (seed=123, n=100)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 20.3: Adapter Evaluation | n_test={args.n_test}")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load adapter ────────────────────────────────────────────────────
    print("\n[2/4] Loading adapter...")
    if not ADAPTER_PATH.exists():
        print(f"  ERROR: Adapter not found at {ADAPTER_PATH}")
        print(f"  Run 'python train_adapter.py --mode train' first.")
        return

    adapter = BottleneckAdapter(d_model=D_MODEL, bottleneck=BOTTLENECK).to(device)
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=device))
    adapter.eval()
    print(f"  Loaded from {ADAPTER_PATH}")

    # ── 3. Load test data ──────────────────────────────────────────────────
    print(
        f"\n[3/4] Loading {args.n_test} TriviaQA validation samples (seed={args.seed})..."
    )
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating...")
    results = {
        "config": {
            "phase": "20.3",
            "n_test": args.n_test,
            "seed": args.seed,
            "layer": LAYER,
            "bottleneck": BOTTLENECK,
            "rank_threshold": RANK_THRESHOLD,
        },
        "per_sample": [],
        "summary": {},
    }

    # Counters
    baseline_ft_correct = 0
    adapter_ft_correct = 0
    baseline_em_correct = 0
    adapter_em_correct = 0

    # Per-category counters
    for cat in ["KC", "KW", "DK"]:
        for prefix in ["baseline_ft", "adapter_ft", "baseline_em", "adapter_em"]:
            results["summary"][f"{prefix}_{cat}_correct"] = 0
            results["summary"][f"{prefix}_{cat}_total"] = 0

    for s in tqdm(test_samples, desc="  Evaluating"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])

        # ── Baseline (L27) ──
        bl_logits, bl_ft_id, bl_ft_text = compute_first_token_logits(
            model, tokenizer, prompt, device, ln_final, W_U, b_U, adapter=None
        )
        bl_ft_correct = y_true_id is not None and bl_ft_id == y_true_id
        bl_full = generate_full_answer(model, tokenizer, prompt, device, bl_ft_id)
        bl_em_correct = check_correct(bl_full, s["answers"], dataset="triviaqa")

        # ── Adapter (L20 + adapter) ──
        ad_logits, ad_ft_id, ad_ft_text = compute_first_token_logits(
            model, tokenizer, prompt, device, ln_final, W_U, b_U, adapter=adapter
        )
        ad_ft_correct = y_true_id is not None and ad_ft_id == y_true_id
        ad_full = generate_full_answer(model, tokenizer, prompt, device, ad_ft_id)
        ad_em_correct = check_correct(ad_full, s["answers"], dataset="triviaqa")

        # ── Classify sample ──
        category = classify_sample(bl_logits, y_true_id or 0, bl_full, s["answers"])

        # ── Record ──
        if bl_ft_correct:
            baseline_ft_correct += 1
        if ad_ft_correct:
            adapter_ft_correct += 1
        if bl_em_correct:
            baseline_em_correct += 1
        if ad_em_correct:
            adapter_em_correct += 1

        for cat in ["KC", "KW", "DK"]:
            if category == cat:
                results["summary"][f"baseline_ft_{cat}_total"] += 1
                results["summary"][f"adapter_ft_{cat}_total"] += 1
                results["summary"][f"baseline_em_{cat}_total"] += 1
                results["summary"][f"adapter_em_{cat}_total"] += 1
                if bl_ft_correct:
                    results["summary"][f"baseline_ft_{cat}_correct"] += 1
                if ad_ft_correct:
                    results["summary"][f"adapter_ft_{cat}_correct"] += 1
                if bl_em_correct:
                    results["summary"][f"baseline_em_{cat}_correct"] += 1
                if ad_em_correct:
                    results["summary"][f"adapter_em_{cat}_correct"] += 1

        results["per_sample"].append(
            {
                "question": s["question"],
                "answers": s["answers"],
                "category": category,
                "y_true_id": y_true_id,
                "baseline_ft_id": bl_ft_id,
                "baseline_ft_text": bl_ft_text,
                "baseline_ft_correct": bl_ft_correct,
                "adapter_ft_id": ad_ft_id,
                "adapter_ft_text": ad_ft_text,
                "adapter_ft_correct": ad_ft_correct,
                "baseline_full": bl_full,
                "baseline_em_correct": bl_em_correct,
                "adapter_full": ad_full,
                "adapter_em_correct": ad_em_correct,
            }
        )

    # ── Summary ──
    n = len(test_samples)
    results["summary"]["n_total"] = n
    results["summary"]["baseline_ft_accuracy"] = baseline_ft_correct / n
    results["summary"]["adapter_ft_accuracy"] = adapter_ft_correct / n
    results["summary"]["baseline_em_accuracy"] = baseline_em_correct / n
    results["summary"]["adapter_em_accuracy"] = adapter_em_correct / n
    results["summary"]["ft_delta"] = (adapter_ft_correct - baseline_ft_correct) / n
    results["summary"]["em_delta"] = (adapter_em_correct - baseline_em_correct) / n

    # Per-category accuracies
    for cat in ["KC", "KW", "DK"]:
        for prefix in ["baseline_ft", "adapter_ft", "baseline_em", "adapter_em"]:
            total = results["summary"][f"{prefix}_{cat}_total"]
            correct = results["summary"][f"{prefix}_{cat}_correct"]
            acc = correct / max(total, 1)
            results["summary"][f"{prefix}_{cat}_accuracy"] = acc

    # ── Gate checks ──
    kw_ft_total = results["summary"]["baseline_ft_KW_total"]
    kw_ad_ft_correct = results["summary"]["adapter_ft_KW_correct"]
    kw_bl_ft_correct = results["summary"]["baseline_ft_KW_correct"]

    kw_ad_acc = kw_ad_ft_correct / max(kw_ft_total, 1)
    kw_bl_acc = kw_bl_ft_correct / max(kw_ft_total, 1)

    kc_ft_total = results["summary"]["baseline_ft_KC_total"]
    kc_ad_ft_correct = results["summary"]["adapter_ft_KC_correct"]
    kc_bl_ft_correct = results["summary"]["baseline_ft_KC_correct"]

    kc_ad_acc = kc_ad_ft_correct / max(kc_ft_total, 1)
    kc_bl_acc = kc_bl_ft_correct / max(kc_ft_total, 1)

    gate_p2031 = kw_ad_acc > kw_bl_acc  # P20.3.1: KW Δ > 0
    gate_p2032 = kc_ad_acc >= kc_bl_acc * 0.8  # P20.3.2: KC not severely degraded

    results["summary"]["gates"] = {
        "P20.3.1": {
            "description": "Adapter KW first-token accuracy > baseline",
            "baseline_kw_ft_acc": kw_bl_acc,
            "adapter_kw_ft_acc": kw_ad_acc,
            "delta": kw_ad_acc - kw_bl_acc,
            "pass": gate_p2031,
        },
        "P20.3.2": {
            "description": "Adapter KC accuracy >= baseline × 0.8",
            "baseline_kc_ft_acc": kc_bl_acc,
            "adapter_kc_ft_acc": kc_ad_acc,
            "threshold": kc_bl_acc * 0.8,
            "pass": gate_p2032,
        },
    }

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(
        f"  N = {n} | KC={kc_ft_total} KW={kw_ft_total} DK={results['summary']['baseline_ft_DK_total']}"
    )
    print(f"\n  First-token accuracy:")
    print(f"    Baseline: {baseline_ft_correct}/{n} = {baseline_ft_correct / n:.1%}")
    print(f"    Adapter:  {adapter_ft_correct}/{n} = {adapter_ft_correct / n:.1%}")
    print(f"    Delta:    {(adapter_ft_correct - baseline_ft_correct) / n:+.1%}")
    print(f"\n  Exact-match accuracy:")
    print(f"    Baseline: {baseline_em_correct}/{n} = {baseline_em_correct / n:.1%}")
    print(f"    Adapter:  {adapter_em_correct}/{n} = {adapter_em_correct / n:.1%}")
    print(f"    Delta:    {(adapter_em_correct - baseline_em_correct) / n:+.1%}")
    print(f"\n  Per-category first-token:")
    for cat in ["KC", "KW", "DK"]:
        bl_a = results["summary"][f"baseline_ft_{cat}_accuracy"]
        ad_a = results["summary"][f"adapter_ft_{cat}_accuracy"]
        total = results["summary"][f"baseline_ft_{cat}_total"]
        print(
            f"    {cat} (n={total}): baseline={bl_a:.1%} adapter={ad_a:.1%} delta={ad_a - bl_a:+.1%}"
        )
    print(f"\n  Gates:")
    for gate_name, gate_info in results["summary"]["gates"].items():
        status = "✅ PASS" if gate_info["pass"] else "❌ FAIL"
        print(f"    {gate_name}: {status} — {gate_info['description']}")
    print(f"{'=' * 60}")

    # ── Save ───────────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {RESULTS_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 20.3: Adapter Bottleneck Injection"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: extract h_L20 + train adapter | eval: evaluate adapter vs baseline",
    )
    # Training args
    parser.add_argument(
        "--n_train",
        type=int,
        default=500,
        help="Number of TriviaQA TRAIN samples (default: 500)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Training epochs (default: 5)"
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.01,
        help="L2 regularization weight (default: 0.01)",
    )
    # Eval args
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of TriviaQA validation samples for eval (default: 100)",
    )
    # Shared
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data selection"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        train_adapter(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
