"""Phase 20 Direction C: Contrastive Learning — replace δ margin with softmax in Q/V space.

Theory: docs/phase20-8b-failure-analysis.md §2 Direction C
Plan:   ~/.claude/plans/CLARIFY/phase20-8b-validation.md

Core idea:
  The δ margin penalty (Direction B) is a hard scalar gate — it fires only when
  g(d*) > g(t*) + m.  Large models have more legitimate late-layer refinement,
  so the scalar δ signal confuses override with refinement (§1.4 of the failure
  analysis).  Instead of a hard margin, use a softmax contrastive loss over ALL
  token channel gains, plus an optional hidden-state prototype loss.

  Loss = CE(softmax(g_all / τ), y_true) + α · BCE(sim(proj(Δh), proto), is_correct)

  where g_all = logits_L27 - logits_ref  (channel gains for all tokens)
        Δh    = h_last - h_ref           (representation drift)

  LoRA: r=8, α=16, target=["q_proj","v_proj"], last 8 layers (same as Phase 20.1)

Gates (same as Phase 20.1):
  P20.1.2: KW exact match Δ > 0
  P20.1.3: KC exact match degradation ≤ 1

Usage:
  # Channel-gain contrastive (primary mode)
  python train_contrastive.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --temperature 0.5

  # + hidden-state prototype auxiliary loss
  python train_contrastive.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --temperature 0.5 \\
      --proto_loss --proto_alpha 0.1

  # Large training set (Direction A variant with contrastive)
  python train_contrastive.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 2000 --batch_size 1 --epochs 1 --temperature 0.5
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Environment ────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_offline = os.environ.get("HF_ALLOW_ONLINE", "") != "1"
if _offline:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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

# ── Re-use utilities from train_lora_delta ─────────────────────────────────────
from train_lora_delta import (
    NUM_DELTA_LAYERS,
    RANK_THRESHOLD,
    _get_delta_layers,
    _compute_auroc,
    load_triviaqa_train,
    get_first_answer_token_id,
    TriviaQADataset,
    collate_lora_batch,
    classify_sample,
    generate_with_model,
    evaluate as evaluate_delta,  # re-use eval infrastructure
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"

# ── Model path resolution ──────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-1.7B"


def _find_model_path() -> str:
    import os as _os

    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        _os.path.join(
            _os.environ.get("HF_HOME", _os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = _os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if _os.path.isdir(local):
            if _os.path.isfile(_os.path.join(local, "config.json")):
                return local
            snaps = _os.path.join(local, "snapshots")
            if _os.path.isdir(snaps):
                for s in sorted(_os.listdir(snaps)):
                    sp = _os.path.join(snaps, s)
                    if _os.path.isfile(_os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID


MODEL_PATH = _find_model_path()


def _get_contrastive_dir(temperature: float, use_proto: bool) -> Path:
    """Checkpoint directory keyed by temperature and proto flag."""
    tag = f"c_contrastive_t{temperature}"
    if use_proto:
        tag += "_proto"
    return OUTPUT_DIR / tag


def _get_contrastive_results_path(temperature: float, use_proto: bool) -> Path:
    tag = f"c_contrastive_t{temperature}"
    if use_proto:
        tag += "_proto"
    return OUTPUT_DIR / f"{tag}.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Projection head for hidden-state prototype loss
# ═══════════════════════════════════════════════════════════════════════════════


class PrototypeHead(nn.Module):
    """2-layer MLP projecting Δh → contrastive embedding space."""

    def __init__(self, d_model: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Learnable "truth" prototype
        self.prototype = nn.Parameter(torch.randn(out_dim) / math.sqrt(out_dim))

    def forward(self, delta_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_normalised, cosine_sim_to_prototype)."""
        z = self.net(delta_h.float())
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototype, dim=-1)
        sim = (z * p).sum(dim=-1)  # cosine similarity, range [-1, 1]
        return z, sim


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_contrastive(args):
    """Train LoRA with contrastive channel-gain loss (+ optional prototype loss)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_proto = getattr(args, "proto_loss", False)
    ckpt_dir = _get_contrastive_dir(args.temperature, use_proto)
    results_path = _get_contrastive_results_path(args.temperature, use_proto)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    mode_str = "channel-contrastive"
    if use_proto:
        mode_str += " + prototype"
    print(
        f"Phase 20 Direction C: {mode_str} | "
        f"n_train={args.n_train} | τ={args.temperature}"
    )

    # ── 1. Load tokenizer + base model ──────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    layer_early, layer_late = _get_delta_layers(model)
    d_model = model.config.hidden_size
    print(
        f"  Model: {model.config.num_hidden_layers} layers, "
        f"d_model={d_model}, target L{layer_early}-L{layer_late}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load training data ──────────────────────────────────────────────
    print(f"\n[2/5] Loading {args.n_train} TriviaQA train samples...")
    train_samples = load_triviaqa_train(n_samples=args.n_train, seed=args.seed)
    train_dataset = TriviaQADataset(train_samples, tokenizer)
    print(f"  Valid samples: {len(train_dataset)}/{args.n_train}")

    # ── 3. Apply LoRA ──────────────────────────────────────────────────────
    print(f"\n[3/5] Applying LoRA (r={args.lora_r}, α={args.lora_alpha})...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_target_layers = list(range(layer_early, layer_late + 1))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=lora_target_layers,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # ── 3.5. Prototype head (if enabled) ───────────────────────────────────
    proto_head = None
    proto_optimizer = None
    if use_proto:
        proto_head = PrototypeHead(d_model=d_model).to(device)
        proto_optimizer = torch.optim.AdamW(
            proto_head.parameters(), lr=args.lr, weight_decay=0.01
        )
        print(
            f"  Prototype head: {sum(p.numel() for p in proto_head.parameters()):,} params"
        )

    # ── 4. Register hooks for reference layer ──────────────────────────────
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        try:
            layers = model.model.model.layers
        except AttributeError:
            layers = model.model.layers

    h_ref_cache = {}
    h_last_cache = {}

    def _capture_ref(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        h_ref_cache["h"] = hs.detach()  # reference is always fixed (no gradient)

    def _capture_last(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Do NOT detach — prototype loss needs gradients through late layers
        h_last_cache["h"] = hs

    ref_handle = layers[layer_early].register_forward_hook(_capture_ref)
    last_handle = None
    if use_proto:
        last_handle = layers[layer_late].register_forward_hook(_capture_last)

    # Get norm and lm_head
    try:
        norm = model.base_model.model.model.norm
        lm_head = model.base_model.model.lm_head
    except AttributeError:
        try:
            norm = model.model.model.norm
            lm_head = model.model.lm_head
        except AttributeError:
            norm = model.model.norm
            lm_head = model.lm_head

    # ── 5. Train ──────────────────────────────────────────────────────────
    proto_str = f" + proto(α={args.proto_alpha})" if use_proto else ""
    print(
        f"\n[5/5] Training | lr={args.lr} batch_size={args.batch_size} "
        f"epochs={args.epochs} τ={args.temperature}{proto_str}"
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_lora_batch(b, tokenizer),
    )

    train_losses = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_ce = 0.0
        epoch_contrastive = 0.0
        epoch_proto = 0.0
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            y_true_ids = batch["y_true_ids"].to(device)
            B = input_ids.shape[0]

            # Clear ref and last cache
            h_ref_cache.clear()
            h_last_cache.clear()

            # Forward (output_hidden_states needed for prototype loss)
            outputs = model(input_ids=input_ids, labels=labels)
            ce_loss = outputs.loss  # standard CE on answer token

            # Get L27 logits — DO NOT detach: contrastive loss needs gradients
            logits_L27 = outputs.logits  # [B, seq, vocab]

            # Last non-masked position per sample
            last_positions = []
            for b_idx in range(B):
                non_mask = (labels[b_idx] != -100).nonzero(as_tuple=True)[0]
                if len(non_mask) > 0:
                    last_positions.append(non_mask[-1].item())
                else:
                    last_positions.append(labels.shape[1] - 1)

            g_L27 = torch.stack(
                [logits_L27[b, last_positions[b], :] for b in range(B)]
            ).float()  # [B, vocab]

            # Reference logits
            h_early = h_ref_cache.get("h")
            if h_early is None:
                contrastive_loss = torch.tensor(0.0, device=device)
                proto_loss = torch.tensor(0.0, device=device)
            else:
                # ── Channel-gain contrastive loss ──────────────────────────
                # g_all(t) = logit_L27(t) - logit_ref(t): per-token channel gain
                h_early_last = torch.stack(
                    [h_early[b, last_positions[b], :] for b in range(B)]
                )
                h_early_norm = norm(h_early_last.to(dtype=norm.weight.dtype))
                g_ref = lm_head(h_early_norm).float()  # [B, vocab]

                g_all = (g_L27 - g_ref) / args.temperature  # [B, vocab]

                # Softmax contrastive: drive y_true to have highest channel gain
                contrastive_loss = F.cross_entropy(g_all, y_true_ids)

                # ── Hidden-state prototype loss (optional) ─────────────────
                proto_loss = torch.tensor(0.0, device=device)
                if use_proto and proto_head is not None:
                    h_last = h_last_cache.get("h")
                    if h_last is not None:
                        # Δh = h_last - h_ref at the answer position
                        h_last_pos = torch.stack(
                            [h_last[b, last_positions[b], :] for b in range(B)]
                        )  # [B, d_model]
                        delta_h = h_last_pos - h_early_last  # [B, d_model]

                        # Project and compute similarity to truth prototype
                        _, sim = proto_head(delta_h)  # [B], cosine sim in [-1, 1]

                        # Binary target: KC samples (d==t*) = 1, others = 0
                        with torch.no_grad():
                            is_kc = (g_L27.argmax(dim=-1) == y_true_ids).float()
                        proto_loss = F.binary_cross_entropy_with_logits(sim, is_kc)

            loss = ce_loss + args.lambda_delta * contrastive_loss
            if proto_loss.item() > 0:
                loss = loss + args.proto_alpha * proto_loss

            optimizer.zero_grad()
            if proto_optimizer is not None:
                proto_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if proto_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(proto_head.parameters(), 1.0)
            optimizer.step()
            if proto_optimizer is not None:
                proto_optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_contrastive += contrastive_loss.item()
            epoch_proto += (
                proto_loss.item()
                if isinstance(proto_loss, torch.Tensor)
                else proto_loss
            )
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "contrast": f"{contrastive_loss.item():.4f}",
                }
            )

            del input_ids, labels, outputs, logits_L27
            if n_batches % 10 == 0:
                torch.cuda.empty_cache()

        avg_ce = epoch_ce / max(n_batches, 1)
        avg_contrastive = epoch_contrastive / max(n_batches, 1)
        avg_proto = epoch_proto / max(n_batches, 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(
            {
                "epoch": epoch + 1,
                "ce": avg_ce,
                "contrastive": avg_contrastive,
                "proto": avg_proto,
                "total": avg_loss,
            }
        )
        print(
            f"    Epoch {epoch + 1}: ce={avg_ce:.4f} contrastive={avg_contrastive:.4f} "
            f"proto={avg_proto:.4f} total={avg_loss:.4f}"
        )

        ep_dir = ckpt_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(str(ep_dir))
        # Save prototype head if used
        if use_proto and proto_head is not None:
            torch.save(proto_head.state_dict(), str(ep_dir / "proto_head.pt"))
        print(f"    -> Saved to {ep_dir}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    # ── 6. Cleanup & save metadata ────────────────────────────────────────
    ref_handle.remove()
    if last_handle is not None:
        last_handle.remove()
    print(f"\n[6/6] Saving metadata...")
    results = {
        "config": {
            "phase": "20-C",
            "direction": "contrastive",
            "mode": "channel-contrastive" + (" + prototype" if use_proto else ""),
            "n_train": args.n_train,
            "n_valid": len(train_dataset),
            "seed": args.seed,
            "layers": f"L{layer_early}-L{layer_late}",
            "lora_target_layers": lora_target_layers,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "lambda_delta": args.lambda_delta,  # weight of contrastive term
            "proto_loss": use_proto,
            "proto_alpha": args.proto_alpha if use_proto else None,
            "model_path": MODEL_PATH,
        },
        "train_losses": train_losses,
        "best_loss": best_loss,
        "n_trainable_params": n_trainable,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {results_path}")

    del model
    if proto_head is not None:
        del proto_head
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 20 Direction C: Contrastive Learning (replaces δ)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: contrastive LoRA | eval: evaluate vs baseline",
    )
    # Training
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for softmax contrastive over channel gains. "
        "Lower = sharper (closer to hard argmax). "
        "Sweep candidate: [0.1, 0.3, 0.5, 0.7, 1.0].",
    )
    parser.add_argument(
        "--lambda_delta",
        type=float,
        default=0.1,
        help="Weight of contrastive loss term (same role as λ in δ-corrective).",
    )
    # Prototype auxiliary loss
    parser.add_argument(
        "--proto_loss",
        action="store_true",
        help="Add hidden-state prototype auxiliary loss.",
    )
    parser.add_argument(
        "--proto_alpha",
        type=float,
        default=0.1,
        help="Weight of prototype loss relative to CE + contrastive.",
    )
    # Eval
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Explicit path to LoRA adapter for eval.",
    )
    parser.add_argument(
        "--skip_lora",
        action="store_true",
        help="Evaluate baseline only.",
    )
    # Shared
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path override.",
    )
    args = parser.parse_args()

    if args.model_path:
        global MODEL_PATH
        MODEL_PATH = args.model_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        train_contrastive(args)
    elif args.mode == "eval":
        # Re-use evaluation from train_lora_delta
        # Build a synthetic args for evaluate()
        class EvalArgs:
            pass

        eval_args = EvalArgs()
        eval_args.mode = "eval"
        eval_args.n_test = args.n_test
        eval_args.seed = args.seed
        eval_args.model_path = args.model_path
        eval_args.skip_lora = args.skip_lora
        if args.lora_checkpoint:
            eval_args.lora_checkpoint = args.lora_checkpoint
        else:
            eval_args.lora_checkpoint = str(
                _get_contrastive_dir(
                    args.temperature, getattr(args, "proto_loss", False)
                )
                / "epoch_1"
            )
        eval_args.lambda_delta = args.lambda_delta  # for results path naming
        evaluate_delta(eval_args)


if __name__ == "__main__":
    main()
