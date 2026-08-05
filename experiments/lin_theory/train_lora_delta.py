"""Phase 20.1: LoRA δ-Corrective Fine-tuning — reduce L20→L27 distractor amplification.

Theory: docs/theory-intervention-failure.md
Plan:   ~/.claude/plans/CLARIFY/phase20-training-intervention.md §20.1

Core idea:
  TLDC shows that L27 over-amplifies distractor tokens relative to L20.
  Instead of post-hoc logit adjustment (TLDC), use LoRA to modify L20-L27
  Q/V projections so the model doesn't over-amplify distractors in the first place.

  Channel gain: g(t|x) = y_L27(t) - y_L20(t)
  distractor d = argmax(y_L27)

  Loss = CE(y_true) + λ·max(0, g(d) - g(y_true) + m)

  LoRA: r=8, α=16, target=["q_proj","v_proj"], layers 20-27

Gates (see plan §20.1.6):
  P20.1.1: g(d) - g(t*) median decreases > 20%
  P20.1.2: KW exact match Δ > 0
  P20.1.3: KC exact match degradation ≤ 1 sample

Usage:
  python train_lora_delta.py --mode train --n_train 200 --n_test 100
  python train_lora_delta.py --mode eval
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

# ── Constants ──────────────────────────────────────────────────────────────────
LAYER_EARLY = 20  # h_L20 extraction
LAYER_LATE = 27  # h_L27 = final layer
RANK_THRESHOLD = 50
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"
LORA_DIR = OUTPUT_DIR / "s20_1_lora_delta"
RESULTS_PATH = OUTPUT_DIR / "s20_1_lora_delta.json"

# Qwen3-1.7B model path (auto-detected or use cache)
MODEL_ID = "Qwen/Qwen3-1.7B"


def _find_model_path() -> str:
    """Auto-detect local HF model path."""
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
            # Check for config.json directly
            if _os.path.isfile(_os.path.join(local, "config.json")):
                return local
            # Check snapshots
            snaps = _os.path.join(local, "snapshots")
            if _os.path.isdir(snaps):
                for s in sorted(_os.listdir(snaps)):
                    sp = _os.path.join(snaps, s)
                    if _os.path.isfile(_os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID  # fallback


MODEL_PATH = _find_model_path()


# ═══════════════════════════════════════════════════════════════════════════════
# Data: TriviaQA train split + prompt formatting
# ═══════════════════════════════════════════════════════════════════════════════


def load_triviaqa_train(n_samples: int, seed: int = 42) -> list[dict]:
    """Load TriviaQA TRAIN split samples."""
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


def get_first_answer_token_id(tokenizer, answers: list[str]) -> int | None:
    """Return first token ID of the first non-empty answer alias (with leading space)."""
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


class TriviaQADataset(Dataset):
    """Tokenized TriviaQA samples for LoRA training."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 768):
        self.data = []
        for s in samples:
            prompt = format_prompt(
                s["question"], s.get("context", ""), dataset="triviaqa"
            )
            y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
            if y_true_id is None:
                continue
            # Tokenize: prompt + answer (for CE loss on answer token)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[-max_length:]
            # Full input = prompt + answer_token (we compute loss only on answer pos)
            input_ids = prompt_ids + [y_true_id]
            # Labels: -100 for prompt, y_true_id for answer position
            labels = [-100] * len(prompt_ids) + [y_true_id]
            self.data.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "y_true_id": y_true_id,
                    "question": s["question"],
                    "answers": s["answers"],
                    "prompt_len": len(prompt_ids),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_lora_batch(batch: list[dict], tokenizer) -> dict:
    """Pad sequences to max length in batch."""
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "y_true_ids": torch.tensor([b["y_true_id"] for b in batch], dtype=torch.long),
        "prompt_lens": [b["prompt_len"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_lora_delta(args):
    """Main training routine: LoRA δ-corrective fine-tuning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Phase 20.1: LoRA δ-Corrective Training | n_train={args.n_train}")

    # ── 1. Load tokenizer + base model ────────────────────────────────────
    print("\n[1/6] Loading model...")
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
    model.eval()  # base model frozen
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load training data ────────────────────────────────────────────
    print(f"\n[2/6] Loading {args.n_train} TriviaQA train samples...")
    train_samples = load_triviaqa_train(n_samples=args.n_train, seed=args.seed)
    train_dataset = TriviaQADataset(train_samples, tokenizer)
    print(f"  Valid samples: {len(train_dataset)}/{args.n_train}")

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────
    print(f"\n[3/6] Applying LoRA (r={args.lora_r}, α={args.lora_alpha})...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(LAYER_EARLY, LAYER_LATE + 1)),
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # ── 4. Register hook to capture h_L20 ─────────────────────────────────
    h_L20_cache = {}

    def _capture_h_L20(module, input, output):
        # output is a tuple (hidden_states,) or just hidden_states
        hs = output[0] if isinstance(output, tuple) else output
        h_L20_cache["h"] = hs.detach()  # [B, seq, d_model]

    # Find the L20 block
    # With PeftModel: model.base_model.model.model.layers[i]
    # But with modules_to_save or other wrappers, the path may vary
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        try:
            layers = model.model.model.layers
        except AttributeError:
            layers = model.model.layers
    h_L20_handle = layers[LAYER_EARLY].register_forward_hook(_capture_h_L20)

    # Get norm and lm_head for L20 logit computation
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
    print(
        f"\n[5/6] Training | lr={args.lr} batch_size={args.batch_size} "
        f"epochs={args.epochs} lambda={args.lambda_delta} margin={args.margin}"
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
        epoch_delta = 0.0
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            y_true_ids = batch["y_true_ids"].to(device)  # [B]
            B = input_ids.shape[0]

            # Clear cache
            h_L20_cache.clear()

            # Forward through full model (computes CE loss internally if labels given)
            outputs = model(input_ids=input_ids, labels=labels)
            ce_loss = outputs.loss  # scalar, mean over non-masked tokens

            # Get L27 logits at last position
            logits_L27 = outputs.logits.detach()  # [B, seq, vocab]

            # Find the last non-masked position for each sample
            # Labels are -100 for prompt, token_id for answer
            # The answer position is the last non-(-100) position
            last_positions = []
            for b_idx in range(B):
                non_mask = (labels[b_idx] != -100).nonzero(as_tuple=True)[0]
                if len(non_mask) > 0:
                    last_positions.append(non_mask[-1].item())
                else:
                    last_positions.append(labels.shape[1] - 1)

            # Gather L27 logits at answer positions
            logits_L27_last = torch.stack(
                [logits_L27[b, last_positions[b], :] for b in range(B)]
            )  # [B, vocab]

            # Get h_L20 at answer positions → L20 logits
            h_L20 = h_L20_cache.get("h")  # [B, seq, d_model]
            if h_L20 is None:
                # Fallback: skip δ regularization if hook didn't fire
                delta_loss = torch.tensor(0.0, device=device)
            else:
                h_L20_last = torch.stack(
                    [h_L20[b, last_positions[b], :] for b in range(B)]
                )  # [B, d_model]

                # Compute L20 logits
                h_L20_norm = norm(h_L20_last.to(dtype=norm.weight.dtype))
                logits_L20 = lm_head(h_L20_norm)  # [B, vocab]

                # Channel gain: g(t) = y_L27(t) - y_L20(t)
                g_L27 = logits_L27_last.float()
                g_L20 = logits_L20.float()

                # Distractor: argmax of L27 logits
                d_ids = g_L27.argmax(dim=-1)  # [B]

                # g(d) - g(t*) for each sample
                g_d = g_L27[torch.arange(B), d_ids] - g_L20[torch.arange(B), d_ids]
                g_tstar = (
                    g_L27[torch.arange(B), y_true_ids]
                    - g_L20[torch.arange(B), y_true_ids]
                )

                # δ penalty: max(0, g(d) - g(t*) + m)
                penalty = F.relu(g_d - g_tstar + args.margin)
                delta_loss = penalty.mean()

            loss = ce_loss + args.lambda_delta * delta_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_delta += (
                delta_loss.item()
                if isinstance(delta_loss, torch.Tensor)
                else delta_loss
            )
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "delta": f"{delta_loss.item() if hasattr(delta_loss, 'item') else delta_loss:.4f}",
                }
            )

            del input_ids, labels, outputs, logits_L27
            if n_batches % 10 == 0:
                torch.cuda.empty_cache()

        avg_ce = epoch_ce / max(n_batches, 1)
        avg_delta = epoch_delta / max(n_batches, 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(
            {
                "epoch": epoch + 1,
                "ce": avg_ce,
                "delta": avg_delta,
                "total": avg_loss,
            }
        )
        print(
            f"    Epoch {epoch + 1}: ce={avg_ce:.4f} delta={avg_delta:.4f} total={avg_loss:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(str(LORA_DIR))
            print(f"    -> Saved LoRA adapter to {LORA_DIR}")

    # ── 6. Cleanup & save metadata ────────────────────────────────────────
    h_L20_handle.remove()
    print(f"\n[6/6] Saving metadata...")
    results = {
        "config": {
            "phase": "20.1",
            "n_train": args.n_train,
            "n_valid": len(train_dataset),
            "seed": args.seed,
            "layers": f"L{LAYER_EARLY}-L{LAYER_LATE}",
            "target_modules": ["q_proj", "v_proj"],
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lambda_delta": args.lambda_delta,
            "margin": args.margin,
            "model_path": MODEL_PATH,
        },
        "train_losses": train_losses,
        "best_loss": best_loss,
        "n_trainable_params": n_trainable,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {RESULTS_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def classify_sample(
    logits: torch.Tensor, y_true_id: int | None, generated: str, answers: list[str]
) -> str:
    """KC/KW/DK classification."""
    is_correct = check_correct(generated, answers, dataset="triviaqa")
    if is_correct:
        return "KC"
    if y_true_id is None:
        return "DK"
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == y_true_id).nonzero(as_tuple=True)[0].item() + 1
    return "KW" if rank <= RANK_THRESHOLD else "DK"


@torch.no_grad()
def generate_with_model(
    model, tokenizer, prompt: str, device: str, max_new: int = 20
) -> tuple[str, int, torch.Tensor]:
    """Generate answer greedily. Returns (full_text, first_token_id, first_token_logits)."""
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :].float().cpu()  # [vocab_size]
    first_token_id = int(logits.argmax().item())

    # Continue generation
    gen_ids = [first_token_id]
    current_ids = input_ids
    for _ in range(max_new - 1):
        next_tok = torch.tensor([[gen_ids[-1]]], device=device)
        current_ids = torch.cat([current_ids, next_tok], dim=1)
        with torch.no_grad():
            out = model(input_ids=current_ids)
        nid = int(out.logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gen_ids.append(nid)

    full_text = tokenizer.decode(gen_ids).strip()
    return full_text, first_token_id, logits


def evaluate(args):
    """Evaluate LoRA model vs baseline on test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 20.1: LoRA δ-Corrective Evaluation | n_test={args.n_test}")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)

    # ── 2. Evaluate baseline ──────────────────────────────────────────────
    print("\n[1/3] Evaluating baseline...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    base_model.eval()

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    bl_results = []
    for s in tqdm(test_samples, desc="  Baseline"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        full_text, ft_id, logits = generate_with_model(
            base_model, tokenizer, prompt, device
        )
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        ft_correct = y_true_id is not None and ft_id == y_true_id
        em_correct = check_correct(full_text, s["answers"], dataset="triviaqa")
        category = classify_sample(logits, y_true_id, full_text, s["answers"])
        bl_results.append(
            {
                "question": s["question"],
                "answers": s["answers"],
                "y_true_id": y_true_id,
                "ft_id": ft_id,
                "ft_correct": ft_correct,
                "em_correct": em_correct,
                "full_text": full_text,
                "category": category,
            }
        )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Evaluate LoRA model ─────────────────────────────────────────────
    print("\n[2/3] Evaluating LoRA model...")
    if not LORA_DIR.exists():
        print(f"  ERROR: LoRA adapter not found at {LORA_DIR}")
        print(f"  Run 'python train_lora_delta.py --mode train' first.")
        return

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    lora_model = PeftModel.from_pretrained(base, str(LORA_DIR))
    lora_model.eval()

    lora_results = []
    for s in tqdm(test_samples, desc="  LoRA"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        full_text, ft_id, logits = generate_with_model(
            lora_model, tokenizer, prompt, device
        )
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        ft_correct = y_true_id is not None and ft_id == y_true_id
        em_correct = check_correct(full_text, s["answers"], dataset="triviaqa")
        category = classify_sample(logits, y_true_id, full_text, s["answers"])
        lora_results.append(
            {
                "question": s["question"],
                "answers": s["answers"],
                "y_true_id": y_true_id,
                "ft_id": ft_id,
                "ft_correct": ft_correct,
                "em_correct": em_correct,
                "full_text": full_text,
                "category": category,
            }
        )

    del base, lora_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary & Gates ────────────────────────────────────────────────────
    print(f"\n[3/3] Computing summary...")
    n = len(test_samples)
    bl_em = sum(1 for r in bl_results if r["em_correct"])
    lo_em = sum(1 for r in lora_results if r["em_correct"])
    bl_ft = sum(1 for r in bl_results if r["ft_correct"])
    lo_ft = sum(1 for r in lora_results if r["ft_correct"])

    # Per-category
    categories = ["KC", "KW", "DK"]
    cat_stats = {}
    for cat in categories:
        cat_stats[cat] = {
            "n": sum(1 for r in bl_results if r["category"] == cat),
            "bl_em": 0,
            "lo_em": 0,
            "bl_ft": 0,
            "lo_ft": 0,
        }
    for i in range(n):
        cat = bl_results[i]["category"]
        if bl_results[i]["em_correct"]:
            cat_stats[cat]["bl_em"] += 1
        if lora_results[i]["em_correct"]:
            cat_stats[cat]["lo_em"] += 1
        if bl_results[i]["ft_correct"]:
            cat_stats[cat]["bl_ft"] += 1
        if lora_results[i]["ft_correct"]:
            cat_stats[cat]["lo_ft"] += 1

    kw_n = cat_stats["KW"]["n"]
    kc_n = cat_stats["KC"]["n"]

    # Gate P20.1.2: KW exact match Δ > 0
    kw_bl_em = cat_stats["KW"]["bl_em"]
    kw_lo_em = cat_stats["KW"]["lo_em"]
    gate_p2012 = kw_lo_em > kw_bl_em

    # Gate P20.1.3: KC exact match degradation ≤ 1
    kc_bl_em = cat_stats["KC"]["bl_em"]
    kc_lo_em = cat_stats["KC"]["lo_em"]
    kc_degradation = kc_bl_em - kc_lo_em
    gate_p2013 = kc_degradation <= 1

    results = {
        "config": {
            "phase": "20.1",
            "n_test": n,
            "seed": args.seed,
            "rank_threshold": RANK_THRESHOLD,
        },
        "summary": {
            "n_total": n,
            "baseline_em_accuracy": bl_em / n,
            "lora_em_accuracy": lo_em / n,
            "em_delta": (lo_em - bl_em) / n,
            "baseline_ft_accuracy": bl_ft / n,
            "lora_ft_accuracy": lo_ft / n,
            "ft_delta": (lo_ft - bl_ft) / n,
            "per_category": cat_stats,
            "gates": {
                "P20.1.2": {
                    "description": "KW exact match Δ > 0",
                    "baseline_kw_em": kw_bl_em,
                    "lora_kw_em": kw_lo_em,
                    "kw_n": kw_n,
                    "delta": kw_lo_em - kw_bl_em,
                    "pass": gate_p2012,
                },
                "P20.1.3": {
                    "description": "KC exact match degradation ≤ 1",
                    "baseline_kc_em": kc_bl_em,
                    "lora_kc_em": kc_lo_em,
                    "kc_n": kc_n,
                    "degradation": kc_degradation,
                    "pass": gate_p2013,
                },
            },
        },
        "per_sample": [],
    }

    # Per-sample records
    for i in range(n):
        results["per_sample"].append(
            {
                "question": bl_results[i]["question"],
                "category": bl_results[i]["category"],
                "baseline": {
                    "ft_correct": bl_results[i]["ft_correct"],
                    "em_correct": bl_results[i]["em_correct"],
                    "full_text": bl_results[i]["full_text"],
                },
                "lora": {
                    "ft_correct": lora_results[i]["ft_correct"],
                    "em_correct": lora_results[i]["em_correct"],
                    "full_text": lora_results[i]["full_text"],
                },
            }
        )

    # Print
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  N = {n} | KC={kc_n} KW={kw_n} DK={cat_stats['DK']['n']}")
    print(f"\n  Exact-match accuracy:")
    print(f"    Baseline: {bl_em}/{n} = {bl_em / n:.1%}")
    print(f"    LoRA:     {lo_em}/{n} = {lo_em / n:.1%}")
    print(f"    Delta:    {(lo_em - bl_em) / n:+.1%}")
    print(f"\n  First-token accuracy:")
    print(f"    Baseline: {bl_ft}/{n} = {bl_ft / n:.1%}")
    print(f"    LoRA:     {lo_ft}/{n} = {lo_ft / n:.1%}")
    print(f"    Delta:    {(lo_ft - bl_ft) / n:+.1%}")
    print(f"\n  Per-category EM:")
    for cat in categories:
        cs = cat_stats[cat]
        bl_a = cs["bl_em"] / max(cs["n"], 1)
        lo_a = cs["lo_em"] / max(cs["n"], 1)
        print(
            f"    {cat} (n={cs['n']}): baseline={bl_a:.1%} lora={lo_a:.1%} delta={lo_a - bl_a:+.1%}"
        )
    print(f"\n  Gates:")
    for gname, ginfo in results["summary"]["gates"].items():
        status = "✅ PASS" if ginfo["pass"] else "❌ FAIL"
        print(f"    {gname}: {status} — {ginfo['description']}")
    print(f"{'=' * 60}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {RESULTS_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 20.1: LoRA δ-Corrective Fine-tuning"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: LoRA δ-corrective | eval: evaluate vs baseline",
    )
    # Training
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    # Eval
    parser.add_argument("--n_test", type=int, default=100)
    # Shared
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path override (e.g. /root/.../Qwen3-1.7B)",
    )
    args = parser.parse_args()

    # Resolve model path before importing model
    if args.model_path:
        global MODEL_PATH
        MODEL_PATH = args.model_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        train_lora_delta(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
