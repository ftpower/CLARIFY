"""Phase 20.2: DPO Token-Preference Training — first-token y_true vs argmax pairs.

Theory: docs/theory-intervention-failure.md
Plan:   ~/.claude/plans/CLARIFY/phase20-training-intervention.md §20.2

Core idea:
  Phase 13.3 DPO failed because reward was v·h (can answer "right or wrong" but
  not "what IS right"). Instead, directly use y_true as chosen and argmax(L27)
  as rejected — first-token only. This directly optimizes accuracy.

  y_w = y_true first token (chosen)
  y_l = argmax(L27) at last prompt position (rejected)
  Filter: only keep samples where y_w ≠ y_l

  DPO loss (Rafailov et al. 2023), first-token variant:
    log_ratio_w = log P_θ(y_w|prompt) - log P_ref(y_w|prompt)
    log_ratio_l = log P_θ(y_l|prompt) - log P_ref(y_l|prompt)
    loss = -log σ(β * (log_ratio_w - log_ratio_l))

  LoRA: r=16, α=32, target=["q_proj","v_proj","o_proj"], all layers

Gates (see plan §20.2.6):
  P20.2.2: KW first-token accuracy Δ > 0
  P20.2.3: KC first-token accuracy degradation ≤ 1 sample

Usage:
  python train_dpo_token.py --mode train --n_train 500 --n_test 100
  python train_dpo_token.py --mode eval
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
RANK_THRESHOLD = 50
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"
DPO_LORA_DIR = OUTPUT_DIR / "s20_2_dpo"
DPO_PAIRS_FILE = OUTPUT_DIR / "s20_2_dpo_pairs.json"
RESULTS_PATH = OUTPUT_DIR / "s20_2_dpo.json"

MODEL_ID = "Qwen/Qwen3-1.7B"


def _find_model_path() -> str:
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if os.path.isdir(local):
            if os.path.isfile(os.path.join(local, "config.json")):
                return local
            snaps = os.path.join(local, "snapshots")
            if os.path.isdir(snaps):
                for s in sorted(os.listdir(snaps)):
                    sp = os.path.join(snaps, s)
                    if os.path.isfile(os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID


MODEL_PATH = _find_model_path()


# ═══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════════


def load_triviaqa_train(n_samples: int, seed: int = 42) -> list[dict]:
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
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Preference pair construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_preference_pairs(
    model, tokenizer, samples: list[dict], device: str
) -> tuple[list[dict], dict]:
    """Build first-token preference pairs from TriviaQA samples.

    For each sample:
      y_w = first token of y_true (chosen)
      y_l = argmax of L27 logits (rejected)
      Filter: y_w ≠ y_l

    Also pre-computes reference log-probs for both y_w and y_l.
    """
    pairs = []
    stats = {"total": len(samples), "y_w_eq_y_l": 0, "y_w_none": 0, "valid": 0}

    for s in tqdm(samples, desc="  Building pairs"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            stats["y_w_none"] += 1
            continue

        # Get L27 logits at last prompt position
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, :1024]

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :].float()  # [vocab_size]
        y_l_id = int(logits.argmax().item())

        if y_true_id == y_l_id:
            stats["y_w_eq_y_l"] += 1
            continue  # no preference signal (model already outputs correct)

        # Pre-compute reference log-probs
        log_probs = torch.log_softmax(logits, dim=-1)
        ref_lp_w = log_probs[y_true_id].item()
        ref_lp_l = log_probs[y_l_id].item()

        pairs.append(
            {
                "question": s["question"],
                "answers": s["answers"],
                "context": s.get("context", ""),
                "prompt": prompt,
                "y_w_id": y_true_id,
                "y_l_id": y_l_id,
                "ref_lp_w": ref_lp_w,
                "ref_lp_l": ref_lp_l,
                "y_w_text": tokenizer.decode([y_true_id]),
                "y_l_text": tokenizer.decode([y_l_id]),
            }
        )
        stats["valid"] += 1

    return pairs, stats


class DPODataset(Dataset):
    """First-token DPO dataset: prompt → (y_w, y_l, ref_lp_w, ref_lp_l)."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 768):
        self.data = []
        for p in pairs:
            prompt_ids = tokenizer.encode(
                format_prompt(p["question"], p.get("context", ""), dataset="triviaqa"),
                add_special_tokens=True,
            )
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[-max_length:]
            self.data.append(
                {
                    "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                    "y_w_id": p["y_w_id"],
                    "y_l_id": p["y_l_id"],
                    "ref_lp_w": p["ref_lp_w"],
                    "ref_lp_l": p["ref_lp_l"],
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_dpo_batch(batch: list[dict], tokenizer) -> dict:
    """Pad prompt sequences."""
    pad_id = tokenizer.pad_token_id or 0
    prompt_ids = torch.nn.utils.rnn.pad_sequence(
        [b["prompt_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    return {
        "prompt_ids": prompt_ids,
        "y_w_ids": torch.tensor([b["y_w_id"] for b in batch], dtype=torch.long),
        "y_l_ids": torch.tensor([b["y_l_id"] for b in batch], dtype=torch.long),
        "ref_lp_w": torch.tensor([b["ref_lp_w"] for b in batch], dtype=torch.float32),
        "ref_lp_l": torch.tensor([b["ref_lp_l"] for b in batch], dtype=torch.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_dpo(args):
    """Build preference pairs + DPO training."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Phase 20.2: DPO Token-Preference Training | n_train={args.n_train}")

    # ── 1. Load model + tokenizer ─────────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Build preference pairs ─────────────────────────────────────────
    print(f"\n[2/5] Building preference pairs from {args.n_train} samples...")
    train_samples = load_triviaqa_train(n_samples=args.n_train, seed=args.seed)
    pairs, pair_stats = build_preference_pairs(
        ref_model, tokenizer, train_samples, device
    )
    print(
        f"  Valid pairs: {pair_stats['valid']}/{args.n_train} "
        f"(y_w=y_l: {pair_stats['y_w_eq_y_l']}, y_w=None: {pair_stats['y_w_none']})"
    )

    if len(pairs) < 10:
        print("  ERROR: Too few valid pairs for DPO training!")
        del ref_model
        return

    # Save pairs
    with open(DPO_PAIRS_FILE, "w") as f:
        json.dump(
            {
                "config": {"n_train": args.n_train, "seed": args.seed},
                "stats": pair_stats,
                "n_pairs": len(pairs),
                "pairs": pairs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Pairs saved to {DPO_PAIRS_FILE}")

    # Split train/val (80/20)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(pairs))
    split = int(len(pairs) * 0.8)
    train_pairs = [pairs[i] for i in indices[:split]]
    val_pairs = [pairs[i] for i in indices[split:]]
    print(f"  Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    train_dataset = DPODataset(train_pairs, tokenizer)
    val_dataset = DPODataset(val_pairs, tokenizer)

    # Free reference model after pair construction
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Create policy model with LoRA ──────────────────────────────────
    print(f"\n[3/5] Creating policy model with LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
    )
    policy_model = get_peft_model(policy_model, lora_config)
    if hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
    policy_model.train()
    n_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # ── 4. Train ──────────────────────────────────────────────────────────
    print(
        f"\n[4/5] DPO Training | beta={args.beta} lr={args.lr} "
        f"epochs={args.epochs} batch_size={args.batch_size}"
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dpo_batch(b, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_dpo_batch(b, tokenizer),
    )

    train_losses = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        policy_model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            prompt_ids = batch["prompt_ids"].to(device)
            y_w_ids = batch["y_w_ids"].to(device)  # [B]
            y_l_ids = batch["y_l_ids"].to(device)  # [B]
            ref_lp_w = batch["ref_lp_w"].to(device)  # [B]
            ref_lp_l = batch["ref_lp_l"].to(device)  # [B]
            B = prompt_ids.shape[0]

            # Forward: get logits at last position
            outputs = policy_model(input_ids=prompt_ids)
            logits = outputs.logits[:, -1, :].float()  # [B, vocab_size]

            # Policy log-probs for y_w and y_l (first-token only)
            log_probs = torch.log_softmax(logits, dim=-1)  # [B, vocab_size]
            policy_lp_w = log_probs[torch.arange(B), y_w_ids]  # [B]
            policy_lp_l = log_probs[torch.arange(B), y_l_ids]  # [B]

            # DPO loss: -log σ(β * (log_ratio_w - log_ratio_l))
            log_ratio_w = policy_lp_w - ref_lp_w
            log_ratio_l = policy_lp_l - ref_lp_l
            loss = -F.logsigmoid(args.beta * (log_ratio_w - log_ratio_l)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (log_ratio_w > log_ratio_l).float().mean().item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{(log_ratio_w > log_ratio_l).float().mean().item():.2f}",
                    "gap": f"{(log_ratio_w - log_ratio_l).mean().item():.3f}",
                }
            )

            del outputs, logits, log_probs
            if n_batches % 20 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)

        # Validation
        policy_model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_n = 0
        with torch.no_grad():
            for v_batch in val_loader:
                prompt_ids = v_batch["prompt_ids"].to(device)
                y_w_ids = v_batch["y_w_ids"].to(device)
                y_l_ids = v_batch["y_l_ids"].to(device)
                ref_lp_w = v_batch["ref_lp_w"].to(device)
                ref_lp_l = v_batch["ref_lp_l"].to(device)
                Bv = prompt_ids.shape[0]

                outputs = policy_model(input_ids=prompt_ids)
                logits = outputs.logits[:, -1, :].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                policy_lp_w = log_probs[torch.arange(Bv), y_w_ids]
                policy_lp_l = log_probs[torch.arange(Bv), y_l_ids]
                log_ratio_w = policy_lp_w - ref_lp_w
                log_ratio_l = policy_lp_l - ref_lp_l
                v_loss = (
                    -F.logsigmoid(args.beta * (log_ratio_w - log_ratio_l)).mean().item()
                )
                val_loss += v_loss
                val_acc += (log_ratio_w > log_ratio_l).float().mean().item()
                val_n += 1
                del outputs, logits, log_probs

        val_loss /= max(val_n, 1)
        val_acc /= max(val_n, 1)
        train_losses.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_acc": avg_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"    Epoch {epoch + 1}: train_loss={avg_loss:.4f} train_acc={avg_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy_model.save_pretrained(str(DPO_LORA_DIR))
            print(f"    -> Saved LoRA adapter to {DPO_LORA_DIR}")

    # ── 5. Save metadata ──────────────────────────────────────────────────
    print(f"\n[5/5] Saving metadata...")
    results = {
        "config": {
            "phase": "20.2",
            "n_train": args.n_train,
            "seed": args.seed,
            "target_modules": ["q_proj", "v_proj", "o_proj"],
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "beta": args.beta,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model_path": MODEL_PATH,
        },
        "pair_stats": pair_stats,
        "n_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "train_losses": train_losses,
        "best_val_loss": best_val_loss,
        "n_trainable_params": n_trainable,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {RESULTS_PATH}")

    del policy_model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def classify_sample(
    logits: torch.Tensor, y_true_id: int | None, generated: str, answers: list[str]
) -> str:
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
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :].float().cpu()
    first_token_id = int(logits.argmax().item())

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 20.2: DPO Evaluation | n_test={args.n_test}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)

    # ── Baseline ──────────────────────────────────────────────────────────
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
                "ft_correct": ft_correct,
                "em_correct": em_correct,
                "full_text": full_text,
                "category": category,
            }
        )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── DPO model ─────────────────────────────────────────────────────────
    print("\n[2/3] Evaluating DPO model...")
    if not DPO_LORA_DIR.exists():
        print(f"  ERROR: DPO adapter not found at {DPO_LORA_DIR}")
        return

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    dpo_model = PeftModel.from_pretrained(base, str(DPO_LORA_DIR))
    dpo_model.eval()

    dpo_results = []
    for i, s in enumerate(tqdm(test_samples, desc="  DPO")):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        full_text, ft_id, logits = generate_with_model(
            dpo_model, tokenizer, prompt, device
        )
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        ft_correct = y_true_id is not None and ft_id == y_true_id
        em_correct = check_correct(full_text, s["answers"], dataset="triviaqa")
        dpo_results.append(
            {
                "ft_correct": ft_correct,
                "em_correct": em_correct,
                "full_text": full_text,
                # category from baseline (property of the sample, not the model)
            }
        )
    del base, dpo_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary & Gates ───────────────────────────────────────────────────
    print(f"\n[3/3] Computing summary...")
    n = len(test_samples)

    # Per-category from baseline classification
    categories = ["KC", "KW", "DK"]
    cat_stats = {
        cat: {"n": 0, "bl_ft": 0, "dpo_ft": 0, "bl_em": 0, "dpo_em": 0}
        for cat in categories
    }
    for i in range(n):
        cat = bl_results[i]["category"]
        cat_stats[cat]["n"] += 1
        if bl_results[i]["ft_correct"]:
            cat_stats[cat]["bl_ft"] += 1
        if dpo_results[i]["ft_correct"]:
            cat_stats[cat]["dpo_ft"] += 1
        if bl_results[i]["em_correct"]:
            cat_stats[cat]["bl_em"] += 1
        if dpo_results[i]["em_correct"]:
            cat_stats[cat]["dpo_em"] += 1

    bl_ft = sum(1 for r in bl_results if r["ft_correct"])
    dpo_ft = sum(1 for r in dpo_results if r["ft_correct"])
    bl_em = sum(1 for r in bl_results if r["em_correct"])
    dpo_em = sum(1 for r in dpo_results if r["em_correct"])

    kw_n = cat_stats["KW"]["n"]
    kc_n = cat_stats["KC"]["n"]
    kw_dpo_ft = cat_stats["KW"]["dpo_ft"]
    kw_bl_ft = cat_stats["KW"]["bl_ft"]
    kc_dpo_ft = cat_stats["KC"]["dpo_ft"]
    kc_bl_ft = cat_stats["KC"]["bl_ft"]

    gate_p2022 = kw_dpo_ft > kw_bl_ft  # P20.2.2: KW first-token Δ > 0
    gate_p2023 = (kc_bl_ft - kc_dpo_ft) <= 1  # P20.2.3: KC degradation ≤ 1

    results = {
        "config": {"phase": "20.2", "n_test": n, "seed": args.seed},
        "summary": {
            "n_total": n,
            "baseline_ft": bl_ft / n,
            "dpo_ft": dpo_ft / n,
            "ft_delta": (dpo_ft - bl_ft) / n,
            "baseline_em": bl_em / n,
            "dpo_em": dpo_em / n,
            "em_delta": (dpo_em - bl_em) / n,
            "per_category": cat_stats,
            "gates": {
                "P20.2.2": {
                    "description": "KW first-token accuracy Δ > 0",
                    "baseline_kw_ft": kw_bl_ft,
                    "dpo_kw_ft": kw_dpo_ft,
                    "kw_n": kw_n,
                    "delta": kw_dpo_ft - kw_bl_ft,
                    "pass": gate_p2022,
                },
                "P20.2.3": {
                    "description": "KC first-token degradation ≤ 1",
                    "baseline_kc_ft": kc_bl_ft,
                    "dpo_kc_ft": kc_dpo_ft,
                    "kc_n": kc_n,
                    "degradation": kc_bl_ft - kc_dpo_ft,
                    "pass": gate_p2023,
                },
            },
        },
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  N = {n} | KC={kc_n} KW={kw_n} DK={cat_stats['DK']['n']}")
    print(f"\n  First-token accuracy:")
    print(f"    Baseline: {bl_ft}/{n} = {bl_ft / n:.1%}")
    print(f"    DPO:      {dpo_ft}/{n} = {dpo_ft / n:.1%}")
    print(f"    Delta:    {(dpo_ft - bl_ft) / n:+.1%}")
    print(f"\n  Exact-match accuracy:")
    print(f"    Baseline: {bl_em}/{n} = {bl_em / n:.1%}")
    print(f"    DPO:      {dpo_em}/{n} = {dpo_em / n:.1%}")
    print(f"    Delta:    {(dpo_em - bl_em) / n:+.1%}")
    print(f"\n  Per-category FT:")
    for cat in categories:
        cs = cat_stats[cat]
        bl_a = cs["bl_ft"] / max(cs["n"], 1)
        dpo_a = cs["dpo_ft"] / max(cs["n"], 1)
        print(
            f"    {cat} (n={cs['n']}): baseline={bl_a:.1%} dpo={dpo_a:.1%} delta={dpo_a - bl_a:+.1%}"
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
        description="Phase 20.2: DPO Token-Preference Training"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: build pairs + DPO | eval: evaluate vs baseline",
    )
    # Training
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
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

    if args.model_path:
        global MODEL_PATH
        MODEL_PATH = args.model_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        train_dpo(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
