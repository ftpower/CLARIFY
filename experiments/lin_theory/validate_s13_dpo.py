"""Phase 13.3: DPO Truth Reward Fine-tuning.

Theory: docs/theory-intervention-failure.md Section 13.4.
Three modes: build (preference pairs), train (DPO LoRA), eval (evaluate).

Usage:
  python validate_s13_dpo.py --mode build --n_train 500 --n_calibrate 100
  python validate_s13_dpo.py --mode train --betas 0.1 0.3 0.5
  python validate_s13_dpo.py --mode eval
"""

import argparse, json, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from src.data_loader import load_triviaqa, load_hellaswag, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    greedy_generate,
    get_first_answer_token_id,
)

# ── Constants ──
MODEL_PATH = (
    "/home/user_ft/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)
LAYER = 27
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"
PAIRS_FILE = OUTPUT_DIR / "s13_dpo_pairs.json"
RESULTS_FILE = OUTPUT_DIR / "s13_dpo_results.json"


# ═══════════════════════════════════════════════════════════
# Helper: temperature generation
# ═══════════════════════════════════════════════════════════


def _sample_token(logits, temperature):
    if temperature <= 0:
        return int(logits.argmax().item())
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def temperature_generate(model, prompt_tokens, device, temperature, max_new=20):
    """Generate with temperature sampling. Returns (gids, full_tokens)."""
    tokens = prompt_tokens.clone()
    gids = []
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(tokens)
        nid = _sample_token(logits[0, -1, :], temperature)
        if nid == model.tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
    return gids, tokens


def _find_last_layer(hf_model):
    """Find the last transformer layer module, handling bare HF and PeftModel."""
    # Try PeftModel path
    try:
        return hf_model.base_model.model.model.layers[-1]
    except (AttributeError, TypeError):
        pass
    # Try bare HF model path
    try:
        return hf_model.model.layers[-1]
    except (AttributeError, TypeError):
        pass
    # Fallback: search by name
    for name, module in hf_model.named_modules():
        if name.endswith(".layers.27") or name.endswith(".layers.27"):
            return module
    raise RuntimeError("Cannot find last transformer layer")


def _to_1d(t):
    """Ensure tensor is 1D [d_model], taking first batch item if 2D."""
    if t.dim() == 2:
        return t[0]
    return t


def extract_h_at_last_token(model, tokens, device):
    """Extract hidden state at last position of tokens via hook."""
    hook_name = f"blocks.{LAYER}.hook_resid_post"
    residual = {}

    def _hook(act, hook=None):
        residual["h"] = act[:, -1, :].detach()
        return act

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])
    return residual["h"].float().squeeze(0)


# ═══════════════════════════════════════════════════════════
# Mode 1: Build preference pairs
# ═══════════════════════════════════════════════════════════


def compute_v_answer(model, tokenizer, n_calibrate, device, seed=42):
    """Compute v from answer-level h (not prompt-level)."""
    samples = load_triviaqa(n_samples=n_calibrate, seed=seed)
    h_correct, h_incorrect = [], []
    for s in tqdm(samples, desc="  Calibrate v_answer"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        gids, full_tokens = temperature_generate(
            model, tokens, device, temperature=0, max_new=20
        )
        ans = model.tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")
        h_vec = extract_h_at_last_token(model, full_tokens, device).cpu().numpy()
        if is_correct:
            h_correct.append(h_vec)
        else:
            h_incorrect.append(h_vec)
    if len(h_correct) == 0 or len(h_incorrect) == 0:
        raise RuntimeError(
            f"Need at least 1 correct+incorrect, got {len(h_correct)}/{len(h_incorrect)}"
        )
    h_correct = np.stack(h_correct, 0)
    h_incorrect = np.stack(h_incorrect, 0)
    v = h_correct.mean(0) - h_incorrect.mean(0)
    v = v / np.linalg.norm(v)
    return torch.from_numpy(v).float().to(device), {
        "n_correct": len(h_correct),
        "n_incorrect": len(h_incorrect),
        "v_norm_raw": float(np.linalg.norm(v)),
        "type": "answer_level",
    }


def build_preference_pairs(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(
        f"Phase 13.3a: Build pairs | n_train={args.n_train} n_calibrate={args.n_calibrate} T={args.temperatures}"
    )

    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, _, _, _ = load_model_and_unembed(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"\n[2/4] Computing v_answer...")
    t0 = time.time()
    v, v_stats = compute_v_answer(model, tokenizer, args.n_calibrate, device, args.seed)
    print(
        f"  v_answer in {time.time() - t0:.1f}s | correct={v_stats['n_correct']} incorrect={v_stats['n_incorrect']}"
    )

    print(f"\n[3/4] Loading {args.n_train} TriviaQA samples...")
    samples = load_triviaqa(n_samples=args.n_train, seed=args.seed)

    print(f"\n[4/4] Generating candidates...")
    pairs, stats = (
        [],
        {
            "chosen_correct": 0,
            "rejected_correct": 0,
            "chosen_better_gt": 0,
            "total": 0,
            "gaps": [],
        },
    )

    for s in tqdm(samples, desc="  Building pairs"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        candidates = []
        for T in args.temperatures:
            gids, full_tokens = temperature_generate(
                model, tokens.clone(), device, T, max_new=20
            )
            ans = tokenizer.decode(gids).strip()
            h_last = (
                extract_h_at_last_token(model, full_tokens, device)
                if gids
                else extract_h_at_last_token(model, tokens, device)
            )
            score = float(torch.dot(v, h_last.to(device)).item())
            candidates.append(
                {
                    "text": ans,
                    "score": score,
                    "correct": check_correct(ans, s["answers"], dataset="triviaqa"),
                }
            )

        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda c: c["score"], reverse=True)
        chosen, rejected = candidates[0], candidates[-1]
        pairs.append(
            {
                "question": s["question"],
                "context": s.get("context", ""),
                "answers": s["answers"],
                "chosen": {"text": chosen["text"], "score": chosen["score"]},
                "rejected": {"text": rejected["text"], "score": rejected["score"]},
            }
        )
        if chosen["correct"]:
            stats["chosen_correct"] += 1
        if rejected["correct"]:
            stats["rejected_correct"] += 1
        if chosen["correct"] and not rejected["correct"]:
            stats["chosen_better_gt"] += 1
        stats["total"] += 1
        stats["gaps"].append(chosen["score"] - rejected["score"])

    print(
        f"\n  Pairs: {len(pairs)} | Chosen correct: {stats['chosen_correct'] / max(stats['total'], 1):.1%} | "
        f"Rejected correct: {stats['rejected_correct'] / max(stats['total'], 1):.1%} | "
        f"Chosen better by GT: {stats['chosen_better_gt'] / max(stats['total'], 1):.1%} | "
        f"Mean gap: {np.mean(stats['gaps']):.1f}"
    )

    with open(PAIRS_FILE, "w") as f:
        json.dump(
            {
                "config": {
                    "n_train": args.n_train,
                    "n_calibrate": args.n_calibrate,
                    "temperatures": args.temperatures,
                    "layer": LAYER,
                    "seed": args.seed,
                },
                "v_numpy": v.cpu().numpy().tolist(),
                "v_stats": v_stats,
                "pairs": pairs,
                "stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Saved to {PAIRS_FILE}")


# ═══════════════════════════════════════════════════════════
# Mode 2: DPO Training
# ═══════════════════════════════════════════════════════════


class PreferenceDataset(Dataset):
    def __init__(
        self, pairs, tokenizer, max_length=512, ref_lp_chosen=None, ref_lp_rejected=None
    ):
        self.data = []
        eos = tokenizer.eos_token_id
        for i, pair in enumerate(pairs):
            prompt_ids = tokenizer.encode(
                format_prompt(pair["question"], pair["context"], dataset="triviaqa"),
                add_special_tokens=True,
            )
            c_ids = tokenizer.encode(pair["chosen"]["text"], add_special_tokens=False)
            r_ids = tokenizer.encode(pair["rejected"]["text"], add_special_tokens=False)
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[-max_length:]
            if c_ids and c_ids[-1] != eos:
                c_ids.append(eos)
            if r_ids and r_ids[-1] != eos:
                r_ids.append(eos)
            item = {
                "prompt_ids": prompt_ids,
                "chosen_ids": c_ids,
                "rejected_ids": r_ids,
            }
            if ref_lp_chosen is not None:
                item["ref_lp_chosen"] = float(ref_lp_chosen[i])
            if ref_lp_rejected is not None:
                item["ref_lp_rejected"] = float(ref_lp_rejected[i])
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_dpo_batch(batch, tokenizer):
    c_in, r_in, c_lab, r_lab, rc, rr = [], [], [], [], [], []
    for item in batch:
        p = item["prompt_ids"]
        c_in.append(torch.tensor(p + item["chosen_ids"], dtype=torch.long))
        r_in.append(torch.tensor(p + item["rejected_ids"], dtype=torch.long))
        c_lab.append(
            torch.tensor([-100] * len(p) + item["chosen_ids"], dtype=torch.long)
        )
        r_lab.append(
            torch.tensor([-100] * len(p) + item["rejected_ids"], dtype=torch.long)
        )
        rc.append(item.get("ref_lp_chosen", 0.0))
        rr.append(item.get("ref_lp_rejected", 0.0))
    pad = tokenizer.pad_token_id or 0
    return {
        "chosen_input_ids": torch.nn.utils.rnn.pad_sequence(
            c_in, batch_first=True, padding_value=pad
        ),
        "rejected_input_ids": torch.nn.utils.rnn.pad_sequence(
            r_in, batch_first=True, padding_value=pad
        ),
        "chosen_labels": torch.nn.utils.rnn.pad_sequence(
            c_lab, batch_first=True, padding_value=-100
        ),
        "rejected_labels": torch.nn.utils.rnn.pad_sequence(
            r_lab, batch_first=True, padding_value=-100
        ),
        "ref_lp_chosen": torch.tensor(rc, dtype=torch.float32),
        "ref_lp_rejected": torch.tensor(rr, dtype=torch.float32),
    }


def compute_seq_log_prob(model, input_ids, labels):
    """Compute sum of log P over answer tokens. Memory-efficient: fp16 logsumexp."""
    with torch.cuda.amp.autocast():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100).float()
    shift_labels_clamped = shift_labels.clamp(min=0)
    target_logits = shift_logits.gather(
        dim=-1, index=shift_labels_clamped.unsqueeze(-1)
    ).squeeze(-1)
    log_sum_exp = torch.logsumexp(shift_logits, dim=-1)
    return ((target_logits.float() - log_sum_exp.float()) * mask).sum(dim=-1)


def precompute_ref_log_probs(ref_model, dataset, tokenizer, device, batch_size=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_dpo_batch(b, tokenizer),
    )
    all_c, all_r = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Precomputing ref log probs"):
            all_c.append(
                compute_seq_log_prob(
                    ref_model,
                    batch["chosen_input_ids"].to(device),
                    batch["chosen_labels"].to(device),
                ).cpu()
            )
            all_r.append(
                compute_seq_log_prob(
                    ref_model,
                    batch["rejected_input_ids"].to(device),
                    batch["rejected_labels"].to(device),
                ).cpu()
            )
    return torch.cat(all_c), torch.cat(all_r)


def train_dpo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(
        f"Phase 13.3b: DPO Training | betas={args.betas} lr={args.lr} epochs={args.epochs} bs={args.batch_size} grad_accum={args.grad_accum}"
    )

    # Load pairs
    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    all_pairs = pairs_data["pairs"]
    v = torch.tensor(pairs_data["v_numpy"], dtype=torch.float32)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_pairs))
    split = int(len(all_pairs) * 0.8)
    train_pairs = [all_pairs[i] for i in indices[:split]]
    val_pairs = [all_pairs[i] for i in indices[split:]]
    print(f"\n[1/5] Pairs: train={len(train_pairs)} val={len(val_pairs)}")

    # Load tokenizer + ref model + precompute ref log probs
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)

    print("[2/5] Precomputing reference log probs...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_simple = PreferenceDataset(train_pairs, tokenizer)
    val_simple = PreferenceDataset(val_pairs, tokenizer)
    train_ref_c, train_ref_r = precompute_ref_log_probs(
        ref_model, train_simple, tokenizer, device, args.batch_size
    )
    val_ref_c, val_ref_r = precompute_ref_log_probs(
        ref_model, val_simple, tokenizer, device, args.batch_size
    )
    print(
        f"  Ref: train chosen={train_ref_c.mean():.2f} rejected={train_ref_r.mean():.2f}"
    )

    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    # Create datasets with ref log probs
    train_dataset = PreferenceDataset(
        train_pairs, tokenizer, ref_lp_chosen=train_ref_c, ref_lp_rejected=train_ref_r
    )
    val_dataset = PreferenceDataset(
        val_pairs, tokenizer, ref_lp_chosen=val_ref_c, ref_lp_rejected=val_ref_r
    )

    # Train per beta
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[3/5] DPO training for beta in {args.betas}")
    all_results = {}

    for beta in args.betas:
        print(f"\n  beta={beta}")
        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
        ).to(device)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        policy_model = get_peft_model(policy_model, lora_config)
        if hasattr(policy_model, "gradient_checkpointing_enable"):
            policy_model.gradient_checkpointing_enable()
        policy_model.train()
        print(
            f"  Trainable: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad):,}"
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
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            policy_model.train()
            epoch_loss, n_steps = 0.0, 0
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"    Epoch {epoch + 1}/{args.epochs}")

            for step, batch in enumerate(pbar):
                c_ids = batch["chosen_input_ids"].to(device)
                c_lab = batch["chosen_labels"].to(device)
                r_ids = batch["rejected_input_ids"].to(device)
                r_lab = batch["rejected_labels"].to(device)
                rc = batch["ref_lp_chosen"].to(device)
                rr = batch["ref_lp_rejected"].to(device)

                lp_c = compute_seq_log_prob(policy_model, c_ids, c_lab)
                lp_r = compute_seq_log_prob(policy_model, r_ids, r_lab)
                ratio_c = lp_c - rc
                ratio_r = lp_r - rr
                loss = (
                    -F.logsigmoid(beta * (ratio_c - ratio_r)).mean() / args.grad_accum
                )
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum
                n_steps += 1

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * args.grad_accum:.4f}",
                        "rew": f"{(ratio_c - ratio_r).mean().item():.3f}",
                        "acc": f"{(ratio_c > ratio_r).float().mean().item():.2f}",
                    }
                )
                del c_ids, c_lab, r_ids, r_lab, rc, rr, lp_c, lp_r, loss
                if step % 20 == 0:
                    torch.cuda.empty_cache()

            avg_loss = epoch_loss / max(n_steps, 1)
            # Validation
            policy_model.eval()
            val_loss, val_acc, val_n = 0.0, 0.0, 0
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_dpo_batch(b, tokenizer),
            )
            with torch.no_grad():
                for v_batch in val_loader:
                    c_ids = v_batch["chosen_input_ids"].to(device)
                    c_lab = v_batch["chosen_labels"].to(device)
                    r_ids = v_batch["rejected_input_ids"].to(device)
                    r_lab = v_batch["rejected_labels"].to(device)
                    rc = v_batch["ref_lp_chosen"].to(device)
                    rr = v_batch["ref_lp_rejected"].to(device)
                    lp_c = compute_seq_log_prob(policy_model, c_ids, c_lab)
                    lp_r = compute_seq_log_prob(policy_model, r_ids, r_lab)
                    vloss = (
                        -F.logsigmoid(beta * ((lp_c - rc) - (lp_r - rr))).mean().item()
                    )
                    val_loss += vloss
                    val_acc += (lp_c - rc > lp_r - rr).float().mean().item()
                    val_n += 1
                    del c_ids, c_lab, r_ids, r_lab, lp_c, lp_r
            val_loss /= max(val_n, 1)
            val_acc /= max(val_n, 1)
            print(
                f"    Epoch {epoch + 1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                adapter_path = OUTPUT_DIR / f"s13_dpo_adapter_beta{beta}"
                policy_model.save_pretrained(str(adapter_path))
                print(f"    -> Saved adapter to {adapter_path}")

        all_results[f"beta_{beta}"] = {"best_val_loss": best_val_loss}
        del policy_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[5/5] Done. Summary:")
    for beta in args.betas:
        print(
            f"  beta={beta}: best_val_loss={all_results[f'beta_{beta}']['best_val_loss']:.4f}"
        )

    existing = json.load(open(RESULTS_FILE)) if RESULTS_FILE.exists() else {}
    existing["train"] = {"config": vars(args), "results": all_results}
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# Mode 3: Evaluation
# ═══════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_triviaqa(model, tokenizer, v, samples, device, max_new=20):
    correct, scores, total = 0, [], 0
    last_layer = _find_last_layer(model)
    for s in tqdm(samples, desc="  TriviaQA"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        if prompt_ids.shape[1] > 1024:
            prompt_ids = prompt_ids[:, :1024]
        gen_ids = prompt_ids.clone()
        generated = []
        for _ in range(max_new):
            with torch.no_grad():
                out = model(input_ids=gen_ids)
            nid = int(out.logits[0, -1, :].argmax().item())
            if nid == tokenizer.eos_token_id:
                break
            generated.append(nid)
            gen_ids = torch.cat([gen_ids, torch.tensor([[nid]], device=device)], dim=1)
        ans = tokenizer.decode(generated).strip()
        # v·h via hook (memory efficient)
        captured = {}

        def _hook(m, i, o):
            hs = o[0]
            captured["h"] = hs[:, -1, :].detach() if hs.dim() == 3 else hs.detach()

        h = last_layer.register_forward_hook(_hook)
        model(input_ids=gen_ids)
        h.remove()
        scores.append(
            float(torch.dot(v.to(device), _to_1d(captured["h"].float())).item())
        )
        if check_correct(ans, s["answers"], dataset="triviaqa"):
            correct += 1
        total += 1
    return {
        "accuracy": correct / max(total, 1),
        "n_correct": correct,
        "n_total": total,
        "vh_mean": float(np.mean(scores)),
        "vh_std": float(np.std(scores)),
    }


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, samples, device):
    correct, total = 0, 0
    for s in tqdm(samples, desc="  HellaSwag"):
        prompt = format_prompt(s["question"], s["context"], dataset="hellaswag")
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(
            device
        )
        if ids.shape[1] > 1024:
            ids = ids[:, :1024]
        nid = int(model(input_ids=ids).logits[0, -1, :].argmax().item())
        if check_correct(
            tokenizer.decode([nid]).strip(), s["answers"], dataset="hellaswag"
        ):
            correct += 1
        total += 1
    return {"accuracy": correct / max(total, 1), "n_correct": correct, "n_total": total}


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Phase 13.3c: Evaluation | n_test={args.n_test} n_hellaswag={args.n_hellaswag}"
    )

    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    v = torch.tensor(pairs_data["v_numpy"], dtype=torch.float32)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)
    triviaqa_test = load_triviaqa(n_samples=args.n_test, seed=args.seed + 999)
    hellaswag_test = load_hellaswag(n_samples=args.n_hellaswag, seed=args.seed + 999)

    # Baseline
    print("\n[1/3] Baseline evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    base_model.eval()
    bl_tqa = evaluate_triviaqa(base_model, tokenizer, v, triviaqa_test, device)
    bl_hs = evaluate_hellaswag(base_model, tokenizer, hellaswag_test, device)
    print(f"  TriviaQA baseline: {bl_tqa['accuracy']:.1%} vh={bl_tqa['vh_mean']:.4f}")
    print(f"  HellaSwag baseline: {bl_hs['accuracy']:.1%}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    all_eval = {"baseline": {"triviaqa": bl_tqa, "hellaswag": bl_hs}}

    for beta in args.betas:
        adapter_path = OUTPUT_DIR / f"s13_dpo_adapter_beta{beta}"
        if not adapter_path.exists():
            print(f"\n  beta={beta}: adapter not found, skipping")
            continue
        print(f"\n[2/3] Evaluating beta={beta}...")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
        ).to(device)
        policy = PeftModel.from_pretrained(base, str(adapter_path))
        policy.eval()
        tqa = evaluate_triviaqa(policy, tokenizer, v, triviaqa_test, device)
        hs = evaluate_hellaswag(policy, tokenizer, hellaswag_test, device)
        d_acc = tqa["accuracy"] - bl_tqa["accuracy"]
        d_hs = hs["accuracy"] - bl_hs["accuracy"]
        d_vh = tqa["vh_mean"] - bl_tqa["vh_mean"]
        print(
            f"  TriviaQA: {tqa['accuracy']:.1%} (delta={d_acc:+.1%}) vh_delta={d_vh:+.4f}"
        )
        print(f"  HellaSwag: {hs['accuracy']:.1%} (delta={d_hs:+.1%})")
        all_eval[f"beta_{beta}"] = {
            "triviaqa": tqa,
            "hellaswag": hs,
            "delta_acc": d_acc,
            "delta_hs": d_hs,
            "delta_vh": d_vh,
        }
        del base, policy
        gc.collect()
        torch.cuda.empty_cache()

    # Gates
    print("\n[3/3] Gate check:")
    for beta in args.betas:
        k = f"beta_{beta}"
        if k not in all_eval:
            continue
        e = all_eval[k]
        t1 = "PASS" if e["delta_vh"] > 0 else "FAIL"
        t2 = "PASS" if e["delta_acc"] > 0.05 else "FAIL"
        t3 = "PASS" if e["delta_acc"] >= 0 else "FAIL"
        t4 = "PASS" if e["delta_hs"] > -0.03 else "FAIL"
        print(
            f"  beta={beta}: T1(vh_up)={t1} T2(acc_gt_5%)={t2} T3(val_ge_0)={t3} T4(hs_gt_-3%)={t4}"
        )

    existing = json.load(open(RESULTS_FILE)) if RESULTS_FILE.exists() else {}
    existing["eval"] = {"config": vars(args), "results": all_eval}
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {RESULTS_FILE}")


# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13.3: DPO Truth Reward Fine-tuning"
    )
    parser.add_argument("--mode", choices=["build", "train", "eval"], required=True)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_calibrate", type=int, default=100)
    parser.add_argument("--temperatures", type=float, nargs="*", default=[0.3, 1.0])
    parser.add_argument("--betas", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--n_hellaswag", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "build":
        build_preference_pairs(args)
    elif args.mode == "train":
        train_dpo(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
