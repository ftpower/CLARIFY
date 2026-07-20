"""VIB bottleneck experiment: train at L15, evaluate on knowledge-filtered HellaSwag.

Based on AdaVIB architecture adapted for pure-text LLM:
- Insert VIB bottleneck at blocks.15.hook_resid_post
- Train: CE(4-way letter) + beta * KL(N(mu,sigma) || N(0,I))
- Inference: deterministic mu (no noise), residual connection h += delta

Usage:
    python main_vib_bottleneck.py
    python main_vib_bottleneck.py --n_train 500 --n_epochs 5 --beta 1e-4
    python main_vib_bottleneck.py --eval_only  # skip training, load checkpoint
"""

import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt, check_correct
from src.vib_bottleneck import VIBBottleneck


# ── helpers ─────────────────────────────────────────────────────────────


def _get_letter_ids(tokenizer) -> dict[str, int]:
    ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        ids[letter] = tok_ids[0] if len(tok_ids) == 1 else tok_ids[-1]
    return ids


def _letter_ce_loss(logits_last: torch.Tensor, letter_ids: dict, correct: str):
    """4-way CE loss: -log softmax over {A,B,C,D} logits at correct letter."""
    lid = torch.tensor(
        [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
    )
    letter_logits = logits_last[lid].float()  # [4], fp32 for numerical stability
    target = torch.tensor(
        ["A", "B", "C", "D"].index(correct), device=logits_last.device
    )
    return F.cross_entropy(letter_logits.unsqueeze(0), target.unsqueeze(0))


def _predict_letter(logits_last: torch.Tensor, letter_ids: dict) -> str:
    """Greedy prediction among {A,B,C,D}."""
    lid = torch.tensor(
        [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
    )
    letter_logits = logits_last[lid]
    idx = letter_logits.argmax().item()
    return ["A", "B", "C", "D"][idx]


# ── training ─────────────────────────────────────────────────────────────


def train_vib(
    model,
    vib: VIBBottleneck,
    train_samples: list[dict],
    letter_ids: dict[str, int],
    args,
):
    """Train VIB bottleneck on HellaSwag train split."""
    vib.train()
    for param in model.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(vib.parameters(), lr=args.lr)

    hook_point = f"blocks.{args.vib_layer}.hook_resid_post"
    n_batches = max(1, args.n_train // args.batch_size)

    print(
        f"Training VIB at {hook_point}, {args.n_train} samples, "
        f"batch={args.batch_size}, epochs={args.n_epochs}, beta={args.beta}"
    )

    for epoch in range(args.n_epochs):
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kl = 0.0
        n_correct = 0

        indices = np.random.permutation(len(train_samples))[: args.n_train]
        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch + 1}")

        for bi in pbar:
            batch_idx = indices[bi : bi + args.batch_size]
            opt.zero_grad()

            batch_loss = 0.0
            batch_ce = 0.0
            batch_kl = 0.0
            batch_correct = 0

            for ii, idx in enumerate(batch_idx):
                sample = train_samples[idx]
                prompt = format_prompt(
                    sample["question"], sample["context"], dataset="hellaswag"
                )
                correct_letter = sample["answers"][1].upper()
                tokens = model.to_tokens(prompt, prepend_bos=True)
                last_pos = tokens.shape[1] - 1

                # Run with VIB hook (training mode = reparameterization)
                # No torch.no_grad() — VIB params need gradients
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_point, vib.make_hook(is_training=True))],
                )
                logits_last = logits[0, last_pos, :]

                ce = _letter_ce_loss(logits_last, letter_ids, correct_letter)
                kl = vib.last_kl
                loss = ce + args.beta * kl
                loss = loss / len(batch_idx)  # average over batch

                loss.backward()

                batch_loss += loss.item()
                batch_ce += ce.item()
                batch_kl += kl.item()

                pred_letter = _predict_letter(logits_last, letter_ids)
                if pred_letter == correct_letter:
                    batch_correct += 1

            opt.step()

            epoch_loss += batch_loss * len(batch_idx)
            epoch_ce += batch_ce
            epoch_kl += batch_kl
            n_correct += batch_correct

            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "ce": f"{batch_ce / len(batch_idx):.4f}",
                    "kl": f"{batch_kl / len(batch_idx):.6f}",
                    "acc": f"{batch_correct / len(batch_idx):.2f}",
                }
            )

        n_total = len(indices)
        print(
            f"  Epoch {epoch + 1}: loss={epoch_loss / n_total:.4f}, "
            f"ce={epoch_ce / n_total:.4f}, kl={epoch_kl / n_total:.6f}, "
            f"acc={n_correct / n_total:.4f}"
        )


# ── evaluation ───────────────────────────────────────────────────────────


def evaluate_vib(
    model,
    vib: VIBBottleneck,
    test_samples: list[dict],
    letter_ids: dict[str, int],
    args,
) -> dict:
    """Evaluate VIB on test set. Compares with-VIB vs no-VIB accuracy."""
    vib.eval()
    hook_point = f"blocks.{args.vib_layer}.hook_resid_post"

    results = {
        "n_samples": len(test_samples),
        "no_vib": {"correct": 0, "per_sample": []},
        "with_vib": {"correct": 0, "per_sample": []},
    }

    for sample in tqdm(test_samples, desc="Evaluating"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        # Baseline: no VIB
        with torch.no_grad():
            logits_base = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_base_last = logits_base[0, last_pos, :]
        pred_base = _predict_letter(logits_base_last, letter_ids)
        is_correct_base = pred_base == correct_letter
        results["no_vib"]["correct"] += int(is_correct_base)
        results["no_vib"]["per_sample"].append(
            {
                "pred": pred_base,
                "correct": correct_letter,
                "is_correct": is_correct_base,
            }
        )

        # With VIB (deterministic inference)
        with torch.no_grad():
            logits_vib = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, vib.make_hook(is_training=False))],
            )
        logits_vib_last = logits_vib[0, last_pos, :]
        pred_vib = _predict_letter(logits_vib_last, letter_ids)
        is_correct_vib = pred_vib == correct_letter
        results["with_vib"]["correct"] += int(is_correct_vib)
        results["with_vib"]["per_sample"].append(
            {"pred": pred_vib, "correct": correct_letter, "is_correct": is_correct_vib}
        )

    n = len(test_samples)
    acc_base = results["no_vib"]["correct"] / n
    acc_vib = results["with_vib"]["correct"] / n
    delta = acc_vib - acc_base

    # Count flips
    corrected = 0
    broken = 0
    for b, v in zip(results["no_vib"]["per_sample"], results["with_vib"]["per_sample"]):
        if not b["is_correct"] and v["is_correct"]:
            corrected += 1
        elif b["is_correct"] and not v["is_correct"]:
            broken += 1

    results["summary"] = {
        "accuracy_no_vib": float(acc_base),
        "accuracy_with_vib": float(acc_vib),
        "delta": float(delta),
        "corrected": corrected,
        "broken": broken,
    }

    print(
        f"\nEvaluation ({n} samples):\n"
        f"  No VIB:  acc={acc_base:.4f}\n"
        f"  With VIB: acc={acc_vib:.4f}\n"
        f"  Delta:    {delta:+.4f}\n"
        f"  Corrected: {corrected}, Broken: {broken}"
    )

    return results


# ── main ─────────────────────────────────────────────────────────────────


def main(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = load_model(device=device, model_id=args.model)
    model.eval()
    letter_ids = _get_letter_ids(model.tokenizer)
    print(f"Letter token IDs: {letter_ids}")

    # ── Create VIB module ───────────────────────────────────────────
    vib = VIBBottleneck(d_model=model.cfg.d_model, d_bottleneck=args.d_bottleneck).to(
        device
    )
    n_params = sum(p.numel() for p in vib.parameters())
    print(
        f"VIB params: {n_params:,} (d_model={model.cfg.d_model}, d_bottleneck={args.d_bottleneck})"
    )

    checkpoint_path = output_dir / "vib_checkpoint.pt"

    if not args.eval_only:
        # ── Load train split ────────────────────────────────────────
        print(f"Loading HellaSwag train split...")
        ds_train = load_dataset(
            "Rowan/hellaswag", split="train", trust_remote_code=False
        )
        ds_train = ds_train.shuffle(seed=args.seed)
        label_letters = ["A", "B", "C", "D"]
        train_samples = []
        for item in ds_train.select(range(min(args.n_train_pool, len(ds_train)))):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            correct_ending = endings[label]
            label_letter = label_letters[label]
            choices_text = "\n".join(
                f"{label_letters[i]}. {endings[i]}" for i in range(4)
            )
            train_samples.append(
                {
                    "question": ctx,
                    "answers": [correct_ending, label_letter],
                    "context": choices_text,
                }
            )
        print(f"Train pool: {len(train_samples)} samples")

        # ── Train ───────────────────────────────────────────────────
        train_vib(model, vib, train_samples, letter_ids, args)

        # Save checkpoint
        torch.save(
            {
                "vib_state_dict": vib.state_dict(),
                "args": vars(args),
                "d_model": model.cfg.d_model,
                "d_bottleneck": args.d_bottleneck,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    else:
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        vib.load_state_dict(ckpt["vib_state_dict"])
        print(f"Loaded VIB (trained with {ckpt.get('args', {})})")

    # ── Load test set (knowledge-filtered) ──────────────────────────
    print("Loading HellaSwag validation for evaluation...")
    val_samples = load_hellaswag(n_samples=args.n_test, seed=args.seed + 1)

    # Apply knowledge filter: keep samples where model knows the answer
    print("Computing P(correct) for knowledge filtering...")
    test_samples = []
    for sample in tqdm(val_samples, desc="Filtering"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, tokens.shape[1] - 1, :]
        lid = torch.tensor(
            [letter_ids[l] for l in ["A", "B", "C", "D"]], device=logits_last.device
        )
        letter_logits = logits_last[lid]
        probs = torch.softmax(letter_logits, dim=-1)
        idx = ["A", "B", "C", "D"].index(sample["answers"][1].upper())
        p_correct = float(probs[idx].item())

        if p_correct > args.knowledge_threshold:
            test_samples.append(sample)

    print(
        f"Knowledge filter (P(correct) > {args.knowledge_threshold}): "
        f"{len(test_samples)}/{len(val_samples)} samples"
    )

    # ── Evaluate ────────────────────────────────────────────────────
    eval_results = evaluate_vib(model, vib, test_samples, letter_ids, args)

    # Save results
    eval_results["args"] = vars(args)
    results_file = output_dir / "vib_results.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved to {results_file}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_train_pool", type=int, default=2000)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--d_bottleneck", type=int, default=512)
    parser.add_argument("--vib_layer", type=int, default=15)
    # Evaluation
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--knowledge_threshold", type=float, default=0.3)
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
