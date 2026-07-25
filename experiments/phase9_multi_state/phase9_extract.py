"""Phase 9 Extraction: Multi-state data collection for detection + intervention.

One pass extracts everything needed for 9.1 (detection) and 9.2 (intervention):

For each sample:
  1. Normal prompt → extract h/a/m at ALL layers (last input token) + greedy generate → label
  2. Answer-augmented prompt → extract h*/a*/m* at selected layers (last token)

Output:
  - Per-sample h/a/m at all layers + labels → 9.1 detection analysis
  - Per-sample m* at L20 → 9.2 intervention (answer-aware target state)

Usage:
    python phase9_extract.py --n_samples 200 --intervention_layer 20
    python phase9_extract.py --n_samples 100 --intervention_layer 20 --no_answer_aug
"""

import argparse
import json
import os
import sys
import time
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
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_normal_and_answer_augmented(
    model, tokenizer, samples, device, n_layers,
    intervention_layer, max_new_tokens=20,
    do_answer_aug=True,
):
    """Extract h/a/m from normal prompts + h*/a*/m* from answer-augmented prompts.

    Returns:
        records: list of per-sample dicts
        summary: dict with overall stats
    """
    records = []
    correct_count = 0

    t0 = time.time()
    for s in tqdm(samples, desc="Extract phase9"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")

        # ═══════════════════════════════════════════════════════════════════
        # Pass 1: Normal prompt → h/a/m at ALL layers + generation + label
        # ═══════════════════════════════════════════════════════════════════
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        residual = {}
        fwd_hooks = []

        # Hook resid_post at ALL layers
        for li in range(n_layers):
            def _resid_hook(act, hook=None, _layer=li):
                residual[("resid", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_resid_post", _resid_hook))

        # Hook attn_out and mlp_out at ALL layers
        for li in range(n_layers):
            def _attn_hook(act, hook=None, _layer=li):
                residual[("attn", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_attn_out", _attn_hook))

            def _mlp_hook(act, hook=None, _layer=li):
                residual[("mlp", _layer)] = act[:, -1, :].detach()
                return act
            fwd_hooks.append((f"blocks.{li}.hook_mlp_out", _mlp_hook))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # Collect per-layer states
        h_all = {}
        a_all = {}
        m_all = {}
        for li in range(n_layers):
            h_all[str(li)] = residual[("resid", li)].float().cpu().numpy().flatten().tolist()
            a_all[str(li)] = residual[("attn", li)].float().cpu().numpy().flatten().tolist()
            m_all[str(li)] = residual[("mlp", li)].float().cpu().numpy().flatten().tolist()

        # Generate for correctness label
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

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")
        if is_correct:
            correct_count += 1
        label = 1 if is_correct else 0

        # ═══════════════════════════════════════════════════════════════════
        # Pass 2: Answer-augmented prompt → h*/a*/m* at intervention_layer
        # ═══════════════════════════════════════════════════════════════════
        # Use all ground-truth answers (for correctness checking later)
        gt_answers = s["answers"] if s["answers"] else [""]
        gt_answer = gt_answers[0]

        # ═══════════════════════════════════════════════════════════════════
        # Pass 2: Answer-augmented prompt → h*/a*/m* at intervention_layer
        # ═══════════════════════════════════════════════════════════════════
        h_star, a_star, m_star = None, None, None
        if do_answer_aug:
            aug_prompt = prompt + " " + gt_answer

            aug_tokens = model.to_tokens(aug_prompt, prepend_bos=True)
            if aug_tokens.shape[1] > 1024:
                aug_tokens = aug_tokens[:, :1024]

            aug_residual = {}

            def _aug_resid_hook(act, hook=None):
                aug_residual["resid"] = act[:, -1, :].detach()
                return act

            def _aug_attn_hook(act, hook=None):
                aug_residual["attn"] = act[:, -1, :].detach()
                return act

            def _aug_mlp_hook(act, hook=None):
                aug_residual["mlp"] = act[:, -1, :].detach()
                return act

            aug_hooks = [
                (f"blocks.{intervention_layer}.hook_resid_post", _aug_resid_hook),
                (f"blocks.{intervention_layer}.hook_attn_out", _aug_attn_hook),
                (f"blocks.{intervention_layer}.hook_mlp_out", _aug_mlp_hook),
            ]

            with torch.no_grad():
                model.run_with_hooks(aug_tokens, fwd_hooks=aug_hooks)

            h_star = aug_residual["resid"].float().cpu().numpy().flatten().tolist()
            a_star = aug_residual["attn"].float().cpu().numpy().flatten().tolist()
            m_star = aug_residual["mlp"].float().cpu().numpy().flatten().tolist()

        # ═══════════════════════════════════════════════════════════════════
        # Store
        # ═══════════════════════════════════════════════════════════════════
        records.append({
            "question": s["question"],
            "context": s.get("context", ""),
            "gt_answers": gt_answers,   # ALL acceptable answers
            "gt_answer": gt_answer,     # First answer (for answer-aug)
            "generated": ans[:200],
            "label": label,
            "input_len": input_len,
            "h": h_all,          # {layer_str: [d] list}
            "a": a_all,
            "m": m_all,
            "h_star": h_star,    # [d] list at intervention_layer
            "a_star": a_star,
            "m_star": m_star,
        })

    elapsed = time.time() - t0
    summary = {
        "n_samples": len(records),
        "n_correct": correct_count,
        "correct_rate": correct_count / len(records) if records else 0,
        "extraction_time_s": elapsed,
    }
    return records, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Multi-state extraction")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--intervention_layer", type=int, default=20,
                        help="Layer for answer-augmented extraction (9.2 intervention)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_answer_aug", action="store_true",
                        help="Skip answer-augmented extraction (9.1 only)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 9: Multi-State Extraction")
    print(f"  Samples: {args.n_samples}  Intervention layer: {args.intervention_layer}")
    print(f"  Answer-aug: {not args.no_answer_aug}")
    print(f"{'='*60}\n")

    # Load model & data
    print("Loading model & data...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
    n_layers = model.cfg.n_layers
    print(f"  Model: {args.model}, {n_layers} layers, d={model.cfg.d_model}")
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Extract
    print(f"\nExtracting h/a/m (all layers) + labels...")
    if not args.no_answer_aug:
        print(f"Also extracting h*/a*/m* at L{args.intervention_layer} (answer-augmented)...")

    records, summary = extract_normal_and_answer_augmented(
        model, tokenizer, samples, device, n_layers,
        args.intervention_layer,
        do_answer_aug=not args.no_answer_aug,
    )

    print(f"\n  Extraction complete: {summary['extraction_time_s']:.0f}s")
    print(f"  Correct: {summary['n_correct']}/{summary['n_samples']} "
          f"({summary['correct_rate']:.1%})")

    # Save
    save_path = output_dir / "phase9_extract.json"
    with open(save_path, "w") as f:
        json.dump({
            "summary": summary,
            "config": {
                "n_samples": args.n_samples,
                "intervention_layer": args.intervention_layer,
                "model": args.model,
                "n_layers": n_layers,
                "d_model": model.cfg.d_model,
                "seed": args.seed,
                "answer_aug": not args.no_answer_aug,
            },
            "records": records,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    # Also save a compact version without full per-layer states (for quick inspection)
    compact_path = output_dir / "phase9_extract_compact.json"
    compact_records = []
    for r in records:
        compact_records.append({
            "question": r["question"],
            "context": r.get("context", ""),
            "gt_answers": r.get("gt_answers", [r.get("gt_answer", "")]),
            "gt_answer": r["gt_answer"],
            "generated": r["generated"],
            "label": r["label"],
            # Only keep L20 states in compact version
            "h_L20": r["h"].get(str(args.intervention_layer), None),
            "a_L20": r["a"].get(str(args.intervention_layer), None),
            "m_L20": r["m"].get(str(args.intervention_layer), None),
            "h_star": r["h_star"],
            "a_star": r["a_star"],
            "m_star": r["m_star"],
        })
    with open(compact_path, "w") as f:
        json.dump({
            "summary": summary,
            "config": {
                "n_samples": args.n_samples,
                "intervention_layer": args.intervention_layer,
            },
            "records": compact_records,
        }, f, indent=2)
    print(f"  Saved: {compact_path}")

    print(f"\n{'='*60}")
    print(f"Phase 9 extraction complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
