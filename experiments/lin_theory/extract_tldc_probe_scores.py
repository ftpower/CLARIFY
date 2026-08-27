"""Extract per-sample probe features (h at question-end position) for gated-TLDC simulation.

Theory: docs/theory-gated-tldc.md §5.1 (stage 0).

Mirrors validate_s14_tldc.py sample loading exactly (same seeds, same skip
condition, same prompt construction) so the output can be joined 1:1 with the
per-sample TLDC archives (s14_tldc_samples.json) by (sample_id, question).

The hidden state is extracted at the SAME position and layer as the detection
probe (detect_lr_probe_cv.py): blocks.L28.hook_resid_post at the last token of
the 1024-window prompt, i.e. BEFORE generation — the gate can therefore be
applied pre-decode.

Usage (8B on server):
    unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
      experiments/lin_theory/extract_tldc_probe_scores.py \
      --model Qwen/Qwen3-8B \
      --seed_test 123
    # then --seed_test 456
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

from src.data_loader import load_triviaqa, format_prompt, check_correct_exact
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    extract_h_at_layer,
    greedy_generate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--n_test", type=int, default=300)
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--layer", type=int, default=28,
                        help="Detection peak layer (8B: L28)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).parent.parent / "outputs" / "lin_theory_8b")
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    model_tag = args.model.split("/")[-1]

    print("=" * 64)
    print(f"Gated-TLDC stage 0: extract probe features (h_L{args.layer}, question-end)")
    print(f"  model={args.model} n_test={args.n_test} seed_test={args.seed_test}")
    print("=" * 64)

    print("[1/3] Loading model...")
    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"[2/3] Extracting h_L{args.layer} + baseline labels (seed={args.seed_test})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]

    entries = []
    skipped = 0
    for i, sample in enumerate(tqdm(test_samples, desc="  Extract")):
        prompt = format_prompt(sample["question"], sample["context"], dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, sample["answers"])
        if y_true_id is None:
            skipped += 1
            continue
        h, _logits, _tokens, _pos = extract_h_at_layer(
            model, tokenizer, prompt, device, args.layer
        )
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = int(check_correct_exact(gen_text, sample["answers"]))
        entries.append({
            "sample_id": i,
            "question": sample["question"][:80],
            "is_correct": is_correct,
            "h": h[0].float().cpu().numpy().tolist(),
        })

    print(f"  extracted {len(entries)} / {len(test_samples)} (skipped {skipped})")

    print("[3/3] Saving...")
    out_path = output_dir / f"probe_scores_seed{args.seed_test}_{model_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "layer": args.layer,
            "seed_test": args.seed_test,
            "n_test": args.n_test,
            "n_entries": len(entries),
            "skipped": skipped,
            "entries": entries,
        }, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
