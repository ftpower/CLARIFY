"""Quick verification: TLDC β=0.1 with multiple seeds + different early layers."""

import json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Path setup
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_triviaqa, format_prompt, check_correct
from common import (
    load_model_and_unembed,
    get_first_answer_token_id,
    extract_h_at_layer,
    greedy_generate,
)
from validate_s14_tldc import (
    compute_early_exit_logits,
    tldc_greedy_generate,
    get_y_true_rank,
)

device = "cuda"
print("Loading model...")
model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)

results = {}

for seed in [456, 789]:
    print(f"\n{'=' * 60}")
    print(f"Seed={seed}")

    test_samples = load_triviaqa(n_samples=50, seed=seed)
    test_samples = test_samples[:50]

    # Classify
    test_entries = []
    for i, s in enumerate(tqdm(test_samples, desc="  Classify")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            continue
        _, logits, _, _ = extract_h_at_layer(model, tokenizer, prompt, device, 20)
        rank = get_y_true_rank(logits, y_true_id)
        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, s["answers"], dataset="triviaqa")
        if rank <= 50:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"
        test_entries.append(
            {
                "prompt": prompt,
                "answers": s["answers"],
                "subset": subset,
                "is_correct": is_correct,
            }
        )

    kw = [e for e in test_entries if e["subset"] == "know_wrong"]
    kc = [e for e in test_entries if e["subset"] == "know_correct"]
    dk = [e for e in test_entries if e["subset"] == "dont_know"]
    bl = sum(1 for e in test_entries if e["is_correct"]) / len(test_entries)
    kw_bl = sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))

    print(
        f"  KC={len(kc)}, KW={len(kw)}, DK={len(dk)}, Baseline={bl:.1%}, KW_bl={kw_bl:.1%}"
    )

    # Test β=0.1
    for beta in [0.05, 0.08, 0.10]:
        correct = defaultdict(int)
        count = defaultdict(int)
        for e in tqdm(test_entries, desc=f"  β={beta:.2f}"):
            gen_text, _ = tldc_greedy_generate(
                model,
                tokenizer,
                e["prompt"],
                device,
                20,
                W_U,
                b_U,
                ln_final,
                beta,
            )
            is_c = check_correct(gen_text, e["answers"], dataset="triviaqa")
            if is_c:
                correct[e["subset"]] += 1
                correct["all"] += 1
            count[e["subset"]] += 1
            count["all"] += 1

        kw_delta = correct["know_wrong"] / max(1, count["know_wrong"]) - kw_bl
        all_delta = correct["all"] / max(1, count["all"]) - bl
        dk_delta = correct["dont_know"] / max(1, count["dont_know"]) - (
            sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))
        )
        print(
            f"  β={beta:.2f}: KW={correct['know_wrong']}/{count['know_wrong']} Δ={kw_delta:+.1%}  "
            f"DK={correct['dont_know']}/{count['dont_know']} Δ={dk_delta:+.1%}  "
            f"All={correct['all']}/{count['all']} Δ={all_delta:+.1%}"
        )

        results[f"seed={seed}_β={beta:.2f}"] = {
            "kw_delta": kw_delta,
            "all_delta": all_delta,
            "dk_delta": dk_delta,
            "kw_correct": correct["know_wrong"],
            "kw_total": count["know_wrong"],
        }

print("\n" + "=" * 60)
print("SUMMARY")
for k, v in results.items():
    print(
        f"  {k}: KW Δ={v['kw_delta']:+.1%} DK Δ={v['dk_delta']:+.1%} All Δ={v['all_delta']:+.1%}"
    )

print("\nDone!")
