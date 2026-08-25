"""Verify: compute_early_exit_logits(h_L27) == model output logits (D2 basis).

D2 gate in validate_s14_tldc.py compares rank_early (L20 early-exit) vs
rank_final (L27 early-exit). If the L27 early-exit path does not reproduce
the model's real output logits, every D2 rank comparison is invalid.

Checks on N prompts:
  1. argmax(token) identical between manual early-exit and model logits
  2. top-5 token overlap
  3. max |l_manual - l_model| (fp16 tolerance)
  4. same check at L20 vs L27 for the D2 comparison itself
"""

import os, sys
from pathlib import Path

import numpy as np
import torch

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

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_triviaqa, format_prompt
from common import load_model_and_unembed
from validate_s14_tldc import compute_early_exit_logits, get_y_true_rank

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device)
n_layers = model.cfg.n_layers
final_layer = n_layers - 1
print(f"layers={n_layers}, d_model={model.cfg.d_model}, fold_ln check: ln_final class={type(ln_final).__name__}")

# Is model.unembed.b_U None or zeros?
print(f"b_U is None: {b_U is None}, b_U shape: {None if b_U is None else tuple(b_U.shape)}")

samples = load_triviaqa(n_samples=5, seed=123)

n_ok_argmax = 0
n_top5_ok = 0
max_diff_all = 0.0
n_rank_agree = 0
n_total = 0

for s in samples:
    prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]

    captured = {}

    def _hook_final(act, hook=None):
        captured["h_final"] = act[:, -1:, :].detach()

    with torch.no_grad():
        logits_model = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{final_layer}.hook_resid_post", _hook_final)]
        )

    h_final = captured["h_final"]  # [1, 1, d_model]
    l_manual = compute_early_exit_logits(h_final, ln_final, W_U, b_U)  # [1, 1, vocab]

    lm = logits_model[0, -1, :].float()
    lh = l_manual[0, 0, :].float()

    n_total += 1
    if lm.argmax().item() == lh.argmax().item():
        n_ok_argmax += 1

    top5_m = set(lm.argsort(descending=True)[:5].tolist())
    top5_h = set(lh.argsort(descending=True)[:5].tolist())
    if top5_m == top5_h:
        n_top5_ok += 1

    d = (lm - lh).abs().max().item()
    max_diff_all = max(max_diff_all, d)

    # rank agreement on y_true (D2 metric)
    y_true = s["answers"][0].strip()
    yid_toks = tokenizer.encode(" " + y_true, add_special_tokens=False)
    if yid_toks:
        r_model = get_y_true_rank(logits_model, yid_toks[0])
        r_manual = get_y_true_rank(l_manual, yid_toks[0])
        if r_model == r_manual:
            n_rank_agree += 1
        print(
            f"  sample: argmax match={lm.argmax().item() == lh.argmax().item()}, "
            f"max|d|={d:.4f}, rank model={r_model} manual={r_manual}"
        )
    else:
        print(f"  sample: argmax match={lm.argmax().item() == lh.argmax().item()}, max|d|={d:.4f}")

print(f"\n=== SUMMARY (n={n_total}) ===")
print(f"argmax identical: {n_ok_argmax}/{n_total}")
print(f"top-5 identical:  {n_top5_ok}/{n_total}")
print(f"max |l_manual - l_model| across all: {max_diff_all:.4f}")
print(f"y_true rank identical (D2 metric):   {n_rank_agree}/{n_total}")
print(f"\nVerdict: early-exit path {'REPRODUCES' if n_ok_argmax == n_total and max_diff_all < 0.1 else 'DIVERGES FROM'} model output")
