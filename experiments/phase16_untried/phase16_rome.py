"""Phase 16.3 (REVIEW): ROME-style Rank-1 FFN Weight Edit — protocol-clean rerun.

ROME (Meng et al. 2022) edits factual knowledge via rank-1 weight update.
We adapt: push FFN output weights to amplify the "truth direction" signal.

Key difference: modifies WEIGHTS (permanent change to computation),
not transient activation perturbations that can be compensated.

Mechanism:
  W_out += λ · outer(k, c) / ||k||^2
  where c = truth direction (output, d_model=2048)
        k = mean MLP intermediate activation (d_mlp=6144, from mlp.hook_post)

Effect: for any input x, output shifts along c proportionally to k·x.

REVIEW fixes (8-point audit, docs/protocol/evaluation-protocol.md §3):
  ① prompt 截断保尾部: tokens[:, -1024:] (was tokens[:, :1024] 保头切尾)
  ② exact 标签: check_correct_exact 全线使用; phase9 records 里存的 fuzzy
     `label` 不再采信 — calibration 标签一律用模型生成 + exact 重算
  ③/⑥ held-out 选参: c/k 只在 calibration 上拟合 (--load records 或
     load_triviaqa(seed_cal)), (层, λ) 只在 VAL 上选 (--n_val/--seed_val,
     与 cal∪test 问题去重), test 只报告
  ④ rank 1-indexed: get_y_true_rank (top-50 = know), 口径与
     train_lora_delta.classify_sample / validate_s14_tldc 一致 —
     rank 取模型**最终层** logits; ℓ* 的 early-exit rank 另存为
     lens 参考字段, 不参与任何决策
  ⑤ 干预路径最终 logits 一律取模型真实前向输出; ℓ* early-exit logits
     (lens) 仅参考 (13.5%→0.18% cublas 舍入伪影教训)
  ⑦ 所有 hook 捕获显式 .detach(); 权重编辑走 weight.data (autograd 外);
     λ=0 必须严格退化 baseline (显式断言)
  ⑧ 无 max(auroc,1-auroc) 路径 (本脚本无 AUROC)

测试集 = load_triviaqa(n_test, seed_test), 与 TLDC 线
(validate_s14_tldc.py --n_test 300 --seed_test 123/456) 同一批样本。

Usage (1.7B 本地, 双 seed):
  python phase16_rome.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract_compact.json \
    --n_test 300 --seed_test 123 --n_val 100 --seed_val 789 --layer_early 20
  python phase16_rome.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract_compact.json \
    --n_test 300 --seed_test 456 --n_val 100 --seed_val 789 --layer_early 20

Known limitations (documented, not silent):
  - --load records 只用作 calibration 样本池 (question/context/gt_answers);
    存的 h/a/m 全部忽略 (当年用保头截断 prompt + fuzzy 标签提取, 两项都违反
    协议), 本脚本在协议条件下全部重算。
  - calibration records 与 test 批次重叠的问题按 question 字符串剔除,
    保证 c/k 严格拟合在 test 集合之外。
  - 输出: experiments/outputs/phase16_rome_review/
      phase16_rome_seed<seed>.json        汇总 (config/test/results)
      phase16_rome_seed<seed>_samples.json per-sample 档案
      phase16_rome_summary.json           跨 seed 汇总 (存在多个 seed 时)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
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
from src.data_loader import load_triviaqa, format_prompt, check_correct_exact


# ═══════════════════════════════════════════════════════════════
# Protocol helpers
# 本地复刻自 experiments/lin_theory/{validate_s14_tldc.py, common.py} —
# 不 import lin_theory 模块, 避免与并行改动中的 lin_theory/ 产生耦合。
# ═══════════════════════════════════════════════════════════════


def clopper_pearson(k, n, alpha=0.05):
    """Exact 95% CI for a binomial proportion (Clopper-Pearson), pure Python.

    No scipy dependency. Returns (lo, hi).
    (Copied verbatim from experiments/lin_theory/validate_s14_tldc.py)
    """
    from math import comb

    if n == 0:
        return (0.0, 0.0)
    if k == 0:
        return (0.0, 1 - (alpha / 2) ** (1.0 / n))
    if k == n:
        return ((alpha / 2) ** (1.0 / n), 1.0)

    def _cdf_le_k(p):
        # P(X <= k) for X ~ Binomial(n, p)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    def _sf_ge_k(p):
        # P(X >= k) = 1 - P(X <= k-1); increasing in p
        return 1.0 - sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k))

    # lower bound: solve P(X >= k) = alpha/2 (increasing fn)
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        if _sf_ge_k(m) < alpha / 2:
            a = m
        else:
            b = m
    lo = (a + b) / 2
    # upper bound: solve P(X <= k) = alpha/2 (decreasing fn)
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        if _cdf_le_k(m) > alpha / 2:
            a = m
        else:
            b = m
    hi = (a + b) / 2
    return (lo, hi)


def get_y_true_rank(logits, y_true_id):
    """Rank 1 = highest probability (1-indexed, top-50 = ranks 1..50).

    code-review-2026-08-24 Medium 5: was 0-indexed, so `rank <= 50` included
    the top-51. Now consistent with train_lora_delta.classify_sample.
    Also flattens to 1-D so the nonzero()[0] index is the rank position
    (fixes a historical bug where 2-D input made every rank collapse to 1).
    (Copied verbatim from experiments/lin_theory/validate_s14_tldc.py)
    """
    # 2026-09-21 修复：early-exit logits 形状为 [1, vocab]（dim==2），旧判定 dim()>1
    # 会走 [0,-1,:] → IndexError（与 main_subspace_intervention.py 同源 bug）
    row = logits[0, -1, :].float() if logits.dim() > 2 else logits.float()
    row = row.reshape(-1)
    sorted_ids = row.argsort(descending=True)
    rank = (sorted_ids == y_true_id).nonzero(as_tuple=True)[0].item() + 1
    return rank


def compute_early_exit_logits(h, ln_final, W_U, b_U):
    """Compute logits from hidden state at any layer via early exit.

    Applies the same RMSNorm + W_U mapping as the final layer.
    ⚠ LENS/early-exit path — REFERENCE ONLY (protocol ⑤): the final logits
    used for any decision/generation must come from the model's real
    forward pass, never from this recomputation (cublas rounding artifact
    lesson: GPU 13.5% step-level argmax disagreement vs CPU 0%).
    Stays in float16 to avoid OOM from W_U.float() (~1.2 GB).
    (Copied verbatim from experiments/lin_theory/validate_s14_tldc.py)
    """
    dtype = next(ln_final.parameters()).dtype
    device = h.device

    h_f16 = h.to(dtype=dtype)  # [..., d_model] float16
    h_norm = ln_final(h_f16)  # [..., d_model] float16
    # Stay in float16 to avoid allocating ~1.2 GB for W_U.float()
    logits = h_norm @ W_U.to(dtype)  # [..., vocab_size] float16
    if b_U is not None:
        logits = logits + b_U.to(dtype)
    return logits


def get_first_answer_token_id(tokenizer, answers):
    """Return the first token ID of the first non-empty answer alias.

    IMPORTANT: The model generates answer tokens with a leading space
    (e.g., " Paris" not "Paris") because the prompt ends with "Answer:".
    We prepend a space to match the actual generated token distribution.
    We preserve the original case since tokenization is case-sensitive.
    (Copied verbatim from experiments/lin_theory/common.py)
    """
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        # Try with leading space (matches generation context after "Answer:")
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


def greedy_generate(model, tokenizer, prompt, device, max_new=20):
    """Greedy generation from the model's REAL logits (protocol ⑤).

    Prompt window: 1024 tokens, keep TAIL (Question) — code review Critical 1.
    (Adapted from experiments/lin_theory/common.py, same loop semantics)
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]  # keep TAIL (Question) — Critical 1

    with torch.no_grad():
        logits = model(tokens)
    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]
    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)
    return tokenizer.decode(gids).strip()


# ═══════════════════════════════════════════════════════════════
# ROME core
# ═══════════════════════════════════════════════════════════════


def fit_rome_direction_and_k(
    model, tokenizer, cal_samples, layer, device, max_k_samples=50
):
    """Fit c (truth direction) and k (mean MLP intermediate) on CALIBRATION.

    c = mean(h|exact-correct) - mean(h|exact-wrong) at ℓ* resid_post,
    L2-normalized. k = mean of blocks.{layer}.mlp.hook_post at last token
    position over the first max_k_samples calibration samples.

    Protocol: labels are recomputed here with greedy generation +
    check_correct_exact (②); stored phase9 fuzzy `label` is never read.
    Prompt keeps TAIL under the 1024 window (①); hook captures are
    explicitly detached (⑦).
    """
    h_correct, h_wrong, all_k = [], [], []

    for s in tqdm(cal_samples, desc=f"  Fit c/k L{layer}"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, -1024:]  # keep TAIL (Question) — Critical 1

        captured = {}

        def _h(act, hook=None):
            captured["h"] = act[0, -1, :].detach()  # protocol ⑦
            return act

        def _k(act, hook=None):
            # [d_mlp] 1-D: torch.outer(k, c) requires 1-D inputs
            captured["k"] = act[0, -1, :].detach()  # protocol ⑦
            return act

        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (f"blocks.{layer}.hook_resid_post", _h),
                    (f"blocks.{layer}.mlp.hook_post", _k),
                ],
            )

        # Exact label from greedy generation on the real model (② ⑤)
        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(19):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)
        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct_exact(ans, s["answers"])  # exact (②)

        h_vec = captured["h"].float().cpu().numpy().flatten()
        if is_correct:
            h_correct.append(h_vec)
        else:
            h_wrong.append(h_vec)
        if len(all_k) < max_k_samples:
            all_k.append(captured["k"])

    if not h_correct or not h_wrong:
        raise RuntimeError(
            "calibration: need >=1 exact-correct and >=1 exact-wrong sample, "
            f"got {len(h_correct)}/{len(h_wrong)}"
        )

    c = np.mean(h_correct, axis=0) - np.mean(h_wrong, axis=0)
    c = c / (np.linalg.norm(c) + 1e-8)
    k = torch.stack(all_k, dim=0).mean(dim=0)  # [d_mlp]
    stats = {
        "n_correct": len(h_correct),
        "n_incorrect": len(h_wrong),
        "n_k": len(all_k),
        "c_norm": float(np.linalg.norm(c)),
        "k_norm": float(torch.norm(k).item()),
    }
    return torch.from_numpy(c).float().to(device), k.float().to(device), stats


@contextmanager
def temporary_W_out_edit(model, layer: int, delta_W: torch.Tensor):
    """Temporarily edit blocks.{layer}.mlp.W_out, restoring after exit.

    Edits weight.data (outside autograd) — protocol ⑦: no gradient flows
    through the edit; generation afterwards reads the model's REAL logits.
    """
    weight = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    original = weight.data.clone()
    with torch.no_grad():
        weight.data = weight.data + delta_W.to(weight.device).to(weight.dtype)
    try:
        yield
    finally:
        weight.data = original


def classify_entries(
    model, tokenizer, samples, device, layer_early, rank_threshold,
    W_U, b_U, ln_final,
):
    """Classify samples into know_correct / know_wrong / dont_know.

    Protocol ②④: rank is 1-indexed on the model's FINAL-layer logits
    (same as validate_s14_tldc.py main(); consistent with
    train_lora_delta.classify_sample and theory §1.2.1); rank <= 50 => know;
    then split KC/KW by check_correct_exact on greedy generation.
    The ℓ* early-exit rank is stored as a lens REFERENCE field only (⑤).
    """
    entries = []
    for i, s in enumerate(tqdm(samples, desc="  Classify")):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            continue

        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, -1024:]  # keep TAIL (Question) — Critical 1

        captured = {}

        def _hook(act, hook=None):
            captured["h"] = act[:, -1:, :].detach()  # protocol ⑦
            return act

        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens, fwd_hooks=[(f"blocks.{layer_early}.hook_resid_post", _hook)]
            )

        # FINAL-layer logits (real model output) — protocol ④⑤
        rank = get_y_true_rank(logits, y_true_id)
        # ℓ* early-exit rank — lens REFERENCE only, never used for decisions
        l_early = compute_early_exit_logits(captured["h"], ln_final, W_U, b_U)
        rank_lstar = get_y_true_rank(l_early, y_true_id)

        gen_text = greedy_generate(model, tokenizer, prompt, device)
        is_correct = check_correct_exact(gen_text, s["answers"])  # exact (②)

        if rank <= rank_threshold:
            subset = "know_correct" if is_correct else "know_wrong"
        else:
            subset = "dont_know"

        entries.append(
            {
                "sample_id": i,
                "rank": rank,
                "rank_lstar_early_exit": rank_lstar,  # lens reference (⑤)
                "is_correct": is_correct,
                "subset": subset,
                "prompt": prompt,
                "answers": s["answers"],
                "question": s["question"][:80],
                "y_true_id": y_true_id,
            }
        )
    return entries


def subset_stats(entries):
    """Baseline per-subset counts/rates for a classified entry list."""
    kw = [e for e in entries if e["subset"] == "know_wrong"]
    kc = [e for e in entries if e["subset"] == "know_correct"]
    dk = [e for e in entries if e["subset"] == "dont_know"]
    n = max(1, len(entries))
    return {
        "n_total": len(entries),
        "n_know_correct": len(kc),
        "n_know_wrong": len(kw),
        "n_dont_know": len(dk),
        "baseline_rate": float(sum(1 for e in entries if e["is_correct"]) / n),
        "kc_baseline_rate": float(sum(1 for e in kc if e["is_correct"]) / max(1, len(kc))),
        "kw_baseline_rate": float(sum(1 for e in kw if e["is_correct"]) / max(1, len(kw))),
        "dk_baseline_rate": float(sum(1 for e in dk if e["is_correct"]) / max(1, len(dk))),
    }


def evaluate_config(model, tokenizer, entries, device, layer, lam, delta_W):
    """Generate all entries under one temporary W_out edit (real model
    forward, ⑤); per-subset exact-matching counts (②)."""
    from collections import defaultdict

    correct_by_subset = defaultdict(int)
    count_by_subset = defaultdict(int)
    flags = {}

    with temporary_W_out_edit(model, layer, delta_W):
        for e in tqdm(entries, desc=f"    L{layer} λ={lam:+.2f}", leave=False):
            gen_text = greedy_generate(model, tokenizer, e["prompt"], device)
            ok = check_correct_exact(gen_text, e["answers"])  # exact (②)
            flags[e["sample_id"]] = bool(ok)
            correct_by_subset[e["subset"]] += int(ok)
            count_by_subset[e["subset"]] += 1
            correct_by_subset["all"] += int(ok)
            count_by_subset["all"] += 1
    return flags, correct_by_subset, count_by_subset


def build_config_result(correct_by_subset, count_by_subset, baseline_rates):
    """Per-subset {correct,total,rate,delta,ci95} vs baseline (template-format)."""
    res = {}
    for s in ["know_wrong", "know_correct", "dont_know", "all"]:
        c, n = correct_by_subset[s], count_by_subset[s]
        if n > 0:
            rate = c / n
            delta = rate - baseline_rates[s]
            ci_lo, ci_hi = clopper_pearson(c, n)  # CI on post-intervention rate
        else:
            rate, delta, ci_lo, ci_hi = 0.0, 0.0, 0.0, 0.0
        res[s] = {
            "correct": c,
            "total": n,
            "rate": float(rate),
            "delta": float(delta),
            "ci95": [float(ci_lo), float(ci_hi)],
        }
    return res


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 16 ROME (REVIEW): Rank-1 FFN Weight Edit"
    )
    parser.add_argument(
        "--load",
        default=None,
        help="Optional phase9 extract JSON (records: question/context/gt_answers). "
        "Used ONLY as the calibration sample pool; stored h/a/m/label are "
        "ignored (head-truncated prompts + fuzzy labels) and recomputed here.",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="Model repo id or local snapshot path (8B: --model Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--n_calibrate", type=int, default=200,
        help="Calibration samples when --load is not given",
    )
    parser.add_argument("--seed_cal", type=int, default=42)
    parser.add_argument(
        "--n_val", type=int, default=100,
        help="Validation samples (same TriviaQA pool, deduped vs cal/test)",
    )
    parser.add_argument(
        "--seed_val", type=int, default=789,
        help="Val draw seed (must differ from seed_test; dedup vs cal/test)",
    )
    parser.add_argument(
        "--n_test", type=int, default=300,
        help="Test samples = load_triviaqa(n_test, seed_test) — same batch as "
        "the TLDC line (protocol n>=300)",
    )
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument(
        "--layer_early", type=int, default=20,
        help="ROME edit layer ℓ* (1.7B: 20). Layers to sweep: --layers",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Edit-layer sweep (selected on VAL); default = [--layer_early]",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="*",
        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        help="λ sweep (selected on VAL); 0.0 must strictly degenerate to baseline",
    )
    parser.add_argument(
        "--rank_threshold", type=int, default=50,
        help="Knowability rank threshold (top-50)",
    )
    parser.add_argument(
        "--select_metric", choices=["all", "kw"], default="all",
        help="Val selection metric: overall rate (original ROME objective) or KW Δ",
    )
    parser.add_argument(
        "--n_k", type=int, default=50,
        help="Number of calibration samples for the mean MLP intermediate k",
    )
    parser.add_argument(
        "--report_full_test_sweep", action="store_true",
        help="Also evaluate the FULL (layer,λ) sweep on test (transparency; "
        "selection is still done on val only — test never selects)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    layers_eff = args.layers if args.layers else [args.layer_early]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (_sys_parent / "outputs" / "phase16_rome_review")
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 64)
    print("Phase 16.3 REVIEW: ROME rank-1 FFN weight edit (protocol-clean)")
    print(f"  Model: {args.model}")
    print(f"  Edit layers: {layers_eff} (ℓ* default L{args.layer_early})")
    print(f"  λ sweep: {args.lambdas} (selected on val, metric={args.select_metric})")
    print(f"  cal: {'--load records' if args.load else f'load_triviaqa(n={args.n_calibrate}, seed={args.seed_cal})'}")
    print(f"  val: n={args.n_val} seed={args.seed_val} | test: n={args.n_test} seed={args.seed_test}")
    print("=" * 64)

    # ── Load model ────────────────────────────────────────────
    print("\n[1/6] Loading model...")
    t0 = time.time()
    model = load_model(device=device, model_id=args.model)
    tokenizer = model.tokenizer
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    ln_final = model.ln_final
    n_layers = model.cfg.n_layers
    final_layer = n_layers - 1
    print(f"  Model: {n_layers} layers, d_model={model.cfg.d_model}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Split: test (TLDC batch) / cal / val, pairwise question-disjoint ──
    print("\n[2/6] Building cal/val/test (question-disjoint)...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]
    test_qs = {s["question"] for s in test_samples}

    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        raw_records = data["records"]
        cal_pool = [
            {
                "question": r["question"],
                "context": r.get("context", ""),
                "answers": r.get("gt_answers") or ([r["gt_answer"]] if r.get("gt_answer") else []),
            }
            for r in raw_records
        ]
        cal_source = f"load:{args.load}"
    else:
        cal_pool = load_triviaqa(n_samples=args.n_calibrate, seed=args.seed_cal)
        cal_source = f"load_triviaqa(n={args.n_calibrate}, seed={args.seed_cal})"

    # Exclude calibration samples that overlap the test batch (strictly no
    # test leakage into c/k); expected overlap is small (~5/300) because
    # phase9 records come from load_triviaqa(seed=42) — same source, different
    # shuffle (documented, phase9_extract.py:233).
    cal = [s for s in cal_pool if s["question"] not in test_qs]
    n_cal_excluded = len(cal_pool) - len(cal)

    cal_qs = {s["question"] for s in cal}
    forbidden = test_qs | cal_qs
    need = args.n_val + len(forbidden) + 50  # buffer guarantees n_val after dedup
    val_pool = load_triviaqa(n_samples=need, seed=args.seed_val)
    val_samples = [s for s in val_pool if s["question"] not in forbidden][: args.n_val]
    assert len(val_samples) == args.n_val, "val dedup pool exhausted"
    print(
        f"  cal: {len(cal)} samples ({cal_source}; "
        f"{n_cal_excluded} excluded for test overlap)"
    )
    print(f"  val: {len(val_samples)} (seed={args.seed_val}, deduped vs cal/test)")
    print(f"  test: {len(test_samples)} (seed={args.seed_test}) — TLDC batch")

    # ── Fit c/k on calibration ────────────────────────────────
    print("\n[3/6] Fitting ROME c/k on calibration (exact labels)...")
    c_by_layer, k_by_layer, ksq_by_layer, fit_stats = {}, {}, {}, {}
    for layer in layers_eff:
        c, k, stats = fit_rome_direction_and_k(
            model, tokenizer, cal, layer, device, max_k_samples=args.n_k
        )
        c_by_layer[layer] = c
        k_by_layer[layer] = k
        ksq_by_layer[layer] = torch.dot(k, k) + 1e-12
        fit_stats[str(layer)] = stats
        print(
            f"  L{layer}: ||c||={stats['c_norm']:.4f}, ||k||={stats['k_norm']:.1f} "
            f"(exact correct/wrong={stats['n_correct']}/{stats['n_incorrect']}, "
            f"n_k={stats['n_k']})"
        )

    # ── Classify val & test (KW/KC/DK, exact labels, rank on final logits) ──
    print("\n[4/6] Classifying val/test by knowability...")
    val_entries = classify_entries(
        model, tokenizer, val_samples, device, args.layer_early,
        args.rank_threshold, W_U, b_U, ln_final,
    )
    test_entries = classify_entries(
        model, tokenizer, test_samples, device, args.layer_early,
        args.rank_threshold, W_U, b_U, ln_final,
    )
    val_stats = subset_stats(val_entries)
    test_stats = subset_stats(test_entries)
    val_baseline_rates = {
        "know_wrong": val_stats["kw_baseline_rate"],
        "know_correct": val_stats["kc_baseline_rate"],
        "dont_know": val_stats["dk_baseline_rate"],
        "all": val_stats["baseline_rate"],
    }
    test_baseline_rates = {
        "know_wrong": test_stats["kw_baseline_rate"],
        "know_correct": test_stats["kc_baseline_rate"],
        "dont_know": test_stats["dk_baseline_rate"],
        "all": test_stats["baseline_rate"],
    }

    def _print_split(tag, st):
        print(
            f"  {tag}: KC={st['n_know_correct']} KW={st['n_know_wrong']} "
            f"DK={st['n_dont_know']} / {st['n_total']} | "
            f"All={st['baseline_rate']:.1%} KW={st['kw_baseline_rate']:.1%} "
            f"KC={st['kc_baseline_rate']:.1%} DK={st['dk_baseline_rate']:.1%}"
        )

    _print_split("val ", val_stats)
    _print_split("test", test_stats)

    # ── Val sweep: select (layer, λ) — test NOT touched (⑥) ──
    print(
        f"\n[5/6] Val sweep ({len(layers_eff)} layers × {len(args.lambdas)} λ) "
        f"— selection metric = {args.select_metric}..."
    )
    val_sweep = {}
    best_key, best_score = None, -float("inf")
    for layer in layers_eff:
        for lam in args.lambdas:
            with torch.no_grad():  # protocol ⑦: no grad through the edit
                delta_W = lam * torch.outer(
                    k_by_layer[layer].detach(), c_by_layer[layer].detach()
                ) / ksq_by_layer[layer].detach()
            flags, cb, ct = evaluate_config(
                model, tokenizer, val_entries, device, layer, lam, delta_W
            )
            res = build_config_result(cb, ct, val_baseline_rates)
            key = f"L{layer}_lam{lam:+.2f}"
            val_sweep[key] = {"layer": layer, "lambda": lam, **res}

            if abs(lam) < 1e-9:
                # protocol ⑦ verification: λ=0 must strictly degenerate baseline
                assert res["all"]["rate"] == val_baseline_rates["all"], (
                    f"λ=0 does NOT strictly degenerate to val baseline "
                    f"({res['all']['rate']} != {val_baseline_rates['all']})"
                )
                print(f"    λ=0 degeneration check: ✅ strictly == val baseline")

            score = (
                res["know_wrong"]["delta"]
                if args.select_metric == "kw"
                else res["all"]["rate"]
            )
            if score > best_score:
                best_score = score
                best_key = key
            print(
                f"    {key}: all={res['all']['rate']:.1%} "
                f"KWΔ={res['know_wrong']['delta']:+.1%} "
                f"KCΔ={res['know_correct']['delta']:+.1%} DKΔ={res['dont_know']['delta']:+.1%}"
            )

    assert best_key is not None, "val sweep produced no configs (empty --lambdas?)"
    best_layer = val_sweep[best_key]["layer"]
    best_lambda = val_sweep[best_key]["lambda"]
    print(f"  → selected on val: {best_key}")

    # ── Test: report only (selected config; optional full sweep) ──
    print("\n[6/6] Test (report only — no selection on test)...")
    results_test = {}
    per_sample_flags = {e["sample_id"]: {} for e in test_entries}

    configs_on_test = [("selected", best_layer, best_lambda)]
    if args.report_full_test_sweep:
        configs_on_test += [
            ("sweep", layer, lam) for layer in layers_eff for lam in args.lambdas
        ]

    for tag, layer, lam in configs_on_test:
        with torch.no_grad():  # protocol ⑦
            delta_W = lam * torch.outer(
                k_by_layer[layer].detach(), c_by_layer[layer].detach()
            ) / ksq_by_layer[layer].detach()
        flags, cb, ct = evaluate_config(
            model, tokenizer, test_entries, device, layer, lam, delta_W
        )
        res = build_config_result(cb, ct, test_baseline_rates)
        key = f"L{layer}_lam{lam:+.2f}"
        if tag == "selected":
            results_test["selected"] = {"layer": layer, "lambda": lam, **res}
        else:
            results_test.setdefault("test_sweep", {})[key] = {
                "layer": layer, "lambda": lam, **res
            }
        for sid, ok in flags.items():
            per_sample_flags[sid][key] = ok

    sel = results_test["selected"]
    print("\n  ── Test summary (selected config, Δ vs baseline) ──")
    print(f"  {'subset':>12}  {'rate':>8}  {'Δ':>8}  {'95% CI (rate)':>18}")
    print(f"  {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 18}")
    for s, label in [
        ("know_wrong", "KW (target)"),
        ("know_correct", "KC"),
        ("dont_know", "DK"),
        ("all", "All"),
    ]:
        r = sel[s]
        print(
            f"  {label:>12}  {r['rate']:>8.1%}  {r['delta']:>+8.1%}  "
            f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]"
        )

    # Statistical note (mirrors validate_s14_tldc.py 2026-08-25 note):
    # KW baseline is a DEFINED value (KW = rank<=thr AND greedy-wrong), not a
    # sampled proportion. Under H0 "intervention has zero effect", P(any KW
    # rescue)=0 → any KW correct count > 0 rejects "no effect"; the effect
    # size is given by the CI above, not by a Fisher-style proportion test.
    print("\n  ── Statistical note ──")
    print(
        f"  KW baseline {test_stats['n_know_wrong']} samples have rate 0 by "
        f"construction (KW = greedy-wrong). Under H0 'zero effect', P(any "
        f"rescue)=0 → any KW correct count > 0 rejects 'no effect'; effect "
        f"size = the CI above. Double seed (123/456) is required for a verdict."
    )

    # ── Save (template-aligned: config / test / results + per_sample) ──
    notes = [
        "rank: 1-indexed on FINAL-layer model logits (consistent with "
        "train_lora_delta.classify_sample / validate_s14_tldc.py); "
        "rank_lstar_early_exit in per-sample is a lens REFERENCE only.",
        "final logits always from the real model forward; ℓ* early-exit "
        "logits (lens) never replace them (cublas rounding lesson).",
        "--load records used only as calibration sample pool; stored h/a/m "
        "and fuzzy labels ignored, everything recomputed under protocol.",
        "calibration samples overlapping the test batch are excluded by "
        "question string (strict test-leakage-free c/k).",
        "λ=0 strictly degenerates to baseline (asserted on val).",
    ]
    output = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "layers_eff": layers_eff,
            "final_layer": final_layer,
            "cal_source": cal_source,
            "n_cal_used": len(cal),
            "n_cal_excluded_test_overlap": n_cal_excluded,
            "notes": notes,
        },
        "calibration_fit": fit_stats,
        "val": {
            **val_stats,
            "selection": {
                "metric": args.select_metric,
                "key": best_key,
                "layer": best_layer,
                "lambda": best_lambda,
            },
            "sweep": val_sweep,
        },
        "test": test_stats,
        "results": results_test,
        "per_sample_path": f"phase16_rome_seed{args.seed_test}_samples.json",
    }

    out_path = output_dir / f"phase16_rome_seed{args.seed_test}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Per-sample archive (template-aligned: config + samples)
    per_sample = {
        "config": {
            "seed_test": args.seed_test,
            "rank_threshold": args.rank_threshold,
            "layer_early": args.layer_early,
            "selected_key": best_key,
        },
        "notes": notes,
        "samples": {
            str(e["sample_id"]): {
                "subset": e["subset"],
                "rank": e["rank"],
                "rank_lstar_early_exit": e["rank_lstar_early_exit"],
                "question": e["question"],
                "baseline_correct": bool(e["is_correct"]),
                **per_sample_flags[e["sample_id"]],
            }
            for e in test_entries
        },
    }
    samples_path = output_dir / f"phase16_rome_seed{args.seed_test}_samples.json"
    with open(samples_path, "w") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    print(f"Saved per-sample to {samples_path}")

    # Cross-seed rollup of headline numbers (only seeds present in output_dir)
    seed_files = sorted(
        p for p in output_dir.glob("phase16_rome_seed*.json")
        if "_samples" not in p.name
    )
    rollup = {"seeds": {}}
    for p in seed_files:
        with open(p) as f:
            d = json.load(f)
        seed = d["config"]["seed_test"]
        rollup["seeds"][str(seed)] = {
            "test": d["test"],
            "selected": d["results"].get("selected"),
        }
    if rollup["seeds"]:
        rollup_path = output_dir / "phase16_rome_summary.json"
        with open(rollup_path, "w") as f:
            json.dump(rollup, f, indent=2, ensure_ascii=False)
        print(f"Cross-seed rollup saved to {rollup_path}")


if __name__ == "__main__":
    main()
