"""Phase 16 前置诊断：A. 过滤"模型已知"的问题 + C. Token-level 评估。

诊断目标：
  A: 多少问题是模型"知道答案但生成了错误内容"？
     → 通过正确答案 token 的 rank/prob 判断模型是否具备该知识
  C: Token-level 指标能否检测到干预效应？
     → 连续指标（rank Δ, logprob Δ）比离散生成正确率更敏感

用法：
  python diagnose_known_questions.py \
    --load ../phase9_multi_state/outputs_phase9/phase9_extract.json \
    --n_test 50

输出：
  - 已知/未知分类统计
  - 不同 rank 阈值下的可干预子集大小
  - Truth direction 干预在 token-level 的效果
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

# ── Path setup ──────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import format_prompt


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════


def get_answer_first_tokens(
    generated_text: str, tokenizer, max_tokens: int = 5
) -> list[int]:
    """从生成文本中提取前几个 token ID（去除特殊 token 和空白）。"""
    # Tokenize with special tokens to get correct eos/bos
    ids = tokenizer.encode(generated_text.strip(), add_special_tokens=False)
    # Filter to meaningful tokens (skip very short/whitespace-only tokens)
    result = []
    for tid in ids:
        tok = tokenizer.decode([tid]).strip()
        if tok and len(tok) > 0:
            result.append(tid)
            if len(result) >= max_tokens:
                break
    return result if result else ids[:max_tokens]


def compute_token_rank(logits: torch.Tensor, target_id: int) -> int:
    """计算 target_id 在 logits 中的 rank（1 = 最高概率）。"""
    # Higher rank = worse; rank 1 = top prediction
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
    return rank


def compute_token_logprob(logits: torch.Tensor, target_id: int) -> float:
    """计算 target_id 的 log probability。"""
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[target_id].item()


def get_layer_truth_direction(records, layer: int, state_key: str = "h") -> np.ndarray:
    """从 train 集计算 truth direction v = mean(correct) - mean(wrong)。"""
    correct_vecs = []
    wrong_vecs = []
    for r in records:
        vec = np.array(r[state_key][str(layer)], dtype=np.float32)
        if r["label"] == 1:
            correct_vecs.append(vec)
        else:
            wrong_vecs.append(vec)
    if not correct_vecs or not wrong_vecs:
        raise ValueError(f"Empty vectors for layer {layer}, key {state_key}")

    mu_c = np.mean(correct_vecs, axis=0)
    mu_w = np.mean(wrong_vecs, axis=0)
    v = mu_c - mu_w
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


# ═══════════════════════════════════════════════════════════════════
# 主诊断逻辑
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Diagnose known vs unknown questions")
    parser.add_argument("--load", required=True, help="phase9 extraction JSON")
    parser.add_argument("--n_test", type=int, default=50, help="Number of test samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=20, help="Intervention layer")
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[-1.0, -0.5, 0.5, 1.0],
        help="Alpha values to test",
    )
    parser.add_argument(
        "--rank_thresholds",
        nargs="*",
        type=int,
        default=[1, 3, 5, 10, 50, 100, 500],
        help="Rank thresholds for 'known' classification",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. 加载模型 ──────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    model = load_model(device=device, model_id="Qwen/Qwen3-1.7B")
    tokenizer = model.tokenizer
    print(
        f"  Model: {model.cfg.model_name}, layers={model.cfg.n_layers}, d={model.cfg.d_model}"
    )

    # ── 2. 加载数据 ──────────────────────────────────────────────
    print("\n[2/5] Loading data...")
    with open(args.load) as f:
        data = json.load(f)
    all_records = data["records"]
    total_correct = sum(1 for r in all_records if r["label"] == 1)
    print(f"  Total records: {len(all_records)}")
    print(f"  Correct: {total_correct} ({total_correct / len(all_records):.1%})")

    # Split train/test (match Phase 11 convention: last n_test as test)
    n_test = min(args.n_test, len(all_records) // 2)
    n_train = len(all_records) - n_test
    train_records = all_records[:n_train]
    test_records = all_records[n_train:]
    print(f"  Train: {n_train}, Test: {n_test}")
    print(
        f"  Test correct: {sum(1 for r in test_records if r['label'] == 1)} / {n_test}"
    )

    # ── 3. Token-level 分析（无干预）──────────────────────────────
    print("\n[3/5] Analyzing token-level knowledge (no intervention)...")

    results = []  # list of dicts with per-question stats

    for i, rec in enumerate(tqdm(test_records, desc="Token analysis")):
        question = rec["question"]
        context = rec.get("context", "")
        gt_answers = rec["gt_answers"]
        label = rec["label"]

        # Format prompt and run forward pass
        prompt = format_prompt(question, context, dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        input_len = tokens.shape[1]

        with torch.no_grad():
            logits = model(tokens)  # [1, seq_len, vocab]
        last_logits = logits[0, -1, :]  # [vocab]

        # For each GT answer, find best rank among first few tokens
        best_rank = float("inf")
        best_logprob = float("-inf")
        best_gt_token = None

        for ans in gt_answers:
            ans_tokens = get_answer_first_tokens(ans, tokenizer, max_tokens=3)
            for tid in ans_tokens:
                rank = compute_token_rank(last_logits, tid)
                lp = compute_token_logprob(last_logits, tid)
                if rank < best_rank:
                    best_rank = rank
                    best_logprob = lp
                    best_gt_token = tokenizer.decode([tid])

        # Also check what the model actually generated (from record)
        generated_tokens = get_answer_first_tokens(
            rec["generated"], tokenizer, max_tokens=3
        )
        generated_first_id = generated_tokens[0] if generated_tokens else None

        results.append(
            {
                "question": question[:80],
                "label": label,
                "best_rank": best_rank,
                "best_logprob": best_logprob,
                "best_gt_token": best_gt_token,
                "generated_first_id": generated_first_id,
                "generated_first_token": tokenizer.decode([generated_first_id])
                if generated_first_id
                else "?",
                "gt_answers": gt_answers,
            }
        )

    # ── 4. 统计分析 ──────────────────────────────────────────────
    print("\n[4/5] Statistical analysis...")

    # Correct vs wrong by rank
    correct_ranks = [r["best_rank"] for r in results if r["label"] == 1]
    wrong_ranks = [r["best_rank"] for r in results if r["label"] == 0]

    print(f"\n  Correct answers (n={len(correct_ranks)}):")
    if correct_ranks:
        print(f"    Median rank: {np.median(correct_ranks):.0f}")
        print(f"    Mean rank: {np.mean(correct_ranks):.0f}")
        print(
            f"    Rank 1: {sum(1 for r in correct_ranks if r == 1)} / {len(correct_ranks)}"
        )
        print(
            f"    Rank <= 5: {sum(1 for r in correct_ranks if r <= 5)} / {len(correct_ranks)}"
        )
        print(
            f"    Rank <= 50: {sum(1 for r in correct_ranks if r <= 50)} / {len(correct_ranks)}"
        )

    print(f"\n  Wrong answers (n={len(wrong_ranks)}):")
    if wrong_ranks:
        print(f"    Median rank: {np.median(wrong_ranks):.0f}")
        print(f"    Mean rank: {np.mean(wrong_ranks):.0f}")
        print(
            f"    Rank 1: {sum(1 for r in wrong_ranks if r == 1)} / {len(wrong_ranks)}"
        )
        print(
            f"    Rank <= 5: {sum(1 for r in wrong_ranks if r <= 5)} / {len(wrong_ranks)}"
        )
        print(
            f"    Rank <= 50: {sum(1 for r in wrong_ranks if r <= 50)} / {len(wrong_ranks)}"
        )

    # "Known" classification at various thresholds
    print(f"\n  'Known' classification by rank threshold:")
    print(
        f"  {'Threshold':<12} {'Known':<8} {'Known+Correct':<15} {'Known+Wrong':<15} {'Max Δ':<10}"
    )
    print(f"  {'-' * 60}")

    n_test_correct = sum(1 for r in results if r["label"] == 1)
    for thresh in args.rank_thresholds:
        known = [r for r in results if r["best_rank"] <= thresh]
        known_correct = [r for r in known if r["label"] == 1]
        known_wrong = [r for r in known if r["label"] == 0]
        max_delta = len(known_wrong)  # best case: all known+wrong → correct
        print(
            f"  rank<={thresh:<6} {len(known):<8} {len(known_correct):<15} {len(known_wrong):<15} {max_delta:<10}"
        )

    # ── 5. Token-level 干预效应 ──────────────────────────────────
    print("\n[5/5] Token-level intervention effect...")

    # Compute truth direction from train set (all 28 layers)
    print("  Computing truth directions...")
    v_h_dict = {}
    for lyr in range(model.cfg.n_layers):
        v_h_dict[lyr] = get_layer_truth_direction(train_records, lyr, "h")

    # Test: for each alpha, compute Δ logprob of correct token
    test_indices = list(range(len(test_records)))
    intervention_results = {}

    for alpha in args.alphas:
        print(f"  Testing α={alpha:+.1f}...")
        deltas = []  # Δ logprob for each question

        for idx in tqdm(test_indices, desc=f"  α={alpha:+.1f}", leave=False):
            rec = test_records[idx]
            question = rec["question"]
            context = rec.get("context", "")
            gt_answers = rec["gt_answers"]

            prompt = format_prompt(question, context, dataset="triviaqa")
            tokens = model.to_tokens(prompt, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            input_len = tokens.shape[1]

            # ── Baseline (no intervention) ──
            with torch.no_grad():
                logits_base = model(tokens)
            last_base = logits_base[0, -1, :]

            # Best logprob among GT tokens
            best_lp_base = float("-inf")
            best_tid = None
            for ans in gt_answers:
                ans_tokens = get_answer_first_tokens(ans, tokenizer, max_tokens=3)
                for tid in ans_tokens:
                    lp = compute_token_logprob(last_base, tid)
                    if lp > best_lp_base:
                        best_lp_base = lp
                        best_tid = tid

            # ── Intervention ──
            hook_name = f"blocks.{args.layer}.hook_resid_post"
            mod_vec = torch.tensor(
                alpha * v_h_dict[args.layer], dtype=torch.float32, device=device
            )

            def _intervene(act, hook=None):
                act[0, input_len - 1, :] = act[0, input_len - 1, :] + mod_vec
                return act

            with torch.no_grad():
                logits_int = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, _intervene)]
                )
            last_int = logits_int[0, -1, :]

            best_lp_int = float("-inf")
            for ans in gt_answers:
                ans_tokens = get_answer_first_tokens(ans, tokenizer, max_tokens=3)
                for tid in ans_tokens:
                    lp = compute_token_logprob(last_int, tid)
                    if lp > best_lp_int:
                        best_lp_int = lp

            delta = best_lp_int - best_lp_base
            deltas.append(delta)

        mean_delta = np.mean(deltas) if deltas else 0.0
        std_delta = np.std(deltas) if deltas else 0.0

        # Also compute separately for "known-wrong" subset
        known_wrong_deltas = [
            d
            for d, r in zip(deltas, results)
            if r["best_rank"] <= 50 and r["label"] == 0
        ]
        known_wrong_mean = np.mean(known_wrong_deltas) if known_wrong_deltas else 0.0

        intervention_results[alpha] = {
            "mean_delta_logprob": mean_delta,
            "std_delta_logprob": std_delta,
            "frac_positive": sum(1 for d in deltas if d > 0) / len(deltas)
            if deltas
            else 0,
            "n_known_wrong": len(known_wrong_deltas),
            "mean_delta_known_wrong": known_wrong_mean,
        }

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  A. Knowledge coverage:")
    n_known = sum(1 for r in results if r["best_rank"] <= 50)
    n_known_wrong = sum(1 for r in results if r["best_rank"] <= 50 and r["label"] == 0)
    print(
        f"     'Known' (rank<=50): {n_known}/{len(results)} ({n_known / len(results):.0%})"
    )
    print(
        f"     'Known but wrong':  {n_known_wrong}/{len(results)} ({n_known_wrong / len(results):.0%})"
    )
    if n_known_wrong == 0:
        print(
            f"     ⚠  ZERO 'known but wrong' questions → intervention has no room to work!"
        )
        print(f"     → Model doesn't 'forget' correct answers — it never knew them.")
        print(f"     → 1.7B is simply too small for TriviaQA open-domain QA.")

    print(f"\n  C. Token-level intervention effect (layer={args.layer}):")
    print(f"     {'α':<8} {'Mean Δlp':<12} {'Frac>0':<10} {'KnownWrong Δlp':<16}")
    print(f"     {'-' * 50}")
    for alpha in sorted(intervention_results.keys()):
        r = intervention_results[alpha]
        print(
            f"     {alpha:<+8.1f} {r['mean_delta_logprob']:<+12.6f} {r['frac_positive']:<10.2f} "
            f"{r['mean_delta_known_wrong']:<+16.6f}  (n={r['n_known_wrong']})"
        )

    # Overall verdict
    print(f"\n  VERDICT:")
    all_near_zero = all(
        abs(r["mean_delta_logprob"]) < 0.01 for r in intervention_results.values()
    )
    if all_near_zero and n_known_wrong > 0:
        print(f"  → Token-level intervention ALSO shows zero effect.")
        print(
            f"  → Even when model 'knows' the answer, truth direction doesn't boost it."
        )
        print(f"  → Confirms: v is a readout direction, not a control direction.")
        print(f"  → Phase 16 (ITI/ROME/RepE) is the right next step.")
    elif n_known_wrong == 0:
        print(f"  → Baseline knowledge too low to measure intervention effect.")
        print(f"  → Recommendations:")
        print(f"    1. Switch to 8B model (higher baseline accuracy)")
        print(f"    2. Or switch to multiple-choice task (e.g., TruthfulQA MC)")
        print(f"    3. Or use continuous metrics on all samples (logprob shift)")
    else:
        print(f"  → Token-level effect detected! Investigate further.")

    # ── Save ────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "outputs_phase16"
    output_dir.mkdir(exist_ok=True)

    output = {
        "config": {
            "n_test": n_test,
            "n_train": n_train,
            "layer": args.layer,
            "alphas": args.alphas,
            "rank_thresholds": args.rank_thresholds,
        },
        "summary": {
            "n_test": len(results),
            "n_test_correct": n_test_correct,
            "n_known_rank50": n_known,
            "n_known_wrong_rank50": n_known_wrong,
        },
        "per_question": [
            {
                "question": r["question"],
                "label": r["label"],
                "best_rank": int(r["best_rank"]),
                "best_logprob": float(r["best_logprob"]),
                "best_gt_token": r["best_gt_token"],
                "generated_first_token": r["generated_first_token"],
            }
            for r in results
        ],
        "intervention_results": {
            str(k): {
                kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()
            }
            for k, v in intervention_results.items()
        },
    }

    output_path = output_dir / "diagnose_known_questions.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
