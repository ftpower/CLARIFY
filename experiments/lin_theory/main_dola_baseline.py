"""DoLa 同协议基线复现（旧干预范式复核 · 3 个必跑之一）。

为什么需要它（发表前提）
------------------------
`docs/plans/current.md` §一 P0-新：**没有同协议基线 ⇒ "同类方法显著优势"不可度量**。
本脚本在同一评测协议（同一 TriviaQA 样本、同一 prompt/截断、同一 exact 标签、同一 KW/KC/DK
划分、同一 per-sample 档案格式）下复现 DoLa 三档：
  baseline（vanilla greedy） / dola-static（固定 premature 层） / dola-dynamic（逐 token JSD 选层）

协议对齐（与 `experiments/lin_theory/validate_s14_tldc.py` 完全一致，保证分母可比）
--------------------------------------------------------------------
  · 模型：HookedTransformer（`common.load_model_and_unembed`）
  · 样本：`load_triviaqa(n_samples=n_test, seed=seed_test)`（1.7B: seed123/456，8B 同）
  · prompt：`format_prompt(..., dataset="triviaqa")`；>1024 token 时 **保尾部**（`tokens[:, -1024:]`）
  · 标签：`check_correct_exact`（**exact，不用 fuzzy** — 8 点清单第 ② 项）
  · 知识划分：**最终层真实 logits** 上 y_true 的 **1-indexed rank**，`rank<=50` 为 know，
    再按 exact 对错分 KC（know_correct）/ KW（know_wrong）/ DK（dont_know）。
    ⚠️ 口径核实（2026-09-21）：`common.extract_h_at_layer` 的 hook 只捕获 h 并 pass-through，
    返回的 logits 是模型**最终层输出** → `validate_s14_tldc.py` 的 rank 就是最终层 rank（`--layer_early`
    只决定捕获哪一层的 h）。本脚本同口径，并把 ℓ* early-exit rank 另存为诊断字段 `rank_early`。
  · 统计：KW/KC/DK/All 的 Δ + Clopper-Pearson 95% CI（**无任何 max(auroc,1-auroc) 符号翻转**）

数值纪律（2026-08-25 教训：lens 重算 logits 有 cublas 舍入伪影，实测 13.5% 步级 argmax 不一致）
--------------------------------------------------------------------
  · **成熟层（mature）项一律取模型真实输出 logits**（forward 的返回值），不用 lens 重算；
  · 只有 premature 层（中间层）必须经 `ln_final + W_U` 投影 —— 这是 DoLa 方法本身的要求，不是验证路径，
    脚本内以 `--verify_lens` 打印"真实 logits vs lens 重算成熟层"的 argmax 一致率作为自检。

I19 卡（DoLa, ICLR 2024）给出的 6 条实现对齐要点
------------------------------------------------
  ① APC（adaptive plausibility constraint）α=0.1，并有 −1000 硬截断变体 → `--alpha/--apc_variant`
  ② 候选 premature 层＝**偶数层**、按 bucket 划分、在验证集上选桶 → `--candidate_stride/--bucket`
  ③ TruthfulQA-MC 无 post-softmax（本项目为开放生成，不适用；记录偏差）
  ④ rp=1.2（repetition penalty）仅生成侧、且 vanilla 侧同样施加 → 本项目协议**不启用 rp**，
     为保持"同协议"可比性显式记录该偏差（`config.rp` 恒为 1.0）
  ⑤ greedy + 行为指标判效 → 本项目用 exact-match（与 KW/KC 口径一致）
  ⑥ 行为指标判效，不用内部信号代验 → 本脚本只报行为口径

用法
----
    # 本地 1.7B（RTX 5060 8GB）
    python experiments/lin_theory/main_dola_baseline.py \
      --model Qwen/Qwen3-1.7B --layer_early 20 --mode dola-dynamic \
      --n_test 300 --seed_test 123 --alpha 0.1 --save_samples

    # 服务器 8B（硬性命令格式见 CLAUDE.md）
    unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
      experiments/lin_theory/main_dola_baseline.py \
      --model Qwen/Qwen3-8B \
      ...（每参数一行）

输出：`<output_dir>/dola_baseline_<mode>_seed<seed>.json`（config/test/results/per_sample）
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── 复用干净管线（与 validate_s14_tldc.py 同一套标签与分类口径）────────────────
sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    load_model_and_unembed,
    load_triviaqa,
    format_prompt,
    check_correct_exact,
    get_first_answer_token_id,
)

MAX_CTX = 1024  # 与全项目一致：超长时保尾部（Question 在尾部）


# ═════════════════════════════════════════════════════════════════════════════
# 统计工具（与模板一致）
# ═════════════════════════════════════════════════════════════════════════════


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) binomial CI. Returns (lo, hi) or (0.0, 1.0) if n==0."""
    from scipy.stats import beta as _beta

    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def get_y_true_rank(logits, y_true_id):
    """1-indexed rank of y_true token in logits.

    支持三种形状：``[vocab]``、``[1, vocab]``（early-exit 投影）、``[1, seq, vocab]``（模型输出）。
    2026-09-21 修复：模板版对 2-D 输入会 `lg[y_true_id]` 索引 batch 维 → IndexError
    （CPU 冒烟实测；ROME/子空间两脚本同源 bug 已一并修复）。
    """
    if logits.ndim == 3:
        lg = logits[0, -1, :]
    else:
        lg = logits.reshape(-1)
    lg = lg.float()
    return int((lg > lg[y_true_id]).sum().item()) + 1


# ═════════════════════════════════════════════════════════════════════════════
# DoLa 机制
# ═════════════════════════════════════════════════════════════════════════════


def _project(h, ln_final, W_U_T, b_U):
    """中间层 hidden → logits（经 ln_final + unembed）。仅用于 premature 层。"""
    return F.linear(ln_final(h.unsqueeze(0) if h.ndim == 1 else h), W_U_T, b_U)


def _js_divergence(mature_logits, pre_logits):
    """Jensen-Shannon divergence between mature and premature distributions."""
    m_lp = F.log_softmax(mature_logits.float(), dim=-1)
    p_lp = F.log_softmax(pre_logits.float(), dim=-1)
    m_p, p_p = m_lp.exp(), p_lp.exp()
    M = 0.5 * (m_p + p_p)
    kl1 = F.kl_div(m_lp, M, reduction="sum")
    kl2 = F.kl_div(p_lp, M, reduction="sum")
    return float(0.5 * (kl1 + kl2).item())


def _apc_mask(mature_logits, alpha, variant):
    """Adaptive plausibility constraint（I19 要点①）。

    hard : p_mature < alpha * max(p_mature) 的 token 置 −inf
    -1000: 同上但置 −1000（DoLa 官方的软截断变体）
    alpha<=0 或 >=1 时返回 None（不施加约束）。
    """
    if alpha is None or alpha <= 0.0:
        return None
    probs = F.softmax(mature_logits.float(), dim=-1)
    thr = alpha * probs.max()
    mask = probs < thr
    if not bool(mask.any()):
        return None
    fill = float("-inf") if variant == "hard" else -1000.0
    return torch.where(mask, torch.full_like(mature_logits.float(), fill), mature_logits.float())


@torch.no_grad()
def dola_next_token(
    model,
    tokens,
    candidates,
    mode,
    premature_layer,
    alpha,
    apc_variant,
    W_U_T,
    b_U,
    ln_final,
    verify_lens=False,
):
    """One DoLa decoding step → (next_token_id, selected_layer, js_best, lens_agree).

    成熟层项取模型真实 logits；premature 层经投影（DoLa 方法本身要求）。
    """
    storage = {}
    hooks = []
    mature_key = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    keys = [mature_key] + [f"blocks.{L}.hook_resid_post" for L in candidates]

    def _save(key):
        def hook(act, hook=None):
            storage[key] = act[0, -1, :].detach()  # 断梯度（8 点清单第 ⑦ 项）
            return act

        return hook

    for k in keys:
        hooks.append((k, _save(k)))

    real_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)  # [1, seq, vocab]
    mature_logits = real_logits[0, -1, :]  # 真实 logits（不用 lens 代验）

    lens_agree = None
    if verify_lens:
        lens_mature = _project(storage[mature_key], ln_final, W_U_T, b_U)[0]
        lens_agree = bool(int(lens_mature.argmax()) == int(mature_logits.argmax()))

    # ── 选 premature 层 ────────────────────────────────────────────────
    if mode == "dola-static":
        selected = premature_layer
        js_best = None
    else:  # dola-dynamic：逐 token 取 JSD 最大的候选层
        js_best, selected = -1.0, candidates[0]
        for L in candidates:
            pre_logits = _project(storage[f"blocks.{L}.hook_resid_post"], ln_final, W_U_T, b_U)[0]
            js = _js_divergence(mature_logits, pre_logits)
            if js > js_best:
                js_best, selected = js, L

    pre_logits = _project(storage[f"blocks.{selected}.hook_resid_post"], ln_final, W_U_T, b_U)[0]

    # ── DoLa 对比分数（文献式：log_softmax(mature) − log_softmax(premature)）──
    diff = F.log_softmax(mature_logits.float(), dim=-1) - F.log_softmax(pre_logits.float(), dim=-1)

    constrained = _apc_mask(mature_logits, alpha, apc_variant)
    if constrained is not None:
        diff = diff + (constrained - mature_logits.float())  # APC：把不合规 token 压到 −inf/−1000

    next_id = int(diff.argmax().item())
    return next_id, selected, js_best, lens_agree


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    mode="baseline",
    candidates=(),
    premature_layer=None,
    alpha=0.1,
    apc_variant="hard",
    max_new=20,
    W_U_T=None,
    b_U=None,
    ln_final=None,
    verify_lens=False,
):
    """Greedy generation；mode='baseline' 时退化为 vanilla greedy（真实 logits）。"""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > MAX_CTX:
        tokens = tokens[:, -MAX_CTX:]  # 保尾部

    gids, trace, lens_agree_n, lens_agree_k = [], [], 0, 0

    for step in range(max_new):
        if mode == "baseline":
            logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            sel, js = None, None
        else:
            nid, sel, js, agree = dola_next_token(
                model, tokens, list(candidates), mode, premature_layer,
                alpha, apc_variant, W_U_T, b_U, ln_final, verify_lens=verify_lens,
            )
            if agree is not None:
                lens_agree_k += int(agree)
                lens_agree_n += 1
            trace.append({"step": step, "selected_layer": int(sel), "js": js, "token_id": nid})

        gids.append(nid)
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    text = tokenizer.decode(gids).strip()
    return text, trace, (lens_agree_k, lens_agree_n)


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="DoLa 同协议基线（复核用，protocol-aligned）")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                    help="repo id 或服务器快照路径（8B: Qwen/Qwen3-8B）")
    ap.add_argument("--mode", type=str, default="baseline",
                    choices=["baseline", "dola-static", "dola-dynamic"])
    ap.add_argument("--premature_layer", type=int, default=None,
                    help="dola-static 的固定 premature 层（动态模式忽略）")
    ap.add_argument("--candidate_stride", type=int, default=2,
                    help="候选 premature 层步长（I19: 偶数层 → 2）")
    ap.add_argument("--bucket", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="把候选层限制在 [LO,HI]（验证集选桶用；默认不限制）")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="APC 系数（I19: 0.1；<=0 关闭）")
    ap.add_argument("--apc_variant", type=str, default="hard", choices=["hard", "-1000"])
    ap.add_argument("--layer_early", type=int, default=28,
                    help="ℓ* 层：仅用于**诊断字段** rank_early（知识划分用最终层真实 logits，与 TLDC 线同口径）；8B 惯例 28，1.7B 本线 20")
    ap.add_argument("--rank_threshold", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed_test", type=int, default=123)
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--verify_lens", action="store_true",
                    help="打印真实 logits 与 lens 重算成熟层的 argmax 一致率（伪影自检）")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--save_samples", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) if args.output_dir else (Path(__file__).parent.parent / "outputs" / "dola_baseline_review")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 72)
    print("DoLa 同协议基线（复核）")
    print(f"  model={args.model} | mode={args.mode} | ℓ*={args.layer_early} | APC α={args.alpha}({args.apc_variant})")
    print(f"  n_test={args.n_test} seed_test={args.seed_test} rank_thr={args.rank_threshold}")
    print("  ⚠️ 协议偏差记录：本复现**不启用 rp=1.2**（I19 要点④），以与本项目 TLDC 线同协议可比")
    print("=" * 72)

    t0 = time.time()
    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    n_layers = model.cfg.n_layers
    W_U_T = model.unembed.W_U.T.contiguous()
    print(f"  loaded in {time.time() - t0:.1f}s | n_layers={n_layers} d_model={model.cfg.d_model}")

    mature_layer = n_layers - 1
    candidates = [l for l in range(0, n_layers, args.candidate_stride) if l < mature_layer]
    if args.bucket:
        candidates = [l for l in candidates if args.bucket[0] <= l <= args.bucket[1]]
    if not candidates and args.mode != "baseline":
        raise SystemExit("候选 premature 层为空：检查 --candidate_stride/--bucket")
    if args.mode == "dola-static":
        if args.premature_layer is None:
            raise SystemExit("dola-static 需要 --premature_layer（或在验证集上选层后传入）")
        if args.premature_layer >= mature_layer:
            raise SystemExit(f"premature 层必须 < mature(L{mature_layer})")
        # 静态层必须被 hook 覆盖，否则取不到该层 hidden（冒烟测试发现的 bug）
        candidates = sorted(set(candidates) | {args.premature_layer})
    print(f"  mature=L{mature_layer} | candidates={candidates}")

    # ── 分类 + baseline ────────────────────────────────────────────────
    print(f"\n[1/2] 分类 {args.n_test} 个测试样本（seed={args.seed_test}）…")
    samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)[: args.n_test]
    entries = []
    for i, s in enumerate(tqdm(samples, desc="  classify")):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        if y_true_id is None:
            continue
        toks = model.to_tokens(prompt, prepend_bos=True)
        if toks.shape[1] > MAX_CTX:
            toks = toks[:, -MAX_CTX:]
        storage = {}
        hook_pt = f"blocks.{args.layer_early}.hook_resid_post"

        def _cap(act, hook=None, _s=storage):
            _s["h"] = act[0, -1, :].detach()
            return act

        with torch.no_grad():
            logits = model.run_with_hooks(toks, fwd_hooks=[(hook_pt, _cap)])
        rank = get_y_true_rank(logits, y_true_id)  # 最终层真实 logits（与 TLDC 线同口径）
        l_early = F.linear(ln_final(storage["h"].unsqueeze(0)), W_U_T, b_U)
        rank_early = get_y_true_rank(l_early, y_true_id)  # ℓ* early-exit rank（诊断字段）
        base_text, _, _ = generate(model, tokenizer, prompt, device, mode="baseline",
                                   max_new=args.max_new, W_U_T=W_U_T, b_U=b_U, ln_final=ln_final)
        ok = check_correct_exact(base_text, s["answers"])  # exact（8 点清单 ②）
        subset = ("know_correct" if ok else "know_wrong") if rank <= args.rank_threshold else "dont_know"
        entries.append({"sample_id": i, "rank": rank, "rank_early": rank_early,
                        "baseline_correct": bool(ok), "subset": subset,
                        "question": s["question"][:80], "answers": s["answers"],
                        "baseline_text": base_text[:120], "prompt": prompt})

    n_kc = sum(e["subset"] == "know_correct" for e in entries)
    n_kw = sum(e["subset"] == "know_wrong" for e in entries)
    n_dk = sum(e["subset"] == "dont_know" for e in entries)
    base_rate = sum(e["baseline_correct"] for e in entries) / max(1, len(entries))
    print(f"  KC={n_kc} KW={n_kw} DK={n_dk} (total={len(entries)}) | baseline acc={base_rate:.1%}")
    assert len(entries) == n_kc + n_kw + n_dk, "三分类不构成划分（口径事故，立即排查）"

    # ── 干预档 ─────────────────────────────────────────────────────────
    print(f"\n[2/2] 运行条件：{args.mode} …")
    per_sample, lens_k, lens_n = [], 0, 0
    for e in tqdm(entries, desc=f"  {args.mode}"):
        text, trace, (ak, an) = generate(
            model, tokenizer, e["prompt"], device, mode=args.mode,
            candidates=candidates, premature_layer=args.premature_layer,
            alpha=args.alpha, apc_variant=args.apc_variant,
            max_new=args.max_new, W_U_T=W_U_T, b_U=b_U, ln_final=ln_final,
            verify_lens=args.verify_lens,
        )
        lens_k += ak
        lens_n += an
        ok = check_correct_exact(text, e["answers"])
        per_sample.append({
            "sample_id": e["sample_id"], "rank": e["rank"], "rank_early": e["rank_early"],
            "subset": e["subset"],
            "baseline_correct": e["baseline_correct"], "intervened_correct": bool(ok),
            "gen_text": text[:120], "n_steps": len(trace),
            "selected_layers": [t["selected_layer"] for t in trace],
            "js": [t["js"] for t in trace],
        })

    def agg(name):
        grp = [r for r in per_sample if r["subset"] == name]
        n = len(grp)
        k = sum(r["intervened_correct"] for r in grp)
        base_k = sum(r["baseline_correct"] for r in grp)
        rate, base = (k / n if n else 0.0), (base_k / n if n else 0.0)
        return {"total": n, "correct": k, "rate": rate, "baseline_rate": base,
                "delta": rate - base, "ci95": list(clopper_pearson(k, n))}

    results = {g: agg(g) for g in ("know_correct", "know_wrong", "dont_know", "all")
               if g != "all"}
    n_all, k_all = len(per_sample), sum(r["intervened_correct"] for r in per_sample)
    b_all = sum(r["baseline_correct"] for r in per_sample)
    results["all"] = {"total": n_all, "correct": k_all, "rate": k_all / max(1, n_all),
                      "baseline_rate": b_all / max(1, n_all),
                      "delta": (k_all - b_all) / max(1, n_all),
                      "ci95": list(clopper_pearson(k_all, n_all))}
    # 救回/破坏双口径（D21 范式：与 TLDC 线同口径，便于横向比较）
    rescued = sum(1 for r in per_sample if (not r["baseline_correct"]) and r["intervened_correct"])
    broken = sum(1 for r in per_sample if r["baseline_correct"] and (not r["intervened_correct"]))
    kw_rescued = sum(1 for r in per_sample if r["subset"] == "know_wrong" and r["intervened_correct"])
    kc_broken = sum(1 for r in per_sample if r["subset"] == "know_correct" and not r["intervened_correct"])

    print("\n  ── 结果（Δ vs baseline；95% CI 为干预后比率）──")
    for g in ("know_wrong", "know_correct", "dont_know", "all"):
        r = results[g]
        print(f"  {g:>12}: n={r['total']:>3} base={r['baseline_rate']:.3f} → {r['rate']:.3f} "
              f"(Δ={r['delta']:+.3f}, CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}])")
    print(f"  救回={rescued} 破坏={broken} | KW 救回 {kw_rescued}/{n_kw} | KC 破坏 {kc_broken}/{n_kc}")
    # 自洽断言（同"β=0 必须严格退化 baseline"纪律）：baseline 档必须恒等复现自身
    if args.mode == "baseline" and (rescued or broken):
        print(f"  ⚠️ [自检失败] baseline 档出现 救回={rescued}/破坏={broken}，"
              f"说明贪心生成不确定或标签口径有 bug → 结果不可用")
    if lens_n:
        print(f"  [自检] 真实 logits vs lens 成熟层 argmax 一致率 = {lens_k / lens_n:.4f} (n={lens_n})")

    out = {
        "config": {**{k: v for k, v in vars(args).items()}},
        "protocol": {
            "labels": "exact (check_correct_exact)",
            "know_rule": f"rank(final real logits) <= {args.rank_threshold} (1-indexed); "
                         f"L{args.layer_early} early-exit rank stored as diagnostic only",
            "truncation": "keep tail (<=1024)",
            "mature_term": "model real logits (not lens)",
            "rp": 1.0,
            "note": "no repetition penalty to stay protocol-identical with the project TLDC line",
        },
        "test": {"n_total": len(entries), "n_know_correct": n_kc, "n_know_wrong": n_kw,
                 "n_dont_know": n_dk, "baseline_rate": base_rate},
        "results": results,
        "dual_criteria": {"rescued": rescued, "broken": broken,
                          "kw_rescued": kw_rescued, "kc_broken": kc_broken,
                          "kw_total": n_kw, "kc_total": n_kc},
        "lens_selfcheck": {"agree": lens_k, "n": lens_n},
        "per_sample": per_sample,
    }
    tag = f"{args.mode}_seed{args.seed_test}" + (f"_L{args.premature_layer}" if args.mode == "dola-static" else "")
    out_path = output_dir / f"dola_baseline_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
