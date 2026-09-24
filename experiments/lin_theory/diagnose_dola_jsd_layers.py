"""DoLa 图 2 式 JSD 分化预分析（S1 **判停点**）。

为什么先做这一步：I19 卡（DoLa, ICLR 2024, 已评审）明写——GPT2-Medium（335M）**全面失效**，
作者归因"层间知识无分化"；并给出红线：**1.7B 级结果不得外推为"方法无效"**。
所以本脚本用论文图 2 的同一量（末层 vs 偶数早层的 JSD，逐位置）在**我们的模型**上先量分化程度，
再决定是否值得进 S2（MC 评测）。方案：`docs/protocol/dola-native-reproduction-20260924.md` §5。

预注册判停规则（写于跑之前，见方案 §5.1-G；三条件**全过**才算"分化存在"）
--------------------------------------------------------------------------
设 `m(d)` = 深度 d 的 JSD 均值（默认识别面 = **续写位置**，可用 `--verdict_on prompt` 换成题面位置），
`anchor` = JSD(成熟层分布 ‖ 均匀分布) 的均值（"最大可能的层间差异"的标度参照），
`cnt(d)` = 逐位置 argmax JSD 选层的计数：

  (i)   **相对分化**：max_d m(d) / min_d m(d) ≥ **1.5**
  (ii)  **选层非均匀**：对 cnt(d) 做 χ² 均匀性检验，**p < 0.05**
  (iii) **绝对尺度**：max_d m(d) ≥ **0.05 × anchor**

  · 三条全过 ⇒ `分化存在` ⇒ 进 S2（MC 评测）
  · 一条都不过 ⇒ `无分化（判停）` ⇒ **只报诊断、不进 S2**（红线：不得外推"方法无效"）
  · 部分过 ⇒ `灰区` ⇒ 记录后仍可进 S2，但结论必须并列披露分化不足

零 GPU 自检：`python3 experiments/lin_theory/diagnose_dola_jsd_layers.py --selftest`
（bucket 规则、判停逻辑、锚点退化三项，秒级）

用法：
    # S1（本地 1.7B，约 10 分钟）
    python3 experiments/lin_theory/diagnose_dola_jsd_layers.py --model Qwen/Qwen3-1.7B --n_questions 100
    # 服务器 8B（硬性命令格式见 CLAUDE.md）
    unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
      experiments/lin_theory/diagnose_dola_jsd_layers.py --model Qwen/Qwen3-8B ...

输出：`experiments/outputs/dola_mc_repro/jsd_profile_<model>_<tag>.json` + 同名 `.md`（含层-分化曲线表）
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_dola_mc import (  # noqa: E402
    DEFAULT_DATA,
    build_prompt_and_answer,
    candidates_in_bucket,
    default_buckets,
    depth_to_hook_name,
    jsd_mean,
    jsd_true_mean,
    load_questions,
    refs_of,
    validate_questions,
)

# ── 预注册阈值（改动须同步方案 §5.1-G 并注明日期）──────────────────────────────
THR_JSD_RATIO = 1.5
THR_CHI2_P = 0.05
THR_ANCHOR_FRAC = 0.05


def jsd_to_uniform(mature_logits):
    """严格 JSD(成熟层分布 ‖ 均匀分布)，逐位置；作为"层间差异"的绝对标度锚点。

    用严格 JSD（有界 ≤ ln2）而非官方 R 口径：R 在 p 有近零分量时可发散，不能当标度锚点。
    """
    p = F.softmax(mature_logits.float(), dim=-1)
    V = p.shape[-1]
    u = torch.full_like(p, 1.0 / V)
    M = 0.5 * (p + u)
    from main_dola_mc import _kl_proper

    return 0.5 * (_kl_proper(p, M) + _kl_proper(u, M))


def verdict_from_stats(mean_by_depth, argmax_counts, anchor, profile="answer"):
    """预注册判停规则（纯逻辑，可离线测）。"""
    depths = sorted(mean_by_depth)
    vals = [mean_by_depth[d] for d in depths]
    lo, hi = min(vals), max(vals)
    ratio = float(hi / lo) if lo > 0 else float("inf")
    c1 = ratio >= THR_JSD_RATIO
    cnt = np.array([argmax_counts.get(d, 0) for d in depths], dtype=float)
    chi2_p = float("nan")
    c2 = False
    if cnt.sum() > 0 and (cnt > 0).sum() >= 2:
        try:
            from scipy.stats import chisquare

            chi2_p = float(chisquare(cnt).pvalue)
            c2 = chi2_p < THR_CHI2_P
        except Exception:
            pass
    c3 = bool(hi >= THR_ANCHOR_FRAC * anchor) if anchor > 0 else False
    n_pass = int(c1) + int(c2) + int(c3)
    if n_pass == 3:
        verdict = "分化存在"
    elif n_pass == 0:
        verdict = "无分化（判停）"
    else:
        verdict = "灰区（分化不足，进 S2 但须并列披露）"
    return {"profile": profile, "max_mean_jsd": hi, "min_mean_jsd": lo, "ratio": ratio,
            "ratio_threshold": THR_JSD_RATIO, "c1_relative": bool(c1),
            "chi2_p": chi2_p, "chi2_p_threshold": THR_CHI2_P, "c2_nonuniform": bool(c2),
            "anchor_jsd_to_uniform": float(anchor), "anchor_frac_threshold": THR_ANCHOR_FRAC,
            "c3_absolute": bool(c3), "n_pass": n_pass, "verdict": verdict,
            "argmax_counts": {str(d): int(argmax_counts.get(d, 0)) for d in depths}}


@torch.no_grad()
def profile_question(model, prompt, cont, depths, prompt_positions=64, max_ctx=0):
    """单题：一次前向 → 每层在{续写位置, 题面尾段位置}上的 JSD 与成熟层到均匀分布的锚点。"""
    prefix_ids = model.to_tokens(prompt, prepend_bos=True)
    full_ids = model.to_tokens(prompt + cont, prepend_bos=True)
    prefix_len = int(prefix_ids.shape[1])
    if max_ctx and full_ids.shape[1] > max_ctx:
        raise SystemExit(f"序列长度 {full_ids.shape[1]} 超 --max_ctx {max_ctx}")

    seq_len = int(full_ids.shape[1])
    ans_pos = torch.arange(prefix_len - 1, seq_len - 1, device=full_ids.device)
    p_lo = max(0, prefix_len - 1 - prompt_positions)
    prm_pos = torch.arange(p_lo, prefix_len - 1, device=full_ids.device)
    pos_all = torch.cat([ans_pos, prm_pos]).unique()

    store = {}
    fwd_hooks = []
    for d in depths:
        name = depth_to_hook_name(model, d)

        def _cap(act, hook=None, _d=d):
            store[_d] = act[0, pos_all, :].detach().float()
            return act

        fwd_hooks.append((name, _cap))
    real_logits = model.run_with_hooks(full_ids, fwd_hooks=fwd_hooks)
    mature_all = real_logits[0, :, :].float()

    idx_map = {int(p): i for i, p in enumerate(pos_all.tolist())}
    ans_idx = torch.tensor([idx_map[int(p)] for p in ans_pos.tolist()])
    prm_idx = torch.tensor([idx_map[int(p)] for p in prm_pos.tolist()])
    mature_ans = mature_all[ans_pos]
    mature_prm = mature_all[prm_pos]

    W_U_T = model.unembed.W_U.T.contiguous()
    b_U = getattr(model.unembed, "b_U", None)
    out = {"depths": [int(d) for d in depths],
           "jsd_answer": {int(d): [] for d in depths},
           "jsd_prompt": {int(d): [] for d in depths},
           "r_answer": {int(d): [] for d in depths},
           "r_prompt": {int(d): [] for d in depths}}
    for d in depths:
        pre = F.linear(model.ln_final(store[d]), W_U_T, b_U)
        out["jsd_answer"][int(d)] = [float(x) for x in jsd_true_mean(mature_ans, pre[ans_idx]).tolist()]
        out["r_answer"][int(d)] = [float(x) for x in jsd_mean(mature_ans, pre[ans_idx]).tolist()]
        if len(prm_idx):
            out["jsd_prompt"][int(d)] = [float(x) for x in jsd_true_mean(mature_prm, pre[prm_idx]).tolist()]
            out["r_prompt"][int(d)] = [float(x) for x in jsd_mean(mature_prm, pre[prm_idx]).tolist()]
    out["anchor_answer"] = [float(x) for x in jsd_to_uniform(mature_ans).tolist()]
    if len(prm_idx):
        out["anchor_prompt"] = [float(x) for x in jsd_to_uniform(mature_prm).tolist()]
    return out


def run(args):
    from common import load_model_and_unembed

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "dola_mc_repro")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.data, args.n_questions, args.seed_subset)
    validate_questions(questions, verbose=False)
    t0 = time.time()
    model, tokenizer, _, _, _ = load_model_and_unembed(device, args.model)
    n_layers = model.cfg.n_layers
    depths = [d for d in range(0, n_layers, args.candidate_stride)]
    buckets = default_buckets(n_layers)
    print("=" * 78)
    print("DoLa JSD 分化预分析（S1 判停点）")
    print(f"  model={args.model} device={device} n_questions={len(questions)}")
    print(f"  n_layers={n_layers} mature_depth={n_layers} candidates={depths}")
    print(f"  buckets={buckets}（论文规则推广：nb=max(2,round(n_layers/20))）")
    print(f"  识别面：续写位置 + 题面尾 {args.prompt_positions} 位置；主判面 = {args.verdict_on}")
    print("=" * 78)

    acc = {"jsd_answer": {d: [] for d in depths}, "jsd_prompt": {d: [] for d in depths},
           "r_answer": {d: [] for d in depths}, "r_prompt": {d: [] for d in depths},
           "anchor_answer": [], "anchor_prompt": [],
           "sel_answer": {d: 0 for d in depths}, "sel_prompt": {d: 0 for d in depths},
           "selJ_answer": {d: 0 for d in depths}, "selJ_prompt": {d: 0 for d in depths}}
    n_used = 0
    for qi, q in enumerate(questions):
        ref_true, _, _ = refs_of(q)
        ans = ref_true[0]
        prompt, cont = build_prompt_and_answer(q["question"], ans)
        r = profile_question(model, prompt, cont, depths,
                             prompt_positions=args.prompt_positions, max_ctx=args.max_ctx)
        for d in depths:
            acc["jsd_answer"][d].extend(r["jsd_answer"][d])
            acc["jsd_prompt"][d].extend(r["jsd_prompt"][d])
            acc["r_answer"][d].extend(r["r_answer"][d])
            acc["r_prompt"][d].extend(r["r_prompt"][d])
        acc["anchor_answer"].extend(r["anchor_answer"])
        acc["anchor_prompt"].extend(r["anchor_prompt"])
        # 逐位置 argmax 层：官方选层量 R（部署口径）与严格 JSD 各统计一次
        for key, sel_key, selJ_key in (("jsd_answer", "sel_answer", "selJ_answer"),
                                       ("jsd_prompt", "sel_prompt", "selJ_prompt")):
            arr = np.array([r[key][d] for d in depths])       # [n_depth, n_pos] 严格 JSD
            arrR = np.array([r["r_" + key.split("_")[1]][d] for d in depths])
            if arr.size:
                for j in arr.argmax(axis=0):
                    acc[selJ_key][depths[int(j)]] += 1
                for j in arrR.argmax(axis=0):
                    acc[sel_key][depths[int(j)]] += 1
        n_used += 1
        if (qi + 1) % 20 == 0:
            print(f"  [{qi + 1}/{len(questions)}] {time.time() - t0:.0f}s")

    stats = {}
    for profile, key, sel_key, anch_key, rkey in (
            ("answer", "jsd_answer", "sel_answer", "anchor_answer", "r_answer"),
            ("prompt", "jsd_prompt", "sel_prompt", "anchor_prompt", "r_prompt")):
        mean_by_depth = {d: float(np.mean(acc[key][d])) if acc[key][d] else 0.0 for d in depths}
        med_by_depth = {d: float(np.median(acc[key][d])) if acc[key][d] else 0.0 for d in depths}
        q10 = {d: float(np.percentile(acc[key][d], 10)) if acc[key][d] else 0.0 for d in depths}
        q90 = {d: float(np.percentile(acc[key][d], 90)) if acc[key][d] else 0.0 for d in depths}
        r_mean = {d: float(np.mean(acc[rkey][d])) if acc[rkey][d] else 0.0 for d in depths}
        anchor = float(np.mean(acc[anch_key])) if acc[anch_key] else 0.0
        # 主判据：严格 JSD 的分化程度 + **部署口径 R** 的选层分布（方法实际选的是 R 的 argmax）
        v = verdict_from_stats(mean_by_depth, acc[sel_key], anchor, profile=profile)
        v_selJ = verdict_from_stats(mean_by_depth, acc["selJ_" + profile], anchor,
                                    profile=profile + "(严格JSD选层)")
        rho = float("nan")
        try:
            from scipy.stats import spearmanr

            rho = float(spearmanr(list(mean_by_depth.keys()), list(mean_by_depth.values())).statistic)
        except Exception:
            pass
        stats[profile] = {"mean_by_depth": {str(d): mean_by_depth[d] for d in depths},
                          "median_by_depth": {str(d): med_by_depth[d] for d in depths},
                          "p10_by_depth": {str(d): q10[d] for d in depths},
                          "p90_by_depth": {str(d): q90[d] for d in depths},
                          "mean_by_depth_x1e5": {str(d): mean_by_depth[d] * 1e5 for d in depths},
                          "r_mean_by_depth_x1e5": {str(d): r_mean[d] * 1e5 for d in depths},
                          "spearman_depth_vs_jsd": rho,
                          "n_positions": len(acc[key][depths[0]]) if acc[key][depths[0]] else 0,
                          "verdict": v, "verdict_sensitivity_true_jsd_selection": v_selJ}

    main_v = stats[args.verdict_on]["verdict"]
    print(f"\n  ── 层-分化曲线（严格 JSD 均值 ×1e5，识别面={args.verdict_on}）──")
    mx = max(1e-12, max(stats[args.verdict_on]["mean_by_depth_x1e5"].values()))
    for d in depths:
        m = stats[args.verdict_on]["mean_by_depth_x1e5"][str(d)]
        bar = "█" * max(1, int(round(m / mx * 40)))
        print(f"    d={d:>2}: {m:8.3f}  {bar}")
    print(f"  anchor JSD(成熟‖均匀) = {main_v['anchor_jsd_to_uniform'] * 1e5:.3f} ×1e-5"
          f" | Spearman(depth, JSD) = {stats[args.verdict_on]['spearman_depth_vs_jsd']:+.3f}")
    print(f"  (i) 相对分化 max/min = {main_v['ratio']:.3f} (≥{THR_JSD_RATIO}) → {main_v['c1_relative']}")
    print(f"  (ii) 选层 χ² 均匀性 p = {main_v['chi2_p']:.4g} (<{THR_CHI2_P}) → {main_v['c2_nonuniform']}"
          f"（选层量＝官方 R 口径 argmax）")
    print(f"  (iii) 绝对尺度 max/anchor = {main_v['max_mean_jsd'] / max(1e-12, main_v['anchor_jsd_to_uniform']):.4f}"
          f" (≥{THR_ANCHOR_FRAC}) → {main_v['c3_absolute']}")
    print(f"\n  ★ S1 判停：{main_v['verdict']}（{main_v['n_pass']}/3 条通过）")
    print(f"    敏感性（若用严格 JSD 的 argmax 选层）："
          f"{stats[args.verdict_on]['verdict_sensitivity_true_jsd_selection']['verdict']}"
          f"（{stats[args.verdict_on]['verdict_sensitivity_true_jsd_selection']['n_pass']}/3）")
    if main_v["verdict"].startswith("无分化"):
        print("  ⇒ 按红线**停在此步**：只报诊断，不进 S2，不得外推\"方法无效\"（I19 卡红线）")

    tag = args.tag or f"n{n_used}"
    model_tag = args.model.split("/")[-1]
    path = out_dir / f"jsd_profile_{model_tag}_{tag}.json"
    json.dump({"config": vars(args), "model": args.model, "n_layers": n_layers,
               "mature_depth": n_layers, "depths": depths, "buckets": buckets,
               "n_questions_used": n_used, "thresholds": {
                   "ratio": THR_JSD_RATIO, "chi2_p": THR_CHI2_P, "anchor_frac": THR_ANCHOR_FRAC},
               "quantity_note": "主曲线=严格 JSD；选层计数（ii）=官方 dola.py 的 R=0.5[KL(M‖q_N)+KL(M‖q_M)]",
               "stats": stats, "main_verdict": main_v, "timing_sec": time.time() - t0},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = path.with_suffix(".md")
    lines = [f"# DoLa JSD 分化预分析（{path.name}）\n",
             f"- model={args.model}，n_layers={n_layers}，mature_depth={n_layers}，题数={n_used}",
             f"- 候选早层（偶数含 0）={depths}；bucket={buckets}",
             f"- 预注册阈值：ratio≥{THR_JSD_RATIO}、χ² p<{THR_CHI2_P}、max/anchor≥{THR_ANCHOR_FRAC}",
             f"- 主曲线＝**严格 JSD**；条件 (ii) 的选层计数用**官方 R 口径**（部署实际选层量）",
             f"- **判停结论（主判面={args.verdict_on}）：{main_v['verdict']}（{main_v['n_pass']}/3）**\n",
             "| depth | 严格 JSD ×1e5 (mean) | median | p10 | p90 | 官方 R ×1e5 | R 选层计数 | J 选层计数 |",
             "|---|---|---|---|---|---|---|---|"]
    for d in depths:
        s = stats[args.verdict_on]
        lines.append(f"| {d} | {s['mean_by_depth_x1e5'][str(d)]:.3f} | {s['median_by_depth'][str(d)] * 1e5:.3f} | "
                     f"{s['p10_by_depth'][str(d)] * 1e5:.3f} | {s['p90_by_depth'][str(d)] * 1e5:.3f} | "
                     f"{s['r_mean_by_depth_x1e5'][str(d)]:.3f} | "
                     f"{main_v['argmax_counts'][str(d)]} | "
                     f"{s['verdict_sensitivity_true_jsd_selection']['argmax_counts'][str(d)]} |")
    lines.append(f"\n- anchor JSD(成熟‖均匀) = {main_v['anchor_jsd_to_uniform'] * 1e5:.3f} ×1e-5；"
                 f"Spearman(depth, mean 严格 JSD) = {stats[args.verdict_on]['spearman_depth_vs_jsd']:+.3f}")
    lines.append(f"- 对照论文图 2（LLaMA-7B，×10^5 量级、随深度递降、实体/日期 token 深处仍高）")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {path}\nSaved → {md}")


# ── 零 GPU 自检（判停逻辑 + bucket 规则）────────────────────────────────────────


def selftest():
    print("=" * 78)
    print("JSD 诊断脚本零 GPU 自检")
    print("=" * 78)
    fails = []
    # 1) bucket 规则与论文模式一致（32→[0,16)/[16,32)；40→2 桶；60→3；80→4）
    for n, expect in ((32, [(0, 16), (16, 32)]), (40, [(0, 20), (20, 40)]),
                      (60, [(0, 20), (20, 40), (40, 60)]), (80, [(0, 20), (20, 40), (40, 60), (60, 80)]),
                      (28, [(0, 14), (14, 28)]), (36, [(0, 18), (18, 36)])):
        got = default_buckets(n)
        ok = got == expect
        print(f"  [1] default_buckets({n}) = {got} {'PASS' if ok else 'FAIL 期望 ' + str(expect)}")
        if not ok:
            fails.append(f"buckets:{n}")
    # 2) 桶内候选=偶数含 0、排除成熟层
    cnd = candidates_in_bucket(0, 14, 28)
    ok = cnd == [0, 2, 4, 6, 8, 10, 12]
    print(f"  [2] candidates_in_bucket(0,14,28) = {cnd} {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("candidates")

    # 3) 判停逻辑：平坦 ⇒ 无分化；强分化 ⇒ 分化存在；部分 ⇒ 灰区
    flat = {d: 1.0e-3 for d in range(0, 14, 2)}
    flat_counts = {d: 100 for d in range(0, 14, 2)}
    v_flat = verdict_from_stats(flat, flat_counts, anchor=1.0)
    ok_flat = v_flat["verdict"].startswith("无分化")
    print(f"  [3a] 平坦谱 → {v_flat['verdict']} {'PASS' if ok_flat else 'FAIL'}（{v_flat['n_pass']}/3）")
    if not ok_flat:
        fails.append("verdict:flat")

    diff = {d: 1.0e-5 * (2 ** i) for i, d in enumerate(range(0, 14, 2))}
    diff_counts = {d: (1000 if d == 12 else 1) for d in range(0, 14, 2)}
    v_diff = verdict_from_stats(diff, diff_counts, anchor=0.005)
    ok_diff = v_diff["verdict"] == "分化存在"
    print(f"  [3b] 强分化谱 → {v_diff['verdict']} {'PASS' if ok_diff else 'FAIL'}（{v_diff['n_pass']}/3）")
    if not ok_diff:
        fails.append("verdict:diff")

    part = {d: 1.0e-5 for d in range(0, 14, 2)}
    part[0] = 3.0e-5
    v_part = verdict_from_stats(part, {d: 50 for d in range(0, 14, 2)}, anchor=0.01)
    ok_part = v_part["verdict"].startswith("灰区")
    print(f"  [3c] 部分通过谱 → {v_part['verdict']} {'PASS' if ok_part else 'FAIL'}（{v_part['n_pass']}/3）")
    if not ok_part:
        fails.append("verdict:partial")

    # 4) 锚点：均匀分布 → JSD(·‖uniform) = 0；单点分布 → 有限 V 的解析值（V→∞ 才等于 ln2）
    V = 1024
    uni = torch.zeros(1, V)
    a_uni = float(jsd_to_uniform(uni).mean().item())
    onehot = torch.full((1, V), -50.0)
    onehot[0, 0] = 50.0
    a_one = float(jsd_to_uniform(onehot).mean().item())
    m0 = (1.0 + 1.0 / V) / 2.0
    exp_one = 0.5 * (math.log(1.0 / m0)
                     + (1.0 / V) * (math.log(1.0 / V) - math.log(m0))
                     + ((V - 1.0) / V) * math.log(2.0))
    ok_a = abs(a_uni) < 1e-9 and abs(a_one - exp_one) < 1e-3
    print(f"  [4] 锚点：均匀={a_uni:.3g}（应 0）、单点={a_one:.6f}"
          f"（有限 V={V} 解析值 {exp_one:.6f}，ln2 上界 {math.log(2):.6f}）{'PASS' if ok_a else 'FAIL'}")
    if not ok_a:
        fails.append("anchor")

    print(f"\n  总判：{'PASS ✅' if not fails else 'FAIL ❌ ' + str(fails)}")
    if fails:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description="DoLa 图 2 式 JSD 分化预分析（S1 判停点）")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n_questions", type=int, default=100)
    ap.add_argument("--seed_subset", type=int, default=42)
    ap.add_argument("--candidate_stride", type=int, default=2, help="候选早层步长（论文=偶数层）")
    ap.add_argument("--prompt_positions", type=int, default=64, help="题面尾段位置数（诊断面之一）")
    ap.add_argument("--verdict_on", type=str, default="answer", choices=["answer", "prompt"])
    ap.add_argument("--max_ctx", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == "__main__":
    main()
