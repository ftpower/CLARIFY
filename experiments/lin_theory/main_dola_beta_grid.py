"""E2：DoLa 原生域对比强度 λ 网格 —— C1 在 DoLa 参数化下的唯一存活形态的验证。

理论（执行前，见 docs/protocol/dola-c1-validation-20260925.md §5.3）：
    官方 MC 口径下 DoLa 分数 = Σ_p diff_p(y_p)（λ=1 官方；转录已 O1/O2 逐位核验）
    ⇒ score(λ) = λ·score(1) ⇒ **选项排序/MC1/MC3 对一切 λ>0 恒等、MC2 仅温度缩放**
    ⇒ 官方口径下"逐样本对比强度"无决策效应（定理级）；
    唯一有决策效应的口径 = 归一化变体（λ·diff 再做 log_softmax，λ 缩放不等价于分数缩放）。
本脚本：一次前向/选项，扫 λ∈{0.125..4.0} × 两侧口径 × {baseline, static_d12, dyn_b1_14_28}；
    每问题另算 λ*(q)（归一化口径下使排序偏离官方 R₀ 的最小 λ）。
自检（零 GPU）：λ=1 官方口径与转录算子逐位一致（随机 logits ×300）+ 惰性定理算子级验证。
判读（零 GPU，--judge）：① 官方口径 MC1/MC3 跨 λ 恒等验证；② 归一化逐 λ 曲线；
    ③ λ* 分布与判停规则；④ λ=1 官方口径 vs 已发布 full817.json 逐位协议自检。

用法：
    python experiments/lin_theory/main_dola_beta_grid.py --selftest
    python experiments/lin_theory/main_dola_beta_grid.py --model Qwen/Qwen3-1.7B
    python experiments/lin_theory/main_dola_beta_grid.py --judge <结果 json>

输出：experiments/outputs/dola_mc_betagrid/betagrid_<model>_n817.json（+ judge_betagrid.md）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_PARENT = Path(__file__).resolve().parent
for _p in [str(_PARENT), str(_PARENT.parent / "phase2_entropy"),
           str(_PARENT.parent / "phase4_generalization"), str(_PARENT.parent / "phase5_cross_task")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import load_model_and_unembed  # noqa: E402
from main_dola_mc import (  # noqa: E402
    MC_calcs, _diff_logits, _make_project_fn, build_prompt_and_answer,
    candidates_in_bucket, depth_to_hook_name, encode_pair, jsd_mean,
    load_questions, refs_of,
)

LAMBDAS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
STATIC_DEPTH = 12
BUCKET = (14, 28)          # dyn_b1_14_28（两折一致选中的桶）
CONDITIONS = ("baseline", "static_d12", "dyn_b1_14_28")


# ═════════════════════════════════════════════════════════════════════════════
# λ 网格算子（纯函数；selftest 与转录算子逐位对照）
# ═════════════════════════════════════════════════════════════════════════════


def score_off(mature_logits, pre_logits, cont_ids, lam=1.0):
    """官方口径：λ·Σ_p diff_p(y_p)，diff = logsm_N − logsm_M（λ=1 即转录算子）。"""
    diff = _diff_logits(mature_logits, pre_logits, False, 0.0, -1000.0)
    idx = torch.arange(cont_ids.shape[0])
    return float((lam * diff[idx, cont_ids]).sum().item())


def grid_scores(mature_logits, pre_logits, cont_ids, lambdas):
    """{λ: {"off", "ps"}}。

    off：score_off(λ) = λ·score_off(1)（惰性定理的算子表述）；
    ps（归一化口径）：d_λ = λ·diff，分数 = Σ_p [d_λ(y_p) − logsumexp(d_λ)]（缩放后再归一化，
    λ 缩放不等价于分数缩放 ⇒ 排序可随 λ 变化）。
    """
    idx = torch.arange(cont_ids.shape[0])
    diff = _diff_logits(mature_logits, pre_logits, False, 0.0, -1000.0)  # [npos, V]
    off1 = float(diff[idx, cont_ids].sum().item())
    out = {}
    for lam in lambdas:
        dlam = lam * diff
        ps = float((dlam[idx, cont_ids] - dlam.logsumexp(dim=-1)).sum().item())
        out[str(lam)] = {"off": lam * off1, "ps": ps, "lam": lam}
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 自检（零 GPU）
# ═════════════════════════════════════════════════════════════════════════════


def selftest():
    torch.manual_seed(0)
    ok = []
    n_bad = 0
    for _ in range(300):
        V, npos = 40, 6
        mature = torch.randn(npos, V) * 3
        pre = torch.randn(npos, V) * 3
        ids = torch.randint(0, V, (npos,))
        mine = score_off(mature, pre, ids, 1.0)
        refv = float(_diff_logits(mature, pre, False, 0.0, -1000.0)
                     [torch.arange(npos), ids].sum().item())
        n_bad += int(abs(mine - refv) > 1e-6)
    ok.append(("S-A λ=1 官方口径 ≡ 转录算子（300 组随机 logits）", n_bad == 0, f"mismatch={n_bad}"))

    mature = torch.randn(5, 30) * 3
    pre = torch.randn(5, 30) * 3
    ids = torch.randint(0, 30, (5,))
    s1 = score_off(mature, pre, ids, 1.0)
    lin = all(abs(score_off(mature, pre, ids, lam) - lam * s1) < 1e-4 * (1 + abs(lam * s1))
              for lam in LAMBDAS)
    ok.append(("S-B 惰性定理算子级：score_off(λ) ≡ λ·score_off(1)（全部 λ）", lin, ""))

    # 归一化口径：缩放后再归一化 ⇒ 分数随 λ 非平凡变化
    gs = grid_scores(mature, pre, ids, LAMBDAS)
    vals = {round(gs[str(lam)]["ps"], 8) for lam in LAMBDAS}
    ok.append(("S-C 归一化口径随 λ 变化（非恒等）", len(vals) >= 3, f"{len(vals)} 个不同值"))
    # 同题不同 λ 下官方口径的**排序**不变（惰性定理的排序级验证）
    gs2 = grid_scores(mature, pre, ids, LAMBDAS)
    ok.append(("S-D 官方口径分数跨 λ 仅差常数（排序不变）",
               all(abs((gs2[str(lam)]["off"] - lam * s1)) < 1e-9 for lam in LAMBDAS), ""))
    print("[selftest]")
    for name, good, extra in ok:
        print(f"  {'✅' if good else '❌'} {name} {extra}")
    return 0 if all(g for _, g, _ in ok) else 1


# ═════════════════════════════════════════════════════════════════════════════
# 前向（一次前向/选项 → 全 λ × 两侧口径 × 条件）
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def score_choice_grid(model, prompt, cont_text, depths, max_ctx=0, lens_check=True):
    _, full_ids, cont_ids, prefix_len = encode_pair(model, prompt, cont_text, max_ctx=max_ctx)
    n_pos = int(cont_ids.shape[0])
    if n_pos == 0:
        return None
    positions = torch.arange(prefix_len - 1, full_ids.shape[1] - 1, device=full_ids.device)
    store = {}
    fwd_hooks = []
    for d in sorted(set(depths)):
        name = depth_to_hook_name(model, d)

        def _cap(act, hook=None, _d=d):
            store[_d] = act[0, positions, :].detach().float()
            return act

        fwd_hooks.append((name, _cap))
    if lens_check:
        name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

        def _cap_mature(act, hook=None):
            store["mature_lens_h"] = act[0, positions, :].detach().float()
            return act

        fwd_hooks.append((name, _cap_mature))
    real_logits = model.run_with_hooks(full_ids, fwd_hooks=fwd_hooks)
    mature = real_logits[0, prefix_len - 1: full_ids.shape[1] - 1, :].float()
    project_fn = _make_project_fn(model, store, mature)

    mature_logsm = F.log_softmax(mature.float(), dim=-1)
    out = {}
    # baseline：与转录一致（Σ logsm_N(y_p)）；无对比项 ⇒ 两侧口径同值（baseline__ps1 ≡ baseline）
    bs = float(mature_logsm[torch.arange(n_pos), cont_ids].sum().item())
    out["baseline"] = {str(lam): {"off": bs, "ps": bs, "lam": lam} for lam in LAMBDAS}
    out["static_d12"] = grid_scores(mature, project_fn(STATIC_DEPTH), cont_ids, LAMBDAS)
    # dyn_b1_14_28：逐位置桶内 argmax JSD（选层与 λ 无关，只算一次）
    bucket = candidates_in_bucket(BUCKET[0], BUCKET[1], model.cfg.n_layers)
    jsd_tab = {d: jsd_mean(mature, project_fn(d)) for d in bucket}
    stack = torch.stack([jsd_tab[d] for d in bucket], dim=0)
    sel_pos = stack.argmax(dim=0)
    sel = [int(bucket[int(s)]) for s in sel_pos]
    dyn = {str(lam): {"off": 0.0, "ps": 0.0, "lam": lam} for lam in LAMBDAS}
    for j, d in enumerate(bucket):
        pos_mask = sel_pos == j
        if not bool(pos_mask.any()):
            continue
        gs = grid_scores(mature[pos_mask], project_fn(d)[pos_mask], cont_ids[pos_mask], LAMBDAS)
        for lam in LAMBDAS:
            dyn[str(lam)]["off"] += gs[str(lam)]["off"]
            dyn[str(lam)]["ps"] += gs[str(lam)]["ps"]
    out["dyn_b1_14_28"] = dyn
    out["_sel"] = sel
    out["_n_pos"] = n_pos
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 判读（零 GPU）
# ═════════════════════════════════════════════════════════════════════════════


def _mcv(pq, cond, lam, conv, k):
    return np.array([r["mc"][cond][str(lam)][conv][k] for r in pq], dtype=float)


def judge(path):
    d = json.loads(Path(path).read_text())
    pq, published = d["per_question"], d.get("published_check", {})
    lines = ["# E2 判读：DoLa 原生域对比强度 λ 网格（判据：dola-c1-validation-20260925.md §5.3）\n",
             f"数据：`{Path(path).name}`（n={len(pq)}）\n",
             "## ① 惰性定理实测（官方口径：MC1/MC3 跨 λ 恒等，MC2 仅温度缩放）\n",
             "| 条件 | MC1 跨 λ 极差 | MC3 跨 λ 极差 | MC2(λ=0.125) | MC2(λ=1) | MC2(λ=4) |",
             "|---|---|---|---|---|---|"]
    inert_ok = True
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        mc1 = np.stack([_mcv(pq, cond, lam, "off", 0) for lam in LAMBDAS])
        mc2 = np.stack([_mcv(pq, cond, lam, "off", 1) for lam in LAMBDAS])
        mc3 = np.stack([_mcv(pq, cond, lam, "off", 2) for lam in LAMBDAS])
        r1 = float(np.abs(mc1.max(axis=0) - mc1.min(axis=0)).max())
        r3 = float(np.abs(mc3.max(axis=0) - mc3.min(axis=0)).max())
        inert_ok &= (r1 == 0.0 and r3 == 0.0)
        lines.append(f"| `{cond}` | {r1:g} | {r3:g} | {mc2[0].mean():.4f} | "
                     f"{mc2[LAMBDAS.index(1.0)].mean():.4f} | {mc2[-1].mean():.4f} |")
    lines.append(f"\n- 惰性定理（MC1/MC3 逐题跨 λ 极差 = 0）：{'✅ 实测成立' if inert_ok else '❌ 违反（须查实现）'}")

    lines += ["\n## ② 归一化口径逐 λ 曲线（主条件 dyn_b1_14_28，相对 baseline 的绝对百分点）\n",
              "| λ | ΔMC1 | ΔMC2 | ΔMC3 |", "|---|---|---|---|"]
    for lam in LAMBDAS:
        lines.append(f"| {lam} | {(_mcv(pq,'dyn_b1_14_28',lam,'ps',0)-_mcv(pq,'baseline',lam,'ps',0)).mean()*100:+.2f} | "
                     f"{(_mcv(pq,'dyn_b1_14_28',lam,'ps',1)-_mcv(pq,'baseline',lam,'ps',1)).mean()*100:+.2f} | "
                     f"{(_mcv(pq,'dyn_b1_14_28',lam,'ps',2)-_mcv(pq,'baseline',lam,'ps',2)).mean()*100:+.2f} |")

    # ③ λ*（归一化排序偏离官方 R₀ = 官方 λ=1 排序的最小 λ）
    lam_star, n_none = [], 0
    for r in pq:
        keys = sorted(r["choice_scores"]["off_1.0"], key=int)
        off = np.array([r["choice_scores"]["off_1.0"][k] for k in keys])
        r0 = np.argsort(-off, kind="stable")
        changed = None
        for lam in LAMBDAS:
            ps = np.array([r["choice_scores"][f"ps_{lam}"][k] for k in keys])
            if not np.array_equal(np.argsort(-ps, kind="stable"), r0):
                changed = lam
                break
        if changed is None:
            n_none += 1
        else:
            lam_star.append(changed)
    share_none = n_none / len(pq)
    lines += ["\n## ③ λ\\*（归一化排序偏离官方 R₀ 的最小 λ）\n"]
    if lam_star:
        a = np.array(lam_star)
        lines.append(f"- 存在 λ\\* 的题：{len(lam_star)}/{len(pq)}（{1 - share_none:.1%}）；"
                     f"λ\\* 分布：median {np.median(a):.3f}、p25 {np.percentile(a, 25):.3f}、"
                     f"p75 {np.percentile(a, 75):.3f}、min {a.min():.3f}、max {a.max():.3f}")
    else:
        lines.append("- 全部题目无 λ\\*（归一化排序在 λ∈[0.125,4] 内从不偏离官方排序）")
    trigger = share_none >= 0.80
    lines.append(f"- 无 λ\\* 占比：**{share_none:.1%}** ⇒ 判停规则（≥80% ⇒ C1 原生域同样惰性 ⇒ 关闭）："
                 f"{'⚠️ 触发 ⇒ 关闭' if trigger else '未触发（进入 λ*(q) vs λ=1 配对比较）'}")

    # ④ 协议自检：λ=1 官方口径 vs 已发布 full817 summary
    if published:
        lines += ["\n## ④ 协议自检（λ=1 官方口径 vs 已发布 full817 summary）\n"]
        all_ok = True
        for cond, got in published.items():
            okv = abs(got["mc2"] - got["ref_mc2"]) < 1e-9
            all_ok &= okv
            lines.append(f"- `{cond}`: 本跑 MC2={got['mc2']:.10f} vs 发布 {got['ref_mc2']:.10f} ⇒ "
                         f"{'✅' if okv else '❌ 不一致（须查）'}")
        lines.append(f"\n协议自检总体：{'✅ 逐位一致' if all_ok else '❌'}")
    out = Path(path).parent / "judge_betagrid.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="E2：DoLa 原生域 λ 网格")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n_questions", type=int, default=None)
    ap.add_argument("--seed_subset", type=int, default=42)
    ap.add_argument("--max_ctx", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--judge", type=str, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.judge:
        return judge(args.judge)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (_PARENT.parent / "outputs" / "dola_mc_betagrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _, _, _ = load_model_and_unembed(device, args.model)
    questions = load_questions(_PARENT.parent / "data" / "truthfulqa_mc_817.json",
                               args.n_questions, args.seed_subset)
    depths = sorted(set([STATIC_DEPTH] + candidates_in_bucket(BUCKET[0], BUCKET[1], model.cfg.n_layers)))
    per_question = []
    for qi, q in enumerate(questions):
        ref_true, ref_false, ref_best = refs_of(q)
        if ref_best not in ref_true:
            ref_true = [ref_best] + [a for a in ref_true if a != ref_best]
        cache = {}
        for ans in list(ref_true) + list(ref_false):
            if ans in cache:
                continue
            prompt, cont = build_prompt_and_answer(q["question"], ans)
            cache[ans] = score_choice_grid(model, prompt, cont, depths, max_ctx=args.max_ctx)
        rec = {"qi": qi, "n_true": len(ref_true), "n_false": len(ref_false),
               "choice_scores": {}, "mc": {}}
        for cond in CONDITIONS:
            rec["mc"][cond] = {}
            for lam in LAMBDAS:
                st_o = [cache[a][cond][str(lam)]["off"] for a in ref_true]
                sf_o = [cache[a][cond][str(lam)]["off"] for a in ref_false]
                mc_off = MC_calcs(st_o, sf_o, ref_true, ref_best)
                st_p = [cache[a][cond][str(lam)]["ps"] for a in ref_true]
                sf_p = [cache[a][cond][str(lam)]["ps"] for a in ref_false]
                mc_ps = MC_calcs(st_p, sf_p, ref_true, ref_best)
                rec["mc"][cond][str(lam)] = {
                    "off": [mc_off["MC1"], mc_off["MC2"], mc_off["MC3"]],
                    "ps": [mc_ps["MC1"], mc_ps["MC2"], mc_ps["MC3"]]}
        rec["choice_scores"]["off_1.0"] = {str(i): float(cache[a]["dyn_b1_14_28"]["1.0"]["off"])
                                           for i, a in enumerate(ref_true + ref_false)}
        for lam in LAMBDAS:
            rec["choice_scores"][f"ps_{lam}"] = {str(i): float(cache[a]["dyn_b1_14_28"][str(lam)]["ps"])
                                                 for i, a in enumerate(ref_true + ref_false)}
        per_question.append(rec)
        if qi % 100 == 0:
            print(f"  {qi}/{len(questions)}")
    pub_path = _PARENT.parent / "outputs" / "dola_mc_repro" / "dola_mc_Qwen3-1.7B_full817.json"
    published = {}
    if pub_path.exists() and len(questions) == 817:
        pub = json.loads(pub_path.read_text())
        for cond in ["static_d12", "dyn_b1_14_28"]:
            got = float(np.mean(_mcv(per_question, cond, "1.0", "off", 1)))
            published[cond] = {"mc2": got, "ref_mc2": pub["summary"][cond]["MC2"]}
    tag = Path(str(args.model).rstrip("/")).name
    out_path = out_dir / f"betagrid_{tag}_n{len(questions)}.json"
    out_path.write_text(json.dumps({"config": vars(args), "lambdas": LAMBDAS,
                                    "conditions": list(CONDITIONS),
                                    "per_question": per_question,
                                    "published_check": published}, ensure_ascii=False))
    print(f"[写出] {out_path}")
    print(f"判读：python experiments/lin_theory/main_dola_beta_grid.py --judge {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
