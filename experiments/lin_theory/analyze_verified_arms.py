"""验证器臂判读（V1/V3）——零 GPU，只读 `main_tldc_controls.py` 的输出。

依据 docs/protocol/review-runbook-20260921.md §5.8b（**预注册，禁止事后更改判据**）
与 docs/plans/current.md §P0-TLDC 的 V1/V3 条目：

  规则（已实现于臂内，θ 预注册唯一出处＝`verify_decision`）：干预后在 t+1 步读 β*_min，
  ≥θ 则**保留**翻转、<θ 则**回退**；None（下一步 R 空＝怎么翻都翻不动）视作最稳固 ⇒ 保留。

  判读（V1/V3，与门控族 §9 的判据**不同**，不得混用）：
    ① **净效应**：net(verified) vs net(real)（全体事件数与 pp）——投影预期 θ=0.5 下
       seed123 n=300：**−6 → +3 事件**（见 `posthoc_direction/seed123`）。
       ⚠️ 验证器**必然丢一部分救回**（回退即放弃翻转），故 KW 救回"不显著低于 real"
       **不是**本路线的判据（与 T3 门控的 (i) 条相反）。
    ② **配对交换比**（本路线真正的选择性度量）：
       avoided = KC 中"仅 real 破坏、verified 未破坏"；lost = KW 中"仅 real 救回、verified 未救回"。
       比值 avoided:lost > 1 才说明验证器**择优保留**（而非等比缩小）。
       ⚠️ 事件数小时比值不稳（§9d 教训）⇒ 必须与绝对事件数并列报。
    ③ 配对检验（描述性）：KC 破坏侧单侧 McNemar（H1＝verified 破坏更少）、
       KW 救回侧单侧 McNemar（H1＝verified 救回更少，属**预期代价**）；全样本配对 McNemar。
    ④ 验证器诊断：逐子集 flip_rate / keep_rate_of_flips / mean_kept_beta / mean_reverted_beta。
    ⑤ **安慰剂对照（2026-09-23 新增，V1 的 footprint 混淆排除）**：`verified_rand` 与 `verified_sym`
       共用同一翻转规则，但保留/回退由 Bernoulli(p) 随机决定，p 逐子集＝V1 实测 `keep_rate_of_flips`
       （KW 0.6053 / KC 0.6991 / DK 0.6506）⇒ 只保留"扰动步数（footprint）"、破坏"选择信息"。
       判据（`placebo_verdict`，先注册后看结果）：sym 相对 rand 净事件 >0 且交换比 >1 ⇒ 增益不可由
       footprint 解释；否则 ⇒ 1.7B 上验证器路线不成立（判停仍在 V2/8B）。
       ⚠️ 轨迹在首次决策分歧后分叉 ⇒ 两臂翻转集只在分叉前相同（轨迹级限制，非实现缺陷）。

  判停线：**不在本脚本、也不在 1.7B**——V2（`geometry-sym-8b`）预注册判停：8B `post_beta`
  AUROC < 0.60 ⇒ 验证器路线关闭；V1 只作"投影是否兑现"的实测检验（1.7B 属 H2 区）。

用法：
    python experiments/lin_theory/analyze_verified_arms.py \
        experiments/outputs/tldc_verified/tldc_controls_123_real-verified_sym.json \
        [--betas 0.2] [--projection_json experiments/outputs/posthoc_direction_1p7b/seed123/posthoc_direction.json] \
        [--output_dir experiments/outputs/tldc_verified/judge]

  ⚠️ 多文件（多 seed）自动按 seed 前缀复合键 pool；同 tag 重复输入会报警（同 analyze_gated_arms.py 约定）。
"""

import argparse
import json
import sys
from pathlib import Path

from analyze_gated_arms import (  # 复用同一套装载与配对口径
    load_files,
    mcnemar_one_sided_less,
    paired_events,
)
from analyze_transition_matrix import clopper_pearson, fmt_ci, fmt_pct, mcnemar_exact, transition

VERIFY_ARMS = ("verified_sym",)
VERIFY_RAND_ARMS = ("verified_rand",)
VERIFY_FAMILY = VERIFY_ARMS + VERIFY_RAND_ARMS
PLACEBO_ARM_DEFAULT = "verified_rand"
BETA_PRIMARY_DEFAULT = 0.20
KW, KC, DK = "know_wrong", "know_correct", "dont_know"


def placebo_verdict(r, dnet_events):
    """真验证器 vs 安慰剂（footprint 匹配）的判定文字（**先注册规则，后看结果**）。

    规则（判据只此一条，按顺序生效）：
      ① dnet_events > 0 且交换比 > 1（避免破坏 : 丢失救回，以安慰剂为基准）⇒ 增益不可由 footprint 解释；
      ② dnet_events > 0 但交换比 ≤ 1 ⇒ 净事件提升但选择性不占优 ⇒ 只报数、不下机制结论；
      ③ dnet_events ≤ 0 ⇒ 增益可由 footprint 解释 ⇒ **1.7B 上验证器路线不成立**（判停依 V2/8B）。
    """
    ratio = r["swap_ratio"]
    if dnet_events > 0 and ratio is not None and ratio > 1:
        return "**不可由 footprint 解释**（净事件更高 ∧ 交换比 >1）⇒ β*_min 选择信息有效"
    if dnet_events > 0:
        return ("净事件更高但**选择性不占优**（交换比 ≤1 或分母 0）⇒ 只报数、不下机制结论"
                "（事件数小，§9d 教训）")
    return "**可由 footprint 解释**（净事件未高于安慰剂）⇒ 1.7B 上验证器路线不成立"


def all_sample_paired(real_bucket, ctrl_bucket):
    """全样本配对（不分桶）：gain＝仅 verified 正确，loss＝仅 real 正确。

    恒等式：gain − loss ≡ net(verified) − net(real)（事件数）。
    """
    shared = set(real_bucket) & set(ctrl_bucket)
    gain = loss = 0
    for sid in shared:
        _, _, real_after = real_bucket[sid]
        ctrl_after = ctrl_bucket[sid][2]
        real_ok, ctrl_ok = bool(real_after), bool(ctrl_after)
        if ctrl_ok and not real_ok:
            gain += 1
        elif real_ok and not ctrl_ok:
            loss += 1
    return gain, loss, len(shared)


def verify_diagnostics(gates, pooled, arm, beta):
    """逐子集聚合验证器统计（keep/回退比例、被保留与被回退的 β* 均值）。"""
    rows = gates.get((arm, beta))
    bucket = pooled.get((arm, beta))
    if not rows or not bucket:
        return None
    agg = {}
    for sid, gs in rows.items():
        if sid not in bucket:
            continue
        sub = bucket[sid][0]
        a = agg.setdefault(sub, {"n": 0, "flips": 0, "steps": 0, "kept": 0, "reverted": 0,
                                 "kept_beta_sum": 0.0, "n_kept_beta": 0,
                                 "rev_beta_sum": 0.0, "n_rev_beta": 0,
                                 "n_noflip": 0, "theta": None})
        a["n"] += 1
        a["flips"] += int(gs.get("n_flip") or 0)
        a["steps"] += int(gs.get("n_steps") or 0)
        a["kept"] += int(gs.get("n_kept") or 0)
        a["reverted"] += int(gs.get("n_reverted") or 0)
        a["n_noflip"] += int(gs.get("n_noflip") or 0)
        if gs.get("mean_kept_beta") is not None:
            a["kept_beta_sum"] += gs["mean_kept_beta"] * int(gs.get("n_kept") or 0)
            a["n_kept_beta"] += int(gs.get("n_kept") or 0)
        if gs.get("mean_reverted_beta") is not None:
            a["rev_beta_sum"] += gs["mean_reverted_beta"] * int(gs.get("n_reverted") or 0)
            a["n_rev_beta"] += int(gs.get("n_reverted") or 0)
        if gs.get("verify_theta") is not None:
            a["theta"] = gs["verify_theta"]
    for a in agg.values():
        a["flip_rate_of_steps"] = (a["flips"] / a["steps"]) if a["steps"] else None
        a["keep_rate_of_flips"] = (a["kept"] / a["flips"]) if a["flips"] else None
        a["mean_kept_beta"] = (a["kept_beta_sum"] / a["n_kept_beta"]) if a["n_kept_beta"] else None
        a["mean_reverted_beta"] = (a["rev_beta_sum"] / a["n_rev_beta"]) if a["n_rev_beta"] else None
        for k in ("kept_beta_sum", "rev_beta_sum", "n_kept_beta", "n_rev_beta"):
            a.pop(k, None)
    return agg


def judge_one(pooled, real_key, ctrl_key):
    """单臂配对判读 → 统计字典（可单测：只依赖两个 bucket 字典）。"""
    real_b, ctrl_b = pooled[real_key], pooled[ctrl_key]
    # 救回侧（base=False）：b=仅 real 救回（＝丢失的救回），c=仅 verified 救回（＝新增救回）
    lost_rescues, extra_rescues, n_kw = paired_events(real_b, ctrl_b, KW, False)
    # 破坏侧（base=True）：b=仅 real 破坏（＝避免的破坏），c=仅 verified 破坏（＝新增破坏）
    avoided_breaks, new_breaks, n_kc = paired_events(real_b, ctrl_b, KC, True)
    gain, loss, n_all = all_sample_paired(real_b, ctrl_b)
    in_scope = (avoided_breaks + extra_rescues) - (lost_rescues + new_breaks)
    return {
        "n_kw": n_kw, "n_kc": n_kc, "n_all": n_all,
        "lost_rescues": lost_rescues, "extra_rescues": extra_rescues,
        "avoided_breaks": avoided_breaks, "new_breaks": new_breaks,
        "net_delta_events_kwkc": in_scope,
        "gain_all": gain, "loss_all": loss, "net_delta_events_all": gain - loss,
        "p_break_one_sided": mcnemar_one_sided_less(avoided_breaks, new_breaks),
        "p_rescue_one_sided": mcnemar_one_sided_less(lost_rescues, extra_rescues),
        "p_all_two_sided": mcnemar_exact(gain, loss),
        "swap_ratio": (avoided_breaks / lost_rescues) if lost_rescues else None,
    }


def swap_verdict(r):
    """交换比文字判定（⚠️ 分母为 0 时**不能**按"不可算"简单归为不利）。

    avoided=0 & lost=0 ⇒ 无差异；lost=0 & avoided>0 ⇒ 零丢失救回（最优形态，比值无上界）；
    其余按比值是否 >1 判。⚠️ 事件数小时比值不稳（§9d）⇒ 判定须与绝对事件数并列读。
    """
    av, ls = r["avoided_breaks"], r["lost_rescues"]
    if av == 0 and ls == 0:
        return "无差异（两侧事件均为 0）"
    if ls == 0:
        return "**有利（零丢失救回，比值无上界）**"
    if av == 0:
        return "不利（只丢救回、未避破坏）"
    return "有利（择优保留）" if av / ls > 1 else "不利（近似等比缩小）"


def selftest():
    """极小合成例：核对配对记账与恒等式（gain − loss ≡ net 差）。"""
    def mk(rows):
        return {f"x:{i}": r for i, r in enumerate(rows)}

    # real: KW 错→对 3（救回），KC 对→错 2（破坏），DK 无变化
    real = mk([(KW, False, True)] * 3 + [(KC, True, False)] * 2 + [(DK, False, False)] * 2)
    # verified: 丢掉 1 个救回、避免 2 个破坏、另有 1 个 DK 由错变对
    ctrl = mk([(KW, False, True)] * 2 + [(KW, False, False)] * 1
              + [(KC, True, True)] * 2 + [(DK, False, True)] * 1 + [(DK, False, False)] * 1)
    s = judge_one({"real": real, "ver": ctrl}, "real", "ver")
    assert (s["lost_rescues"], s["extra_rescues"]) == (1, 0), s
    assert (s["avoided_breaks"], s["new_breaks"]) == (2, 0), s
    assert s["swap_ratio"] == 2.0, s
    # KW/KC 内部净变化 = (2+0)-(1+0) = +1；全样本 = (1+2 增益) - (1 损失) = +2
    assert s["net_delta_events_kwkc"] == 1, s
    assert (s["gain_all"], s["loss_all"], s["net_delta_events_all"]) == (3, 1, 2), s
    # 恒等式：net_delta_all ≡ net(verified) − net(real)
    t_r, t_v = transition(list(real.values())), transition(list(ctrl.values()))
    assert s["net_delta_events_all"] == t_v["net_events"] - t_r["net_events"], s
    # 退化例：完全相同的两臂 ⇒ 全零、比值 None、p=1
    s0 = judge_one({"real": real, "ver": dict(real)}, "real", "ver")
    assert (s0["lost_rescues"], s0["avoided_breaks"], s0["net_delta_events_all"]) == (0, 0, 0), s0
    assert s0["swap_ratio"] is None and s0["p_break_one_sided"] == 1.0, s0
    # 交换比文字判定：分母为 0 的两种极端不得被误判为"不利"（初版 bug）
    assert swap_verdict({"avoided_breaks": 9, "lost_rescues": 0}).startswith("**有利"), "零丢失应判最优"
    assert swap_verdict({"avoided_breaks": 0, "lost_rescues": 3}).startswith("不利")
    assert swap_verdict({"avoided_breaks": 0, "lost_rescues": 0}).startswith("无差异")
    # 安慰剂判定（规则先注册）：三条分支各自可达，且不得把"净提升但选择性不占优"判成有效
    assert "不可由 footprint 解释" in placebo_verdict({"swap_ratio": 2.0}, 3)
    assert "选择性不占优" in placebo_verdict({"swap_ratio": 0.5}, 3)
    assert "选择性不占优" in placebo_verdict({"swap_ratio": None}, 3)
    assert "可由 footprint 解释" in placebo_verdict({"swap_ratio": 2.0}, 0)
    assert "可由 footprint 解释" in placebo_verdict({"swap_ratio": 1.5}, -2)
    print("[selftest] analyze_verified_arms 11/11 PASS")


def main():
    ap = argparse.ArgumentParser(description="验证器臂判读（V1/V3，§5.8b 预注册）")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--betas", nargs="+", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None,
                    help=f"主判据档（默认 {BETA_PRIMARY_DEFAULT}，不在输入中则取最大 β）")
    ap.add_argument("--projection_json", type=str, default=None,
                    help="事后投影表 JSON（per-seed 档，如 posthoc_direction_1p7b/seed123/posthoc_direction.json）")
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        if not args.paths:
            return
    if not args.paths:
        raise SystemExit("需要至少一个输入 JSON（或 --selftest）")

    pooled, gates, metas = load_files(args.paths)
    if not pooled:
        raise SystemExit("未从输入文件中解析到任何 correct_<arm>_beta<β> 字段")
    arms = sorted({a for a, _ in pooled})
    verify_present = [a for a in arms if a in VERIFY_FAMILY]
    if not verify_present:
        raise SystemExit(f"输入中无验证器臂（可选 {VERIFY_FAMILY}）——本脚本只判 V1/V3")
    if "real" not in arms:
        raise SystemExit("输入中无 real 臂——配对判读不可用（须同批包含 real）")

    betas = sorted({b for _, b in pooled})
    if args.betas:
        betas = [b for b in betas if b in args.betas]
    if args.beta is not None:
        beta_main = args.beta if args.beta in betas else None
    else:
        beta_main = BETA_PRIMARY_DEFAULT if BETA_PRIMARY_DEFAULT in betas else max(betas)
    if beta_main is None:
        raise SystemExit(f"主判据档 β={args.beta} 不在输入中（可用：{betas}）")

    proj = None
    if args.projection_json:
        pj = Path(args.projection_json)
        if pj.exists():
            d = json.loads(pj.read_text())
            proj = {"path": str(pj), "n": d.get("n"), "theta": d.get("projection", {})}
        else:
            print(f"  ⚠️ 投影表不存在，跳过投影对照：{pj}")

    md = ["# 验证器臂判读（V1/V3，预注册 §5.8b）", ""]
    md.append(f"- 输入：{', '.join(m['file'] for m in metas)}")
    md.append(f"- 臂：{arms}；β：{betas}；主判据档 β={beta_main}")
    md.append(f"- real 基准样本数：{len(pooled.get(('real', beta_main), {}))}")
    md.append("")
    md.append("> 判停线**不在本报告**：V2（8B `geometry-sym-8b`）预注册 `post_beta` AUROC < 0.60 ⇒ "
              "验证器路线关闭；1.7B 属 H2 区，本报告只作**投影兑现检验**。")
    md.append("")

    # ── 各臂逐格 ──
    md += [f"## 各臂逐格（β={beta_main}）", "",
           "| 臂 | n | 救回（全体，CP95） | KW 救回 (CP95) | KC 破坏 (CP95) | 净事件 | 净 pp |",
           "|---|---|---|---|---|---|---|"]
    stats = {}
    for arm in arms:
        bucket = pooled.get((arm, beta_main))
        if not bucket:
            continue
        t = transition(list(bucket.values()))
        stats[arm] = t
        ps = t["per_subset"]
        kw, kc = ps.get(KW, {}), ps.get(KC, {})
        res_ci = clopper_pearson(t["cells"]["W_to_C"], t["cells"]["W_to_C"] + t["cells"]["W_to_W"])
        md.append(
            f"| `{arm}` | {t['n']} | {t['cells']['W_to_C']} ({fmt_ci(res_ci)}) "
            f"| {kw.get('rescue', 0)}/{kw.get('base_wrong', 0)} ({fmt_ci(kw.get('rescue_ci'))}) "
            f"| {kc.get('break', 0)}/{kc.get('base_right', 0)} ({fmt_ci(kc.get('break_ci'))}) "
            f"| {t['net_events']:+d} | {t['net_pp']:+.2f} |")
    md.append("")

    verdicts = []
    for arm in verify_present:
        key = (arm, beta_main)
        if key not in pooled:
            continue
        is_placebo = arm in VERIFY_RAND_ARMS
        lab = "安慰剂" if is_placebo else "验证器"
        r = judge_one(pooled, ("real", beta_main), key)
        t_r, t_v = stats["real"], stats[arm]
        dnet_events = t_v["net_events"] - t_r["net_events"]
        dnet_pp = t_v["net_pp"] - t_r["net_pp"]

        md += [f"## {'安慰剂' if is_placebo else '主判据'}（β={beta_main}）：`{arm}` vs `real`", "",
               f"| 量 | real | {lab} | Δ |", "|---|---|---|---|",
               f"| 净事件 | {t_r['net_events']:+d} | {t_v['net_events']:+d} | **{dnet_events:+d}** |",
               f"| 净 pp | {t_r['net_pp']:+.2f} | {t_v['net_pp']:+.2f} | **{dnet_pp:+.2f}** |",
               f"| KW 救回 | {t_r['per_subset'].get(KW, {}).get('rescue', 0)}"
               f"/{t_r['per_subset'].get(KW, {}).get('base_wrong', 0)}"
               f" | {t_v['per_subset'].get(KW, {}).get('rescue', 0)}"
               f"/{t_v['per_subset'].get(KW, {}).get('base_wrong', 0)} | — |",
               f"| KC 破坏 | {t_r['per_subset'].get(KC, {}).get('break', 0)}"
               f"/{t_r['per_subset'].get(KC, {}).get('base_right', 0)}"
               f" | {t_v['per_subset'].get(KC, {}).get('break', 0)}"
               f"/{t_v['per_subset'].get(KC, {}).get('base_right', 0)} | — |",
               ""]

        md += ["### 配对交换比（本路线真正的选择性度量）", "",
               f"- 避免的破坏（KC：仅 real 破坏）= **{r['avoided_breaks']}**"
               f"（n_kc={r['n_kc']}）；新增破坏（仅{lab}）= {r['new_breaks']}",
               f"- 丢失的救回（KW：仅 real 救回）= **{r['lost_rescues']}**"
               f"（n_kw={r['n_kw']}）；新增救回（仅{lab}）= {r['extra_rescues']}",
               f"- **交换比 = 避免破坏 : 丢失救回 = "
               f"{r['avoided_breaks']}:{r['lost_rescues']}"
               + (f" = **{r['swap_ratio']:.2f}:1**" if r["swap_ratio"] is not None else "（分母 0）")
               + f"** ⇒ {swap_verdict(r)}",
               f"- 配对检验：KC 破坏侧单侧 McNemar p = **{r['p_break_one_sided']:.4f}**"
               f"（H1＝{lab}破坏更少）；KW 救回侧单侧 p = {r['p_rescue_one_sided']:.4f}"
               f"（H1＝{lab}救回更少，属**预期代价**，非失败条件）",
               f"- 全样本配对：仅{lab}正确 {r['gain_all']} vs 仅 real 正确 {r['loss_all']}"
               f"（n={r['n_all']}）⇒ 双侧 McNemar p = **{r['p_all_two_sided']:.4f}**",
               ""]

        if proj and not is_placebo:
            th = proj["theta"].get("0.5", {})
            pnet = th.get("net")
            md += ["### 投影对照（事后档案，θ=0.5 预注册档）", ""]
            if pnet is None:
                md += ["- ⚠️ 投影表缺 θ=0.5 档，跳过对照", ""]
            else:
                # ⚠️ 口径：投影的 net 与实测的 net 都是**绝对净事件**（n 相同时方可直接相减）；
                #    不可拿"实测 Δ"去比"投影绝对值"（本脚本初版犯过此错）。
                md += [
                    f"- 投影（{Path(proj['path']).parent.name}，n={proj['n']['total']}，"
                    f"救回 {proj['n']['rescue']}/破坏 {proj['n']['break']}）："
                    f"θ=0 ⇒ {proj['theta'].get('0.0', {}).get('net')} 事件，"
                    f"**θ=0.5 ⇒ {pnet:+d} 事件**"
                    + ("" if proj["n"]["total"] == t_v["n"]
                       else f" ⚠️ 投影 n={proj['n']['total']} ≠ 实测 n={t_v['n']}，仅同向参考"),
                    f"- 实测：real {t_r['net_events']:+d} → 验证器 **{t_v['net_events']:+d}** 事件"
                    f"（Δ{dnet_events:+d}）",
                    f"- 兑现：{'✅ 净事件提升' if dnet_events > 0 else '❌ 净事件未提升'}"
                    f"；绝对净事件差异 {t_v['net_events'] - pnet:+d} 事件"
                    f"（实测 {t_v['net_events']:+d} vs 投影 {pnet:+d}）",
                    "",
                ]
        verdicts.append({"arm": arm, "beta": beta_main, "is_placebo": is_placebo, **r,
                         "net_real": t_r["net_events"], "net_verified": t_v["net_events"],
                         "dnet_events": dnet_events, "dnet_pp": dnet_pp})

    # ── 安慰剂对照：真验证器 vs footprint 匹配安慰剂（V1 的混淆排除）──
    placebo_verdicts = []
    placebo_arms = [a for a in verify_present if a in VERIFY_RAND_ARMS and (a, beta_main) in pooled]
    for pa in placebo_arms:
        for arm in [a for a in verify_present if a in VERIFY_ARMS and (a, beta_main) in pooled]:
            t_p, t_v = stats[pa], stats[arm]
            r = judge_one(pooled, (pa, beta_main), (arm, beta_main))
            dnet = t_v["net_events"] - t_p["net_events"]
            md += [f"## 安慰剂对照（β={beta_main}）：`{arm}` vs `{pa}`（footprint 匹配）", "",
                   "> 两臂共用同一翻转规则，只有保留/回退的决策依据不同（β*_min ≥θ vs Bernoulli(p)，"
                   "p 逐子集＝V1 实测保留率）⇒ 本块回答「V1 的净收益是否只是**扰动更少**带来的」。",
                   "",
                   "| 量 | 安慰剂 | 真验证器 | Δ |", "|---|---|---|---|",
                   f"| 净事件 | {t_p['net_events']:+d} | {t_v['net_events']:+d} | **{dnet:+d}** |",
                   f"| 净 pp | {t_p['net_pp']:+.2f} | {t_v['net_pp']:+.2f} | **{t_v['net_pp'] - t_p['net_pp']:+.2f}** |",
                   f"| KW 救回 | {t_p['per_subset'].get(KW, {}).get('rescue', 0)}"
                   f"/{t_p['per_subset'].get(KW, {}).get('base_wrong', 0)}"
                   f" | {t_v['per_subset'].get(KW, {}).get('rescue', 0)}"
                   f"/{t_v['per_subset'].get(KW, {}).get('base_wrong', 0)} | — |",
                   f"| KC 破坏 | {t_p['per_subset'].get(KC, {}).get('break', 0)}"
                   f"/{t_p['per_subset'].get(KC, {}).get('base_right', 0)}"
                   f" | {t_v['per_subset'].get(KC, {}).get('break', 0)}"
                   f"/{t_v['per_subset'].get(KC, {}).get('base_right', 0)} | — |",
                   "",
                   "### 配对（以安慰剂为基准）", "",
                   f"- 避免的破坏（仅安慰剂破坏）= **{r['avoided_breaks']}**（n_kc={r['n_kc']}）；"
                   f"新增破坏（仅真验证器）= {r['new_breaks']}",
                   f"- 丢失的救回（仅安慰剂救回）= **{r['lost_rescues']}**（n_kw={r['n_kw']}）；"
                   f"新增救回（仅真验证器）= {r['extra_rescues']}",
                   f"- **交换比 = {r['avoided_breaks']}:{r['lost_rescues']}"
                   + (f" = **{r['swap_ratio']:.2f}:1**" if r["swap_ratio"] is not None else "（分母 0）")
                   + f"** ⇒ {swap_verdict(r)}",
                   f"- 配对检验：KC 破坏侧单侧 McNemar p = **{r['p_break_one_sided']:.4f}**；"
                   f"KW 救回侧单侧 p = {r['p_rescue_one_sided']:.4f}；"
                   f"全样本双侧 McNemar p = **{r['p_all_two_sided']:.4f}**"
                   f"（仅真验证器正确 {r['gain_all']} vs 仅安慰剂正确 {r['loss_all']}）",
                   f"- **判定（规则先注册）**：Δnet = {dnet:+d} 事件 ⇒ {placebo_verdict(r, dnet)}",
                   ""]
            placebo_verdicts.append({"arm": arm, "placebo_arm": pa, "beta": beta_main, **r,
                                     "net_placebo": t_p["net_events"],
                                     "net_verified": t_v["net_events"],
                                     "dnet_events": dnet,
                                     "verdict": placebo_verdict(r, dnet)})
    if verify_present and not placebo_arms:
        md += [f"## 安慰剂对照：未跑（输入无 `{PLACEBO_ARM_DEFAULT}`）", "",
               f"> footprint 混淆未排除 ⇒ V1 净收益**尚不能**归因于 β*_min 选择信息。"
               f"补跑：`bash scripts/review_local.sh verified-rand`（同批含 real/verified_sym/"
               f"{PLACEBO_ARM_DEFAULT}）。",
               ""]

    # ── 验证器诊断 ──
    md += ["## 验证器诊断（按子集）", "",
           "| 臂 | 子集 | n | 翻转率/步 | 保留率/翻转 | 保留 β* 均值 | 回退 β* 均值 | 回退数 | θ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for arm in verify_present:
        agg = verify_diagnostics(gates, pooled, arm, beta_main)
        if not agg:
            md.append(f"| `{arm}` | — | — | — | — | — | — | — | ⚠️ 无 gate_stats（旧档案？） |")
            continue
        for sub in (KW, KC, DK):
            a = agg.get(sub)
            if not a:
                continue
            mk = f"{a['mean_kept_beta']:.3f}" if a["mean_kept_beta"] is not None else "—"
            mr = f"{a['mean_reverted_beta']:.3f}" if a["mean_reverted_beta"] is not None else "—"
            md.append(f"| `{arm}` | {sub} | {a['n']} | {fmt_pct(a['flip_rate_of_steps'])} "
                      f"| {fmt_pct(a['keep_rate_of_flips'])} | {mk} | {mr} "
                      f"| {a['reverted']} | {a['theta']} |")
    md.append("")

    md += ["## 并列 caveat（写进论文前必读）", "",
           "1. 投影假设「回退即回到基线结果」——档案是多步干预，真实验证臂才是**实测**（本报告即实测）。",
           "2. 事件数小 ⇒ 交换比不稳（§9d 教训）⇒ 必须与绝对事件数并列报，不得单以比值立论。",
           "3. 1.7B 处 H2 区（救回＝通用扰动）⇒ 本报告结论**不得外推到 8B**；判停看 V2。",
           "4. θ=0.5 为预注册主档；改 θ 即破预注册，须在报告中显式声明。",
           "5. **footprint 对照**：`verified_rand` 的 p 由 V1 实测保留率标定（只看决策计数、不看结果），"
           "两臂的**期望扰动步数**相同；但轨迹在首次决策分歧后分叉 ⇒ 翻转集只在分叉前相同，"
           "且 Bernoulli 抽样使实际步数有 ~±2–4% 波动 ⇒ 判读须同时看两臂实测 footprint（诊断表）。",
           ""]

    report_md = "\n".join(x for x in md if x is not None)
    out = Path(args.output_dir) if args.output_dir else (
        Path(args.paths[0]).parent / "judge")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verified_arms_report.md").write_text(report_md, encoding="utf-8")
    (out / "verified_arms_report.json").write_text(json.dumps(
        {"inputs": [m["file"] for m in metas], "arms": arms, "beta_main": beta_main,
         "betas": betas, "verdicts": verdicts, "placebo_verdicts": placebo_verdicts},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(report_md)
    print(f"\n[saved] {out / 'verified_arms_report.md'}")


if __name__ == "__main__":
    main()
