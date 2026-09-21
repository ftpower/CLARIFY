"""T2 分叉可预测性分析（零 GPU）——TLDC 改进的门控可行性前置。

═══════════════════════════════════════════════════════════════════════════
动机（2026-09-21 晚，TLDC 改进主线第一件）
═══════════════════════════════════════════════════════════════════════════
今日五透镜判停结论：1.7B 上 TLDC 救回＝通用扰动；首分叉翻转 100% 由**压支**驱动（抬支 0/72）；
命题 5 轨迹级上界被违反（观测救回 7.0% > 基线轨迹 P(h=1) 4.2%）。
⇒ 改进方向不再是"怎么把正确 token 抬起来"，而是**"分叉在什么条件下发生、能否事前识别"**。
本脚本回答：**在分叉发生的那一步（基线前缀，与 TLDC 轨迹同前缀），末层/参考层的几何量能否
把它与普通步区分开？** 等价于 P5.3 的几何版、D21 的失败模式检查。

可检验预测（预注册，禁止事后更改）：
  T2-A：分叉步的末层 margin（l_L 前二差）显著低于非分叉步 ⇒ 门控有信号。
  T2-B：判别力 AUC ≥ 0.70 视为"可用"；0.60–0.70 边缘；≤0.60 视为无信号（同 D21 AUROC_fail≈0.5 失败模式）。
  T2-C：窄 margin 门（margin<τ）的**召回/负担曲线**：若某 τ 能覆盖 ≥60% 分叉步且只影响 ≤30% 步，
        则候选①"窄 margin 触发 + 单步回验证"值得实现；否则门控类改进直接排除。

输入（均已落盘，零 GPU）：
  --geo  geometry archive（baseline 轨迹，每步 final_top10/early_top10/geo）
  --tok  per-token archive（TLDC 轨迹 gids vs baseline gids_bl → 首分叉步）
用法：
  python experiments/lin_theory/analyze_fork_predictors.py \
    --geo experiments/outputs/geometry_archive/geo_baseline_seed123.json \
    --tok experiments/outputs/lin_theory/s15_2b_tldc_per_token.json
"""

import argparse
import json
from pathlib import Path


def _logit(entry):
    return float(entry[2])


def step_features(step):
    """从 geometry archive 的单步记录抽取可分叉性特征（全来自基线前缀，零额外前向）。"""
    f10 = step["final_top10"]
    e10 = step["early_top10"]
    geo = step["geo"]
    margin_L = _logit(f10[0]) - _logit(f10[1])
    margin_E = _logit(e10[0]) - _logit(e10[1])
    spread_L = _logit(f10[0]) - _logit(f10[-1])
    # 压支幅度：末层 argmax 在参考层的相对落后（δ = l_ℓ* − l_L 的负部）
    am_id = f10[0][0]
    e_of_am = next((_logit(e) for e in e10 if e[0] == am_id), None)
    damp_am = (_logit(f10[0]) - e_of_am) if e_of_am is not None else None
    return {
        "margin_L": margin_L,
        "margin_E": margin_E,
        "spread_L": spread_L,
        "damp_am": damp_am,
        "R_size": geo["R_size"],
        "beta_star_min": geo["beta_star_min"],
        "yt_in_R": int(bool(geo["yt"]["in_R"])),
        "yt_delta": geo["yt"]["delta"],
    }


def auc_mann_whitney(pos, neg):
    """秩和 AUC（pos=分叉步，neg=非分叉步）；含并列平均秩。"""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    allv = sorted(pos + neg)
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks[allv[i]] = avg
        i = j + 1
    r_pos = sum(ranks[v] for v in pos)
    n1, n2 = len(pos), len(neg)
    u = r_pos - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


def main():
    ap = argparse.ArgumentParser(description="T2 分叉可预测性分析（零 GPU）")
    ap.add_argument("--geo", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--output_dir", default="experiments/outputs/tldc_fork_analysis_20260921")
    args = ap.parse_args()

    geo_doc = json.loads(Path(args.geo).read_text())
    geo_by_q = {}
    for s in geo_doc["samples"]:
        key = s["question"][:60] if isinstance(s["question"], str) else s["question"]
        geo_by_q[key] = {st["step"]: st for st in s["steps"]}

    tok = json.loads(Path(args.tok).read_text())
    fork_feats, other_feats = [], []
    n_fork = n_nofork = n_mismatch = 0
    for s in tok:
        q = s["question"][:60]
        steps = geo_by_q.get(q)
        if steps is None:
            n_mismatch += 1
            continue
        gids, gids_bl = s["gids"], s["gids_bl"]
        t = next((i for i in range(min(len(gids), len(gids_bl))) if gids[i] != gids_bl[i]), None)
        if t is None:
            n_nofork += 1
        else:
            n_fork += 1
        for st, rec in steps.items():
            f = step_features(rec)
            f["step"] = st
            (fork_feats if st == t else other_feats).append(f)

    feats = ["margin_L", "margin_E", "spread_L", "damp_am", "R_size", "beta_star_min", "yt_delta"]
    md = ["# T2 分叉可预测性分析（零 GPU）", "",
          f"- 输入：`{Path(args.geo).name}` × `{Path(args.tok).name}`；对齐失败 {n_mismatch} 例",
          f"- 有分叉样本 {n_fork}，无分叉样本 {n_nofork}；分叉步 {len(fork_feats)} vs 非分叉步 {len(other_feats)}",
          "", "## 判别力（秩和 AUC：分叉步 vs 非分叉步）", "",
          "> AUC 是「特征值大 ⇒ 更可能是分叉步」的判别力；**AUC<0.5 表示方向相反**"
          "（该特征取小值才是分叉），此时有效判别力 = 1−AUC。",
          "", "| 特征 | 分叉步均值 | 非分叉步均值 | AUC | 有效判别力 | 判读 |",
          "|---|---|---|---|---|---|"]
    results = {}
    for f in feats:
        p = [r[f] for r in fork_feats]
        n = [r[f] for r in other_feats]
        a = auc_mann_whitney(p, n)
        mp = sum(x for x in p if x is not None) / max(1, sum(1 for x in p if x is not None))
        mn = sum(x for x in n if x is not None) / max(1, sum(1 for x in n if x is not None))
        eff = None if a is None else max(a, 1 - a)
        direction = "低值⇒分叉" if (a is not None and a < 0.5) else "高值⇒分叉"
        tag = "—" if eff is None else ("**可用**" if eff >= 0.70 else ("边缘" if eff >= 0.60 else "无信号"))
        results[f] = {"auc": a, "effective_auc": eff, "direction": direction,
                      "mean_fork": mp, "mean_other": mn}
        md.append(f"| {f} | {mp:.3f} | {mn:.3f} | {a:.3f} | {eff:.3f} | {tag}（{direction}） |")

    # 门控召回/负担曲线（候选①）
    md += ["", "## 窄 margin 门（margin_L < τ）的召回 / 负担", "",
           "| τ | 覆盖分叉步 | 影响步占比 | 收益比 |", "|---|---|---|---|"]
    curve = []
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
        rf = sum(1 for r in fork_feats if r["margin_L"] is not None and r["margin_L"] < tau)
        ra = sum(1 for r in other_feats if r["margin_L"] is not None and r["margin_L"] < tau)
        recall = rf / max(1, len(fork_feats))
        burden = (rf + ra) / max(1, len(fork_feats) + len(other_feats))
        ratio = recall / burden if burden else None
        curve.append({"tau": tau, "recall": recall, "burden": burden, "lift": ratio})
        md.append(f"| {tau} | {recall*100:.1f}% | {burden*100:.1f}% | "
                  f"{'—' if ratio is None else f'{ratio:.2f}×'} |")

    best = max((c for c in curve if c["lift"]), key=lambda c: c["lift"], default=None)
    md += ["", f"**最佳 τ（按收益比）**：{best['tau'] if best else '—'}"
              f"（召回 {best['recall']*100:.1f}%、负担 {best['burden']*100:.1f}%、"
              f"收益比 {best['lift']:.2f}×）" if best else "", ""]
    # 预注册判读（T2-A/B：按有效判别力；T2-C：在所有 τ 中找满足"召回≥60% 且 负担≤30%"者）
    eff_margin = results["margin_L"]["effective_auc"]
    eff_beta = results["beta_star_min"]["effective_auc"]
    ok_signal = max(x for x in (eff_margin, eff_beta) if x is not None) >= 0.70
    qualifiers = [c for c in curve if c["recall"] >= 0.60 and c["burden"] <= 0.30]
    ok_gate = len(qualifiers) > 0
    md += [f"**T2-A/B 判定**：margin_L 有效判别力={eff_margin:.3f}（{results['margin_L']['direction']}）、"
           f"beta_star_min={eff_beta:.3f} ⇒ " +
           ("**门控有信号（AUC≥0.70）**" if ok_signal else "信号不足（同 D21 失败模式风险）"),
           f"**T2-C 判定**：满足「召回≥60% 且 负担≤30%」的 τ = "
           + (", ".join(f"{c['tau']}" for c in qualifiers) if qualifiers else "无")
           + (" ⇒ **门控类改进值得实现**（候选①）" if ok_gate else " ⇒ **门控类改进应排除/降级**"),
           ""]

    # ── 门控的净效应预估：低 margin 步在各子集的分布（决定门控能否改善 net）──
    md += ["## 低 margin 步的子集分布（门控净效应预估）", "",
           "> 若 KW 样本的低 margin 步数 ≫ KC 样本，则「只在低 margin 步干预」会**多留救回机会、少造成破坏** ⇒ 净效应改善；",
           "> 若两组相当，则门控只是等比例缩小救回与破坏，**净效应不变**。", "",
           "| 子集 | n | 平均低 margin 步数(τ=0.2) | 至少 1 步低 margin 的样本占比 |",
           "|---|---|---|---|"]
    from collections import defaultdict
    sub_stat = defaultdict(lambda: {"n": 0, "steps": 0, "has": 0})
    for s in geo_doc["samples"]:
        sub = s["subset"]
        cnt = sum(1 for st in s["steps"]
                  if (lambda m: m is not None and m < 0.2)(step_features(st)["margin_L"]))
        sub_stat[sub]["n"] += 1
        sub_stat[sub]["steps"] += cnt
        sub_stat[sub]["has"] += int(cnt > 0)
    gate_stat = {}
    for sub, d in sorted(sub_stat.items()):
        avg = d["steps"] / max(1, d["n"])
        frac = d["has"] / max(1, d["n"])
        gate_stat[sub] = {"n": d["n"], "avg_low_margin_steps": avg, "frac_samples_with_low": frac}
        md.append(f"| {sub} | {d['n']} | {avg:.2f} | {frac*100:.1f}% |")
    kw_avg = gate_stat.get("know_wrong", {}).get("avg_low_margin_steps")
    kc_avg = gate_stat.get("know_correct", {}).get("avg_low_margin_steps")
    md += ["", f"**净效应预估**：KW 平均 {kw_avg:.2f} 步 vs KC 平均 {kc_avg:.2f} 步 ⇒ "
           + ("**KW 明显更多 ⇒ 门控有净增益空间**" if (kw_avg or 0) > 1.3 * (kc_avg or 1e-9)
              else "**两组相当 ⇒ 门控预计不改变净效应（只缩小规模）**"), ""]

    print("\n".join(md))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fork_predictors.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "fork_predictors.json").write_text(
        json.dumps({"features": results, "gate_curve": curve,
                    "n_fork_steps": len(fork_feats), "n_other_steps": len(other_feats),
                    "n_fork_samples": n_fork, "n_nofork_samples": n_nofork},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'fork_predictors.md'}")


if __name__ == "__main__":
    main()
