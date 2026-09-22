"""TLDC 可选择性审计（零 GPU）——回答"只做 TLDC 改进还有什么路"。

输入（均已落盘）：
  · 几何档案（干预前，基线轨迹）：experiments/outputs/geometry_archive_8b/geo_baseline_seed123.json
  · TLDC real 臂（干预后，β=0.20）：experiments/outputs/tldc_gated_8b/tldc_controls_123_*_tau0.2.json
  两者以 sample_id join（同 seed/同分类口径，交集 300/300）。

三项检验：
  A. **β 外推**：各 β 下 step0 的 argmax 是否命中正确 token（过冲＝β 增大反而丢失正确起点）；
     KC 样本"至少一步被改动"的比例（破坏侧附带上界，随 β 的增长率）。
  B. **R-受限扰动上界**：抬支只能作用于 R={c: l_ℓ*(c) > l_ℓ*(a)}（引理 1），
     故 step0 救回的硬前提是 yt∈R ⇒ 给出任何 R-受限算子的救回上界。
  C. **可选择性判决（核心）**：用干预前可观测的 step0 几何量，分别判别
     "被救回"（KW 内）与"被破坏"（KC 内）。若同一特征对两者的 AUROC 近似相等
     ⇒ 不存在能改善"救回/破坏"交换比的阈值选择器 ⇒ 调度/选择类改进空间关闭。
     参照 D21（Knowing-Saying Gap 卡）实测 AUROC_fail≈0.5 的失败模式。

判读纪律：AUROC 用 Mann-Whitney（含并列 0.5 计分），无 scipy 依赖；数值只作描述，
判停阈值不事后更改（本脚本只做测量，不设判据）。

用法：
  python experiments/lin_theory/analyze_tldc_selectivity.py \
    --geo experiments/outputs/geometry_archive_8b/geo_baseline_seed123.json \
    --tldc experiments/outputs/tldc_gated_8b/tldc_controls_123_real-gated_margin-gated_betastar-gated_damp_tau0.2.json
"""

import argparse
import json
import statistics as st
from pathlib import Path

BETAS = ["0.00", "0.01", "0.02", "0.03", "0.05", "0.08", "0.10", "0.15", "0.20", "0.30", "0.50"]


def step0_top2_margin(step):
    """末层前二 margin（final_top10 已按降序存 [id, text, logit]）。"""
    f = step["final_top10"]
    return f[0][2] - f[1][2]


def auroc(pos, neg, fn):
    a = [x for x in (fn(s) for s in pos) if x is not None]
    b = [x for x in (fn(s) for s in neg) if x is not None]
    if not a or not b:
        return float("nan")
    w = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return w / (len(a) * len(b))


def med(sids, fn):
    v = [x for x in (fn(s) for s in sids) if x is not None]
    return st.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description="TLDC 可选择性审计（零 GPU）")
    ap.add_argument("--geo", required=True)
    ap.add_argument("--tldc", required=True, help="real 臂所在的结果 JSON（β=0.20）")
    ap.add_argument("--beta", type=str, default="0.20", help="TLDC real 臂所用 β（用于 C 段落标签）")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    geo = json.loads(Path(args.geo).read_text())
    G = {s["sample_id"]: s for s in geo["samples"]}
    tld = json.loads(Path(args.tldc).read_text())
    arm_key = f"correct_real_beta{float(args.beta):g}"
    T = {int(k): v for k, v in tld["samples"].items() if arm_key in v}
    shared = sorted(set(G) & set(T))
    assert shared, f"两档案 sample_id 无交集（检查 arm 键 {arm_key}）"

    md = ["# TLDC 可选择性审计（零 GPU）", "",
          f"- 几何档案：`{Path(args.geo).name}`；干预后档案：`{Path(args.tldc).name}`（{arm_key}）",
          f"- join 交集：{len(shared)}/{len(G)} 样本", ""]

    kw = [s for s in shared if T[s]["subset"] == "know_wrong" and not T[s]["baseline_correct"]]
    kc = [s for s in shared if T[s]["subset"] == "know_correct" and T[s]["baseline_correct"]]
    rescue = [s for s in kw if T[s][arm_key]]
    brk = [s for s in kc if not T[s][arm_key]]

    # ── A. β 外推 ──
    md += ["## A. β 外推检验（step0 命中正确 token / 破坏侧扰动面）", "",
           "| β | step0 argmax==yt（sym） | step0 argmax==yt（damp） | KC 样本≥1 步被改动 |",
           "|---|---|---|---|"]
    for b in BETAS:
        hit = sum(1 for s in kw if G[s]["steps"][0]["grid"]["sym"][b] == G[s]["y_true_id"])
        hit_d = sum(1 for s in kw if G[s]["steps"][0]["grid"]["damp"][b] == G[s]["y_true_id"])
        kc_chg = sum(1 for s in kc if any(st_["grid"]["sym"][b] != st_["chosen_id"] for st_ in G[s]["steps"]))
        md.append(f"| {float(b):.2f} | {hit}/{len(kw)} | {hit_d}/{len(kw)} | "
                  f"{kc_chg}/{len(kc)} = {kc_chg/len(kc)*100:.1f}% |")
    base_hit = sum(1 for s in kw if G[s]["steps"][0]["chosen_id"] == G[s]["y_true_id"])
    md += ["", f"> 基线（β=0）step0 命中数 = {base_hit}/{len(kw)}；**若某 β 的命中数低于该值 ⇒ 过冲**"
              "（干预反而摧毁了原本正确的起点）。", ""]

    # ── B. R-受限上界 ──
    inR = sum(1 for s in kw if G[s]["steps"][0]["geo"]["yt"]["in_R"])
    need = [s for s in kw if G[s]["steps"][0]["chosen_id"] != G[s]["y_true_id"]]
    inR_need = sum(1 for s in need if G[s]["steps"][0]["geo"]["yt"]["in_R"])
    md += ["## B. R-受限扰动上界（引理 1：抬支只能作用 R 内 token）", "",
           f"- KW 样本 step0 `yt∈R`：**{inR}/{len(kw)} = {inR/len(kw)*100:.1f}%**（任何 R-受限算子的 step0 救回上界）",
           f"- 其中「step0 尚不正确、需被救」的子集：`yt∈R` {inR_need}/{len(need)} = "
           f"{inR_need/len(need)*100:.1f}%",
           "- ⚠️ 注意：**破坏候选同样住在 R 内**（要越过 a 必须 l_ℓ*(c) > l_ℓ*(a)）",
           "  ⇒ R-受限只剔除「惰性质量」，不改变救回/破坏候选的共存结构。", ""]

    # ── C. 可选择性判决 ──
    kw_no = [s for s in kw if s not in rescue]
    kc_no = [s for s in kc if s not in brk]
    feats = {
        "top2_margin": lambda s: step0_top2_margin(G[s]["steps"][0]),
        "m_min": lambda s: G[s]["steps"][0]["geo"]["m_min"],
        "beta_star_min": lambda s: G[s]["steps"][0]["geo"]["beta_star_min"],
        "R_size": lambda s: G[s]["steps"][0]["geo"]["R_size"],
    }
    md += ["## C. 可选择性判决（核心）", "",
           f"KW {len(kw)}（救回 {len(rescue)}）｜KC {len(kc)}（破坏 {len(brk)}）；特征均取自**干预前**的 step0", "",
           "| step0 特征 | 中位（被救回） | 中位（KW 未救回） | 中位（被破坏） | 中位（KC 未破坏） "
           "| AUROC 救回 | AUROC 破坏 | 差距 |", "|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for name, fn in feats.items():
        a1, a2 = auroc(rescue, kw_no, fn), auroc(brk, kc_no, fn)
        gap = abs(a1 - a2) if a1 == a1 and a2 == a2 else float("nan")
        verdict[name] = {"auroc_rescue": a1, "auroc_break": a2, "gap": gap}
        md.append(f"| `{name}` | {med(rescue,fn):.3f} | {med(kw_no,fn):.3f} | {med(brk,fn):.3f} "
                  f"| {med(kc_no,fn):.3f} | {a1:.3f} | {a2:.3f} | {gap:.3f} |")
    md += ["",
           "> **判读口径**：要改善「救回/破坏」交换比，需要某特征对救回的判别力**明显高于**对破坏的判别力",
           "> （或反向使用）。两者 AUROC 近似相等（差距 ≈0）⇒ 任何该特征上的阈值选择器都同等放大两者，",
           "> **不可能提高交换比**；若某特征对破坏的判别力更强，用它做门控只会更糟（优先砍掉救回）。",
           "> 参照 D21（Knowing-Saying Gap）实测 `AUROC_fail≈0.5` 的失败模式。", ""]

    if verdict.get("R_size", {}).get("gap") is not None:
        g = verdict["R_size"]["gap"]
        md += [f"**本档结论**：R_size 的救回/破坏 AUROC = "
               f"{verdict['R_size']['auroc_rescue']:.3f} / {verdict['R_size']['auroc_break']:.3f}"
               f"（差距 {g:.3f}）⇒ " +
               ("**无可选择性**：不存在靠该几何量改善交换比的调度/门控类改进。"
                if g < 0.10 else "存在可选择性空间，值得进一步设计选择器。"), ""]

    text = "\n".join(md)
    print(text)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tldc_selectivity.md").write_text(text, encoding="utf-8")
        (out / "tldc_selectivity.json").write_text(
            json.dumps({"config": {"geo": args.geo, "tldc": args.tldc, "beta": args.beta},
                        "n": {"kw": len(kw), "kc": len(kc), "rescue": len(rescue), "break": len(brk)},
                        "auroc": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[saved] {out / 'tldc_selectivity.md'}")


if __name__ == "__main__":
    main()
