"""逐格转移矩阵分析（T15② 口径）——零 GPU，只读既有档案。

方法学锚点：
  - T15「Mirage of Performance Gains」（NeurIPS 2025，已评审）卡片「可做」②：
    「Acc+双口径与 FN→TP/TN→FP 逐格转移矩阵」；「不可做」：不得只在单一口径报告
    干预涨点而不排除"单向分布偏移"与"等效解码退化"两种伪提升。
    → 论文/论文结论卡.md [T15]；出处 论文/论文补充/新论文补充-2026-09/理论/Mirage_of_Performance_Gains/
  - 判读口径：docs/protocol/review-runbook-20260921.md §5 第 7 条
    （救回/破坏必须带 DK 参考组 + Fisher 检验；KW 结构上不可能破坏、KC 结构上不可能救回）。
  - 对照协议参照：D01 五重因果对照（matched-random/label-shuffle/打乱门控/oracle 门控）、
    T03 随机方向 push、I02 oracle vs blind —— 本脚本只做"逐格"这一层，不含对照臂。

输入（全部为已落盘档案，不重跑模型）：
  - ROME 1.7B：experiments/outputs/phase16_rome_review/phase16_rome_seed{123,456}[_samples].json
    （λ 由主档 val.selection.key 自动解析，禁止硬编码）
  - TLDC 8B：experiments/outputs/lin_theory_8b/seed{123,456}_8b/s14_tldc_samples.json
  - TLDC 1.7B：experiments/outputs/lin_theory/s14_tldc_samples.json（仅 seed123 → 标注探索性）

输出：experiments/outputs/transition_matrix_20260921/transition_matrix.{md,json}

用法：
    python experiments/lin_theory/analyze_transition_matrix.py
    python experiments/lin_theory/analyze_transition_matrix.py --output_dir <dir>
"""

import argparse
import json
import math
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# 纯 Python 精确统计（不依赖 scipy，保证任何环境可复跑）
# ═════════════════════════════════════════════════════════════════════════════


def _binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact(b, c):
    """McNemar 精确检验（二项，双侧）——不一致对 b vs c。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(_binom_pmf(i, n) for i in range(0, k + 1))
    return min(1.0, 2 * tail)


def fisher_greater(a, b, c, d):
    """Fisher 精确检验（单侧 greater）：表 [[a,b],[c,d]]，a/c 为事件数，
    b/d 为非事件数。P = Σ_{x≥a} 超几何概率。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if r1 == 0 or c1 == 0 or n == 0:
        return 1.0
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    denom = math.comb(n, c1)

    def hyper(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / denom

    return min(1.0, sum(hyper(x) for x in range(a, hi + 1)))


def clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson 精确 CI（纯 Python）。

    P(X ≥ j | p) 关于 p 单调递增，故用单调二分求根：
      p_L 解 P(X ≥ k)   = α/2      （k>0）
      p_U 解 P(X ≥ k+1) = 1 − α/2  （k<n）
    边界 k=0 / k=n 用闭式：上界 1−(α/2)^(1/n)、下界 (α/2)^(1/n)。
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        return (0.0, 1 - (alpha / 2) ** (1 / n))
    if k == n:
        return ((alpha / 2) ** (1 / n), 1.0)

    def ge(j, p):
        """P(X ≥ j | p)，n 次伯努利。"""
        return sum(_binom_pmf(i, n, p) for i in range(j, n + 1))

    def solve(target, j):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if ge(j, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return (solve(alpha / 2, k), solve(1 - alpha / 2, k + 1))


# ═════════════════════════════════════════════════════════════════════════════
# 转移矩阵核心
# ═════════════════════════════════════════════════════════════════════════════


def transition(rows):
    """rows: [(subset, baseline_correct, final_correct)] → 逐格计数与比率。

    四格（T15② 口径）：W→C 救回 / C→W 破坏 / W→W 未救回 / C→C 保持。
    注意 KW 组结构上不可能出现 C→W、KC 组结构上不可能出现 W→C（标签定义使然）。
    """
    cells = {"W_to_C": 0, "C_to_W": 0, "W_to_W": 0, "C_to_C": 0}
    per_sub = {}
    for sub, base, final in rows:
        key = ("W_to_C" if (not base and final) else
               "C_to_W" if (base and not final) else
               "W_to_W" if not base else "C_to_C")
        cells[key] += 1
        s = per_sub.setdefault(
            sub, {"n": 0, "base_wrong": 0, "base_right": 0, "rescue": 0, "break": 0}
        )
        s["n"] += 1
        s["base_wrong" if not base else "base_right"] += 1
        if key == "W_to_C":
            s["rescue"] += 1
        elif key == "C_to_W":
            s["break"] += 1

    n = len(rows)
    rescue, brk = cells["W_to_C"], cells["C_to_W"]
    for s in per_sub.values():
        s["rescue_rate"] = s["rescue"] / s["base_wrong"] if s["base_wrong"] else None
        s["break_rate"] = s["break"] / s["base_right"] if s["base_right"] else None
        if s["base_wrong"]:
            s["rescue_ci"] = clopper_pearson(s["rescue"], s["base_wrong"])
        if s["base_right"]:
            s["break_ci"] = clopper_pearson(s["break"], s["base_right"])

    base_acc = sum(1 for _, b, _ in rows if b) / n if n else 0.0
    final_acc = sum(1 for _, _, f in rows if f) / n if n else 0.0
    # 事件归属占比：救回事件中属于 KW 的比例、破坏事件中属于 KC 的比例
    # —— 直接量化「分类法能解释多少」，余下部分落在 DK 参考组（通用扰动嫌疑）。
    kw_res = per_sub.get("know_wrong", {}).get("rescue", 0)
    kc_brk = per_sub.get("know_correct", {}).get("break", 0)
    return {
        "n": n,
        "cells": cells,
        "net_events": rescue - brk,
        "net_pp": (rescue - brk) / n * 100 if n else 0.0,
        "baseline_acc": base_acc,
        "intervened_acc": final_acc,
        "mcnemar_p": mcnemar_exact(rescue, brk),
        "rescue_kw_share": (kw_res / rescue) if rescue else None,
        "break_kc_share": (kc_brk / brk) if brk else None,
        "per_subset": per_sub,
    }


def kw_vs_dk(t):
    """救回：KW vs DK（Fisher greater）；破坏：KC vs DK（Fisher greater）。"""
    ps = t["per_subset"]
    out = {}
    if "know_wrong" in ps and "dont_know" in ps:
        kw, dk = ps["know_wrong"], ps["dont_know"]
        out["rescue_kw_vs_dk"] = {
            "kw": f"{kw['rescue']}/{kw['base_wrong']}",
            "dk": f"{dk['rescue']}/{dk['base_wrong']}",
            "ratio": (kw["rescue_rate"] / dk["rescue_rate"]) if dk["rescue_rate"] else None,
            "p": fisher_greater(
                kw["rescue"], kw["base_wrong"] - kw["rescue"],
                dk["rescue"], dk["base_wrong"] - dk["rescue"],
            ),
        }
    if "know_correct" in ps and "dont_know" in ps:
        kc, dk = ps["know_correct"], ps["dont_know"]
        out["break_kc_vs_dk"] = {
            "kc": f"{kc['break']}/{kc['base_right']}",
            "dk": f"{dk['break']}/{dk['base_right']}",
            "ratio": (kc["break_rate"] / dk["break_rate"]) if dk["break_rate"] else None,
            "p": fisher_greater(
                kc["break"], kc["base_right"] - kc["break"],
                dk["break"], dk["base_right"] - dk["break"],
            ),
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 档案装载
# ═════════════════════════════════════════════════════════════════════════════


def load_rome(root, seed):
    """→ (rows, meta)。λ 从主档 val.selection.key 解析，禁止硬编码。"""
    main = json.loads((root / f"phase16_rome_seed{seed}.json").read_text())
    key = main["val"]["selection"]["key"]
    samples = json.loads((root / f"phase16_rome_seed{seed}_samples.json").read_text())["samples"]
    rows = []
    for v in samples.values():
        if v.get("baseline_correct") is None or v.get(key) is None:
            continue
        rows.append((v["subset"], bool(v["baseline_correct"]), bool(v[key])))
    return rows, {"key": key, "selection": main["val"]["selection"]}


def load_tldc(path):
    """→ {β: rows}；β 从样本键 correct_beta* 解析（保留原始键字符串，避免
    0.10→'0.1' / 0.20→'0.2' 的浮点重格式化错位）。"""
    d = json.loads(Path(path).read_text())
    samples = d["samples"]
    first = next(iter(samples.values()))
    raw_keys = {
        float(k.replace("correct_beta", "")): k
        for k in first
        if k.startswith("correct_beta")
    }
    out = {}
    for b in sorted(raw_keys):
        key = raw_keys[b]
        rows = [
            (v["subset"], bool(v["baseline_correct"]), bool(v[key]))
            for v in samples.values()
            if v.get("baseline_correct") is not None and v.get(key) is not None
        ]
        out[b] = rows
    return out, d.get("config", {})


def pool(*row_lists):
    return [r for rl in row_lists for r in rl]


# ═════════════════════════════════════════════════════════════════════════════
# 报告
# ═════════════════════════════════════════════════════════════════════════════


def fmt_pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def fmt_ci(ci):
    return "—" if ci is None else f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"


def arm_md(title, t, extra=None):
    c = t["cells"]
    lines = [
        f"### {title}",
        "",
        f"- 样本 n={t['n']}；基线准确率 {t['baseline_acc'] * 100:.1f}% → 干预后 {t['intervened_acc'] * 100:.1f}%"
        f"（净 **{t['net_pp']:+.1f}pp**，McNemar 精确 p={t['mcnemar_p']:.4f}）",
        f"- 事件归属：救回 {c['W_to_C']} 例中 KW 占 "
        f"{fmt_pct(t['rescue_kw_share'])}；破坏 {c['C_to_W']} 例中 KC 占 {fmt_pct(t['break_kc_share'])}"
        f"（余下落在 DK 参考组＝通用扰动嫌疑）",
        "",
        "| 逐格转移 | 计数 | 口径 |",
        "|---|---|---|",
        f"| W→C（救回） | **{c['W_to_C']}** | 分母＝基线错 |",
        f"| C→W（破坏） | **{c['C_to_W']}** | 分母＝基线对 |",
        f"| W→W（未救回） | {c['W_to_W']} | — |",
        f"| C→C（保持） | {c['C_to_C']} | — |",
        "",
        "| 子集 | n | 救回/分母 | 救回率 (CP95) | 破坏/分母 | 破坏率 (CP95) |",
        "|---|---|---|---|---|---|",
    ]
    for sub in ("know_wrong", "know_correct", "dont_know"):
        if sub not in t["per_subset"]:
            continue
        s = t["per_subset"][sub]
        lines.append(
            f"| {sub} | {s['n']} | {s['rescue']}/{s['base_wrong']} | "
            f"{fmt_pct(s['rescue_rate'])} {fmt_ci(s.get('rescue_ci'))} | "
            f"{s['break']}/{s['base_right']} | {fmt_pct(s['break_rate'])} {fmt_ci(s.get('break_ci'))} |"
        )
    if extra:
        lines += ["", extra]
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="逐格转移矩阵（T15② 口径，零 GPU）")
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="experiments/outputs/transition_matrix_20260921",
    )
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    md = [
        "# 逐格转移矩阵报告（T15② 口径）— 2026-09-21",
        "",
        "> 数据源＝已落盘档案（零 GPU，无重跑）；统计＝纯 Python 精确检验（McNemar 精确 / Fisher 精确 / CP95）。",
        "> 口径依据：`论文/论文结论卡.md` [T15]「可做」②；判读依据：`docs/protocol/review-runbook-20260921.md` §5 第 7 条。",
        "> ⚠️ KW 组结构上不可能出现 C→W、KC 组结构上不可能出现 W→C（标签定义使然）⇒ 跨组比较必须带 DK 参考组。",
        "",
    ]

    # ── ROME 1.7B λ=20 ──
    rome_root = root / "experiments" / "outputs" / "phase16_rome_review"
    if rome_root.exists():
        md += ["## 一、ROME 1.7B（λ 由 val 选中）", ""]
        rome_rows = {}
        for seed in (123, 456):
            if not (rome_root / f"phase16_rome_seed{seed}_samples.json").exists():
                continue
            rows, meta = load_rome(rome_root, seed)
            rome_rows[seed] = rows
            t = transition(rows)
            report[f"rome_seed{seed}"] = {"meta": meta, **t, "compare": kw_vs_dk(t)}
            md.append(arm_md(f"ROME seed {seed}（{meta['key']}，val 选参 metric={meta['selection']['metric']}）", t))
        if len(rome_rows) >= 2:
            t = transition(pool(*rome_rows.values()))
            report["rome_pooled"] = {**t, "compare": kw_vs_dk(t)}
            md.append(arm_md("ROME 双 seed pooled", t))

    # ── TLDC 8B β 曲线 ──
    tldc8_root = root / "experiments" / "outputs" / "lin_theory_8b"
    if tldc8_root.exists():
        md += ["## 二、TLDC 8B（β 全档透明报告，无 val 选参）", ""]
        per_seed = {}
        for seed in (123, 456):
            p = tldc8_root / f"seed{seed}_8b" / "s14_tldc_samples.json"
            if p.exists():
                per_seed[seed] = load_tldc(p)[0]
        betas = sorted(next(iter(per_seed.values())).keys()) if per_seed else []
        hdr = (
            "| β | 救回 (KW) | 救回率 | 破坏 (KC) | 破坏率 | DK 救回率 | DK 破坏率 | "
            "净 pp | KW/DK 救回比 | Fisher p |\n|---|---|---|---|---|---|---|---|---|---|\n"
        )
        rows_md = []
        for b in betas:
            rows = pool(*[per_seed[s][b] for s in per_seed])
            t = transition(rows)
            cmp_ = kw_vs_dk(t)
            ps = t["per_subset"]
            kw, kc, dk = ps.get("know_wrong", {}), ps.get("know_correct", {}), ps.get("dont_know", {})
            rk = cmp_.get("rescue_kw_vs_dk", {})
            report[f"tldc8b_beta{b}"] = {**t, "compare": cmp_}
            rk_ratio = rk.get("ratio")
            rk_p = rk.get("p")
            ratio_s = f"{rk_ratio:.2f}×" if isinstance(rk_ratio, float) else "—"
            p_s = f"{rk_p:.4f}" if isinstance(rk_p, float) else "—"
            rows_md.append(
                f"| {b} | {kw.get('rescue')}/{kw.get('base_wrong')} | {fmt_pct(kw.get('rescue_rate'))} | "
                f"{kc.get('break')}/{kc.get('base_right')} | {fmt_pct(kc.get('break_rate'))} | "
                f"{fmt_pct(dk.get('rescue_rate'))} | {fmt_pct(dk.get('break_rate'))} | "
                f"{t['net_pp']:+.1f} | {ratio_s} | {p_s} |"
            )
        md += ["双 seed pooled（n=600）：", "", hdr + "\n".join(rows_md), ""]
        for b in (0.03, 0.20):
            if b in betas:
                rows = pool(*[per_seed[s][b] for s in per_seed])
                t = transition(rows)
                md.append(arm_md(f"TLDC 8B 双 seed pooled @ β={b}", t))

    # ── TLDC 1.7B（单 seed，探索性）──
    p17 = root / "experiments" / "outputs" / "lin_theory" / "s14_tldc_samples.json"
    if p17.exists():
        md += ["## 三、TLDC 1.7B（⚠️ 仅 seed123 单 seed，探索性，不作结论）", ""]
        per_beta, cfg = load_tldc(p17)
        for b in (0.03, 0.20):
            if b in per_beta:
                t = transition(per_beta[b])
                report[f"tldc17b_beta{b}"] = {**t, "compare": kw_vs_dk(t)}
                md.append(arm_md(f"TLDC 1.7B seed{cfg.get('seed_test')} @ β={b}", t))

    (out_dir / "transition_matrix.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "transition_matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(md))
    print(f"\n[saved] {out_dir / 'transition_matrix.md'}")
    print(f"[saved] {out_dir / 'transition_matrix.json'}")


if __name__ == "__main__":
    main()
