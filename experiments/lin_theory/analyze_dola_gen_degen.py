"""DoLa 生成侧退化归因分离：判读脚本（零 GPU）。

判据：`docs/protocol/dola-gen-degeneration-separation-20260928.md` §1–§5（执行前固定）。
读入 rp=1.0／rp=1.2 两档产物（`per_question[].conditions[c].text`），按执行前固定的
非 ASCII 阈值（占比 >0.10）统计 `degen(c, rp)`，先过仪器失效门再按 P1／P2／P3 分支，
并复用各档 `results` 块并列报行为指标、配对事件与自检计数。

退化量的统计只依赖原始文本字符，不依赖别名判分代理（协议 §1）。

用法
----
    python3 experiments/lin_theory/analyze_dola_gen_degen.py \\
        --json10 experiments/outputs/dola_generation_20260927/gen_Qwen3-1.7B_seprp10.json \\
        --json12 experiments/outputs/dola_generation_20260927/gen_Qwen3-1.7B_seprp12.json \\
        --out_dir experiments/outputs/dola_generation_20260927
    python3 experiments/lin_theory/analyze_dola_gen_degen.py --selftest   # 零 GPU 自检

缺档行为：只给一档 ⇒ 只报该档读数并标注「缺档，不作判定」；两档齐备才出分支与交互量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── 执行前固定的常数（协议 §1/§3/§4）────────────────────────────────────────
DEGEN_THR = 0.10          # 非 ASCII 字符占比 > 0.10 记为退化样本
SENS_THR = 0.25           # 敏感性阈值（只报不判）
P1_LE = 0.10              # P1：两档 degen ≤ 10%
P2_GE = 0.20              # P2：任一档 degen ≥ 20%
GATE_GT = 0.10            # 仪器失效门：baseline degen > 10% ⇒ 不判定
CONDS = ["baseline", "dola_static", "dola_dynamic"]
MAIN_REF = "主档先验（n=300，rp=1.2）：baseline 0/300、dola_static 93/300、dola_dynamic 83/300"


def non_ascii_frac(text: str) -> float:
    """非 ASCII 字符占比；空文本记 1.0（视为完全退化）。"""
    if not text:
        return 1.0
    return sum(1 for ch in text if not ch.isascii()) / len(text)


def degen_counts(recs: list, cond: str, thr: float = DEGEN_THR):
    """返回 (退化题数, 总题数)。"""
    bad = sum(1 for r in recs if non_ascii_frac(r["conditions"][cond]["text"]) > thr)
    return bad, len(recs)


def _load(path: str):
    d = json.loads(Path(path).read_text())
    recs = d["per_question"]
    meta = d["meta"]["config"]
    return recs, meta, d.get("results")


def _pct(bad: int, n: int) -> str:
    return f"{bad}/{n}（{bad / n:.1%}）"


def _bootstrap_ci(bads: list, n: int, n_boot: int = 4000, seed: int = 42, alpha: float = 0.05):
    """退化率的 bootstrap 95% CI（描述性）。"""
    import random
    rng = random.Random(seed)
    s = len(bads)
    if s == 0:
        return (float("nan"), float("nan"))
    ests = []
    for _ in range(n_boot):
        idx = [rng.randrange(s) for _ in range(s)]
        ests.append(sum(bads[i] for i in idx) / s)
    ests.sort()
    lo, hi = int(alpha / 2 * n_boot), int((1 - alpha / 2) * n_boot) - 1
    return (ests[lo], ests[max(hi, 0)])


def _label_frac(results: dict, cond: str, key: str) -> float:
    if isinstance(results, dict) and "conditions" in results:
        v = results["conditions"].get(cond) or {}
        return float(v.get(key, float("nan")))
    return float("nan")


def _render_table(recs, results, thr):
    lines = ["| 条件 | 退化（非 ASCII>10%） | 退化（>25%，只报不判） | truth_like | reject | unmatched | 截断比例 |",
             "|---|---|---|---|---|---|---|"]
    bads_all = {}
    for c in CONDS:
        bad, n = degen_counts(recs, c, DEGEN_THR)
        bad_s, _ = degen_counts(recs, c, SENS_THR)
        bads_all[c] = [1 if non_ascii_frac(r["conditions"][c]["text"]) > DEGEN_THR else 0 for r in recs]
        tl = _label_frac(results, c, "truth_like")
        rj = _label_frac(results, c, "reject")
        um = _label_frac(results, c, "unmatched")
        tf = _label_frac(results, c, "truncated_frac")
        lines.append(f"| `{c}` | {_pct(bad, n)} | {_pct(bad_s, n)} | "
                     f"{tl:.3%} | {rj:.3%} | {um:.3%} | {tf:.3%} |")
    return lines, bads_all


def verdict(rates10: dict, rates12: dict | None) -> tuple[str, str]:
    """先过仪器失效门，再按 P1/P2/P3 分支（协议 §5）。"""
    gate = [rates10["baseline"] > GATE_GT]
    if rates12 is not None:
        gate.append(rates12["baseline"] > GATE_GT)
    if any(gate):
        return ("仪器失效", f"baseline 自身退化（degen > {GATE_GT:.0%}），不判定，转入排查")
    if rates10["dola_static"] <= P1_LE and rates10["dola_dynamic"] <= P1_LE:
        return ("P1", "退化主要由「对比 × rp=1.2」交互造成（M1 交互论获支持）")
    if rates10["dola_static"] >= P2_GE or rates10["dola_dynamic"] >= P2_GE:
        return ("P2", "退化主要由对比算子本身造成（M2 算子论获支持）")
    return ("P3", "不确定区间（并列披露，不作结论）")


def render_judge(recs10, meta10, results10, recs12=None, meta12=None, results12=None):
    L = ["# DoLa 生成侧退化归因分离：判读报告\n",
         "判据：`docs/protocol/dola-gen-degeneration-separation-20260928.md` §1–§5（执行前设定）。\n"]
    L.append(f"- 档 rp=1.0：`{meta10.get('model')}`，n={meta10.get('n_questions')}，"
             f"max_new={meta10.get('max_new')}，APC relative_top={meta10.get('relative_top')}")
    if meta12 is not None:
        L.append(f"- 档 rp=1.2：同上配置，仅 rp 不同")
    L.append(f"- 参照：{MAIN_REF}\n")
    L.append("## rp=1.0 档\n")
    lines, bads10 = _render_table(recs10, results10, DEGEN_THR)
    L += lines
    rates10 = {c: sum(bads10[c]) / len(bads10[c]) for c in CONDS}
    n = len(bads10[CONDS[0]])
    ci10 = {c: _bootstrap_ci(bads10[c], n) for c in CONDS}
    L.append("\n退化率 bootstrap 95% CI（描述性）："
             + "；".join(f"`{c}` [{ci10[c][0]:.3f}, {ci10[c][1]:.3f}]" for c in CONDS))
    L.append("")
    if recs12 is not None:
        L.append("## rp=1.2 档\n")
        lines12, bads12 = _render_table(recs12, results12, DEGEN_THR)
        L += lines12
        rates12 = {c: sum(bads12[c]) / len(bads12[c]) for c in CONDS}
        ci12 = {c: _bootstrap_ci(bads12[c], len(bads12[CONDS[0]])) for c in CONDS}
        L.append("\n退化率 bootstrap 95% CI（描述性）："
                 + "；".join(f"`{c}` [{ci12[c][0]:.3f}, {ci12[c][1]:.3f}]" for c in CONDS))
        delta = (rates12["dola_dynamic"] - rates10["dola_dynamic"]) - \
                (rates12["baseline"] - rates10["baseline"])
        L.append(f"\n交互量（描述性）：Δ = [degen(dyn,1.2)−degen(dyn,1.0)] − [degen(bl,1.2)−degen(bl,1.0)]"
                 f" = **{delta:+.1%}**")
        branch, msg = verdict(rates10, rates12)
        L.append(f"\n## 判定：**{branch}**\n")
        L.append(f"- {msg}")
        if isinstance(results10, dict) and isinstance(results12, dict):
            for tag, res in (("rp=1.0", results10), ("rp=1.2", results12)):
                sc = res.get("selfcheck") or {}
                L.append(f"- {tag} 协议自检：lens 一致率 {sc.get('lens_agree')}/{sc.get('lens_n')}")
                p = (res.get("paired") or {}).get("dola_dynamic") or {}
                L.append(f"- {tag} 配对（相对 baseline，主条件 dola_dynamic）：救回 {p.get('rescued')}／"
                         f"破坏 {p.get('broken')}／净 {p.get('net')}（Δtruth_like {p.get('truth_like_delta_pp')}pp）")
    else:
        L.append("\n## 判定：缺档（仅 rp=1.0 产物）——只报读数，不作判定\n")
    L.append("\n**禁止事项**（协议 §5）：不得据本档改主档判读；不得外推「DoLa 无效」（I19）；"
             "不得事后调整阈值。")
    return "\n".join(L)


def _selftest() -> int:
    import math
    ok = True

    def chk(name, cond, a=None, b=None, tol=0.0):
        nonlocal ok
        good = bool(cond) if b is None else (bool(cond) and abs(a - b) <= tol)
        print(f"[S] {name}: {'PASS' if good else 'FAIL'}")
        ok = ok and good

    chk("空文本记 1.0", non_ascii_frac("") == 1.0)
    chk("纯 ASCII 记 0", non_ascii_frac("hello world") == 0.0)
    chk("半 ASCII", True, non_ascii_frac("abc中"), 0.25, 1e-12)
    chk("全非 ASCII", non_ascii_frac("中文测试") == 1.0)
    # 假记录构造
    def recs_with(rates, n=100):
        out = []
        per = {"baseline": round(rates["baseline"] * n), "dola_static": round(rates["dola_static"] * n),
               "dola_dynamic": round(rates["dola_dynamic"] * n)}
        for i in range(n):
            cond = {}
            for c in CONDS:
                junk = "ascii ok text"
                if i < per[c]:
                    junk = "中文乱码" + "中" * 5
                cond[c] = {"text": junk, "label": "unmatched", "truncated": True}
            out.append({"qi": i, "question": "q", "conditions": cond})
        return out
    r1 = recs_with({"baseline": 0.0, "dola_static": 0.05, "dola_dynamic": 0.08})
    r10 = {c: degen_counts(r1, c)[0] / 100 for c in CONDS}
    chk("B1 分支（交互为主）", verdict(r10, r10)[0] == "P1")
    r2 = recs_with({"baseline": 0.0, "dola_static": 0.35, "dola_dynamic": 0.25})
    r20 = {c: degen_counts(r2, c)[0] / 100 for c in CONDS}
    chk("B2 分支（算子为主）", verdict(r20, r20)[0] == "P2")
    r3 = recs_with({"baseline": 0.0, "dola_static": 0.15, "dola_dynamic": 0.08})
    r30 = {c: degen_counts(r3, c)[0] / 100 for c in CONDS}
    chk("B3 不确定区间", verdict(r30, r30)[0] == "P3")
    r4 = recs_with({"baseline": 0.15, "dola_static": 0.05, "dola_dynamic": 0.05})
    r40 = {c: degen_counts(r4, c)[0] / 100 for c in CONDS}
    chk("仪器失效门", verdict(r40, r40)[0] == "仪器失效")
    chk("阈值敏感性 0.25", degen_counts(r2, "dola_static", SENS_THR)[0] == 35)
    # 渲染冒烟
    md = render_judge(r1, {"model": "stub", "n_questions": 100, "max_new": 20, "relative_top": 0.0},
                      None, r2, {"model": "stub"}, None)
    chk("报告渲染（含判定与交互量）", "判定" in md and "交互量" in md)
    print("=" * 72)
    print("自检结果：" + ("全部 PASS" if ok else "存在 FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="DoLa 生成侧退化归因分离判读（零 GPU）")
    ap.add_argument("--json10", type=str, default=None, help="rp=1.0 档产物（主判定输入）")
    ap.add_argument("--json12", type=str, default=None, help="rp=1.2 档产物（交互量与对照）")
    ap.add_argument("--out_dir", type=str, default=None, help="给定则把报告写入该目录 *_degen_report.md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.json10 and not args.json12:
        ap.error("须至少给 --json10 或 --json12")
    recs10 = meta10 = res10 = None
    if args.json10:
        recs10, meta10, res10 = _load(args.json10)
    recs12 = meta12 = res12 = None
    if args.json12:
        recs12, meta12, res12 = _load(args.json12)
    if recs10 is None:  # 只有 rp=1.2 ⇒ 以它为唯一档，只报读数
        recs10, meta10, res10 = recs12, meta12, res12
        recs12 = meta12 = res12 = None
        md = render_judge(recs10, meta10, res10)
    else:
        md = render_judge(recs10, meta10, res10, recs12, meta12, res12)
    print(md)
    if args.out_dir:
        out = Path(args.out_dir) / "dola_gen_degen_report.md"
        out.write_text(md)
        print(f"[judge] 报告已写入 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
