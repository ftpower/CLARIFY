"""秩代理口径复算的判读（零 GPU）：三套口径 × 干预结果 join → 救回/破坏率对照 + 预注册判据。

输入
  - `experiments/outputs/rank_audit_proxy/rank_audit_seed{123,456}_<tag>.json`（上一步 GPU 产出）
  - 干预结果（按 question join）：
      1.7B：`experiments/outputs/lin_theory/s14_tldc_samples.json`（seed123）
            `experiments/outputs/lin_theory/seed456/s14_tldc_samples.json`（seed456）
            → `baseline_correct` + `correct_beta0.20`（real 臂 β=0.20）
      8B  ：`experiments/outputs/tldc_controls_8b/tldc_controls_{123,456}_*.json`
            → `baseline_correct` + `correct_real_beta0.2`

判据（预注册于 `audit_rank_proxy_recompute.py` 头部，跑前写死）
  主判量 = KW 计数与救回率在 B(fixed_first)/C(min_alias) 口径下的变化
  - |ΔKW|/KW_old >= 10%  或  救回率变化 >= 2pp  ⇒ 论文改用新口径并重算受影响数字
  - < 5% 且 < 1pp                              ⇒ 保留现口径 + 审计章披露
  - 其余＝灰区 ⇒ 报数不判

口径定义
  old_first  ：第一非空别名的原始首 token（含空格 artifact；已发布口径）
  fixed_first：第一非空别名的首个非空白 token（修 artifact）
  min_alias  ：min over 全部别名（各取首个非空白 token）的 rank（B1）

用法：
    ~/miniconda3/envs/pytorch_env0/bin/python experiments/lin_theory/analyze_rank_audit.py
可选 --tag Qwen3-8B（8B 结果落盘后同命令判读）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO / "experiments" / "outputs" / "rank_audit_proxy"
OUT_DIR = REPO / "experiments" / "outputs" / "rank_audit_proxy"

CRITERIA = ("old", "fixed_first", "min_alias")

INTERVENTION_SOURCES = {
    "Qwen3-1.7B": {
        123: ("s14", REPO / "experiments/outputs/lin_theory/s14_tldc_samples.json", "correct_beta0.20"),
        456: ("s14", REPO / "experiments/outputs/lin_theory/seed456/s14_tldc_samples.json", "correct_beta0.20"),
    },
    "Qwen3-8B": {
        123: ("ctrl", REPO / "experiments/outputs/tldc_controls_8b/tldc_controls_123_real-shuffle-gauss-anti-wrong_late-wrong_zero.json", "correct_real_beta0.2"),
        456: ("ctrl", REPO / "experiments/outputs/tldc_controls_8b/tldc_controls_456_real-shuffle.json", "correct_real_beta0.2"),
    },
}


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def load_outcomes(tag: str, seed: int):
    """question -> (baseline_correct, intervened_correct)

    ⚠️ `s14_tldc_samples.json` 的 `question` 字段被截断到 80 字符（validate_s14_tldc.py 写入时
    `sample["question"][:80]`）⇒ 同时登记全长与 80 截断两个键，避免静默漏配。
    """
    spec = INTERVENTION_SOURCES.get(tag, {}).get(seed)
    if spec is None:
        return None
    _, path, key = spec
    if not path.exists():
        return None
    d = load_json(path)
    out = {}
    for s in d["samples"].values():
        if key not in s:
            continue
        val = (bool(s["baseline_correct"]), bool(s[key]))
        q = str(s["question"])
        out[q] = val
        if len(q) > 80:
            out.setdefault(q[:80], val)
    return out


def lookup(outcomes, question: str):
    if outcomes is None:
        return None
    q = str(question)
    if q in outcomes:
        return outcomes[q]
    return outcomes.get(q[:80])


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def evaluate(entries, outcomes, crit, thr=50):
    """在给定口径下重算子集与救回/破坏率。"""
    res = {"n": 0, "kc": 0, "kw": 0, "dk": 0, "rescue": 0, "break": 0,
           "n_join": 0, "all_delta": 0}
    for e in entries:
        sub = e[crit]["subset"]
        res["n"] += 1
        res[{"know_correct": "kc", "know_wrong": "kw", "dont_know": "dk"}[sub]] += 1
        o = lookup(outcomes, e["question"])
        if o is None:
            continue
        res["n_join"] += 1
        base, inter = o
        res["all_delta"] += int(inter) - int(base)
        if sub == "know_wrong" and (not base) and inter:
            res["rescue"] += 1
        if sub == "know_correct" and base and (not inter):
            res["break"] += 1
    res["rate_rescue"], res["rescue_lo"], res["rescue_hi"] = wilson(res["rescue"], res["kw"])
    res["rate_break"], res["break_lo"], res["break_hi"] = wilson(res["break"], res["kc"])
    res["net_events"] = res["rescue"] - res["break"]
    res["net_pp_all"] = res["all_delta"] / res["n"] * 100
    res["join_rate"] = res["n_join"] / max(1, res["n"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", type=str, default="Qwen3-1.7B")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    args = ap.parse_args()

    report, pooled = {}, {c: {"kwargs": 0} for c in CRITERIA}
    for seed in args.seeds:
        path = AUDIT_DIR / f"rank_audit_seed{seed}_{args.tag}.json"
        if not path.exists():
            print(f"[warn] 缺档 {path}")
            continue
        entries = load_json(path)["entries"]
        outcomes = load_outcomes(args.tag, seed)
        if outcomes is None:
            print(f"[warn] {args.tag} seed{seed}: 无干预结果可 join，只报标签计数")
        block = {"seed": seed, "n": len(entries),
                 "space_artifact": sum(1 for e in entries if e["candidates"]
                                       and e["candidates"][0]["space_artifact"]),
                 "alias_unstable_know": sum(1 for e in entries if len(
                     {c["rank_final"] <= 50 for c in e["candidates"]}) > 1)}
        for crit in CRITERIA:
            block[crit] = evaluate(entries, outcomes or {}, crit)
        if outcomes is not None:
            jr = min(block[c]["join_rate"] for c in CRITERIA)
            if jr < 0.999:
                print(f"[warn] seed{seed}: 干预结果 join 率仅 {jr*100:.1f}% —— 救回/破坏可能被低估")
            else:
                print(f"[ok] seed{seed}: 干预结果 join 100%（{block['old']['n_join']}/{block['old']['n']}）")
        report[seed] = block

    # ── markdown ─────────────────────────────────────────────────────────
    tag = args.tag
    md = [f"# 秩代理口径复算判读 — {tag}", "",
          "判据（跑前预注册）：|ΔKW|/KW_old ≥10% 或 救回率变化 ≥2pp ⇒ 改用新口径重算；<5% 且 <1pp ⇒ 保留现口径。", ""]
    for seed, b in report.items():
        md += [f"## seed{seed}（n={b['n']}；空格 artifact {b['space_artifact']}；别名改判 know {b['alias_unstable_know']}）", "",
               "| 口径 | KC | KW | DK | 救回 (KW 内) | 破坏 (KC 内) | 净事件 | All Δ (pp) | join |",
               "|---|---|---|---|---|---|---|---|---|"]
        for crit in CRITERIA:
            r = b[crit]
            md.append(f"| {crit} | {r['kc']} | {r['kw']} | {r['dk']} | "
                      f"{r['rescue']}/{r['kw']} = {r['rate_rescue']*100:.1f}% "
                      f"(CP95 {r['rescue_lo']*100:.1f}–{r['rescue_hi']*100:.1f}) | "
                      f"{r['break']}/{r['kc']} = {r['rate_break']*100:.1f}% | "
                      f"{r['net_events']:+d} | {r['net_pp_all']:+.2f} | {r['n_join']}/{r['n']} |")
        md.append("")
    # 判据核算
    md += ["## 预注册判据核算", ""]
    for seed, b in report.items():
        old, fx, mn = b["old"], b["fixed_first"], b["min_alias"]
        d_kw_fx = (fx["kw"] - old["kw"]) / max(1, old["kw"])
        d_kw_mn = (mn["kw"] - old["kw"]) / max(1, old["kw"])
        d_rs_fx = (fx["rate_rescue"] - old["rate_rescue"]) * 100 if old["kw"] and fx["kw"] else float("nan")
        d_rs_mn = (mn["rate_rescue"] - old["rate_rescue"]) * 100 if old["kw"] and mn["kw"] else float("nan")
        md += [f"- **seed{seed}**：KW 变化 fixed_first **{d_kw_fx*100:+.1f}%**、min_alias **{d_kw_mn*100:+.1f}%**；"
               f"救回率变化 fixed_first **{d_rs_fx:+.1f}pp**、min_alias **{d_rs_mn:+.1f}pp**"]
    (OUT_DIR / f"judge_{tag}.md").write_text("\n".join(md) + "\n")
    with open(OUT_DIR / f"judge_{tag}.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n".join(md))
    print(f"[out] {OUT_DIR/f'judge_{tag}.md'}")


if __name__ == "__main__":
    main()
