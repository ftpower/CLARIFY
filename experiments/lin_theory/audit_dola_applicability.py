"""L0-2（候选 C3）：DoLa 适用性预检清单 —— 三条件的操作化定义与在现有产物上的回算。

判据来源（**执行前设定**）：
    docs/protocol/dola-l0-analysis-20260925.md §2
    ① 层间分化：严格 JSD 与官方 R 的 max/min 均 ≥ 1.5
    ② baseline 余量：官方口径 baseline MC2 < 0.70
    ③ 选层非退化：dynamic 条件的选层分布最大份额 ≤ 0.95 且跨度 ≥ 4 层

数据（零 GPU，产物已在手）：
    默认：experiments/outputs/dola_mc_repro/jsd_profile_Qwen3-1.7B_n100.json
          experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817ps.json
    换模型：`--jsd_json` / `--mc_json`（模型级负例走
    `docs/protocol/dola-c3-negative-20260927.md` 的两条命令产出）

判别力核验（负例）：
    内部负例＝同产物中的退化的 dynamic 条件（`dyn_b0_0_14`），预期 ③ 不通过；
    模型级负例＝小模型（pythia-1b-deduped / opt-125m），判据见上述协议文件。

用法：
    python3 experiments/lin_theory/audit_dola_applicability.py [--selftest]
        [--jsd_json P] [--mc_json P] [--out_dir D] [--out_name NAME] [--label STR]
输出：<out_dir>/<out_name>.json + <out_name>_report.md（默认 dola_l0_20260925/l0_c3.*）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "outputs" / "dola_l0_20260925"
JSD_JSON = REPO / "experiments/outputs/dola_mc_repro/jsd_profile_Qwen3-1.7B_n100.json"
MC_JSON = REPO / "experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817ps.json"

T_RATIO = 1.5          # ① 分化比值阈值（与 S1 事前设定一致）
T_BASELINE_MC2 = 0.70  # ② baseline 余量
T_DOMINANT = 0.95      # ③ 选层占优份额上界
T_SPAN = 4             # ③ 选层跨度下界（层）


def check_differentiation(jsd, profile="answer"):
    st = jsd["stats"][profile]
    strict = {int(k): float(v) for k, v in st["mean_by_depth_x1e5"].items()}
    official_r = {int(k): float(v) for k, v in st["r_mean_by_depth_x1e5"].items()}
    def ratio(d):
        vals = [v for v in d.values()]
        return max(vals) / min(vals) if min(vals) > 0 else float("inf")
    rs, ro = ratio(strict), ratio(official_r)
    return {
        "profile": profile,
        "strict_jsd_max_min_ratio": rs,
        "official_R_max_min_ratio": ro,
        "strict_jsd_argmax_depth": max(strict, key=strict.get),
        "official_R_argmax_depth": max(official_r, key=official_r.get),
        "official_R_min_depth": min(official_r, key=official_r.get),
        "spearman_depth_vs_strict_jsd": st.get("spearman_depth_vs_jsd"),
        "ok": bool(rs >= T_RATIO and ro >= T_RATIO),
        "n_positions": st.get("n_positions"),
    }


def check_selection(mc, cond):
    dist = mc["premature_layer_dist"][cond]
    total = sum(dist.values())
    top = max(dist.items(), key=lambda kv: kv[1])
    depths = [int(k) for k in dist]
    span = max(depths) - min(depths)
    return {
        "condition": cond, "n_positions": total,
        "dominant_depth": int(top[0]), "dominant_share": top[1] / total,
        "span": span, "n_distinct_depths": len(set(dist)),
        "ok": bool(top[1] / total <= T_DOMINANT and span >= T_SPAN),
    }


def main():
    ap = argparse.ArgumentParser(description="L0-2 C3 适用性预检清单")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jsd_json", default=str(JSD_JSON), help="JSD 逐层剖面产物")
    ap.add_argument("--mc_json", default=str(MC_JSON), help="MC 打分产物（含 baseline 与选层分布）")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--out_name", default="l0_c3", help="产物名（不含扩展名）")
    ap.add_argument("--label", default="", help="报告标题附加说明（如模型名）")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsd_p, mc_p = Path(args.jsd_json), Path(args.mc_json)

    jsd = json.loads(jsd_p.read_text())
    mc = json.loads(mc_p.read_text())

    d_answer = check_differentiation(jsd, "answer")
    d_prompt = check_differentiation(jsd, "prompt")
    baseline_mc2 = float(mc["summary"]["baseline"]["MC2"])
    c2 = {"baseline_MC2_official": baseline_mc2, "threshold": T_BASELINE_MC2,
          "ok": bool(baseline_mc2 < T_BASELINE_MC2)}
    dyn_conds = sorted(mc["premature_layer_dist"].keys())
    sel = {c: check_selection(mc, c) for c in dyn_conds}

    res = {
        "thresholds": {"ratio": T_RATIO, "baseline_mc2": T_BASELINE_MC2,
                       "dominant_share": T_DOMINANT, "span": T_SPAN},
        "c1_differentiation_answer": d_answer,
        "c1_differentiation_prompt": d_prompt,
        "c2_baseline_margin": c2,
        "c3_selection": sel,
        "configs": {},
    }
    for c in dyn_conds:
        passed = [d_answer["ok"], c2["ok"], sel[c]["ok"]]
        res["configs"][c] = {
            "n_pass": int(sum(passed)),
            "verdict": "通过预检" if all(passed) else "预检不通过",
            "failed_items": [n for n, ok in
                             zip(["①层间分化", "②baseline余量", "③选层非退化"], passed) if not ok],
        }
    # 判别力核验：内部负例
    neg = [c for c in dyn_conds if not sel[c]["ok"]]
    res["discriminative_power"] = {
        "internal_negative_examples": neg,
        "ok": bool(len(neg) >= 1),
        "note": "预期：退化的 dynamic 条件（桶内选层集中于最浅层）在 ③ 上不通过",
    }
    res["sources"] = {"jsd_json": str(jsd_p), "mc_json": str(mc_p), "label": args.label}
    (out_dir / f"{args.out_name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    L = ["# L0-2（C3）DoLa 适用性预检清单报告\n",
         ("模型级负例（" + args.label + "）" if args.label else "")
         + "判据：`docs/protocol/dola-l0-analysis-20260925.md` §2（执行前设定）"
         + ("；模型级负例判据见 `docs/protocol/dola-c3-negative-20260927.md`" if args.label else "")
         + "；\n数据："
         f"`{jsd_p.name}` + `{mc_p.name}`（零 GPU）。\n",
         "## 三条件回算\n",
         "| # | 条件 | 观测 | 阈值 | 结果 |", "|---|---|---|---|---|",
         f"| ① | 层间分化（answer 剖面，严格 JSD） | max/min = {d_answer['strict_jsd_max_min_ratio']:.2f} "
         f"（峰深度 d{d_answer['strict_jsd_argmax_depth']}，n={d_answer['n_positions']}） | ≥ {T_RATIO} | "
         f"{'✅' if d_answer['strict_jsd_max_min_ratio'] >= T_RATIO else '❌'} |",
         f"| ① | 层间分化（answer 剖面，官方 R） | max/min = {d_answer['official_R_max_min_ratio']:.2f} "
         f"（峰深度 d{d_answer['official_R_argmax_depth']}，最浅 d{d_answer['official_R_min_depth']}） | ≥ {T_RATIO} | "
         f"{'✅' if d_answer['official_R_max_min_ratio'] >= T_RATIO else '❌'} |",
         f"| ② | baseline 余量 | 官方口径 baseline MC2 = {baseline_mc2:.4f} | < {T_BASELINE_MC2} | "
         f"{'✅' if c2['ok'] else '❌'} |",
         f"| ③ | 选层非退化（逐 dynamic 条件） | 见表下 | 份额 ≤ {T_DOMINANT} ∧ 跨度 ≥ {T_SPAN} | "
         f"{'✅' if all(v['ok'] for v in sel.values()) else '❌（部分条件）'} |",
         "\n### ③ 逐条件选层分布\n",
         "| dynamic 条件 | 位置数 | 占优深度 | 占优份额 | 跨度 | 结果 |", "|---|---|---|---|---|---|"]
    for c, v in sel.items():
        L.append(f"| `{c}` | {v['n_positions']} | d{v['dominant_depth']} | {v['dominant_share']:.1%} | "
                 f"{v['span']} | {'✅' if v['ok'] else '❌ 退化'} |")
    L.append("\n### 配置级判定\n")
    L.append("| 配置 | 通过项 | 判定 | 未通过项 |")
    L.append("|---|---|---|---|")
    for c, v in res["configs"].items():
        L.append(f"| `{c}` | {v['n_pass']}/3 | {v['verdict']} | {'、'.join(v['failed_items']) or '—'} |")
    dp = res["discriminative_power"]
    L.append(f"\n**判别力核验**：内部负例＝{dp['internal_negative_examples']}（③ 不通过）⇒ "
             f"清单在'选层退化'这一轴上有判别力：{'✅' if dp['ok'] else '❌'}。"
             "模型级负例（GPT2 级无分化模型）需一次前向，不在本轮零 GPU 范围。\n")
    L.append(f"**敏感性**：prompt 剖面下严格 JSD max/min = "
             f"{d_prompt['strict_jsd_max_min_ratio']:.2f}、官方 R max/min = "
             f"{d_prompt['official_R_max_min_ratio']:.2f}（① 在两侧剖面上均"
             f"{'通过' if d_prompt['ok'] else '不通过'}）。\n")
    (out_dir / f"{args.out_name}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
