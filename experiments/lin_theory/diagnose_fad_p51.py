"""FAD P5.1（几何可用率）+ S5 P-S5.1（抬压分支归因）零 GPU 判读。

理论/协议锚点：
  - FAD 引理 1（R/m/Δ₀/β*）、命题 5（P(r) ≤ P(g=1)·P(h=1|g=1)）、P5.1 判据：
    docs/paper/paper-route-method-schemes.md §2.2、§2.5、§5
  - S5 判停 P-S5.1：论文/论文结论卡.md [I24] 迁移；判据 ≥70%
  - 档案 schema 与判读协议：docs/protocol/geometry-archive-spec.md §6、§10
  - 判读纪律：runbook §5（救回/破坏必须带 DK 参考组；预注册阈值不得事后更改）

预注册判据（照 spec §6.4 与 S5 卡，禁止事后改）：
  P5.1（总闸）：P(g=1) < 5% 且 KW/KC 组差异 Fisher p>0.05 ⇒ 几何不可用 ⇒ FAD 判停（转 B 路线）。
  P-S5.1：救回事件中"抬支单独驱动"占比 ≥70% 且破坏事件中"压支单独驱动"占比 ≥70%，否则关闭 S5。

输入（均已落盘，零 GPU）：
  --geo    geometry archive（dump_geometry_archive.py 产物，可多个 seed）
  --tok    （可选）per-token archive（analyze_tldc_per_token.py 产物）→ 启用 P-S5.1
           需与 geo 同 seed/同批次（脚本会做 question 对齐校验）

用法：
  python experiments/lin_theory/diagnose_fad_p51.py \
    --geo experiments/outputs/geometry_archive/geo_baseline_seed123.json \
          experiments/outputs/geometry_archive/geo_baseline_seed456.json \
    --tok experiments/outputs/lin_theory/s15_2b_tldc_per_token.json \
    --tok_beta 0.10
"""

import argparse
import json
from pathlib import Path

from analyze_transition_matrix import clopper_pearson, fisher_greater

BETAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
# 预注册阈值（spec §6.4 / S5 卡）
P_G1_FLOOR = 0.05
S5_BRANCH_THRESHOLD = 0.70


def load_geo(paths):
    """→ {sample_key: {"subset":…, "steps":[…]}}，多 seed 以 seed:sid 复合键 pool。"""
    out = {}
    metas = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        meta = d["meta"]
        metas.append({"file": Path(p).name, "seed": meta["seed_test"],
                      "model": meta["model"], "layer_early": meta["layer_early"]})
        for s in d["samples"]:
            out[f"{meta['seed_test']}:{s['sample_id']}"] = s
    return out, metas


def p51(geo):
    """几何可用率 + 可负担性 + β* 分布（命题 5 的两个因子）。"""
    per_group = {}
    rows = []
    for key, s in geo.items():
        steps = s["steps"]
        yt = [st["geo"]["yt"] for st in steps]
        in_r = [bool(t["in_R"]) for t in yt]
        betas_yt = [t["beta_star"] for t in yt if t["in_R"]]
        # g(x) = ∃t: c* ∈ R_t（spec 定义）；另报承诺步（step 0）单独口径
        g_any = any(in_r)
        g_step0 = bool(in_r[0]) if in_r else False
        # h(x,β) = 1[min_{t: c*∈R_t} β* ≤ β]
        min_beta = min(betas_yt) if betas_yt else None
        first_in_r = next((i for i, v in enumerate(in_r) if v), None)
        r = {"key": key, "subset": s["subset"], "g_any": g_any, "g_step0": g_step0,
             "min_beta_star": min_beta, "first_in_r_step": first_in_r,
             "n_steps_in_r": sum(in_r), "n_steps": len(steps)}
        rows.append(r)
        per_group.setdefault(s["subset"], []).append(r)

    def rate(rs, f):
        return (sum(1 for r in rs if f(r)) / len(rs)) if rs else None

    summary = {}
    for sub, rs in per_group.items():
        g = rate(rs, lambda r: r["g_any"])
        ci = clopper_pearson(sum(1 for r in rs if r["g_any"]), len(rs))
        summary[sub] = {
            "n": len(rs), "P_g_any": g, "P_g_any_ci": ci,
            "P_g_step0": rate(rs, lambda r: r["g_step0"]),
            "mean_steps_in_r": (sum(r["n_steps_in_r"] for r in rs) / len(rs)) if rs else None,
        }
        # 可负担性曲线 P(h=1)、P(h=1|g=1)
        affordability = {}
        gs = [r for r in rs if r["g_any"]]
        for b in BETAS:
            if b == 0.0:
                continue
            h = sum(1 for r in gs if r["min_beta_star"] is not None and r["min_beta_star"] <= b)
            affordability[b] = {
                "P_h": h / len(rs) if rs else None,
                "P_h_given_g": h / len(gs) if gs else None,
                "ceiling_pp": 100 * h / len(rs) if rs else None,
            }
        summary[sub]["affordability"] = affordability

    # KW vs KC（及 vs DK）的 P(g=1) 差异
    tests = {}
    for a, b_ in (("know_wrong", "know_correct"), ("know_wrong", "dont_know"), ("know_correct", "dont_know")):
        if a in per_group and b_ in per_group:
            ga = sum(1 for r in per_group[a] if r["g_any"])
            gb = sum(1 for r in per_group[b_] if r["g_any"])
            tests[f"{a}_vs_{b_}"] = {
                "a": f"{ga}/{len(per_group[a])}", "b": f"{gb}/{len(per_group[b_])}",
                "fisher_greater_p": fisher_greater(
                    ga, len(per_group[a]) - ga, gb, len(per_group[b_]) - gb),
            }
    return {"per_group": summary, "tests": tests, "rows": rows}


def ps51(geo, tok_path, tok_beta):
    """S5 分支归因：首分叉步上，real TLDC 的选择由哪支驱动。"""
    tok = json.loads(Path(tok_path).read_text())
    # question 对齐校验（防批次错位）
    geo_by_q = {}
    for key, s in geo.items():
        geo_by_q.setdefault(s["question"][:60], key)
    mism = 0
    events = {"rescue": [], "break": [], "other": []}
    no_flip = 0
    for s in tok:
        q = s["question"][:60]
        gkey = geo_by_q.get(q)
        if gkey is None:
            mism += 1
            continue
        gsteps = {st["step"]: st for st in geo[gkey]["steps"]}
        gids, gids_bl = s["gids"], s["gids_bl"]
        t = None
        for i in range(min(len(gids), len(gids_bl))):
            if gids[i] != gids_bl[i]:
                t = i
                break
        if t is None:
            no_flip += 1
            continue
        if t not in gsteps:
            mism += 1
            continue
        grid = gsteps[t]["grid"]
        bkey = f"{tok_beta:.2f}"
        chosen = gids[t]
        lift = grid["lift"].get(bkey) == chosen
        damp = grid["damp"].get(bkey) == chosen
        sym = grid["sym"].get(bkey) == chosen
        if not sym:
            label = "unexplained"  # 对称算子在该前缀下不复现实际选择 → 协议/数值不符
        elif lift and damp:
            label = "both_ambiguous"
        elif lift:
            label = "lift_only"
        elif damp:
            label = "damp_only"
        else:
            label = "needs_both"  # 单支都不行，需两支协同（对称才翻）
        rec = {"q": q[:40], "step": t, "label": label}
        if s["subset"] == "know_wrong" and not s["baseline_correct"] and s["is_correct"]:
            events["rescue"].append(rec)
        elif s["subset"] == "know_correct" and s["baseline_correct"] and not s["is_correct"]:
            events["break"].append(rec)
        else:
            events["other"].append(rec)

    def shares(evs):
        n = len(evs)
        return {
            "n": n,
            "lift_only": sum(1 for e in evs if e["label"] == "lift_only") / n if n else None,
            "damp_only": sum(1 for e in evs if e["label"] == "damp_only") / n if n else None,
            "needs_both": sum(1 for e in evs if e["label"] == "needs_both") / n if n else None,
            "both_ambiguous": sum(1 for e in evs if e["label"] == "both_ambiguous") / n if n else None,
            "unexplained": sum(1 for e in evs if e["label"] == "unexplained") / n if n else None,
        }

    out = {"alignment_mismatch": mism, "no_flip_samples": no_flip, "beta": tok_beta}
    for k, v in events.items():
        out[k] = shares(v)
        out[k]["examples"] = v[:5]
    # 预注册判据
    r, b_ = out["rescue"], out["break"]
    ok = (r["lift_only"] is not None and r["lift_only"] >= S5_BRANCH_THRESHOLD and
          b_["damp_only"] is not None and b_["damp_only"] >= S5_BRANCH_THRESHOLD)
    out["verdict"] = "P-S5.1 通过（单侧归因成立）" if ok else "P-S5.1 不通过 ⇒ 关闭 S5"
    return out


def main():
    ap = argparse.ArgumentParser(description="FAD P5.1 / S5 P-S5.1 零 GPU 判读")
    ap.add_argument("--geo", nargs="+", required=True)
    ap.add_argument("--tok", type=str, default=None)
    ap.add_argument("--tok_beta", type=float, default=0.10)
    ap.add_argument("--output_dir", type=str, default="experiments/outputs/fad_diagnostics_20260921")
    args = ap.parse_args()

    geo, metas = load_geo(args.geo)
    files_desc = ", ".join(f"{m['file']}(seed{m['seed']})" for m in metas)
    print("=" * 76)
    print("FAD P5.1 / S5 P-S5.1 零 GPU 判读")
    print(f"  档案：{files_desc}")
    print(f"  样本 {len(geo)}，ℓ*={metas[0]['layer_early']}")
    print("=" * 76)

    res = p51(geo)
    md = ["# FAD P5.1 / S5 P-S5.1 判读报告", "",
          f"- 输入：{', '.join(m['file'] for m in metas)}（ℓ*={metas[0]['layer_early']}，"
          f"模型 {metas[0]['model']}）",
          f"- 样本数 {len(geo)}；判据预注册见 `docs/protocol/geometry-archive-spec.md` §6", ""]
    md += ["## 一、P5.1 几何可用率（总闸）", "",
           "| 子集 | n | P(g=1)（∃t，CP95） | P(g=1)（step0） | 平均 in_R 步数 |",
           "|---|---|---|---|---|"]
    for sub, s in res["per_group"].items():
        ci = s["P_g_any_ci"]
        md.append(f"| {sub} | {s['n']} | {s['P_g_any']*100:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] | "
                  f"{s['P_g_step0']*100:.1f}% | {s['mean_steps_in_r']:.1f} |")
    md += ["", "组间 Fisher（greater）：", ""]
    for k, v in res["tests"].items():
        md.append(f"- {k}: {v['a']} vs {v['b']} → p={v['fisher_greater_p']:.4f}")
    md += ["", "### 命题 5 天花板：P(h=1)（可负担性）", "",
           "| β | KW P(h=1) | KW P(h\\|g=1) | KC P(h=1) | KC P(h\\|g=1) | DK P(h=1) |",
           "|---|---|---|---|---|---|"]
    for b in BETAS:
        if b == 0.0:
            continue
        cells = []
        for sub in ("know_wrong", "know_correct", "dont_know"):
            a = res["per_group"].get(sub, {}).get("affordability", {}).get(b)
            cells.append(a)
        if not any(cells):
            continue
        def f(c, k):
            return "—" if c is None or c[k] is None else f"{c[k]*100:.1f}%"
        md.append(f"| {b} | {f(cells[0],'P_h')} | {f(cells[0],'P_h_given_g')} | "
                  f"{f(cells[1],'P_h')} | {f(cells[1],'P_h_given_g')} | {f(cells[2],'P_h')} |")

    # 总闸判定
    kw = res["per_group"].get("know_wrong", {})
    kc = res["per_group"].get("know_correct", {})
    p_kw_kc = res["tests"].get("know_wrong_vs_know_correct", {}).get("fisher_greater_p")
    g_ok = kw.get("P_g_any") is not None and kw["P_g_any"] >= P_G1_FLOOR
    sep_ok = p_kw_kc is not None and p_kw_kc <= 0.05
    md += ["", f"**P5.1 判定**：KW 组 P(g=1)={kw.get('P_g_any', float('nan'))*100:.1f}%"
              f"（阈值 ≥{P_G1_FLOOR*100:.0f}%），KW vs KC 分离 p={p_kw_kc if p_kw_kc is None else round(p_kw_kc,4)}"
              f"（阈值 ≤0.05） ⇒ " +
              ("**几何可用（继续 FAD）**" if (g_ok and sep_ok) else
               ("**几何可用但组间不分离**（有几何但无知识区分度）" if g_ok else
                "**P(g=1) 近零 ⇒ 结构性封顶 ⇒ FAD 判停（转 B 路线）**")), ""]

    out = {"p51": {k: v for k, v in res.items() if k != "rows"}, "meta": metas}
    if args.tok:
        s5 = ps51(geo, args.tok, args.tok_beta)
        out["ps51"] = s5
        md += ["## 二、S5 P-S5.1 抬压分支归因（首分叉步）", "",
               f"- 输入 per-token 档案 β={args.tok_beta}；对齐失败 {s5['alignment_mismatch']} 例，无分叉样本 {s5['no_flip_samples']} 例",
               "", "| 事件 | n | 抬支单独 | 压支单独 | 需两支协同 | 两支歧义 | 无法解释 |",
               "|---|---|---|---|---|---|---|"]
        for k in ("rescue", "break", "other"):
            e = s5[k]
            def g(kk):
                return "—" if e[kk] is None else f"{e[kk]*100:.1f}%"
            md.append(f"| {k} | {e['n']} | {g('lift_only')} | {g('damp_only')} | {g('needs_both')} | "
                      f"{g('both_ambiguous')} | {g('unexplained')} |")
        md += ["", f"**P-S5.1 判定**（阈值：救回抬支≥70% 且破坏压支≥70%）：{s5['verdict']}", ""]

    print("\n".join(md))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fad_diagnostics.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "fad_diagnostics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'fad_diagnostics.md'}")


if __name__ == "__main__":
    main()
