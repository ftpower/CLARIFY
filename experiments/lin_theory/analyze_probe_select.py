"""RII 方向 A·A2 探测-选择档判读（零 GPU）——探测档案（弱扰动）作信号源 × 最终干预档案（β=0.20）作标签源。

判据（执行前设定，2026-09-28）：docs/protocol/rii-response-informed-intervention-20260928.md §5
  A2-S0 档案自洽：探测档与基线档在分叉步之前 chosen_id 逐位一致（只核首个分叉步之前）。
  A2-S1 主判据：AUROC(救回 vs 破坏 | 探测响应 post_beta) ≥ 0.70 ∧ CP95 下界 > 0.5
              ∧（双 seed 时）双 seed 同向 ⇒ 进入 A2-S2；≤ 0.60 ⇒ A2 关闭；其余 ⇒ 不确定区间。
  A2-S2 选择投影：θ=0.5（主档）投影净事件 > 无选择参照 ∧ 超过随机选择安慰剂分布 97.5 分位 ⇒ 进入 A2-S3。
  ⚠️ 单 seed 时照常出报告，但标记「双 seed 同向对照缺失，不得据此判停/关闭」。

档案语义（继承 runbook §5.8b 实测确认）：每档每步的 final_top10/early_top10/geo 是干预前该步的读出，
只有 chosen_id 是干预后的 ⇒ t+1 步特征才是合法事后信号（本脚本读探测分叉步 t_p 的下一步）。

用法：
  python experiments/lin_theory/analyze_probe_select.py \
    --seeds 123 456 --arch experiments/outputs/geometry_archive \
    --probe_betas 0.03 0.05 0.08 --final_beta 0.20 \
    --out_dir experiments/outputs/rii_probe_select
  自检：python experiments/lin_theory/analyze_probe_select.py --selftest
"""

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

ARCH = "experiments/outputs/geometry_archive"
THETAS = [0.3, 0.5, 0.7]
THETA_MAIN = 0.5
N_PLACEBO = 2000
PLACEBO_SEED = 0


# ── 档案读取 ────────────────────────────────────────────────────────────────
def fname_beta(beta):
    """档案命名口径：geo_sym_b{beta:.2f} 去小数点（与 dump_geometry_archive.py 一致）。"""
    return ("b" + f"{beta:.2f}".replace(".", ""))


def load_archive(arch, op, seed, beta=None):
    p = Path(arch)
    if op == "baseline":
        fp = p / f"geo_baseline_seed{seed}.json"
    else:
        fp = p / f"geo_{op}_{fname_beta(beta)}_seed{seed}.json"
    if not fp.exists():
        return None, fp.name
    return {s["sample_id"]: s for s in json.loads(fp.read_text())["samples"]}, fp.name


def diverge_step(base, perturbed):
    for t in range(min(len(base["steps"]), len(perturbed["steps"]))):
        if base["steps"][t]["chosen_id"] != perturbed["steps"][t]["chosen_id"]:
            return t
    return None


def label_of(sample):
    """救回/破坏/不变（相对基线；基线与干预两档案内均存 baseline_correct 与 is_correct）。"""
    base_ok, post_ok = bool(sample["baseline_correct"]), bool(sample["is_correct"])
    if not base_ok and post_ok:
        return "rescue"
    if base_ok and not post_ok:
        return "break"
    return "nochange"


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def boot_ci(pos, neg, n=2000, seed=0):
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    v = sorted(auroc([rnd.choice(pos) for _ in pos], [rnd.choice(neg) for _ in neg]) for _ in range(n))
    return v[int(0.025 * n)], v[int(0.975 * n)]


# ── 判定纯函数（selftest 覆盖）──────────────────────────────────────────────
def verdict_s1(auroc_val, ci_lo, n_seeds, same_direction):
    """A2-S1：返回 (分支, 说明)。single-seed 标记缺失对照。"""
    if auroc_val is None or math.isnan(auroc_val):
        return "invalid", "事件数不足（任一类别为空）"
    if n_seeds < 2:
        flag = "（双 seed 同向对照缺失，不得据此判停/关闭）"
        if auroc_val <= 0.60:
            return "grey", f"单 seed {auroc_val:.3f} ≤ 0.60 {flag}"
        if auroc_val >= 0.70 and ci_lo > 0.5:
            return "enter", f"单 seed {auroc_val:.3f} ≥ 0.70 ∧ 下界 {ci_lo:.3f} > 0.5 {flag}"
        return "grey", f"单 seed 落不确定区间 {auroc_val:.3f} {flag}"
    if auroc_val <= 0.60:
        return "close", f"{auroc_val:.3f} ≤ 0.60 ⇒ A2 关闭"
    if auroc_val >= 0.70 and ci_lo > 0.5 and same_direction:
        return "enter", f"{auroc_val:.3f} ≥ 0.70 ∧ 下界 {ci_lo:.3f} > 0.5 ∧ 双 seed 同向 ⇒ 进入 A2-S2"
    return "grey", "落不确定区间（记数不判）"


def verdict_s2(proj_net, no_sel_net, placebo_p975):
    if proj_net is None or no_sel_net is None or placebo_p975 is None:
        return "invalid", "样本不足"
    if proj_net > no_sel_net and proj_net > placebo_p975:
        return "enter", (f"投影净事件 {proj_net:+d} > 无选择 {no_sel_net:+d} "
                         f"且 > 安慰剂 97.5 分位 {placebo_p975:+d} ⇒ 进入 A2-S3")
    return "report", (f"投影净事件 {proj_net:+d}（无选择 {no_sel_net:+d}、"
                      f"安慰剂 97.5 分位 {placebo_p975:+d}）⇒ 只报数不下结论")


# ── 主流程 ──────────────────────────────────────────────────────────────────
def collect(seeds, arch, probe_betas, final_beta):
    """按探测 β 聚合：(探测信号, 最终干预标签, 探测自身标签) 行。"""
    rows = {b: [] for b in probe_betas}
    meta = {"missing": [], "files": set()}
    for seed in seeds:
        B, fB = load_archive(arch, "baseline", seed)
        F, fF = load_archive(arch, "sym", seed, beta=final_beta)
        if B is None or F is None:
            meta["missing"].append(f"seed{seed}: baseline={fB} final={fF}")
            continue
        meta["files"].update({fB, fF})
        for pb in probe_betas:
            P, fP = load_archive(arch, "sym", seed, beta=pb)
            if P is None:
                meta["missing"].append(f"seed{seed}: probe_b{pb:.2f}={fP}")
                continue
            meta["files"].add(fP)
            for sid, ps in P.items():
                b, fs = B.get(sid), F.get(sid)
                if b is None or fs is None:
                    continue
                t_p = diverge_step(b, ps)
                sig = None
                if t_p is not None and t_p + 1 < len(ps["steps"]):
                    sig = ps["steps"][t_p + 1]["geo"]["beta_star_min"]
                # A2-S0 档案自洽：分叉步之前 chosen_id 须与基线逐位一致
                upto = t_p if t_p is not None else min(len(b["steps"]), len(ps["steps"]))
                s0_viol = sum(
                    1 for t in range(upto)
                    if b["steps"][t]["chosen_id"] != ps["steps"][t]["chosen_id"])
                rows[pb].append({
                    "seed": seed, "subset": ps["subset"], "t_p": t_p,
                    "signal": sig, "s0_viol": s0_viol,
                    "label_final": label_of(fs),
                    "label_probe": label_of(ps),
                })
    return rows, meta


def analyze_beta(rows, theta_main=THETA_MAIN):
    """单 β 的完整判读块；返回 dict 供报告渲染与判据调用。"""
    out = {}
    out["n_samples"] = len(rows)
    out["n_diverged"] = sum(1 for r in rows if r["t_p"] is not None)
    out["n_signal"] = sum(1 for r in rows if r["signal"] is not None)
    out["s0_violations"] = sum(r.get("s0_viol", 0) for r in rows)
    out["s0_pass"] = out["s0_violations"] == 0
    pc = {s: 0 for s in ("know_wrong", "know_correct", "dont_know")}
    for r in rows:
        if r["t_p"] is not None:
            pc[r["subset"]] += 1
    out["diverged_by_subset"] = pc
    out["probe_rescue"] = sum(1 for r in rows if r["label_probe"] == "rescue")
    out["probe_break"] = sum(1 for r in rows if r["label_probe"] == "break")

    pos = [r["signal"] for r in rows if r["signal"] is not None and r["label_final"] == "rescue"]
    neg = [r["signal"] for r in rows if r["signal"] is not None and r["label_final"] == "break"]
    out["n_rescue"], out["n_break"] = len(pos), len(neg)
    out["auroc"] = auroc(pos, neg)
    out["ci_lo"], out["ci_hi"] = boot_ci(pos, neg)

    per_seed = {}
    for s in {r["seed"] for r in rows}:
        sp = [r["signal"] for r in rows if r["seed"] == s and r["signal"] is not None
              and r["label_final"] == "rescue"]
        sn = [r["signal"] for r in rows if r["seed"] == s and r["signal"] is not None
              and r["label_final"] == "break"]
        per_seed[s] = auroc(sp, sn)
    out["per_seed"] = per_seed
    out["same_direction"] = all(v > 0.5 for v in per_seed.values()) or \
        all(v < 0.5 for v in per_seed.values())

    # 选择投影（仅信号非缺失样本；未选样本按基线结果计，与 runbook §5.8b 投影同口径）
    sel_pool = [r for r in rows if r["signal"] is not None]
    rescue_all = sum(1 for r in sel_pool if r["label_final"] == "rescue")
    break_all = sum(1 for r in sel_pool if r["label_final"] == "break")
    out["no_sel_net"] = rescue_all - break_all
    proj = {}
    rnd = random.Random(PLACEBO_SEED)
    for th in THETAS:
        kept = [r for r in sel_pool if r["signal"] >= th]
        k = len(kept)
        net = sum(1 for r in kept if r["label_final"] == "rescue") - \
            sum(1 for r in kept if r["label_final"] == "break")
        placebo_nets = []
        for _ in range(N_PLACEBO):
            sub = rnd.sample(sel_pool, k)
            placebo_nets.append(
                sum(1 for r in sub if r["label_final"] == "rescue") -
                sum(1 for r in sub if r["label_final"] == "break"))
        placebo_nets.sort()
        proj[th] = {
            "kept": k, "net": net,
            "placebo_mean": st.mean(placebo_nets),
            "placebo_p025": placebo_nets[int(0.025 * (N_PLACEBO - 1))],
            "placebo_p975": placebo_nets[int(0.975 * (N_PLACEBO - 1))],
        }
    out["proj"] = proj
    out["verdict_s1"] = verdict_s1(out["auroc"], out["ci_lo"],
                                   len(out["per_seed"]), out["same_direction"])
    m = proj.get(theta_main)
    out["verdict_s2"] = verdict_s2(m["net"] if m else None, out["no_sel_net"],
                                   m["placebo_p975"] if m else None)
    return out


def render(rows, meta, probe_betas, final_beta, arch, out_dir):
    lines = ["# RII 方向 A·A2 探测-选择档判读报告",
             "",
             f"- 判据：`docs/protocol/rii-response-informed-intervention-20260928.md` §5（执行前设定）",
             f"- 档案：`{arch}`；探测 β ∈ {probe_betas}；最终干预 β = {final_beta}；"
             f"θ 主档 = {THETA_MAIN}（敏感性 {THETAS}）",
             f"- 缺档告警：{meta['missing'] or '无'}",
             ""]
    for pb in probe_betas:
        if pb not in rows or not rows[pb]:
            lines.append(f"## β_p = {pb:.2f}：无数据（缺档）\n")
            continue
        o = analyze_beta(rows[pb])
        lines.append(f"## β_p = {pb:.2f}（探测档）\n")
        lines.append(f"- **A2-S0 档案自洽**：分叉步之前 chosen_id 不一致计数 **{o['s0_violations']}**"
                     f"（{'PASS' if o['s0_pass'] else 'FAIL'}）")
        lines.append(f"- 样本 {o['n_samples']}｜探测分叉 {o['n_diverged']}"
                     f"（KW {o['diverged_by_subset']['know_wrong']}／"
                     f"KC {o['diverged_by_subset']['know_correct']}／"
                     f"DK {o['diverged_by_subset']['dont_know']}）｜"
                     f"探测自身救回 {o['probe_rescue']}／破坏 {o['probe_break']}")
        lines.append(f"- 信号非缺失 {o['n_signal']}；最终干预标签：救回 {o['n_rescue']}／破坏 {o['n_break']}")
        lines.append(f"- 可分性（探测响应 → 最终救回/破坏）：AUROC **{o['auroc']:.3f}**"
                     f" CP95 [{o['ci_lo']:.3f}, {o['ci_hi']:.3f}]｜"
                     f"分 seed { {int(k): round(v, 3) for k, v in o['per_seed'].items()} }"
                     f"（同向={o['same_direction']}）")
        lines.append(f"- **A2-S1 判定：{o['verdict_s1'][0]}** — {o['verdict_s1'][1]}")
        lines.append("")
        lines.append("| θ | 保留数 | 投影净事件 | 无选择参照 | 安慰剂均值 | 安慰剂 97.5 分位 |")
        lines.append("|---|---|---|---|---|---|")
        for th in THETAS:
            m = o["proj"][th]
            mark = " ← 主档" if th == THETA_MAIN else ""
            lines.append(f"| {th:.1f}{mark} | {m['kept']} | **{m['net']:+d}** | "
                         f"{o['no_sel_net']:+d} | {m['placebo_mean']:+.2f} | {m['placebo_p975']:+d} |")
        lines.append(f"\n- **A2-S2 判定（θ={THETA_MAIN:.1f}）：{o['verdict_s2'][0]}** — {o['verdict_s2'][1]}")
        lines.append("")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / "rii_probe_select_report.md"
    rep.write_text("\n".join(lines))
    data = {f"b{int(pb * 100):03d}": analyze_beta(rows[pb]) for pb in probe_betas
            if rows.get(pb)}
    data["meta"] = {"missing": meta["missing"], "files": sorted(meta["files"]),
                    "thetas": THETAS, "theta_main": THETA_MAIN}
    (out / "rii_probe_select.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[saved] {rep}")


# ── 自检（合成档案，无模型）────────────────────────────────────────────────
def _mk_step(chosen, beta_star_min, margin_a, margin_b):
    return {"chosen_id": chosen,
            "final_top10": [[0, 0, margin_a], [1, 0, margin_b]],
            "early_top10": [[0, 0, margin_a], [1, 0, margin_b]],
            "geo": {"beta_star_min": beta_star_min, "R_size": 2}}


def _mk_sample(sid, subset, base_ok, post_ok, steps):
    return {"sample_id": sid, "subset": subset, "baseline_correct": base_ok,
            "is_correct": post_ok, "steps": steps}


def selftest():
    # 4 样本：A rescue、B break、C nochange、D rescue（信号阈值边界）
    A = _mk_sample("A", "know_wrong", False, True,
                   [_mk_step(1, 0.2, 1.0, 0.5), _mk_step(2, 0.9, 1.0, 0.5)])
    B = _mk_sample("B", "know_correct", True, False,
                   [_mk_step(1, 0.2, 1.0, 0.5), _mk_step(2, 0.3, 1.0, 0.5)])
    C = _mk_sample("C", "dont_know", False, False,
                   [_mk_step(1, 0.2, 1.0, 0.5), _mk_step(2, 0.6, 1.0, 0.5)])
    D = _mk_sample("D", "know_wrong", False, True,
                   [_mk_step(1, 0.2, 1.0, 0.5), _mk_step(2, 0.8, 1.0, 0.5)])
    rows = [{"seed": 123, "subset": s["subset"], "t_p": 0, "signal": s["steps"][1]["geo"]["beta_star_min"],
             "label_final": {"A": "rescue", "B": "break", "C": "nochange", "D": "rescue"}[s["sample_id"]],
             "label_probe": "nochange"} for s in (A, B, C, D)]
    o = analyze_beta(rows)
    assert o["n_samples"] == 4 and o["n_signal"] == 4
    assert o["n_rescue"] == 2 and o["n_break"] == 1
    assert o["s0_pass"] and o["s0_violations"] == 0
    rows_s0 = [dict(r, s0_viol=1) for r in rows]
    assert analyze_beta(rows_s0)["s0_violations"] == 4 and not analyze_beta(rows_s0)["s0_pass"]
    # AUROC: pos=[0.9,0.8], neg=[0.3] → (2+2)/2... 手算：0.9>0.3 ✓、0.8>0.3 ✓ ⇒ 2/2 = 1.0
    assert abs(o["auroc"] - 1.0) < 1e-9
    # θ=0.5：保留 signal≥0.5 → A(0.9), C(0.6), D(0.8)（B 0.3 落选）⇒ 净 = rescue(A,D)=2 − 0 = +2
    m = o["proj"][0.5]
    assert m["kept"] == 3 and m["net"] == 2 and o["no_sel_net"] == 1
    # 安慰剂可复现：同一随机种子两次抽样一致
    rows2 = [dict(r) for r in rows]
    o2 = analyze_beta(rows2)
    assert o2["proj"][0.5]["placebo_p975"] == m["placebo_p975"]
    # 判定纯函数
    assert verdict_s1(0.75, 0.55, 2, True)[0] == "enter"
    assert verdict_s1(0.55, 0.4, 2, True)[0] == "close"
    assert verdict_s1(0.68, 0.52, 2, True)[0] == "grey"
    assert verdict_s1(0.75, 0.55, 1, True)[0] == "enter"  # 单 seed：附缺失标记
    assert verdict_s2(5, 3, 4)[0] == "enter"
    assert verdict_s2(5, 6, 4)[0] == "report"
    assert verdict_s2(None, 3, 4)[0] == "invalid"
    # 档案命名口径
    assert fname_beta(0.20) == "b020" and fname_beta(0.05) == "b005"
    print("selftest: 全部 PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    ap.add_argument("--arch", default=ARCH)
    ap.add_argument("--probe_betas", nargs="+", type=float, default=[0.03, 0.05, 0.08])
    ap.add_argument("--final_beta", type=float, default=0.20)
    ap.add_argument("--out_dir", default="experiments/outputs/rii_probe_select")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    rows, meta = collect(args.seeds, args.arch, args.probe_betas, args.final_beta)
    render(rows, meta, args.probe_betas, args.final_beta, args.arch, args.out_dir)


if __name__ == "__main__":
    main()
