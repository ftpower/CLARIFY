"""T3 门控族判读——零 GPU，只读 `main_tldc_controls.py` 的输出。

依据 docs/protocol/placebo-control-protocol.md §9（**预注册，禁止事后更改判据**）：

  主判据（β=0.20，同批必含 real 作基准；τ=0.2 主档、0.3 次档）
    (i)   KW 样本配对 McNemar `gated vs real`：gated 救回**不显著低于** real
          （b=仅 real 救回，c=仅 gated 救回；不成立 ⇔ b>c 且双侧 p<0.05）
    (ii)  gated 的 **KC 破坏率 < real**（配对 McNemar 单侧，破坏事件减少）
          （d=仅 real 破坏，e=仅 gated 破坏；单侧 p = P(X≤e | Binom(d+e,0.5))）
          ⚠️ 预注册原文只写"KC 破坏率 < real（配对 McNemar 单侧）"，未写点比较与
          显著性哪个为准 ⇒ 本脚本**两个口径都报**（`ii_point` / `ii_sig`），
          判定按两者同时满足记"强成立"、仅点比较满足记"弱成立"，并在报告中显式标注。
    (iii) net(gated) > net(real)（全体净效应 pp）
    ⇒ 三条同时满足才算"改进成立"。
  次判据：gated_betastar vs gated_margin——谁门开比例更低且不损救回。
  诊断（必报）：每样本 gate_stats.gated_frac 分布、按子集（KW/KC/DK）的门开率。

  判停（预注册）：(ii) 不成立 ⇒ 门控只等比缩小规模、无净增益 ⇒ 门控族关闭；
                  (i) 不成立 ⇒ 门开着也救不回 ⇒ 回到"通用扰动"结论。

用法：
    python experiments/lin_theory/analyze_gated_arms.py \
        experiments/outputs/tldc_gated/tldc_controls_123_real-gated_*_tau0.2.json \
        [--betas 0.2] [--output_dir experiments/outputs/tldc_gated/judge_tau0.2]

  ⚠️ 多文件（多 seed）自动按 seed 前缀复合键 pool；同 tag 重复输入会报警
     （与 analyze_control_arms.py 同一约定）。
"""

import argparse
import json
import math
from pathlib import Path

from analyze_transition_matrix import (  # 复用同一套精确统计与逐格口径
    clopper_pearson,
    fisher_greater,
    mcnemar_exact,
    transition,
)

GATED_ARMS = ("gated_margin", "gated_betastar", "gated_damp")
BETA_PRIMARY_DEFAULT = 0.20


def mcnemar_one_sided_less(d, e):
    """单侧 McNemar：H1 = 不一致对偏向 d 侧（e 更少）。d=仅 real 破坏，e=仅 gated 破坏。

    p = P(X ≤ e)，X ~ Binom(d+e, 0.5)。d+e=0 时返回 1.0（无信息）。
    """
    n = d + e
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) * 0.5 ** n for i in range(0, e + 1))
    return min(1.0, tail)


def load_files(paths):
    """→ (pooled, metas)；pooled[(arm, beta)] = {seed:sid → (subset, base, final)}。

    额外收集门控诊断：gates[(arm, beta)][seed:sid] = gate_stats 字典。
    """
    pooled, gates, metas, tags = {}, {}, [], {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        meta = d.get("meta", {})
        cfg = {}
        if isinstance(d.get("report"), dict) and isinstance(d["report"].get("config"), dict):
            cfg = d["report"]["config"]
        elif isinstance(meta.get("config"), dict):
            cfg = meta["config"]
        seed = cfg.get("seed_test")
        tag = f"s{seed}" if seed is not None else Path(p).stem
        metas.append({"file": Path(p).name, "seed": seed, "gate_tau": cfg.get("gate_tau")})
        if tag in tags:
            print(f"  ⚠️ [pool 警告] {Path(p).name} 与前一个输入同 tag={tag}"
                  f"（前一个：{tags[tag]}）⇒ 同 seed 重复输入会造成样本覆盖，请确认是否误传")
        tags[tag] = Path(p).name

        for sid, v in d["samples"].items():
            for k in v:
                if not k.startswith("correct_"):
                    continue
                arm, _, b = k[len("correct_"):].rpartition("_beta")
                bucket = pooled.setdefault((arm, float(b)), {})
                if v.get("baseline_correct") is not None:
                    bucket[f"{tag}:{sid}"] = (v["subset"], bool(v["baseline_correct"]), bool(v[k]))
                gs = (v.get("gate_stats") or {}).get(f"{arm}_beta{float(b):g}")
                if gs is not None:
                    gates.setdefault((arm, float(b)), {})[f"{tag}:{sid}"] = gs
    return pooled, gates, metas


def paired_events(real_bucket, ctrl_bucket, subset, base_value):
    """在指定子集、指定基线结果上配对比较（按 key 取交集，防错位）。

    返回 (b_or_d, c_or_e, n)：b = 仅 real 发生事件，c = 仅对照发生事件，
    事件定义为"结果正确"（base_value=False 时 = 救回）或"结果错误"（base_value=True 时 = 破坏）。
    """
    shared = set(real_bucket) & set(ctrl_bucket)
    b = c = n = 0
    for sid in shared:
        sub, base, real_after = real_bucket[sid]
        if sub != subset or bool(base) != base_value:
            continue
        n += 1
        ctrl_after = ctrl_bucket[sid][2]
        real_ok = bool(real_after)
        ctrl_ok = bool(ctrl_after)
        if base_value:      # 破坏事件：正确 → 错误
            if (not real_ok) and ctrl_ok:
                b += 1      # 仅 real 破坏
            elif (not ctrl_ok) and real_ok:
                c += 1      # 仅 gated 破坏
        else:               # 救回事件：错误 → 正确
            if real_ok and not ctrl_ok:
                b += 1
            elif ctrl_ok and not real_ok:
                c += 1
    return b, c, n


def gate_diagnostics(gates, arm, beta):
    """门开比例：整体分位 + 按子集样本级"至少开一次门"比例。"""
    rows = gates.get((arm, beta))
    if not rows:
        return None
    fracs, by_sub = [], {}
    for sid, gs in rows.items():
        f = gs.get("gated_frac")
        if f is None:
            continue
        fracs.append(f)
        sub = sid  # 子集由调用方另传；此处仅按样本聚合
    if not fracs:
        return None
    fracs_sorted = sorted(fracs)
    n = len(fracs_sorted)
    return {
        "n_samples": n,
        "mean_frac": sum(fracs_sorted) / n,
        "median_frac": fracs_sorted[n // 2],
        "min_frac": fracs_sorted[0],
        "max_frac": fracs_sorted[-1],
        "any_open_frac": sum(1 for f in fracs_sorted if f > 0) / n,
        "all_open_frac": sum(1 for f in fracs_sorted if f >= 1.0) / n,
        "by_subset": by_sub,
    }


def gate_open_by_subset(gates, pooled, arm, beta):
    """按子集统计"该样本整段生成里至少开过一次门"的比例（T2 预测 KW≈56%、KC≈37%）。"""
    rows = gates.get((arm, beta))
    bucket = pooled.get((arm, beta))
    if not rows or not bucket:
        return None
    agg = {}
    for sid, gs in rows.items():
        if sid not in bucket:
            continue
        sub = bucket[sid][0]
        f = gs.get("gated_frac")
        if f is None:
            continue
        a = agg.setdefault(sub, {"n": 0, "any_open": 0, "mean_frac_sum": 0.0})
        a["n"] += 1
        a["any_open"] += int(f > 0)
        a["mean_frac_sum"] += f
    for a in agg.values():
        a["any_open_frac"] = a["any_open"] / a["n"] if a["n"] else None
        a["mean_gated_frac"] = a["mean_frac_sum"] / a["n"] if a["n"] else None
    return agg


def main():
    ap = argparse.ArgumentParser(description="T3 门控族判读（三条预注册判据）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--betas", nargs="+", type=float, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    pooled, gates, metas = load_files(args.paths)
    if not pooled:
        raise SystemExit("未从输入文件中解析到任何 correct_<arm>_beta<β> 字段")

    arms = sorted({a for a, _ in pooled})
    betas = sorted({b for _, b in pooled})
    if args.betas:
        betas = [b for b in betas if b in args.betas]
    gated_present = [a for a in arms if a in GATED_ARMS]
    if not gated_present:
        raise SystemExit(f"输入中无门控臂（可选 {GATED_ARMS}）——本脚本只判 T3")
    if "real" not in arms:
        raise SystemExit("输入中无 real 臂——三条判据均需 real 作同批基准，配对检验不可用")

    beta_main = BETA_PRIMARY_DEFAULT if BETA_PRIMARY_DEFAULT in betas else max(betas)
    taus = sorted({m["gate_tau"] for m in metas if m.get("gate_tau") is not None})

    md = ["# T3 门控族判读报告（预注册判据，placebo-control-protocol.md §9）", ""]
    md.append(f"- 输入：{', '.join(m['file'] for m in metas)}")
    md.append(f"- 臂：{arms}；β：{betas}；主判据档 β={beta_main}；τ：{taus or '未记录'}")
    md.append(f"- real 基准样本数：{len(pooled.get(('real', beta_main), {}))}")
    md.append("")

    # ── 各臂总表 ──
    md += [f"## 各臂逐格（β={beta_main}）", "",
           "| 臂 | n | KW 救回 (CP95) | KW/DK | Fisher p | KC 破坏 | 净 pp | McNemar p |",
           "|---|---|---|---|---|---|---|---|"]
    stats = {}
    for arm in arms:
        bucket = pooled.get((arm, beta_main))
        if not bucket:
            continue
        t = transition(list(bucket.values()))
        ps = t["per_subset"]
        kw, kc, dk = ps.get("know_wrong", {}), ps.get("know_correct", {}), ps.get("dont_know", {})
        ratio = fp = None
        if kw.get("base_wrong") and dk.get("base_wrong") and dk.get("rescue_rate"):
            ratio = kw["rescue_rate"] / dk["rescue_rate"]
            fp = fisher_greater(kw["rescue"], kw["base_wrong"] - kw["rescue"],
                                dk["rescue"], dk["base_wrong"] - dk["rescue"])
        ci = clopper_pearson(kw["rescue"], kw["base_wrong"]) if kw.get("base_wrong") else None
        stats[arm] = {"t": t, "ratio": ratio, "fisher": fp, "ci": ci}
        md.append(
            f"| {arm} | {t['n']} | {kw.get('rescue')}/{kw.get('base_wrong')} "
            f"({(kw.get('rescue_rate') or 0)*100:.1f}%"
            f"{'' if ci is None else f', CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]'}) | "
            f"{'—' if ratio is None else f'{ratio:.2f}×'} | {'—' if fp is None else f'{fp:.4f}'} | "
            f"{kc.get('break')}/{kc.get('base_right')} ({(kc.get('break_rate') or 0)*100:.1f}%) | "
            f"{t['net_pp']:+.1f} | {t['mcnemar_p']:.4f} |"
        )
    md.append("")

    # ── 三条预注册判据 ──
    real_t = stats.get("real", {}).get("t")
    md += [f"## 主判据（β={beta_main}，gated vs real 配对）", "",
           "| 门控臂 | (i) KW 救回 b/c, p | (i) 成立 | (ii) KC 破坏 d/e, 单侧 p "
           "| (ii) 点比较 | (ii) 显著 | (iii) net real→gated | (iii) 成立 | 判定 |",
           "|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for arm in gated_present:
        if (arm, beta_main) not in pooled:
            continue
        # (i) KW 救回：b=仅 real 救回，c=仅 gated 救回
        b, c, n_kw = paired_events(pooled[("real", beta_main)], pooled[(arm, beta_main)],
                                   "know_wrong", False)
        p_i = mcnemar_exact(b, c)
        i_ok = not (b > c and p_i < 0.05)          # 不显著低于 real
        # (ii) KC 破坏：d=仅 real 破坏，e=仅 gated 破坏
        d_, e_, n_kc = paired_events(pooled[("real", beta_main)], pooled[(arm, beta_main)],
                                     "know_correct", True)
        p_ii = mcnemar_one_sided_less(d_, e_)
        real_brk = real_t["per_subset"].get("know_correct", {}).get("break_rate")
        g_brk = stats[arm]["t"]["per_subset"].get("know_correct", {}).get("break_rate")
        ii_point = (real_brk is not None and g_brk is not None and g_brk < real_brk)
        ii_sig = (e_ < d_ and p_ii < 0.05)
        # (iii) 净效应
        net_real, net_g = real_t["net_pp"], stats[arm]["t"]["net_pp"]
        iii_ok = net_g > net_real
        all_ok = i_ok and ii_point and iii_ok
        # 退化护栏（2026-09-22 冒烟暴露）：KW 两臂均无救回事件时，
        # (i) 与 (ii) 都会"自动通过"，net 改善全部来自破坏减少 ⇒ 不足以称"改进"。
        rescue_real = real_t["per_subset"].get("know_wrong", {}).get("rescue", 0)
        rescue_gated = stats[arm]["t"]["per_subset"].get("know_wrong", {}).get("rescue", 0)
        degen = []
        if b + c == 0:
            degen.append("(i) 无救回事件差异 b+c=0")
        if d_ + e_ == 0:
            degen.append("(ii) 无破坏事件差异 d+e=0")
        degenerate = bool(degen) and (rescue_real + rescue_gated == 0)
        if all_ok and degenerate:
            verdict_txt = "**不可判定（退化：两臂均无救回事件）**"
        elif all_ok:
            verdict_txt = "**改进成立**"
        else:
            verdict_txt = "**不成立**"
        verdict[arm] = {"i_ok": i_ok, "i_b": b, "i_c": c, "i_p": p_i, "i_n_kw": n_kw,
                        "ii_point": ii_point, "ii_sig": ii_sig, "ii_d": d_, "ii_e": e_,
                        "ii_p": p_ii, "ii_n_kc": n_kc,
                        "iii_ok": iii_ok, "net_real": net_real, "net_gated": net_g,
                        "rescue_real": rescue_real, "rescue_gated": rescue_gated,
                        "degenerate": degenerate, "degenerate_reasons": degen,
                        "improvement": all_ok and not degenerate}
        md.append(
            f"| {arm} | b={b}, c={c}, p={p_i:.4f} | {'✅' if i_ok else '❌'} "
            f"| d={d_}, e={e_}, p={p_ii:.4f} | {'✅' if ii_point else '❌'} "
            f"| {'✅' if ii_sig else '❌'} "
            f"| {net_real:+.1f} → {net_g:+.1f} | {'✅' if iii_ok else '❌'} "
            f"| {verdict_txt} |"
        )
        if degenerate:
            md.append(f"| ↳ | 退化原因：{'; '.join(degen)}（real 救回 {rescue_real}、"
                      f"gated 救回 {rescue_gated}）⇒ 三条判据在无救回事件时不可判定 | | | | | | |")
    md.append("")
    md += ["> (i) b=仅 real 救回、c=仅 gated 救回（KW 基线错样本）；不成立 ⇔ b>c 且双侧 p<0.05。",
           "> (ii) d=仅 real 破坏、e=仅 gated 破坏（KC 基线对样本）；单侧 p=P(X≤e)。",
           "> ⚠️ 预注册原文未写 (ii) 以点比较还是显著性为准 ⇒ 两个口径并列报，",
           ">    `点比较 ✅ + 显著 ❌` 只能记「弱成立」，论文中必须写明口径。",
           "> (iii) net = (救回 − 破坏)/n × 100pp，全体样本。",
           "> ⚠️ **退化护栏**：若 real 与 gated **两臂均无救回事件**，(i)(ii) 会自动通过而 net 改善",
           ">   全部来自破坏减少 ⇒ 判为「不可判定（退化）」，**不得**据此主张改进（2026-09-22 冒烟暴露）。",
           "> **判停（预注册）**：(ii) 不成立 ⇒ 门控只等比缩小规模、无净增益 ⇒ 门控族关闭；",
           ">   (i) 不成立 ⇒ 门开着也救不回 ⇒ 回到「通用扰动」结论。", ""]

    # ── 门控诊断（必报） ──
    md += ["## 门控诊断（必报）", "",
           "| 臂 | 样本数 | 平均 gated_frac | 中位 | 至少开一次门 | 全程开门 |",
           "|---|---|---|---|---|---|"]
    diags = {}
    for arm in gated_present:
        gd = gate_diagnostics(gates, arm, beta_main)
        if gd is None:
            continue
        diags[arm] = gd
        md.append(f"| {arm} | {gd['n_samples']} | {gd['mean_frac']:.3f} | {gd['median_frac']:.3f} "
                  f"| {gd['any_open_frac']*100:.1f}% | {gd['all_open_frac']*100:.1f}% |")
    md.append("")
    md += ["### 按子集门开率（T2 预测：τ=0.2 时 KW≈56%、KC≈37%）", "",
           "| 臂 | 子集 | n | 至少开一次门 | 平均 gated_frac |", "|---|---|---|---|---|"]
    subs_diag = {}
    for arm in gated_present:
        agg = gate_open_by_subset(gates, pooled, arm, beta_main)
        if not agg:
            continue
        subs_diag[arm] = agg
        for sub in ("know_wrong", "know_correct", "dont_know"):
            a = agg.get(sub)
            if not a:
                continue
            md.append(f"| {arm} | {sub} | {a['n']} | {(a['any_open_frac'] or 0)*100:.1f}% "
                      f"| {a['mean_gated_frac']:.3f} |")
    md.append("")

    # ── 次判据：省不省 ──
    if len([a for a in gated_present if a != "gated_damp"]) >= 2:
        md += ["## 次判据（谁更省且不损救回）", ""]
        for arm in gated_present:
            if arm in diags and arm in verdict:
                md.append(f"- `{arm}`：门开比例（平均 gated_frac）{diags[arm]['mean_frac']:.3f}、"
                          f"KW 救回 b/c={verdict[arm]['i_b']}/{verdict[arm]['i_c']}、"
                          f"KC 破坏 d/e={verdict[arm]['ii_d']}/{verdict[arm]['ii_e']}")
        md.append("")

    print("\n".join(md))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gated_arms_report.md").write_text("\n".join(md), encoding="utf-8")
        (out / "gated_arms_report.json").write_text(json.dumps(
            {"config": {"betas": betas, "beta_main": beta_main, "taus": taus, "arms": arms},
             "arm_stats": {a: {"net_pp": s["t"]["net_pp"], "transition": s["t"],
                               "ratio_kw_dk": s["ratio"], "fisher_p": s["fisher"], "kw_ci": s["ci"]}
                           for a, s in stats.items()},
             "verdict": verdict, "gate_diagnostics": diags, "gate_open_by_subset": subs_diag},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[saved] {out / 'gated_arms_report.md'}")


if __name__ == "__main__":
    main()
