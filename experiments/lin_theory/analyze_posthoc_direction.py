"""§5.8 事后信号"方向可分性"判读（零 GPU）——干预后档案 × 干预前档案。

预注册：docs/protocol/review-runbook-20260921.md §5.8（登记于档案落盘之前）。
主判据：AUROC(救回 vs 破坏 | 事后特征) ≥ 0.70（单 seed CP95 下界 >0.5；双 seed 同向）。
附加约束：该特征须**不是干预前信息的重述**（用 baseline 同索引步做对照）。

⚠️ **档案语义（2026-09-22 实测确认，勿误读）**：每一档每步存的 `final_top10/early_top10/geo`
是**干预前**该步模型自身的读出（分叉步处与 baseline 完全相同 273/273），**只有 `chosen_id` 是干预后的**。
因此：
  · t+1 步的特征（轨迹已分叉 ⇒ 新前缀上的读出）= **合法的事后/一步前瞻信号**；
  · t   步的特征（与 baseline 同值）= 干预前特征，不得标为"事后"。

用法：
  python experiments/lin_theory/analyze_posthoc_direction.py --seeds 123 456
"""

import argparse
import json
import math
import random
import statistics as st
from collections import Counter
from pathlib import Path

ARCH = "experiments/outputs/geometry_archive"


def load(seed, op):
    f = f"geo_{op}_seed{seed}.json" if op == "baseline" else f"geo_{op}_b020_seed{seed}.json"
    p = Path(ARCH) / f
    if not p.exists():
        return None
    return {s["sample_id"]: s for s in json.loads(p.read_text())["samples"]}


def diverge_step(base, sym):
    for t in range(min(len(base["steps"]), len(sym["steps"]))):
        if base["steps"][t]["chosen_id"] != sym["steps"][t]["chosen_id"]:
            return t
    return None


def agree(step):
    return int(step["early_top10"][0][0] == step["final_top10"][0][0])


def margin(step):
    f = step["final_top10"]
    return f[0][2] - f[1][2]


def collect(seeds):
    rows = []
    for seed in seeds:
        B, S = load(seed, "baseline"), load(seed, "sym")
        if B is None or S is None:
            print(f"  [skip] seed{seed} 缺档案")
            continue
        for sid, s in S.items():
            b = B.get(sid)
            if b is None:
                continue
            base_ok, post_ok = bool(s["baseline_correct"]), bool(s["is_correct"])
            lab = ("rescue" if (not base_ok and post_ok) else
                   "break" if (base_ok and not post_ok) else "nochange")
            t = diverge_step(b, s)
            r = {"seed": seed, "subset": s["subset"], "label": lab, "t": t,
                 "sid": f"{seed}:{sid}"}
            if t is not None and t + 1 < len(s["steps"]):
                nxt_s = s["steps"][t + 1]
                # 合法事后特征（新前缀上、下一步即可算出）
                r["post_beta"] = nxt_s["geo"]["beta_star_min"]
                r["post_Rsize"] = nxt_s["geo"]["R_size"]
                r["post_margin"] = margin(nxt_s)
                r["post_agree"] = agree(nxt_s)
                # 干预前对照（baseline 同索引步；仅用于「是否旧信息」检验）
                # ⚠️ baseline 轨迹可能比 sym 短（EOS 位置不同）⇒ 需越界保护
                if t + 1 < len(b["steps"]):
                    nxt_b = b["steps"][t + 1]
                    r["pre_beta"] = nxt_b["geo"]["beta_star_min"]
                    r["pre_margin"] = margin(nxt_b)
            rows.append(r)
    return rows


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


def spearman(x, y):
    rx = {i: r for r, i in enumerate(sorted(range(len(x)), key=lambda i: x[i]))}
    ry = {i: r for r, i in enumerate(sorted(range(len(y)), key=lambda i: y[i]))}
    a = [rx[i] for i in range(len(x))]
    b = [ry[i] for i in range(len(y))]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["123", "456"])
    ap.add_argument("--out", default="experiments/outputs/posthoc_direction_1p7b")
    args = ap.parse_args()

    rows = collect(args.seeds)
    resc = [r for r in rows if r["label"] == "rescue"]
    brk = [r for r in rows if r["label"] == "break"]
    md = ["# §5.8 事后信号方向可分性（干预后档案 × 干预前档案，零 GPU）", "",
          f"- 档：`geo_sym_b020_seed{','.join(args.seeds)}.json` × `geo_baseline_seed{','.join(args.seeds)}.json`",
          f"- 样本 {len(rows)}｜救回 {len(resc)}｜破坏 {len(brk)}",
          "- 判据：AUROC(救回 vs 破坏 | 事后特征) ≥ 0.70，CP95 下界 >0.5，双 seed 同向", ""]

    # 分叉步分布
    md += ["## 分叉步分布（诊断）", "", "| 标签 | 未分叉 | 中位 t | 分布 |", "|---|---|---|---|"]
    for lab in ("rescue", "break", "nochange"):
        ts = [r["t"] for r in rows if r["label"] == lab and r["t"] is not None]
        none = sum(1 for r in rows if r["label"] == lab and r["t"] is None)
        dist = " ".join(f"{k}:{v}" for k, v in sorted(Counter(ts).items())[:8]) if ts else "—"
        md.append(f"| {lab} | {none} | {st.median(ts) if ts else '—'} | {dist} |")
    md.append("")

    # 主判据
    feats = [("post_beta", "干预后 t+1 步 β*_min", True),
             ("post_margin", "干预后 t+1 步 top-2 margin", True),
             ("post_Rsize", "干预后 t+1 步 R_size", True),
             ("post_agree", "干预后 t+1 步 early/final 一致", True),
             ("pre_beta", "【对照】干预前同索引步 β*_min", False),
             ("pre_margin", "【对照】干预前同索引步 margin", False)]
    md += ["## 主判据：方向可分性", "",
           "| 特征 | 事后性 | 中位(救回) | 中位(破坏) | AUROC(救回 vs 破坏) | CP95 | ≥0.70 |",
           "|---|---|---|---|---|---|---|"]
    verdict = {}
    for key, name, is_post in feats:
        p = [r[key] for r in resc if r.get(key) is not None]
        n = [r[key] for r in brk if r.get(key) is not None]
        a = auroc(p, n)
        lo, hi = boot_ci(p, n)
        ok = "✅" if (a == a and a >= 0.70 and lo > 0.5) else "❌"
        verdict[key] = {"auroc": a, "ci": [lo, hi], "n_post": len(p), "n_brk": len(n),
                        "median_rescue": st.median(p) if p else None,
                        "median_break": st.median(n) if n else None, "is_post": is_post}
        md.append(f"| `{key}` | {'事后 ✅' if is_post else '干预前'} | "
                  f"{st.median(p):.3f} | {st.median(n):.3f} | **{a:.3f}** | [{lo:.3f}, {hi:.3f}] | {ok} |")
    md.append("")

    # 三项对照
    best_key = max((k for k, v in verdict.items() if v["is_post"]),
                   key=lambda k: verdict[k]["auroc"] if verdict[k]["auroc"] == verdict[k]["auroc"] else -1)
    md += [f"## 三项对照（对最优事后特征 `{best_key}`）", ""]
    for seed in args.seeds:
        p = [r[best_key] for r in resc if r["seed"] == seed and r.get(best_key) is not None]
        n = [r[best_key] for r in brk if r["seed"] == seed and r.get(best_key) is not None]
        md.append(f"- **分 seed**：seed{seed} 救回 n={len(p)} / 破坏 n={len(n)} → AUROC **{auroc(p, n):.3f}**")
    for lo_, hi_, nm in ((0, 1, "t≤1"), (2, 3, "t=2–3"), (4, 99, "t≥4")):
        p = [r[best_key] for r in resc if r.get(best_key) is not None and lo_ <= r["t"] <= hi_]
        n = [r[best_key] for r in brk if r.get(best_key) is not None and lo_ <= r["t"] <= hi_]
        md.append(f"- **分层 t {nm}**：救回 {len(p)} / 破坏 {len(n)} → AUROC {auroc(p, n):.3f}")
    ts = [r["t"] for r in rows if r["t"] is not None and r.get(best_key) is not None]
    bs = [r[best_key] for r in rows if r["t"] is not None and r.get(best_key) is not None]
    md += [f"- **与 t 的 Spearman**：rho = {spearman(ts, bs):+.3f}（≈0 ⇒ 非步索引假象）",
           f"- **干预前对照**：同索引步 `{best_key.replace('post', 'pre')}` 的 AUROC = "
           f"{verdict[best_key.replace('post','pre')]['auroc']:.3f}（≈0.5 ⇒ 信息由干预产生）", ""]

    # 验证投影（近似）
    md += ["## 一步前瞻验证的净效应投影（近似）", "",
           "规则：干预后在 t+1 步读 β*_min，**≥θ 则保留翻转、<θ 则回退**（回退视为回到基线结果）。",
           "", "| θ | 保留的救回 | 保留的破坏 | 投影净事件 | 备注 |", "|---|---|---|---|---|"]
    proj = {}
    for th in (0.0, 0.2, 0.3, 0.5, 0.7, 0.9):
        rk = sum(1 for r in resc if r.get(best_key) is not None and r[best_key] >= th)
        bk = sum(1 for r in brk if r.get(best_key) is not None and r[best_key] >= th)
        proj[th] = {"rescue_kept": rk, "break_kept": bk, "net": rk - bk}
        md.append(f"| {th:.1f} | {rk}/{len(resc)} | {bk}/{len(brk)} | **{rk-bk:+d}** | "
                  f"{'—' if th else '无验证 = 现状'} |")
    md += ["", "> ⚠️ 投影假设「回退即回到基线结果」（单步干预设计下成立；档案是多步干预，故为近似估计）。"
               "真实验证臂必须实跑。", ""]

    text = "\n".join(md)
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posthoc_direction.md").write_text(text, encoding="utf-8")
    (out / "posthoc_direction.json").write_text(json.dumps(
        {"n": {"total": len(rows), "rescue": len(resc), "break": len(brk)},
         "verdict": verdict, "projection": proj}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out / 'posthoc_direction.md'}")


if __name__ == "__main__":
    main()
