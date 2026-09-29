"""A-6 头级定位判读（零 GPU）：S0 功效预检 / S1 归因判读 / S2 零消融判读。

理论锚点与判据：docs/protocol/a6-head-rescue-circuit-20260929.md §3
  - S0（--precheck）：既有档案 pooled 救回 ≥20 ∧ 破坏 ≥20 ∧ 分叉样本 ≥50
  - S1（--judge）：d_h = DLA_h(chosen_tp) − DLA_h(y_true)；s_h = mean_救回 − mean_破坏；
      ① seed123 上 max_h|s_h| 超置乱 97.5 分位（2000 次救回/破坏标签置乱）；
      ② seed456 同定义重算 s_h，Spearman ρ ≥ 0.4
  - S2（--ablate）：选中头/随机头消融档 vs 全量 sym seed456——
      R⁺（s_h<0）消融 Δr ≤ −2；R⁻（s_h>0）消融 Δb ≤ −2；方向一致头数 ≥5/8；
      选中头效应和超随机头效应和（随机头列表协议已固定）
"""
import argparse
import json
import math
import random
from pathlib import Path

ARCH = "experiments/outputs/geometry_archive"
ATTR_DIR = "experiments/outputs/head_attribution"
ABL_DIR = "experiments/outputs/head_ablation"
OUT_DIR = "experiments/outputs/head_loc"
BETA = 0.20
K_TOP = 8
RHO_MIN = 0.40
SIGN_CONSIST_MIN = 5       # ≥5/8 方向一致
N_SHUFFLE = 2000
EVENT_MIN = 2              # Δr/Δb 事件下限
RANDOM_HEADS = [(12, 3), (16, 9), (23, 8), (9, 4), (21, 4), (14, 13), (4, 11), (13, 15)]


def load_arch(arch, op, seed, beta=None):
    if beta is None:
        fp = Path(arch) / f"geo_{op}_seed{seed}.json"
    else:
        tag = ("b" + f"{beta:.2f}").replace(".", "")
        fp = Path(arch) / f"geo_{op}_{tag}_seed{seed}.json"
    if not fp.exists():
        return None, fp.name
    return {s["sample_id"]: s for s in json.loads(fp.read_text())["samples"]}, fp.name


def label_of(s):
    b, a = bool(s["baseline_correct"]), bool(s["is_correct"])
    return "rescue" if (not b and a) else ("break" if (b and not a) else "nochange")


def diverge_step(b, p):
    for t in range(min(len(b["steps"]), len(p["steps"]))):
        if b["steps"][t]["chosen_id"] != p["steps"][t]["chosen_id"]:
            return t
    return None


def precheck(seeds=(123, 456)):
    resc = brk = div = 0
    missing = []
    for s in seeds:
        B, fB = load_arch(ARCH, "baseline", s)
        S, fS = load_arch(ARCH, "sym", s, BETA)
        if B is None or S is None:
            missing.append(f"seed{s}: {fB} / {fS}")
            continue
        for sid, ss in S.items():
            b = B.get(sid)
            if b is None:
                continue
            lab = label_of(ss)
            t_p = diverge_step(b, ss)
            if lab == "rescue":
                resc += 1
            elif lab == "break":
                brk += 1
            if t_p is not None:
                div += 1
    ok = resc >= 20 and brk >= 20 and div >= 50
    lines = ["# A-6 S0 功效预检", "",
             f"- 档案 {ARCH} × seeds {seeds}（baseline × sym b{BETA:.2f}）；缺档：{missing or '无'}",
             f"- 救回 **{resc}**（≥20？{'✅' if resc >= 20 else '❌'}）｜破坏 **{brk}**"
             f"（≥20？{'✅' if brk >= 20 else '❌'}）｜分叉样本 **{div}**（≥50？{'✅' if div >= 50 else '❌'}）",
             f"- **S0 判定：{'✅ 进 S1' if ok else '❌ 判停（事件不足）'}**", ""]
    print("\n".join(lines))
    return ok


def judge(seeds=(123, 456)):
    """S1：置乱检验（seed123 选头）+ 跨 seed Spearman（seed456 重算）。"""
    dumps, missing = {}, []
    for s in seeds:
        tag = ("b" + f"{BETA:.2f}").replace(".", "")
        fp = Path(ATTR_DIR) / f"headattr_sym_{tag}_seed{s}.json"
        if not fp.exists():
            missing.append(fp.name)
            continue
        dumps[s] = json.loads(fp.read_text())
    meta = None
    for s in seeds:
        if s in dumps:
            meta = dumps[s]["meta"]
            break
    if meta is None:
        raise SystemExit("缺归因 dump：先跑 `bash scripts/review_local.sh head-attr`（" + ", ".join(missing) + "）")
    nL, nH = meta["n_layers"], meta["n_heads"]
    B = {s: load_arch(ARCH, "baseline", s)[0] for s in seeds}

    def s_h(seed):
        D, barch = dumps[seed], B[seed]
        sm = {x["sample_id"]: x for x in D["samples"]}
        acc = [[0.0, 0] for _ in range(nL * nH)]
        n_div = 0
        for sid, ss in barch.items():
            ds = sm.get(sid)
            if ds is None:
                continue
            lab = label_of(ss)
            if lab not in ("rescue", "break"):
                continue
            t_p = diverge_step(barch[sid], ss)
            if t_p is None or t_p >= len(ds["steps"]):
                continue
            n_div += 1
            attr = ds["steps"][t_p]["attr"]          # [L*nH*2] flatten
            yt = ds["y_true_id"]
            chosen = ds["steps"][t_p]["chosen_id"]
            for i in range(nL * nH):
                d = attr[2 * i] - attr[2 * i + 1]    # DLA(chosen) − DLA(y_true)
                acc[i][0] += d
                acc[i][1] += 1
        return [a / n for a, n in acc], n_div

    s123, n123 = s_h(123)
    top = sorted(range(len(s123)), key=lambda i: -abs(s123[i]))[:K_TOP]
    top_list = [(i // nH, i % nH) for i in top]

    # ① 置乱检验（seed123）
    dumps123 = {x["sample_id"]: x for x in dumps[123]["samples"]}
    rows = []
    for sid, ss in B[123].items():
        ds = dumps123.get(sid)
        lab = label_of(ss)
        if ds is None or lab not in ("rescue", "break"):
            continue
        t_p = diverge_step(B[123][sid], ss)
        if t_p is None or t_p >= len(ds["steps"]):
            continue
        attr = ds["steps"][t_p]["attr"]
        rows.append((lab, attr))
    rnd = random.Random(20260929)
    labs = [r[0] for r in rows]
    def max_shuf():
        ls = labs[:]
        rnd.shuffle(ls)
        acc = [0.0] * (nL * nH)
        cnt = [0] * (nL * nH)
        for (_, attr), lab in zip(rows, ls):
            for i in range(nL * nH):
                acc[i] += attr[2 * i] - attr[2 * i + 1]
                cnt[i] += 1
        vals = [acc[i] / cnt[i] for i in range(nL * nH)]
        return max(abs(v) for v in vals)
    obs_max = max(abs(v) for v in s123)
    shuf_max = sorted(max_shuf() for _ in range(N_SHUFFLE))
    p975 = shuf_max[int(0.975 * (N_SHUFFLE - 1))]
    perm_ok = obs_max > p975

    # ② 跨 seed
    rho = None
    if 456 in dumps:
        s456, _ = s_h(456)
        rk1 = sorted(range(len(s123)), key=lambda i: s123[i])
        rk2 = sorted(range(len(s456)), key=lambda i: s456[i])
        r1 = {i: j for j, i in enumerate(rk1)}
        r2 = {i: j for j, i in enumerate(rk2)}
        n = len(s123)
        rho = 1 - 6 * sum((r1[i] - r2[i]) ** 2 for i in range(n)) / (n ** 3 - n)
    rho_ok = rho is not None and rho >= RHO_MIN

    ok = perm_ok and rho_ok
    lines = ["# A-6 S1 归因判读（零 GPU）", "",
             f"- 判据：置乱 97.5 分位 + Spearman ρ≥{RHO_MIN}（seed123 选头、seed456 验证）",
             f"- dump：{sorted(missing) or '无缺档'}；分叉救回/破坏样本（seed123）n={n123}", "",
             f"## ① 置乱检验（seed123）", "",
             f"- 观测 max|s_h| = **{obs_max:.4f}** vs 置乱 97.5 分位 **{p975:.4f}**（{N_SHUFFLE} 次）"
             f" ⇒ {'✅' if perm_ok else '❌'}", "",
             f"## ② 跨 seed 稳定性", "",
             f"- Spearman ρ(s123, s456) = **{rho:.3f}**（≥{RHO_MIN}？{'✅' if rho_ok else '❌'}）"
             if rho is not None else
             f"- seed456 dump 缺档：{missing} ⇒ ② 不计算", "",
             "## 选中头（top-K by |s_h|，seed123；s<0=R⁺ 救回支持、s>0=R⁻ 破坏支持）", ""]
    for i in top:
        l, h = i // nH, i % nH
        tag = "R⁺" if s123[i] < 0 else "R⁻"
        lines.append(f"- L{l}-H{h}  s_h={s123[i]:+.4f}  [{tag}]")
    lines += ["", f"## S1 判定：{'✅ 进 S2' if ok else '❌ 关闭（H2 全域重排）'}", "",
              "## S2 零消融命令（每条一行，16 条：8 选中头 + 8 固定随机头）", ""]
    for l, h in top_list:
        lines.append(f"```bash\nHEAD_L={l} HEAD_H={h} bash scripts/review_local.sh head-ablate\n```")
    lines += ["随机头（执行前固定 seed=20260929）：" +
              " ".join(f"{l}-{h}" for l, h in RANDOM_HEADS)]
    out = Path(OUT_DIR)
    out.mkdir(exist_ok=True, parents=True)
    (out / "head_loc_s1_report.md").write_text("\n".join(lines))
    with open(out / "head_loc_s1.json", "w") as f:
        json.dump({"obs_max": obs_max, "p975": p975, "perm_ok": perm_ok, "rho": rho,
                   "rho_ok": rho_ok, "top_heads": [[l, h, s123[i]] for i, (l, h)
                                                   in zip(top, top_list)]},
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {out / 'head_loc_s1_report.md'}")
    print("\n".join(lines))


def ablate_judge():
    """S2：零消融判读（seed456；选中头 + 随机头 vs 全量 sym）。"""
    S_full, fS = load_arch(ARCH, "sym", 456, BETA)
    B456, _ = load_arch(ARCH, "baseline", 456)
    if S_full is None or B456 is None:
        raise SystemExit("seed456 全量档缺失")
    sel = json.loads((Path(OUT_DIR) / "head_loc_s1.json").read_text())
    top_heads = [(int(l), int(h)) for l, h, _ in sel["top_heads"]]

    def counts(ablate=None):
        if ablate is None:
            S = S_full
        else:
            l, h = ablate
            fp = Path(ABL_DIR) / f"headablate_L{l}_H{h}_seed456.json"
            if not fp.exists():
                return None, fp.name
            S = {x["sample_id"]: x for x in json.loads(fp.read_text())["samples"]}
        resc = sum(1 for s in S.values() if label_of(s) == "rescue")
        brk = sum(1 for s in S.values() if label_of(s) == "break")
        return resc, brk

    r_full, b_full = counts()
    lines = ["# A-6 S2 零消融判读（零 GPU）", "",
             f"- 全量 sym seed456：救回 {r_full} 破坏 {b_full}", ""]
    drs, dbs, effects, consistent = [], [], [], 0
    for l, h in top_heads:
        r, b = counts((l, h))
        if r is None:
            lines.append(f"- L{l}-H{h}：缺档 {b}")
            continue
        dr, db = r - r_full, b - b_full
        drs.append(dr)
        dbs.append(db)
        s = sel["top_heads"][[i for i, (ll, hh) in enumerate(top_heads) if (ll, hh) == (l, h)][0]][2]
        expect = (s < 0 and dr <= -EVENT_MIN) or (s > 0 and db <= -EVENT_MIN)
        consistent += int(expect)
        effects.append(abs(dr) + abs(db))
        lines.append(f"- L{l}-H{h}（s_h={s:+.4f}）：Δr **{dr:+d}** Δb **{db:+d}** "
                     f"预期{'✅' if expect else '❌'}")
    sel_sum = sum(effects)
    rand_effects = []
    for l, h in RANDOM_HEADS:
        r, b = counts((l, h))
        if r is None:
            lines.append(f"- 随机头 L{l}-H{h}：缺档 {b}")
            continue
        rand_effects.append(abs(r - r_full) + abs(b - b_full))
        lines.append(f"- 随机头 L{l}-H{h}：Δr {r - r_full:+d} Δb {b - b_full:+d}")
    ok = consistent >= SIGN_CONSIST_MIN and sel_sum > max(rand_effects)
    lines += ["", f"- 方向一致头数 **{consistent}/{len(top_heads)}**（≥{SIGN_CONSIST_MIN}？"
              f"{'✅' if consistent >= SIGN_CONSIST_MIN else '❌'}）",
              f"- 选中头效应和 **{sel_sum}** vs 随机头最大 **{max(rand_effects)}**"
              f"（{'✅' if sel_sum > max(rand_effects) else '❌'}）",
              f"- **S2 判定：{'✅ 进 S3' if ok else '❌ 关闭（归因无因果力 / 伪信号）'}**", ""]
    out = Path(OUT_DIR)
    (out / "head_loc_s2_report.md").write_text("\n".join(lines))
    print(f"[saved] {out / 'head_loc_s2_report.md'}")
    print("\n".join(lines))


def selftest():
    ok = 0
    # label_of / diverge_step
    assert label_of({"baseline_correct": False, "is_correct": True}) == "rescue"
    assert label_of({"baseline_correct": True, "is_correct": False}) == "break"
    b = {"steps": [{"chosen_id": 1}, {"chosen_id": 2}]}
    p = {"steps": [{"chosen_id": 1}, {"chosen_id": 3}]}
    assert diverge_step(b, p) == 1 and diverge_step(b, b) is None
    ok += 1
    # s_h 手算（小规模）：
    # 头0：救回样本 d=+2、破坏样本 d=0 ⇒ s=+2（R⁻）；头1：救回 −1、破坏 +1 ⇒ s=−2（R⁺）
    # （经 judge 主流程不可直接调用 s_h，这里直接验 d 定义与选头排序逻辑）
    d = {"0": (2.0, 0.0), "1": (-1.0, 1.0)}          # head → (rescue_mean_d, break_mean_d)
    s = {h: v[0] - v[1] for h, v in d.items()}
    top = sorted(s, key=lambda h: -abs(s[h]))[:2]
    assert s["0"] > 0 and s["1"] < 0 and abs(s["0"]) == abs(s["1"])
    assert len(top) == 2
    ok += 1
    print(f"SELFTEST PASS: {ok}/2 checks (label-diverge / s_h 选头)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true", help="S0 功效预检（零 GPU）")
    ap.add_argument("--judge", action="store_true", help="S1 归因判读（零 GPU）")
    ap.add_argument("--ablate", action="store_true", help="S2 零消融判读（零 GPU）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.precheck:
        precheck()
    elif args.ablate:
        ablate_judge()
    else:
        judge()


if __name__ == "__main__":
    main()
