"""方向二·CRG 承诺反转门控 S0 判读（零 GPU）。

理论锚点与判据：docs/protocol/crg-commitment-revision-gate-20260929.md §5
  - 承诺观测：δ_t = l₁(y_true) − l₁(a_t)（二元答案集 log-odds 承诺；档案 geo.yt.l1 / geo.l1_a）
  - 首承诺反转点 t*：δ 的符号相对上一步（非零符号）翻转的首个步
  - 首分叉步 t_p：sym 档与 baseline 档 chosen_id 首次不一致（既有口径）
  - 反转幅度 |Δ₀| = |geo.yt.l0 − geo.l0_a|（首反转步，参考层口径）
  - 事后黏性 = sym 档 t_p+1 步的 geo.beta_star_min（A1 同源信号）
  - 判定：A∧B∧C 全过 ⇒ 进 S1；A 过而 B/C 不过 ⇒ 退化为 A1、关闭独立立项；A 不过 ⇒ 关闭。
"""
import argparse
import json
import math
from pathlib import Path

ARCH_DEFAULT = "experiments/outputs/geometry_archive"
OUT_DEFAULT = "experiments/outputs/crg_gate"
FINAL_BETA = 0.20
COINCIDE_MIN = 0.80      # S0-A 重合率
AUROC_MAG_MIN = 0.60     # S0-B 反转幅度可分性
CI_LOSS_TOL = 0.05       # S0-C 事后黏性 AUROC 允许损失
RESCUE_KEEP_MIN = 0.80   # S0-C 救回保留率
SHARE_MAX = 0.70         # S0-C 步数占比（footprint 压缩 ≥30%）
EXIST_LO, EXIST_HI = 0.05, 0.60  # S0 仪器：反转点存在率区间


def load(arch, op, seed, beta=None):
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


def revision_step(sample):
    """承诺修订点：y_true 在 final top-10 候选短名单中的进出事件（c_t = I[yt ∈ top10_t]）。
    返回 (t*, |Δ₀|_t*)，t* 为首个 c_t ≠ c_{t−1} 的步（t≥1）；无进出 ⇒ (None, None)。

    仪器修正说明（2026-09-29，协议 §7）：原二元承诺 δ_t = l1(yt) − l1(a_t) 结构退化——
    a_t 为 l1 的 argmax ⇒ δ_t ≤ 0 恒成立（等号仅当 a_t == yt），符号翻转事件不存在
    ⇒ 改为「候选短名单进出」口径（T14 候选集承诺的可计算特例；A/B/C 阈值不变）。
    """
    yt = sample["y_true_id"]
    prev = None
    for t, st in enumerate(sample["steps"]):
        ids = [x[0] for x in st.get("final_top10", [])]
        c = 1 if yt in ids else 0
        if prev is not None and c != prev:
            mag = abs(st["geo"]["yt"]["l0"] - st["geo"]["l0_a"])
            return t, mag
        prev = c
    return None, None


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def collect(arch, seeds):
    rows = []
    meta = {"missing": []}
    for seed in seeds:
        B, fB = load(arch, "baseline", seed)
        S, fS = load(arch, "sym", seed, FINAL_BETA)
        if B is None or S is None:
            meta["missing"].append(f"seed{seed}: baseline={fB} sym={fS}")
            continue
        for sid, s in S.items():
            b = B.get(sid)
            if b is None:
                continue
            lab = label_of(s)
            t_p = diverge_step(b, s)
            t_s, mag = revision_step(b)
            post = None
            if t_p is not None and t_p + 1 < len(s["steps"]):
                post = s["steps"][t_p + 1]["geo"]["beta_star_min"]
            rows.append({"seed": seed, "sid": sid, "label": lab, "t_p": t_p,
                         "t_s": t_s, "mag": mag, "post": post,
                         "subset": s["subset"]})
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=ARCH_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    rows, meta = collect(args.arch, args.seeds)
    div = [r for r in rows if r["t_p"] is not None]
    flip = [r for r in div if r["t_s"] is not None]
    resc = [r for r in div if r["label"] == "rescue"]
    brk = [r for r in div if r["label"] == "break"]

    exist_rate = len(flip) / len(div) if div else float("nan")
    coincide = [r for r in flip if r["t_s"] == r["t_p"]]
    coinc_rate = len(coincide) / len(flip) if flip else float("nan")

    # B：反转幅度可分性（在反转样本上，救回 vs 破坏）
    mag_pos = [r["mag"] for r in flip if r["label"] == "rescue"]
    mag_neg = [r["mag"] for r in flip if r["label"] == "break"]
    b_auroc = auroc(mag_pos, mag_neg)

    # C：对齐子集（t*==t_p，事后黏性非缺失，救回/破坏）上的幅度中位分割
    aligned = [r for r in coincide if r["label"] in ("rescue", "break") and r["post"] is not None]
    overall = auroc([r["post"] for r in aligned if r["label"] == "rescue"],
                    [r["post"] for r in aligned if r["label"] == "break"])
    c_ok = None
    if aligned:
        mags = sorted(r["mag"] for r in aligned)
        med = mags[len(mags) // 2]
        high = [r for r in aligned if r["mag"] >= med]
        hi_auroc = auroc([r["post"] for r in high if r["label"] == "rescue"],
                         [r["post"] for r in high if r["label"] == "break"])
        keep_rate = (sum(1 for r in high if r["label"] == "rescue")
                     / max(1, sum(1 for r in aligned if r["label"] == "rescue")))
        share = len(high) / len(aligned)
        c_ok = (not math.isnan(hi_auroc)) and hi_auroc >= overall - CI_LOSS_TOL \
            and keep_rate >= RESCUE_KEEP_MIN and share <= SHARE_MAX
    inst_ok = EXIST_LO <= exist_rate <= EXIST_HI
    a_ok = coinc_rate >= COINCIDE_MIN
    b_ok = b_auroc >= AUROC_MAG_MIN

    lines = ["# CRG S0 判读（零 GPU）", "",
             f"- 判据：`docs/protocol/crg-commitment-revision-gate-20260929.md` §5"
             f"（A 重合率 ≥{COINCIDE_MIN}；B 幅度 AUROC ≥{AUROC_MAG_MIN}；"
             f"C 黏性损失 ≤{CI_LOSS_TOL} ∧ 救回保留 ≥{RESCUE_KEEP_MIN} ∧ 占比 ≤{SHARE_MAX}）",
             f"- 档案：`{args.arch}` × seeds {args.seeds}（baseline × sym b{FINAL_BETA:.2f}）",
             f"- 缺档：{meta['missing'] or '无'}",
             f"- 分叉样本 {len(div)}（救回 {len(resc)}／破坏 {len(brk)}）", ""]
    if div:
        lines += [f"## 仪器", "",
                  f"- 反转点存在率 **{exist_rate:.3f}** ∈ [{EXIST_LO:.2f}, {EXIST_HI:.2f}] ⇒ "
                  f"{'✅' if inst_ok else '❌'}", "",
                  f"## S0-A 重合", "",
                  f"- 首反转 ≡ 首分叉：**{len(coincide)}/{len(flip)} = {coinc_rate:.3f}**"
                  f"（要求 ≥{COINCIDE_MIN}）⇒ {'✅' if a_ok else '❌'}", "",
                  f"## S0-B 幅度可分性", "",
                  f"- AUROC(|Δ₀|, 救回 vs 破坏) = **{b_auroc:.3f}**"
                  f"（n={len(mag_pos)}/{len(mag_neg)}；要求 ≥{AUROC_MAG_MIN}）⇒ {'✅' if b_ok else '❌'}", "",
                  f"## S0-C 黏性保持与 footprint", "",
                  f"- 对齐子集 n={len(aligned)}；全体黏性 AUROC **{overall:.3f}**；"
                  f"幅度高位半 AUROC **{hi_auroc:.3f}**、救回保留 **{keep_rate:.3f}**、"
                  f"占比 **{share:.3f}**" if aligned and c_ok is not None else
                  f"- 对齐子集 n={len(aligned)}（不足 ⇒ C 不计算）", ""]
    if inst_ok and a_ok and b_ok and c_ok:
        verdict, why = "enter", "A∧B∧C 全过 ⇒ 进 S1"
    elif a_ok and not (b_ok and c_ok):
        verdict, why = "degenerate", ("A 过而 B/C 不过 ⇒ CRG 退化为 A1（when 轴无增量），"
                                      "关闭独立立项，审计并入 A1")
    else:
        verdict, why = "close", ("A 不过（反转点≠决策点）或仪器未过 ⇒ 关闭")
    lines += [f"## 判定", "", f"- **{verdict}** — {why}", ""]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "crg_gate_report.md").write_text("\n".join(lines))
    with open(out_dir / "crg_gate.json", "w") as f:
        json.dump({"exist_rate": exist_rate, "coinc_rate": coinc_rate,
                   "b_auroc": b_auroc, "overall_auroc": overall,
                   "n_div": len(div), "n_flip": len(flip), "n_aligned": len(aligned)},
                  f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_dir / 'crg_gate_report.md'}")
    print("\n".join(lines))


def selftest():
    import random
    ok = 0
    # auroc 手工值
    assert abs(auroc([1, 2, 3], [0]) - 1.0) < 1e-9
    assert abs(auroc([0], [1, 2, 3]) - 0.0) < 1e-9
    assert abs(auroc([1, 2], [1, 2]) - 0.5) < 1e-9
    ok += 1
    # revision_step（候选短名单进出）：y_true 在 t=2 进入 top10 ⇒ t*=2，mag=|l0(yt)−l0(a)|
    b = {"y_true_id": 9, "steps": [
        {"geo": {"yt": {"l0": 3.0}, "l0_a": 1.0}, "final_top10": [(1, "a", 0.5), (2, "b", 0.4)]},
        {"geo": {"yt": {"l0": 3.0}, "l0_a": 1.0}, "final_top10": [(1, "a", 0.5), (2, "b", 0.4)]},
        {"geo": {"yt": {"l0": 3.0}, "l0_a": 1.0}, "final_top10": [(9, "g", 0.5), (1, "a", 0.4)]},
    ]}
    t_s, mag = revision_step(b)
    assert t_s == 2 and abs(mag - 2.0) < 1e-9
    # 无进出
    b2 = {"y_true_id": 9, "steps": [
        {"geo": {"yt": {"l0": 3.0}, "l0_a": 1.0}, "final_top10": [(9, "g", 0.5), (1, "a", 0.4)]},
        {"geo": {"yt": {"l0": 3.0}, "l0_a": 1.0}, "final_top10": [(9, "g", 0.5), (1, "a", 0.4)]},
    ]}
    assert revision_step(b2) == (None, None)
    ok += 1
    # diverge_step / label_of
    b3 = {"steps": [{"chosen_id": 1}, {"chosen_id": 2}]}
    p3 = {"steps": [{"chosen_id": 1}, {"chosen_id": 3}]}
    assert diverge_step(b3, p3) == 1 and diverge_step(b3, b3) is None
    assert label_of({"baseline_correct": False, "is_correct": True}) == "rescue"
    assert label_of({"baseline_correct": True, "is_correct": False}) == "break"
    assert label_of({"baseline_correct": True, "is_correct": True}) == "nochange"
    ok += 1
    print(f"SELFTEST PASS: {ok}/3 checks (auroc / revision / diverge-label)")
    return ok


if __name__ == "__main__":
    main()
