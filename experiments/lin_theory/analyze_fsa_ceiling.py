"""方向一·FSA 可拦性上界判读（S0/S1，零 GPU）。

理论锚点（规则 1）与判据：docs/protocol/fsa-flip-set-arbitration-20260929.md §5
  - S0（基线档案双 seed）：
      仪器  分歧步占比 r_div ∈ [1%, 30%]（步级；全无分歧=无仲裁空间、全分歧=参考层退化）
      主判  潜在救回 ≥ 5 ∧ 潜在救回 ≥ 潜在破坏 + 2（pooled 双 seed）：
            潜在救回 = baseline 答错样本中，首个分歧步 t_d 的共识集 C_{t_d} 含 y_true
                      且 a_{t_d} ≠ y_true（上界口径：oracle 从共识集翻出正确答案）；
            潜在破坏 = baseline 答对样本中，存在分歧步使 a_t = y_true（仲裁将覆盖正确 token）。
      判定  主判过 ⇒ 进 S1；不过 ⇒ FSA 关闭（审计素材），不占用 GPU。
  - S1（fsa 真机档案自洽，--verify）：
      无分歧步 chosen == geo.a_id；分歧步 chosen ∈ C ∪ {a}（C 空时 == a）；触发率 ∈ [1%, 30%]。

红线：l_final 为真实 logits（T04）；仅用 top-10 与 geo 标量，不做任何 lens 代验；
k 敏感性 {5,20} 只报不判（主档 k=10）。
"""
import argparse
import json
import math
from pathlib import Path

ARCH_DEFAULT = "experiments/outputs/geometry_archive"
OUT_DEFAULT = "experiments/outputs/fsa_ceiling"
K_MAIN = 10
KS = [5, 10, 20]
R_DIV_LO, R_DIV_HI = 0.01, 0.30   # 分歧步占比仪器区间
N_RESCUE_MIN = 5                   # 潜在救回事件数下限（fail-closed）
NET_MARGIN = 2                     # 净可拦 ≥ 潜在破坏 + 2


def analyze_sample(sample, k):
    """返回该样本的 S0 读数 dict（k 为共识集半径）。"""
    y_true = sample["y_true_id"]
    is_correct = bool(sample["is_correct"])
    n_steps = 0
    n_div = 0
    t_d = None
    C_d = None
    a_d = None
    covered_break = False  # 存在分歧步使 a_t == y_true（答对样本 ⇒ 潜在破坏）
    for st in sample.get("steps", []):
        n_steps += 1
        a = int(st["geo"]["a_id"])
        e_ids = [x[0] for x in st.get("early_top10", [])][:k]
        f_ids = [x[0] for x in st.get("final_top10", [])][:k]
        div = a not in e_ids
        if div:
            n_div += 1
            if t_d is None:
                t_d = st["step"]
                a_d = a
                C_d = [x for x in f_ids if x in e_ids]
            if a == y_true:
                covered_break = True
    rescue = False
    if (not is_correct) and t_d is not None and a_d != y_true and y_true in (C_d or []):
        rescue = True
    return {
        "n_steps": n_steps, "n_div": n_div, "t_d": t_d, "a_d": a_d, "C_d": C_d,
        "is_correct": is_correct, "subset": sample.get("subset"),
        "rescue": rescue, "break": is_correct and covered_break,
        "consensus_empty": (C_d is not None and not C_d),
    }


def verdict(rescue_n, break_n, r_div):
    """S0 判定纯函数：返回 (分支, 说明)。"""
    if not (R_DIV_LO <= r_div <= R_DIV_HI):
        return "close", (f"仪器未过：分歧步占比 {r_div:.4f} ∉ "
                         f"[{R_DIV_LO:.2f}, {R_DIV_HI:.2f}]")
    if rescue_n < N_RESCUE_MIN:
        return "close", f"潜在救回 {rescue_n} < {N_RESCUE_MIN}（事件数不足，fail-closed）"
    if rescue_n < break_n + NET_MARGIN:
        return "close", f"净可拦 {rescue_n - break_n:+d} < +{NET_MARGIN}（潜在救回 {rescue_n} / 潜在破坏 {break_n}）"
    return "enter", f"潜在救回 {rescue_n} ≥ {N_RESCUE_MIN} ∧ 净可拦 {rescue_n - break_n:+d} ≥ +{NET_MARGIN} ⇒ 进 S1"


def selftest():
    ok = 0
    # 合成样本 1：答错，首分歧步共识集含 y_true ⇒ 潜在救回
    s1 = {
        "y_true_id": 7, "is_correct": False, "subset": "know_wrong",
        "steps": [
            {"step": 0, "geo": {"a_id": 1},
             "early_top10": [(1, "a", 0.5), (2, "b", 0.4)], "final_top10": [(1, "a", 0.5), (2, "b", 0.4)]},
            {"step": 1, "geo": {"a_id": 3},  # 分歧步：3 ∉ E={7,2}；C={3,7}∩{7,2}={7} 含 y_true
             "early_top10": [(7, "g", 0.5), (2, "b", 0.4)], "final_top10": [(3, "c", 0.5), (7, "g", 0.3)]},
        ],
    }
    r1 = analyze_sample(s1, k=2)
    assert r1["t_d"] == 1 and r1["C_d"] == [7] and r1["rescue"] is True
    assert r1["n_div"] == 1 and not r1["break"] and not r1["consensus_empty"]
    ok += 1
    # 合成样本 2：答对，分歧步 a == y_true ⇒ 潜在破坏
    s2 = {
        "y_true_id": 5, "is_correct": True, "subset": "know_correct",
        "steps": [
            {"step": 0, "geo": {"a_id": 5},  # 分歧步且 a == y_true
             "early_top10": [(1, "a", 0.5), (2, "b", 0.4)], "final_top10": [(5, "e", 0.5), (1, "a", 0.4)]},
        ],
    }
    r2 = analyze_sample(s2, k=2)
    assert r2["break"] is True and not r2["rescue"]
    ok += 1
    # 合成样本 3：答错，分歧步共识集空 ⇒ 不算救回、consensus_empty
    s3 = {
        "y_true_id": 7, "is_correct": False, "subset": "dont_know",
        "steps": [
            {"step": 0, "geo": {"a_id": 3},
             "early_top10": [(1, "a", 0.5), (2, "b", 0.4)], "final_top10": [(3, "c", 0.5), (4, "d", 0.4)]},
        ],
    }
    r3 = analyze_sample(s3, k=2)
    assert r3["t_d"] == 0 and r3["consensus_empty"] is True and not r3["rescue"]
    ok += 1
    # 判定函数三态
    assert verdict(8, 4, 0.05)[0] == "enter"
    assert verdict(8, 4, 0.005)[0] == "close"   # 触发率过低
    assert verdict(3, 1, 0.05)[0] == "close"    # 事件不足
    assert verdict(6, 5, 0.05)[0] == "close"    # 净可拦 < +2
    ok += 1
    print(f"SELFTEST PASS: {ok}/4 checks (sample×3 / verdict)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=ARCH_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", default=None, help="S1 档案自洽（fsa/fsa-rand 真机档案）")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    if args.verify:
        verify_archive(Path(args.verify))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    lines = ["# FSA S0 可拦性上界判读（零 GPU）", "",
             f"- 判据：`docs/protocol/fsa-flip-set-arbitration-20260929.md` §5"
             f"（主档 k={K_MAIN}；仪器 r_div ∈ [{R_DIV_LO:.2f}, {R_DIV_HI:.2f}]；"
             f"潜在救回 ≥{N_RESCUE_MIN} ∧ 净可拦 ≥ +{NET_MARGIN}）",
             f"- 档案：`{args.arch}` × seeds {args.seeds}（baseline 档）", ""]

    missing = []
    for s in args.seeds:
        if not (Path(args.arch) / f"geo_baseline_seed{s}.json").exists():
            missing.append(f"seed{s}")
    if missing:
        print("  ⚠️ 缺档（该 seed 跳过）：" + ", ".join(missing))

    summary = {}
    for k in args.ks:
        rows = []
        for s in args.seeds:
            fp = Path(args.arch) / f"geo_baseline_seed{s}.json"
            if not fp.exists():
                continue
            data = json.loads(fp.read_text())
            for smp in data["samples"]:
                r = analyze_sample(smp, k)
                r["seed"] = s
                rows.append(r)
        n_steps = sum(r["n_steps"] for r in rows)
        n_div = sum(r["n_div"] for r in rows)
        r_div = n_div / n_steps if n_steps else float("nan")
        rescue_n = sum(1 for r in rows if r["rescue"])
        break_n = sum(1 for r in rows if r["break"])
        empty_n = sum(1 for r in rows if r["consensus_empty"])
        trig = sum(1 for r in rows if r["t_d"] is not None)
        summary[k] = dict(r_div=r_div, rescue=rescue_n, break_n=break_n, empty=empty_n,
                          trig=trig, n=len(rows), steps=n_steps)
        branch, why = verdict(rescue_n, break_n, r_div)
        tag = "【主档 k=10】" if k == K_MAIN else ""
        lines += [f"## k={k} {tag}", "",
                  f"- 样本 {len(rows)}｜步数 {n_steps}｜分歧步 {n_div}（占比 **{r_div:.3f}**）｜"
                  f"触发样本 {trig}（{trig / len(rows) * 100:.1f}%）",
                  f"- 共识集空（分歧步且 C=∅）样本 {empty_n}｜共识集大小中位（只报）见 JSON",
                  f"- **潜在救回 {rescue_n}｜潜在破坏 {break_n}｜净可拦 {rescue_n - break_n:+d}**",
                  f"- **S0 判定：{branch}** — {why}", ""]
    if K_MAIN in summary:
        sm = summary[K_MAIN]
        lines += ["## 主档判定", "",
                  f"- 仪器：r_div = {sm['r_div']:.3f} ∈ [{R_DIV_LO:.2f}, {R_DIV_HI:.2f}] ⇒ "
                  f"{'✅' if R_DIV_LO <= sm['r_div'] <= R_DIV_HI else '❌'}",
                  f"- 主判：潜在救回 {sm['rescue']}（≥{N_RESCUE_MIN}？"
                  f"{'✅' if sm['rescue'] >= N_RESCUE_MIN else '❌'}）∧ 净可拦 "
                  f"{sm['rescue'] - sm['break_n']:+d}（≥+{NET_MARGIN}？"
                  f"{'✅' if sm['rescue'] >= sm['break_n'] + NET_MARGIN else '❌'}）",
                  f"- **终判：{verdict(sm['rescue'], sm['break_n'], sm['r_div'])[0]}**", ""]
    report = out_dir / "fsa_ceiling_report.md"
    report.write_text("\n".join(lines))
    with open(out_dir / "fsa_ceiling.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {report}")
    print("\n".join(lines))


def verify_archive(path):
    """S1 档案自洽（无模型）：fsa/fsa-rand 档每步判定一致性。"""
    data = json.loads(path.read_text())
    meta = data.get("meta", {})
    if meta.get("operator") not in ("fsa", "fsa-rand"):
        raise SystemExit(f"非 FSA 档案：operator={meta.get('operator')}")
    k = meta.get("fsa_k") or K_MAIN
    n_err = n_step = n_div = 0
    for smp in data["samples"]:
        y_true = smp["y_true_id"]
        for st in smp["steps"]:
            n_step += 1
            fi = st.get("fsa", {})
            a = int(st["geo"]["a_id"])
            chosen = int(st["chosen_id"])
            e_ids = [x[0] for x in st.get("early_top10", [])][:k]
            f_ids = [x[0] for x in st.get("final_top10", [])][:k]
            div = a not in e_ids
            if div:
                n_div += 1
            C = [x for x in f_ids if x in e_ids]
            ok = True
            if chosen != fi.get("candidate"):
                ok = False
            if (not div) and chosen != a:
                ok = False
            if div and C and chosen not in C:
                ok = False
            if div and not C and chosen != a:
                ok = False
            if not ok:
                n_err += 1
                print(f"  [ERR] {smp['sample_id']} step {st['step']}: chosen={chosen} "
                      f"fsa={fi} a={a} E={e_ids} C={C}")
    r_div = n_div / n_step if n_step else float("nan")
    passed = n_err == 0 and R_DIV_LO <= r_div <= R_DIV_HI
    print(f"VERIFY {path.name}: {n_step} steps, 分歧 {n_div} ({r_div:.3f}), {n_err} errors → "
          f"{'PASS' if passed else 'FAIL'}（触发率区间检查 {'✅' if R_DIV_LO <= r_div <= R_DIV_HI else '❌'}）")
    return passed


if __name__ == "__main__":
    main()
