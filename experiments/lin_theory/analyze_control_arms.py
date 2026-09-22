"""对照臂判读（主判据 H1 vs H2）——读 main_tldc_controls.py 的输出，一条命令出判定。

依据 docs/protocol/placebo-control-protocol.md §3 预注册判据：
  主判据（β=0.20）：KW 基线错样本上，real vs shuffle 的**配对 McNemar**
    b = 仅 real 救回，c = 仅 shuffle 救回
    p<0.05 且 b>c  ⇒ H1（δ 内容起作用）成立 → FAD/S5 前提保住
    否则            ⇒ H2（通用扰动）不能被排除 → 相关主张降级
  次判据（β=0.03）：同法，仅一致性检查。
  描述性（不进判据）：gauss / anti / wrong_* 的同向率与 KW/DK 比。

用法：
  python experiments/lin_theory/analyze_control_arms.py \
    experiments/outputs/tldc_controls/tldc_controls_123_*.json \
    [--betas 0.2 0.03] [--primary_arm shuffle] [--output_dir <dir>]
"""

import argparse
import json
from pathlib import Path

from analyze_transition_matrix import (  # 复用同一套精确统计与逐格口径
    clopper_pearson,
    fisher_greater,
    mcnemar_exact,
    transition,
)

BETA_PRIMARY_DEFAULT = 0.20


def load_files(paths):
    """→ {(arm, beta): {sample_id: (subset, base, final)}}，多文件（多 seed）自动 pool。

    ⚠️ 以 sample_id 为键（不用列表顺序），保证 real-vs-对照的配对不会因样本集差异错位。
    注意：不同 seed 的 sample_id 会冲突，故多 seed pool 用 **seed 前缀** 复合键规避。
    ⚠️ 2026-09-22 修复：`main_tldc_controls.py` 把 config 写在 `report.config`（**不是** `meta.config`），
    原实现读 `meta.config` ⇒ seed 恒为 "?" ⇒ 双 seed 输入互相覆盖、pooled 静默只剩一个 seed
    （8B 上会把 122 个 KW 当成 64 个，pooled p 从 0.0127 变成 0.2266）。现改为 report.config 优先、
    meta 兜底，并对同 tag 的重复输入报警。
    """
    pooled = {}
    metas = []
    tags = {}
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
        metas.append({"file": Path(p).name, "meta": meta, "seed": seed, "tag": tag})
        if tag in tags:
            print(f"  ⚠️ [pool 警告] {Path(p).name} 与前一个输入同 tag={tag}"
                  f"（前一个：{tags[tag]}）⇒ 同 seed 重复输入会造成样本覆盖，请确认是否误传")
        tags[tag] = Path(p).name
        samples = d["samples"]
        arms, betas = set(), set()
        for v in samples.values():
            for k in v:
                if k.startswith("correct_"):
                    arm, _, b = k[len("correct_"):].rpartition("_beta")
                    arms.add(arm)
                    betas.add(float(b))
        for arm in arms:
            for b in betas:
                key = f"correct_{arm}_beta{b}"
                bucket = pooled.setdefault((arm, b), {})
                for sid, v in samples.items():
                    if key in v and v.get("baseline_correct") is not None:
                        bucket[f"{tag}:{sid}"] = (v["subset"], bool(v["baseline_correct"]), bool(v[key]))
    return pooled, metas


def paired(real_bucket, ctrl_bucket):
    """KW 基线错样本上的配对计数（按 sample_id 取交集，避免错位）。

    b = 仅 real 救回，c = 仅对照救回。
    """
    shared = set(real_bucket) & set(ctrl_bucket)
    b = c = n_kw = 0
    for sid in shared:
        sub, base, real_after = real_bucket[sid]
        if sub != "know_wrong" or base:
            continue
        n_kw += 1
        ctrl_after = ctrl_bucket[sid][2]
        if real_after and not ctrl_after:
            b += 1
        elif ctrl_after and not real_after:
            c += 1
    return b, c, mcnemar_exact(b, c), n_kw


def main():
    ap = argparse.ArgumentParser(description="对照臂判读（H1 vs H2）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--betas", nargs="+", type=float, default=None)
    ap.add_argument("--primary_arm", type=str, default="shuffle")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    pooled, metas = load_files(args.paths)
    if not pooled:
        raise SystemExit("未从输入文件中解析到任何 correct_<arm>_beta<β> 字段")

    arms = sorted({a for a, _ in pooled})
    betas = sorted({b for _, b in pooled})
    if args.betas:
        betas = [b for b in betas if b in args.betas]
    beta_main = BETA_PRIMARY_DEFAULT if BETA_PRIMARY_DEFAULT in betas else (max(betas) if betas else None)

    md = ["# 对照臂判读报告（H1 知识特异 vs H2 通用扰动）", ""]
    md.append(f"- 输入：{', '.join(m['file'] for m in metas)}")
    md.append(f"- 臂：{arms}；β：{betas}；主判据档 β={beta_main}，主对照臂 `{args.primary_arm}`")
    md.append("")

    stats = {}
    for b in betas:
        md += [f"## β={b}", "",
               "| 臂 | n | KW 救回 (CP95) | DK 救回 | KW/DK | Fisher p | KC 破坏 | DK 破坏 | 净 pp | McNemar p |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            bucket = pooled.get((arm, b))
            if not bucket:
                continue
            rows = list(bucket.values())
            t = transition(rows)
            ps = t["per_subset"]
            kw, kc, dk = ps.get("know_wrong", {}), ps.get("know_correct", {}), ps.get("dont_know", {})
            ratio = None
            fp = None
            if kw.get("base_wrong") and dk.get("base_wrong") and dk.get("rescue_rate"):
                ratio = kw["rescue_rate"] / dk["rescue_rate"]
                fp = fisher_greater(kw["rescue"], kw["base_wrong"] - kw["rescue"],
                                    dk["rescue"], dk["base_wrong"] - dk["rescue"])
            ci = clopper_pearson(kw["rescue"], kw["base_wrong"]) if kw.get("base_wrong") else None
            stats[(arm, b)] = {"t": t, "ratio": ratio, "fisher": fp, "ci": ci}
            md.append(
                f"| {arm} | {t['n']} | {kw.get('rescue')}/{kw.get('base_wrong')} "
                f"({kw['rescue_rate']*100:.1f}%{'' if ci is None else f', CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]'}) | "
                f"{dk.get('rescue')}/{dk.get('base_wrong')} ({100*(dk.get('rescue_rate') or 0):.1f}%) | "
                f"{'—' if ratio is None else f'{ratio:.2f}×'} | {'—' if fp is None else f'{fp:.4f}'} | "
                f"{kc.get('break')}/{kc.get('base_right')} ({100*(kc.get('break_rate') or 0):.1f}%) | "
                f"{dk.get('break')}/{dk.get('base_right')} ({100*(dk.get('break_rate') or 0):.1f}%) | "
                f"{t['net_pp']:+.1f} | {t['mcnemar_p']:.4f} |"
            )
        md.append("")

    # ── 主判据：配对 McNemar（需 real 与对照同批） ──
    verdict = []
    if beta_main is not None:
        md += [f"## 主判据（β={beta_main}，配对 McNemar，KW 基线错样本）", "",
               "| 对照臂 | b（仅 real 救回） | c（仅对照救回） | 配对 p | 判定 |", "|---|---|---|---|---|"]
        if ("real", beta_main) not in pooled:
            md.append("| — | — | — | — | ⚠️ 输入中无 real 臂，配对检验不可用（请在同一批运行中包含 real） |")
            verdict.append("no-real")
        else:
            for arm in arms:
                if arm == "real" or (arm, beta_main) not in pooled:
                    continue
                b_, c_, p_, n_kw = paired(pooled[("real", beta_main)], pooled[(arm, beta_main)])
                if arm == args.primary_arm:
                    ok = (p_ < 0.05 and b_ > c_)
                    tag = "**H1 成立**（δ 内容起作用）" if ok else "**H2 不能排除**（通用扰动）"
                    verdict.append(("H1" if ok else "H2", b_, c_, p_, n_kw))
                else:
                    tag = "描述性"
                md.append(f"| {arm} | {b_} | {c_} | {p_:.4f} | {tag}（配对 KW n={n_kw}） |")
        md.append("")
        md += ["> 判据（预注册，禁止事后更改）：p<0.05 且 b>c ⇒ H1；否则 ⇒ H2 不能排除。",
               "> 功效提醒：若 b、c 都只有个位数，本设计判不了小效应——此时只能写\"未能排除 H2\"，",
               "> 不得反向主张 H1。", ""]

    print("\n".join(md))
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "control_arms_report.md").write_text("\n".join(md), encoding="utf-8")
        (out / "control_arms_report.json").write_text(
            json.dumps({f"{a}_beta{b}": {"transition": s["t"], "ratio_kw_dk": s["ratio"],
                                         "fisher_p": s["fisher"], "kw_ci": s["ci"]}
                        for (a, b), s in stats.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n[saved] {out / 'control_arms_report.md'}")


if __name__ == "__main__":
    main()
