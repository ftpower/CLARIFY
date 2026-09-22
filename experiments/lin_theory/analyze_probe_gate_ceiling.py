"""探针门控 TLDC 的"精度 → 兑现率"映射（零 GPU）——重锚结题口径用。

回答的问题：在本协议（TriviaQA / n=300 / TLDC 无真值单遍 / KW-KC-DK 划分）下，
**样本级语义门控**能兑现多少 oracle 上限？探针要练到多准才值得投？

三层上限（同 docs/protocol/review-runbook §5.7 口径）：
  U0 现状       ：β=0.20 固定、无门控
  U1 选择 oracle：β=0.20 + 完美选择（只干预会被救回的样本）
  U2 联合 oracle：7 档 β 逐样本最优 + 完美选择（破坏=0 由"不干预"保证）＝确证上限

数据（均已落盘）：
  · per-sample 逐 β 结果：experiments/outputs/lin_theory_8b/seed{123,456}_8b/s14_tldc_samples.json
  · 探针隐状态 h@L28    ：probe_scores/probe_scores_seed{123,456}_Qwen3-8B.json

探针分数：留一法 truth direction（μ_correct − μ_wrong，排除自身）——避免 in-sample 乐观。
门控策略：样本级二值门（score < θ 则整条轨迹施加 β=0.20），与 gated_simulation_8b 同构。

用法：
  python experiments/lin_theory/analyze_probe_gate_ceiling.py \
    --out experiments/outputs/probe_gate_ceiling_8b
"""

import argparse
import json
from pathlib import Path

BETAS = ["0.01", "0.03", "0.05", "0.08", "0.10", "0.15", "0.20"]
BETA_MAIN = "0.20"


def loo_truth_scores(H, y):
    """留一法 truth direction 分数：score_i = (mu_c - mu_w)·h_i，mu 用其余样本估计。

    纯 Python 实现（不依赖 numpy/scipy）：本档案 300 x 2048，秒级完成。
    """
    n = len(H)
    d = len(H[0])
    s_c = [0.0] * d
    s_w = [0.0] * d
    nc = nw = 0
    for i in range(n):
        row, acc = H[i], (s_c if y[i] else s_w)
        for j in range(d):
            acc[j] += row[j]
        if y[i]:
            nc += 1
        else:
            nw += 1
    scores = []
    for i in range(n):
        if y[i]:
            denom_c, denom_w = max(nc - 1, 1), max(nw, 1)
        else:
            denom_c, denom_w = max(nc, 1), max(nw - 1, 1)
        row = H[i]
        sc = 0.0
        for j in range(d):
            if y[i]:
                mu_c = (s_c[j] - row[j]) / denom_c
                mu_w = s_w[j] / denom_w
            else:
                mu_c = s_c[j] / denom_c
                mu_w = (s_w[j] - row[j]) / denom_w
            sc += (mu_c - mu_w) * row[j]
        scores.append(sc)
    return scores


def simulate(S, score, theta):
    """样本级门控模拟：score<θ 则施加 β=0.20，否则保持基线。返回计数。"""
    resc = brk = 0
    base_ok = 0
    for i, k in enumerate(S):
        v = S[k]
        base = bool(v["baseline_correct"])
        base_ok += base
        open_gate = score[i] < theta
        final = bool(v[f"correct_beta{BETA_MAIN}"]) if open_gate else base
        if not base and final:
            resc += 1
        elif base and not final:
            brk += 1
    return resc, brk, resc - brk, base_ok


def ceiling(S):
    """U1/U2 上限。"""
    bw = [k for k in S if not S[k]["baseline_correct"]]
    u1 = sum(1 for k in bw if S[k][f"correct_beta{BETA_MAIN}"])
    u2 = sum(1 for k in bw if any(S[k][f"correct_beta{b}"] for b in BETAS))
    return u1, u2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/outputs/probe_gate_ceiling_8b")
    args = ap.parse_args()

    md = ["# 探针门控 TLDC：精度 → 兑现率映射（8B，零 GPU）", ""]
    summary = {}
    for seed in ("123", "456"):
        s14 = json.load(open(f"experiments/outputs/lin_theory_8b/seed{seed}_8b/s14_tldc_samples.json"))
        S = s14["samples"]
        keys = list(S.keys())
        pr = json.load(open(f"probe_scores/probe_scores_seed{seed}_Qwen3-8B.json"))
        P = {str(e["sample_id"]): e for e in pr["entries"]}
        # 对齐校验
        assert all(k in P for k in keys), "probe 档案与 s14 样本 id 不匹配"
        y = [bool(P[k]["is_correct"]) for k in keys]
        y_base = [bool(S[k]["baseline_correct"]) for k in keys]
        mism = sum(1 for a, b in zip(y, y_base) if a != b)
        H = [P[k]["h"] for k in keys]
        score = loo_truth_scores(H, y)

        u1, u2 = ceiling(S)
        n = len(keys)
        r0, b0, net0, _ = simulate(S, [-1e9] * n, 0.0)  # 全开门 = 无门控
        md += [f"## seed{seed}", "",
               f"- 样本 {n}｜基线对 {sum(y_base)}｜探针档案与 s14 的 `is_correct` 不一致 {mism} 个（应为 0）",
               f"- **U0 现状（无门控）**：救回 {r0}、破坏 {b0}、净 {net0:+d} = {net0/n*100:+.2f}pp",
               f"- **U1 选择 oracle（β=0.20）**：{u1} 事件 = +{u1/n*100:.2f}pp",
               f"- **U2 联合 oracle（7 档 β）**：{u2} 事件 = +{u2/n*100:.2f}pp ← 确证上限", ""]

        # 阈值扫描
        qs = sorted(set([float(x) for x in score]))
        cand = [qs[0] - 1e-9] + [qs[min(len(qs) - 1, int(len(qs) * p / 100))] for p in range(2, 101, 2)]
        rows = []
        for th in cand:
            resc, brk, net, _ = simulate(S, score, th)
            open_mask = [s < th for s in score]
            # recall = 在"真错"样本上的开门率；spec = 在"真对"样本上的关门率
            wrong = [i for i in range(n) if not y_base[i]]
            right = [i for i in range(n) if y_base[i]]
            rec = sum(1 for i in wrong if open_mask[i]) / max(len(wrong), 1)
            spec = sum(1 for i in right if not open_mask[i]) / max(len(right), 1)
            a_bal = (rec + spec) / 2
            model_net = a_bal * u1 - (1 - a_bal) * b0
            rows.append((th, sum(open_mask), rec, spec, resc, brk, net, model_net))
        best = max(rows, key=lambda r: r[6])
        md += ["| θ | 开门数 | recall | spec | 净事件 | 兑现率(U2) | 同 a 随机模型期望 | 惩罚 |",
               "|---|---|---|---|---|---|---|---|"]
        for r in rows[::max(1, len(rows) // 12)] + [best]:
            md.append(f"| {r[0]:+.1f} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[6]:+d} | "
                      f"{r[6]/u2*100:.0f}% | {r[7]:+.1f} | {r[6]-r[7]:+.1f} |")
        md += ["", f"- **实测最优门控**：θ={best[0]:+.2f}（recall {best[2]:.2f}/spec {best[3]:.2f}）"
                  f"→ 净 {best[6]:+d} = {best[6]/n*100:+.2f}pp，兑现率 **{best[6]/u2*100:.0f}%**（含事后再选 θ，乐观）", ""]

        # 精度阶梯：随机误差模型 net(a) = a·U1 − (1−a)·B
        B = b0
        ladder = []
        for a in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
            exp_net = a * u1 - (1 - a) * B
            ladder.append((a, exp_net, exp_net / u2 * 100))
        md += ["**精度阶梯（随机误差模型）** `net(a) = a·U1 − (1−a)·B`：", "",
               "| 平衡准确率 a | 期望净事件 | 兑现率(U2) |", "|---|---|---|"]
        for a, e, rr in ladder:
            md.append(f"| {a:.2f} | {e:+.1f} | {rr:.0f}% |")
        md += ["", "> ⚠️ 该模型假设探针误差与「是否可救或将被破坏」**独立**。"
                   "实测最优门控低于该期望的部分 = **误差相关性惩罚**（探针错在要紧样本上）。", ""]
        summary[seed] = {"n": n, "U0": net0, "U1": u1, "U2": u2,
                         "best_net": best[6], "best_theta": best[0],
                         "best_recall": best[2], "best_spec": best[3],
                         "realization": best[6] / u2, "ladder": ladder}

    text = "\n".join(md)
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_gate_ceiling.md").write_text(text, encoding="utf-8")
    (out / "probe_gate_ceiling.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"\n[saved] {out / 'probe_gate_ceiling.md'}")


if __name__ == "__main__":
    main()
