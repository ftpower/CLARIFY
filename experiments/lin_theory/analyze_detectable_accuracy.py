# -*- coding: utf-8 -*-
"""可检测准确率（detectable accuracy）评估：阈值无关 AUROC + 风险—覆盖率曲线。

目的：补上项目叙事里缺的一层——"检测器在某个工作点上，把被它判为正确的样本留下来，
准确率有多高、代价（弃权/干预比例）多大"。全部指标建立在**折外（out-of-fold）**分数上，
避免 in-sample 高估（见 docs/protocol/evaluation-protocol.md 规则 1/3/4）。

协议（与主结果一致）：
  - 分层 5 折 CV；StandardScaler + LogisticRegression 全部在训练折内拟合，折外打分
  - 标签为 exact match 对错（存盘口径，与原实验同源）
  - 覆盖率 c = 保留比例（按 P(correct) 从高到低保留 top-c）；阈值无关部分用 AURAC
  - 区间估计用 Clopper-Pearson 精确区间

数据源（均为本地存盘，CPU 可跑，无需 GPU）：
  1) 8B  / TriviaQA : probe_scores/probe_scores_seed{123,456}_Qwen3-8B.json
                      （逐样本 h_L28，n=300/seed；基线正确数经 s14 档案交叉核对）
  2) 1.7B/ TriviaQA : experiments/outputs/lin_theory/detect_lr_probe_hidden.npz
                      （h_L0..L27，n=200，seed=42；逐层 CV 选峰值层）
  3) 1.7B/ HellaSwag : experiments/phase2_entropy/outputs/knowledge_filtered_data.json
                      （max_prob 为可部署输出面信号；p_correct 是金标索引量，禁止当检测分数用）

用法：
    python experiments/lin_theory/analyze_detectable_accuracy.py
输出：
    experiments/outputs/detectable_accuracy/detectable_accuracy.json + 控制台表
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "outputs" / "detectable_accuracy"
COVERAGES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
# 绘图用细网格（覆盖率 1.00→0.20，步长 0.01）：报告点数值仍以 COVERAGES 为准，
# 细网格只用于绘制"准确率—覆盖率"曲线（避免图上只有 6 个折点）。
CURVE_GRID = [round(1.00 - 0.01 * i, 2) for i in range(81)]
N_FOLDS = 5


def cp95(k, n):
    """Clopper-Pearson 95% 精确区间。"""
    if n == 0:
        return (0.0, 1.0)
    lo = float(beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return (lo, hi)


def oof_scores(X, y, seed=0):
    """折外 P(correct)：分层 5 折，Scaler+LR 全部折内拟合。

    seed 为**折划分**种子，需与来源实验一致：
      - detect_lr_probe_cv.py（检测主结果）: random_state=0
      - simulate_gated_tldc.py（门控模拟）  : random_state=TLDC seed（123/456）
    """
    scores = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return scores


def mean_fold_auroc(scores, y, seed=0):
    """协议口径 AUROC：与 cross_val_score 相同——逐折算 AUROC 再取均值（非池化）。

    与 detect_lr_probe_cv.py 的 `_probe_cv` 等价，用于逐层峰值层选择与结果核对。
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_aucs = []
    for _, te in skf.split(np.zeros(len(y)), y):
        if len(np.unique(y[te])) < 2:
            continue
        fold_aucs.append(roc_auc_score(y[te], scores[te]))
    return float(np.mean(fold_aucs)) if fold_aucs else float("nan")


def risk_coverage(scores, y, coverages=COVERAGES):
    """按 P(correct) 降序保留 top-c，返回每档覆盖率下的保留准确率与 CP95。"""
    order = np.argsort(-scores)
    y_sorted = y[order]
    n = len(y)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        kept = y_sorted[:k]
        acc = float(kept.mean())
        lo, hi = cp95(int(kept.sum()), k)
        rows.append({
            "coverage": round(c, 2),
            "n_kept": k,
            "n_kept_correct": int(kept.sum()),
            "retained_accuracy": acc,
            "cp95": [lo, hi],
            "errors_removed": int((1 - y).sum() - (1 - kept).sum()),
            "errors_total": int((1 - y).sum()),
            "correct_dropped": int(y.sum() - kept.sum()),
        })
    return rows


def risk_coverage_curve(scores, y, grid=CURVE_GRID):
    """细网格风险—覆盖率曲线（绘图用；只保留绘图所需字段，报告点见 risk_coverage）。"""
    order = np.argsort(-scores)
    y_sorted = y[order]
    n = len(y)
    rows = []
    for c in grid:
        k = max(1, int(round(c * n)))
        kept = y_sorted[:k]
        lo, hi = cp95(int(kept.sum()), k)
        rows.append({
            "coverage": round(float(c), 2),
            "n_kept": k,
            "retained_accuracy": float(kept.mean()),
            "cp95": [lo, hi],
        })
    return rows


def aurac(scores, y):
    """AURAC：准确率—覆盖率曲线下面积（覆盖率 0→1，随机基线 = 整体准确率）。"""
    order = np.argsort(-scores)
    y_sorted = y[order]
    cum_acc = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(cum_acc, cov))


def summarize(tag, scores, y, extra=None, fold_seed=None):
    base = float(y.mean())
    auroc = float(roc_auc_score(y, scores))
    rc = risk_coverage(scores, y)
    out = {
        "tag": tag,
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "baseline_accuracy": base,
        "auroc_oof_pooled": auroc,
        "aurac": aurac(scores, y),
        "aurac_random": base,
        "risk_coverage": rc,
        "curve": risk_coverage_curve(scores, y),
    }
    if fold_seed is not None:
        out["auroc_fold_mean"] = mean_fold_auroc(scores, y, seed=fold_seed)
        out["fold_seed"] = fold_seed
    if extra:
        out.update(extra)
    print(f"\n=== {tag} ===")
    fm = out.get("auroc_fold_mean")
    print(f"n={len(y)}  基线准确率={base:.4f}  折外池化 AUROC={auroc:.4f}"
          + (f"  折均值 AUROC={fm:.4f}（协议口径）" if fm is not None else "")
          + f"  AURAC={out['aurac']:.4f}（随机基线 {base:.4f}）")
    print(f"{'覆盖率':>6} {'保留n':>6} {'保留准确率':>10} {'CP95':>18} {'剔除错误':>8} {'误删正确':>8}")
    for r in rc:
        print(f"{r['coverage']:>6.2f} {r['n_kept']:>6d} {r['retained_accuracy']:>10.4f} "
              f"[{r['cp95'][0]:.3f},{r['cp95'][1]:.3f}]".rjust(18) +
              f" {r['errors_removed']:>5d}/{r['errors_total']:<3d} {r['correct_dropped']:>8d}")
    return out


def load_8b_triviaqa():
    """8B / TriviaQA：逐样本 h_L28（probe_scores/），与 TLDC 档案交叉核对标签。"""
    blocks = []
    for seed in (123, 456):
        sp = REPO / "probe_scores" / f"probe_scores_seed{seed}_Qwen3-8B.json"
        ap = REPO / "experiments" / "outputs" / "lin_theory_8b" / f"seed{seed}_8b" / "s14_tldc_samples.json"
        sc = json.load(open(sp, encoding="utf-8"))
        ar = json.load(open(ap, encoding="utf-8"))
        by_key = {(e["sample_id"], e["question"]): e for e in sc["entries"]}
        X, y, mism = [], [], 0
        for sid, rec in ar["samples"].items():
            key = (int(sid), rec["question"])
            e = by_key[key]
            if bool(e["is_correct"]) != bool(rec["baseline_correct"]):
                mism += 1
            X.append(e["h"])
            y.append(int(e["is_correct"]))
        blocks.append({
            "seed": seed,
            "X": np.asarray(X, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int32),
            "label_mismatch": mism,
            "archive": ar,
        })
    return blocks


def load_probe_npz(path, fold_seed=0):
    """通用：逐层隐藏状态 npz（h_L0..h_L{n-1} + y）→ 逐层 CV 选峰值层后取折外分数。

    峰值层用**协议口径**（折均值 AUROC，与 cross_val_score 一致）选择。
    """
    z = np.load(path, allow_pickle=True)
    y = z["y"].astype(np.int32)
    layers = sorted(int(k[3:]) for k in z.files if k.startswith("h_L"))
    per_layer = {}
    for li in layers:
        s = oof_scores(z[f"h_L{li}"], y, seed=fold_seed)
        per_layer[li] = mean_fold_auroc(s, y, seed=fold_seed)
    peak = max(per_layer, key=per_layer.get)
    scores = oof_scores(z[f"h_L{peak}"], y, seed=fold_seed)
    return scores, y, {
        "source": str(path),
        "n_layers": len(layers),
        "peak_layer": int(peak),
        "peak_layer_auroc_fold_mean": per_layer[peak],
        "layer_auroc_fold_mean": {int(k): v for k, v in per_layer.items()},
    }


def load_oof_json(path):
    """通用：detect_lr_probe_cv.py --dump 的逐样本折外分数 JSON。"""
    d = json.load(open(path, encoding="utf-8"))
    return np.asarray(d["scores_peak"], dtype=float), np.asarray(d["y"], dtype=np.int32), {
        "source": str(path),
        "best_layer": d.get("best_layer"),
        "config": d.get("config", {}),
        "note": d.get("note", ""),
    }


def load_1p7b_triviaqa(fold_seed=0):
    """1.7B / TriviaQA：h_L0..L27，逐层 CV 选峰值层（选择口径与主结果一致）。"""
    return load_probe_npz(REPO / "experiments" / "outputs" / "lin_theory" /
                          "detect_lr_probe_hidden.npz", fold_seed=fold_seed)


def load_1p7b_hellaswag():
    """1.7B / HellaSwag：可部署信号 = max_prob（输出面）；p_correct 为金标索引量，禁用。"""
    data = json.load(open(REPO / "experiments" / "phase2_entropy" / "outputs" /
                          "knowledge_filtered_data.json", encoding="utf-8"))
    y = np.asarray([int(d["is_correct"]) for d in data], dtype=np.int32)
    M = np.asarray([d["max_prob"] for d in data], dtype=float)      # (n, n_layers+1)
    scores = M[:, -1]                                               # max_p @L28（末层，报告口径）
    per_layer = {int(i): float(roc_auc_score(y, M[:, i])) for i in range(M.shape[1])}
    return scores, y, {
        "signal": "max_prob @L28 (output-surface, deployable)",
        "auroc_in_sample": float(roc_auc_score(y, scores)),
        "auroc_best_layer": max(per_layer.values()),
        "best_layer": int(max(per_layer, key=per_layer.get)),
        "note": "p_correct（金标索引）AUROC=0.872 不可作检测分数，仅用于知识筛选分析",
    }


def gated_net_vs_flag(blocks, beta="0.03", flag_rates=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """8B：门控净效应 vs 干预比例——flagged（低 P(correct)）= 施干预，其余保持基线。"""
    rows = []
    for fr in flag_rates:
        tot_n = tot_gain = tot_kw = tot_kc = 0
        for b in blocks:
            scores = b["scores"]
            y = b["y"]
            ar = b["archive"]["samples"]
            thr = np.quantile(scores, fr)          # 最低 fr 比例被 flag
            flagged = scores <= thr
            n = len(y)
            gain = kw = kc = 0
            for i, (sid, rec) in enumerate(ar.items()):
                base = int(b["y"][i])
                if not flagged[i]:
                    continue
                after = int(rec.get(f"correct_beta{beta}", base))
                gain += after - base
                kw += int(base == 0 and after == 1)
                kc += int(base == 1 and after == 0)
            tot_n += n
            tot_gain += gain
            tot_kw += kw
            tot_kc += kc
        rows.append({"flag_rate": fr, "n": tot_n, "net_pp": 100.0 * tot_gain / tot_n,
                     "kw_rescued": tot_kw, "kc_broken": tot_kc})
    return rows


def main():
    ap = argparse.ArgumentParser(description="可检测准确率 / 风险—覆盖率评估")
    ap.add_argument("--main_npz", type=str, default=None,
                    help="8B 主结果集逐层隐藏状态 npz（服务器回传 detect_lr_probe_hidden.npz）")
    ap.add_argument("--main_oof", type=str, default=None,
                    help="8B 主结果集逐样本折外分数 JSON（服务器回传 detect_lr_probe_oof.json）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"protocol": ("5-fold stratified CV, in-fold StandardScaler+LR, out-of-fold scores; "
                            "coverage = top-c by P(correct); exact-match labels; CP95 intervals"),
               "coverages": COVERAGES, "blocks": []}

    # ① 8B / TriviaQA（两个 seed，分别 + pooled）
    blocks = load_8b_triviaqa()
    seed_rows = []
    for b in blocks:
        b["scores"] = oof_scores(b["X"], b["y"], seed=b["seed"])   # 折种子=TLDC seed（门控协议）
        seed_rows.append(summarize(f"8B / TriviaQA / seed{b['seed']} (h_L28, LR probe)",
                                   b["scores"], b["y"], fold_seed=b["seed"],
                                   extra={"label_mismatch_vs_archive": b["label_mismatch"]}))
    Xp = np.vstack([b["X"] for b in blocks])
    yp = np.concatenate([b["y"] for b in blocks])
    sp = np.concatenate([b["scores"] for b in blocks])
    pooled = summarize("8B / TriviaQA / POOLED (2 seeds, n=600)", sp, yp, fold_seed=123)
    gate = gated_net_vs_flag(blocks, beta="0.03")
    print("\n--- 8B 门控净效应 vs 干预比例（β=0.03，双 seed 合并 n=600）---")
    for r in gate:
        print(f"  干预比例 {r['flag_rate']:.0%}: 净效应 {r['net_pp']:+.2f}pp  "
              f"(KW 救回 {r['kw_rescued']}, KC 破坏 {r['kc_broken']})")
    results["blocks"] += seed_rows + [pooled]
    results["gated_net_vs_flag_rate_8b_beta0.03"] = gate

    # ② 1.7B / TriviaQA（折种子 0，与 detect_lr_probe_cv.py 一致）
    s17, y17, meta17 = load_1p7b_triviaqa(fold_seed=0)
    results["blocks"].append(summarize("1.7B / TriviaQA (h_L%d, LR probe)" % meta17["peak_layer"],
                                       s17, y17, fold_seed=0, extra=meta17))

    # ②b 8B 检测主结果集（n=200 seed=42，与 0.8509@L28 同源）——由服务器回传后分析
    if args.main_npz:
        s, y, meta = load_probe_npz(Path(args.main_npz), fold_seed=0)
        results["blocks"].append(summarize(
            "8B / TriviaQA 主结果集 (n=%d, npz h_L%d)" % (len(y), meta["peak_layer"]),
            s, y, fold_seed=0, extra=meta))
    if args.main_oof:
        s, y, meta = load_oof_json(Path(args.main_oof))
        results["blocks"].append(summarize(
            "8B / TriviaQA 主结果集 (n=%d, OOF JSON, L%s)" % (len(y), meta.get("best_layer")),
            s, y, fold_seed=0, extra=meta))

    # ③ 1.7B / HellaSwag（输出面 max_prob）
    shs, yhs, metahs = load_1p7b_hellaswag()
    results["blocks"].append(summarize("1.7B / HellaSwag (max_prob, output surface)",
                                       shs, yhs, extra=metahs))

    out = OUT_DIR / "detectable_accuracy.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已保存：{out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
