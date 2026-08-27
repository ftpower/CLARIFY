#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate publication figures for the proposal / report from experiment JSONs.

Figures are regenerated from the experiment JSONs (single source of truth), so
after any re-run of the experiments, re-running this script refreshes every
figure — numbers are never hand-copied.

Usage (run with the project env, e.g. conda pytorch_env0):

    MPLCONFIGDIR=/tmp/mpl python docs/thesis/make_figures.py
    MPLCONFIGDIR=/tmp/mpl python docs/thesis/make_figures.py --only tldc

Output: docs/thesis/figures/<name>.png (300dpi) + <name>.pdf (vector).
Embed into the docx via 【图:<name>:图题】 placeholders in 开题报告草稿.txt,
handled by make_docx.py.

Figure inventory (chapter-5 numbering used in the proposal):
  fig_detection_layers    图 5.1 逐层线性探测 AUROC（TriviaQA，5 折 CV）
  fig_detection_rankfilter 图 5.2 知识筛选分组 AUROC（rank≤20/50/100）
  fig_detection_task      图 5.3 检测任务依赖性（TriviaQA vs HellaSwag）
  fig_tldc_dose           图 5.4 TLDC β 剂量-响应（双 seed，ΔKW/ΔKC + CP95 CI）
  fig_tldc_bars           图 5.5 TLDC vs 基线 分组合计（β=0.05，双 seed）
  fig_lora_beta_sweep     图 5.6 KL 正则强度 β 扫描（n=100/点，各类别 EM Δ）
  fig_lora_window_kl      图 5.7 窗口 KL 前后对比（n=1000：纯 CE vs KL）

⚠️ 已知的待重跑数字（P0，2026-08-26 未跑）：
  - Phase 24 β sweep 与窗口 KL 的 β=0 基线为修复前数字（见 fig_lora_window_kl 的
    BETA0_DOC 常量，SOURCE: docs/phase24-kl-tradeoff.md 表 1）；重跑后更新 JSON
    或常量即可。
  - HellaSwag 0.936 / 0.905 来自 phase4 旧管线（5 折 CV，协议合格，数据源
    comprehensive_analysis.json）；跨任务 0.54/0.66 标 ❌ 待重测，未入图。
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "thesis" / "figures"
L_THEORY = REPO / "experiments" / "outputs" / "lin_theory"
C2_FILE = REPO / "experiments" / "phase7_three_directions" / "outputs_phase7" / "C2_truth_direction.json"
PHASE4_FILE = REPO / "experiments" / "phase4_generalization" / "outputs" / "comprehensive_analysis.json"

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_SEED123 = "#1f77b4"
C_SEED456 = "#d62728"
C_GRAY = "#8c8c8c"
C_DARK = "#222222"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png / {name}.pdf")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1) detection
# ---------------------------------------------------------------------------
def fig_detection_layers():
    """图 5.1 逐层线性探测 AUROC（TriviaQA，5 折 CV）。"""
    print("fig_detection_layers: reading", L_THEORY / "detect_lr_probe_cv.json", "&", C2_FILE)
    d = _load(L_THEORY / "detect_lr_probe_cv.json")
    c2 = _load(C2_FILE)

    layers = [p["layer"] for p in d["per_layer"]]
    auroc = np.array([p["auroc"] for p in d["per_layer"]])
    std = np.array([p["auroc_std"] for p in d["per_layer"]])

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(layers, auroc, color=C_SEED123, lw=1.6, marker="o", ms=3,
            label="LR probe (per-layer)")
    ax.fill_between(layers, auroc - std, auroc + std, color=C_SEED123, alpha=0.15,
                    label="±1 std (5-fold CV)")
    # best probe layer
    b = d["best_layer"]
    ax.annotate(f"best L{b} = {d['best_auroc']:.4f}",
                xy=(b, d["best_auroc"]), xytext=(b - 16, d["best_auroc"] + 0.045),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_DARK),
                fontsize=8.5)
    # truth direction reference (single best-layer value, clean protocol)
    ax.axhline(c2["best_auroc"], color=C_SEED456, ls="--", lw=1.2)
    ax.text(27.5, c2["best_auroc"] + 0.008, f"truth direction L{c2['best_layer']}\n= {c2['best_auroc']:.4f}",
            ha="right", va="bottom", fontsize=8, color=C_SEED456)
    ax.axhline(0.5, color=C_GRAY, ls=":", lw=1)
    ax.text(0.3, 0.505, "chance", fontsize=8, color=C_GRAY)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.45, 0.9)
    ax.set_xticks(range(0, 28, 2))
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_detection_layers")


def fig_detection_rankfilter():
    """图 5.2 知识筛选分组 AUROC（rank≤20/50/100 vs 全样本）。"""
    print("fig_detection_rankfilter: reading", L_THEORY / "detect_lr_probe_rankfilter.json")
    d = _load(L_THEORY / "detect_lr_probe_rankfilter.json")
    groups = d["groups"]
    order = ["full", "rank_le_20", "rank_le_50", "rank_le_100"]
    labels = ["full\n(n=200)", "rank≤20\n(n=72)", "rank≤50\n(n=93)", "rank≤100\n(n=110)"]
    vals = [groups[k]["best_auroc"] for k in order]
    errs = [groups[k]["best_auroc_std"] for k in order]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.arange(len(order))
    ax.bar(x, vals, yerr=errs, width=0.6, color=C_SEED123, alpha=0.85,
           capsize=4, error_kw=dict(lw=1))
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
    # full-sample reference + "no gain" bracket
    ax.axhline(groups["full"]["best_auroc"], color=C_GRAY, ls=":", lw=1)
    ax.annotate("", xy=(0.15, groups["full"]["best_auroc"] - 0.05),
                xytext=(0.15, groups["full"]["best_auroc"] + 0.055),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=C_DARK))
    ax.text(0.3, groups["full"]["best_auroc"] + 0.062, "no gain vs full",
            fontsize=8, color=C_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Best-layer AUROC (5-fold CV)")
    ax.set_ylim(0.55, 0.95)
    ax.set_title("Knowledge-rank filtering (TriviaQA, 1.7B)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig_detection_rankfilter")


def fig_detection_task():
    """图 5.3 检测任务依赖性：TriviaQA vs HellaSwag × 内部表示/表面特征。"""
    print("fig_detection_task: reading", L_THEORY / "detect_lr_probe_cv.json",
          "&", L_THEORY / "detect_js_lr_cv.json", "&", PHASE4_FILE)
    lr = _load(L_THEORY / "detect_lr_probe_cv.json")
    js = _load(L_THEORY / "detect_js_lr_cv.json")
    p4 = _load(PHASE4_FILE)
    p4v = p4["evaluation_matrix"]["values"]

    # TriviaQA（修复后协议）：LR probe 0.7708；表面特征 = 7 特征 joint（0.607）
    tqa_internal = lr["best_auroc"]
    tqa_surface = js["joint"]["joint_all"]["mean"]
    # HellaSwag（phase4 5 折 CV 矩阵，行 index 0 = HellaSwag 1.7B train+CV）：
    # joint_lr 0.936、max_p 0.905（见 code-review-2026-08-24.md「JS/LR 检测重测」对照节）
    hs_internal = p4v["joint_lr,0"]
    hs_surface = p4v["max_p,0"]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    methods = ["Internal repr.\n(LR probe / joint LR)", "Surface\n(max_p / joint)"]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, [tqa_internal, tqa_surface], w, label="TriviaQA (open-gen)",
                color=C_SEED123, alpha=0.9)
    b2 = ax.bar(x + w / 2, [hs_internal, hs_surface], w, label="HellaSwag (multi-choice)",
                color=C_SEED456, alpha=0.9)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.85, color=C_GRAY, ls=":", lw=1.2)
    ax.text(0.55, 0.962, "target 0.85", fontsize=8, color=C_GRAY, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Detection is task-dependent (Qwen3-1.7B)", fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.subplots_adjust(right=0.68)
    fig.tight_layout()
    _save(fig, "fig_detection_task")


# ---------------------------------------------------------------------------
# 2) TLDC
# ---------------------------------------------------------------------------
# 注意：seed123 的 s14_tldc.json 摘要曾被后续运行覆盖（只留 β∈{0.0,0.1,0.2}），
# 但 per-sample 档案 s14_tldc_samples.json 含全部 β 的逐样本正确性，可无损重建
# 完整扫描摘要（两 seed 均验证 0 mismatch）。因此本脚本统一从 samples 档案重建，
# 存储摘要仅作交叉校验。

def _tldc_from_samples(samples_path, summary_path=None):
    """Rebuild the per-β TLDC summary from the per-sample archive.

    Returns (out, betas, baseline) where
      out[(subset, beta_str)] = {rate, delta, n, ci95}  (CP95 Clopper-Pearson)
      baseline[subset]        = baseline correctness rate
      betas                   = sorted β strings, e.g. ["0.01", ..., "0.20"]
    """
    d = _load(samples_path)
    samples = list(d["samples"].values())
    prefix = "correct_beta"
    betas = sorted({k[len(prefix):] for k in samples[0].keys()
                    if k.startswith(prefix)}, key=float)
    subs = ["know_wrong", "know_correct", "dont_know"]
    by_sub = {s: [x for x in samples if x["subset"] == s] for s in subs}
    base = {s: float(np.mean([x["baseline_correct"] for x in xs]))
            for s, xs in by_sub.items()}
    base["all"] = float(np.mean([x["baseline_correct"] for x in samples]))

    out = {}
    for b in betas:
        for s, xs in by_sub.items():
            corr = np.array([x[prefix + b] for x in xs], dtype=float)
            out[(s, b)] = {"rate": float(corr.mean()),
                           "delta": float(corr.mean()) - base[s],
                           "n": len(corr)}
        allc = np.array([x[prefix + b] for x in samples], dtype=float)
        out[("all", b)] = {"rate": float(allc.mean()),
                           "delta": float(allc.mean()) - base["all"],
                           "n": len(allc)}
    # CP95 exact Clopper-Pearson CI of the rate (baseline KW/KC are definitional
    # 0/1, so the rate CI is also the delta CI for the two headline subsets).
    for (s, b), v in out.items():
        n, x = v["n"], round(v["rate"] * v["n"])
        lo = beta_dist.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
        hi = beta_dist.ppf(0.975, x + 1, n - x) if x < n else 1.0
        v["ci95"] = [float(lo), float(hi)]
    # cross-check against the stored summary where its keys still exist
    if summary_path is not None and summary_path.exists():
        d2 = _load(summary_path)
        for k, v in d2["results"]["betas"].items():
            b = k.split("=", 1)[1]
            if b not in betas:
                continue
            for s in subs + ["all"]:
                if (s, b) not in out:
                    continue
                r, delta = out[(s, b)]["rate"], out[(s, b)]["delta"]
                if abs(r - v[s]["rate"]) > 1e-9 or abs(delta - v[s]["delta"]) > 1e-9:
                    print(f"  ⚠️ mismatch vs {summary_path.name}: {s} β={b}")
    return out, betas, base


def _best_beta_for_bars(seed_data):
    """Pick the β where BOTH seeds' ΔKW CP95 lower bound > 0 (else fallback 0.05)."""
    cands = []
    for beta_str in ("0.05", "0.03", "0.08", "0.10", "0.15", "0.20", "0.01"):
        if all(out[("know_wrong", beta_str)]["ci95"][0] > 0
               for _, out, _ in seed_data):
            cands.append(beta_str)
    return cands[0] if cands else "0.05"


def fig_tldc_dose():
    """图 5.4 TLDC β 剂量-响应（双 seed，ΔKW 左轴 / ΔKC 右轴，CP95 CI 带）。"""
    seeds = [("seed 123", "s14_tldc_samples.json", "s14_tldc.json", C_SEED123),
             ("seed 456", "seed456/s14_tldc_samples.json", "seed456/s14_tldc.json", C_SEED456)]
    print("fig_tldc_dose: rebuilding from per-sample archives (verified 0-mismatch):")
    seed_data = []
    for label, samp, summ, color in seeds:
        print(f"  {label}: {samp} + cross-check {summ}")
        out, betas, base = _tldc_from_samples(L_THEORY / samp, L_THEORY / summ)
        seed_data.append((label, out, betas, color))

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    for label, out, betas, color in seed_data:
        xs = np.array([float(b) for b in betas])
        kw_d = np.array([out[("know_wrong", b)]["delta"] for b in betas])
        kw_lo = np.array([out[("know_wrong", b)]["ci95"][0] for b in betas])
        kw_hi = np.array([out[("know_wrong", b)]["ci95"][1] for b in betas])
        kc_d = np.array([out[("know_correct", b)]["delta"] for b in betas])
        kc_lo = np.array([out[("know_correct", b)]["ci95"][0] for b in betas])
        kc_hi = np.array([out[("know_correct", b)]["ci95"][1] for b in betas])
        ax.plot(xs, kw_d * 100, color=color, marker="o", ms=3.5, lw=1.5,
                label=f"{label}: ΔKW (know-wrong rescue)")
        ax.fill_between(xs, kw_lo * 100, kw_hi * 100, color=color, alpha=0.12)
        ax2.plot(xs, (kc_d) * 100, color=color, marker="s", ms=3.5, lw=1.2,
                 ls="--", label=f"{label}: ΔKC (know-correct cost)")
        ax2.fill_between(xs, (kc_lo - 0) * 100, (kc_hi - 0) * 100, color=color, alpha=0.08)

    ax.axhline(0, color=C_DARK, lw=0.8)
    ax2.axhline(0, color=C_DARK, lw=0.8)
    # KC cost criterion line
    ax2.axhline(-5, color=C_GRAY, ls=":", lw=1)
    ax2.text(0.205, -6.2, "KC cost ≤5% criterion", fontsize=7.5, color=C_GRAY, ha="right")

    ax.set_xlabel("β (TLDC perturbation strength)")
    ax.set_ylabel("Δ rate, know-wrong (pp)")
    ax2.set_ylabel("Δ rate, know-correct (pp)")
    ax.set_ylim(-2, 14)
    ax2.set_ylim(-30, 8)
    ax.set_xticks([0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2])
    ax.set_xticklabels(["0.01", "0.03", "0.05", "0.08", "0.10", "0.15", "0.20"])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_tldc_dose")


def fig_tldc_bars():
    """图 5.5 TLDC vs 基线 分组合计（β=0.05，双 seed 面板）。"""
    seeds = [("seed 123", "s14_tldc_samples.json", "s14_tldc.json"),
             ("seed 456", "seed456/s14_tldc_samples.json", "seed456/s14_tldc.json")]
    print("fig_tldc_bars: rebuilding from per-sample archives")
    seed_data = []
    for label, samp, summ in seeds:
        out, betas, base = _tldc_from_samples(L_THEORY / samp, L_THEORY / summ)
        seed_data.append((label, out, base))
    beta_str = _best_beta_for_bars(seed_data)
    print(f"  showcase β = {beta_str} (both seeds' ΔKW CI lower bound > 0)")

    cats = [("KW", "know_wrong"), ("KC", "know_correct"), ("DK", "dont_know"), ("All", "all")]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, (label, out, base) in zip(axes, seed_data):
        names = [c[0] for c in cats]
        bl = [base[c[1]] * 100 for c in cats]
        tl = [out[(c[1], beta_str)]["rate"] * 100 for c in cats]
        tl_lo = [out[(c[1], beta_str)]["ci95"][0] * 100 for c in cats]
        tl_hi = [out[(c[1], beta_str)]["ci95"][1] * 100 for c in cats]
        x = np.arange(len(names))
        w = 0.34
        ax.bar(x - w / 2, bl, w, label="Baseline", color=C_GRAY, alpha=0.75)
        ax.bar(x + w / 2, tl, w,
               yerr=[np.array(tl) - np.array(tl_lo), np.array(tl_hi) - np.array(tl)],
               label=f"TLDC β={beta_str}", color=C_SEED123, alpha=0.9, capsize=3,
               error_kw=dict(lw=0.9))
        for xi, v in zip(x, tl):
            ax.text(xi + w / 2, v + 1.8, f"{v:.0f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(f"{label} (n=300)", fontsize=9)
        ax.set_ylim(0, 112)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("TLDC at β where both seeds' ΔKW CI lower bound > 0", fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_tldc_bars")


# ---------------------------------------------------------------------------
# 3) LoRA / Phase 24
# ---------------------------------------------------------------------------
SWEEP_FILES = [
    ("0.03", "s24_kl0.03.json"),
    ("0.05", "s24_kl0.05.json"),
    ("0.1", "s24_kl0.1.json"),
    ("0.3", "s24_kl0.3_n100.json"),
    ("1.0", "s24_kl1.0.json"),
    ("3.0", "s24_kl3.0.json"),
]
# β=0（纯 CE）n=1000 基线，SOURCE: docs/phase24-kl-tradeoff.md 表 1（2026-08 修复前数字，
# s24_kl0.0.json 缺失；P0 β sweep 重跑后应改为读 JSON）。
BETA0_DOC = {"KW": 16, "KC": -50, "net": -34}


def _s24_cat_deltas_pp(path):
    """Per-category EM rate delta in percentage points: 100*(lo_em - bl_em)/n."""
    d = _load(path)
    pc = d["summary"]["per_category"]
    out = {}
    for cat, v in pc.items():
        out[cat] = 100.0 * (v["lo_em"] - v["bl_em"]) / v["n"]
    return out


def fig_lora_beta_sweep():
    """图 5.6 KL 正则强度 β 扫描（n=100/点，KC/KW/净效应 EM 率差，pp）。"""
    print("fig_lora_beta_sweep: reading", ", ".join(f"{p}" for _, p in SWEEP_FILES))
    betas, kc, kw, net = [], [], [], []
    for b, f in SWEEP_FILES:
        d = _s24_cat_deltas_pp(L_THEORY / f)
        betas.append(float(b))
        kw.append(d["KW"])
        kc.append(d["KC"])
        net.append(d["KW"] + d["KC"])
    betas = np.array(betas)

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(betas, kw, color=C_SEED123, marker="o", ms=4, lw=1.5, label="ΔKW (fix wrong answers)")
    ax.plot(betas, kc, color=C_SEED456, marker="s", ms=4, lw=1.5, label="ΔKC (forgetting cost)")
    ax.plot(betas, net, color=C_DARK, marker="^", ms=4, lw=1.5, ls="--", label="net = ΔKW + ΔKC")
    ax.axhline(0, color=C_GRAY, lw=0.8)
    for xi, v in zip(betas, net):
        ax.annotate(f"{v:+.0f}", (xi, v), textcoords="offset points",
                    xytext=(0, -11), ha="center", fontsize=7.5, color=C_DARK)
    ax.set_xscale("log")
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlabel("KL regularization strength β (log scale)")
    ax.set_ylabel("EM rate Δ vs baseline (pp)")
    # y-range auto-fit to the data (β=0.05 ΔKC dips to ≈ -32pp)
    allv = np.concatenate([kw, kc, net])
    pad = max((allv.max() - allv.min()) * 0.15, 3.0)
    ax.set_ylim(allv.min() - pad, allv.max() + pad)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig_lora_beta_sweep")


def fig_lora_window_kl():
    """图 5.7 窗口 KL 前后对比（n=1000：纯 CE vs KL，EM 计数差）。"""
    print("fig_lora_window_kl: reading", L_THEORY / "s24_kl0.3.json",
          "+ β=0 constants from docs/phase24-kl-tradeoff.md (pre-rerun)")
    d = _load(L_THEORY / "s24_kl0.3.json")
    pc = d["summary"]["per_category"]
    kl = {"KW": pc["KW"]["lo_em"] - pc["KW"]["bl_em"],
          "KC": pc["KC"]["lo_em"] - pc["KC"]["bl_em"],
          "net": (pc["KW"]["lo_em"] - pc["KW"]["bl_em"]) + (pc["KC"]["lo_em"] - pc["KC"]["bl_em"])}
    bl0 = BETA0_DOC

    names = ["KW fixed", "KC forgotten", "net (KW + KC)"]
    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x - w / 2, [bl0["KW"], bl0["KC"], bl0["net"]], w, label="β=0 (pure CE)",
           color=C_GRAY, alpha=0.8)
    ax.bar(x + w / 2, [kl["KW"], kl["KC"], kl["net"]], w, label="window KL β=0.3",
           color=C_SEED123, alpha=0.9)
    for xi, (v0, v1) in enumerate(zip([bl0["KW"], bl0["KC"], bl0["net"]],
                                      [kl["KW"], kl["KC"], kl["net"]])):
        ax.text(xi - w / 2, v0 + (1.5 if v0 >= 0 else -4.5), f"{v0:+d}", ha="center", fontsize=8)
        ax.text(xi + w / 2, v1 + (1.5 if v1 >= 0 else -4.5), f"{v1:+d}", ha="center", fontsize=8)
    ax.axhline(0, color=C_DARK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Δ EM count (n=1000)")
    ax.set_ylim(-70, 25)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig_lora_window_kl")


# ---------------------------------------------------------------------------
ALL_FIGS = {
    "detection": [fig_detection_layers, fig_detection_rankfilter, fig_detection_task],
    "tldc": [fig_tldc_dose, fig_tldc_bars],
    "lora": [fig_lora_beta_sweep, fig_lora_window_kl],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=list(ALL_FIGS),
                    help="regenerate only this group (repeatable); default: all")
    args = ap.parse_args()
    groups = args.only or list(ALL_FIGS)
    print(f"Output dir: {OUT}")
    for g in groups:
        print(f"== {g} ==")
        for fn in ALL_FIGS[g]:
            fn()
    print("done.")


if __name__ == "__main__":
    main()
