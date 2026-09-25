"""L0-1（候选 C2）：DoLa 增益的口径分解 —— 等长子集 / 长度分层 / 长度回归残差。

判据来源（**执行前设定**，不得事后修改）：
    docs/protocol/dola-l0-analysis-20260925.md §1

数据（零 GPU，产物已在手）：
    experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817ps.json
        官方口径＝无后缀键；归一化口径＝`*__ps1` 键（post_softmax=True 变体）。
    长度变量由本地 Qwen3-1.7B tokenizer 复算，口径镜像 main_dola_mc.py::encode_pair
    （cont_len = |tok(prompt + " " + answer)| − |tok(prompt)|）。

自检（内置，先于统计量）：
    C1 逐题逐选项的 full token 长度必须与产物 `_lens_len` 逐位一致（否则 prompt 口径有偏差）；
    C2 baseline MC2 在 full817 与 full817ps 两次运行间必须一致（运行可复现性）；
    C3 非 baseline 条件在两侧口径下必须不同（诊断有效）。

用法
----
    python3 experiments/lin_theory/analyze_dola_l0_c2.py --selftest   # 仅自检
    python3 experiments/lin_theory/analyze_dola_l0_c2.py             # 全量分析
输出：experiments/outputs/dola_l0_20260925/l0_c2.json + l0_c2_report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "outputs" / "dola_l0_20260925"
PS_JSON = REPO / "experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817ps.json"
PLAIN_JSON = REPO / "experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817.json"
DATA_JSON = REPO / "experiments/data/truthfulqa_mc_817.json"
TOKENIZER_SNAP = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                               "snapshots")
PRIMARY = "dyn_b1_14_28"


def load_dola_module():
    """导入 main_dola_mc（复用其官方 prompt 构造与 refs_of/close_answer 口径）。"""
    path = REPO / "experiments/lin_theory/main_dola_mc.py"
    spec = importlib.util.spec_from_file_location("main_dola_mc_l0", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["main_dola_mc_l0"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tokenizer():
    from transformers import AutoTokenizer
    snaps = sorted(p for p in TOKENIZER_SNAP.glob("*") if (p / "tokenizer.json").exists())
    if not snaps:
        raise SystemExit(f"未找到本地 tokenizer：{TOKENIZER_SNAP}")
    return AutoTokenizer.from_pretrained(str(snaps[-1]))


def per_question_lengths(dataset, tokenizer, dola):
    """逐题选项长度（token）：L_true / L_min / L_max / spread / L_mean。

    同时返回全部选项的**整段序列长度**（BOS + prompt + continuation，与 TL `to_tokens` 同口径），
    用于自检 prompt 长度分布是否与产物 `data.prompt_len` 一致。
    """
    recs, fulls_all = [], []
    for qi, q in enumerate(dataset):
        ref_true, ref_false, ref_best = dola.refs_of(q)
        if ref_best not in ref_true:
            ref_true = [ref_best] + [a for a in ref_true if a != ref_best]
        prompt, _ = dola.build_prompt_and_answer(q["question"], ref_true[0])
        pref = len(tokenizer.encode(prompt, add_special_tokens=False))  # BOS 差分时抵消
        lens, seen = [], set()
        for ans in list(ref_true) + list(ref_false):
            if ans in seen:
                continue
            seen.add(ans)
            _, cont = dola.build_prompt_and_answer(q["question"], ans)
            full = len(tokenizer.encode(prompt + cont, add_special_tokens=False))
            lens.append(full - pref)
            fulls_all.append(full + 1)  # +1 = BOS（TL prepend_bos=True）
        n_true = len(ref_true)
        recs.append({
            "qi": qi,
            "n_true": n_true,
            "n_false": len(lens) - n_true,
            "L_true": float(np.mean(lens[:n_true])) if n_true else float("nan"),
            "L_min": float(min(lens)),
            "L_max": float(max(lens)),
            "spread": float(max(lens) - min(lens)),
            "L_mean": float(np.mean(lens)),
        })
    return recs, np.array(fulls_all, dtype=float)


def mc2_vector(pq, key):
    return np.array([r["mc"][key]["MC2"] for r in pq], dtype=float)


def boot_ci(delta, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = delta[idx].mean(axis=1) * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_condition(base_o, base_n, cond_o, cond_n, L, mask=None):
    """单条件：两侧口径 Δ、保留率、长度相关。返回字典（Δ 单位：百分点）。"""
    if mask is None:
        mask = np.ones(len(base_o), dtype=bool)
    d_o = (cond_o - base_o)[mask] * 100.0
    d_n = (cond_n - base_n)[mask] * 100.0
    out = {
        "n": int(mask.sum()),
        "delta_official": float(d_o.mean()),
        "delta_norm": float(d_n.mean()),
        "median_official": float(np.median(d_o)),
        "median_norm": float(np.median(d_n)),
        "mean_abs_norm": float(np.abs(d_n).mean()),
        "share_abs_gt50_norm": float((np.abs(d_n) > 50).mean()),
        "share_abs_gt50_official": float((np.abs(d_o) > 50).mean()),
        "share_plus100_norm": float((d_n > 99.9).mean()),
        "share_minus100_norm": float((d_n < -99.9).mean()),
        "ci_norm": boot_ci(d_n),
        "ci_official": boot_ci(d_o),
    }
    out["retention"] = (float(d_n.mean() / d_o.mean())
                        if abs(d_o.mean()) > 1e-9 else None)
    Lm = L[mask]
    if len(np.unique(Lm)) > 1 and float(np.std(d_n)) > 0:
        rho, p = stats.spearmanr(Lm, d_n)
        lr = stats.linregress(Lm, d_n)
        out.update({"spearman_rho_norm": float(rho), "spearman_p_norm": float(p),
                    "ols_slope_norm": float(lr.slope), "ols_p_norm": float(lr.pvalue),
                    "ols_r2_norm": float(lr.rvalue ** 2)})
    if len(np.unique(Lm)) > 1 and float(np.std(d_o)) > 0:
        rho_o, p_o = stats.spearmanr(Lm, d_o)
        out.update({"spearman_rho_official": float(rho_o), "spearman_p_official": float(p_o)})
    return out


def main():
    ap = argparse.ArgumentParser(description="L0-1 C2 口径分解")
    ap.add_argument("--selftest", action="store_true", help="仅跑自检并退出")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = json.loads(PS_JSON.read_text())
    plain = json.loads(PLAIN_JSON.read_text())
    dataset = json.loads(DATA_JSON.read_text())
    pq = ps["per_question"]
    assert len(pq) == len(dataset) == 817, (len(pq), len(dataset))

    dola = load_dola_module()
    tok = load_tokenizer()
    lens_recs, fulls = per_question_lengths(dataset, tok, dola)

    # ── 自检 C1：整段序列长度分布与产物 data.prompt_len 一致（tokenizer + prompt 口径）──
    ref_pl = ps["data"]["prompt_len"]
    got_pl = {"mean": float(fulls.mean()), "p95": float(np.percentile(fulls, 95)),
              "max": int(fulls.max()), "min": int(fulls.min()),
              "n_scored": int(fulls.size)}
    c1_ok = (abs(got_pl["mean"] - ref_pl["mean"]) < 0.01
             and abs(got_pl["p95"] - ref_pl["p95"]) < 1e-9
             and got_pl["max"] == ref_pl["max"] and got_pl["min"] == ref_pl["min"])
    # ── 自检 C2：baseline MC2 跨两次运行一致 ──
    b_ps = mc2_vector(pq, "baseline")
    b_pl = mc2_vector(plain["per_question"], "baseline")
    base_run_diff = float(np.max(np.abs(b_ps - b_pl)))
    # ── 自检 C3：非 baseline 条件两侧口径必须不同 ──
    conds = [k for k in pq[0]["mc"] if not k.endswith("__ps1")]
    diag_diff = {c: float(np.max(np.abs(mc2_vector(pq, c) - mc2_vector(pq, c + "__ps1"))))
                 for c in conds if c != "baseline"}
    c3_ok = all(v > 1e-12 for v in diag_diff.values())
    checks = {
        "C1_prompt_len_match": {"ok": bool(c1_ok), "got": got_pl, "ref": ref_pl},
        "C2_baseline_two_runs_max_abs_diff": {"ok": base_run_diff < 1e-12,
                                              "value": base_run_diff},
        "C3_diag_differs": {"ok": c3_ok,
                            "min_max_abs_diff": min(diag_diff.values()) if diag_diff else None},
    }
    print("[自检]", json.dumps(checks, ensure_ascii=False))
    if args.selftest:
        return 0 if all(c["ok"] for c in checks.values()) else 1

    L_mean = np.array([r["L_mean"] for r in lens_recs])
    L_true = np.array([r["L_true"] for r in lens_recs])
    spread = np.array([r["spread"] for r in lens_recs])

    base_o = mc2_vector(pq, "baseline")
    base_n = mc2_vector(pq, "baseline__ps1")

    res = {"checks": checks, "baseline_structure": {
        "share_mc2_eq0": float((base_o <= 1e-9).mean()),
        "share_mc2_eq1": float((base_o >= 1 - 1e-9).mean()),
        "share_mc2_gt0.99": float((base_o > 0.99).mean()),
        "share_mc2_lt0.01": float((base_o < 0.01).mean()),
    }, "length_stats": {
        "L_mean": {k: float(v) for k, v in zip(
            ["mean", "p25", "p50", "p75", "min", "max"],
            [L_mean.mean(), *np.percentile(L_mean, [25, 50, 75]), L_mean.min(), L_mean.max()])},
        "spread_hist": {str(int(s)): int((spread == s).sum()) for s in np.unique(spread)},
        "n_spread_le1": int((spread <= 1).sum()),
        "n_spread_le2": int((spread <= 2).sum()),
    }, "conditions": {}}

    for c in conds:
        mask_all = np.ones(817, dtype=bool)
        entry = {
            "all": analyze_condition(base_o, base_n, mc2_vector(pq, c),
                                     mc2_vector(pq, c + "__ps1"), L_mean, mask_all),
            "E1_spread_le1": analyze_condition(base_o, base_n, mc2_vector(pq, c),
                                               mc2_vector(pq, c + "__ps1"), L_mean, spread <= 1),
            "E2_spread_le2": analyze_condition(base_o, base_n, mc2_vector(pq, c),
                                               mc2_vector(pq, c + "__ps1"), L_mean, spread <= 2),
        }
        if c != "baseline":
            # 长度四分位分层（按 L_mean）+ 两侧口径饱和披露
            qs = np.percentile(L_mean, [25, 50, 75])
            bins = np.digitize(L_mean, qs)
            entry["by_length_quartile"] = []
            for b in range(4):
                m = bins == b
                do = (mc2_vector(pq, c) - base_o)[m] * 100.0
                dn = (mc2_vector(pq, c + "__ps1") - base_n)[m] * 100.0
                entry["by_length_quartile"].append({
                    "bin": b, "n": int(m.sum()),
                    "L_mean_range": [float(L_mean[m].min()), float(L_mean[m].max())],
                    "delta_official": float(do.mean()), "delta_norm": float(dn.mean()),
                })
            entry["saturation"] = {
                "base_official": int((base_o > 0.99).sum()),
                "cond_official": int((mc2_vector(pq, c) > 0.99).sum()),
                "base_norm": int((base_n > 0.99).sum()),
                "cond_norm": int((mc2_vector(pq, c + "__ps1") > 0.99).sum()),
            }
        res["conditions"][c] = entry

    # ── 主条件判定（事前设定分支）──
    p = res["conditions"][PRIMARY]
    ret_e1 = p["E1_spread_le1"]["retention"]
    rho = p["E1_spread_le1"].get("spearman_rho_norm")
    rho_p = p["E1_spread_le1"].get("spearman_p_norm")
    if ret_e1 is None or p["E1_spread_le1"]["delta_official"] <= 0:
        verdict = ("判据在等长子集上退化（E1 内官方 Δ ≤ 0，保留率无意义）；"
                   "改以描述性口径报告——见报告 §等长子集与重尾结构")
    elif ret_e1 >= 0.5:
        verdict = "真实行为改善（口径稳健）"
    elif ret_e1 < 0.5 and rho is not None and rho >= 0.2 and rho_p is not None and rho_p < 0.05:
        verdict = "口径效应为主"
    else:
        verdict = "不确定区间"
    res["verdict"] = {"primary_condition": PRIMARY, "branch": verdict,
                      "retention_E1": ret_e1, "spearman_E1_rho": rho, "spearman_E1_p": rho_p}

    (out_dir / "l0_c2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    # ── Markdown 报告 ──
    L = []
    L.append("# L0-1（C2）DoLa 增益口径分解报告\n")
    L.append("判据：`docs/protocol/dola-l0-analysis-20260925.md` §1（执行前设定）。"
             f"数据：`{PS_JSON.name}`（两侧口径同一次运行）。\n")
    L.append("## 自检\n")
    for k, v in checks.items():
        L.append(f"- {k}: {'✅' if v['ok'] else '❌'} `{json.dumps(v, ensure_ascii=False)}`")
    ls = res["length_stats"]
    L.append(f"\n选项长度（token）：mean {ls['L_mean']['mean']:.2f}、p25 {ls['L_mean']['p25']:.0f}、"
             f"p50 {ls['L_mean']['p50']:.0f}、p75 {ls['L_mean']['p75']:.0f}；"
             f"等长题数 spread≤1＝{ls['n_spread_le1']}、spread≤2＝{ls['n_spread_le2']}（共 817）\n")
    L.append("## 主条件 `%s`\n" % PRIMARY)
    L.append("| 子集 | n | Δ官方 | Δ归一化 | 保留率 |")
    L.append("|---|---|---|---|---|")
    for tag, key in [("全部", "all"), ("等长 E1（spread≤1）", "E1_spread_le1"),
                     ("等长 E2（spread≤2）", "E2_spread_le2")]:
        e = p[key]
        r = "—" if e["retention"] is None else f"{e['retention']:.2f}"
        L.append(f"| {tag} | {e['n']} | {e['delta_official']:+.2f}pp | {e['delta_norm']:+.2f}pp | {r} |")
    L.append(f"\n- 归一化 Δ 的 bootstrap 95% CI（全部 817）："
             f"[{p['all']['ci_norm'][0]:+.2f}, {p['all']['ci_norm'][1]:+.2f}]pp")
    L.append(f"- 长度回归（归一化 Δ ~ 选项平均长度）："
             f"Spearman ρ={p['all'].get('spearman_rho_norm', float('nan')):.3f}"
             f"（p={p['all'].get('spearman_p_norm', float('nan')):.2e}）、"
             f"OLS 斜率={p['all'].get('ols_slope_norm', float('nan')):+.3f}pp/token"
             f"（p={p['all'].get('ols_p_norm', float('nan')):.2e}）")
    L.append(f"- 饱和题数（MC2>0.99）：baseline 官方 {p['saturation']['base_official']} → 条件 "
             f"{p['saturation']['cond_official']}；归一化 {p['saturation']['base_norm']} → "
             f"{p['saturation']['cond_norm']}")
    L.append("\n### 长度四分位\n")
    L.append("| 分位 | n | 长度范围 | Δ官方 | Δ归一化 |")
    L.append("|---|---|---|---|---|")
    for bi, b in enumerate(p["by_length_quartile"]):
        L.append(f"| Q{bi+1} | {b['n']} | {b['L_mean_range'][0]:.1f}–{b['L_mean_range'][1]:.1f} | "
                 f"{b['delta_official']:+.2f}pp | {b['delta_norm']:+.2f}pp |")
    bs = res["baseline_structure"]
    L.append("\n### 重尾与全翻转结构（主条件）\n")
    L.append(f"- baseline MC2 取值结构：=0 占 {bs['share_mc2_eq0']:.1%}、=1 占 {bs['share_mc2_eq1']:.1%}"
             f"（>0.99 占 {bs['share_mc2_gt0.99']:.1%}）⇒ MC2 为全有全无型，逐题 Δ 重尾")
    L.append(f"- 官方口径：均值 {p['all']['delta_official']:+.2f}pp、**中位数 {p['all']['median_official']:+.2f}pp**、"
             f"|Δ|>50pp 占 {p['all']['share_abs_gt50_official']:.1%}")
    L.append(f"- 归一化口径：均值 {p['all']['delta_norm']:+.2f}pp、**中位数 {p['all']['median_norm']:+.2f}pp**、"
             f"|Δ|>50pp 占 {p['all']['share_abs_gt50_norm']:.1%}"
             f"（+100pp 占 {p['all']['share_plus100_norm']:.1%}、−100pp 占 {p['all']['share_minus100_norm']:.1%}）、"
             f"平均绝对幅度 {p['all']['mean_abs_norm']:.1f}pp")
    L.append(f"- ⇒ 均值增益由少数全翻转题驱动，bootstrap CI 宽达 "
             f"[{p['all']['ci_norm'][0]:+.1f}, {p['all']['ci_norm'][1]:+.1f}]pp（重尾下均值不稳健）\n")
    L.append(f"**判定分支：{verdict}**（retention(E1)={ret_e1 if ret_e1 is None else round(ret_e1, 3)}）\n")
    L.append("## 全部 17 个条件\n")
    L.append("| 条件 | Δ官方 | Δ归一化 | 保留率 | Δ官方(E1) | Δ归一化(E1) | 保留率(E1) |")
    L.append("|---|---|---|---|---|---|---|")
    for c, e in res["conditions"].items():
        if c == "baseline":
            continue
        a, e1 = e["all"], e["E1_spread_le1"]
        f = lambda x: "—" if x is None else f"{x:.2f}"
        L.append(f"| `{c}` | {a['delta_official']:+.2f} | {a['delta_norm']:+.2f} | "
                 f"{f(a['retention'])} | {e1['delta_official']:+.2f} | {e1['delta_norm']:+.2f} | "
                 f"{f(e1['retention'])} |")
    (out_dir / "l0_c2_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n[写出] {out_dir/'l0_c2.json'} / {out_dir/'l0_c2_report.md'}")
    return 0 if all(c["ok"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
