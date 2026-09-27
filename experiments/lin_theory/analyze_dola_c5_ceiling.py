"""C5（DoLa 合成路线）完美选择上界复算 —— U0／A*／U1／U2 框架（零 GPU）。

判据来源（**执行前设定**，不得事后修改）：
    docs/protocol/dola-c5-ceiling-20260927.md §1–§3

背景：C1（闭式对比强度）已在生成域与原生域两处关闭（DoLa 增益为口径属性，不可被闭式强度改良）。
本脚本执行 C5 的「最低验证」＝把 `docs/protocol/intervention-line-verdict-20260923.md` §1 的
U0／U1／U2 上限框架移植到 DoLa 原生域（TruthfulQA-MC，817 题）。

事件口径（题级，沿用产物 MC1 定义）：
    救回 R(c) = #{q : baseline MC1=0 ∧ c MC1=1}
    破坏 B(c) = #{q : baseline MC1=1 ∧ c MC1=0}
    净事件 net(c) = R − B；Δpp = net / n × 100

候选条件集合：
    U0        = 已发布主条件 dyn_b1_14_28（λ=1）
    A*        = 17 个干预条件中净事件最大者（无需选择器即可及的上界）
    U1        = 逐题选择 oracle（17 干预条件 + 不干预）
    U1|3      = 逐题选择 oracle（static_d12 与 dyn_b1_14_28，λ=1 + 不干预；用于隔离强度增量）
    U2        = 联合 oracle（3 条件 × 8 个 λ）

「不干预」条件在指标口径上取**基线自身的 MC 值**（不是 0）。

自检（内置，先于统计量）：
    S1 复现已发布 summary：baseline 与 dyn_b1_14_28 的 MC1／MC2／MC3 均值逐位一致（两口径）；
    S2 事件恒等式：net(c) == (mean MC1(c) − mean MC1(baseline)) × n；
    S3 单调性：net(U1) ≥ max_c net(c)，且 net(U2) ≥ net(U1|3)；
    S4 口径自检：baseline__ps1 与 baseline 的逐题 MC1 决策完全相等；
    S5 强度网格 λ=1 与主产物同条件一致（MC1／MC3 逐位；MC2 容差 1e-5，覆盖 float32 求和次序噪声）；
    S6 复算确定性：同 seed 的 bootstrap 两次调用结果完全相同；
    S7 两次前向的基线决策一致：网格产物 baseline MC1 与主产物 baseline MC1 逐题相同；
    S8 主分析源（第二次运行）与首次运行的**官方口径键**逐位相同（口径来源可追溯）。

用法
----
    python3 experiments/lin_theory/analyze_dola_c5_ceiling.py --selftest   # 合成数据自检（秒级）
    python3 experiments/lin_theory/analyze_dola_c5_ceiling.py             # 全量复算（零 GPU，秒级）
输出：experiments/outputs/dola_c5_ceiling_20260927/c5_ceiling.json + c5_ceiling_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "outputs" / "dola_c5_ceiling_20260927"
# 主分析源：第二次全量运行（官方口径键与首次运行**逐位相同**，见自检 S8；
# 归一化口径键取该次运行——首次运行的 `__ps1` 组受键名冲突缺陷影响，不具效力）。
MAIN_JSON = REPO / "experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817ps.json"
OFFICIAL_REF_JSON = REPO / "experiments/outputs/dola_mc_repro/dola_mc_Qwen3-1.7B_full817.json"
GRID_JSON = REPO / "experiments/outputs/dola_mc_betagrid/betagrid_Qwen3-1.7B_n817.json"
# 归一化口径的已发布参照（第二次运行的判读件，含 17 个 `__ps1` 条件的均值）
NORM_REF_JSON = REPO / "experiments/outputs/dola_mc_repro/judge_dola_mc_Qwen3-1.7B_full817ps.json"

PRIMARY = "dyn_b1_14_28"
SEED = 20260927
BOOT_N = 1000
RAND_DRAWS = 200
TARGET_NET = 41          # 41/817 = +5.00pp（与结题目标同刻度，见判据 §3.1）
RHO_TARGET = 0.60        # 精度阶梯的实现比例目标（判定备忘 §6① 的「底线＝实现比例 60%」）
METRICS = ("MC1", "MC2", "MC3")


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------
def events(base_mc1: np.ndarray, arm_mc1: np.ndarray) -> tuple[int, int]:
    """返回 (救回, 破坏) 事件数。"""
    b = base_mc1 > 0.5
    a = arm_mc1 > 0.5
    return int(np.sum((~b) & a)), int(np.sum(b & (~a)))


def net_curve(base_mc1: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每一列条件返回 (救回数组, 破坏数组)。"""
    b = base_mc1 > 0.5
    a = mat > 0.5
    return np.sum((~b)[:, None] & a, axis=0), np.sum(b[:, None] & (~a), axis=0)


def oracle_choice(mat: np.ndarray) -> np.ndarray:
    """逐行取最大值的列号；列序保证「不干预」在第 0 列、其余按字典序 ⇒ argmax 天然满足并列规则。"""
    return np.argmax(mat, axis=1)


def bootstrap_net(base_mc1: np.ndarray, arm_mc1: np.ndarray, rng: np.random.Generator,
                  n_boot: int = BOOT_N) -> dict:
    """题目级 bootstrap：净事件的 95% 百分位区间。"""
    n = len(base_mc1)
    idx = rng.integers(0, n, size=(n_boot, n))
    b = (base_mc1 > 0.5)[idx]
    a = (arm_mc1 > 0.5)[idx]
    net = np.sum((~b) & a, axis=1) - np.sum(b & (~a), axis=1)
    return {"mean": float(net.mean()),
            "ci95": [float(np.percentile(net, 2.5)), float(np.percentile(net, 97.5))]}


def build_columns(noop: np.ndarray, conds: list[str], mats: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """按「不干预优先、其余字典序」拼列，返回矩阵与条件名（第 0 列为不干预）。"""
    order = sorted(conds)
    return np.column_stack([noop] + [mats[c] for c in order]), ["baseline"] + order


# --------------------------------------------------------------------------------------
# 自检（合成数据；验证事件数学与 oracle 规则，不依赖真实产物）
# --------------------------------------------------------------------------------------
def selftest() -> int:
    n = 6
    base = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    arm_a = np.array([1, 1, 0, 1, 0, 0], dtype=float)   # 救 q0、破 q5
    arm_b = np.array([0, 1, 1, 1, 0, 1], dtype=float)   # 救 q2（不碰 q1/q3/q5）
    arm_c = base.copy()                                  # 恒等条件
    mats = {"aa": arm_a, "bb": arm_b, "cc": arm_c}
    mat, names = build_columns(base, list(mats), mats)
    assert names == ["baseline", "aa", "bb", "cc"], names

    assert events(base, arm_a) == (1, 1), events(base, arm_a)
    assert events(base, arm_b) == (1, 0), events(base, arm_b)
    rescue, broken = net_curve(base, mat)
    assert list(rescue) == [0, 1, 1, 0], (rescue, broken)
    assert list(broken) == [0, 1, 0, 0], (rescue, broken)

    # oracle：逐行最大 + 并列优先不干预（q1/q3 最大同为 1 且 baseline=1 ⇒ 选 baseline；
    # q0 由 aa 救回、q2 由 bb 救回、q4/q5 保持不干预以免被破坏）
    pick = oracle_choice(mat)
    assert list(pick) == [1, 0, 2, 0, 0, 0], list(pick)
    oracle_mc1 = mat[np.arange(n), pick]
    assert events(base, oracle_mc1) == (2, 0), events(base, oracle_mc1)
    for j in range(mat.shape[1]):
        r, b = events(base, mat[:, j])
        assert 2 >= r - b, (j, r, b)

    for j in range(mat.shape[1]):
        r, b = events(base, mat[:, j])
        assert abs((r - b) - (mat[:, j].mean() - base.mean()) * n) < 1e-9

    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)
    assert bootstrap_net(base, oracle_mc1, rng1, 50) == bootstrap_net(base, oracle_mc1, rng2, 50)

    print("[selftest] 合成数据 6/6 通过：事件数学 / 列序与并列规则 / oracle / 单调性 / 恒等式 / 确定性")
    return 0


# --------------------------------------------------------------------------------------
# 真实产物：装载
# --------------------------------------------------------------------------------------
def load_main(path: Path) -> dict:
    d = json.loads(path.read_text())
    pq = d["per_question"]
    conds_all = list(pq[0]["mc"].keys())
    conds = [c for c in conds_all if not c.endswith("__ps1") and c != "baseline"]
    out = {"summary": d["summary"], "n": len(pq), "pq": pq, "conds": conds,
           "folds": {k: np.asarray(v, dtype=int) for k, v in d["data"]["fold_ids"].items()},
           "base": {}, "base_metrics": {}, "mats": {}}
    for tag, suffix in (("official", ""), ("norm", "__ps1")):
        bkey = "baseline" + suffix
        out["base"][tag] = np.array([q["mc"][bkey]["MC1"] for q in pq], dtype=float)
        out["base_metrics"][tag] = {m: np.array([q["mc"][bkey][m] for q in pq], dtype=float)
                                    for m in METRICS}
        out["mats"][tag] = {m: {c: np.array([q["mc"][c + suffix][m] for q in pq], dtype=float)
                                for c in conds} for m in METRICS}
    return out


def _grid_cell(q: dict, cond: str, lam: float, side: str) -> tuple[float, float, float]:
    """取网格产物中某题某条件某 λ 的 (MC1, MC2, MC3)。λ 键按浮点匹配，缺失即报错。"""
    cell = q["mc"][cond]
    key = next((k for k in cell if _is_lam(k, lam)), None)
    if key is None:
        raise KeyError(f"网格产物缺少 λ={lam}（条件 {cond}）")
    v = cell[key][side]
    return float(v[0]), float(v[1]), float(v[2])


def _is_lam(key: str, lam: float) -> bool:
    try:
        return abs(float(key) - lam) < 1e-12
    except ValueError:
        return False


def load_grid(path: Path) -> dict:
    d = json.loads(path.read_text())
    pq = d["per_question"]
    lambdas = [float(x) for x in d["lambdas"]]
    conds = [c for c in d["conditions"] if c != "baseline"]
    out = {"lambdas": lambdas, "conds": conds, "n": len(pq), "base": {}, "arms": {},
           "base_mc2": {}, "base_mc3": {}, "mc2_arms": {}, "mc3_arms": {}}
    for tag, side in (("official", "off"), ("norm", "ps")):
        cells = {c: {lam: [_grid_cell(q, c, lam, side) for q in pq] for lam in lambdas} for c in conds}
        base_cells = [_grid_cell(q, "baseline", lambdas[0], side) for q in pq]
        out["base"][tag] = np.array([c[0] for c in base_cells], dtype=float)
        out["base_mc2"][tag] = np.array([c[1] for c in base_cells], dtype=float)
        out["base_mc3"][tag] = np.array([c[2] for c in base_cells], dtype=float)
        out["arms"][tag] = {(c, lam): np.array([v[0] for v in cells[c][lam]], dtype=float)
                            for c in conds for lam in lambdas}
        out["mc2_arms"][tag] = {(c, lam): np.array([v[1] for v in cells[c][lam]], dtype=float)
                                for c in conds for lam in lambdas}
        out["mc3_arms"][tag] = {(c, lam): np.array([v[2] for v in cells[c][lam]], dtype=float)
                                for c in conds for lam in lambdas}
    return out


# --------------------------------------------------------------------------------------
# 主分析
# --------------------------------------------------------------------------------------
def analyze(main: dict, grid: dict) -> dict:
    n = main["n"]
    res: dict = {"n": n, "primary": PRIMARY, "seed": SEED, "checks": [], "tags": {}}
    pidx = main["conds"].index(PRIMARY)

    for tag in ("official", "norm"):
        base = main["base"][tag]
        basem = main["base_metrics"][tag]
        mc1_all = np.column_stack([main["mats"][tag]["MC1"][c] for c in main["conds"]])
        rec: dict = {}

        # ---- U0 现状与基线 ----
        r0, b0 = events(base, mc1_all[:, pidx])
        rec["U0"] = {"arm": PRIMARY, "rescue": r0, "broken": b0, "net": r0 - b0,
                     "delta_pp": (r0 - b0) / n * 100,
                     "mc1": float(main["mats"][tag]["MC1"][PRIMARY].mean()),
                     "mc2": float(main["mats"][tag]["MC2"][PRIMARY].mean()),
                     "mc3": float(main["mats"][tag]["MC3"][PRIMARY].mean())}
        rec["baseline"] = {"mc1": float(basem["MC1"].mean()), "mc2": float(basem["MC2"].mean()),
                           "mc3": float(basem["MC3"].mean()),
                           "mc2_gt099_share": float(np.mean(basem["MC2"] > 0.99))}

        # ---- A* 与 17 个条件的单条件曲线 ----
        r_vec, b_vec = net_curve(base, mc1_all)
        net_vec = r_vec - b_vec
        best = int(np.argmax(net_vec))
        rec["A_star"] = {"arm": main["conds"][best], "rescue": int(r_vec[best]),
                         "broken": int(b_vec[best]), "net": int(net_vec[best]),
                         "delta_pp": float(net_vec[best]) / n * 100}
        rec["arms_lambda1"] = {c: {"rescue": int(r_vec[i]), "broken": int(b_vec[i]),
                                   "net": int(net_vec[i]), "delta_pp": float(net_vec[i]) / n * 100}
                               for i, c in enumerate(main["conds"])}

        # ---- U1：逐题选择 oracle（17 干预条件 + 不干预）----
        mat, names = build_columns(base, main["conds"],
                                   {c: main["mats"][tag]["MC1"][c] for c in main["conds"]})
        pick = oracle_choice(mat)
        chosen = mat[np.arange(n), pick]
        r1, b1 = events(base, chosen)
        # 指标口径 oracle：各指标逐题取条件内最大（不干预条件取基线自身的指标值）
        orc = {}
        for m in METRICS:
            cols = [basem[m]] + [main["mats"][tag][m][c] for c in sorted(main["conds"])]
            orc[m] = float(np.max(np.column_stack(cols), axis=1).mean())
        b_union = int(np.sum((base > 0.5) & np.any(mc1_all < 0.5, axis=1)))
        rec["U1"] = {"rescue": r1, "broken": b1, "net": r1 - b1, "delta_pp": (r1 - b1) / n * 100,
                     "R_max": r1, "B_max": b_union, "B_oracle_breaks": b1,
                     "oracle_means": orc,
                     "picked_conds": {names[j]: int(np.sum(pick == j)) for j in range(len(names))}}
        rec["U1"]["rho"] = rec["U0"]["net"] / rec["U1"]["net"] if rec["U1"]["net"] else None
        n_wrong = int(np.sum(base < 0.5))
        n_right = int(np.sum(base > 0.5))
        rec["U1"]["n_baseline_wrong"] = n_wrong
        rec["U1"]["n_baseline_right"] = n_right
        rec["U1"]["R_max_over_wrong"] = r1 / n_wrong if n_wrong else None
        rec["U1"]["B_max_over_right"] = b_union / n_right if n_right else None

        # ---- U1|3 与 U2（强度网格）----
        g_base = grid["base"][tag]
        arms3 = {"static_d12": grid["arms"][tag][("static_d12", 1.0)],
                 PRIMARY: grid["arms"][tag][(PRIMARY, 1.0)]}
        mat3, _ = build_columns(g_base, list(arms3), arms3)
        pick3 = oracle_choice(mat3)
        r3, b3 = events(g_base, mat3[np.arange(n), pick3])
        rec["U1_restricted3"] = {"rescue": r3, "broken": b3, "net": r3 - b3,
                                 "delta_pp": (r3 - b3) / n * 100}

        arm_keys = list(grid["arms"][tag].keys())
        cond_names = [f"{c}@{lam:g}" for (c, lam) in arm_keys]
        order = sorted(range(len(cond_names)), key=lambda j: cond_names[j])
        matU2 = np.column_stack([g_base] + [grid["arms"][tag][arm_keys[j]] for j in order])
        namesU2 = ["baseline"] + [cond_names[j] for j in order]
        pickU2 = oracle_choice(matU2)
        r2, b2 = events(g_base, matU2[np.arange(n), pickU2])
        lam_by_q = [namesU2[j].split("@")[1] if "@" in namesU2[j] else "baseline" for j in pickU2]
        orc2 = {}
        for m, src in (("MC1", None), ("MC2", "mc2_arms"), ("MC3", "mc3_arms")):
            if m == "MC1":
                cols = [g_base] + [grid["arms"][tag][arm_keys[j]] for j in order]
            else:
                noop = g_base if m == "MC1" else (grid["base_mc2"][tag] if m == "MC2" else grid["base_mc3"][tag])
                cols = [noop] + [grid[src][tag][arm_keys[j]] for j in order]
            orc2[m] = float(np.max(np.column_stack(cols), axis=1).mean())
        rec["U2"] = {"rescue": r2, "broken": b2, "net": r2 - b2, "delta_pp": (r2 - b2) / n * 100,
                     "n_intervention_conds": len(arm_keys), "oracle_means": orc2,
                     "picked_conds": {nm: int(np.sum(pickU2 == j)) for j, nm in enumerate(namesU2)},
                     "picked_lambda": {lam: int(sum(1 for x in lam_by_q if x == lam))
                                       for lam in sorted({x for x in lam_by_q if x != "baseline"}, key=float)},
                     "picked_no_intervention": int(sum(1 for x in lam_by_q if x == "baseline"))}
        rec["U2"]["rho"] = rec["U0"]["net"] / rec["U2"]["net"] if rec["U2"]["net"] else None
        rec["delta_selection"] = rec["U1"]["net"] - rec["A_star"]["net"]
        rec["delta_strength"] = rec["U2"]["net"] - rec["U1_restricted3"]["net"]

        # ---- 网格上的单一条件最优（无需选择器）----
        r_g, b_g = net_curve(g_base, matU2[:, 1:])
        net_g = r_g - b_g
        jb = int(np.argmax(net_g))
        rec["A_star_grid"] = {"arm": namesU2[1 + jb], "net": int(net_g[jb]),
                              "delta_pp": float(net_g[jb]) / n * 100}

        # ---- 重尾与饱和（C2 必报项）----
        d_pp = (mc1_all[:, pidx] - base) * 100.0
        rec["heavy_tail"] = {"delta_median_pp": float(np.median(d_pp)),
                             "delta_mean_pp": float(d_pp.mean()),
                             "abs_gt_50pp_share": float(np.mean(np.abs(d_pp) > 50)),
                             "u0_mc2_gt099_share": float(np.mean(
                                 main["mats"][tag]["MC2"][PRIMARY] > 0.99))}
        rec["saturation"] = {"baseline_mc2_gt099_share": rec["baseline"]["mc2_gt099_share"],
                             "u0_mc2_gt099_share": rec["heavy_tail"]["u0_mc2_gt099_share"]}

        # ---- 随机条件参考带（安慰剂）----
        rng = np.random.default_rng(SEED)
        nets = []
        for _ in range(RAND_DRAWS):
            j = rng.integers(0, mc1_all.shape[1], size=n)
            rr, bb = events(base, mc1_all[np.arange(n), j])
            nets.append(rr - bb)
        rec["random_cond"] = {"mean_net": float(np.mean(nets)),
                             "ci95": [float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5))],
                             "draws": RAND_DRAWS}

        # ---- 精度阶梯 ----
        R, B = rec["U1"]["R_max"], rec["U1"]["B_max"]
        B_u0 = rec["U0"]["broken"]
        a_req = (RHO_TARGET * R + B) / (R + B) if (R + B) else float("nan")
        a_req_arm = (RHO_TARGET * R + B_u0) / (R + B_u0) if (R + B_u0) else float("nan")
        rec["precision_ladder"] = {"R_max": R, "B_max": B, "B_u0": B_u0, "rho_target": RHO_TARGET,
                                   "a_required": float(a_req), "a_required_u0_break": float(a_req_arm),
                                   "net_at_a": {f"{a:.2f}": float(a * R - (1 - a) * B)
                                                for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)}}

        # ---- 追加描述性分析（非事前设定的判据）：条件数膨胀曲线 ----
        # 按单一条件净事件降序取前 k 个条件，报 R_max(k)（说明上界随候选条件数的膨胀幅度）
        arm_order = list(np.argsort(-net_vec))
        curve = []
        for k in range(1, mc1_all.shape[1] + 1):
            sub = mc1_all[:, arm_order[:k]]
            r_k = int(np.sum((base < 0.5) & np.any(sub > 0.5, axis=1)))
            curve.append({"k": k, "arm": main["conds"][arm_order[k - 1]], "R_max_k": r_k})
        rec["cond_count_curve"] = curve

        # ---- 追加描述性分析（非事前设定的判据）：按真选项数的可救比例 ----
        # 用于排除「MC1 允许多个真选项 ⇒ 上界被结构放大」这一疑虑
        n_true = np.array([q["n_true"] for q in main["pq"]], dtype=int)
        rec["rescue_by_n_true"] = {}
        for name, mask in (("all", np.ones(n, dtype=bool)),
                           ("n_true_eq_1", n_true == 1),
                           ("n_true_ge_5", n_true >= 5)):
            wrong = int(np.sum((base[mask] < 0.5)))
            r_sub = int(np.sum((base[mask] < 0.5) & np.any(mc1_all[mask] > 0.5, axis=1)))
            rec["rescue_by_n_true"][name] = {"n": int(mask.sum()), "n_wrong": wrong, "R_max": r_sub,
                                             "ratio": (r_sub / wrong) if wrong else None}

        # ---- 追加描述性分析（非事前设定的判据）：A* 的两折留出评估 ----
        # 在一折上按净事件条件选择、在另一折上报告（说明「最优固定条件」的留出表现）
        held = {}
        for fsel, frep in (("a", "b"), ("b", "a")):
            ids_s, ids_r = main["folds"][fsel], main["folds"][frep]
            rs, bs = net_curve(base[ids_s], mc1_all[ids_s])
            j = int(np.argmax(rs - bs))
            rr, bb = events(base[ids_r], mc1_all[ids_r, j])
            rr0, bb0 = events(base[ids_r], mc1_all[ids_r, pidx])
            held[f"{fsel}->{frep}"] = {"arm": main["conds"][j], "report_net": int(rr - bb),
                                       "report_delta_pp": float(rr - bb) / len(ids_r) * 100,
                                       "U0_report_net": int(rr0 - bb0),
                                       "U0_report_delta_pp": float(rr0 - bb0) / len(ids_r) * 100}
        rec["heldout_fixed_cond"] = held

        # ---- bootstrap ----
        rng2 = np.random.default_rng(SEED)
        rec["bootstrap"] = {"U0": bootstrap_net(base, mc1_all[:, pidx], rng2),
                            "U1": bootstrap_net(base, chosen, rng2)}

        # ---- 分折稳健性 ----
        rec["folds"] = {}
        for fname, ids in main["folds"].items():
            rf, bf = events(base[ids], mc1_all[ids, pidx])
            matf, _ = build_columns(base[ids], main["conds"],
                                    {c: main["mats"][tag]["MC1"][c][ids] for c in main["conds"]})
            pf = oracle_choice(matf)
            r_or, b_or = events(base[ids], matf[np.arange(len(ids)), pf])
            rec["folds"][fname] = {"n": len(ids), "U0_net": rf - bf, "U1_net": r_or - b_or}

        res["tags"][tag] = rec

    return res


def real_checks(main: dict, grid: dict, res: dict, ref: dict | None = None,
                norm_ref: dict | None = None) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    n = main["n"]
    # S1 复现已发布数字：官方口径对 summary；归一化口径对第二次运行的判读件
    # （判读件 `post_softmax_diagnostic.rows` 给出 17 个 `__ps1` 条件的 MC1/MC2/MC3 均值；
    #  baseline__ps1 ≡ baseline，故对 summary 的 baseline 核对）
    for tag in ("official", "norm"):
        for cond in ("baseline", PRIMARY):
            if cond == "baseline":
                want = {m: float(main["summary"]["baseline"][m]) for m in METRICS} if tag == "official" else \
                       {m: float(main["summary"]["baseline"][m]) for m in METRICS}
                got = {m: float(main["base_metrics"][tag][m].mean()) for m in METRICS}
                src = "summary.baseline" + ("（baseline__ps1 ≡ baseline）" if tag == "norm" else "")
            else:
                if tag == "official":
                    want = {m: float(main["summary"][cond][m]) for m in METRICS}
                    src = "summary." + cond
                else:
                    rows = (norm_ref or {}).get("post_softmax_diagnostic", {}).get("rows", {})
                    if cond + "__ps1" not in rows:
                        continue
                    want = {m: float(rows[cond + "__ps1"][m]) for m in METRICS}
                    src = "判读件 post_softmax_diagnostic.rows." + cond + "__ps1"
                got = {m: float(main["mats"][tag][m][cond].mean()) for m in METRICS}
            for m in METRICS:
                add(f"S1 {tag} {cond} {m}", abs(want[m] - got[m]) < 1e-9,
                    f"复算 {got[m]:.12f} vs 发布 {want[m]:.12f}（{src}）")
    # S2 事件恒等式
    for tag in ("official", "norm"):
        r = res["tags"][tag]["U0"]
        ident = (main["mats"][tag]["MC1"][PRIMARY].mean() - main["base"][tag].mean()) * n
        add(f"S2 {tag} 事件恒等式 net == Δmean×n", abs((r["rescue"] - r["broken"]) - ident) < 1e-6,
            f"net {r['rescue'] - r['broken']} vs {ident:.6f}")
    # S3 单调性
    for tag in ("official", "norm"):
        r = res["tags"][tag]
        ok = r["U1"]["net"] >= r["A_star"]["net"] and r["U2"]["net"] >= r["U1_restricted3"]["net"]
        add(f"S3 {tag} oracle 单调性", ok,
            f"U1 {r['U1']['net']} ≥ A* {r['A_star']['net']}；U2 {r['U2']['net']} ≥ U1|3 {r['U1_restricted3']['net']}")
    # S4 baseline__ps1 ≡ baseline
    same = np.array_equal(main["base"]["official"] > 0.5, main["base"]["norm"] > 0.5)
    add("S4 baseline__ps1 ≡ baseline（逐题 MC1 决策）", same, "逐题决策完全一致" if same else "存在不一致题目")
    # S5 网格 λ=1 与主产物一致（MC1/MC3 逐位；MC2 容差 1e-5，因 float32 求和次序在少数题上有 ~6e-6 量级差异）
    for tag in ("official", "norm"):
        for cond in ("static_d12", PRIMARY):
            for m, key in (("MC1", "arms"), ("MC2", "mc2_arms"), ("MC3", "mc3_arms")):
                a = main["mats"][tag][m][cond]
                b = grid[key][tag][(cond, 1.0)]
                d = float(np.max(np.abs(a - b)))
                tol = 1e-5 if m == "MC2" else 1e-9
                add(f"S5 {tag} {cond} {m}（网格 λ=1 vs 主产物）", d <= tol, f"max|Δ| = {d:.3e}（容差 {tol:g}）")
    # S6 bootstrap 确定性
    o1 = bootstrap_net(main["base"]["official"], main["mats"]["official"]["MC1"][PRIMARY],
                       np.random.default_rng(SEED), 100)
    o2 = bootstrap_net(main["base"]["official"], main["mats"]["official"]["MC1"][PRIMARY],
                       np.random.default_rng(SEED), 100)
    add("S6 bootstrap 复算确定性", o1 == o2, "同 seed 两次调用结果相同")
    # S7 两次前向的基线决策一致
    for tag in ("official", "norm"):
        same7 = np.array_equal(main["base"][tag] > 0.5, grid["base"][tag] > 0.5)
        add(f"S7 {tag} 网格基线与主产物基线决策一致", same7,
            "逐题决策完全一致" if same7 else "存在不一致题目（U2 事件以网格基线为准）")
    # S8 主分析源与首次运行的官方口径键逐位相同（口径来源可追溯）
    if ref is not None:
        for cond in ["baseline"] + main["conds"]:
            for m in METRICS:
                a = (main["base_metrics"]["official"][m] if cond == "baseline"
                     else main["mats"]["official"][m][cond])
                b = (ref["base_metrics"]["official"][m] if cond == "baseline"
                     else ref["mats"]["official"][m].get(cond))
                if b is None:
                    continue
                d = float(np.max(np.abs(a - b)))
                add(f"S8 official {cond} {m}（两次全量运行）", d == 0.0, f"max|Δ| = {d:.3e}")
    return checks


def verdict(res: dict) -> dict:
    """按判据 §3 的分支自动判定（人工复核后写入协议 §4）。"""
    off, nrm = res["tags"]["official"], res["tags"]["norm"]
    u1n, u1o = nrm["U1"]["net"], off["U1"]["net"]
    ratio = u1n / u1o if u1o else float("nan")
    if u1n >= TARGET_NET and ratio >= 0.5:
        branch = "保留为方法路线（进 L1 设计；前置＝选择器可观测性方案）"
    elif u1n >= TARGET_NET or ratio >= 0.5:
        branch = "不确定区间（并列披露；不立方法路线、不关闭）"
    else:
        branch = "关闭为审计素材（上限由口径支配，与 C1 同型）"
    a_req = nrm["precision_ladder"]["a_required"]
    return {"u1_net_norm": u1n, "u1_net_official": u1o, "ratio_norm_over_official": float(ratio),
            "target_net": TARGET_NET, "branch": branch,
            "delta_selection_official": off["delta_selection"], "delta_strength_official": off["delta_strength"],
            "delta_selection_norm": nrm["delta_selection"], "delta_strength_norm": nrm["delta_strength"],
            "a_required_norm": a_req, "ladder_unreachable": bool(a_req >= 0.90),
            "rho_U1_official": off["U1"]["rho"], "rho_U1_norm": nrm["U1"]["rho"],
            "rho_U2_official": off["U2"]["rho"], "rho_U2_norm": nrm["U2"]["rho"]}


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------
def write_report(res: dict, path: Path) -> None:
    n = res["n"]
    L: list[str] = []
    A = L.append
    A("# C5 完美选择上界复算报告（DoLa 原生域，U0／A*／U1／U2）")
    A("")
    A(f"判据（事前设定）：`docs/protocol/dola-c5-ceiling-20260927.md` §1–§3；数据：TruthfulQA-MC n={n}，"
      "Qwen3-1.7B，官方口径与归一化口径并列；零 GPU。")
    A("")
    A("> 全部 oracle 数字为**样本内完美选择**的上界，不可实现（判据 §3.5）。")
    A("")
    A("## 0. 自检")
    A("")
    bad = [c for c in res["checks"] if not c["ok"]]
    A(f"- 共 {len(res['checks'])} 项，通过 {len(res['checks']) - len(bad)} 项，失败 {len(bad)} 项。")
    for c in bad:
        A(f"  - ❌ {c['name']}：{c['detail']}")
    A("")

    A("## 1. 事件口径：现状与三层上界")
    A("")
    A("| 层 | 定义 | 官方口径 救回/破坏/净事件（Δpp） | 归一化口径 救回/破坏/净事件（Δpp） |")
    A("|---|---|---|---|")
    for label, desc, key in (("U0 现状", f"固定条件 `{PRIMARY}`（λ=1）", "U0"),
                             ("A* 单条件最优", "17 个干预条件中净事件最大者（无需选择器）", "A_star"),
                             ("U1 选择 oracle", "逐题选择（17 干预条件 + 不干预）", "U1"),
                             ("U2 联合 oracle", "逐题选择（3 条件 × 8 个 λ）", "U2")):
        cells = []
        for tag in ("official", "norm"):
            r = res["tags"][tag][key]
            arm = f" `{r['arm']}`" if "arm" in r else ""
            cells.append(f"{r['rescue']}/{r['broken']}/{r['net']}（{r['delta_pp']:+.2f}pp）{arm}")
        A(f"| {label} | {desc} | {cells[0]} | {cells[1]} |")
    A("")
    A(f"实现比例 ρ（U0 / U1）：官方 **{res['tags']['official']['U1']['rho']:.3f}**、"
      f"归一化 **{res['tags']['norm']['U1']['rho']:.3f}**；"
      f"实现比例（U0 / U2）：官方 {res['tags']['official']['U2']['rho']:.3f}、"
      f"归一化 {res['tags']['norm']['U2']['rho']:.3f}。")
    A("")
    A("指标口径 oracle 均值（MC1／MC2／MC3）：")
    A("")
    A("| 层 | 官方口径 | 归一化口径 |")
    A("|---|---|---|")
    for label, key in (("baseline（现成）", "baseline"), ("U0 现状", "U0"), ("U1 选择 oracle", "U1"), ("U2 联合 oracle", "U2")):
        cells = []
        for tag in ("official", "norm"):
            r = res["tags"][tag][key]
            vals = [r["mc1"], r["mc2"], r["mc3"]] if key in ("baseline", "U0") else \
                   [r["oracle_means"]["MC1"], r["oracle_means"]["MC2"], r["oracle_means"]["MC3"]]
            cells.append(f"{vals[0]:.4f} / {vals[1]:.4f} / {vals[2]:.4f}")
        A(f"| {label} | {cells[0]} | {cells[1]} |")
    A("")

    A("## 1b. 派生比率（可救／可破坏相对基数）")
    A("")
    A("| 口径 | baseline 答错题数 | R_max | R_max／答错 | baseline 答对题数 | B_max | B_max／答对 |")
    A("|---|---|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        u = res["tags"][tag]["U1"]
        A(f"| {name} | {u['n_baseline_wrong']} | {u['R_max']} | {u['R_max_over_wrong']:.3f} | "
          f"{u['n_baseline_right']} | {u['B_max']} | {u['B_max_over_right']:.3f} |")
    A("")

    A("## 2. 增量归因（选择红利 vs 强度红利）")
    A("")
    A("| 量 | 官方口径 | 归一化口径 |")
    A("|---|---|---|")
    A(f"| Δ_selection = U1 − A* | {res['tags']['official']['delta_selection']:+d} 事件 | "
      f"{res['tags']['norm']['delta_selection']:+d} 事件 |")
    A(f"| Δ_strength = U2 − U1|3 | {res['tags']['official']['delta_strength']:+d} 事件 | "
      f"{res['tags']['norm']['delta_strength']:+d} 事件 |")
    A(f"| A*（网格 16 条件） | {res['tags']['official']['A_star_grid']['net']:+d} 事件"
      f"（{res['tags']['official']['A_star_grid']['arm']}） | {res['tags']['norm']['A_star_grid']['net']:+d} 事件"
      f"（{res['tags']['norm']['A_star_grid']['arm']}） |")
    A("")

    A("## 3. 精度阶梯（选择器需达到的平衡准确率）")
    A("")
    A(f"模型：`net(a) = a·R_max − (1−a)·B_max`（误差与可救／可破坏独立，属乐观假说）。"
      f"达到实现比例 {RHO_TARGET:.0%} 所需 `a`：")
    A("")
    A("| 口径 | R_max | B_max（事前设定＝可破坏并集） | 所需 a（事前设定） | 所需 a（次级＝U0 条件破坏数） |")
    A("|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        pl = res["tags"][tag]["precision_ladder"]
        A(f"| {name} | {pl['R_max']} | {pl['B_max']} | **{pl['a_required']:.3f}** | {pl['a_required_u0_break']:.3f} |")
    A("")
    A("期望净事件随 a（官方口径）："
      + "；".join(f"a={k} → {v:+.1f}" for k, v in res["tags"]["official"]["precision_ladder"]["net_at_a"].items()) + "。")
    A("")

    A("## 4. 安慰剂与不确定度")
    A("")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        r = res["tags"][tag]
        A(f"- **{name}口径**：随机条件参考带（{r['random_cond']['draws']} 次逐题随机取条件）均值 "
          f"{r['random_cond']['mean_net']:+.2f} 事件、95% 区间 "
          f"[{r['random_cond']['ci95'][0]:+.1f}, {r['random_cond']['ci95'][1]:+.1f}]；"
          f"U0 净事件 bootstrap 95% 区间 [{r['bootstrap']['U0']['ci95'][0]:+.1f}, {r['bootstrap']['U0']['ci95'][1]:+.1f}]；"
          f"U1 上界 bootstrap 95% 区间 [{r['bootstrap']['U1']['ci95'][0]:+.1f}, {r['bootstrap']['U1']['ci95'][1]:+.1f}]。")
    A("")
    A("分折稳健性（净事件）：")
    A("")
    A("| 口径 | 折 | n | U0 | U1 |")
    A("|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        for f, v in sorted(res["tags"][tag]["folds"].items()):
            A(f"| {name} | {f} | {v['n']} | {v['U0_net']:+d} | {v['U1_net']:+d} |")
    A("")

    A("## 5. 重尾与饱和披露（C2 必报项）")
    A("")
    A("| 口径 | U0 逐题 Δ 中位数 | U0 逐题 Δ 均值 | \\|Δ\\|>50pp 份额 | baseline MC2>0.99 | U0 MC2>0.99 |")
    A("|---|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        h, s = res["tags"][tag]["heavy_tail"], res["tags"][tag]["saturation"]
        A(f"| {name} | {h['delta_median_pp']:+.2f}pp | {h['delta_mean_pp']:+.2f}pp | "
          f"{h['abs_gt_50pp_share']:.1%} | {s['baseline_mc2_gt099_share']:.1%} | {s['u0_mc2_gt099_share']:.1%} |")
    A("")

    A("## 6. oracle 条件选择分布")
    A("")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        dist = res["tags"][tag]["U2"]["picked_lambda"]
        A(f"- **{name}口径 U2 的 λ 选择**：" + "；".join(f"λ={k} → {v} 题" for k, v in dist.items()))
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        picks = sorted(res["tags"][tag]["U1"]["picked_conds"].items(), key=lambda kv: -kv[1])[:8]
        A(f"- **{name}口径 U1 的条件选择（前 8）**：" + "；".join(f"{k} → {v}" for k, v in picks))
    A("")

    A("## 6b. 追加描述性分析（非事前设定的判据，不动判定分支）")
    A("")
    A("条件数膨胀曲线（按单一条件净事件降序取前 k 个条件的逐题 oracle 可救集合）：")
    A("")
    A("| 口径 | k=1 | k=2 | k=3 | k=5 | k=8 | k=12 | k=17 |")
    A("|---|---|---|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        c = {row["k"]: row["R_max_k"] for row in res["tags"][tag]["cond_count_curve"]}
        A(f"| {name} | {c[1]} | {c[2]} | {c[3]} | {c[5]} | {c[8]} | {c[12]} | {c[17]} |")
    A("")
    A("按真选项数的可救比例（排除「多真选项放大上界」的疑虑）：")
    A("")
    A("| 口径 | 子集 | n | 答错 | R_max | 可救／答错 |")
    A("|---|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        for key, label in (("all", "全部"), ("n_true_eq_1", "n_true = 1"), ("n_true_ge_5", "n_true ≥ 5")):
            v = res["tags"][tag]["rescue_by_n_true"][key]
            ratio = f"{v['ratio']:.3f}" if v["ratio"] is not None else "—"
            A(f"| {name} | {label} | {v['n']} | {v['n_wrong']} | {v['R_max']} | {ratio} |")
    A("")
    A("A* 的两折留出评估（在一折上条件选择、在另一折报告）：")
    A("")
    A("| 口径 | 方向 | 选中条件 | 报告折净事件 | 报告折 U0 净事件 |")
    A("|---|---|---|---|---|")
    for tag, name in (("official", "官方"), ("norm", "归一化")):
        for k, v in res["tags"][tag]["heldout_fixed_cond"].items():
            A(f"| {name} | {k} | {v['arm']} | {v['report_net']:+d}（{v['report_delta_pp']:+.2f}pp） | "
              f"{v['U0_report_net']:+d}（{v['U0_report_delta_pp']:+.2f}pp） |")
    A("")

    v = res["verdict"]
    A("## 7. 判定（按判据 §3 自动生成，人工复核后写入协议 §4）")
    A("")
    A(f"- 主判据：归一化口径 U1 净事件 **{v['u1_net_norm']}**（目标线 {v['target_net']}）；"
      f"归一化／官方比 **{v['ratio_norm_over_official']:.3f}** ⇒ **{v['branch']}**。")
    A(f"- 增量归因：Δ_selection 官方 {v['delta_selection_official']:+d}／归一化 {v['delta_selection_norm']:+d}；"
      f"Δ_strength 官方 {v['delta_strength_official']:+d}／归一化 {v['delta_strength_norm']:+d}。")
    A(f"- 精度阶梯：归一化口径达到实现比例 {RHO_TARGET:.0%} 所需平衡准确率 "
      f"**{v['a_required_norm']:.3f}**（≥0.90 判为不可达：{v['ladder_unreachable']}）。")
    A("")
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--main_json", default=str(MAIN_JSON))
    ap.add_argument("--grid_json", default=str(GRID_JSON))
    ap.add_argument("--official_ref_json", default=str(OFFICIAL_REF_JSON))
    ap.add_argument("--norm_ref_json", default=str(NORM_REF_JSON))
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_p, grid_p = Path(args.main_json), Path(args.grid_json)
    main_d = load_main(main_p)
    grid_d = load_grid(grid_p)
    ref_d = load_main(Path(args.official_ref_json)) if Path(args.official_ref_json).exists() else None
    norm_ref = json.loads(Path(args.norm_ref_json).read_text()) if Path(args.norm_ref_json).exists() else None
    res = analyze(main_d, grid_d)
    res["checks"] = real_checks(main_d, grid_d, res, ref_d, norm_ref)
    res["verdict"] = verdict(res)
    res["inputs"] = {"main_json": str(main_p), "grid_json": str(grid_p),
                     "official_ref_json": str(args.official_ref_json) if ref_d is not None else None}

    bad = [c for c in res["checks"] if not c["ok"]]
    (out_dir / "c5_ceiling.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    write_report(res, out_dir / "c5_ceiling_report.md")

    print(f"自检：{len(res['checks']) - len(bad)}/{len(res['checks'])} 通过"
          + (f"，失败项：{[c['name'] for c in bad]}" if bad else ""))
    v = res["verdict"]
    print(f"主判据：归一化 U1 净事件 {v['u1_net_norm']}（目标 {v['target_net']}），"
          f"归一化/官方比 {v['ratio_norm_over_official']:.3f} ⇒ {v['branch']}")
    print(f"产物：{out_dir}/c5_ceiling.json 与 c5_ceiling_report.md")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
