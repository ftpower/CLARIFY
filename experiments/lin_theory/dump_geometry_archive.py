"""Geometry archive dumper for FAD / S5 zero-GPU diagnostics (P5.1, P-S5.1–P-S5.3).

理论锚点（规则 1：动手前引用理论章节）：
  - FAD 引理 1（翻转集刻画 m/Δ₀、阈值 m/(m+Δ₀)）、命题 5（救回必要性分解 g/h）、
    P5.1 判据：docs/paper/paper-route-method-schemes.md §2.2、§2.5、§5
  - S5 抬压分解（δ 的 ReLU 符号分解、单侧算子 l₊/l₋）、判停诊断 P-S5.1–P-S5.3：
    docs/paper/paper-route-tldc-improvements.md §S5
  - 档案 schema 与判读协议：docs/protocol/geometry-archive-spec.md

为什么需要本脚本（2026-09-21 档案侦察结论）：
  - s14_tldc_samples.json 只存 subset/rank/correct_beta{...}，无 logits；
  - probe_scores_*.json 只存 h（4096 维 @L28），无 logits；
  - s15_2b_tldc_per_token.json（仅 1.7B/seed123）每步只存 top-10×3 + yt_logits，
    无法离线重算单侧算子 argmax（min(l₁, βl₀+(1−β)l₁) 的 argmax 不在 l₁/l' 的 top-10 内）。
  ⇒ P5.1 / P-S5.1 的精确判定必须一次轻前向 dump 全词表几何量 + 预注册 β 网格 argmax。

红线（继承 validate_s14_tldc.py / analyze_tldc_per_token.py）：
  1. l_final 必须是**模型真实 final logits**（禁 logit-lens 代验，T04；lens 重算有
     cublas 舍入伪影，13.5% 步级 argmax 不一致，2026-08-25 实测）。
  2. l_early 按 TLDC 算子定义 = W_U · ln_final(h_ℓ*)（lens 只用于 ℓ* 读出的定义本身）。
  3. exact 标签（check_correct_exact）、rank 1-indexed、截断保尾（>1024 取尾）。
  4. β 网格预注册（BETAS 常量）：多重比较纪律——事后换网格重跑 = 挑结果，禁止。

输出（每样本 × 每步）：
  - geo：全词表几何标量（a/l₀(a)/l₁(a)/|R|/β*_min/c_min/…/y_true 的 m、Δ₀、β*）
  - grid：预注册 β 网格上 lift/damp/sym 三种算子的 argmax token id + 11×11 (β₊,β₋) 前沿
  - early_top10 / final_top10：与 per-token 档案同格式（诊断可视化用）
离线判读（见 spec §5）：
  - P5.1 精确（P(g=1)=∃t: c*∈R_t、KW/KC 可分离性、β* 分布）——阶段 1 基线轨迹即够
  - P-S5.1 精确（首分叉步的分支归因：lift/damp/sym 网格 argmax 对照 y_true）
  - P-S5.2/P-S5.3 轨迹级 KW/KC 需要阶段 2 算子轨迹（--operator lift/damp/sym × β 网格）

用法：
  阶段 1（基线轨迹几何档案，1.7B × 双 seed）：
    python experiments/lin_theory/dump_geometry_archive.py \
      --model Qwen/Qwen3-1.7B --layer_early 20 --n_test 300 --seed_test 123 \
      --operator baseline --output_dir experiments/outputs/geometry_archive
  阶段 2（单侧/对称算子轨迹，P-S5.2/5.3 精确量）：
    同命令加 --operator lift|damp|sym --beta 0.05（β 须 ∈ BETAS，见下方常量）
  自检（纯函数，无模型）：
    python experiments/lin_theory/dump_geometry_archive.py --selftest
  离线校验（读档案，无模型）：
    python experiments/lin_theory/dump_geometry_archive.py --verify <archive.json>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(Path(__file__).parent),
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tldc_per_token import (  # noqa: E402
    classify_samples,
    compute_early_exit_logits,
    get_topk_info,
)
from common import load_model_and_unembed  # noqa: E402
from src.data_loader import check_correct_exact, load_triviaqa  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════════
# 预注册常量（多重比较纪律：禁止事后修改后重跑挑结果）
# ═════════════════════════════════════════════════════════════════════════════

# β 网格：含 s14 TLDC 主扫 {0.01..0.20} + 扩展档；0.0 为 baseline 自洽点。
BETAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
TOP_K = 10  # 每步存档的 top-k（可视化/描点用；判定靠 grid，不靠 top-k）
N_FRONT = len(BETAS)  # (β₊,β₋) 前沿网格尺寸 = 11×11

THEORY_REFS = {
    "fad_lemma1": "docs/paper/paper-route-method-schemes.md §2.2 引理 1（m=margin、Δ₀=参考层偏好差、β*>m/(m+Δ₀) 可翻转）",
    "fad_prop5": "docs/paper/paper-route-method-schemes.md §2.5 命题 5（P(r)≤P(g=1)·P(h=1|g=1)，g=几何可用性）",
    "fad_p51": "docs/paper/paper-route-method-schemes.md §5 P5.1（P(g=1) 非平凡且 KW/KC 可分离；近零⇒结构性封顶判停）",
    "s5": "docs/paper/paper-route-tldc-improvements.md §S5（抬支 l₊=l₁+β·ReLU(δ)、压支 l₋=l₁−β·ReLU(−δ)、P-S5.1–5.3）",
    "spec": "docs/protocol/geometry-archive-spec.md（schema 与判读协议）",
}


# ═════════════════════════════════════════════════════════════════════════════
# 纯函数（可离线单测，无模型依赖）
# ═════════════════════════════════════════════════════════════════════════════


def apply_branches(l0, l1, beta_plus, beta_minus):
    """S5 双支算子：l' = l₁ + β₊·ReLU(δ) − β₋·ReLU(−δ)，δ = l₀ − l₁。

    lift=只有 β₊；damp=只有 β₋；sym=β₊=β₋（恒等 TLDC）；baseline=β₊=β₋=0。
    输入 [..., V]，输出 [..., V]。
    """
    delta = l0.float() - l1.float()
    return l1.float() + beta_plus * torch.relu(delta) - beta_minus * torch.relu(-delta)


def _empty_geo(a_id, a_text, l0_a, l1_a):
    return {
        "a_id": a_id,
        "a_text": a_text,
        "l0_a": float(l0_a),
        "l1_a": float(l1_a),
        "R_size": 0,
        "beta_star_min": None,
        "c_min_id": None,
        "c_min_text": None,
        "m_min": None,
        "d0_min": None,
    }


def compute_geo(l0, l1, y_true_id, tokenizer):
    """全词表几何标量（引理 1）。

    a = argmax(l₁)；翻转集 R = {c : l₀(c) > l₀(a)}（l₁(c)<l₁(a) 因 a 为 argmax 自动成立，
    且 l₀(a)>l₀(a) 为假 ⇒ a ∉ R）；m(a,c)=l₁(a)−l₁(c)，Δ₀(a,c)=l₀(c)−l₀(a)；
    β*_min = min_{c∈R} m/(m+Δ₀)；y_true 的 m/Δ₀/β* 与 in_R 标志。
    输入 [1,V] 或 [V]，全部按 fp32 计算。返回可 JSON 序列化的 dict。
    """
    l0f = l0.float().squeeze()
    l1f = l1.float().squeeze()
    a_id = int(l1f.argmax().item())
    l0_a = float(l0f[a_id].item())
    l1_a = float(l1f[a_id].item())

    mask_R = l0f > l0_a
    out = _empty_geo(a_id, tokenizer.decode([a_id]), l0_a, l1_a)

    R_size = int(mask_R.sum().item())
    out["R_size"] = R_size
    if R_size > 0:
        m = l1_a - l1f  # >0 on R（argmax 唯一时严格；平局时 =0 ⇒ β*=0，语义=零成本翻转）
        d0 = l0f - l0_a  # >0 on R（严格，由 mask 保证）
        ratios = torch.where(mask_R, m / (m + d0), torch.full_like(m, float("inf")))
        beta_min = ratios.min()
        c_min = int(ratios.argmin().item())
        out["beta_star_min"] = float(beta_min.item())
        out["c_min_id"] = c_min
        out["c_min_text"] = tokenizer.decode([c_min])
        out["m_min"] = float((l1_a - l1f[c_min]).item())
        out["d0_min"] = float((l0f[c_min] - l0_a).item())

    yt_in_R = bool(mask_R[y_true_id].item()) and y_true_id != a_id
    out["yt"] = {
        "l0": float(l0f[y_true_id].item()),
        "l1": float(l1f[y_true_id].item()),
        "delta": float((l0f[y_true_id] - l1f[y_true_id]).item()),
        "in_R": yt_in_R,
        "m": float((l1_a - l1f[y_true_id]).item()) if yt_in_R else None,
        "d0": float((l0f[y_true_id] - l0_a).item()) if yt_in_R else None,
        "beta_star": (
            float(((l1_a - l1f[y_true_id]) / ((l1_a - l1f[y_true_id]) + (l0f[y_true_id] - l0_a))).item())
            if yt_in_R
            else None
        ),
    }
    return out


def compute_grid(l0, l1, betas):
    """预注册网格上的三种算子 argmax + (β₊,β₋) 前沿。

    返回 {"lift": {"0.01": id, ...}, "damp": {...}, "sym": {...},
          "front2d": [[id × 11] × 11]}（front2d[i][j] = (β₊=betas[i], β₋=betas[j])）。
    """
    l0f = l0.float().squeeze()
    l1f = l1.float().squeeze()
    delta = l0f - l1f
    dp = torch.relu(delta)
    dm = torch.relu(-delta)

    grid = {"lift": {}, "damp": {}, "sym": {}}
    for b in betas:
        key = f"{b:.2f}"
        grid["lift"][key] = int((l1f + b * dp).argmax().item())
        grid["damp"][key] = int((l1f - b * dm).argmax().item())
        grid["sym"][key] = int((l1f + b * delta).argmax().item())

    b_t = torch.tensor(betas, dtype=torch.float32, device=l1f.device)
    logs = (
        l1f[None, None, :]
        + b_t[:, None, None] * dp[None, None, :]
        - b_t[None, :, None] * dm[None, None, :]
    )  # [nB, nB, V]
    grid["front2d"] = logs.argmax(dim=-1).tolist()
    return grid


def selftest():
    """纯函数单测（无模型）。手算期望值见各注释。"""
    ok = 0

    # ── 案例 1：翻转集/β*/网格 argmax（手算值）──
    l1 = torch.tensor([[3.0, 2.0, 1.0]])
    l0 = torch.tensor([[1.0, 3.5, 0.5]])
    y_true_id = 1

    class _Tok:
        @staticmethod
        def decode(ids):
            return "|".join(str(i) for i in ids)

    geo = compute_geo(l0, l1, y_true_id, _Tok)
    assert geo["a_id"] == 0 and geo["l0_a"] == 1.0 and geo["l1_a"] == 3.0
    assert geo["R_size"] == 1, geo["R_size"]  # 仅 token1: l0=3.5 > l0(a)=1.0
    assert abs(geo["beta_star_min"] - (1.0 / 3.5)) < 1e-6  # m=1, Δ₀=2.5 → 1/3.5≈0.2857
    assert geo["c_min_id"] == 1 and abs(geo["m_min"] - 1.0) < 1e-9 and abs(geo["d0_min"] - 2.5) < 1e-9
    assert geo["yt"]["in_R"] and abs(geo["yt"]["beta_star"] - (1.0 / 3.5)) < 1e-6
    ok += 1

    betas = [0.0, 0.2, 0.5, 0.8]
    grid = compute_grid(l0, l1, betas)
    # lift(0.5)=[3, 2.75, 1]→0；lift(0.8)=[3, 3.2, 1]→1（翻转）
    # damp(0.5)=[2, 2, 0.75]→0（平局取首）；damp(0.8)=[1.4, 2, 0.6]→1
    # sym(0.2)=[2.6, 2.3, 0.9]→0；sym(0.5)=[2, 2.75, 0.75]→1；sym(0.8)=[1.4, 3.2, 0.6]→1
    assert grid["lift"]["0.00"] == 0 and grid["lift"]["0.20"] == 0
    assert grid["lift"]["0.50"] == 0 and grid["lift"]["0.80"] == 1
    assert grid["damp"]["0.00"] == 0 and grid["damp"]["0.50"] == 0 and grid["damp"]["0.80"] == 1
    assert grid["sym"]["0.00"] == 0 and grid["sym"]["0.20"] == 0
    assert grid["sym"]["0.50"] == 1 and grid["sym"]["0.80"] == 1
    # front2d[i][j]: (β₊=betas[i], β₋=betas[j])——[3]=0.8, [2]=0.5
    assert grid["front2d"][3][0] == 1  # lift-only(0.8)
    assert grid["front2d"][0][3] == 1  # damp-only(0.8)
    assert grid["front2d"][3][3] == 1  # sym(0.8)
    assert grid["front2d"][0][0] == 0  # β=0 baseline
    ok += 1

    # ── 案例 2：R 为空（参考层不偏好任何 token 胜过末层 argmax）──
    l1b = torch.tensor([[1.0, 5.0]])
    l0b = torch.tensor([[3.0, 3.0]])
    geo2 = compute_geo(l0b, l1b, 0, _Tok)
    assert geo2["a_id"] == 1 and geo2["R_size"] == 0
    assert geo2["beta_star_min"] is None and geo2["c_min_id"] is None
    assert geo2["yt"]["in_R"] is False and geo2["yt"]["beta_star"] is None
    ok += 1

    # ── 案例 3：apply_branches 与算子恒等 l' = l₊ + l₋ − l₁ ──
    lift = apply_branches(l0, l1, 0.5, 0.0)
    damp = apply_branches(l0, l1, 0.0, 0.5)
    sym = apply_branches(l0, l1, 0.5, 0.5)
    base = apply_branches(l0, l1, 0.0, 0.0)
    assert torch.allclose(base, l1)
    assert torch.allclose(lift + damp - l1, sym, atol=1e-6)  # 恒等式（S5 §S5）
    assert int(lift.argmax()) == 0 and int(sym.argmax()) == 1  # β=0.5：lift 未翻、sym 翻
    ok += 1

    print(f"SELFTEST PASS: {ok}/4 checks (flip-set / grid / empty-R / branch identity)")


# ═════════════════════════════════════════════════════════════════════════════
# 轨迹 dump（需模型 + GPU）
# ═════════════════════════════════════════════════════════════════════════════

_OP_BRANCHES = {
    "baseline": (0.0, 0.0),
    "sym": ("b", "b"),
    "lift": ("b", 0.0),
    "damp": (0.0, "b"),
}


def dump_sample(
    model,
    tokenizer,
    entry,
    device,
    layer_early,
    W_U,
    b_U,
    ln_final,
    operator,
    beta,
    max_new,
):
    """在指定算子的 greedy 轨迹上逐步 dump 几何量。

    l_early = W_U·ln_final(h_ℓ*)（TLDC 定义本身）；l_final = 模型真实 logits（T04 红线）。
    每步同时存档：geo（全词表几何标量）、grid（预注册 β 网格 argmax）、top-k 描点。
    """
    y_true_id = entry["y_true_id"]
    tokens = model.to_tokens(entry["prompt"], prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]  # 截断保尾（code review Critical 1）

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    captured = {}

    def _hook_early(act, hook=None):
        captured["h_early"] = act[:, -1:, :].detach()
        return act

    beta_plus, beta_minus = _OP_BRANCHES[operator]
    bp = beta if beta_plus == "b" else float(beta_plus)
    bm = beta if beta_minus == "b" else float(beta_minus)

    steps_log = []
    gids = []
    nid = None
    for step in range(max_new):
        with torch.no_grad():
            logits_final = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_early, _hook_early)]
            )
        h_early = captured["h_early"]
        l_early = compute_early_exit_logits(h_early, ln_final, W_U, b_U)
        l_final = logits_final[0, -1:, :].float()  # TRUE final logits（禁 lens 代验）

        l_combined = apply_branches(l_early, l_final, bp, bm)
        nid = int(l_combined.argmax(dim=-1).item())
        gids.append(nid)

        e_f = l_early.float().squeeze()
        f_f = l_final.float().squeeze()
        steps_log.append(
            {
                "step": step,
                "chosen_id": nid,
                "geo": compute_geo(l_early, l_final, y_true_id, tokenizer),
                "grid": compute_grid(l_early, l_final, BETAS),
                "early_top10": get_topk_info(e_f, tokenizer, k=TOP_K),
                "final_top10": get_topk_info(f_f, tokenizer, k=TOP_K),
            }
        )

        # 自洽校验：若 β 在预注册网格内，算子 argmax 必须与网格一致
        if operator in ("sym", "lift", "damp") and beta in BETAS:
            grid_argmax = steps_log[-1]["grid"][operator][f"{beta:.2f}"]
            if grid_argmax != nid:
                print(
                    f"  [WARN] step {step}: operator argmax {nid} != grid {operator}"
                    f"@{beta} argmax {grid_argmax}"
                )

        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    gen_text = tokenizer.decode(gids).strip()
    return {"gids": gids, "gen_text": gen_text, "steps": steps_log}


# ═════════════════════════════════════════════════════════════════════════════
# 离线校验（无模型）
# ═════════════════════════════════════════════════════════════════════════════


def verify_archive(path):
    """读档案做 schema/一致性校验（无模型）。返回 (ok, n_err, n_step)。"""
    path = Path(path)
    data = json.loads(path.read_text())
    n_err = 0
    n_step = 0

    def _err(msg):
        nonlocal n_err
        n_err += 1
        print(f"  [ERR] {msg}")

    meta = data.get("meta")
    if not isinstance(meta, dict) or "operator" not in meta:
        _err("meta 缺失或不是 dict"); return 0, n_err, 0
    n_betas = len(meta.get("betas", []))
    if n_betas != N_FRONT:
        _err(f"meta.betas 长度 {n_betas} != 预注册 {N_FRONT}")

    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        _err("samples 缺失/为空"); return 0, n_err, 0

    for s in samples:
        sid = s.get("sample_id")
        for st in s.get("steps", []):
            n_step += 1
            geo = st.get("geo", {})
            grid = st.get("grid", {})
            if geo.get("a_id") is None:
                _err(f"sample {sid} step {st.get('step')}: geo.a_id 缺失")
                continue
            # β=0 自洽：三算子 argmax = l₁ argmax = geo.a_id
            for op in ("lift", "damp", "sym"):
                v = grid.get(op, {}).get("0.00")
                if v is None:
                    _err(f"sample {sid} step {st.get('step')}: grid.{op}['0.00'] 缺失")
                elif int(v) != int(geo["a_id"]):
                    _err(f"sample {sid} step {st.get('step')}: grid.{op}[0.00]={v} != a_id={geo['a_id']}")
            # baseline 轨迹：chosen == a_id
            if meta.get("operator") == "baseline" and st.get("chosen_id") != geo["a_id"]:
                _err(f"sample {sid} step {st.get('step')}: chosen {st.get('chosen_id')} != a_id {geo['a_id']}")
            # front2d 形状
            fd = grid.get("front2d")
            if not (isinstance(fd, list) and len(fd) == n_betas and all(len(r) == n_betas for r in fd)):
                _err(f"sample {sid} step {st.get('step')}: front2d 形状 != {n_betas}×{n_betas}")
            # yt.in_R ↔ yt.beta_star 一致性
            yt = geo.get("yt", {})
            if bool(yt.get("in_R")) != (yt.get("beta_star") is not None):
                _err(f"sample {sid} step {st.get('step')}: yt.in_R 与 yt.beta_star 不一致")
            # R_size 与 beta_star_min 一致性
            if geo.get("R_size", 0) > 0 and geo.get("beta_star_min") is None:
                _err(f"sample {sid} step {st.get('step')}: R_size>0 但 beta_star_min=None")
            if geo.get("R_size", 0) == 0 and geo.get("beta_star_min") is not None:
                _err(f"sample {sid} step {st.get('step')}: R_size=0 但 beta_star_min 非 None")

    print(
        f"VERIFY {path.name}: {len(samples)} samples, {n_step} steps, "
        f"{n_err} errors → {'PASS' if n_err == 0 else 'FAIL'}"
    )
    return len(samples), n_err, n_step


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="FAD/S5 geometry archive dumper")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed_test", type=int, default=123)
    parser.add_argument("--n_test", type=int, default=300)
    parser.add_argument("--layer_early", type=int, default=20, help="ℓ*（1.7B=20，8B=28）")
    parser.add_argument("--rank_threshold", type=int, default=50)
    parser.add_argument(
        "--operator",
        type=str,
        default="baseline",
        choices=["baseline", "sym", "lift", "damp"],
        help="轨迹算子：baseline=纯 greedy（阶段 1）；sym/lift/damp=阶段 2 单侧/对称轨迹",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="阶段 2 算子强度（须 ∈ BETAS）")
    parser.add_argument("--max_new", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "outputs" / "geometry_archive"),
    )
    parser.add_argument("--selftest", action="store_true", help="纯函数单测（无模型），随后退出")
    parser.add_argument("--verify", type=str, default=None, help="离线校验已存档案（无模型），随后退出")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.verify:
        verify_archive(args.verify)
        return

    if args.operator != "baseline":
        if args.beta not in BETAS:
            sys.exit(
                f"预注册纪律：--beta={args.beta} 不在 BETAS={BETAS} 内。"
                "禁止用网格外 β 生成阶段 2 轨迹（事后挑结果）——若确需新 β，"
                "先改 BETAS 常量并重新登记 spec。"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 72)
    print("Geometry archive dumper (FAD P5.1 / S5 P-S5.1–5.3)")
    print(f"  model={args.model} ℓ*={args.layer_early} op={args.operator} "
          f"β={args.beta} seed={args.seed_test} n={args.n_test}")
    print("=" * 72)

    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    final_layer = model.cfg.n_layers - 1
    print(f"  loaded: {model.cfg.n_layers} layers, L={final_layer}, "
          f"V={model.cfg.d_vocab_out}")

    print(f"[1/3] classify (seed={args.seed_test})...")
    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)
    test_samples = test_samples[: args.n_test]
    entries = classify_samples(
        model, tokenizer, test_samples, device, args.layer_early, args.rank_threshold
    )
    n_kw = sum(1 for e in entries if e["subset"] == "know_wrong")
    n_kc = sum(1 for e in entries if e["subset"] == "know_correct")
    print(f"  KC={n_kc}, KW={n_kw}, DK={len(entries) - n_kc - n_kw}")

    print(f"[2/3] dump ({args.operator} trajectory)...")
    samples_out = []
    from tqdm import tqdm

    for e in tqdm(entries, desc="  dump"):
        res = dump_sample(
            model, tokenizer, e, device, args.layer_early, W_U, b_U, ln_final,
            args.operator, args.beta, args.max_new,
        )
        samples_out.append(
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "rank": e["rank"],
                "question": e["question"],
                "answers": e["answers"],
                "y_true_id": e["y_true_id"],
                "baseline_correct": e["is_correct"],  # classify 阶段的基线 greedy 判定
                "is_correct": check_correct_exact(res["gen_text"], e["answers"]),  # exact
                "gids": res["gids"],
                "gen_text": res["gen_text"],
                "steps": res["steps"],
            }
        )

    out = {
        "meta": {
            "model": args.model,
            "layer_early": args.layer_early,
            "final_layer": final_layer,
            "operator": args.operator,
            "beta": args.beta,
            "seed_test": args.seed_test,
            "n_test": args.n_test,
            "rank_threshold": args.rank_threshold,
            "max_new": args.max_new,
            "betas": BETAS,
            "top_k": TOP_K,
            "theory_refs": THEORY_REFS,
            "script": Path(__file__).name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "samples": samples_out,
    }
    fname = f"geo_{args.operator}"
    if args.operator != "baseline":
        fname += f"_b{args.beta:.2f}".replace(".", "")
    out_path = output_dir / f"{fname}_seed{args.seed_test}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[3/3] saved → {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    # 自校验刚写出的档案
    verify_archive(out_path)


if __name__ == "__main__":
    main()
