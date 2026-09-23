"""TLDC 零机制/安慰剂对照臂（2026-09-21）——回答「救回是知识特异还是通用扰动」。

═══════════════════════════════════════════════════════════════════════════════
理论先行（规则 1）
═══════════════════════════════════════════════════════════════════════════════
【问题形式化】
  干预算子（TLDC）：在每步 t 用双读出构造扰动 Δ_t = β·(l_ℓ*(t) − l_L(t)) = β·δ_t，
  施加于末层 logits：l' = l_L + Δ_t，贪心取 argmax。
  样本按基线行为分层：KW（rank≤50 且 exact 错）、KC（rank≤50 且 exact 对）、DK（rank>50）。
  观测：KW 组救回率 r_KW、KC 组破坏率 b_KC、DK 组同向率（参考）。

【机制假说】
  H1（知识特异）：救回依赖 δ_t 的**内容**——即参考层对正确 token 的相对偏好。
      ⇒ 破坏 δ 与 token 的对应关系（但保持幅度分布）应当显著降低 r_KW。
  H2（通用扰动）：救回只依赖**扰动的幅度**（把 logits 推离原 argmax，偶发翻到正确 token）。
      ⇒ 任何同幅度扰动应给出与真实 δ 同量级的 r_KW（且与 DK 参考组同量级）。

【可检验预测】（预注册，见 docs/protocol/placebo-control-protocol.md）
  P-A（shuffle 臂，主判据）：臂 A1 把 δ 的**词表条目随机置换**（多重集/范数完全保持，
     仅破坏 token 对齐）。若 H1 成立 → r_KW(A1) 显著低于 r_KW(real)；若 H2 成立 → 两者无差异。
  P-B（gauss 臂）：同 L2 范数的各向同性高斯扰动，检验"仅幅度"假说。
  P-C（wrong-layer 臂）：用**别的层**的读出（ℓ'≠ℓ*，范数匹配）做同样的对比插值。
      若 ℓ* 不特殊（任何层都行）→ r_KW 与 real 无差异 ⇒ FAD/I01「选层准则」组件贬值。
  P-D（anti 臂）：Δ 取负号（−β·δ）。若方向有信息 → r_KW 应≈0 而 b_KC 升高。
  P-E（real 臂，协议自检）：必须复现 validate_s14_tldc.py 已发布的 (KW,KC) 数字
      （同 seed/同 n/同 β），否则说明本脚本协议漂移，其余臂作废。

【失败模式预判】
  F1. KW 救回事件本身稀少（8B 双 seed pooled β=0.20：21/122）⇒ 单臂检验功效有限；
      故主判据用**配对** McNemar（real vs 对照，同批 KW 样本），而非两组独立比率。
  F2. shuffle 臂若与 DK 参考率同量级，说明"救回"在该臂下就是运气——这正是 H2 的证据，
      但需注意 DK 与 KW 的 rank 分布不同，解读须并列报两列。
  F3. wrong-layer 臂的 δ 范数与 real 不同 ⇒ 必须**重标定到同一范数**，否则混淆"层"与"幅度"。
  F4. GPU 数值：末层 l_L 必须模型真实 logits（禁 lens 代验，T04）；ℓ* 读出用 lens 是算子定义本身。
  F5. 纯 logits 锐化/top-k 截断**不是**有效对照：它们是单调变换，不改变 argmax ⇒ 在 greedy
      解码下恒为 0 救回 0 破坏（T15 的 R2/APC 臂只在采样解码下才有意义）——故本套不设该臂，
      以免把"结构性零效应"误读为"无通用扰动"。

【红线】
  - 不得只报 real 臂的 KW 救回率而不报对照臂与 DK 参考（runbook §5 第 7 条）。
  - 不得事后更换 β 网格或主判据（多重比较纪律；β=0.20 为主、0.03 为次，其余描述性）。
  - 引用 T15 时注意其域限：MLLM 对象幻觉 / POPE / ≤13B。

═══════════════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════════════
  自检（无模型）：      python experiments/lin_theory/main_tldc_controls.py --selftest
  本地 1.7B 冒烟：      --n_test 30 --arms real shuffle --betas 0.2
  本地 1.7B 全量：      --n_test 300 --arms real shuffle gauss anti wrong_late wrong_zero \
                        --betas 0.03 0.2
  服务器 8B：          同上 + --model <Qwen3-8B 快照路径> --layer_early 28
"""

import argparse
import json
import os
import sys
import time
import zlib
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
)
from analyze_transition_matrix import (  # noqa: E402
    clopper_pearson,
    fisher_greater,
    mcnemar_exact,
)
from common import load_model_and_unembed  # noqa: E402
from src.data_loader import check_correct_exact, load_triviaqa  # noqa: E402

# 预注册：β 网格（与 s14 一致）+ 主/次判据档
BETAS_ALLOWED = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
BETA_PRIMARY = 0.20  # 主判据档（KW 救回最强、知识特异最明显）
BETA_SECONDARY = 0.03  # 次判据档（最低工作点）

ARM_DOC = {
    "real": "真实 TLDC（Δ=β·δ）——协议自检臂，须复现已发布数字",
    "shuffle": "δ 词表随机置换（范数/多重集完全保持，仅破坏 token 对齐）— 主判据",
    "gauss": "同 L2 范数各向同性高斯（无结构，仅幅度）",
    "anti": "Δ=−β·δ（方向取反）",
    "wrong_late": "用 L24 读出替代 ℓ*（范数重标定到 ||δ||）",
    "wrong_zero": "用 L0 读出替代 ℓ*（范数重标定到 ||δ||）",
    # ── T3 门控族（2026-09-21 晚新增；占位核验见 docs/plans/current.md §P0-TLDC T3）──
    "gated_margin": "应用门 G1：仅当末层前二 margin < τ 才施加 β·δ（判据=引理 1 的 m）",
    "gated_betastar": "应用门 G2：仅当 β*_min ≤ β（可负担）才施加 β·δ",
    "gated_damp": "应用门 G3：margin < τ 时只施加压支 −β·ReLU(−δ)（翻转 100% 由压支驱动的推论）",
    # ── 验证器臂（2026-09-22 晚新增；§5.8 事后方向可分性实测的落地）──
    "verified_sym": "验证器臂：每步算对称 TLDC 翻转；若与基线 argmax 不同，则做【一步前瞻】——"
                    "在新前缀上读 β*_min，≥θ 保留翻转、<θ 回退为基线 token"
                    "（θ 由 --verify_theta 预注册，默认 0.5；None＝R 空＝翻不动＝保留）",
    # ── 验证器安慰剂臂（2026-09-23 新增：V1 的 footprint 混淆排除）──
    "verified_rand": "安慰剂验证臂（footprint 匹配）：与 verified_sym **同一翻转规则**，但**不读** β*_min ——"
                     "保留/回退用 seeded Bernoulli(p) 随机决定，p 由 --verify_keep_prob 逐子集标定"
                     "（＝ V1 实测 keep_rate_of_flips，只按决策计数标定、不看结果）⇒ 只保留「扰动步数」"
                     "成分、破坏「选择信息」成分。⚠️ 首次决策分歧后轨迹分叉 ⇒ 翻转集只在分叉前与"
                     "verified_sym 相同（与门控臂同款的轨迹级限制，非实现缺陷）。"
                     "判读：verified_sym vs verified_rand",
}
GATED_ARMS = ("gated_margin", "gated_betastar", "gated_damp")
VERIFY_ARMS = ("verified_sym",)          # 真验证器（用 β*_min 信息）
VERIFY_RAND_ARMS = ("verified_rand",)    # 安慰剂（同机器、随机决策）
VERIFY_FAMILY = VERIFY_ARMS + VERIFY_RAND_ARMS


def loop_branch(arm):
    """生成循环的分支归属（**单一来源**）："verify" / "gated" / "control"。

    存在的理由（2026-09-23 实跑事故回归）：循环里的分支条件与 `d_ctrl` 守卫若各写一份判断，
    两者可能不一致——原 bug 即分支写 `VERIFY_FAMILY`、守卫写 `VERIFY_ARMS`（只含 verified_sym）
    ⇒ `verified_rand` 进分支后仍被要求算 `d_ctrl`（验证器族从不赋值）→ UnboundLocalError，
    整轮三臂 ~28 分钟 GPU 白跑。现在分支与守卫都取本函数的返回值，并由 selftest 第 8 项锁死映射。
    """
    if arm in VERIFY_FAMILY:
        return "verify"
    if arm in GATED_ARMS:
        return "gated"
    return "control"


def parse_keep_prob(spec):
    """解析 `--verify_keep_prob`：单标量 → 全体同一 p；`sub=p,sub=p` → 逐子集 p（缺失回退 _default）。

    ⚠️ p 是**footprint 匹配参数**（由 V1 实测保留率标定），不是待优化超参、不得按结果调。
    """
    spec = str(spec).strip()
    if "=" not in spec:
        return {"_default": float(spec)}
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    if "_default" not in out:
        out["_default"] = sum(out.values()) / len(out) if out else 0.65
    return out


def keep_prob_for(spec_map, subset):
    """取该子集的保留概率（无该子集条目则用 _default）。"""
    return spec_map.get(subset, spec_map["_default"])


# ═════════════════════════════════════════════════════════════════════════════
# 门控判据（纯函数，可单测）
# ═════════════════════════════════════════════════════════════════════════════


def top2_margin(l_final):
    """末层 logits 前二差（= 引理 1 的 m，当 a 为 argmax 时）。"""
    v = l_final.float().squeeze().topk(2).values
    return float((v[0] - v[1]).item())


def min_flip_beta(l_early, l_final):
    """β*_min = min_{c∈R} m/(m+Δ₀)，R={c: l_ℓ*(c) > l_ℓ*(a)}；R 空返回 None。"""
    l0 = l_early.float().squeeze()
    l1 = l_final.float().squeeze()
    a = int(l1.argmax().item())
    l0a = l0[a]
    mask = l0 > l0a
    if not bool(mask.any().item()):
        return None
    m = l1[a] - l1
    d0 = l0 - l0a
    r = torch.where(mask, m / (m + d0), torch.full_like(m, float("inf")))
    return float(r.min().item())


def verify_decision(beta_next, theta):
    """verified_sym 的一步前瞻决策（纯函数，可单测）。

    β*_min ≥ θ ⇒ 新状态稳固 ⇒ 保留翻转；< θ ⇒ 脆弱 ⇒ 回退。
    `None`＝下一步 R 为空（任何 token 都翻不动）＝最稳固 ⇒ 保留。
    θ 预注册 0.5（runbook §5.8b），**不得按结果更改**。
    """
    return beta_next is None or beta_next >= theta


def gate_open(arm, margin_L, beta_min, beta, tau):
    """门控决策（0/1）。返回 (是否施加, 诊断字典)。

    G1 gated_margin：margin_L < τ；G2 gated_betastar：beta_min ≤ β；
    G3 gated_damp：同 G1（另在生成侧只取压支）。
    """
    if arm == "gated_margin" or arm == "gated_damp":
        return margin_L < tau, {"margin_L": margin_L}
    if arm == "gated_betastar":
        ok = beta_min is not None and beta_min <= beta
        return ok, {"beta_star_min": beta_min}
    raise ValueError(f"{arm} 不是门控臂")


# ═════════════════════════════════════════════════════════════════════════════
# 扰动构造（纯函数，可单测）
# ═════════════════════════════════════════════════════════════════════════════


def make_control_delta(delta, arm, gen, ctrl_delta=None):
    """把真实扰动 δ 变换为对照臂扰动（保持 L2 范数）。

    delta: [1, V] 真实扰动（未乘 β）；ctrl_delta: wrong_* 臂的另一层扰动。
    返回 [1, V]，范数与 delta 一致（wrong_* 臂按比例重标定）。
    """
    if arm == "real":
        return delta
    if arm == "shuffle":
        perm = torch.randperm(delta.shape[-1], generator=gen, device=delta.device)
        return delta[..., perm]
    if arm == "gauss":
        g = torch.randn(delta.shape, generator=gen, device=delta.device, dtype=delta.dtype)
        return g * (delta.norm() / g.norm().clamp_min(1e-12))
    if arm == "anti":
        return -delta
    if arm.startswith("wrong"):
        if ctrl_delta is None:
            raise ValueError(f"{arm} 需要 ctrl_delta")
        scale = delta.norm() / ctrl_delta.norm().clamp_min(1e-12)  # 范数重标定（F3）
        return ctrl_delta * scale
    raise ValueError(f"未知 arm: {arm}")


# ═════════════════════════════════════════════════════════════════════════════
# 生成（带对照扰动）
# ═════════════════════════════════════════════════════════════════════════════


def controlled_greedy_generate(
    model,
    tokenizer,
    prompt,
    device,
    layer_early,
    W_U,
    b_U,
    ln_final,
    beta,
    arm,
    ctrl_layer=None,
    max_new=20,
    sample_id=0,
    gate_tau=0.2,
    verify_theta=0.5,
    keep_prob=0.65,
):
    """贪心生成，每步施加对照/门控扰动；返回 (文本, 扰动范数比列表, 门控/验证统计)。"""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]  # 截断保尾

    hook_early = f"blocks.{layer_early}.hook_resid_post"
    hook_ctrl = f"blocks.{ctrl_layer}.hook_resid_post" if arm.startswith("wrong") else None
    captured = {}
    arm_offset = zlib.crc32(arm.encode()) % 100000

    def _hook(name):
        def fn(act, hook=None):
            captured[name] = act[:, -1:, :].detach()
            return act

        return fn

    hooks = [(hook_early, _hook("h_early"))]
    if hook_ctrl:
        hooks.append((hook_ctrl, _hook("h_ctrl")))

    gids = []
    norm_ratios = []
    gate_flags = []
    gate_diags = []
    vstat = None
    branch = loop_branch(arm)          # "verify" / "gated" / "control"（单一来源，见 loop_branch）
    if branch == "verify":
        vstat = {"n_noflip": 0, "n_flip": 0, "n_kept": 0, "n_kept_inf": 0,
                 "n_reverted": 0, "kept_beta_sum": 0.0, "rev_beta_sum": 0.0,
                 "n_kept_beta": 0, "n_rev_beta": 0}
    nid = None
    for step in range(max_new):
        with torch.no_grad():
            logits_final = model.run_with_hooks(tokens, fwd_hooks=hooks)

        l_final = logits_final[0, -1:, :].float()  # TRUE final logits（T04）
        l_early = compute_early_exit_logits(captured["h_early"], ln_final, W_U, b_U)
        delta = l_early - l_final

        ctrl_delta = None
        if hook_ctrl:
            l_ctrl = compute_early_exit_logits(captured["h_ctrl"], ln_final, W_U, b_U)
            ctrl_delta = l_ctrl - l_final

        # 每 (arm, sample, step) 固定随机种子 → 可复现
        gen = torch.Generator(device=delta.device)
        gen.manual_seed(1234 + 7919 * sample_id + 31 * step + arm_offset)

        if branch == "verify":
            # 一步前瞻验证：先算对称 TLDC 会翻成什么；只在与基线不同时才付出前瞻代价
            nid_base = int(l_final.argmax(dim=-1).item())
            nid_sym = int((l_final + beta * delta).argmax(dim=-1).item())
            if nid_sym == nid_base:
                nid = nid_base
                norm_ratios.append(0.0)
                vstat["n_noflip"] += 1
            else:
                vstat["n_flip"] += 1
                if arm in VERIFY_RAND_ARMS:
                    # 安慰剂：**不读** β*_min，只按子集匹配的保留概率随机决定（seeded ⇒ 可复现）
                    keep = torch.rand((), generator=gen, device=delta.device).item() < keep_prob
                    b_next = None
                else:
                    tokens_try = torch.cat(
                        [tokens, torch.tensor([[nid_sym]], device=device)], dim=1
                    )
                    with torch.no_grad():
                        logits_try = model.run_with_hooks(tokens_try, fwd_hooks=hooks)
                    l_final_try = logits_try[0, -1:, :].float()
                    l_early_try = compute_early_exit_logits(
                        captured["h_early"], ln_final, W_U, b_U
                    )
                    b_next = min_flip_beta(l_early_try, l_final_try)
                    keep = verify_decision(b_next, verify_theta)   # 稳固 ⇒ 保留翻转
                if keep:
                    nid = nid_sym
                    norm_ratios.append(1.0)
                    if b_next is None:
                        vstat["n_kept_inf"] += 1
                    else:
                        vstat["n_kept"] += 1
                        vstat["kept_beta_sum"] += b_next
                        vstat["n_kept_beta"] += 1
                else:                                        # 脆弱 ⇒ 回退到基线 token
                    nid = nid_base
                    norm_ratios.append(0.0)
                    vstat["n_reverted"] += 1
                    if b_next is not None:
                        vstat["rev_beta_sum"] += b_next
                        vstat["n_rev_beta"] += 1
        elif branch == "gated":
            gated, diag = gate_open(
                arm, top2_margin(l_final), min_flip_beta(l_early, l_final), beta, gate_tau
            )
            gate_flags.append(bool(gated))
            if not gated:
                d_ctrl = torch.zeros_like(delta)  # 门关：不动
                norm_ratios.append(0.0)
            elif arm == "gated_damp":
                d_ctrl = -torch.relu(-delta)  # 只取压支（保持 δ 的原幅，不重标定）
                norm_ratios.append(float((d_ctrl.norm() / delta.norm().clamp_min(1e-12)).item()))
            else:
                d_ctrl = delta
                norm_ratios.append(1.0)
            gate_diags.append(diag)
        else:
            d_ctrl = make_control_delta(delta, arm, gen, ctrl_delta)
            norm_ratios.append(float((d_ctrl.norm() / delta.norm().clamp_min(1e-12)).item()))

        if branch != "verify":
            logits_adj = l_final + beta * d_ctrl
            nid = int(logits_adj.argmax(dim=-1).item())
        gids.append(nid)
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)

    gate_stat = None
    if arm in GATED_ARMS:
        gate_stat = {
            "n_steps": len(gate_flags),
            "n_gated": int(sum(gate_flags)),
            "gated_frac": (sum(gate_flags) / len(gate_flags)) if gate_flags else None,
        }
    elif arm in VERIFY_FAMILY:
        n_flip = vstat["n_flip"]
        n_kept = vstat["n_kept"] + vstat["n_kept_inf"]
        gate_stat = {
            "n_steps": len(norm_ratios),
            "n_noflip": vstat["n_noflip"],
            "n_flip": n_flip,
            "n_kept": n_kept,
            "n_kept_finite": vstat["n_kept"],
            "n_kept_inf": vstat["n_kept_inf"],
            "n_reverted": vstat["n_reverted"],
            "flip_rate": (n_flip / len(norm_ratios)) if norm_ratios else None,
            "keep_rate_of_flips": (n_kept / n_flip) if n_flip else None,
            # 均值只对"算了 β*"的步求（安慰剂臂 b_next 恒为 None ⇒ 两项均为 None，不是 0）
            "mean_kept_beta": (vstat["kept_beta_sum"] / vstat["n_kept_beta"]) if vstat["n_kept_beta"] else None,
            "mean_reverted_beta": (vstat["rev_beta_sum"] / vstat["n_rev_beta"]) if vstat["n_rev_beta"] else None,
            "verify_theta": verify_theta,
            "verify_rule": ("bernoulli_keep_prob" if arm in VERIFY_RAND_ARMS else "beta_star_min>=theta"),
            "keep_prob": keep_prob if arm in VERIFY_RAND_ARMS else None,
        }
    return tokenizer.decode(gids).strip(), norm_ratios, gate_stat


# ═════════════════════════════════════════════════════════════════════════════
# 统计（含配对 real-vs-对照）
# ═════════════════════════════════════════════════════════════════════════════


def arm_stats(samples, arm, beta, subset_key="know_wrong"):
    """单臂统计：逐格转移 + 子集率 + DK 参考 + McNemar。"""
    key = f"correct_{arm}_beta{beta}"
    res = brk = 0
    n = 0
    per_sub = {}
    for v in samples.values():
        if key not in v or v.get("baseline_correct") is None:
            continue
        n += 1
        base, final = v["baseline_correct"], v[key]
        s = per_sub.setdefault(v["subset"], {"n": 0, "bw": 0, "br": 0, "rescue": 0, "break": 0})
        s["n"] += 1
        s["bw" if not base else "br"] += 1
        if not base and final:
            res += 1
            s["rescue"] += 1
        elif base and not final:
            brk += 1
            s["break"] += 1
    for s in per_sub.values():
        s["rescue_rate"] = s["rescue"] / s["bw"] if s["bw"] else None
        s["break_rate"] = s["break"] / s["br"] if s["br"] else None

    kw, dk, kc = per_sub.get(subset_key, {}), per_sub.get("dont_know", {}), per_sub.get("know_correct", {})
    out = {
        "arm": arm,
        "beta": beta,
        "n": n,
        "rescue": res,
        "break": brk,
        "net_pp": (res - brk) / n * 100 if n else 0.0,
        "mcnemar_p": mcnemar_exact(res, brk),
        "per_subset": per_sub,
    }
    if kw and dk and kw.get("bw") and dk.get("bw"):
        out["rescue_kw"] = f"{kw['rescue']}/{kw['bw']}"
        out["rescue_dk"] = f"{dk['rescue']}/{dk['bw']}"
        r_dk = dk["rescue_rate"]
        out["ratio_kw_dk"] = (kw["rescue_rate"] / r_dk) if r_dk else None
        out["fisher_kw_vs_dk"] = fisher_greater(
            kw["rescue"], kw["bw"] - kw["rescue"], dk["rescue"], dk["bw"] - dk["rescue"]
        )
        out["kw_rescue_ci"] = clopper_pearson(kw["rescue"], kw["bw"])
    if kc and dk and kc.get("br") and dk.get("br"):
        out["break_kc"] = f"{kc['break']}/{kc['br']}"
        out["break_dk"] = f"{dk['break']}/{dk['br']}"
    return out


def paired_vs_real(samples, ctrl_arm, beta):
    """主判据：KW 样本上 real vs 对照 的配对 McNemar（救回事件）。

    仅统计基线错的 KW 样本：b=real 救回且对照未救回，c=对照救回且 real 未救回。
    """
    kr, kc_ = f"correct_real_beta{beta}", f"correct_{ctrl_arm}_beta{beta}"
    b = c = 0
    for v in samples.values():
        if v.get("subset") != "know_wrong" or v.get("baseline_correct"):
            continue
        if kr not in v or kc_ not in v:
            continue
        if v[kr] and not v[kc_]:
            b += 1
        elif v[kc_] and not v[kr]:
            c += 1
    return {"b_real_only": b, "c_ctrl_only": c, "mcnemar_p": mcnemar_exact(b, c)}


# ═════════════════════════════════════════════════════════════════════════════
# 自检（无模型）
# ═════════════════════════════════════════════════════════════════════════════


def selftest():
    torch.manual_seed(0)
    d = torch.randn(1, 64)
    ctrl = torch.randn(1, 64) * 3.0
    gen = torch.Generator().manual_seed(7)

    # 1) 范数匹配：所有对照臂 ||Δ_ctrl|| == ||δ||
    for arm, extra in [("shuffle", None), ("gauss", None), ("anti", None), ("wrong_late", ctrl)]:
        out = make_control_delta(d, arm, gen, extra)
        assert out.shape == d.shape
        assert abs(out.norm().item() - d.norm().item()) < 1e-4, (arm, out.norm().item(), d.norm().item())
    print("  [1/4] 范数匹配（shuffle/gauss/anti/wrong_late）✓")

    # 2) shuffle 保持多重集（排序后逐元素相等）；gauss 破坏对齐；anti = −δ
    sh = make_control_delta(d, "shuffle", gen)
    assert torch.allclose(sh.sort().values, d.sort().values)
    assert not torch.allclose(sh, d)
    assert torch.allclose(make_control_delta(d, "anti", gen), -d)
    g = make_control_delta(d, "gauss", gen)
    cos = torch.nn.functional.cosine_similarity(g.flatten(), d.flatten(), dim=0).abs().item()
    assert cos < 0.5, f"gauss 不应与 δ 对齐（cos={cos}）"
    print("  [2/4] shuffle 多重集保持 / anti 取反 / gauss 去对齐 ✓")

    # 3) 复现性：同种子两次调用结果一致
    a1 = make_control_delta(d, "gauss", torch.Generator().manual_seed(11))
    a2 = make_control_delta(d, "gauss", torch.Generator().manual_seed(11))
    assert torch.allclose(a1, a2)
    print("  [3/4] 同种子复现 ✓")

    # 4) 配对统计：构造 b=6, c=1 → McNemar p 应 ≈0.125
    p = mcnemar_exact(6, 1)
    assert abs(p - 0.125) < 1e-9, p
    print(f"  [4/6] 配对 McNemar(6,1)= {p:.4f}（期望 0.125）✓")

    # 6) verified_sym 一步前瞻决策（预注册 θ=0.5）
    assert verify_decision(None, 0.5) is True          # R 空 ⇒ 最稳固 ⇒ 保留
    assert verify_decision(0.9, 0.5) is True           # 稳固 ⇒ 保留
    assert verify_decision(0.5, 0.5) is True           # 边界含等号
    assert verify_decision(0.25, 0.5) is False         # 脆弱 ⇒ 回退
    print("  [6/6] 前瞻验证决策（None/≥θ 保留、<θ 回退）✓")

    # 5) 门控判据（T3）：top2_margin / min_flip_beta / gate_open
    lf = torch.tensor([[3.0, 2.9, 0.0]])      # margin = 0.1
    le = torch.tensor([[1.0, 3.5, 0.5]])      # a=0(l1 argmax)；R={1}: l0(1)=3.5>l0(0)=1.0
    assert abs(top2_margin(lf) - 0.1) < 1e-6
    bs = min_flip_beta(le, lf)                # m=3.0-2.9=0.1, Δ₀=3.5-1.0=2.5 → 0.1/2.6
    assert bs is not None and abs(bs - 0.1 / 2.6) < 1e-5, bs
    assert gate_open("gated_margin", 0.1, bs, 0.2, 0.2)[0] is True    # 0.1<0.2 开门
    assert gate_open("gated_margin", 0.25, bs, 0.2, 0.2)[0] is False  # 关门
    assert gate_open("gated_betastar", 0.1, bs, 0.2, 0.2)[0] is True   # β*=0.038≤0.2 开门
    assert gate_open("gated_betastar", 0.1, bs, 0.01, 0.2)[0] is False # β*>β 关门
    assert gate_open("gated_betastar", 0.1, None, 0.2, 0.2)[0] is False  # R 空 → 关门
    assert gate_open("gated_damp", 0.1, bs, 0.2, 0.2)[0] is True
    print("  [5/5] 门控判据（margin/β*_min/gate_open 三臂）✓")

    # 7) 安慰剂臂（verified_rand）footprint 标定解析 + 抽签可复现
    mp = parse_keep_prob("know_wrong=0.6053,know_correct=0.6991,dont_know=0.6506")
    assert abs(keep_prob_for(mp, "know_wrong") - 0.6053) < 1e-12
    assert abs(keep_prob_for(mp, "dont_know") - 0.6506) < 1e-12
    # 缺失子集 ⇒ _default＝已给各档均值（不是 0、不报错）
    assert abs(keep_prob_for(mp, "未列子集") - (0.6053 + 0.6991 + 0.6506) / 3) < 1e-9
    m1 = parse_keep_prob("0.8")
    assert m1 == {"_default": 0.8} and keep_prob_for(m1, "know_wrong") == 0.8
    # 默认标定串必须可解析且与 V1 实测一致
    dflt = parse_keep_prob("know_wrong=0.6053,know_correct=0.6991,dont_know=0.6506")
    assert set(dflt) == {"know_wrong", "know_correct", "dont_know", "_default"}
    # 抽签可复现：同 (sample_id, step) → 同决策（臂内 gen.manual_seed(1234+7919*sid+31*step+arm_offset)）
    g1 = torch.Generator().manual_seed(1234 + 7919 * 3 + 31 * 5)
    g2 = torch.Generator().manual_seed(1234 + 7919 * 3 + 31 * 5)
    assert torch.rand((), generator=g1).item() == torch.rand((), generator=g2).item()
    print("  [7/7] 安慰剂臂 footprint 标定解析（逐子集/标量/缺省）+ 抽签可复现 ✓")

    # 8) 分支归属一致性（2026-09-23 实跑 bug 回归）：循环分支与 d_ctrl 守卫必须同源。
    #    原 bug＝分支写 VERIFY_FAMILY、守卫写 VERIFY_ARMS ⇒ verified_rand 走分支后仍被要求算 d_ctrl。
    assert loop_branch("verified_sym") == "verify"
    assert loop_branch("verified_rand") == "verify", "安慰剂臂属验证器族（不赋值 d_ctrl）"
    assert "verified_rand" not in VERIFY_ARMS and "verified_rand" in VERIFY_FAMILY
    for a in GATED_ARMS:
        assert loop_branch(a) == "gated", a
    for a in ARM_DOC:
        assert loop_branch(a) in ("verify", "gated", "control"), a
        if a.startswith("verified"):        # 命名属验证器族 ⇒ 必须已登记进 VERIFY_FAMILY
            assert loop_branch(a) == "verify", f"{a} 未登记进 VERIFY_FAMILY（会落到 d_ctrl 分支）"
    assert set(VERIFY_FAMILY).isdisjoint(GATED_ARMS)
    print("  [8/8] 分支归属一致（d_ctrl 守卫同源；verified_rand ∈ verify 族）✓")
    print("SELFTEST PASS: 8/8")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="TLDC 零机制/安慰剂对照臂")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer_early", type=int, default=20, help="ℓ*（1.7B=20，8B=28）")
    ap.add_argument("--ctrl_layer", type=int, default=24, help="wrong_late 用的层")
    ap.add_argument("--ctrl_layer_zero", type=int, default=0, help="wrong_zero 用的层")
    ap.add_argument("--arms", nargs="+", default=["real", "shuffle"])
    ap.add_argument("--betas", nargs="+", type=float, default=[BETA_PRIMARY])
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed_test", type=int, default=123)
    ap.add_argument("--rank_threshold", type=int, default=50)
    ap.add_argument("--verify_theta", type=float, default=0.5,
                    help="verified_sym 的一步前瞻阈值（预注册 0.5）：t+1 步 β*_min ≥ θ 保留翻转")
    ap.add_argument("--verify_keep_prob", type=str,
                    default="know_wrong=0.6053,know_correct=0.6991,dont_know=0.6506",
                    help="verified_rand 的保留概率（footprint 匹配参数，非超参）：单标量，或 "
                         "'know_wrong=..,know_correct=..,dont_know=..' 逐子集标定。默认值＝"
                         "V1（1.7B seed123 β=0.20 θ=0.5）实测 keep_rate_of_flips；"
                         "⚠️ 换模型/换 β/换 θ 必须用同批实测重新标定（只看决策计数，不看结果）")
    ap.add_argument("--gate_tau", type=float, default=0.2,
                    help="G1/G3 门控阈值：末层前二 margin < τ 才施加（预注册：0.2 或 0.3）")
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--refresh_classify", action="store_true",
                    help="强制重算分类（默认复用 _classify_cache，省约 11 分钟/轮）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    for b in args.betas:
        if b not in BETAS_ALLOWED:
            sys.exit(f"预注册纪律：β={b} 不在 {BETAS_ALLOWED}（禁止网格外扫描）")
    unknown = [a for a in args.arms if a not in ARM_DOC]
    if unknown:
        sys.exit(f"未知 arm: {unknown}（可选 {list(ARM_DOC)}）")
    keep_prob_map = parse_keep_prob(args.verify_keep_prob)
    if any(a in VERIFY_RAND_ARMS for a in args.arms):
        print(f"  [verified_rand] footprint 匹配保留概率：{keep_prob_map}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "outputs" / "tldc_controls"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("TLDC 零机制/安慰剂对照臂")
    print(f"  model={args.model} ℓ*={args.layer_early} arms={args.arms} β={args.betas}")
    print("=" * 76)

    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)
    print(f"  loaded: {model.cfg.n_layers} layers, V={model.cfg.d_vocab_out}")

    print(f"[1/3] classify (seed={args.seed_test}, n={args.n_test})...")
    # 分类结果缓存：classify 在 1.7B 上约 11 分钟/轮（300 样本 × 每次前向），
    # 且对 (model, ℓ*, rank_threshold, seed, n) 完全确定 ⇒ 跨 arm/β 运行复用。
    cache_dir = out_dir / "_classify_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(str(args.model).rstrip("/")).name
    cache_key = f"{model_tag}_l{args.layer_early}_r{args.rank_threshold}_s{args.seed_test}_n{args.n_test}"
    cache_path = cache_dir / f"classify_{cache_key}.json"
    if cache_path.exists() and not args.refresh_classify:
        entries = json.loads(cache_path.read_text())
        print(f"  [cache] 命中 {cache_path.name}（跳过 classify；--refresh_classify 可强制重算）")
    else:
        test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)[: args.n_test]
        entries = classify_samples(
            model, tokenizer, test_samples, device, args.layer_early, args.rank_threshold
        )
        cache_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        print(f"  [cache] 已写入 {cache_path.name}")
    print(f"  KC={sum(1 for e in entries if e['subset']=='know_correct')}, "
          f"KW={sum(1 for e in entries if e['subset']=='know_wrong')}, "
          f"DK={sum(1 for e in entries if e['subset']=='dont_know')}")

    samples = {}
    print("[2/3] generating under control arms...")
    from tqdm import tqdm

    for arm in args.arms:
        ctrl_layer = None
        if arm == "wrong_late":
            ctrl_layer = args.ctrl_layer
        elif arm == "wrong_zero":
            ctrl_layer = args.ctrl_layer_zero
        for beta in args.betas:
            key = f"correct_{arm}_beta{beta}"
            n_ok = 0
            for e in tqdm(entries, desc=f"  {arm}@β={beta}"):
                text, ratios, gate_stat = controlled_greedy_generate(
                    model, tokenizer, e["prompt"], device, args.layer_early, W_U, b_U,
                    ln_final, beta, arm, ctrl_layer=ctrl_layer, max_new=args.max_new,
                    sample_id=e["sample_id"], gate_tau=args.gate_tau,
                    verify_theta=args.verify_theta,
                    keep_prob=keep_prob_for(keep_prob_map, e["subset"]),
                )
                # 范数自检（F3）：非门控臂须 ≈1；门控臂只校验"开门"步（关门步比值为 0）；
                # gated_damp 只取压支 ⇒ 范数天然 <1，跳过该检查（幅度由 δ⁻ 决定，非重标定问题）
                bad = ([] if arm == "gated_damp"
                       else [r for r in ratios if r != 0.0 and abs(r - 1.0) > 0.05])
                if bad:
                    print(f"  [WARN] sample {e['sample_id']} {arm} 范数比偏离 1.0: {bad[:3]}")
                v = samples.setdefault(
                    e["sample_id"],
                    {"subset": e["subset"], "rank": e["rank"], "question": e["question"],
                     "baseline_correct": e["is_correct"]},
                )
                if gate_stat is not None:
                    v.setdefault("gate_stats", {})[f"{arm}_beta{beta}"] = gate_stat
                v[key] = check_correct_exact(text, e["answers"])  # exact 标签
                n_ok += int(v[key])
            print(f"    {arm}@β={beta}: 生成正确 {n_ok}/{len(entries)}")

    print("[3/3] 统计 + 保存...")
    report = {"config": vars(args), "arms": {}, "paired_vs_real": {}}
    for arm in args.arms:
        for beta in args.betas:
            st = arm_stats(samples, arm, beta)
            report["arms"][f"{arm}_beta{beta}"] = st
            if arm != "real" and ("real" in args.arms):
                report["paired_vs_real"][f"{arm}_beta{beta}"] = paired_vs_real(samples, arm, beta)

    out = {"meta": {"script": Path(__file__).name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "theory": "docs/protocol/placebo-control-protocol.md", "arms_doc": ARM_DOC},
           "report": report, "samples": {str(k): v for k, v in samples.items()}}
    # τ 须进文件名：门控族同一 seed/arms/β 下 τ=0.2 与 τ=0.3 仅差 gate_tau，
    # 原命名（seed_arms）会让次档静默覆盖主档（2026-09-22 修）。
    gate_tag = f"_tau{args.gate_tau}" if any(a in GATED_ARMS for a in args.arms) else ""
    tag = f"{args.seed_test}_{'-'.join(args.arms)}{gate_tag}"
    path = out_dir / f"tldc_controls_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ── 报告 ──
    print("\n" + "=" * 76)
    print(f"{'arm':>12} {'β':>5} {'KW 救回':>12} {'KW/DK':>7} {'Fisher p':>9} "
          f"{'KC 破坏':>10} {'净 pp':>7}")
    for arm in args.arms:
        for beta in args.betas:
            st = report["arms"][f"{arm}_beta{beta}"]
            ratio = f"{st.get('ratio_kw_dk'):.2f}×" if st.get("ratio_kw_dk") else "—"
            fp = f"{st.get('fisher_kw_vs_dk'):.4f}" if st.get("fisher_kw_vs_dk") is not None else "—"
            print(f"{arm:>12} {beta:>5} {st.get('rescue_kw','—'):>12} {ratio:>7} {fp:>9} "
                  f"{st.get('break_kc','—'):>10} {st['net_pp']:>+7.1f}")
    if report["paired_vs_real"]:
        print("\n主判据（KW 样本配对 McNemar：real 独有救回 vs 对照独有救回）")
        for k, v in report["paired_vs_real"].items():
            print(f"  {k}: b={v['b_real_only']}, c={v['c_ctrl_only']}, p={v['mcnemar_p']:.4f}")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
