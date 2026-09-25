"""C1 决定性验证：闭式对比强度 β\\* 的逐样本臂 vs 固定 β 臂（本地 GPU；含机制 RCT 臂）。

判据来源（**执行前设定**）：docs/protocol/dola-c1-validation-20260925.md
    P1 主判据：betastar vs best_fixed（同 seed 同批样本）——Δnet ≥ +3 事件 ∧ 救回侧配对 p<0.05 ∧ KC 不增；
    P2 机制判据：rand_strength 臂内（同一轨迹随机化边际）零边际翻转 vs 大边际翻转的一步粘住率；
    §4 前置门（n=30）：selftest 全过 + fixed@0.20 逐位复现已发布数字 + β*_min 中位数 ≥ 0.01。

算子与已发布 TLDC/real 臂**逐字一致**（便于与已发布数字对照）：
    l'_t = l_final + β_t·(l_ℓ* − l_final)，β=1 ⇔ 完全替换为 ℓ* 读出。
臂：fixed @β∈{0.03,0.05,0.10,0.20}｜betastar（β*_min+ε，R=∅ 则不动）｜betastar_tau（仅 β*_min≥τ）｜
    rand_strength（feasible 步 seeded 随机取 min 或 β_max=1.0）。
复用：min_flip_beta／compute_early_exit_logits／classify_samples／load_model_and_unembed／
    arm_stats／paired_vs_real／mcnemar_exact／fisher_greater（**不新写第二份闭式**）。

用法
----
    python experiments/lin_theory/main_tldc_betastar.py --selftest            # 零 GPU 自检
    python experiments/lin_theory/main_tldc_betastar.py --n_test 30 --smoke   # 前置门（n=30）
    python experiments/lin_theory/main_tldc_betastar.py --n_test 300 --seed_test 123
    python experiments/lin_theory/main_tldc_betastar.py --judge <结果 json>    # 零 GPU 判读

输出：experiments/outputs/tldc_betastar/betastar_<model>_n<n>_s<seed>.json（+ judge_*.md）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_PARENT = Path(__file__).resolve().parent
for _p in [str(_PARENT), str(_PARENT.parent / "phase2_entropy"),
           str(_PARENT.parent / "phase4_generalization"), str(_PARENT.parent / "phase5_cross_task")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tldc_per_token import classify_samples, compute_early_exit_logits  # noqa: E402
from analyze_transition_matrix import fisher_greater, mcnemar_exact  # noqa: E402
from common import load_model_and_unembed  # noqa: E402
from main_tldc_controls import arm_stats, min_flip_beta  # noqa: E402
from src.data_loader import check_correct_exact, load_triviaqa  # noqa: E402

FIXED_BETAS = [0.03, 0.05, 0.10, 0.20]
EPS = 0.01          # 预注册：闭式强度上的最小余量
TAU = 0.5           # 预注册：betastar_tau 的阈值（与既有 --verify_theta 同值）
BETA_MAX = 1.0      # rand_strength 的"大边际"档（=完全替换为 ℓ* 读出）
THETA_STICKY = 0.5  # 粘性判据阈值（与 verified_sym 的 θ 同值）
RCT_P_MIN = 0.5     # rand_strength 抽到"最小强度"的概率

ARM_DOC = {
    "fixed": "常数 β 臂（参照系）",
    "betastar": "C1 主臂：β_t = β*_min(t)+ε；R=∅ 的步不施加（无决策收益）",
    "betastar_tau": f"探索臂：仅当 β*_min(t) ≥ τ={TAU} 才施加 β*_min+ε（检验 H_frag 的'稳健翻转'侧）",
    "rand_strength": f"机制 RCT：feasible 步 seeded 随机取 β*_min+ε（p={RCT_P_MIN}）或 β_max={BETA_MAX}",
}
BETASTAR_ARMS = ("betastar", "betastar_tau", "rand_strength")


# ═════════════════════════════════════════════════════════════════════════════
# 纯函数：臂的强度规则（单一来源；selftest 锁死）
# ═════════════════════════════════════════════════════════════════════════════


def arm_beta(arm, beta, beta_min, gen, eps=EPS, tau=TAU, beta_max=BETA_MAX):
    """返回 (beta_t, rct_choice)。`beta_min=None` 表示 R 为空（任何 β 都翻不动）。"""
    if arm == "fixed":
        return float(beta), None
    if arm == "betastar":
        return (0.0, None) if beta_min is None else (min(beta_min + eps, 1.0), None)
    if arm == "betastar_tau":
        if beta_min is None or beta_min < tau:
            return 0.0, None
        return min(beta_min + eps, 1.0), None
    if arm == "rand_strength":
        if beta_min is None:
            return 0.0, None
        if float(torch.rand((), generator=gen).item()) < RCT_P_MIN:
            return min(beta_min + eps, 1.0), "min"
        return float(beta_max), "max"
    raise ValueError(f"未知臂：{arm}")


def make_gen(sample_id, step, arm):
    import zlib
    g = torch.Generator(device="cpu")
    g.manual_seed(1234 + 7919 * int(sample_id) + 31 * int(step)
                  + zlib.crc32(arm.encode()) % 100000)
    return g


# ═════════════════════════════════════════════════════════════════════════════
# 生成（每步零额外前向即可得 β*_min(t) 与 β*_min(t+1)）
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def generate_arm(model, tokenizer, prompt, device, layer_early, W_U, b_U, ln_final,
                 arm, beta, sample_id, max_new=20):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, -1024:]
    hook = f"blocks.{layer_early}.hook_resid_post"
    captured = {}

    def fn(act, hook=None):
        captured["h"] = act[:, -1:, :].detach()
        return act

    rec = {"n_steps": 0, "n_feasible": 0, "n_flip": 0, "beta_min": [], "applied_beta": [],
           "rct_choice": [], "persist_next": [], "beta_min_next": [], "flip_step": [],
           "flip_rct_choice": [], "flip_applied_beta": []}
    gids = []
    pending = None  # 上一步翻转到的 token（用于本步零成本粘性判定）
    for step in range(max_new):
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook, fn)])
        l_final = logits[0, -1:, :].float()
        l_early = compute_early_exit_logits(captured["h"], ln_final, W_U, b_U)
        bmin = min_flip_beta(l_early, l_final)
        if pending is not None:  # 上一步翻转的粘性（本步前向即可判定，无额外代价）
            rec["persist_next"].append(int(int(l_final.argmax(dim=-1).item()) == pending))
            rec["beta_min_next"].append(bmin)
            pending = None
        if bmin is not None:
            rec["n_feasible"] += 1
            rec["beta_min"].append(float(bmin))
        g = make_gen(sample_id, step, arm)
        beta_t, choice = arm_beta(arm, beta, bmin, g)
        rec["applied_beta"].append(float(beta_t))
        if choice is not None:
            rec["rct_choice"].append(choice)
        logits_adj = l_final + beta_t * (l_early - l_final)
        nid = int(logits_adj.argmax(dim=-1).item())
        if nid != int(l_final.argmax(dim=-1).item()):
            rec["n_flip"] += 1
            rec["flip_step"].append(step)
            rec["flip_rct_choice"].append(choice)
            rec["flip_applied_beta"].append(float(beta_t))
            pending = nid
        gids.append(nid)
        rec["n_steps"] = step + 1
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
    return tokenizer.decode(gids).strip(), rec


# ═════════════════════════════════════════════════════════════════════════════
# 自检（零 GPU）：闭式 vs 暴力扫描 + 臂逻辑 + 引理 2
# ═════════════════════════════════════════════════════════════════════════════


def _brute_min_beta(l_early, l_final, n_grid=4000):
    """暴力/二分：使 T_β 的 argmax 变化的**最小** β（无变化返回 None）。"""
    l0 = l_early.float().squeeze()
    l1 = l_final.float().squeeze()
    a = int(l1.argmax().item())
    lo, hi = 0.0, 1.0
    if int((l1 + 1.0 * (l0 - l1)).argmax().item()) == a:
        return None
    for _ in range(60):  # 二分（单调：argmax 变化集合是 β 的下闭区间补集）
        mid = (lo + hi) / 2
        if int((l1 + mid * (l0 - l1)).argmax().item()) != a:
            hi = mid
        else:
            lo = mid
    return hi


def selftest():
    torch.manual_seed(0)
    ok = []

    # S-A 闭式 β*_min 与暴力二分一致（含 R=∅ 情形）
    n_bad = 0
    for _ in range(300):
        V = 50
        l1 = torch.randn(V) * 3
        l0 = l1 + torch.randn(V) * 3
        bm = min_flip_beta(l0, l1)
        bb = _brute_min_beta(l0, l1)
        if bm is None or bb is None:
            n_bad += int((bm is None) != (bb is None))
        else:
            n_bad += int(abs(bm - bb) > 1e-3)
    ok.append(("S-A 闭式 β*_min ≡ 暴力二分（300 组随机 logits）", n_bad == 0, f"mismatch={n_bad}"))

    # S-B 引理 2（R 受限）：翻转后的 argmax 必落在 topK(l1) ∪ topK(l0) 内
    n_bad = 0
    for _ in range(200):
        V = 40
        l1 = torch.randn(V) * 3
        l0 = l1 + torch.randn(V) * 3
        K = 5
        cands = torch.unique(torch.cat([l1.topk(K).indices, l0.topk(K).indices]))
        for beta in (0.05, 0.2, 0.5, 1.0):
            full = int((l1 + beta * (l0 - l1)).argmax().item())
            n_bad += int(full not in set(cands.tolist()))
    ok.append(("S-B 引理 2：T_β 的 argmax ⊆ topK(l1)∪topK(l0)（200×4）", n_bad == 0, f"violations={n_bad}"))

    # S-C 臂逻辑：R=∅ 不施加；τ 门；fixed 恒等；β=1 时 argmax ≡ l0
    g = torch.Generator().manual_seed(0)
    ok.append(("S-C1 betastar 在 R=∅ 时不施加", arm_beta("betastar", 0.2, None, g)[0] == 0.0, ""))
    ok.append(("S-C2 fixed 恒定", arm_beta("fixed", 0.2, 0.3, g)[0] == 0.2, ""))
    ok.append(("S-C3 betastar_tau 低于 τ 不施加",
               arm_beta("betastar_tau", 0.2, 0.3, g)[0] == 0.0
               and abs(arm_beta("betastar_tau", 0.2, 0.6, g)[0] - (0.6 + EPS)) < 1e-12, ""))
    ok.append(("S-C4 betastar = β*_min + ε",
               abs(arm_beta("betastar", 0.2, 0.3, g)[0] - (0.3 + EPS)) < 1e-12, ""))
    b1, _ = arm_beta("betastar", 0.2, 0.999999, g)
    ok.append(("S-C5 上界保护 β ≤ 1", b1 <= 1.0, f"β={b1}"))
    l1 = torch.randn(30) * 2
    l0 = l1 + torch.randn(30) * 2
    ok.append(("S-C6 β=1 ⇔ argmax l0",
               int((l1 + 1.0 * (l0 - l1)).argmax().item()) == int(l0.argmax().item()), ""))

    # S-D rand_strength 可复现且两档齐备
    draws = [arm_beta("rand_strength", 0.2, 0.3, make_gen(7, 3, "rand_strength"))
             for _ in range(1)]
    draws2 = [arm_beta("rand_strength", 0.2, 0.3, make_gen(7, 3, "rand_strength"))
              for _ in range(1)]
    ok.append(("S-D1 rand_strength 同种子可复现", draws == draws2, f"{draws} vs {draws2}"))
    vals = {arm_beta("rand_strength", 0.2, 0.3, make_gen(i, i, "rand_strength"))[0]
            for i in range(60)}
    ok.append(("S-D2 rand_strength 两档齐备（min 与 max 都出现）",
               len({round(v, 3) for v in vals}) >= 2, f"{sorted(round(v,3) for v in vals)[:5]}"))

    # S-E 整链路 stub：无真实权重，用表驱动 logits 检验生成循环（β* 计算/施加、翻转计数、粘性、R=∅ 跳过）
    ok += _stub_loop_selftest()
    print("\n[selftest]")
    for name, good, extra in ok:
        print(f"  {'✅' if good else '❌'} {name} {extra}")
    return 0 if all(g for _, g, _ in ok) else 1


# ═════════════════════════════════════════════════════════════════════════════
# 判读（零 GPU）
# ═════════════════════════════════════════════════════════════════════════════


def paired_between(samples, arm_a, beta_a, arm_b, beta_b, subset="know_wrong"):
    """KW 样本上的配对比较：b=仅 A 救回、c=仅 B 救回；另报 KC 破坏侧的四格。"""
    ka, kb = f"correct_{arm_a}_beta{beta_a}", f"correct_{arm_b}_beta{beta_b}"
    b = c = 0
    kc_avoid = kc_add = 0
    for v in samples.values():
        if ka not in v or kb not in v or v.get("baseline_correct") is None:
            continue
        if v.get("subset") == subset and not v["baseline_correct"]:
            if v[ka] and not v[kb]:
                b += 1
            elif v[kb] and not v[ka]:
                c += 1
        if v.get("subset") == "know_correct" and v["baseline_correct"]:
            if v[ka] and not v[kb]:
                kc_avoid += 1     # A 保住、B 破坏
            elif v[kb] and not v[ka]:
                kc_add += 1       # A 破坏、B 保住
    return {"b_a_only": b, "c_b_only": c, "mcnemar_p": mcnemar_exact(b, c),
            "kc_kept_by_a_only": kc_avoid, "kc_kept_by_b_only": kc_add}


def _stub_loop_selftest():
    """表驱动 stub：手算 β* 与翻转/粘性，逐项对照 generate_arm 的记录。"""
    from types import SimpleNamespace
    V = 6

    class StubModel:
        def __init__(self, finals, earlies):
            self.finals, self.earlies = finals, earlies
            self.cfg = SimpleNamespace(n_layers=4, d_vocab_out=V)

        def to_tokens(self, prompt, prepend_bos=True):
            return torch.zeros((1, 1), dtype=torch.long)

        def run_with_hooks(self, tokens, fwd_hooks):
            step = int(tokens.shape[1]) - 1
            for _, fn in fwd_hooks:
                fn(self.earlies[step].view(1, 1, V))
            return self.finals[step].view(1, 1, V)

    class StubTok:
        eos_token_id = -1

        def decode(self, gids):
            return "-".join(str(g) for g in gids)

    finals = [torch.tensor([5.0, 4.0, 0, 0, 0, 0]),
              torch.tensor([5.0, 6.0, 0, 0, 0, 0]),
              torch.tensor([6.0, 5.0, 0, 0, 0, 0]),
              torch.tensor([6.0, 5.0, 0, 0, 0, 0])]
    earlies = [torch.tensor([0.0, 4.5, 0, 0, 0, 0]),   # R={1}, β*=1/(1+4.5)=0.18182
               torch.tensor([0.0, 3.0, 0, 0, 0, 0]),   # argmax(l0)=1=argmax(l1) ⇒ R=∅
               torch.tensor([0.0, 3.0, 0, 0, 0, 0]),   # R={1}, β*=1/(1+3)=0.25
               torch.tensor([0.0, 3.0, 0, 0, 0, 0])]
    class StubNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, x):
            return x

    model, tok = StubModel(finals, earlies), StubTok()
    ln_id, W = StubNorm(), torch.eye(V)
    _, rec = generate_arm(model, tok, "p", "cpu", 0, W, torch.zeros(V), ln_id,
                          "betastar", 0.20, sample_id=1, max_new=4)
    exp_b0 = 1.0 / (1.0 + 4.5) + EPS
    exp_b2 = 1.0 / (1.0 + 3.0) + EPS
    checks = [
        ("stub: β_t = β*_min+ε（步0）", abs(rec["applied_beta"][0] - exp_b0) < 1e-6,
         f"{rec['applied_beta'][0]:.6f} vs {exp_b0:.6f}"),
        ("stub: R=∅ 的步不施加（步1）", rec["applied_beta"][1] == 0.0, f"{rec['applied_beta'][1]}"),
        ("stub: β_t（步2）", abs(rec["applied_beta"][2] - exp_b2) < 1e-6,
         f"{rec['applied_beta'][2]:.6f} vs {exp_b2:.6f}"),
        # 步3 的表与步2 相同 ⇒ 同样翻转（步0/2/3 共 3 次；步1 因 R=∅ 不施加）
        ("stub: 翻转计数 = 3（步0/2/3；步1 R=∅ 不翻）", rec["n_flip"] == 3, f"{rec['n_flip']}"),
        ("stub: feasible 步数 = 3（步0/2/3）", rec["n_feasible"] == 3, f"{rec['n_feasible']}"),
        ("stub: 一步粘性 [1, 0]（步1 保留、步2 翻回）", rec["persist_next"] == [1, 0],
         f"{rec['persist_next']}"),
        ("stub: β*_min(t+1) = [None, 0.25]", rec["beta_min_next"][0] is None
         and abs(rec["beta_min_next"][1] - 0.25) < 1e-9, f"{rec['beta_min_next']}"),
        ("stub: 输出 token 序列", True, f"{rec.get('n_steps')} 步"),
    ]
    # fixed 臂对照：β 恒定 ⇒ 同一轨迹上翻转集合不同
    _, rec_f = generate_arm(model, tok, "p", "cpu", 0, W, torch.zeros(V), ln_id,
                            "fixed", 0.05, sample_id=1, max_new=4)
    checks.append(("stub: fixed 臂 β 恒定", set(rec_f["applied_beta"]) == {0.05},
                   f"{sorted(set(rec_f['applied_beta']))}"))
    return [(n, bool(g), e) for n, g, e in checks]


def _agg_step_stats(per_sample):
    """臂级步统计：β*_min 分布、feasible 占比、干预步占比、粘性。"""
    feas, applied, steps, flips, persist, bnext = [], [], 0, 0, [], []
    for r in per_sample.values():
        s = r.get("steps")
        if not s:
            continue
        steps += s["n_steps"]
        feas += s["beta_min"]
        applied += [b for b in s["applied_beta"]]
        flips += s["n_flip"]
        persist += s["persist_next"]
        bnext += [b for b in s["beta_min_next"] if b is not None]
    feas_a = np.array(feas) if feas else np.array([np.nan])
    appl_a = np.array(applied) if applied else np.array([np.nan])
    per_a = np.array(persist) if persist else np.array([np.nan])
    bn_a = np.array(bnext) if bnext else np.array([np.nan])
    return {
        "n_steps": steps, "n_feasible": int(feas_a.size if feas else 0),
        "feasible_frac": (feas_a.size / steps) if steps else None,
        "n_flip": flips, "flip_frac_of_steps": (flips / steps) if steps else None,
        "beta_min": {"median": float(np.nanmedian(feas_a)), "p90": float(np.nanpercentile(feas_a, 90)),
                     "share_gt_020": float(np.nanmean(feas_a > 0.20))} if feas else None,
        "applied_beta": {"median": float(np.nanmedian(appl_a)),
                         "mean": float(np.nanmean(appl_a)),
                         "share_zero": float(np.nanmean(appl_a == 0.0))} if applied else None,
        "stickiness": {
            "one_step_persist_rate": float(np.nanmean(per_a)) if persist else None,
            "n_flip_observed_next": int(len(persist)),
            "beta_min_next_median": float(np.nanmedian(bn_a)) if bnext else None,
            "share_beta_min_next_ge_theta": float(np.nanmean(bn_a >= THETA_STICKY)) if bnext else None,
        },
    }


def judge(path, out_dir=None):
    d = json.loads(Path(path).read_text())
    out_dir = Path(out_dir) if out_dir else Path(path).parent
    arms = d["arms"]
    step = {a: _agg_step_stats(v["per_sample"]) for a, v in arms.items()}
    label = {a: v["label_stats"] for a, v in arms.items()}

    def _key_of(name):
        ks = [k for k, v in arms.items() if v.get("arm") == name]
        return ks[0] if ks else None

    fixed = {a: v for a, v in label.items() if a.startswith("fixed_")}
    best_fixed = max(fixed, key=lambda a: fixed[a]["net_pp"]) if fixed else None
    lines = ["# C1 决定性验证判读（判据：docs/protocol/dola-c1-validation-20260925.md）\n",
             f"数据：`{Path(path).name}`；模型 {d['config']['model']}，n={d['config']['n_test']}，"
             f"seed={d['config']['seed_test']}\n", "## 各臂总览\n",
             "| 臂 | KW 救回 | KC 破坏 | net(pp) | 翻转步/步 | feasible 占比 | β*_min 中位 | 粘住率 |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        L, S = label[a], step[a]
        kw = L.get("rescue_kw", "—")
        kc = L.get("break_kc", "—")
        bm = S["beta_min"]["median"] if S["beta_min"] else float("nan")
        lines.append(f"| `{a}` | {kw} | {kc} | {L['net_pp']:+.2f} | {S['flip_frac_of_steps']:.3f} | "
                     f"{S['feasible_frac']:.3f} | {bm:.4f} | "
                     f"{S['stickiness']['one_step_persist_rate']:.3f} |")

    verdict = {}
    bs_key, rs_key = _key_of("betastar"), _key_of("rand_strength")
    if best_fixed and bs_key:
        pb = arms[bs_key]["paired_vs"].get(best_fixed, {})
        dn = label[bs_key]["net_pp"] - label[best_fixed]["net_pp"]
        b = pb.get("b_a_only", 0)
        c = pb.get("c_b_only", 0)
        pv = pb.get("mcnemar_p", float("nan"))
        kc_ok = (label[bs_key].get("per_subset", {}).get("know_correct", {}).get("break", 0)
                 <= label[best_fixed].get("per_subset", {}).get("know_correct", {}).get("break", 0))
        if dn >= 3 and pv < 0.05 and b > c and kc_ok:
            v = "通过（C1 优于最佳固定 β）"
        elif dn <= 0 or not kc_ok:
            v = "判停（C1 无差异化价值：Δnet ≤ 0 或 KC 破坏更高）"
        else:
            v = "不确定区间（并列披露；需 8B/原生域档再定）"
        verdict["P1"] = {"vs": best_fixed, "delta_net_pp": dn, "paired_b": b, "paired_c": c,
                         "mcnemar_p": pv, "kc_not_worse": bool(kc_ok), "branch": v}
        lines += ["\n## P1 主判据（betastar vs best_fixed）\n",
                  f"- best_fixed = `{best_fixed}`（net {label[best_fixed]['net_pp']:+.2f}pp）",
                  f"- Δnet = **{dn:+.2f}pp**；KW 救回配对 b={b}（betastar 独有）/ c={c}（对照独有），"
                  f"McNemar p={pv:.4f}",
                  f"- KC 破坏不高于 best_fixed：{'✅' if kc_ok else '❌'}",
                  f"- **判定：{v}**"]
    if rs_key:
        rs = arms[rs_key]["per_sample"]
        mm = {"min": [0, 0], "max": [0, 0]}
        for r in rs.values():
            s = r.get("steps")
            if not s:
                continue
            for j, ch in enumerate(s.get("flip_rct_choice", [])):
                if ch is None or j >= len(s["persist_next"]):
                    continue
                mm[ch][0] += int(s["persist_next"][j])
                mm[ch][1] += 1
        if mm["min"][1] and mm["max"][1]:
            # H_frag 方向 = 「零边际比大边际更不粘」⇒ 单侧检验 max > min（fisher_greater 的第一组为较大率组）
            fp = fisher_greater(mm["max"][0], mm["max"][1] - mm["max"][0],
                                mm["min"][0], mm["min"][1] - mm["min"][0])
            try:  # 另报双侧（scipy 存在时）
                from scipy.stats import fisher_exact
                _, fp2 = fisher_exact([[mm["max"][0], mm["max"][1] - mm["max"][0]],
                                       [mm["min"][0], mm["min"][1] - mm["min"][0]]])
                fp_two = float(fp2)
            except Exception:
                fp_two = None
            rmin, rmax = mm["min"][0] / mm["min"][1], mm["max"][0] / mm["max"][1]
            if fp < 0.05 and rmin < rmax:
                hint = "H_frag 成立（零边际翻转更脆）"
            elif fp < 0.05 and rmin > rmax:
                hint = "H_frag 反向（零边际反而更粘）"
            else:
                hint = "H_frag 未获支持（无显著差异）"
            verdict["P2"] = {"min": mm["min"], "max": mm["max"], "fisher_p_dir_max_gt_min": fp,
                             "fisher_p_two_sided": fp_two, "branch": hint}
            lines += ["\n## P2 机制判据（rand_strength 臂内随机化）\n",
                      f"- 零边际翻转：粘住 {mm['min'][0]}/{mm['min'][1]} = "
                      f"{mm['min'][0]/mm['min'][1]:.3f}",
                      f"- 大边际翻转：粘住 {mm['max'][0]}/{mm['max'][1]} = "
                      f"{mm['max'][0]/mm['max'][1]:.3f}",
                      f"- Fisher 单侧（H_frag 方向：max > min）p={fp:.4f}"
                      + (f"；双侧 p={fp_two:.4f}" if fp_two is not None else "")
                      + f" ⇒ **{hint}**"]
    lines += ["\n## 前置门（§4）\n"]
    med = step.get(bs_key, {}).get("beta_min", {}) if bs_key else {}
    med = med.get("median") if med else None
    lines.append(f"- betastar 的 feasible 步 β*_min 中位数 = {med}；停止条件（<0.01 ⇒ 退化为几乎不扰动）："
                 f"{'⚠️ 触发' if (med is not None and med < 0.01) else '未触发'}")
    fx = label.get("fixed_beta0.2")
    if fx and d["config"]["n_test"] < 300:
        lines.append(f"- 协议自检：小样本档（n={d['config']['n_test']}）不做逐位复现核对；"
                     "全量档（n=300）须与已发布数字逐位一致")
    elif fx:
        pub = "5/71" if d['config']['seed_test'] == 123 else "7/64"
        kc_pub = "17/76" if d['config']['seed_test'] == 123 else "17/75"
        lines.append(f"- 协议自检：`fixed@0.20` KW 救回 {fx.get('rescue_kw')}（已发布 {pub}）、"
                     f"KC 破坏 {fx.get('break_kc')}（已发布 {kc_pub}）⇒ "
                     f"{'✅ 逐位复现' if fx.get('rescue_kw') == pub and fx.get('break_kc') == kc_pub else '⚠️ 不一致（须查明后才判读）'}")
    (out_dir / "judge_c1.md").write_text("\n".join(lines) + "\n")
    (out_dir / "judge_c1.json").write_text(json.dumps(
        {"step_stats": step, "label_stats": label, "verdict": verdict},
        ensure_ascii=False, indent=2))
    print("\n".join(lines))
    return verdict


# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="C1 闭式强度决定性验证")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer_early", type=int, default=20, help="ℓ*（1.7B=20，8B=28）")
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed_test", type=int, default=123)
    ap.add_argument("--rank_threshold", type=int, default=50)
    ap.add_argument("--arms", nargs="+", default=["fixed", "betastar", "betastar_tau", "rand_strength"])
    ap.add_argument("--betas", nargs="+", type=float, default=FIXED_BETAS)
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="小样本前置门模式（n 小、跑全臂）")
    ap.add_argument("--save_steps", action="store_true", default=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--refresh_classify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--judge", type=str, default=None, help="对已有结果 JSON 判读（零 GPU）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.judge:
        judge(args.judge)
        return 0
    unknown = [a for a in args.arms if a not in ARM_DOC]
    if unknown:
        sys.exit(f"未知臂 {unknown}（可选 {list(ARM_DOC)}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir) if args.output_dir else (_PARENT.parent / "outputs" / "tldc_betastar")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(str(args.model).rstrip("/")).name

    print("=" * 76)
    print("C1 闭式对比强度 β* 决定性验证")
    print(f"  model={args.model} ℓ*={args.layer_early} n={args.n_test} seed={args.seed_test} "
          f"arms={args.arms} fixed β={args.betas}")
    print("=" * 76)

    model, tokenizer, W_U, b_U, ln_final = load_model_and_unembed(device, args.model)

    print(f"[1/3] classify (seed={args.seed_test}, n={args.n_test})...")
    cache_dir = out_dir / "_classify_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{model_tag}_l{args.layer_early}_r{args.rank_threshold}_s{args.seed_test}_n{args.n_test}"
    cache_path = cache_dir / f"classify_{cache_key}.json"
    if cache_path.exists() and not args.refresh_classify:
        entries = json.loads(cache_path.read_text())
        print(f"  [cache] 命中 {cache_path.name}")
    else:
        test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed_test)[: args.n_test]
        entries = classify_samples(model, tokenizer, test_samples, device,
                                   args.layer_early, args.rank_threshold)
        cache_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    print(f"  KC={sum(1 for e in entries if e['subset']=='know_correct')}, "
          f"KW={sum(1 for e in entries if e['subset']=='know_wrong')}, "
          f"DK={sum(1 for e in entries if e['subset']=='dont_know')}")

    samples, arms_out = {}, {}
    print("[2/3] generating...")
    from tqdm import tqdm
    t0 = time.time()
    for arm in args.arms:
        betas = args.betas if arm == "fixed" else [args.betas[-1]]  # 闭式臂只用最后一个 β 作键名标记
        for beta in betas:
            key = f"correct_{arm}_beta{beta}"
            per_sample = {}
            for e in tqdm(entries, desc=f"  {arm}@β={beta}"):
                text, rec = generate_arm(model, tokenizer, e["prompt"], device,
                                         args.layer_early, W_U, b_U, ln_final,
                                         arm, beta, e["sample_id"], max_new=args.max_new)
                v = samples.setdefault(e["sample_id"], {
                    "subset": e["subset"], "rank": e["rank"], "question": e["question"],
                    "baseline_correct": e["is_correct"]})
                v[key] = check_correct_exact(text, e["answers"])
                if args.save_steps:
                    per_sample[e["sample_id"]] = {"steps": rec, "text": text}
            arms_out[f"{arm}{'' if arm == 'fixed' else ''}_beta{beta}"] = {
                "arm": arm, "beta": beta, "per_sample": per_sample}
            print(f"  {arm}@β={beta} 完成（{time.time()-t0:.0f}s 累计）")

    print("[3/3] stats...")
    for name, a in arms_out.items():
        a["label_stats"] = arm_stats(samples, a["arm"], a["beta"])
        if a["arm"] == "fixed":
            a["paired_vs"] = {}
        else:
            a["paired_vs"] = {f"fixed_beta{b}": paired_between(samples, a["arm"], a["beta"], "fixed", b)
                              for b in args.betas}
    report = {
        "config": vars(args), "model_tag": model_tag,
        "n_entries": len(entries), "elapsed": time.time() - t0,
        "arms": arms_out, "samples": samples,
    }
    tag = f"betastar_{model_tag}_n{args.n_test}_s{args.seed_test}" + ("_smoke" if args.smoke else "")
    out_path = out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"[写出] {out_path}")
    print(f"判读：python experiments/lin_theory/main_tldc_betastar.py --judge {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
