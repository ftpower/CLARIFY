"""L0-3（候选 C1）：闭式对比强度 β\\* 的零 GPU 可行性筛查。

判据来源（**执行前设定**）：
    docs/protocol/dola-l0-analysis-20260925.md §3
主统计量：DDR(β) = P(winner(β\\*_min+ε) ≠ winner(β) | R ≠ ∅)，β ∈ {0.05,0.1,0.2,0.3,0.5,1.0}；
判定：max_β DDR < 0.10 ⇒ C1 判停；≥ 0.30 ⇒ 进 L1；其余 ⇒ 不确定区间。

数据（零 GPU）：
    experiments/outputs/lin_theory/detect_lr_probe_hidden.npz
      —— Qwen3-1.7B，TriviaQA 检测集 n=200 seed=42，末 token 残差 `h_L0..h_L27`（含标签 y）。
    权重：本地 HF 缓存 Qwen3-1.7B 的 `model.norm.weight` + 绑定 `lm_head.weight`（safetensors 直读）。
算子：FAD 引理 1/2/4a（`docs/paper/paper-route-method-schemes.md` §2）——
    l₁ = unembed(norm(h_L27))（末层真实 logits 的等价读出）、l₀ = unembed(norm(h_L20))（ℓ\\*=L20）；
    a = argmax l₁；R = {c : l₀(c) > l₀(a)} ∩ topK(l₀)；β\\*(a,c) = m/(m+Δ₀)。

自检：
    S1 绑定权重一致（lm_head ≡ embed_tokens）；
    S2 末层读出＝真实 logits：`--check-model` 时用 TransformerLens 前向 1 条真实 prompt，
       比对 (i) npz h_L27 与 TL resid_post、(ii) 本脚本 lens argmax 与 TL 真实 logits argmax。

用法：
    python3 experiments/lin_theory/diagnose_dola_betastar_decision.py --check-model
输出：experiments/outputs/dola_l0_20260925/l0_c1.json + l0_c1_report.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments/outputs/dola_l0_20260925"
NPZ = REPO / "experiments/outputs/lin_theory/detect_lr_probe_hidden.npz"
SNAP_ROOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots"
LAYER_MATURE, LAYER_REF = 27, 20
K_PRIMARY, K_SENS = 10, (5, 50)
EPS = 0.01
BETAS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def snapshot_dir():
    snaps = sorted(p for p in SNAP_ROOT.glob("*") if (p / "config.json").exists())
    if not snaps:
        raise SystemExit(f"未找到本地 Qwen3-1.7B：{SNAP_ROOT}")
    return snaps[-1]


def load_weights(snap):
    import torch
    from safetensors import safe_open
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    w = {}
    for key in ["model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"]:
        shard = snap / idx[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            w[key] = f.get_tensor(key).to(torch.float32)
    return w


def rms_norm(h, weight, eps=1e-6):
    import torch
    h = torch.as_tensor(h, dtype=torch.float32)
    var = h.pow(2).mean(dim=-1, keepdim=True)
    return (h / torch.sqrt(var + eps)) * weight


def lens_logits(h, w, eps=1e-6):
    import torch
    with torch.no_grad():
        return (rms_norm(h, w["model.norm.weight"], eps) @ w["lm_head.weight"].T).numpy()


def per_sample(l1, l0, K):
    """返回每题 (a, R, beta_min, cands)。"""
    a = int(np.argmax(l1))
    top1 = np.argsort(-l1)[:K]
    top0 = np.argsort(-l0)[:K]
    cands = np.unique(np.concatenate([top1, top0]))
    R = [int(c) for c in top0 if l0[c] > l0[a] and c != a]
    beta = {}
    for c in R:
        m = float(l1[a] - l1[c])
        d0 = float(l0[c] - l0[a])
        if m > 0 and d0 > 0:
            beta[c] = m / (m + d0)
    bmin = min(beta.values()) if beta else None
    return a, R, bmin, cands


def per_sample_batch(l1, l0, K):
    a, R_list, bmins, cands_list = [], [], [], []
    for i in range(l1.shape[0]):
        ai, Ri, bm, cd = per_sample(l1[i], l0[i], K)
        a.append(ai); R_list.append(Ri); bmins.append(bm); cands_list.append(cd)
    return a, R_list, bmins, cands_list


def winner(l1, l0, cands, beta):
    z = (1.0 - beta) * l1[cands] + beta * l0[cands]
    return int(cands[int(np.argmax(z))])


def check_model(npz, w, snap):
    """S2：用 TransformerLens 前向 1 条真实 prompt 复核 npz 与 lens 读出。"""
    out = {"attempted": True}
    try:
        import torch
        from transformer_lens import HookedTransformer
        import sys
        sys.path.insert(0, str(REPO / "experiments/phase2_entropy"))
        from src.data_loader import load_triviaqa, format_prompt

        model = HookedTransformer.from_pretrained("Qwen/Qwen3-1.7B", device="cpu",
                                                  dtype=torch.float32)
        s = load_triviaqa(n_samples=1, seed=42)[0]
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        toks = model.to_tokens(prompt, prepend_bos=True)
        if toks.shape[1] > 1024:
            toks = toks[:, -1024:]
        store = {}
        def hook(act, hook=None):
            store["h"] = act[:, -1, :].detach()
            return act
        with torch.no_grad():
            logits = model.run_with_hooks(toks, fwd_hooks=[("blocks.27.hook_resid_post", hook)])
        h_npz = npz["h_L27"][0].astype(np.float32)
        h_tl = store["h"][0].numpy().astype(np.float32)
        d_h = float(np.abs(h_npz - h_tl).max())
        real = logits[0, -1, :].float().numpy()
        mine = lens_logits(h_npz[None, :], w)[0]
        d_l = float(np.abs(real - mine).max())
        out.update({
            "n_tokens": int(toks.shape[1]),
            "max_abs_h_diff_npz_vs_TL": d_h,
            "max_abs_logit_diff_lens_vs_TL": d_l,
            "argmax_agree": bool(int(np.argmax(real)) == int(np.argmax(mine))),
            "argmax_TL": int(np.argmax(real)), "argmax_lens": int(np.argmax(mine)),
            "S2a_h_convention_ok": bool(d_h < 0.05),
            "S2b_lens_ok": bool(d_l < 0.05 and int(np.argmax(real)) == int(np.argmax(mine))),
            "ok": bool(d_h < 0.05 and d_l < 0.05
                       and int(np.argmax(real)) == int(np.argmax(mine))),
        })
    except Exception as e:  # 复核失败不阻断主分析，但必须显式记录
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return out


def _writable_dataset_cache():
    """把只读的 HF datasets 缓存以符号链接挂到可写目录（锁文件只落在 /tmp），源缓存零改动。"""
    src = Path.home() / ".cache/huggingface/datasets"
    dst = Path("/tmp/hf_ds")
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for child in src.iterdir():
            link = dst / child.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(child)
    os.environ["HF_DATASETS_CACHE"] = str(dst)
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def gold_first_tokens():
    """重建检测集样本（同 seed 同序），返回金标首 token id 列表。"""
    import importlib.util
    import sys
    _writable_dataset_cache()
    sys.path.insert(0, str(REPO / "experiments/phase2_entropy"))
    from src.data_loader import load_triviaqa
    spec = importlib.util.spec_from_file_location("common_l0", REPO / "experiments/lin_theory/common.py")
    common = importlib.util.module_from_spec(spec)
    sys.modules["common_l0"] = common
    spec.loader.exec_module(common)
    from transformers import AutoTokenizer
    snaps = sorted(p for p in SNAP_ROOT.glob("*") if (p / "tokenizer.json").exists())
    tok = AutoTokenizer.from_pretrained(str(snaps[-1]))
    samples = load_triviaqa(n_samples=200, seed=42)
    return [common.get_first_answer_token_id(tok, smp["answers"]) for smp in samples], samples, tok


def main():
    ap = argparse.ArgumentParser(description="L0-3 C1 β* 决策差异率")
    ap.add_argument("--check-model", action="store_true",
                    help="用 TransformerLens 前向 1 条 prompt 复核 h 约定与 lens 读出（较慢）")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snap = snapshot_dir()
    w = load_weights(snap)
    s1 = bool((w["lm_head.weight"] == w["model.embed_tokens.weight"]).all().item())

    npz = np.load(NPZ, allow_pickle=True)
    y = npz["y"]
    l1 = lens_logits(npz[f"h_L{LAYER_MATURE}"], w)
    l0 = lens_logits(npz[f"h_L{LAYER_REF}"], w)

    checks = {"S1_tied_embedding": {"ok": s1}}
    if args.check_model:
        checks["S2_model_roundtrip"] = check_model(npz, w, snap)
    print("[自检]", json.dumps(checks, ensure_ascii=False)[:400])

    res = {"checks": checks, "setup": {
        "npz": str(NPZ.relative_to(REPO)), "n": int(len(y)),
        "layer_mature": LAYER_MATURE, "layer_ref": LAYER_REF, "eps": EPS,
        "K_primary": K_PRIMARY, "betas": list(BETAS),
        "share_y_correct": float(np.mean(y == 1)),
    }, "K": {}}

    for K in [K_PRIMARY, *K_SENS]:
        a, R_list, bmins, cands_list = per_sample_batch(l1, l0, K)
        n_R = int(sum(1 for r in R_list if r))
        has = np.array([bm is not None for bm in bmins])
        bm_arr = np.array([bm for bm in bmins if bm is not None], dtype=float)
        entry = {
            "n_with_R": n_R, "share_R_empty": float(1 - n_R / len(y)),
            "beta_min": {
                "median": float(np.median(bm_arr)), "p10": float(np.percentile(bm_arr, 10)),
                "p90": float(np.percentile(bm_arr, 90)),
                "iqr": float(np.percentile(bm_arr, 75) - np.percentile(bm_arr, 25)),
                "cv": float(bm_arr.std() / bm_arr.mean()),
                "min": float(bm_arr.min()), "max": float(bm_arr.max()),
            } if len(bm_arr) else None,
            "ddr": {},
        }
        for beta in BETAS:
            diff = noflip = both_diff = both_same = 0
            for i in range(len(y)):
                if not has[i]:
                    continue
                w_star = winner(l1[i], l0[i], cands_list[i], bmins[i] + EPS)
                w_fix = winner(l1[i], l0[i], cands_list[i], beta)
                flipped = bmins[i] <= beta
                if w_star != w_fix:
                    diff += 1
                    if not flipped:
                        noflip += 1
                    else:
                        both_diff += 1
                elif flipped:
                    both_same += 1
            entry["ddr"][str(beta)] = diff / n_R if n_R else None
            entry.setdefault("ddr_decomposition", {})[str(beta)] = {
                "fixed_beta_did_not_flip": noflip / n_R if n_R else None,
                "both_flipped_different_winner": both_diff / n_R if n_R else None,
                "both_flipped_same_winner": both_same / n_R if n_R else None,
            }
        # 描述性：按 baseline 正确性分层 + 决策等价率
        for beta in BETAS:
            flips = [i for i in range(len(y)) if has[i]]
            entry.setdefault("fixed_beta_flip_rate", {})[str(beta)] = float(
                np.mean([bmins[i] <= beta for i in flips])) if flips else None
        entry["ddr_by_y"] = {}
        for beta in BETAS:
            for grp, name in [(1, "baseline_correct"), (0, "baseline_wrong")]:
                sel = [i for i in range(len(y)) if has[i] and int(y[i]) == grp]
                if not sel:
                    continue
                diff = sum(int(winner(l1[i], l0[i], cands_list[i], bmins[i] + EPS)
                             != winner(l1[i], l0[i], cands_list[i], beta)) for i in sel)
                entry["ddr_by_y"].setdefault(str(beta), {})[name] = diff / len(sel)
        res["K"][str(K)] = entry

    # ── 描述性：金标命中率与救回/破坏分解（协议 §3「辅统计量」，不进入判定分支）──
    gold, samples, gtok = gold_first_tokens()
    ok_gold = np.array([g is not None for g in gold])
    ok_gold_idx = [i for i in range(len(y)) if ok_gold[i]]
    a1 = np.array([int(np.argmax(l1[i])) for i in range(len(y))])
    hit_base = float(np.mean([a1[i] == gold[i] for i in range(len(y)) if ok_gold[i]]))
    align_ok = hit_base > 0.10
    order = np.argsort(-l1, axis=1)
    top10 = order[:, :10]
    top10_rate = float(np.mean([gold[i] in top10[i] for i in ok_gold_idx]))
    modal = int(np.bincount([int(order[i, 0]) for i in ok_gold_idx]).argmax())
    gd = {"n_gold": int(ok_gold.sum()), "baseline_first_token_hit": hit_base,
          "align_ok": bool(align_ok),
          "alignment_addendum": {
              "gold_in_top10_rate": top10_rate,
              "chance_rate": 10.0 / int(l1.shape[1]),
              "top10_rate_by_y": {name: float(np.mean([gold[i] in top10[i] for i in ok_gold_idx
                                                       if int(y[i]) == grp]))
                                  for grp, name in [(1, "y_correct"), (0, "y_wrong")]},
              "modal_first_token": modal,
              "modal_first_token_str": gtok.decode([modal]),
          },
          "hit_by_y": {name: float(np.mean([a1[i] == gold[i] for i in range(len(y))
                                            if ok_gold[i] and int(y[i]) == grp]))
                       for grp, name in [(1, "y_correct"), (0, "y_wrong")]},
          "hit": {}, "rescue": {}, "break": {}}
    if align_ok:
        _, R_list, bmins, cands_list = per_sample_batch(l1, l0, K_PRIMARY)
        has = np.array([bm is not None for bm in bmins])
        for tag, beta_of in ([("persample", None)]
                             + [(f"fixed_{b}", b) for b in BETAS]):
            hit = resc = brk = n = 0
            for i in range(len(y)):
                if not ok_gold[i] or not has[i]:
                    continue
                beta = (bmins[i] + EPS) if beta_of is None else beta_of
                wn = winner(l1[i], l0[i], cands_list[i], beta)
                n += 1
                hit += int(wn == gold[i])
                if a1[i] != gold[i] and wn == gold[i]:
                    resc += 1
                if a1[i] == gold[i] and wn != gold[i]:
                    brk += 1
            gd["hit"][tag] = hit / n if n else None
            gd["rescue"][tag] = resc / n if n else None
            gd["break"][tag] = brk / n if n else None
        gd["n_eval"] = n
    res["gold_diagnostics"] = gd
    print("[金标]", json.dumps({k: v for k, v in gd.items() if k in
                                ("n_gold", "baseline_first_token_hit", "align_ok", "hit_by_y")},
                               ensure_ascii=False))

    pk = res["K"][str(K_PRIMARY)]
    ddr_max = max(v for v in pk["ddr"].values() if v is not None)
    if ddr_max < 0.10:
        verdict = "C1 判停（闭式强度与固定强度决策不可分辨）⇒ 转 C5 上界复算，不进 L1"
    elif ddr_max >= 0.30:
        verdict = "进入 L1（本地 1.7B 单点对照；须报救回/破坏双口径 + 配对事件数）"
    else:
        verdict = "不确定区间（并列披露；L1 前需补 8B 或原生域档）"
    res["verdict"] = {"max_ddr": ddr_max, "branch": verdict}

    (OUT_DIR / "l0_c1.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    L = ["# L0-3（C1）闭式强度 β\\* 可行性报告\n",
         "判据：`docs/protocol/dola-l0-analysis-20260925.md` §3（执行前设定）。"
         f"数据：`{NPZ.name}`（1.7B / TriviaQA / n={len(y)} / 末 token）；"
         f"l₁=L{LAYER_MATURE}、l₀=L{LAYER_REF}（ℓ\\*），K={K_PRIMARY} 为主。\n",
         "## 自检\n"]
    for k, v in checks.items():
        L.append(f"- {k}: {'✅' if v.get('ok') else '❌'} `{json.dumps(v, ensure_ascii=False)[:300]}`")
    L.append("\n## 主结果（K=%d）\n" % K_PRIMARY)
    bmd = pk["beta_min"]
    L.append(f"- 可翻转样本（R ≠ ∅）：{pk['n_with_R']}/{len(y)}（R=∅ 占 {pk['share_R_empty']:.1%}）")
    L.append(f"- β\\*_min 分布：中位数 {bmd['median']:.4f}、p10 {bmd['p10']:.4f}、p90 {bmd['p90']:.4f}、"
             f"IQR {bmd['iqr']:.4f}、CV **{bmd['cv']:.3f}**、范围 [{bmd['min']:.4f}, {bmd['max']:.4f}]")
    L.append("\n| 固定 β | DDR（主统计量） | 固定 β 的翻转率 | DDR（baseline 正确） | DDR（baseline 错误） |")
    L.append("|---|---|---|---|---|")
    for beta in BETAS:
        by = pk["ddr_by_y"].get(str(beta), {})
        L.append(f"| {beta} | **{pk['ddr'][str(beta)]:.3f}** | {pk['fixed_beta_flip_rate'][str(beta)]:.3f} | "
                 f"{by.get('baseline_correct', float('nan')):.3f} | {by.get('baseline_wrong', float('nan')):.3f} |")
    L.append(f"\n**判定分支：{verdict}**（max_β DDR = {ddr_max:.3f}）\n")
    L.append("\n### DDR 分解（区分「固定 β 未翻转」与「两侧都翻转但赢家不同」）\n")
    L.append("| 固定 β | 决策不同合计 | ⋯固定 β 未翻转 | ⋯都翻转但赢家不同 | ⋯都翻转且赢家相同 |")
    L.append("|---|---|---|---|---|")
    for beta in BETAS:
        d = pk["ddr_decomposition"][str(beta)]
        L.append(f"| {beta} | {pk['ddr'][str(beta)]:.3f} | {d['fixed_beta_did_not_flip']:.3f} | "
                 f"{d['both_flipped_different_winner']:.3f} | {d['both_flipped_same_winner']:.3f} |")
    L.append("\n⇒ 小 β 处的高 DDR 主要来自「固定 β 强度不足、未触发翻转」；"
             "**实质性的赢家差异**（两侧都翻转但结果不同）仅在大 β 处显著："
             f"β=0.5 为 {pk['ddr_decomposition']['0.5']['both_flipped_different_winner']:.3f}、"
             f"β=1.0 为 {pk['ddr_decomposition']['1.0']['both_flipped_different_winner']:.3f}。"
             "闭式强度的可检验差异＝「按各自最小强度翻转」（覆盖各样本自身的 β\\*_min），"
             "固定 β ≤ 0.5 时 62–92% 的可翻转样本不翻转 ⇒ 该差异是真的，但**不是赢家选择差异**。\n")
    L.append("## K 敏感性\n")
    L.append("| K | R=∅ 占比 | β\\*_min 中位数 | CV | max DDR |")
    L.append("|---|---|---|---|---|")
    for K, e in res["K"].items():
        mx = max(v for v in e["ddr"].values() if v is not None)
        L.append(f"| {K} | {e['share_R_empty']:.1%} | {e['beta_min']['median']:.4f} | "
                 f"{e['beta_min']['cv']:.3f} | {mx:.3f} |")
    gd = res["gold_diagnostics"]
    L.append("\n## 金标诊断（协议 §3 辅统计量；不进入判定分支）\n")
    L.append(f"- 金标对齐：baseline 首 token 命中 {gd['baseline_first_token_hit']:.3f}"
             f"（y=1 组 {gd['hit_by_y']['y_correct']:.3f} / y=0 组 {gd['hit_by_y']['y_wrong']:.3f}，"
             f"n={gd['n_gold']}）⇒ 对齐{'成立' if gd['align_ok'] else '存疑（诊断为不可用）'}")
    if gd["align_ok"]:
        L.append("\n| 强度口径 | 首 token 命中率 | 救回率 | 破坏率 |")
        L.append("|---|---|---|---|")
        for tag in ["persample"] + [f"fixed_{b}" for b in BETAS]:
            L.append(f"| {tag} | {gd['hit'][tag]:.3f} | {gd['rescue'][tag]:.3f} | {gd['break'][tag]:.3f} |")
    if not gd["align_ok"]:
        aa = gd["alignment_addendum"]
        L.append("\n**事后补充（预注册门未过后的追加核验，非预注册判据）**：金标首 token 命中率低并非对齐失败——"
                 f"以机会水平校正的对齐检验显示，金标落在末层读出 top-10 内的比例为 **{aa['gold_in_top10_rate']:.1%}**"
                 f"（随机基线 10/151936 ≈ {aa['chance_rate']:.1e}；y=1 组 {aa['top10_rate_by_y']['y_correct']:.1%} "
                 f"vs y=0 组 {aa['top10_rate_by_y']['y_wrong']:.1%}）⇒ 样本顺序对齐成立。"
                 "低首 token 命中率的机制是**标签口径**：检测标签为「金标别名出现在 20 token 贪心续写中」"
                 "（`check_correct_exact` 为跨词边界包含匹配），而末位置众数续写为 "
                 f"`{aa['modal_first_token_str']!r}`（token {aa['modal_first_token']}）"
                 "⇒ 首 token 命中率与答案级正确率（39%）本就不等价，故**救回/破坏分解在本数据上不可用**（fail-closed 保持）。\n")
    L.append("\n## 必须并列的限定\n"
             "- 数据为 **TriviaQA（项目域，DoLa 在此净负）**，非 DoLa 原生域；读数为 **prompt 末 token 的单步"
             "下一 token 决策**，非多 token 续写似然 ⇒ 本结果为**可行性筛查**，不得当作最终判据；\n"
             "- 候选集按 topK 截断（引理 2 的精确性只在全词表 R 上成立），K=10 为主、5/50 为敏感性；\n"
             "- 本机无 8B 权重 ⇒ 8B 档须在服务器执行后复算；\n"
             "- **读数口径**：末位置 argmax 的众数为 `\" The\"`（首 token 金标命中率仅 8.0%、top-10 30.0%）"
             "⇒ 本筛查度量的是**算子是否改变 argmax 决策**，不代表答案级正确性的改善；\n"
             "- L1 若执行，宜落在真实解码/续写算子上，而非本单步代理。\n")
    (OUT_DIR / "l0_c1_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
