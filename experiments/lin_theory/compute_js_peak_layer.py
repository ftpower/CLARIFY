#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compute_js_peak_layer.py — 计算层间 JS 散度峰值层 ℓ_JS（label-free）。

用途：为「层间 logit 插值类干预（TLDC）」的**对比层选择对照**提供层号——
现用对比层为固定层距（1.7B L20 / 8B L28 = 末层−7），需要一个"由 JS 散度选出的层"
作为方法对照臂（可视为 DoLa 式选层规则的静态版本）。

定义：ℓ_JS = argmax_ℓ  mean_i JS( p_ℓ^i ‖ p_final^i )
  p_ℓ^i：第 i 个样本在**问题末尾 token**处对第 ℓ 层 hook_resid_post 做 logit lens 的下一 token 分布
  p_final：同约定下最后一层的分布

⚠️ 支撑约定（必须与 `detect_js_lr_cv.py` 一致，否则退化）：
  逐层分布与末层分布在**全词表支撑**上几乎不重叠，直接算 JS 会在几乎所有权层饱和到
  ln2（实测 1.7B：L0–L19 全部 0.6931 = ln2，判读无意义）。因此按项目既有约定，
  JS 在**两层 top-10 的并集支撑**上计算，各自重归一化后再算；支撑质量下溢的层对跳过。

两种口径都报（稳健性）：
  A. raw      —— lens = h @ W_U + b_U（不套 ln_final），与 `detect_js_lr_cv.py` 一致（主口径）
  B. ln_final —— lens = ln_final(h) @ W_U + b_U（套最终 LayerNorm）

⚠️ 纪律：**label-free**——只用隐藏状态，不读 y、不按正确与否分层（用标签选层即污染）。

输入：`detect_lr_probe_cv.py --dump` 的逐层隐藏状态 npz（h_L0..h_L{n-1}）
输出：JSON（两种口径的逐层 mean/median JS、峰值层、指定层取值、逐样本 argmax 层直方图）

用法（本地 1.7B，CPU 可跑，约 3–5 分钟）：
    python experiments/lin_theory/compute_js_peak_layer.py \
        --npz experiments/outputs/lin_theory/detect_lr_probe_hidden.npz \
        --model Qwen/Qwen3-1.7B \
        --out experiments/outputs/lin_theory/js_peak_layer.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from common import load_model_and_unembed  # noqa: E402

EPS = 1e-12
TOP_K = 10


def _rel(p):
    """相对仓库路径（绝对/相对输入都能处理，失败则原样返回）。"""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def js_pair(p, q, ids):
    """在给定支撑 ids 上重归一化后计算 JS 散度（nats）；质量下溢返回 None。"""
    a, b = p[ids], q[ids]
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= EPS or sb <= EPS:
        return None
    a, b = a / sa, b / sb
    m = 0.5 * (a + b)
    return 0.5 * (float(np.sum(a * np.log((a + EPS) / (m + EPS))))
                  + float(np.sum(b * np.log((b + EPS) / (m + EPS)))))


def curve_for(probs, topk, n_layers):
    """逐层 JS（对末层），返回长度 n_layers 的数组（无效层为 NaN，末层为 0）。"""
    p_final, final_top = probs[-1], topk[-1]
    vals = []
    for l in range(n_layers):
        if l == n_layers - 1:
            vals.append(0.0)
            continue
        v = js_pair(probs[l], p_final, np.union1d(topk[l], final_top))
        vals.append(np.nan if v is None else v)
    return np.array(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="层间 JS 散度峰值层（label-free）")
    ap.add_argument("--npz", required=True, help="逐层隐藏状态 npz（h_L0..h_L{n-1}）")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers_of_interest", type=str, default="",
                    help="逗号分隔层号，额外报告这些层的 mean JS（如 18,20,23,26）")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    layer_keys = sorted((k for k in z.files if k.startswith("h_L")),
                        key=lambda k: int(k[3:]))
    n_layers = len(layer_keys)
    n_samples = z[layer_keys[0]].shape[0]
    d_model = z[layer_keys[0]].shape[1]
    print(f"npz: {args.npz} | 层数 {n_layers} | 样本 {n_samples} | d_model {d_model}")
    # ⚠️ 不读取 z['y']：选层必须 label-free

    model, _tok, W_U, b_U, ln_final = load_model_and_unembed(
        device=args.device, model_id=args.model)
    W = W_U.detach().float().cpu()                    # [d_model, vocab]
    b = None if b_U is None else b_U.detach().float().cpu()
    print(f"model: {args.model} | W_U {tuple(W.shape)} | 支撑 = top-{TOP_K} 并集（重归一化）")

    H = np.stack([z[k] for k in layer_keys], axis=1)   # (n, L, d)
    js_raw = np.full((n_samples, n_layers), np.nan)
    js_ln = np.full((n_samples, n_layers), np.nan)

    with torch.no_grad():
        for i in range(n_samples):
            h = torch.from_numpy(H[i].astype(np.float32))          # (L, d)

            def _probs(hh):
                lg = hh @ W
                if b is not None:
                    lg = lg + b
                p = torch.softmax(lg, dim=-1).numpy()
                tk = [np.argpartition(p[l], -TOP_K)[-TOP_K:] for l in range(n_layers)]
                return p, tk

            p_raw, tk_raw = _probs(h)
            p_ln, tk_ln = _probs(ln_final(h.to(args.device)).float().cpu())
            js_raw[i] = curve_for(p_raw, tk_raw, n_layers)
            js_ln[i] = curve_for(p_ln, tk_ln, n_layers)
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{n_samples}")

    interest = [int(x) for x in args.layers_of_interest.split(",") if x.strip()]
    out = {
        "npz": _rel(args.npz),
        "model": args.model,
        "n_samples": int(n_samples),
        "n_layers": int(n_layers),
        "support": f"union of top-{TOP_K} tokens of the two layers, renormalized",
        "conventions": {
            "raw": "logit lens = h @ W_U + b_U (no ln_final) — 与 detect_js_lr_cv.py 一致",
            "ln_final": "logit lens = ln_final(h) @ W_U + b_U",
        },
        "label_free": True,
    }
    for name, arr in (("raw", js_raw), ("ln_final", js_ln)):
        mean = np.nanmean(arr, axis=0)
        med = np.nanmedian(arr, axis=0)
        peak = int(np.nanargmax(mean))
        hist = np.bincount(np.nanargmax(np.nan_to_num(arr, nan=-1.0), axis=1),
                           minlength=n_layers)
        out[name] = {
            "mean_js_per_layer": [None if np.isnan(v) else float(v) for v in mean],
            "median_js_per_layer": [None if np.isnan(v) else float(v) for v in med],
            "peak_layer": peak,
            "peak_mean_js": float(mean[peak]),
            "js_at_layers": {int(l): (None if np.isnan(mean[l]) else float(mean[l]))
                             for l in interest},
            "per_sample_argmax_hist": {int(l): int(c) for l, c in enumerate(hist) if c > 0},
        }
        print(f"\n[{name}] 逐层 mean JS（top-{TOP_K} 并集支撑，nat）:")
        for l in range(n_layers):
            v = mean[l]
            tag = "  ← 峰值" if l == peak else ("  *" if l in interest else "")
            print(f"  L{l:<3d} {'nan' if np.isnan(v) else f'{v:.4f}'}{tag}")
        print(f"[{name}] ℓ_JS = L{peak} (mean JS {mean[peak]:.4f})")
        top = sorted([(int(l), int(c)) for l, c in enumerate(hist)], key=lambda kv: -kv[1])[:5]
        print(f"[{name}] 逐样本 argmax 层（前 5）: {top}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已保存：{_rel(args.out)}")


if __name__ == "__main__":
    main()
