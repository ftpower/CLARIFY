"""Phase 23: LoRA δ-Corrective + TLDC 组合干预 — 1.7B 验证脚本。

Theory:  docs/theory-intervention-failure.md §7
Plan:    docs/phase23-lora-tldc-combined.md

4 策略 × β sweep（双层干预）:
  Baseline   = 原始模型, greedy
  TLDC       = 原始模型, TLDC 逐 token 解码
  LoRA       = LoRA checkpoint, greedy
  LoRA+TLDC  = LoRA checkpoint, TLDC 逐 token 解码

阶段 A 诊断:
  A1: LoRA 上参考层 ℓ* 重定位（AUROC 扫描，分避让/覆盖两分支）
  A2: δ 分布对比（baseline vs LoRA，验证 H3——LoRA 压缩 override）
  A3: KW 集合变化（哪些 baseline-KW 被 LoRA 修正）

关键设计:
  - 分类固定: KC/KW/DK 用 baseline 原始模型定义, 4 策略共享 (T4)
  - 串行加载: 1.7B 两个模型不可同时驻留 8GB, 先原始后 LoRA (显存约束)
  - fp16 early-exit: lm_head(norm(h_ref)), 避免 OOM
  - 参考层避让: LoRA target = L20-L27, 参考层分"避让"(L16/L18) 与"覆盖"(L20/L22/L24) (T6)

Usage:
  # 1) 固定分类 (原始模型, ~5min on 5060)
  python eval_lora_tldc.py --mode classify --n_test 100 --seed 123

  # 2) 诊断 (LoRA 模型)
  python eval_lora_tldc.py --mode diagnose --lora_dir .../s20_1_lambda0.005/epoch_1

  # 3) 4 策略 × β 干预对比
  python eval_lora_tldc.py --mode intervene --lora_dir .../s20_1_lambda0.005/epoch_1 \
      --betas 0.03 0.05 0.08 0.10 0.15 0.20
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_offline = os.environ.get("HF_ALLOW_ONLINE", "") != "1"
if _offline:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Path setup ────────────────────────────────────────────────────────────────
_SYS_PARENT = Path(__file__).parent.parent  # experiments/
for _p in [
    str(_SYS_PARENT),
    str(_SYS_PARENT / "phase2_entropy"),
    str(_SYS_PARENT / "phase4_generalization"),
    str(_SYS_PARENT / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct

# 只读复用 lin_theory 既有模块（不改动源码）
from lin_theory.train_lora_delta import (
    MODEL_PATH,
    RANK_THRESHOLD,
    _compute_auroc,
    get_first_answer_token_id,
)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "phase23"
DEFAULT_LORA_DIR = (
    Path(__file__).parent.parent
    / "outputs"
    / "lin_theory"
    / "s20_1_lambda0.005"
    / "epoch_1"
)
DEFAULT_REFS = [16, 18, 20, 22, 24]  # 1.7B: 避让 L16/L18, 覆盖 L20/L22/L24
DEFAULT_BETAS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]


# ═════════════════════════════════════════════════════════════════════════════
# Model loading / module navigation
# ═════════════════════════════════════════════════════════════════════════════


def _get_modules(model):
    """Return (layers, norm, lm_head), navigating PeftModel wrapper."""
    try:
        return (
            model.base_model.model.model.layers,
            model.base_model.model.model.norm,
            model.base_model.model.lm_head,
        )
    except AttributeError:
        try:
            return (
                model.model.model.layers,
                model.model.model.norm,
                model.model.lm_head,
            )
        except AttributeError:
            return model.model.layers, model.model.norm, model.lm_head


def load_model(device: str, model_path: str, lora_dir: str | None = None):
    """Load base (or base + LoRA adapter) model, fp16, eval mode.

    LoRA config reconstruction mirrors train_lora_delta.evaluate() —
    including the inference_mode/peft_type fields added by commit edcf768.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    base.eval()

    if lora_dir is not None:
        from peft import LoraConfig as _LoraConfig
        from peft import PeftModel

        lora_dir = Path(lora_dir)
        if not (lora_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"LoRA adapter not found at {lora_dir}")
        _cfg = json.loads((lora_dir / "adapter_config.json").read_text())
        _lt = _cfg.pop("layers_to_transform", None)
        _peft_cfg_fields = {
            "task_type",
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "layers_to_transform",
            "layers_pattern",
            "rank_pattern",
            "alpha_pattern",
            "fan_in_fan_out",
            "init_lora_weights",
            "use_dora",
            "use_rslora",
            "loftq_config",
            "inference_mode",
            "peft_type",
        }
        _clean_cfg = {k: v for k, v in _cfg.items() if k in _peft_cfg_fields}
        if _lt is not None:
            _clean_cfg["layers_to_transform"] = _lt
        base = PeftModel.from_pretrained(
            base, str(lora_dir), config=_LoraConfig(**_clean_cfg)
        )
        base.eval()
        print(f"  LoRA adapter loaded: {lora_dir}")

    return base, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Core ops: capture hidden state, early-exit logits, generation
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def capture_layers_h(model, tokenizer, prompt, layer_list, device):
    """Single forward, capture resid_post at multiple layers (last token).

    Returns dict {layer: h[1, 1, d]} (fp16, detached).
    """
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    layers, _, _ = _get_modules(model)
    caches = {}
    handles = []

    def _make_hook(li):
        def _hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            caches[li] = hs[:, -1:, :].detach()

        return _hook

    for li in layer_list:
        handles.append(layers[li].register_forward_hook(_make_hook(li)))
    try:
        model(**input_ids)
    finally:
        for h in handles:
            h.remove()
    return caches


def early_exit_logits(model, h_ref):
    """l = lm_head(norm(h_ref)). fp16. h_ref is [1, 1, d] → returns [1, vocab] float32."""
    _, norm, lm_head = _get_modules(model)
    out = lm_head(norm(h_ref)).float()  # [1, 1, vocab]
    return out[:, 0, :]  # [1, vocab]


@torch.no_grad()
def greedy_generate_hf(model, tokenizer, prompt, device, max_new=20):
    """Greedy generation. Returns (full_text, first_token_logits)."""
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]
    out = model(input_ids=input_ids)
    logits = out.logits[0, -1, :].float().cpu()  # first-token logits for classify/rank
    nid = int(logits.argmax().item())
    gen_ids = [nid]
    current = input_ids
    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break
        nxt = torch.tensor([[gen_ids[-1]]], device=device)
        current = torch.cat([current, nxt], dim=1)
        if current.shape[1] > 1024:
            break
        out = model(current)
        nid = int(out.logits[0, -1, :].argmax().item())
        gen_ids.append(nid)
    return tokenizer.decode(gen_ids).strip(), logits


@torch.no_grad()
def tldc_generate_hf(model, tokenizer, prompt, device, ref_layer, beta, max_new=20):
    """TLDC decoding on HF: l_adj = l_final + β·(l_ref − l_final).

    Returns (full_text, first_token_raw_logits, first_token_adj_logits).
    """
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]
    layers, _, _ = _get_modules(model)

    gen_ids = []
    current = input_ids
    first_raw = None
    for step in range(max_new):
        cache = {}

        def _hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            cache["h"] = hs[:, -1:, :].detach()

        handle = layers[ref_layer].register_forward_hook(_hook)
        try:
            out = model(current)
        finally:
            handle.remove()

        l_final = out.logits[0, -1:, :].float()
        h_ref = cache["h"]
        l_ref = early_exit_logits(model, h_ref)
        l_adj = l_final + beta * (l_ref - l_final)
        nid = int(l_adj.argmax(dim=-1).item())

        if step == 0:
            first_raw = l_final[0].cpu()

        if step > 0 and nid == tokenizer.eos_token_id:
            break
        gen_ids.append(nid)
        current = torch.cat([current, torch.tensor([[nid]], device=device)], dim=1)
        if current.shape[1] > 1024:
            break

    return tokenizer.decode(gen_ids).strip(), first_raw


# ═════════════════════════════════════════════════════════════════════════════
# Mode: classify — 固定分类 (baseline 原始模型)
# ═════════════════════════════════════════════════════════════════════════════


def classify(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(device, args.model_path)

    samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    entries = []
    for i, s in enumerate(tqdm(samples, desc="Classify")):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        gen_text, logits = greedy_generate_hf(model, tokenizer, prompt, device)
        is_correct = check_correct(gen_text, s["answers"], dataset="triviaqa")

        # KC/KW/DK via first-token logits rank (mirrors train_lora_delta.classify_sample)
        if is_correct:
            subset = "KC"
        elif y_true_id is None:
            subset = "DK"
        else:
            sorted_idx = torch.argsort(-logits)
            rank = (sorted_idx == y_true_id).nonzero(as_tuple=True)[0].item() + 1
            subset = "KW" if rank <= RANK_THRESHOLD else "DK"

        entries.append(
            {
                "sample_id": i,
                "question": s["question"],
                "context": s.get("context", ""),
                "answers": s["answers"],
                "y_true_id": y_true_id,
                "prompt": prompt,
                "baseline_correct": bool(is_correct),
                "subset": subset,
            }
        )

    counts = defaultdict(int)
    for e in entries:
        counts[e["subset"]] += 1
    print(
        f"\n  N={len(entries)} | KC={counts['KC']} KW={counts['KW']} DK={counts['DK']}"
    )

    out = OUTPUT_DIR / "entries.json"
    with open(out, "w") as f:
        json.dump(
            {"config": vars(args), "entries": entries}, f, indent=2, ensure_ascii=False
        )
    print(f"  Saved {out}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# Mode: diagnose — A1 ℓ* 重定位, A2 δ 分布, A3 KW 变化
# ═════════════════════════════════════════════════════════════════════════════


def _rank_of(logits, y_true_id):
    """Rank of y_true in logits (0-indexed → +1)."""
    sorted_idx = torch.argsort(-logits)
    return (sorted_idx == y_true_id).nonzero(as_tuple=True)[0].item() + 1


def compute_ref_auroc(model, tokenizer, device, samples, refs):
    """Truth-direction AUROC per reference layer (correct vs wrong)."""
    layers, _, _ = _get_modules(model)
    h_correct = {r: [] for r in refs}
    h_wrong = {r: [] for r in refs}
    labels = []

    # First pass: determine correctness
    for s in tqdm(samples, desc="  label", leave=False):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        gen_text, _ = greedy_generate_hf(model, tokenizer, prompt, device)
        labels.append(int(check_correct(gen_text, s["answers"], dataset="triviaqa")))

    # Second pass: extract h per ref layer
    for i, s in enumerate(tqdm(samples, desc="  extract h", leave=False)):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)
        caches = {}
        handles = []

        def _make_hook(li):
            def _hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                caches[li] = hs[0, -1, :].detach().float().cpu().numpy()  # [d_model]

            return _hook

        for li in refs:
            handles.append(layers[li].register_forward_hook(_make_hook(li)))
        with torch.no_grad():
            model(**tokens)
        for h in handles:
            h.remove()

        for li in refs:
            (h_correct if labels[i] else h_wrong)[li].append(caches[li])
        del tokens, caches

    aurocs = {}
    for li in refs:
        c = np.stack(h_correct[li])
        w = np.stack(h_wrong[li])
        if len(c) == 0 or len(w) == 0:
            aurocs[li] = 0.5
            continue
        v = c.mean(0) - w.mean(0)
        v = v / (np.linalg.norm(v) + 1e-8)
        scores = np.concatenate([np.dot(c, v), np.dot(w, v)])
        labels_arr = np.concatenate([np.ones(len(c)), np.zeros(len(w))])
        aurocs[li] = _compute_auroc(scores, labels_arr)
    return aurocs


def compute_delta_dist(model, tokenizer, kw_samples, ref_layer, final_layer, device):
    """Compute g(d*) − g(t*) for each KW sample. Returns list of floats."""
    vals = []
    for e in tqdm(kw_samples, desc="  δ", leave=False):
        y = e["y_true_id"]
        if y is None:
            continue
        hs = capture_layers_h(
            model, tokenizer, e["prompt"], [ref_layer, final_layer], device
        )
        l_ref = early_exit_logits(model, hs[ref_layer])[0]
        l_final = early_exit_logits(model, hs[final_layer])[0]
        g = l_final - l_ref  # channel gain per token
        g_mask = g.clone()
        g_mask[y] = -float("inf")
        d_star = g_mask.argmax().item()
        vals.append((g[d_star] - g[y]).item())
    return vals


def diagnose(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    refs = args.refs if args.refs else DEFAULT_REFS

    # ── Load entries (fixed classification from baseline) ──
    with open(OUTPUT_DIR / "entries.json") as f:
        data = json.load(f)
    entries = data["entries"]
    kw_samples = [e for e in entries if e["subset"] == "KW"]
    all_samples = [
        {"question": e["question"], "context": e["context"], "answers": e["answers"]}
        for e in entries
    ]
    print(f"  Entries: N={len(entries)}, KW={len(kw_samples)}")

    # ── A1: ℓ* re-location on LoRA model (single model residency) ──
    print("\n[A1] ℓ* re-location (truth-direction AUROC on LoRA model)...")
    lora_model, tokenizer = load_model(device, args.model_path, args.lora_dir)
    aurocs = compute_ref_auroc(lora_model, tokenizer, device, all_samples, refs)
    print("  AUROC per ref layer:", {f"L{r}": round(a, 4) for r, a in aurocs.items()})
    best_ref = max(aurocs, key=aurocs.get)
    print(f"  → ℓ* (LoRA) = L{best_ref}")
    del lora_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── A2: δ distribution — baseline vs LoRA (on fixed KW set) ──
    print("\n[A2] δ = l_final − l_ref distribution on KW samples...")
    ref_final = args.delta_final_ref

    base_model, _ = load_model(device, args.model_path)
    g_kw_base = compute_delta_dist(
        base_model, tokenizer, kw_samples, best_ref, ref_final, device
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    lora_model, _ = load_model(device, args.model_path, args.lora_dir)
    g_kw_lora = compute_delta_dist(
        lora_model, tokenizer, kw_samples, best_ref, ref_final, device
    )

    mean_base = float(np.mean(g_kw_base)) if g_kw_base else 0.0
    mean_lora = float(np.mean(g_kw_lora)) if g_kw_lora else 0.0
    ratio = mean_lora / mean_base if mean_base != 0 else float("nan")
    a2_pass = ratio < 0.8
    print(
        f"  mean g(d*)−g(t*) | baseline={mean_base:.3f}, LoRA={mean_lora:.3f}, ratio={ratio:.3f}"
    )
    print(f"  A2 (LoRA δ < baseline × 0.8): {'✅' if a2_pass else '❌'}")

    # ── A3: KW set changes under LoRA ──
    print("\n[A3] KW samples under LoRA greedy...")
    kw_status = []
    for e in tqdm(kw_samples, desc="  LoRA greedy", leave=False):
        gen_text, _ = greedy_generate_hf(lora_model, tokenizer, e["prompt"], device)
        correct = bool(check_correct(gen_text, e["answers"], dataset="triviaqa"))
        kw_status.append(
            {
                "sample_id": e["sample_id"],
                "correct_under_lora": correct,
                "question": e["question"][:60],
                "gen": gen_text[:40],
            }
        )
    n_fixed = sum(1 for k in kw_status if k["correct_under_lora"])
    print(f"  KW fixed by LoRA: {n_fixed}/{len(kw_status)}")

    result = {
        "config": vars(args),
        "A1": {
            "aurocs": {str(r): round(a, 4) for r, a in aurocs.items()},
            "best_ref": best_ref,
        },
        "A2": {
            "mean_delta_base": mean_base,
            "mean_delta_lora": mean_lora,
            "ratio": round(ratio, 4),
            "pass": bool(a2_pass),
        },
        "A3": {
            "n_kw": len(kw_status),
            "n_fixed_by_lora": n_fixed,
            "details": kw_status,
        },
    }
    out = OUTPUT_DIR / "diagnose.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {out}")
    del lora_model
    gc.collect()
    torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# Mode: intervene — 4 策略 × β sweep
# ═════════════════════════════════════════════════════════════════════════════


def _run_strategy(model, tokenizer, entries, device, mode, ref_layer, beta, max_new):
    """Run one strategy over fixed-classified entries. Returns per-sample results."""
    results = []
    for e in tqdm(
        entries,
        desc=f"  {mode}" + (f" β={beta}" if mode == "tldc" else ""),
        leave=False,
    ):
        if mode == "greedy":
            gen_text, _ = greedy_generate_hf(
                model, tokenizer, e["prompt"], device, max_new
            )
        else:
            gen_text, _ = tldc_generate_hf(
                model, tokenizer, e["prompt"], device, ref_layer, beta, max_new
            )
        correct = bool(check_correct(gen_text, e["answers"], dataset="triviaqa"))
        results.append(
            {
                "sample_id": e["sample_id"],
                "subset": e["subset"],
                "correct": correct,
                "gen": gen_text[:40],
            }
        )
    return results


def summarize(results, entries):
    """Aggregate per-subset EM rates. entries provides fixed subset labels."""
    counts = defaultdict(int)
    correct = defaultdict(int)
    for r, e in zip(results, entries):
        s = e["subset"]
        counts[s] += 1
        counts["all"] += 1
        if r["correct"]:
            correct[s] += 1
            correct["all"] += 1
    out = {}
    for s in ["KC", "KW", "DK", "all"]:
        out[s] = {
            "correct": correct[s],
            "total": counts[s],
            "rate": correct[s] / counts[s] if counts[s] else 0.0,
        }
    return out


def intervene(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "entries.json") as f:
        data = json.load(f)
    entries = data["entries"]
    betas = args.betas if args.betas else DEFAULT_BETAS
    ref_layer = args.ref_layer if args.ref_layer is not None else 20  # 1.7B default ℓ*

    # Load diagnosis to prefer relocated ℓ* if available
    diag_path = OUTPUT_DIR / "diagnose.json"
    if diag_path.exists() and args.ref_layer is None:
        try:
            diag = json.loads(diag_path.read_text())
            ref_layer = diag["A1"]["best_ref"]
            print(f"  Using relocated ℓ* from diagnose: L{ref_layer}")
        except Exception:
            pass

    summary = {
        "baseline": None,
        "tldc": {},
        "lora": None,
        "lora_tldc": {},
        "per_sample": {},
    }

    # ── Phase 1: original model → Baseline + TLDC ──
    print("\n[Phase 1] Original model: Baseline + TLDC")
    base_model, tokenizer = load_model(device, args.model_path)
    res_base = _run_strategy(
        base_model, tokenizer, entries, device, "greedy", ref_layer, 0, args.max_new
    )
    summary["baseline"] = summarize(res_base, entries)
    print(
        f"  Baseline EM: {summary['baseline']['all']['correct']}/{summary['baseline']['all']['total']}"
    )

    summary["per_sample"]["baseline"] = res_base
    for beta in betas:
        res = _run_strategy(
            base_model,
            tokenizer,
            entries,
            device,
            "tldc",
            ref_layer,
            beta,
            args.max_new,
        )
        summary["tldc"][f"beta={beta}"] = summarize(res, entries)
        summary["per_sample"][f"tldc_beta={beta}"] = res
        print(
            f"  TLDC β={beta}: KW={summary['tldc'][f'beta={beta}']['KW']['correct']}/"
            f"{summary['tldc'][f'beta={beta}']['KW']['total']}"
        )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: LoRA model → LoRA + LoRA+TLDC ──
    print("\n[Phase 2] LoRA model: LoRA + LoRA+TLDC")
    lora_model, _ = load_model(device, args.model_path, args.lora_dir)
    res_lora = _run_strategy(
        lora_model, tokenizer, entries, device, "greedy", ref_layer, 0, args.max_new
    )
    summary["lora"] = summarize(res_lora, entries)
    print(
        f"  LoRA EM: {summary['lora']['all']['correct']}/{summary['lora']['all']['total']}"
    )
    summary["per_sample"]["lora"] = res_lora
    for beta in betas:
        res = _run_strategy(
            lora_model,
            tokenizer,
            entries,
            device,
            "tldc",
            ref_layer,
            beta,
            args.max_new,
        )
        summary["lora_tldc"][f"beta={beta}"] = summarize(res, entries)
        summary["per_sample"][f"lora_tldc_beta={beta}"] = res
        print(
            f"  LoRA+TLDC β={beta}: KW={summary['lora_tldc'][f'beta={beta}']['KW']['correct']}/"
            f"{summary['lora_tldc'][f'beta={beta}']['KW']['total']}"
        )

    del lora_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Gate evaluation ──
    kw_max_single = max(
        summary["lora"]["KW"]["correct"],
        max(summary["tldc"][k]["KW"]["correct"] for k in summary["tldc"]),
    )
    kw_max_combo = max(
        summary["lora_tldc"][k]["KW"]["correct"] for k in summary["lora_tldc"]
    )
    g1_pass = kw_max_combo > kw_max_single
    print(
        f"\n  G1 (combo KW > max single): {'✅' if g1_pass else '❌'} "
        f"(combo={kw_max_combo}, single={kw_max_single})"
    )

    # KC degradation relative to baseline KC count
    kc_base = summary["baseline"]["KC"]["correct"]
    kc_lora = summary["lora"]["KC"]["correct"]
    kc_deg_lora = kc_base - kc_lora
    kc_combo_min = min(
        summary["lora_tldc"][k]["KC"]["correct"] for k in summary["lora_tldc"]
    )
    kc_deg_combo = kc_base - kc_combo_min
    g2_pass = kc_deg_combo <= kc_deg_lora + 1
    print(
        f"  G2 (combo KC deg ≤ LoRA KC deg + 1): {'✅' if g2_pass else '❌'} "
        f"(combo={kc_deg_combo}, lora={kc_deg_lora})"
    )

    summary["gates"] = {
        "G1": bool(g1_pass),
        "G2": bool(g2_pass),
        "kw_max_single": kw_max_single,
        "kw_max_combo": kw_max_combo,
        "kc_deg_lora": kc_deg_lora,
        "kc_deg_combo": kc_deg_combo,
        "ref_layer": ref_layer,
    }

    out = OUTPUT_DIR / "intervene.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 23: LoRA+TLDC combined (1.7B)")
    parser.add_argument(
        "--mode", required=True, choices=["classify", "diagnose", "intervene"]
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--refs", type=int, nargs="*", default=None, help="A1 candidate ref layers"
    )
    parser.add_argument(
        "--ref_layer", type=int, default=None, help="ℓ* for TLDC (override)"
    )
    parser.add_argument(
        "--delta_final_ref", type=int, default=27, help="final layer for δ computation"
    )
    parser.add_argument("--betas", type=float, nargs="*", default=None)
    parser.add_argument("--max_new", type=int, default=20)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = MODEL_PATH
    if args.lora_dir is None:
        args.lora_dir = str(DEFAULT_LORA_DIR)

    if args.mode == "classify":
        classify(args)
    elif args.mode == "diagnose":
        diagnose(args)
    else:
        intervene(args)


if __name__ == "__main__":
    main()
