"""8B Cross-Model Validation: D2 (JS consistency) + I1 (direction intervention) + S1 (pipeline).

Unified script — single model load, phased execution with disk caching.
Compares 8B results against 1.7B baselines to test cross-model generalisation.

Usage:
    python main_8b_validation.py                        # full pipeline, 500 val samples
    python main_8b_validation.py --n_val 200 --n_dir 200  # quick test
    python main_8b_validation.py --phase d2             # D2 only
    python main_8b_validation.py --phase i1             # I1 only
    python main_8b_validation.py --phase s1             # S1 only (needs D2+I1 results)

Key 8B layer mapping (0-indexed, Qwen3-8B has 36 blocks L0-L35):
    L35 = wAUROC optimal (phase1 "L36"), blocks.35.hook_resid_post
    L17 = mid bottleneck (phase1 "L18"),   blocks.17.hook_resid_post
    L0  = embedding,                       blocks.0.hook_resid_post
    I1 candidate layers: [11, 17, 35] — early, mid, deep
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt

LETTERS = ["A", "B", "C", "D"]
I1_CANDIDATE_LAYERS = [11, 17, 35]
I1_LAMS = [0.1, 0.3, 0.5, 1.0, 2.0]
I1_MODES = ["subtract", "add"]


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════


def _make_save_hook(storage, key):
    def hook(activation, hook=None):
        storage[key] = activation.detach()
        return activation

    return hook


def get_letter_ids(model):
    d = {}
    for letter in LETTERS:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        d[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
    return d


def load_train_samples(n: int, seed: int) -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed)
    label_letters = ["A", "B", "C", "D"]
    samples = []
    for item in ds.select(range(min(n, len(ds)))):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        label_letter = label_letters[label]
        choices_text = "\n".join(f"{label_letters[i]}. {endings[i]}" for i in range(4))
        samples.append(
            {
                "question": ctx,
                "answers": [endings[label], label_letter],
                "context": choices_text,
            }
        )
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Hidden state extraction (shared across D2 + S1)
# ═══════════════════════════════════════════════════════════════════════════


def extract_all_states(model, samples, letter_ids, cache_path):
    """Extract per-layer: choice_probs, max_p, norms, and last-position hidden states.

    Returns dict for D2/D3/D4 + S1 pipeline.
    """
    n_samples = len(samples)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    letter_toks = [letter_ids[l] for l in LETTERS]

    labels = np.zeros(n_samples, dtype=np.int32)
    p_correct_arr = np.zeros(n_samples, dtype=np.float32)
    max_prob = np.zeros((n_samples, n_layers), dtype=np.float32)
    choice_probs = np.zeros((n_samples, n_layers, 4), dtype=np.float32)
    norms = np.zeros((n_samples, n_layers), dtype=np.float32)
    states = np.zeros((n_samples, n_layers, d_model), dtype=np.float16)

    storage = {}
    hooks = [
        (f"blocks.{i}.hook_resid_post", _make_save_hook(storage, f"L{i}"))
        for i in range(n_layers)
    ]

    for idx, sample in enumerate(tqdm(samples, desc="Extracting states")):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        for li in range(n_layers):
            h = storage[f"L{li}"][0, last_pos, :]  # [d_model]
            states[idx, li, :] = h.detach().cpu().to(torch.float16)
            norms[idx, li] = h.norm(p=2).item()

            logits_L = h.to(W_U.device) @ W_U
            if b_U is not None:
                logits_L = logits_L + b_U.to(W_U.device)
            probs_L = torch.softmax(logits_L, dim=-1)
            max_prob[idx, li] = probs_L.max().item()

            cprobs = torch.softmax(logits_L[letter_toks], dim=-1)
            choice_probs[idx, li, :] = cprobs.detach().cpu().to(torch.float32)

        logits_last = logits[0, last_pos, :]
        final_cprobs = torch.softmax(logits_last[letter_toks].float(), dim=-1)
        pred_idx = final_cprobs.argmax().item()
        labels[idx] = int(LETTERS[pred_idx] == correct_letter)
        p_correct_arr[idx] = final_cprobs[LETTERS.index(correct_letter)].item()

    data = {
        "labels": labels,
        "p_correct": p_correct_arr,
        "max_prob": max_prob,
        "choice_probs": choice_probs,
        "norms": norms,
        "states": states,
    }
    np.savez_compressed(cache_path, **{k: v for k, v in data.items() if k != "states"})
    # states saved separately due to size
    np.savez_compressed(str(cache_path).replace(".npz", "_states.npz"), states=states)
    print(f"States cached to {cache_path}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: D2 — Layer-wise JS consistency scan
# ═══════════════════════════════════════════════════════════════════════════


def run_d2(data, output_dir):
    print("\n" + "=" * 60)
    print("D2: Layer-wise JS Consistency (8B)")
    print("=" * 60)

    labels = data["labels"]
    p_correct = data["p_correct"]
    choice_probs = data["choice_probs"]
    max_prob = data["max_prob"]
    n_layers = choice_probs.shape[1]

    # Knowledge filter
    kmask = p_correct > 0.3
    f_labels = labels[kmask]
    f_choice = choice_probs[kmask]
    f_maxp = max_prob[kmask]
    n_f = kmask.sum()
    print(f"Filtered (P>0.3): {n_f}/{len(labels)}, acc={f_labels.mean():.4f}")

    # Scan all layer pairs
    results = []
    for early in tqdm(range(n_layers), desc="D2 scanning"):
        for late in range(early + 1, n_layers):
            p_e = np.clip(f_choice[:, early, :], 1e-10, 1.0)
            p_l = np.clip(f_choice[:, late, :], 1e-10, 1.0)
            m = 0.5 * (p_e + p_l)

            kl_e = np.sum(p_e * np.log(p_e / np.clip(m, 1e-10, 1.0)), axis=1)
            kl_l = np.sum(p_l * np.log(p_l / np.clip(m, 1e-10, 1.0)), axis=1)
            js = 0.5 * (kl_e + kl_l)

            try:
                auroc_js = roc_auc_score(f_labels, js)
            except ValueError:
                auroc_js = 0.5

            # Joint with max_p at late layer
            X = np.stack([f_maxp[:, late], js], axis=1)
            try:
                joint = cross_val_score(
                    LogisticRegression(max_iter=1000),
                    X,
                    f_labels,
                    cv=5,
                    scoring="roc_auc",
                ).mean()
            except Exception:
                joint = float("nan")

            results.append(
                {
                    "early": early,
                    "late": late,
                    "auroc_js": float(auroc_js),
                    "auroc_joint": float(joint),
                    "mean_js": float(js.mean()),
                }
            )

    results.sort(key=lambda r: r["auroc_js"], reverse=True)
    best = results[0]
    print(f"\nBest pair: L{best['early']} vs L{best['late']}")
    print(f"  AUROC(JS)       = {best['auroc_js']:.4f}")
    print(f"  AUROC(joint)    = {best['auroc_joint']:.4f}")

    # Best max_p reference
    best_mp, best_mp_layer = 0.0, -1
    for li in range(n_layers):
        auc = roc_auc_score(f_labels, f_maxp[:, li])
        if auc > best_mp:
            best_mp, best_mp_layer = auc, li
    print(f"  Best max_p:     L{best_mp_layer} = {best_mp:.4f}")

    # Full-set JS
    p_e_full = np.clip(choice_probs[:, best["early"], :], 1e-10, 1.0)
    p_l_full = np.clip(choice_probs[:, best["late"], :], 1e-10, 1.0)
    m_full = 0.5 * (p_e_full + p_l_full)
    js_full = 0.5 * (
        np.sum(p_e_full * np.log(p_e_full / np.clip(m_full, 1e-10, 1.0)), axis=1)
        + np.sum(p_l_full * np.log(p_l_full / np.clip(m_full, 1e-10, 1.0)), axis=1)
    )
    best_mp_full = max(roc_auc_score(labels, max_prob[:, li]) for li in range(n_layers))
    print(
        f"\nFull set: JS AUROC = {roc_auc_score(labels, js_full):.4f}, "
        f"best max_p = {best_mp_full:.4f}"
    )

    # Top-15 pairs
    print(f"\nTop-15 layer pairs:")
    for r in results[:15]:
        print(
            f"  L{r['early']:>2} vs L{r['late']:>2}: JS={r['auroc_js']:.4f}, "
            f"joint={r['auroc_joint']:.4f}"
        )

    out = {
        "n_layers": n_layers,
        "best_pair": {"early": best["early"], "late": best["late"]},
        "auroc_js_filtered": best["auroc_js"],
        "auroc_joint_filtered": best["auroc_joint"],
        "best_maxp_filtered": {"layer": best_mp_layer, "auroc": float(best_mp)},
        "auroc_js_full": float(roc_auc_score(labels, js_full)),
        "best_maxp_full": float(best_mp_full),
        "filtered_n": int(n_f),
        "filtered_acc": float(f_labels.mean()),
        "top_pairs": results[:30],
    }
    with open(output_dir / "d2_8b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'd2_8b_results.json'}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: I1 — Direction computation + intervention evaluation
# ═══════════════════════════════════════════════════════════════════════════


def collect_train_states(model, train_samples, layers, letter_ids):
    """Collect per-layer hidden states from train set for direction computation."""
    accum = {L: {"correct": [], "incorrect": []} for L in layers}
    storage = {}
    hooks = [
        (f"blocks.{L}.hook_resid_post", _make_save_hook(storage, f"L{L}"))
        for L in layers
    ]

    for sample in tqdm(train_samples, desc="Collecting train states"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        logits_last = logits[0, last_pos, :]
        lid = torch.tensor([letter_ids[l] for l in LETTERS], device=logits_last.device)
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        is_correct = LETTERS[probs.argmax().item()] == correct_letter

        for L in layers:
            h = storage[f"L{L}"][0, last_pos, :].cpu()
            if is_correct:
                accum[L]["correct"].append(h)
            else:
                accum[L]["incorrect"].append(h)

    for L in layers:
        print(
            f"  L{L}: {len(accum[L]['correct'])} correct, "
            f"{len(accum[L]['incorrect'])} incorrect"
        )
    return accum


def compute_directions(accum, layers, d_model):
    """Compute mean_diff, PCA, LDA, random directions per layer."""
    all_dirs = {"mean_diff": {}, "pca": {}, "lda": {}, "random": {}}

    for L in layers:
        h_corr = torch.stack(accum[L]["correct"]).float()
        h_incorr = torch.stack(accum[L]["incorrect"]).float()

        # mean-diff
        md = h_incorr.mean(0) - h_corr.mean(0)
        md = md / (md.norm() + 1e-8)
        all_dirs["mean_diff"][L] = md

        # PCA
        residuals = (h_incorr - h_corr.mean(0)).numpy()
        pca = PCA(n_components=1)
        pca.fit(residuals)
        pc1 = torch.from_numpy(pca.components_[0]).float()
        pc1 = pc1 / (pc1.norm() + 1e-8)
        cos_md = float((pc1 * md).sum())
        all_dirs["pca"][L] = pc1
        print(
            f"  L{L} PCA: evr={pca.explained_variance_ratio_[0]:.3%}, "
            f"cos(mean-diff)={cos_md:+.4f}"
        )

        # LDA
        from sklearn.covariance import LedoitWolf

        X_c = h_corr.numpy().astype(np.float64)
        X_i = h_incorr.numpy().astype(np.float64)
        mu_c, mu_i = X_c.mean(0), X_i.mean(0)
        mean_diff = mu_i - mu_c
        X_pool = np.concatenate([X_c - mu_c, X_i - mu_i], axis=0)
        lw = LedoitWolf()
        S_W = lw.fit(X_pool).covariance_
        reg = 0.1 * np.trace(S_W) / d_model
        try:
            w = np.linalg.solve(S_W + reg * np.eye(d_model), mean_diff)
        except np.linalg.LinAlgError:
            w = mean_diff
        lda_d = torch.from_numpy(w).float()
        lda_d = lda_d / (lda_d.norm() + 1e-8)
        all_dirs["lda"][L] = lda_d
        print(f"  L{L} LDA: cos(mean-diff)={float((lda_d * md).sum()):+.4f}")

        # random
        r = torch.randn(d_model)
        all_dirs["random"][L] = r / (r.norm() + 1e-8)

    return all_dirs


def evaluate_i1(model, eval_samples, all_dirs, letter_ids, layers, lams, modes):
    """Evaluate intervention configurations."""
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers

    print("Pre-tokenizing eval samples...")
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append(
            {
                "tokens": tokens,
                "correct_letter": sample["answers"][1].upper(),
            }
        )

    methods = list(all_dirs.keys())
    configs = [
        (m, L, lam, mode)
        for m in methods
        for L in layers
        for lam in lams
        for mode in modes
    ]

    accum = {cfg: {"n_correct": 0} for cfg in configs}
    n_base_correct = 0
    base_per_sample = []
    lid_tensor = torch.tensor([letter_ids[l] for l in LETTERS])

    for item in tqdm(tokenized, desc="Evaluating I1"):
        tokens = item["tokens"]
        correct = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)

        # Baseline
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        probs_base = F.softmax(logits_last[lid].float(), dim=-1)
        is_correct = LETTERS[probs_base.argmax().item()] == correct
        n_base_correct += int(is_correct)
        base_per_sample.append(
            {
                "is_correct": is_correct,
                "p_correct": probs_base[LETTERS.index(correct)].item(),
            }
        )

        for method, L, lam, mode in configs:
            direction = all_dirs[method][L].to(device)
            sign = -1.0 if mode == "subtract" else 1.0

            def make_hook(d, sgn, lam_val):
                def hook(act, hook=None):
                    dd = d.to(act.dtype).to(act.device)
                    proj = act @ dd
                    return act + sgn * lam_val * proj.unsqueeze(-1) * dd.unsqueeze(
                        0
                    ).unsqueeze(0)

                return hook

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (f"blocks.{L}.hook_resid_post", make_hook(direction, sign, lam))
                    ],
                )
            logits_last = logits[0, -1, :]
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            correct_pred = LETTERS[probs.argmax().item()] == correct
            accum[(method, L, lam, mode)]["n_correct"] += int(correct_pred)

    n_total = len(tokenized)
    baseline_acc = n_base_correct / n_total
    baseline_filt = [s for s in base_per_sample if s["p_correct"] > 0.3]
    n_base_f = len(baseline_filt)
    baseline_f_acc = (
        sum(s["is_correct"] for s in baseline_filt) / n_base_f
        if n_base_f >= 20
        else None
    )

    print(f"Baseline: full={baseline_acc:.4f}, filtered(P>0.3)={baseline_f_acc}")

    # Directionality check
    print(
        f"\n{'Method':<12} {'L':<6} {'Sub Δf':>10} {'Add Δf':>10} {'Directional?':>14}"
    )
    print("-" * 56)
    for method in ["mean_diff", "pca", "lda"]:
        for L in layers:
            sub_d, add_d = None, None
            for (m, lyr, lam, mode), a in accum.items():
                if m == method and lyr == L and lam == 1.0:
                    acc = a["n_correct"] / n_total
                    if mode == "subtract":
                        sub_d = acc - baseline_acc
                    else:
                        add_d = acc - baseline_acc
            if sub_d is not None and add_d is not None:
                directional = sub_d > 0 and add_d < 0
                print(
                    f"{method:<12} L{L:<5} {sub_d:>+10.4f} {add_d:>+10.4f} "
                    f"{'YES' if directional else 'no':>14}"
                )

    # Best per method
    print(f"\nBest per method (full set):")
    for method in methods:
        method_res = [(cfg, a) for cfg, a in accum.items() if cfg[0] == method]
        if not method_res:
            continue
        best_cfg, best_a = max(method_res, key=lambda x: x[1]["n_correct"])
        _, best_L, best_lam, best_mode = best_cfg
        best_acc = best_a["n_correct"] / n_total
        print(
            f"  {method:<12} L{best_L:<5} λ={best_lam:<5} {best_mode:<10} "
            f"acc={best_acc:.4f} Δ={best_acc - baseline_acc:+.4f}"
        )

    # Random control
    rand_deltas = []
    for (m, L, lam, mode), a in accum.items():
        if m == "random":
            rand_deltas.append(a["n_correct"] / n_total - baseline_acc)
    if rand_deltas:
        print(
            f"\nRandom control: mean Δ={np.mean(rand_deltas):+.4f}, "
            f"std={np.std(rand_deltas):.4f}"
        )

    results = []
    for (method, L, lam, mode), a in accum.items():
        acc = a["n_correct"] / n_total
        results.append(
            {
                "method": method,
                "layer": L,
                "lambda": lam,
                "mode": mode,
                "accuracy": float(acc),
                "delta": float(acc - baseline_acc),
                "n_correct": a["n_correct"],
            }
        )

    return {
        "results": results,
        "baseline_acc": float(baseline_acc),
        "baseline_filt_acc": float(baseline_f_acc) if baseline_f_acc else None,
        "n_total": n_total,
        "n_base_filtered": n_base_f,
    }


def run_i1(model, output_dir, args):
    print("\n" + "=" * 60)
    print("I1: Direction Intervention (8B)")
    print("=" * 60)

    letter_ids = get_letter_ids(model)
    layers = args.i1_layers
    d_model = model.cfg.d_model

    dir_file = output_dir / "i1_8b_directions.pt"

    # Phase A: Compute directions from train split
    print(f"\nPhase A: Computing directions from train split (n={args.n_dir})...")
    train_samples = load_train_samples(args.n_dir, args.seed)
    accum = collect_train_states(model, train_samples, layers, letter_ids)
    all_dirs = compute_directions(accum, layers, d_model)

    save_dict = {
        m: {str(L): d.cpu() for L, d in ld.items()} for m, ld in all_dirs.items()
    }
    torch.save(save_dict, dir_file)
    print(f"Saved directions to {dir_file}")

    # Phase B: Evaluate interventions on val split
    print(f"\nPhase B: Evaluating interventions on val split (n={args.n_eval})...")
    eval_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)
    eval_data = evaluate_i1(
        model,
        eval_samples,
        all_dirs,
        letter_ids,
        layers,
        I1_LAMS,
        I1_MODES,
    )

    out = {
        "args": {"n_dir": args.n_dir, "n_eval": args.n_eval, "layers": layers},
        "baseline_acc": eval_data["baseline_acc"],
        "baseline_filt_acc": eval_data["baseline_filt_acc"],
        "results": eval_data["results"],
    }
    with open(output_dir / "i1_8b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'i1_8b_results.json'}")
    return eval_data


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: S1 — Detect-then-Intervene pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_s1(model, output_dir, d2_result, i1_result, args):
    print("\n" + "=" * 60)
    print("S1: Detect-then-Intervene Pipeline (8B)")
    print("=" * 60)

    # Read best D2 pair and I1 config from results
    js_early = d2_result["best_pair"]["early"]
    js_late = d2_result["best_pair"]["late"]
    print(f"D2 best pair: L{js_early} vs L{js_late}")

    # Find best I1 config (highest delta among mean_diff, subtract mode)
    i1_configs = [
        r
        for r in i1_result["results"]
        if r["method"] == "mean_diff" and r["mode"] == "subtract"
    ]
    best_i1 = max(i1_configs, key=lambda r: r["delta"])
    int_layer = best_i1["layer"]
    best_lam = best_i1["lambda"]
    print(f"Best I1: L{int_layer}, λ={best_lam}, Δ={best_i1['delta']:+.4f}")

    # Load direction
    dir_file = output_dir / "i1_8b_directions.pt"
    if dir_file.exists():
        directions = torch.load(dir_file, map_location="cpu")
        direction = directions["mean_diff"][str(int_layer)]
        print(f"Loaded mean_diff direction for L{int_layer}")
    else:
        print(f"WARNING: {dir_file} not found, using random direction")
        direction = torch.randn(model.cfg.d_model)
        direction = direction / direction.norm()

    letter_ids = get_letter_ids(model)
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers

    # Load eval samples
    eval_samples = load_hellaswag(n_samples=args.n_val, seed=args.seed)
    print(f"Using {len(eval_samples)} validation samples")

    # Step 1: Extract features (max_p + JS)
    print("\nStep 1: Extracting detection features...")
    storage = {}
    hooks = [
        (f"blocks.{i}.hook_resid_post", _make_save_hook(storage, f"L{i}"))
        for i in range(n_layers)
    ]
    W_U = model.unembed.W_U

    baseline_results = []
    letter_toks = [letter_ids[l] for l in LETTERS]

    for sample in tqdm(eval_samples, desc="Forward pass"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # JS at best pair
        h_e = storage[f"L{js_early}"][0, last_pos, :].to(W_U.device)
        h_l = storage[f"L{js_late}"][0, last_pos, :].to(W_U.device)
        logits_e = h_e @ W_U
        logits_l = h_l @ W_U
        p_e = F.softmax(logits_e[letter_toks].float(), dim=-1).detach().cpu().numpy()
        p_l = F.softmax(logits_l[letter_toks].float(), dim=-1).detach().cpu().numpy()
        eps = 1e-10
        m = 0.5 * (p_e + p_l)
        js = float(
            0.5
            * (
                np.sum(
                    np.clip(p_e, eps, 1.0)
                    * np.log(np.clip(p_e, eps, 1.0) / np.clip(m, eps, 1.0))
                )
                + np.sum(
                    np.clip(p_l, eps, 1.0)
                    * np.log(np.clip(p_l, eps, 1.0) / np.clip(m, eps, 1.0))
                )
            )
        )

        # max_p from final logit lens
        h_final = storage[f"L{n_layers - 1}"][0, last_pos, :].to(W_U.device)
        max_p_val = float(F.softmax((h_final @ W_U).float(), dim=-1).detach().max())

        logits_last = logits[0, last_pos, :]
        choice_probs = F.softmax(logits_last[letter_toks].float(), dim=-1)
        p_c = float(choice_probs[LETTERS.index(correct_letter)])
        is_c = LETTERS[choice_probs.argmax().item()] == correct_letter

        baseline_results.append(
            {
                "js": js,
                "max_p": max_p_val,
                "p_correct": p_c,
                "is_correct": is_c,
                "correct_letter": correct_letter,
                "tokens": tokens,
            }
        )

    n_total = len(baseline_results)
    baseline_acc = sum(r["is_correct"] for r in baseline_results) / n_total
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Knowledge filter
    kmask = np.array([r["p_correct"] > 0.3 for r in baseline_results])
    n_know = kmask.sum()
    k_acc = sum(r["is_correct"] for r, m in zip(baseline_results, kmask) if m) / n_know
    print(f"Knowledge-filtered: {n_know} samples, acc={k_acc:.4f}")

    # Step 2: Train detector (CV LR on JS + max_p)
    print("\nStep 2: Training risk detector...")
    f_js = np.array([r["js"] for r, m in zip(baseline_results, kmask) if m])
    f_mp = np.array([r["max_p"] for r, m in zip(baseline_results, kmask) if m])
    f_labels = np.array(
        [r["is_correct"] for r, m in zip(baseline_results, kmask) if m], dtype=int
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    risk_scores = np.zeros(n_know)
    for tr, te in cv.split(f_js, f_labels):
        X_tr = np.stack([f_js[tr], f_mp[tr]], axis=1)
        X_te = np.stack([f_js[te], f_mp[te]], axis=1)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_tr, f_labels[tr])
        risk_scores[te] = lr.predict_proba(X_te)[:, 1]

    detector_auroc = roc_auc_score(f_labels, risk_scores)
    print(f"Detector AUROC (CV): {detector_auroc:.4f}")

    # Step 3: Intervention at various thresholds
    print("\nStep 3: Detect-then-Intervene...")
    filt_indices = np.where(kmask)[0]

    # Blind intervention (all knowledge-filtered)
    print("Evaluating blind intervention...")
    blind_correct = 0
    direction = direction.to(device)

    def make_proj_hook(d, lam, mode):
        sign = -1.0 if mode == "subtract" else 1.0

        def hook(act, hook=None):
            dd = d.to(act.dtype).to(act.device)
            proj = act @ dd
            return act + sign * lam * proj.unsqueeze(-1) * dd.unsqueeze(0).unsqueeze(0)

        return hook

    for fi, idx in enumerate(tqdm(filt_indices, desc="Blind intervene")):
        r = baseline_results[idx]
        with torch.no_grad():
            logits = model.run_with_hooks(
                r["tokens"],
                fwd_hooks=[
                    (
                        f"blocks.{int_layer}.hook_resid_post",
                        make_proj_hook(direction, best_lam, "subtract"),
                    )
                ],
            )
        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last[letter_toks].float(), dim=-1)
        blind_correct += int(LETTERS[probs.argmax().item()] == r["correct_letter"])

    blind_acc = blind_correct / n_know
    blind_delta = blind_acc - k_acc
    print(f"Blind intervene: {blind_acc:.4f} (Δ={blind_delta:+.4f})")

    # Per-threshold
    risk_thresholds = np.linspace(0.1, 0.9, 9)
    by_threshold = []
    for thresh in risk_thresholds:
        int_c, int_t = 0, 0
        skip_c, skip_t = 0, 0
        for fi, idx in enumerate(filt_indices):
            r = baseline_results[idx]
            if risk_scores[fi] >= thresh:
                with torch.no_grad():
                    logits = model.run_with_hooks(
                        r["tokens"],
                        fwd_hooks=[
                            (
                                f"blocks.{int_layer}.hook_resid_post",
                                make_proj_hook(direction, best_lam, "subtract"),
                            )
                        ],
                    )
                logits_last = logits[0, -1, :]
                probs = F.softmax(logits_last[letter_toks].float(), dim=-1)
                int_c += int(LETTERS[probs.argmax().item()] == r["correct_letter"])
                int_t += 1
            else:
                skip_c += int(r["is_correct"])
                skip_t += 1

        combined = (int_c + skip_c) / (int_t + skip_t)
        delta = combined - k_acc
        by_threshold.append(
            {
                "risk_threshold": float(thresh),
                "n_intervened": int_t,
                "n_skipped": skip_t,
                "intervene_acc": float(int_c / int_t) if int_t > 0 else None,
                "skip_acc": float(skip_c / skip_t) if skip_t > 0 else None,
                "combined_acc": float(combined),
                "delta": float(delta),
            }
        )

    print(
        f"\n{'Thr':<6} {'N_int':>6} {'N_skip':>6} {'IntAcc':>8} {'SkipAcc':>8} "
        f"{'CombAcc':>8} {'Δ':>8}"
    )
    best_r = None
    for r in by_threshold:
        ia = f"{r['intervene_acc']:.4f}" if r["intervene_acc"] else "N/A"
        sa = f"{r['skip_acc']:.4f}" if r["skip_acc"] else "N/A"
        print(
            f"{r['risk_threshold']:<6.2f} {r['n_intervened']:>6} {r['n_skipped']:>6} "
            f"{ia:>8} {sa:>8} {r['combined_acc']:>8.4f} {r['delta']:>+8.4f}"
        )
        if best_r is None or r["delta"] > best_r["delta"]:
            best_r = r

    if best_r:
        print(
            f"\nBest: thr={best_r['risk_threshold']:.2f}, "
            f"n_int={best_r['n_intervened']}, Δ={best_r['delta']:+.4f}"
        )

    print(f"\nSummary:")
    print(f"  Baseline (full):        {baseline_acc:.4f}")
    print(f"  Baseline (knowledge):   {k_acc:.4f}")
    print(f"  Blind intervene:        {blind_acc:.4f}  (Δ={blind_delta:+.4f})")
    if best_r:
        print(
            f"  Detect-then-intervene:  {best_r['combined_acc']:.4f}  (Δ={best_r['delta']:+.4f})"
        )
        print(f"    → n_intervened={best_r['n_intervened']}/{n_know}")
        print(
            f"    → selectivity gain: {best_r['delta'] - blind_delta:+.4f} pp over blind"
        )

    out = {
        "model": "Qwen3-8B",
        "n_total": n_total,
        "baseline_acc": float(baseline_acc),
        "n_knowledge": int(n_know),
        "knowledge_baseline_acc": float(k_acc),
        "blind_intervene_acc": float(blind_acc),
        "blind_delta": float(blind_delta),
        "detector_auroc": float(detector_auroc),
        "d2_best_pair": [js_early, js_late],
        "i1_best_layer": int_layer,
        "i1_best_lambda": best_lam,
        "by_threshold": by_threshold,
        "best_threshold": best_r,
    }
    with open(output_dir / "s1_8b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 's1_8b_results.json'}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="8B Cross-Model Validation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs_8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_val", type=int, default=500, help="Validation samples for D2 + S1"
    )
    parser.add_argument(
        "--n_dir",
        type=int,
        default=300,
        help="Train samples for I1 direction computation",
    )
    parser.add_argument(
        "--n_eval", type=int, default=200, help="Validation samples for I1 evaluation"
    )
    parser.add_argument(
        "--i1_layers",
        type=int,
        nargs="+",
        default=I1_CANDIDATE_LAYERS,
        help="Layers to test for I1 direction intervention",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "d2", "i1", "s1"],
        help="Which phase(s) to run",
    )
    parser.add_argument(
        "--skip_extract", action="store_true", help="Use cached states for D2"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"8B Cross-Model Validation — Phase: {args.phase}")
    print(f"Model: {args.model}, Output: {output_dir}")

    # ── Load model ──
    print(f"\nLoading {args.model}...")
    model = load_model(device=args.device, model_id=args.model)
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Loaded: {n_layers} layers, d_model={model.cfg.d_model}")

    # ── Phase: D2 ──
    if args.phase in ("all", "d2"):
        letter_ids = get_letter_ids(model)
        state_cache = output_dir / "phase3_8b_states.npz"

        if args.skip_extract and state_cache.exists():
            print(f"Loading cached states from {state_cache}")
            cached = np.load(state_cache, allow_pickle=True)
            data = {k: cached[k] for k in cached.files}
        else:
            print(f"Loading HellaSwag val ({args.n_val} samples)...")
            val_samples = load_hellaswag(n_samples=args.n_val, seed=args.seed)
            data = extract_all_states(model, val_samples, letter_ids, state_cache)

        d2_result = run_d2(data, output_dir)
    else:
        d2_result = None

    # ── Phase: I1 ──
    if args.phase in ("all", "i1"):
        i1_eval_data = run_i1(model, output_dir, args)
    else:
        i1_eval_data = None

    # ── Phase: S1 ──
    if args.phase in ("all", "s1"):
        if d2_result is None:
            d2_file = output_dir / "d2_8b_results.json"
            if d2_file.exists():
                with open(d2_file) as f:
                    d2_result = json.load(f)
            else:
                print("ERROR: D2 results not found. Run with --phase d2 first.")
                sys.exit(1)

        if i1_eval_data is None:
            i1_file = output_dir / "i1_8b_results.json"
            if i1_file.exists():
                with open(i1_file) as f:
                    saved = json.load(f)
                    i1_eval_data = {
                        "baseline_acc": saved["baseline_acc"],
                        "baseline_filt_acc": saved["baseline_filt_acc"],
                        "results": saved["results"],
                    }
            else:
                print("ERROR: I1 results not found. Run with --phase i1 first.")
                sys.exit(1)

        run_s1(model, output_dir, d2_result, i1_eval_data, args)

    # ── Cleanup ──
    del model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"8B Validation complete in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
