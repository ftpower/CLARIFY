"""I1: Contrastive Direction Guidance — PCA/LDA hallucination directions.

Replaces AAC-lite's simple mean-diff with PCA and LDA directions to get cleaner,
more directionally consistent hallucination suppression.

Key validation: does PCA/LDA solve the "subtract AND add both help" problem?

Usage:
    python main_i1_directions.py --n_dir 300 --n_eval 200
    python main_i1_directions.py --n_dir 300 --n_eval 200 --layers 11,15 --eval_only
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.model_loader import load_model
from src.data_loader import load_hellaswag, format_prompt


# ═══════════════════════════════════════════════════════════════════════════
# Direction computation
# ═══════════════════════════════════════════════════════════════════════════


def _make_save_hook(storage: dict, key: str):
    def hook(activation, hook=None):
        storage[key] = activation.detach()
        return activation

    return hook


def collect_hidden_states(
    model,
    samples: list[dict],
    candidate_layers: list[int],
    letter_ids: dict[str, int],
) -> dict[int, dict]:
    """Collect hidden states at candidate layers, partitioned by correctness.

    Returns:
        {layer: {"correct": [N_c, d_model], "incorrect": [N_i, d_model]}}
    """
    n_layers = model.cfg.n_layers
    accum = {L: {"correct": [], "incorrect": []} for L in candidate_layers}

    storage = {}
    hooks = []
    for L in candidate_layers:
        key = f"blocks.{L}.hook_resid_post"
        hooks.append((key, _make_save_hook(storage, key)))

    letters = ["A", "B", "C", "D"]

    for sample in tqdm(samples, desc="Collecting hidden states"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Determine correctness
        logits_last = logits[0, last_pos, :]
        lid = torch.tensor([letter_ids[l] for l in letters], device=logits_last.device)
        probs = F.softmax(logits_last[lid].float(), dim=-1)
        pred_idx = probs.argmax().item()
        is_correct = letters[pred_idx] == correct_letter

        for L in candidate_layers:
            key = f"blocks.{L}.hook_resid_post"
            h = storage[key][0, last_pos, :].cpu()  # [d_model]
            if is_correct:
                accum[L]["correct"].append(h)
            else:
                accum[L]["incorrect"].append(h)

    for L in candidate_layers:
        n_corr = len(accum[L]["correct"])
        n_incorr = len(accum[L]["incorrect"])
        print(f"  L{L}: {n_corr} correct, {n_incorr} incorrect")

    return accum


def compute_mean_diff_direction(
    accum: dict,
) -> torch.Tensor:
    """Original AAC-lite direction: normalized(mean_incorrect - mean_correct)."""
    h_corr = torch.stack(accum["correct"]).mean(dim=0)
    h_incorr = torch.stack(accum["incorrect"]).mean(dim=0)
    diff = h_incorr - h_corr
    return diff / (diff.norm() + 1e-8)


def compute_pca_direction(
    accum: dict,
) -> tuple[torch.Tensor, float, float]:
    """PCA direction from error residuals r_i = h_i - centroid_correct.

    Returns:
        direction: [d_model] unit-norm PC1
        explained_variance_ratio: float — fraction of variance explained by PC1
        cos_with_mean_diff: float — cosine similarity with mean-diff direction
    """
    h_corr = torch.stack(accum["correct"]).mean(dim=0)
    residuals = []
    for h in accum["incorrect"]:
        r = (h - h_corr).numpy()
        residuals.append(r)
    R = np.stack(residuals, axis=0)  # [n_error, d_model]

    pca = PCA(n_components=1)
    pca.fit(R)
    pc1 = torch.from_numpy(pca.components_[0]).float()  # [d_model]
    pc1 = pc1 / (pc1.norm() + 1e-8)
    evr = float(pca.explained_variance_ratio_[0])

    # Cosine with mean-diff
    md = compute_mean_diff_direction(accum)
    cos_md = float((pc1 * md).sum())

    return pc1, evr, cos_md


def compute_lda_direction(
    accum: dict,
    reg: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Regularized LDA direction: w = (S_W + γ·trace(S_W)/d·I)^{-1} (μ_1 - μ_0).

    Uses Ledoit-Wolf shrinkage for S_W then adds ridge penalty for inversion stability.
    d_model=2048 >> n_samples=300, so S_W is severely rank-deficient.

    Returns:
        direction: [d_model] unit-norm LDA discriminant axis
        cos_with_pca: float
    """
    from sklearn.covariance import LedoitWolf

    X_c = torch.stack(accum["correct"]).numpy().astype(np.float64)
    X_i = torch.stack(accum["incorrect"]).numpy().astype(np.float64)

    mu_c = X_c.mean(axis=0)
    mu_i = X_i.mean(axis=0)
    mean_diff = mu_i - mu_c  # same as mean-diff direction

    # Center each class
    X_c_ctr = X_c - mu_c
    X_i_ctr = X_i - mu_i

    # Ledoit-Wolf shrinkage for pooled within-class covariance
    lw = LedoitWolf()
    # Pool: concatenate centered data
    X_pool = np.concatenate([X_c_ctr, X_i_ctr], axis=0)
    S_W = lw.fit(X_pool).covariance_  # [d, d] shrunk

    # Ridge regularization for inversion
    d = S_W.shape[0]
    gamma = reg * np.trace(S_W) / d
    S_W_reg = S_W + gamma * np.eye(d)

    # Solve: w = S_W_reg^{-1} @ mean_diff
    try:
        w = np.linalg.solve(S_W_reg, mean_diff)
    except np.linalg.LinAlgError:
        w = mean_diff  # fall back to mean-diff

    direction = torch.from_numpy(w).float()
    direction = direction / (direction.norm() + 1e-8)

    # Cosine with PCA
    pca_dir, _, _ = compute_pca_direction(accum)
    cos_pca = float((direction * pca_dir).sum())

    # Cosine with mean-diff
    md = compute_mean_diff_direction(accum)
    cos_md = float((direction * md).sum())
    print(
        f"  LDA:       cos with PCA = {cos_pca:+.4f}, cos with mean-diff = {cos_md:+.4f}"
    )

    return direction, cos_pca


def compute_random_direction(d_model: int) -> torch.Tensor:
    """Random unit-norm direction (control)."""
    d = torch.randn(d_model)
    return d / (d.norm() + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# Intervention evaluation
# ═══════════════════════════════════════════════════════════════════════════


def make_projection_hook(direction: torch.Tensor, lam: float, mode: str = "subtract"):
    """Return hook that projects hidden states onto/against the given direction."""
    sign = -1.0 if mode == "subtract" else 1.0

    def hook(activation, hook=None):
        d = direction.to(activation.dtype).to(activation.device)
        proj_mag = activation @ d  # [batch, seq]
        projection = proj_mag.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return activation + sign * lam * projection

    return hook


def evaluate_interventions(
    model,
    eval_samples: list[dict],
    all_directions: dict[str, dict[int, torch.Tensor]],  # {method: {layer: direction}}
    letter_ids: dict[str, int],
    layers: list[int],
    lams: list[float],
    modes: list[str],
) -> dict:
    """Evaluate intervention across method x layer x lambda x mode.

    Sample-major loop for efficiency: for each sample, run all configs.
    """
    letters = ["A", "B", "C", "D"]
    n_layers = model.cfg.n_layers

    # Pre-tokenize
    print("Pre-tokenizing eval samples...")
    tokenized = []
    for sample in tqdm(eval_samples, desc="Tokenizing"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="hellaswag"
        )
        correct_letter = sample["answers"][1].upper()
        tokens = model.to_tokens(prompt, prepend_bos=True)
        tokenized.append({"tokens": tokens, "correct_letter": correct_letter})

    # Build config list
    methods = list(all_directions.keys())
    configs = [
        (method, L, lam, mode)
        for method in methods
        for L in layers
        for lam in lams
        for mode in modes
    ]

    # Initialize accumulators
    accum = {cfg: {"n_correct": 0, "per_sample": []} for cfg in configs}

    # Baseline: no intervention
    n_base_correct = 0
    base_per_sample = []
    lid_tensor = torch.tensor([letter_ids[l] for l in letters])

    for idx, item in enumerate(tqdm(tokenized, desc="Evaluating", leave=True)):
        tokens = item["tokens"]
        correct_letter = item["correct_letter"]
        lid = lid_tensor.to(tokens.device)

        # Baseline
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[])
        logits_last = logits[0, -1, :]
        probs_base = F.softmax(logits_last[lid].float(), dim=-1)
        pred_base = letters[probs_base.argmax().item()]
        is_base_correct = pred_base == correct_letter
        p_correct_base = probs_base[letters.index(correct_letter)].item()
        n_base_correct += int(is_base_correct)
        base_per_sample.append(
            {"is_correct": is_base_correct, "p_correct": p_correct_base}
        )

        # Run all intervention configs
        for method, L, lam, mode in configs:
            direction = all_directions[method][L].to(tokens.device)
            hook_point = f"blocks.{L}.hook_resid_post"
            hook_fn = make_projection_hook(direction, lam, mode)

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_point, hook_fn)],
                )

            logits_last = logits[0, -1, :]
            probs = F.softmax(logits_last[lid].float(), dim=-1)
            pred_idx = probs.argmax().item()
            pred = letters[pred_idx]
            is_correct = pred == correct_letter
            p_correct = probs[letters.index(correct_letter)].item()

            accum[(method, L, lam, mode)]["n_correct"] += int(is_correct)
            accum[(method, L, lam, mode)]["per_sample"].append(
                {
                    "pred": pred,
                    "correct": correct_letter,
                    "is_correct": is_correct,
                    "p_correct": p_correct,
                }
            )

    # ── Build results ──
    n_total = len(tokenized)
    baseline_acc = n_base_correct / n_total

    baseline_filtered = [s for s in base_per_sample if s["p_correct"] > 0.3]
    n_base_filt = len(baseline_filtered)
    baseline_filt_acc = (
        sum(s["is_correct"] for s in baseline_filtered) / n_base_filt
        if n_base_filt >= 20
        else None
    )

    results = []
    for method, L, lam, mode in configs:
        a = accum[(method, L, lam, mode)]
        acc = a["n_correct"] / n_total
        delta = acc - baseline_acc

        filtered = [s for s in a["per_sample"] if s["p_correct"] > 0.3]
        n_filt = len(filtered)
        acc_filt = (
            sum(s["is_correct"] for s in filtered) / n_filt if n_filt >= 20 else None
        )
        delta_f = (
            acc_filt - baseline_filt_acc
            if (acc_filt is not None and baseline_filt_acc is not None)
            else None
        )

        results.append(
            {
                "method": method,
                "layer": L,
                "lambda": lam,
                "mode": mode,
                "accuracy": float(acc),
                "delta": float(delta),
                "accuracy_filtered": float(acc_filt) if acc_filt is not None else None,
                "delta_filtered": float(delta_f) if delta_f is not None else None,
                "n_filtered": n_filt,
                "n_correct": a["n_correct"],
            }
        )

    return {
        "results": results,
        "baseline_acc": float(baseline_acc),
        "baseline_filt_acc": float(baseline_filt_acc) if baseline_filt_acc else None,
        "n_total": n_total,
        "n_base_filtered": n_base_filt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_file = output_dir / "i1_directions.pt"

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = load_model(device=device, model_id=args.model)
    model.eval()

    letter_ids = {}
    for letter in ["A", "B", "C", "D"]:
        tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]
    print(f"Letter token IDs: {letter_ids}")

    d_model = model.cfg.d_model
    layers = args.layers

    if args.eval_only:
        print(f"\nLoading directions from {dir_file}")
        ckpt = torch.load(dir_file, map_location=device)
        all_directions = {}
        for method_key, layer_dict in ckpt.items():
            all_directions[method_key] = {
                int(L): d.to(device) for L, d in layer_dict.items()
            }
        print(f"Loaded methods: {list(all_directions.keys())}")
        print(f"Layers: {list(all_directions[list(all_directions.keys())[0]].keys())}")
    else:
        # ── Phase 1: Collect hidden states from train set ──────────
        print(f"\n{'=' * 60}")
        print("Phase 1: Collecting hidden states for direction computation")
        print(f"{'=' * 60}")

        print("Loading HellaSwag train split...")
        ds_train = load_dataset(
            "Rowan/hellaswag", split="train", trust_remote_code=False
        )
        ds_train = ds_train.shuffle(seed=args.seed)
        label_letters = ["A", "B", "C", "D"]
        dir_samples = []
        for item in ds_train.select(range(min(args.n_dir, len(ds_train)))):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            label_letter = label_letters[label]
            choices_text = "\n".join(
                f"{label_letters[i]}. {endings[i]}" for i in range(4)
            )
            dir_samples.append(
                {
                    "question": ctx,
                    "answers": [endings[label], label_letter],
                    "context": choices_text,
                }
            )

        accum = collect_hidden_states(model, dir_samples, layers, letter_ids)

        # ── Phase 2: Compute directions ────────────────────────────
        print(f"\n{'=' * 60}")
        print("Phase 2: Computing directions (mean-diff, PCA, LDA, random)")
        print(f"{'=' * 60}")

        all_directions = {
            "mean_diff": {},
            "pca": {},
            "lda": {},
            "random": {},
        }

        for L in layers:
            print(f"\n--- L{L} ---")

            # Mean-diff (original AAC-lite)
            md = compute_mean_diff_direction(accum[L])
            all_directions["mean_diff"][L] = md.cpu()
            print(
                f"  mean-diff:  ||diff|| before norm = {(torch.stack(accum[L]['incorrect']).mean(0) - torch.stack(accum[L]['correct']).mean(0)).norm().item():.4f}"
            )

            # PCA
            pc1, evr, cos_md = compute_pca_direction(accum[L])
            all_directions["pca"][L] = pc1.cpu()
            print(
                f"  PCA:        PC1 explains {evr:.3%} variance, cos with mean-diff = {cos_md:+.4f}"
            )

            # LDA
            lda_dir, cos_pca = compute_lda_direction(accum[L])
            all_directions["lda"][L] = lda_dir.cpu()

            # Random control
            rand_dir = compute_random_direction(d_model)
            all_directions["random"][L] = rand_dir.cpu()
            print(
                f"  random:    cos with mean-diff = {(rand_dir * md).sum().item():+.4f}"
            )

        # ── Save directions ────────────────────────────────────────
        save_dict = {
            method: {str(L): d.cpu() for L, d in layer_dict.items()}
            for method, layer_dict in all_directions.items()
        }
        torch.save(save_dict, dir_file)
        print(f"\nSaved directions to {dir_file}")

    # ── Phase 3: Evaluate interventions ──────────────────────────────
    print(f"\n{'=' * 60}")
    print("Phase 3: Evaluating intervention directions")
    print(f"{'=' * 60}")

    print("Loading HellaSwag validation for evaluation...")
    eval_samples = load_hellaswag(n_samples=args.n_eval, seed=args.seed + 1)

    eval_data = evaluate_interventions(
        model=model,
        eval_samples=eval_samples,
        all_directions=all_directions,
        letter_ids=letter_ids,
        layers=layers,
        lams=args.lam,
        modes=args.modes.split(","),
    )

    # ── Report ───────────────────────────────────────────────────────
    baseline = eval_data["baseline_acc"]
    baseline_filt = eval_data["baseline_filt_acc"]
    print(
        f"\nBaseline: full={baseline:.4f}, filtered(P>0.3)={baseline_filt:.4f} (n={eval_data['n_base_filtered']})"
        if baseline_filt
        else f"\nBaseline: full={baseline:.4f}"
    )

    # Directionality check
    print(f"\n{'=' * 60}")
    print("Directionality Check (subtract vs add at λ=1.0)")
    print(f"{'=' * 60}")
    print(
        f"{'Method':<12} {'Layer':<6} {'Sub Δf':>10} {'Add Δf':>10} {'Directional?':>14}"
    )
    print("-" * 56)

    directionality_results = {}
    for method in ["mean_diff", "pca", "lda"]:
        for L in layers:
            sub_delta = None
            add_delta = None
            for r in eval_data["results"]:
                if r["method"] == method and r["layer"] == L and r["lambda"] == 1.0:
                    if r["mode"] == "subtract":
                        sub_delta = r["delta_filtered"]
                    elif r["mode"] == "add":
                        add_delta = r["delta_filtered"]

            if sub_delta is not None and add_delta is not None:
                directional = sub_delta > 0 and add_delta < 0
                print(
                    f"{method:<12} L{L:<5} {sub_delta:>+10.4f} {add_delta:>+10.4f} {'YES' if directional else 'no':>14}"
                )
                directionality_results[f"{method}_L{L}"] = {
                    "subtract_delta": sub_delta,
                    "add_delta": add_delta,
                    "directional": directional,
                }

    # Best per method
    print(f"\n{'=' * 60}")
    print("Best result per method (filtered set)")
    print(f"{'=' * 60}")
    print(f"{'Method':<12} {'Layer':<6} {'λ':<8} {'Mode':<10} {'AccF':>8} {'Δf':>8}")
    print("-" * 56)

    for method in ["mean_diff", "pca", "lda"]:
        method_results = [
            r
            for r in eval_data["results"]
            if r["method"] == method and r["delta_filtered"] is not None
        ]
        if not method_results:
            continue
        best = max(method_results, key=lambda r: r["delta_filtered"])
        print(
            f"{method:<12} L{best['layer']:<5} {best['lambda']:<8} {best['mode']:<10} "
            f"{best['accuracy_filtered']:.4f} {best['delta_filtered']:>+.4f}"
        )

    # Random control
    rand_results = [
        r
        for r in eval_data["results"]
        if r["method"] == "random" and r["delta_filtered"] is not None
    ]
    if rand_results:
        rand_deltas = [r["delta_filtered"] for r in rand_results]
        print(
            f"\nRandom direction control: mean Δf = {np.mean(rand_deltas):+.4f}, "
            f"std = {np.std(rand_deltas):.4f}, "
            f"range = [{min(rand_deltas):+.4f}, {max(rand_deltas):+.4f}]"
        )

    # ── Save ──────────────────────────────────────────────────────────
    out = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "baseline_acc": baseline,
        "baseline_filt_acc": baseline_filt,
        "n_dir_samples": args.n_dir,
        "n_eval_samples": args.n_eval,
        "directionality": directionality_results,
        "sweep": eval_data["results"],
    }
    with open(output_dir / "i1_directions_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_dir / 'i1_directions_results.json'}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dir", type=int, default=300)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=[11, 15])
    parser.add_argument("--lam", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--modes", type=str, default="subtract,add")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
