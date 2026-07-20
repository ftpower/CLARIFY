"""Innovation 5 (Plan 2): Zero-Training Attn/FFN Functional Differentiation.

Hypothesis (from InternalInspector + ROME/Knowledge Neurons):
  - TriviaQA (factual QA):  FFN output norm > Attention output norm → FFN-dominated
                             signals are more predictive of hallucination.
  - SQuAD (reading comp):   Attention output norm > FFN output norm → Attn-dominated
                             signals are more predictive.
  - HellaSwag (commonsense): Balanced → either anomaly could signal hallucination.

Validation via controlled experiment:
  - TriviaQA: AUROC(FFN norm) > AUROC(Attention norm)
  - SQuAD:    AUROC(Attention norm) > AUROC(FFN norm)
  - Random labels (control): Both → 0.5

Usage:
    python main_attn_ffn_task_analysis.py --n_samples 200 --device cuda
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase4_generalization"))

from src.model_loader import load_model
from src.data_loader import (
    load_hellaswag, load_triviaqa, load_squad,
    format_prompt, check_correct,
)
from phase4_utils.hidden_state_extended import (
    extract_all_sub_layer_states, generate_with_temperature,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction — per-layer Attn/FFN norm extraction across 3 datasets
# ═══════════════════════════════════════════════════════════════════════════════


def extract_task_features(
    model,
    samples: list[dict],
    dataset_name: str,
    letter_ids: dict[str, int] | None = None,
) -> dict:
    """Extract per-layer Attn and FFN L2 norms, plus correctness labels.

    For HellaSwag: uses 4-choice logit-lens correctness.
    For TriviaQA/SQuAD: uses generation-based correctness.

    Returns:
        labels: [N] int
        attn_norms: [N, n_layers] float32
        ffn_norms: [N, n_layers] float32
        ratios: [N, n_layers] float32 — attn_norm / ffn_norm
    """
    n_layers = model.cfg.n_layers
    all_attn = []
    all_ffn = []
    labels_list = []

    for sample in tqdm(samples, desc=f"Extracting {dataset_name}"):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset=dataset_name
        )
        sub = extract_all_sub_layer_states(model, prompt)

        # Per-layer norms
        attn_norms = np.array(
            [sub["attn"][li].norm(p=2).item() for li in range(n_layers)],
            dtype=np.float32,
        )
        ffn_norms = np.array(
            [sub["ffn"][li].norm(p=2).item() for li in range(n_layers)],
            dtype=np.float32,
        )

        # Correctness
        if dataset_name == "hellaswag":
            letters = ["A", "B", "C", "D"]
            letter_tok_ids = [letter_ids[l] for l in letters]
            correct_letter = sample["answers"][1].upper()
            logits_last = sub["logits"]
            cf = logits_last[letter_tok_ids]
            pf = torch.softmax(cf.float(), dim=-1)
            is_correct = letters[pf.argmax().item()] == correct_letter
        else:
            answer = generate_with_temperature(
                model, prompt, temperature=0.0, top_p=1.0, max_new_tokens=20,
            )
            is_correct = check_correct(answer, sample["answers"], dataset=dataset_name)

        all_attn.append(attn_norms)
        all_ffn.append(ffn_norms)
        labels_list.append(int(is_correct))

    N = len(samples)
    return {
        "labels": np.array(labels_list, dtype=np.int32),
        "attn_norms": np.stack(all_attn, axis=0),  # [N, n_layers]
        "ffn_norms": np.stack(all_ffn, axis=0),    # [N, n_layers]
        "ratios": np.stack(all_attn, axis=0) / (np.stack(all_ffn, axis=0) + 1e-8),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Hypothesis test
# ═══════════════════════════════════════════════════════════════════════════════


def test_functional_differentiation(
    data_hellaswag: dict,
    data_triviaqa: dict,
    data_squad: dict,
    n_layers: int,
) -> dict:
    """Test the Attn/FFN functional differentiation hypothesis across datasets."""

    results = {
        "per_layer": {},
        "hypothesis_tests": {},
    }

    for dataset_name, data in [
        ("hellaswag", data_hellaswag),
        ("triviaqa", data_triviaqa),
        ("squad", data_squad),
    ]:
        labels = data["labels"]
        attn_norms = data["attn_norms"]
        ffn_norms = data["ffn_norms"]
        ratios = data["ratios"]

        # Per-layer AUROC
        attn_aurocs = []
        ffn_aurocs = []
        ratio_aurocs = []

        for li in range(n_layers):
            try:
                attn_auroc = roc_auc_score(labels, attn_norms[:, li])
            except ValueError:
                attn_auroc = float("nan")
            try:
                ffn_auroc = roc_auc_score(labels, ffn_norms[:, li])
            except ValueError:
                ffn_auroc = float("nan")
            try:
                ratio_auroc = roc_auc_score(labels, ratios[:, li])
            except ValueError:
                ratio_auroc = float("nan")

            attn_aurocs.append(attn_auroc)
            ffn_aurocs.append(ffn_auroc)
            ratio_aurocs.append(ratio_auroc)

        # Best layers
        best_attn_idx = int(np.nanargmax(attn_aurocs))
        best_ffn_idx = int(np.nanargmax(ffn_aurocs))
        best_ratio_idx = int(np.nanargmax(ratio_aurocs))

        results["per_layer"][dataset_name] = {
            "attn_aurocs": [float(a) if not np.isnan(a) else None for a in attn_aurocs],
            "ffn_aurocs": [float(a) if not np.isnan(a) else None for a in ffn_aurocs],
            "ratio_aurocs": [float(a) if not np.isnan(a) else None for a in ratio_aurocs],
            "best_attn": {"layer": best_attn_idx, "auroc": float(attn_aurocs[best_attn_idx])},
            "best_ffn": {"layer": best_ffn_idx, "auroc": float(ffn_aurocs[best_ffn_idx])},
            "best_ratio": {"layer": best_ratio_idx, "auroc": float(ratio_aurocs[best_ratio_idx])},
        }

    # ── Hypothesis tests ──
    tests = {}

    # H1: TriviaQA — FFN > Attn
    tq_best_ffn = results["per_layer"]["triviaqa"]["best_ffn"]["auroc"]
    tq_best_attn = results["per_layer"]["triviaqa"]["best_attn"]["auroc"]
    tests["triviaqa_ffn_gt_attn"] = {
        "hypothesis": "TriviaQA: FFN norm AUROC > Attn norm AUROC",
        "ffn_auroc": tq_best_ffn,
        "attn_auroc": tq_best_attn,
        "supported": tq_best_ffn > tq_best_attn,
        "delta": tq_best_ffn - tq_best_attn,
    }

    # H2: SQuAD — Attn > FFN
    sq_best_ffn = results["per_layer"]["squad"]["best_ffn"]["auroc"]
    sq_best_attn = results["per_layer"]["squad"]["best_attn"]["auroc"]
    tests["squad_attn_gt_ffn"] = {
        "hypothesis": "SQuAD: Attn norm AUROC > FFN norm AUROC",
        "ffn_auroc": sq_best_ffn,
        "attn_auroc": sq_best_attn,
        "supported": sq_best_attn > sq_best_ffn,
        "delta": sq_best_attn - sq_best_ffn,
    }

    # H3: Ratio (combined) is better than either alone on HellaSwag
    hs_best_ratio = results["per_layer"]["hellaswag"]["best_ratio"]["auroc"]
    hs_best_ffn = results["per_layer"]["hellaswag"]["best_ffn"]["auroc"]
    hs_best_attn = results["per_layer"]["hellaswag"]["best_attn"]["auroc"]
    tests["hellaswag_ratio_gt_either"] = {
        "hypothesis": "HellaSwag: Attn/FFN ratio AUROC > max(Attn, FFN) alone",
        "ratio_auroc": hs_best_ratio,
        "max_single": max(hs_best_ffn, hs_best_attn),
        "supported": hs_best_ratio > max(hs_best_ffn, hs_best_attn),
        "delta": hs_best_ratio - max(hs_best_ffn, hs_best_attn),
    }

    results["hypothesis_tests"] = tests

    # ── Overall verdict ──
    all_supported = all(t["supported"] for t in tests.values())
    results["claim5_supported"] = all_supported

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_task_report(results: dict):
    """Pretty-print the functional differentiation analysis."""

    print(f"\n{'=' * 60}")
    print("Per-Dataset Best AUROC: Attn vs FFN vs Ratio")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<14} {'Best Attn':>12} {'Best FFN':>12} {'Best Ratio':>12}")
    print("-" * 52)

    for ds in ["hellaswag", "triviaqa", "squad"]:
        pd = results["per_layer"][ds]
        print(
            f"{ds:<14} "
            f"L{pd['best_attn']['layer']}:{pd['best_attn']['auroc']:.4f}  "
            f"L{pd['best_ffn']['layer']}:{pd['best_ffn']['auroc']:.4f}  "
            f"L{pd['best_ratio']['layer']}:{pd['best_ratio']['auroc']:.4f}"
        )

    print(f"\n{'=' * 60}")
    print("Hypothesis Tests")
    print(f"{'=' * 60}")

    for key, test in results["hypothesis_tests"].items():
        icon = "✅" if test["supported"] else "❌"
        print(f"\n{icon} {test['hypothesis']}")
        print(f"   Δ = {test['delta']:+.4f}")

    if results["claim5_supported"]:
        print(f"\n✅ Claim 5 SUPPORTED: Attn/FFN ratio is a zero-training, ")
        print("   task-dependent hallucination diagnostic.")
    else:
        print(f"\n⚠ Claim 5 NOT fully supported — review per-dataset results.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Innovation 5: Attn/FFN Functional Differentiation"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_random_control", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "attn_ffn_task_features.npz"

    if args.skip_extract and cache_path.exists():
        print(f"Loading cached from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data_all = cached["data_all"].item()
    else:
        print(f"Loading model {args.model}...")
        model = load_model(device=args.device, model_id=args.model)
        model.eval()
        n_layers = model.cfg.n_layers
        print(f"Model: {n_layers} layers")

        letter_ids = {}
        for letter in ["A", "B", "C", "D"]:
            tok_ids = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
            letter_ids[letter] = tok_ids[-1] if len(tok_ids) >= 1 else tok_ids[0]

        # Load three datasets
        print(f"\nLoading HellaSwag (n={args.n_samples})...")
        hs_samples = load_hellaswag(n_samples=args.n_samples, seed=args.seed)
        print(f"Loading TriviaQA (n={args.n_samples})...")
        tq_samples = load_triviaqa(n_samples=args.n_samples, seed=args.seed)
        print(f"Loading SQuAD (n={args.n_samples})...")
        sq_samples = load_squad(n_samples=args.n_samples, seed=args.seed)

        print(f"\n{'=' * 60}")
        print("Extracting Attn/FFN features across 3 datasets")
        print(f"{'=' * 60}")

        data_hs = extract_task_features(model, hs_samples, "hellaswag", letter_ids)
        data_tq = extract_task_features(model, tq_samples, "triviaqa")
        data_sq = extract_task_features(model, sq_samples, "squad")

        data_all = {
            "hellaswag": data_hs,
            "triviaqa": data_tq,
            "squad": data_sq,
        }

        np.savez_compressed(cache_path, data_all=data_all)
        print(f"Cached to {cache_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    n_layers = data_all["hellaswag"]["attn_norms"].shape[1]

    # ── Random label control ──
    if not args.skip_random_control:
        print(f"\n{'=' * 60}")
        print("Random Label Control")
        print(f"{'=' * 60}")

        for ds_name in ["hellaswag", "triviaqa", "squad"]:
            data = data_all[ds_name]
            N = len(data["labels"])
            rand_labels = np.random.RandomState(args.seed).randint(0, 2, N)

            attn_aurocs_rand = []
            ffn_aurocs_rand = []
            for li in range(n_layers):
                try:
                    attn_aurocs_rand.append(
                        roc_auc_score(rand_labels, data["attn_norms"][:, li])
                    )
                except ValueError:
                    attn_aurocs_rand.append(0.5)
                try:
                    ffn_aurocs_rand.append(
                        roc_auc_score(rand_labels, data["ffn_norms"][:, li])
                    )
                except ValueError:
                    ffn_aurocs_rand.append(0.5)

            print(
                f"  {ds_name}: random-label Attn AUROC = {np.mean(attn_aurocs_rand):.3f} ± "
                f"{np.std(attn_aurocs_rand):.3f}, "
                f"FFN AUROC = {np.mean(ffn_aurocs_rand):.3f} ± {np.std(ffn_aurocs_rand):.3f}"
            )

            if np.mean(attn_aurocs_rand) > 0.55 or np.mean(ffn_aurocs_rand) > 0.55:
                print(f"  ⚠ WARNING: Random label AUROC > 0.55 — possible confound!")

    # ── Hypothesis test ──
    print(f"\n{'=' * 60}")
    print("Functional Differentiation Analysis")
    print(f"{'=' * 60}")

    results = test_functional_differentiation(
        data_all["hellaswag"], data_all["triviaqa"], data_all["squad"], n_layers,
    )
    print_task_report(results)

    # Save
    output = {
        "config": {"n_samples": args.n_samples, "model": args.model, "seed": args.seed},
        "dataset_stats": {
            ds: {
                "n_samples": len(data_all[ds]["labels"]),
                "accuracy": float(data_all[ds]["labels"].mean()),
            }
            for ds in ["hellaswag", "triviaqa", "squad"]
        },
        "results": results,
    }

    with open(output_dir / "attn_ffn_task_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'attn_ffn_task_results.json'}")


if __name__ == "__main__":
    main()
