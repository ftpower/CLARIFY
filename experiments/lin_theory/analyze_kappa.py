"""Phase 21 Step 1: κ-Spikiness distribution analysis — no training, pure measurement.

Theory: docs/theory-kappa-spikiness.md
Plan:   ~/.claude/plans/CLARIFY/phase21-kappa-spikiness.md

Measures channel-gain spikiness κ for KC/KW/DK samples and tests whether κ
can discriminate override from legitimate refinement where scalar δ cannot.

κ = g(d*) - median{g(t) | t in top-k minus t*}

where g(t) = y_last(t) - y_ref(t) is the per-token channel gain.

Gate P21.1: κ-AUROC (KW vs KC+DK) >= 0.70

Usage:
  # 1.7B
  python analyze_kappa.py --n_samples 500

  # 8B
  python analyze_kappa.py \
      --model_path /root/autodl-tmp/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
      --n_samples 500
"""

import argparse
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

# ── Path setup ─────────────────────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import format_prompt, check_correct

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_DELTA_LAYERS = 8  # Same as train_lora_delta.py
RANK_THRESHOLD = 50
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"


def _get_delta_layers(model) -> tuple[int, int]:
    """Return (ref_layer, last_layer) for the last NUM_DELTA_LAYERS."""
    n_layers = model.config.num_hidden_layers
    last = n_layers - 1
    ref = max(0, n_layers - NUM_DELTA_LAYERS)
    return ref, last


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    U = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def load_triviaqa_train(n_samples: int, seed: int = 42) -> list[dict]:
    """Load TriviaQA train split."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        question = item["question"]
        answers = item["answer"]["aliases"]
        search_contexts = item["search_results"]["search_context"]
        context = "\n\n".join(ctx for ctx in search_contexts if ctx)
        samples.append(
            {
                "question": question,
                "answers": answers,
                "context": context,
            }
        )
    return samples


def get_first_answer_token_id(tokenizer, answers: list[str]) -> int | None:
    """Return first token ID of the first non-empty answer alias (with leading space)."""
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


def classify_sample(
    logits: torch.Tensor,
    y_true_id: int | None,
    generated: str,
    answers: list[str],
) -> str:
    """KC/KW/DK classification (same as train_lora_delta.py)."""
    is_correct = check_correct(generated, answers, dataset="triviaqa")
    if is_correct:
        return "KC"
    if y_true_id is None:
        return "DK"
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == y_true_id).nonzero(as_tuple=True)[0].item() + 1
    return "KW" if rank <= RANK_THRESHOLD else "DK"


@torch.no_grad()
def generate_answer(
    model, tokenizer, prompt: str, device: str, max_new: int = 20
) -> str:
    """Greedy generate from prompt. Returns decoded answer text."""
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    first_id = int(logits.argmax().item())

    gen_ids = [first_id]
    current_ids = input_ids
    for _ in range(max_new - 1):
        if gen_ids[-1] == tokenizer.eos_token_id:
            break
        next_tok = torch.tensor([[gen_ids[-1]]], device=device)
        current_ids = torch.cat([current_ids, next_tok], dim=1)
        out = model(input_ids=current_ids)
        nid = int(out.logits[0, -1, :].argmax().item())
        gen_ids.append(nid)

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def compute_kappa(
    g_L27: np.ndarray,  # [vocab], channel gains at last layer
    y_true_id: int,
    top_k: int = 10,
) -> float:
    """Compute spikiness κ for a single sample.

    κ = g(d*) - median{g(t) | t in top-k minus t*}

    Args:
        g_L27: channel gain vector of shape [vocab_size]
        y_true_id: token id of correct answer
        top_k: number of top tokens to consider (excluding y_true)

    Returns:
        κ (float): spikiness score. Higher = more isolated peak = more likely override.
    """
    # Exclude y_true from consideration
    g_masked = g_L27.copy()
    g_masked[y_true_id] = -np.inf

    # Top-k by channel gain (excluding y_true)
    top_k_indices = np.argpartition(-g_masked, min(top_k, len(g_masked) - 1))[:top_k]
    top_k_gains = g_L27[top_k_indices]

    if len(top_k_gains) == 0:
        return 0.0

    d_star = top_k_indices[0]  # distractor = argmax
    g_dstar = g_L27[d_star]
    median_gain = np.median(top_k_gains)

    return float(g_dstar - median_gain)


def compute_delta(g_L27: np.ndarray, y_true_id: int) -> float:
    """Original scalar δ = g(d*) - g(t*)."""
    g_masked = g_L27.copy()
    g_masked[y_true_id] = -np.inf
    d_star = int(np.argmax(g_masked))
    return float(g_L27[d_star] - g_L27[y_true_id])


def analyze_kappa(args):
    """Main analysis: compute κ for each sample, compare distributions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = args.model_path or _find_model_path()
    print(
        f"Phase 21 Step 1: κ-Spikiness Analysis | n={args.n_samples} | k={args.top_k}"
    )

    # ── 1. Load model + tokenizer ──────────────────────────────────────────
    print("\n[1/4] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    ref_layer, last_layer = _get_delta_layers(model)
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(
        f"  Model: {model.config.num_hidden_layers} layers, d_model={d_model}, "
        f"vocab={vocab_size}, ref=L{ref_layer}, last=L{last_layer}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load data ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading {args.n_samples} TriviaQA train samples...")
    samples = load_triviaqa_train(n_samples=args.n_samples, seed=args.seed)

    # ── 3. Register hooks ──────────────────────────────────────────────────
    try:
        layers = model.model.layers
        norm = model.model.norm
        lm_head = model.lm_head
    except AttributeError:
        try:
            layers = model.base_model.model.model.layers
            norm = model.base_model.model.model.norm
            lm_head = model.base_model.model.lm_head
        except AttributeError:
            layers = model.model.model.layers
            norm = model.model.model.norm
            lm_head = model.model.lm_head

    h_ref_cache = {}
    h_last_cache = {}

    def _hook_ref(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        h_ref_cache["h"] = hs[:, -1, :].detach()

    def _hook_last(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        h_last_cache["h"] = hs[:, -1, :].detach()

    handle_ref = layers[ref_layer].register_forward_hook(_hook_ref)
    handle_last = layers[last_layer].register_forward_hook(_hook_last)

    # ── 4. Compute κ per sample ────────────────────────────────────────────
    print(f"\n[3/4] Computing κ for {len(samples)} samples...")
    results = []
    kappa_by_cat = defaultdict(list)
    delta_by_cat = defaultdict(list)

    for s in tqdm(samples):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        h_ref_cache.clear()
        h_last_cache.clear()

        with torch.no_grad():
            _ = model(**tokens)

        # Get channel gains
        h_ref = h_ref_cache.get("h")
        h_last = h_last_cache.get("h")
        if h_ref is None or h_last is None:
            continue

        # Compute logits from both layers
        h_ref_norm = norm(h_ref.to(dtype=norm.weight.dtype))
        logits_ref = lm_head(h_ref_norm).float().cpu().numpy().flatten()  # [vocab]

        h_last_norm = norm(h_last.to(dtype=norm.weight.dtype))
        logits_last = lm_head(h_last_norm).float().cpu().numpy().flatten()  # [vocab]

        g = logits_last - logits_ref  # channel gain [vocab]
        logits_first = torch.from_numpy(logits_last)  # for classification
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])

        # Generate answer for KC/KW/DK classification
        generated = generate_answer(model, tokenizer, prompt, device)
        category = classify_sample(logits_first, y_true_id, generated, s["answers"])

        # Compute κ and δ
        if y_true_id is not None:
            kappa_val = compute_kappa(g, y_true_id, top_k=args.top_k)
            delta_val = compute_delta(g, y_true_id)
        else:
            # If no answer token, use g's max as d* and skip (no t*)
            # This sample goes to DK, use a dummy t* = argmax
            kappa_val = 0.0
            delta_val = 0.0

        kappa_by_cat[category].append(kappa_val)
        delta_by_cat[category].append(delta_val)

        results.append(
            {
                "question": s["question"][:80],
                "category": category,
                "kappa": kappa_val,
                "delta": delta_val,
            }
        )

    handle_ref.remove()
    handle_last.remove()

    # ── 5. Report ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Results\n{'=' * 60}")

    # Per-category statistics
    categories = ["KC", "KW", "DK"]
    print(f"\n  n_total = {len(results)}")
    for cat in categories:
        vals = kappa_by_cat[cat]
        if vals:
            print(
                f"  {cat} (n={len(vals)}): "
                f"κ median={np.median(vals):.4f} mean={np.mean(vals):.4f} "
                f"std={np.std(vals):.4f} "
                f"min={np.min(vals):.4f} max={np.max(vals):.4f}"
            )
        else:
            print(f"  {cat} (n=0): no samples")

    # δ statistics for comparison
    print(f"\n  --- δ (scalar, for comparison) ---")
    for cat in categories:
        vals = delta_by_cat[cat]
        if vals:
            print(
                f"  {cat} (n={len(vals)}): "
                f"δ median={np.median(vals):.4f} mean={np.mean(vals):.4f} "
                f"std={np.std(vals):.4f}"
            )

    # AUROC: κ for KW vs KC+DK
    if len(kappa_by_cat["KW"]) > 0:
        kw_vals = np.array(kappa_by_cat["KW"])
        non_kw_vals = np.concatenate(
            [
                np.array(kappa_by_cat["KC"]),
                np.array(kappa_by_cat["DK"]),
            ]
        )
        scores = np.concatenate([kw_vals, non_kw_vals])
        labels = np.array([1] * len(kw_vals) + [0] * len(non_kw_vals))
        kappa_auroc = _compute_auroc(scores, labels)

        # δ AUROC for comparison
        kw_delta = np.array(delta_by_cat["KW"])
        non_kw_delta = np.concatenate(
            [
                np.array(delta_by_cat["KC"]),
                np.array(delta_by_cat["DK"]),
            ]
        )
        delta_scores = np.concatenate([kw_delta, non_kw_delta])
        delta_auroc = _compute_auroc(delta_scores, labels)

        print(f"\n  κ-AUROC (KW vs KC+DK): {kappa_auroc:.4f}")
        print(f"  δ-AUROC (KW vs KC+DK): {delta_auroc:.4f}")
        print(f"  Δ AUROC (κ - δ):      {kappa_auroc - delta_auroc:+.4f}")
        gate = kappa_auroc >= 0.70
        print(f"\n  Gate P21.1 (κ-AUROC >= 0.70): {'✅ PASS' if gate else '❌ FAIL'}")

    # Distribution percentiles
    if len(kappa_by_cat["KW"]) >= 3 and len(kappa_by_cat["KC"]) >= 3:
        from scipy.stats import mannwhitneyu

        try:
            stat, pval = mannwhitneyu(
                kappa_by_cat["KW"], kappa_by_cat["KC"], alternative="greater"
            )
            print(f"\n  Mann-Whitney U (KW > KC): p={pval:.4f}")
        except ImportError:
            # Manual MW approximation for large samples
            pass

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUT_DIR / "kappa_analysis.json"
    summary = {
        "config": {
            "n_samples": args.n_samples,
            "top_k": args.top_k,
            "seed": args.seed,
            "ref_layer": ref_layer,
            "last_layer": last_layer,
        },
        "per_category": {
            cat: {
                "n": len(kappa_by_cat[cat]),
                "kappa_median": float(np.median(kappa_by_cat[cat]))
                if kappa_by_cat[cat]
                else None,
                "kappa_mean": float(np.mean(kappa_by_cat[cat]))
                if kappa_by_cat[cat]
                else None,
                "kappa_std": float(np.std(kappa_by_cat[cat]))
                if kappa_by_cat[cat]
                else None,
                "delta_median": float(np.median(delta_by_cat[cat]))
                if delta_by_cat[cat]
                else None,
            }
            for cat in categories
        },
        "auroc": {
            "kappa": float(kappa_auroc) if len(kappa_by_cat["KW"]) > 0 else None,
            "delta": float(delta_auroc) if len(kappa_by_cat["KW"]) > 0 else None,
        },
        "gate_p211": bool(gate) if len(kappa_by_cat["KW"]) > 0 else None,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {out_path}")

    del model
    torch.cuda.empty_cache()


def _find_model_path() -> str:
    """Auto-detect local HF model path for 1.7B."""
    import os as _os

    MODEL_ID = "Qwen/Qwen3-1.7B"
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        _os.path.join(
            _os.environ.get("HF_HOME", _os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = _os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if _os.path.isdir(local):
            if _os.path.isfile(_os.path.join(local, "config.json")):
                return local
            snaps = _os.path.join(local, "snapshots")
            if _os.path.isdir(snaps):
                for s in sorted(_os.listdir(snaps)):
                    sp = _os.path.join(snaps, s)
                    if _os.path.isfile(_os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID


def main():
    parser = argparse.ArgumentParser(
        description="Phase 21 Step 1: κ-Spikiness analysis"
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path override (required for 8B).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    analyze_kappa(args)


if __name__ == "__main__":
    main()
