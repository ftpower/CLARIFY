"""A.1: g_vec distribution analysis — test if top-k channel gain vector
outperforms scalar 未 in distinguishing override (KW) from non-override (KC+DK).

Theory: scalar 未 = g(d*) - g(t*) collapses 150K-dim logit space into 1 number.
Hypothesis: top-k gain vector [g(t*), g(d*), g(t_2)-g(t*), ...] preserves
distribution shape information that scalar 未 discards.

Gate A.1: g_vec-AUROC > 未-AUROC + 0.03 on 1.7B LOOCV.

Usage:
  # 1.7B
  python analyze_g_vec.py --n_samples 500

  # 8B
  python analyze_g_vec.py \
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

from src.data_loader import format_prompt, check_correct  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_DELTA_LAYERS = 8
RANK_THRESHOLD = 50
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"


def _get_delta_layers(model) -> tuple[int, int]:
    n_layers = model.config.num_hidden_layers
    last = n_layers - 1
    ref = max(0, n_layers - NUM_DELTA_LAYERS)
    return ref, last


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC (no sklearn dependency)."""
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
    from datasets import load_dataset  # noqa: E402

    ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        question = item["question"]
        answers = item["answer"]["aliases"]
        search_contexts = item["search_results"]["search_context"]
        context = "\n\n".join(ctx for ctx in search_contexts if ctx)
        samples.append({"question": question, "answers": answers, "context": context})
    return samples


def get_first_answer_token_id(tokenizer, answers: list[str]) -> int | None:
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


# ── g_vec computation ──────────────────────────────────────────────────────────


def compute_g_vec(g: np.ndarray, y_true_id: int, k: int = 5) -> np.ndarray:
    """Extract top-k channel gain feature vector.

    Returns:
        g_vec: shape [k+1]
          [g(t*), g(d*), g(t_2)-g(t*), g(t_3)-g(t*), ..., g(t_k)-g(t*)]
    """
    g_masked = g.copy()
    g_masked[y_true_id] = -np.inf

    sorted_idx = np.argsort(-g_masked)  # descending gain
    d_star = sorted_idx[0]
    g_tstar = float(g[y_true_id])
    g_dstar = float(g[d_star])

    features = [g_tstar, g_dstar]
    for i in range(1, k):  # t_2 ... t_k
        if i < len(sorted_idx):
            features.append(float(g[sorted_idx[i]] - g_tstar))
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float64)  # [k+1]


def compute_scalar_delta(g: np.ndarray, y_true_id: int) -> float:
    """Scalar 未 = g(d*) - g(t*)."""
    g_masked = g.copy()
    g_masked[y_true_id] = -np.inf
    d_star = int(np.argmax(g_masked))
    return float(g[d_star] - g[y_true_id])


# ── Logistic Regression with LOOCV (numpy-only, no sklearn dependency) ─────────


class LogisticRegression:
    """L2-regularised logistic regression trained via gradient descent."""

    def __init__(self, l2: float = 1.0, lr: float = 0.01, n_iter: int = 2000):
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.w = None  # [d]
        self.b = None  # scalar

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: [n, d], y: [n] in {0, 1}."""
        n, d = X.shape
        # Standardise features
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        X_s = (X - self.X_mean) / self.X_std

        # Class-balanced initialisation
        w = np.zeros(d, dtype=np.float64)
        b = np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))

        pos_weight = (1 - y).sum() / (y.sum() + 1e-6)

        for _ in range(self.n_iter):
            logits = X_s @ w + b
            p = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))

            # Weighted BCE + L2
            grad_w = (X_s.T @ ((p - y) * np.where(y == 1, pos_weight, 1.0))) / n
            grad_w += self.l2 * w
            grad_b = ((p - y) * np.where(y == 1, pos_weight, 1.0)).mean()

            w -= self.lr * grad_w
            b -= self.lr * grad_b

        self.w = w
        self.b = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = (X - self.X_mean) / self.X_std
        logits = X_s @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(logits, -100, 100)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def loocv_auroc(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> float:
    """Leave-one-out cross-validated AUROC using logistic regression."""
    n = len(y)
    scores = np.empty(n)
    pos_count = y.sum()

    if pos_count < 3:
        # Not enough positive samples for meaningful LOOCV
        # Fall back: train on all data, predict on all data (optimistic)
        clf = LogisticRegression(l2=l2)
        clf.fit(X, y)
        scores = clf.predict_proba(X)
        return _compute_auroc(scores, y)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]

        # Guard: training set must have both classes
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            scores[i] = 0.5
            continue

        clf = LogisticRegression(l2=l2)
        clf.fit(X_train, y_train)
        scores[i] = float(clf.predict_proba(X[i : i + 1])[0])

    return _compute_auroc(scores, y)


# ── Main analysis ─────────────────────────────────────────────────────────────


def analyze_g_vec(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = args.model_path or _find_model_path()
    k = args.top_k
    print(f"A.1: g_vec Analysis | n={args.n_samples} | k={k}")

    # ── 1. Load model + tokenizer ──────────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

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
    print(f"\n[2/5] Loading {args.n_samples} TriviaQA train samples...")
    samples = load_triviaqa_train(n_samples=args.n_samples, seed=args.seed)

    # ── 3. Register hooks ──────────────────────────────────────────────────
    try:
        layers = model.model.layers
        norm_fn = model.model.norm
        lm_head = model.lm_head
    except AttributeError:
        try:
            layers = model.base_model.model.model.layers
            norm_fn = model.base_model.model.model.norm
            lm_head = model.base_model.model.lm_head
        except AttributeError:
            layers = model.model.model.layers
            norm_fn = model.model.model.norm
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

    # ── 4. Extract g_vec per sample ────────────────────────────────────────
    print(f"\n[3/5] Extracting g_vec for {len(samples)} samples...")
    X_all = []  # g_vec features [n, k+1]
    y_all = []  # 1=KW, 0=KC+DK
    delta_all = []  # scalar 未 for comparison
    categories = []
    metadata = []

    for s in tqdm(samples):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        h_ref_cache.clear()
        h_last_cache.clear()

        with torch.no_grad():
            _ = model(**tokens)

        h_ref = h_ref_cache.get("h")
        h_last = h_last_cache.get("h")
        if h_ref is None or h_last is None:
            continue

        # Compute channel gains g(t) = logits_last(t) - logits_ref(t)
        h_ref_norm = norm_fn(h_ref.to(dtype=norm_fn.weight.dtype))
        logits_ref = lm_head(h_ref_norm).float().detach().cpu().numpy().flatten()

        h_last_norm = norm_fn(h_last.to(dtype=norm_fn.weight.dtype))
        logits_last = lm_head(h_last_norm).float().detach().cpu().numpy().flatten()

        g = logits_last - logits_ref  # [vocab]

        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        logits_first = torch.from_numpy(logits_last)
        generated = generate_answer(model, tokenizer, prompt, device)
        category = classify_sample(logits_first, y_true_id, generated, s["answers"])

        if y_true_id is not None and y_true_id < len(g):
            g_vec = compute_g_vec(g, y_true_id, k=k)
            delta_val = compute_scalar_delta(g, y_true_id)
        else:
            g_vec = np.zeros(k + 1, dtype=np.float64)
            delta_val = 0.0

        X_all.append(g_vec)
        y_all.append(1 if category == "KW" else 0)
        delta_all.append(delta_val)
        categories.append(category)
        metadata.append({"question": s["question"][:80], "category": category})

    handle_ref.remove()
    handle_last.remove()

    X = np.array(X_all, dtype=np.float64)  # [n, k+1]
    y = np.array(y_all, dtype=np.int64)  # [n]
    deltas = np.array(delta_all, dtype=np.float64)  # [n]

    n_kw = int(y.sum())
    print(f"\n  Samples: {len(X)} total, {n_kw} KW, {len(X) - n_kw} KC+DK")

    # ── 5. Evaluate ────────────────────────────────────────────────────────
    print(f"\n[4/5] LOOCV evaluation...")

    # A. Full g_vec AUROC
    if n_kw >= 3:
        gvec_auroc = loocv_auroc(X, y, l2=args.l2)
    else:
        gvec_auroc = 0.5
        print("  WARNING: <3 KW samples, AUROC set to 0.5")

    # B. Scalar 未 AUROC
    delta_auroc = _compute_auroc(deltas, y)

    # C. Per-feature ablation
    ablation_results = {}
    if n_kw >= 3:
        feature_names = ["g(t*)", "g(d*)"] + [
            f"g(t_{i})-g(t*)" for i in range(2, k + 1)
        ]
        for j, name in enumerate(feature_names):
            X_ablated = np.delete(X, j, axis=1)
            abl_auroc = loocv_auroc(X_ablated, y, l2=args.l2)
            ablation_results[name] = {
                "auroc": float(abl_auroc),
                "delta_vs_full": float(gvec_auroc - abl_auroc),
            }

    # D. Per-category g(d*) and g(t*) stats
    cat_stats = {}
    for cat in ["KC", "KW", "DK"]:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if idx:
            cat_X = X[idx]
            cat_d = deltas[idx]
            cat_stats[cat] = {
                "n": len(idx),
                "g_tstar_mean": float(cat_X[:, 0].mean()),
                "g_tstar_std": float(cat_X[:, 0].std()),
                "g_dstar_mean": float(cat_X[:, 1].mean()),
                "g_dstar_std": float(cat_X[:, 1].std()),
                "delta_mean": float(cat_d.mean()),
                "delta_std": float(cat_d.std()),
            }

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n[5/5] Results\n{'=' * 60}")

    print(f"\n  Per-category channel gains:")
    for cat in ["KC", "KW", "DK"]:
        if cat in cat_stats:
            s = cat_stats[cat]
            print(
                f"  {cat} (n={s['n']}): "
                f"g(t*)={s['g_tstar_mean']:+.2f}±{s['g_tstar_std']:.2f}, "
                f"g(d*)={s['g_dstar_mean']:+.2f}±{s['g_dstar_std']:.2f}, "
                f"未={s['delta_mean']:+.2f}±{s['delta_std']:.2f}"
            )

    print(f"\n  AUROC (KW vs KC+DK):")
    print(f"    g_vec ({k + 1}-dim):  {gvec_auroc:.4f}")
    print(f"    scalar 未 (1-dim):  {delta_auroc:.4f}")
    delta_auroc_diff = gvec_auroc - delta_auroc
    print(f"    未 AUROC (g_vec - 未): {delta_auroc_diff:+.4f}")

    gate_a1 = delta_auroc_diff > 0.03
    print(
        f"\n  Gate A.1 (g_vec AUROC > 未 AUROC + 0.03): "
        f"{'✅ PASS' if gate_a1 else '❌ FAIL'}"
    )

    if ablation_results:
        print(f"\n  Feature ablation (drop one, LOOCV AUROC):")
        for name, res in sorted(
            ablation_results.items(), key=lambda x: x[1]["delta_vs_full"], reverse=True
        ):
            marker = " ← most informative" if res["delta_vs_full"] > 0.02 else ""
            print(
                f"    drop {name:20s}: AUROC={res['auroc']:.4f} "
                f"(未={res['delta_vs_full']:+.4f}){marker}"
            )

    # ── Save ────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUT_DIR / "g_vec_analysis.json"
    summary = {
        "config": {
            "n_samples": args.n_samples,
            "k": k,
            "seed": args.seed,
            "l2": args.l2,
            "ref_layer": ref_layer,
            "last_layer": last_layer,
            "model": MODEL_PATH,
        },
        "samples": {"total": len(X), "kw": n_kw, "kc_dk": len(X) - n_kw},
        "auroc": {
            "g_vec": float(gvec_auroc),
            "scalar_delta": float(delta_auroc),
            "delta": float(delta_auroc_diff),
        },
        "gate_a1": bool(gate_a1),
        "ablation": ablation_results,
        "per_category": cat_stats,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {out_path}")

    del model
    torch.cuda.empty_cache()


def _find_model_path() -> str:
    MODEL_ID = "Qwen/Qwen3-1.7B"
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if os.path.isdir(local):
            if os.path.isfile(os.path.join(local, "config.json")):
                return local
            snaps = os.path.join(local, "snapshots")
            if os.path.isdir(snaps):
                for s in sorted(os.listdir(snaps)):
                    sp = os.path.join(snaps, s)
                    if os.path.isfile(os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID


def main():
    parser = argparse.ArgumentParser(description="A.1: g_vec distribution analysis")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
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

    analyze_g_vec(args)


if __name__ == "__main__":
    main()
