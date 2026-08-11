"""Synthesize KW samples via knowledge conflict injection.

For KC samples (model answers correctly), inject misleading context containing the
model's own top distractor (d*). If the model now generates a wrong answer while
y_true stays in the top-50 logits, we have a synthetic KW sample with a real 未 signal.

Theory: The injected context pushes up the distractor's logit, creating genuine
g(d*) > g(t*) override. The 未 signal comes from the forward-pass logits, so the
training gradient is real — unlike temperature sampling which only changes which
token is selected, not the logits themselves.

Usage:
  # 1.7B (local test)
  python synthesize_kw_conflict.py --n_samples 200 --n_conflict 100

  # 8B (server)
  python synthesize_kw_conflict.py \
      --model_path /root/autodl-tmp/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
      --n_samples 1000 --n_conflict 500
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

from analyze_g_vec import (  # noqa: E402
    _get_delta_layers,
    _compute_auroc,
    load_triviaqa_train,
    get_first_answer_token_id,
    generate_answer,
    compute_scalar_delta,
    NUM_DELTA_LAYERS,
    RANK_THRESHOLD,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"


# ── Misleading context templates ──────────────────────────────────────────────

CONFLICT_TEMPLATES = [
    # Assertive: state the distractor as fact
    "Recent research indicates that {dstar}. {original_context}",
    "According to authoritative sources, {dstar}. {original_context}",
    "It is now widely accepted that {dstar}. {original_context}",
    # Suggestive: hint at the distractor
    "Some experts argue that {dstar}. {original_context}",
    "New evidence suggests that {dstar}. {original_context}",
    "Historical records point to {dstar}. {original_context}",
    # Embedding: weave distractor into the context
    "{original_context}\n\nNotably, {dstar}.",
]

# Template strengths: weak → medium → strong
# Weak:   suggestive hint, model may ignore
# Medium: assertive claim, model may conflict
# Strong: authoritative override, risks pushing to DK
CONFLICT_TEMPLATES_BY_STRENGTH = {
    "weak": [
        "Some readers have wondered whether {dstar}.",
        "One possible answer might be {dstar}.",
        "It has been asked if {dstar}.",
    ],
    "medium": [
        "Some experts argue that {dstar} is the correct answer.",
        "Recent evidence points to {dstar} as the answer.",
        "Several sources claim that {dstar} is correct.",
    ],
    "strong": [
        "According to authoritative research, {dstar} is definitively the correct answer.",
        "It is now firmly established that {dstar} is the correct answer.",
        "All credible sources confirm that {dstar} is the right answer.",
    ],
}


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


# ── Distractor decoding ───────────────────────────────────────────────────────


def decode_distractor_phrase(
    model, tokenizer, prompt: str, y_true_id: int, device: str, max_tokens: int = 10
) -> str | None:
    """Generate a short phrase starting from the top distractor token.

    Runs a short greedy decode from the model's first forward pass, forcing the
    first generated token to be the distractor d* (not y_true). This produces a
    semantically meaningful wrong-answer phrase that can be injected as misleading
    context.
    """
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        output = model(**tokens)
    logits = output.logits[0, -1, :]  # [vocab]

    # Find d* = best token that isn't y_true
    masked = logits.clone()
    masked[y_true_id] = -float("inf")
    d_id = int(masked.argmax().item())

    # Generate continuation from d*
    gen_ids = [d_id]
    current = torch.cat(
        [tokens.input_ids, torch.tensor([[d_id]], device=device)], dim=1
    )
    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(current)
        nid = int(out.logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gen_ids.append(nid)
        current = torch.cat([current, torch.tensor([[nid]], device=device)], dim=1)

    phrase = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Quality filter
    if len(phrase) < 2 or all(c in ".,;:!?-'\"()[]{} " for c in phrase):
        return None
    # Skip if the decoded phrase contains obvious garbage patterns
    if phrase.count("�") > 0:  # replacement character
        return None
    return phrase


def decode_top_distractor(
    logits: torch.Tensor, y_true_id: int | None, tokenizer
) -> str | None:
    """Decode the top distractor token (single token, fast fallback)."""
    if y_true_id is None or y_true_id >= len(logits):
        return None
    masked = logits.clone()
    masked[y_true_id] = -float("inf")
    d_id = int(masked.argmax().item())
    text = tokenizer.decode([d_id], skip_special_tokens=True).strip()
    if len(text) < 2 or all(c in ".,;:!?-'\"()[]{}" for c in text):
        return None
    return text


def decode_top_k_distractors(
    logits: torch.Tensor, y_true_id: int | None, tokenizer, k: int = 3
) -> list[str]:
    """Decode top-k distractor tokens to text."""
    if y_true_id is None or y_true_id >= len(logits):
        return []
    masked = logits.clone()
    masked[y_true_id] = -float("inf")
    top_k_ids = masked.argsort(descending=True)[:k].tolist()
    texts = []
    for tid in top_k_ids:
        text = tokenizer.decode([tid], skip_special_tokens=True).strip()
        if len(text) >= 2 and not all(c in ".,;:!?-'\"()[]{}" for c in text):
            texts.append(text)
    return texts


# ── Conflict injection ────────────────────────────────────────────────────────


def make_conflict_prompts(
    question: str,
    original_context: str,
    dstar_text: str,
) -> list[tuple[str, str]]:
    """Generate conflicting prompt variants with 3 strength levels.

    Returns list of (variant_label, full_prompt) tuples.
    Labels: weak_0/1/2, medium_0/1/2, strong_0/1/2
    """
    variants = []
    for strength in ["weak", "medium", "strong"]:
        for i, template in enumerate(CONFLICT_TEMPLATES_BY_STRENGTH[strength]):
            misleading = template.format(dstar=dstar_text)
            if original_context.strip():
                combined = misleading + " " + original_context.strip()
            else:
                combined = misleading
            prompt = format_prompt(question, combined, dataset="triviaqa")
            variants.append((f"{strength}_{i}", prompt))
    return variants


# ── Sample evaluation ─────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_sample(
    model, tokenizer, prompt: str, answers: list[str], device: str, max_new: int = 20
) -> dict:
    """Run forward pass + generation on one sample, return all metrics.

    Returns dict with:
      logits_last, y_true_id, rank, generated, is_correct, category, delta, g_tstar, g_dstar
    """
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    output = model(**tokens)
    logits_last = output.logits[0, -1, :].float().detach().cpu()

    y_true_id = get_first_answer_token_id(tokenizer, answers)
    generated = generate_answer(model, tokenizer, prompt, device, max_new)
    is_correct = check_correct(generated, answers, dataset="triviaqa")

    # Rank of y_true
    if y_true_id is not None and y_true_id < len(logits_last):
        rank = int((logits_last > logits_last[y_true_id]).sum().item()) + 1
        # Compute delta
        g = logits_last.numpy()  # no ref subtraction needed for rank/delta
        delta_val = compute_scalar_delta(g, y_true_id)
        g_tstar = float(g[y_true_id])
        g_masked = g.copy()
        g_masked[y_true_id] = -np.inf
        g_dstar = float(g[int(np.argmax(g_masked))])
    else:
        rank = 999999
        delta_val = 0.0
        g_tstar = 0.0
        g_dstar = 0.0

    # Category
    if is_correct:
        category = "KC"
    elif y_true_id is None:
        category = "DK"
    elif rank <= RANK_THRESHOLD:
        category = "KW"
    else:
        category = "DK"

    return {
        "logits_last": logits_last,
        "y_true_id": y_true_id,
        "rank": rank,
        "generated": generated,
        "is_correct": is_correct,
        "category": category,
        "delta": delta_val,
        "g_tstar": g_tstar,
        "g_dstar": g_dstar,
    }


# ── Main synthesis pipeline ────────────────────────────────────────────────────


def synthesize_kw_conflict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = args.model_path or _find_model_path()
    print(f"KW Synthesis via Knowledge Conflict | model={MODEL_PATH}")
    print(f"  n_samples={args.n_samples}, n_conflict={args.n_conflict}")

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    ref_layer, last_layer = _get_delta_layers(model)
    print(
        f"  Model: {model.config.num_hidden_layers} layers, "
        f"d={model.config.hidden_size}, ref=L{ref_layer}, last=L{last_layer}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load samples ────────────────────────────────────────────────────
    print(f"\n[2/5] Loading {args.n_samples} TriviaQA train samples...")
    all_samples = load_triviaqa_train(n_samples=args.n_samples, seed=args.seed)
    # Take first n_conflict samples for conflict injection (to limit runtime)
    conflict_samples = all_samples[: args.n_conflict]
    print(f"  Will attempt conflict injection on {len(conflict_samples)} samples")

    # ── 3. Baseline evaluation + conflict injection ────────────────────────
    print(f"\n[3/5] Baseline eval + conflict injection...")

    summary = {
        "n_total": len(all_samples),
        "n_conflict_tested": len(conflict_samples),
        "n_kc_baseline": 0,
        "n_kw_baseline": 0,
        "n_dk_baseline": 0,
        "n_conflict_attempted": 0,  # samples where d* text was valid
        "n_synthetic_kw": 0,  # conflict turned KC into KW
        "n_conflict_becomes_dk": 0,  # conflict pushed rank > 50
        "n_conflict_stays_kc": 0,  # conflict didn't affect generation
    }

    baseline_results = []  # per-sample: full evaluate_sample dict + question/answers/context
    synthetic_kw = []  # successfully synthesized KW samples
    all_conflict_results = []  # all conflict injection attempts (for analysis)

    for s in tqdm(conflict_samples, desc="Baseline"):
        question = s["question"]
        answers = s["answers"]
        original_context = s.get("context", "")

        prompt_baseline = format_prompt(question, original_context, dataset="triviaqa")
        base = evaluate_sample(model, tokenizer, prompt_baseline, answers, device)

        baseline_results.append(
            {
                "sample": s,
                "base": base,  # full (tensors ok, converted at save time)
            }
        )

        if base["category"] == "KC":
            summary["n_kc_baseline"] += 1
        elif base["category"] == "KW":
            summary["n_kw_baseline"] += 1
        else:
            summary["n_dk_baseline"] += 1

    # ── 4. Conflict injection on KC samples ────────────────────────────────
    print(f"\n[4/5] Injecting conflicts on {summary['n_kc_baseline']} KC samples...")

    for entry in tqdm(baseline_results, desc="Conflict"):
        s = entry["sample"]
        base = entry["base"]
        question = s["question"]
        answers = s["answers"]
        original_context = s.get("context", "")

        # Only inject conflict on KC samples with a valid distractor
        if base["category"] != "KC":
            continue
        if base["y_true_id"] is None:
            continue

        # Multi-token distractor phrase (more meaningful than single token)
        prompt_clean = format_prompt(question, original_context, dataset="triviaqa")
        dstar_text = decode_distractor_phrase(
            model, tokenizer, prompt_clean, base["y_true_id"], device, max_tokens=8
        )
        # Fallback to single token if multi-token fails
        if dstar_text is None:
            dstar_text = decode_top_distractor(
                base["logits_last"], base["y_true_id"], tokenizer
            )
        if dstar_text is None:
            continue

        summary["n_conflict_attempted"] += 1

        # Generate conflict prompts with 3 strength levels
        conflict_prompts = make_conflict_prompts(question, original_context, dstar_text)

        # Test each conflict variant; track best outcome PER SAMPLE and PER STRENGTH
        sample_success = False
        sample_best_category = "KC"
        sample_best_strength = None
        for variant_label, prompt in conflict_prompts:
            result = evaluate_sample(model, tokenizer, prompt, answers, device)
            result["variant"] = variant_label
            result["dstar_text"] = dstar_text
            result["rank_before"] = base["rank"]
            result["delta_before"] = base["delta"]
            result["category_before"] = base["category"]

            all_conflict_results.append(result)

            if result["category"] == "KW":
                sample_success = True
                sample_best_category = "KW"
                strength = variant_label.split("_")[0]  # weak/medium/strong
                sample_best_strength = strength
                # Get the template used
                template_idx = int(variant_label.split("_")[1])
                template_used = CONFLICT_TEMPLATES_BY_STRENGTH[strength][template_idx]
                synthetic_kw.append(
                    {
                        "question": question,
                        "answers": answers,
                        "context": original_context,
                        "conflict_context": template_used.format(dstar=dstar_text),
                        "variant": variant_label,
                        "strength": strength,
                        "dstar_text": dstar_text,
                        "rank_before": base["rank"],
                        "rank_after": result["rank"],
                        "delta_before": round(base["delta"], 4),
                        "delta_after": round(result["delta"], 4),
                        "generated_after": result["generated"][:120],
                    }
                )
                break  # One successful KW per sample is enough
            elif result["category"] == "DK":
                sample_best_category = "DK"

        # Per-sample counting (not per-variant)
        if sample_success:
            summary["n_synthetic_kw"] += 1
            summary.setdefault("kw_by_strength", {})
            summary["kw_by_strength"][sample_best_strength] = (
                summary["kw_by_strength"].get(sample_best_strength, 0) + 1
            )
        elif sample_best_category == "DK":
            summary["n_conflict_becomes_dk"] += 1
        else:
            summary["n_conflict_stays_kc"] += 1

    # ── 5. Report + Save ────────────────────────────────────────────────────
    print(f"\n[5/5] Results\n{'=' * 60}")

    n_kc = summary["n_kc_baseline"]
    n_attempted = summary["n_conflict_attempted"]
    n_synth = summary["n_synthetic_kw"]
    n_dk = summary["n_conflict_becomes_dk"]
    n_kc_stay = summary["n_conflict_stays_kc"]

    print(
        f"\n  Baseline: {summary['n_kc_baseline']} KC, "
        f"{summary['n_kw_baseline']} KW, {summary['n_dk_baseline']} DK"
    )
    print(f"  Conflict attempted: {n_attempted}/{n_kc} KC samples (had valid d* token)")
    print(f"  Synthetic KW: {n_synth} ({n_synth / max(1, n_attempted) * 100:.1f}%)")
    print(f"  Became DK:     {n_dk} ({n_dk / max(1, n_attempted) * 100:.1f}%)")
    print(
        f"  Stayed KC:     {n_kc_stay} ({n_kc_stay / max(1, n_attempted) * 100:.1f}%)"
    )

    # Analyze delta shifts
    if all_conflict_results:
        kw_results = [r for r in all_conflict_results if r["category"] == "KW"]
        if kw_results:
            deltas_after = [r["delta"] for r in kw_results]
            deltas_before = [r["delta_before"] for r in kw_results]
            print(f"\n  Synthetic KW delta stats:")
            print(
                f"    Before conflict: 未 = {np.mean(deltas_before):.2f} "
                f"± {np.std(deltas_before):.2f}"
            )
            print(
                f"    After conflict:  未 = {np.mean(deltas_after):.2f} "
                f"± {np.std(deltas_after):.2f}"
            )
            print(
                f"    Mean 未 shift:    {np.mean(deltas_after) - np.mean(deltas_before):+.2f}"
            )

    synthesis_rate = n_synth / max(1, n_attempted)
    print(f"\n  Synthesis rate: {synthesis_rate:.1%}")
    print(f"  Gate (synthesis_rate > 0.05): {'✅' if synthesis_rate > 0.05 else '❌'}")

    # Per-strength breakdown
    if "kw_by_strength" in summary:
        print(f"\n  KW by template strength:")
        for strength in ["weak", "medium", "strong"]:
            n = summary["kw_by_strength"].get(strength, 0)
            print(f"    {strength}: {n} ({n / max(1, n_synth) * 100:.0f}%)")

    # Projection: if n_train=2000 with original KW ratio ~3.5% (~70 natural KW),
    # add synthetic KW from ~60% KC → (2000 * 0.60 * synthesis_rate) extra KW
    projected_natural_kw = 2000 * 0.035  # ~70
    projected_synthetic_kw = 2000 * (n_kc / len(conflict_samples)) * synthesis_rate
    print(f"\n  Projection for n_train=2000:")
    print(f"    Natural KW:  ~{projected_natural_kw:.0f}")
    print(f"    Synthetic KW: ~{projected_synthetic_kw:.0f}")
    print(f"    Total KW:     ~{projected_natural_kw + projected_synthetic_kw:.0f}")

    # ── Save ────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUT_DIR / "synthesize_kw_conflict.json"

    # Convert baseline tensors for JSON serialization
    baseline_json = []
    for entry in baseline_results:
        b = entry["base"]
        baseline_json.append(
            {
                "question": entry["sample"]["question"][:80],
                "category": b["category"],
                "rank": b["rank"],
                "delta": b["delta"],
                "is_correct": b["is_correct"],
                "generated": b["generated"][:120],
            }
        )

    output = {
        "config": {
            "n_samples": args.n_samples,
            "n_conflict": args.n_conflict,
            "seed": args.seed,
            "model": MODEL_PATH,
            "ref_layer": ref_layer,
            "last_layer": last_layer,
        },
        "summary": summary,
        "projection": {
            "n_train_2000_natural_kw": round(projected_natural_kw),
            "n_train_2000_synthetic_kw": round(projected_synthetic_kw),
            "n_train_2000_total_kw": round(
                projected_natural_kw + projected_synthetic_kw
            ),
        },
        "synthetic_kw_samples": synthetic_kw,
        "baseline": baseline_json,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")

    # Also save a plain sample list for train_lora_delta.py consumption
    if synthetic_kw:
        samples_path = OUTPUT_DIR / "synthetic_kw_samples.json"
        with open(samples_path, "w") as f:
            json.dump(synthetic_kw, f, indent=2, ensure_ascii=False)
        print(f"  Synthetic KW samples saved to {samples_path}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize KW samples via knowledge conflict injection"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Total TriviaQA train samples to load",
    )
    parser.add_argument(
        "--n_conflict",
        type=int,
        default=200,
        help="Number of samples to attempt conflict injection on "
        "(first N of total, to limit runtime)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path (default: auto-detect Qwen3-1.7B)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    synthesize_kw_conflict(args)


if __name__ == "__main__":
    main()
