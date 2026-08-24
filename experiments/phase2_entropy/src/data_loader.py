"""Data loading: HellaSwag, TriviaQA, SQuAD."""

import re

from datasets import load_dataset


def load_hellaswag(n_samples: int = 200, seed: int = 42):
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    label_letters = ["A", "B", "C", "D"]
    samples = []
    for item in ds:
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        correct_ending = endings[label]
        label_letter = label_letters[label]
        choices_text = "\n".join(
            f"{label_letters[i]}. {endings[i]}" for i in range(4)
        )
        samples.append({
            "question": ctx,
            "answers": [correct_ending, label_letter],
            "context": choices_text,
        })
    return samples


def load_triviaqa(n_samples: int = 200, seed: int = 42):
    ds = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        question = item["question"]
        answers = item["answer"]["aliases"]
        search_contexts = item["search_results"]["search_context"]
        context = "\n\n".join(ctx for ctx in search_contexts if ctx)
        samples.append({"question": question, "answers": answers, "context": context})
    return samples


def load_squad(n_samples: int = 200, seed: int = 42):
    ds = load_dataset("squad_v2", split="validation", trust_remote_code=False)
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        samples.append({
            "question": item["question"],
            "answers": item["answers"]["text"],
            "context": item["context"],
        })
    return samples


# ── Prompt truncation (code-review-2026-08-24 Critical 1) ────────────────────
# TriviaQA search_context can reach hundreds of KB. Keeping it whole pushed
# prompts past the 1024-token window, and the old head-truncation
# `tokens[:, :1024]` then cut off the Question — which sits at the END of the
# prompt. Fix: truncate the CONTEXT (head paragraphs + char ceiling), never
# the question. Callers keep `tokens[:, -1024:]` as a defense-in-depth net.
_MAX_CONTEXT_PARAS = 3
_MAX_CONTEXT_CHARS = 2400  # ≈600 tokens; keeps full prompt well under 1024


def _truncate_context(context: str) -> str:
    """Keep the first few paragraphs of search context, capped by chars."""
    if not context:
        return context
    paras = [p.strip() for p in context.split("\n\n") if p.strip()]
    ctx = "\n\n".join(paras[:_MAX_CONTEXT_PARAS])
    if len(ctx) > _MAX_CONTEXT_CHARS:
        cut = ctx[:_MAX_CONTEXT_CHARS]
        # Cut at the last whitespace to avoid splitting a word mid-way
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        ctx = cut
    return ctx


def format_prompt(question: str, context: str = "", dataset: str = "hellaswag") -> str:
    if dataset == "hellaswag":
        return (
            f"Complete the sentence with the most natural ending. "
            f"Answer with a single letter A, B, C, or D.\n\n"
            f"Context: {question}\n"
            f"{context}\n\n"
            f"Answer:"
        )
    if dataset == "squad":
        return (
            f"Read the passage and answer the question with a short phrase.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    if context:
        return (
            f"Based on the provided context, answer the question with a single word "
            f"or short phrase.\n\n"
            f"Context: {_truncate_context(context)}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    return (
        f"Answer the question with a single word or short phrase.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def check_correct(prediction: str, answers: list[str], dataset: str = "hellaswag") -> bool:
    pred_lower = prediction.strip().lower()
    if dataset == "hellaswag":
        pred_letter = pred_lower[0] if pred_lower else ""
        label_letter = answers[1].lower()
        return pred_letter == label_letter
    pred_words = set(pred_lower.split())
    for ans in answers:
        ans_lower = ans.lower().strip()
        ans_words = set(ans_lower.split())
        if ans_words & pred_words:
            return True
        if len(pred_lower) >= 3 and len(ans_lower) >= 3:
            if ans_lower in pred_lower or pred_lower in ans_lower:
                return True
    return False


def check_correct_exact(prediction: str, answers: list[str]) -> bool:
    """Exact match with word-boundary semantics (code-review High 3 / Low 7).

    Conservative label: the answer must appear verbatim (case-insensitive) in
    the prediction, either as the whole prediction or as a whitespace/punct
    delimited span. Aliases of length <= 3 skip the span rule to avoid false
    positives like "the" or "paris" ⊂ "parisian".
    """
    pred_lower = prediction.strip().lower()
    for ans in answers:
        ans_lower = ans.strip().lower()
        if not ans_lower:
            continue
        if pred_lower == ans_lower:
            return True
        if len(ans_lower) <= 3:
            continue
        # \w-boundaries: match only when not adjacent to more word chars
        if re.search(rf"(?<![\w]){re.escape(ans_lower)}(?![\w])", pred_lower):
            return True
    return False
