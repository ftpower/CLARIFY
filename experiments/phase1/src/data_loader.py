"""Load QA datasets: TriviaQA (rc), SQuAD v2, and HellaSwag."""

from datasets import load_dataset


def load_triviaqa(n_samples: int = 100, seed: int = 42):
    """Load TriviaQA validation samples (rc config with context passages).

    Returns list of dicts with keys: question, answers (list of aliases), context (str).
    """
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


def load_squad(n_samples: int = 100, seed: int = 42):
    """Load SQuAD v2 validation samples (answerable questions only).

    Returns list of dicts with keys: question, answers (list of answer texts), context (str).
    """
    ds = load_dataset("squad_v2", split="validation", trust_remote_code=False)
    # Keep only answerable questions
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        samples.append(
            {
                "question": item["question"],
                "answers": item["answers"]["text"],
                "context": item["context"],
            }
        )
    return samples


def load_hellaswag(n_samples: int = 100, seed: int = 42):
    """Load HellaSwag validation samples (commonsense sentence completion, 4-choice).

    Returns list of dicts with keys: question (ctx), answers (correct ending + label letter),
    context (A/B/C/D choices formatted).
    """
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    label_letters = ["A", "B", "C", "D"]
    samples = []
    for item in ds:
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])  # 0-3
        correct_ending = endings[label]
        label_letter = label_letters[label]
        choices_text = "\n".join(f"{label_letters[i]}. {endings[i]}" for i in range(4))

        samples.append(
            {
                "question": ctx,
                "answers": [correct_ending, label_letter],
                "context": choices_text,
            }
        )
    return samples


def format_prompt(question: str, context: str = "", dataset: str = "triviaqa") -> str:
    """Format a question with context into a model prompt."""
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
    # Default: TriviaQA instruct format
    if context:
        return (
            f"Based on the provided context, answer the question with a single word "
            f"or short phrase.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    return (
        f"Answer the question with a single word or short phrase.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def check_correct(prediction: str, answers: list[str]) -> bool:
    """Check if prediction matches any ground-truth answer (case-insensitive)."""
    pred_lower = prediction.strip().lower()
    for ans in answers:
        if ans.lower() in pred_lower or pred_lower in ans.lower():
            return True
    return False
