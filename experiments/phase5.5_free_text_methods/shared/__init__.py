# Shared data loading for Phase 5.5 experiments
"""Unified TriviaQA data loading used by all Phase 5.5 methods.

Provides:
  - load_triviaqa_samples(): load raw samples with fixed seed
  - load_model_and_data(): convenience: load model + samples in one call

All methods use the SAME 200 samples (seed=42) as Phase 5.1.
"""

import os
import sys
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_sys_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_sys_parent / "phase2_entropy"))
sys.path.insert(0, str(_sys_parent / "phase5_cross_task"))

from src.model_loader import load_model as _load_model
from src.data_loader import load_triviaqa as _load_triviaqa, format_prompt, check_correct


def load_triviaqa_samples(n_samples: int = 200, seed: int = 42) -> list[dict]:
    """Load TriviaQA samples with fixed seed. Same as Phase 5.1."""
    return _load_triviaqa(n_samples=n_samples, seed=seed)


def load_model_and_data(
    n_samples: int = 200,
    seed: int = 42,
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
):
    """Convenience: load model and TriviaQA samples in one call.

    Returns:
        model: HookedTransformer
        samples: list[dict]
        W_U: unembedding matrix
        b_U: unembedding bias (or None)
    """
    print(f"Loading model {model_id}...")
    model = _load_model(device=device, model_id=model_id)
    W_U = model.unembed.W_U.to(device)
    b_U = model.unembed.b_U
    if b_U is not None:
        b_U = b_U.to(device)
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    print(f"Loading TriviaQA ({n_samples} samples)...")
    samples = load_triviaqa_samples(n_samples=n_samples, seed=seed)
    print(f"  Loaded {len(samples)} samples")

    return model, samples, W_U, b_U
