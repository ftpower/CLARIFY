"""Common utilities for LIN theory validation (Phase A).

Provides model loading, truth direction computation, analytical gradient
computation, and generation helpers. All scripts in this directory depend on this.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import load_triviaqa, format_prompt, check_correct


# ═════════════════════════════════════════════════════════════════════════════
# 1. Model loading with unembedding attributes
# ═════════════════════════════════════════════════════════════════════════════


def load_model_and_unembed(device="cuda", model_id="Qwen/Qwen3-1.7B"):
    """Load HookedTransformer + return (model, tokenizer, W_U, b_U, ln_final)."""
    model = load_model(device=device, model_id=model_id)
    tokenizer = model.tokenizer
    W_U = model.unembed.W_U  # [d_model, vocab_size], float16 on device
    b_U = model.unembed.b_U  # [vocab_size] on device, or None
    ln_final = model.ln_final  # RMSNorm module
    return model, tokenizer, W_U, b_U, ln_final


# ═════════════════════════════════════════════════════════════════════════════
# 2. Truth direction v = mean(correct) - mean(wrong)
# ═════════════════════════════════════════════════════════════════════════════


def compute_v(model, tokenizer, n_calibrate, device, layer, seed=42):
    """Compute truth direction v from calibration samples at a specific layer.

    Reuses the exact algorithm from D_intervention_lean.py:
      1. Generate answers for calibration samples
      2. Label correct/wrong via check_correct
      3. Extract hidden state at given layer (last token position)
      4. v = mean(h_correct) - mean(h_incorrect), L2-normalized

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        n_calibrate: number of TriviaQA samples for calibration
        device: "cuda" or "cpu"
        layer: layer index to extract hidden states from
        seed: random seed for TriviaQA sample selection (default 42)

    Returns:
        v: torch.Tensor [d_model] on device, float32, unit-norm
        stats: dict with n_correct, n_incorrect, v_norm
    """
    samples = load_triviaqa(n_samples=n_calibrate, seed=seed)
    h_correct = []
    h_incorrect = []

    hook_name = f"blocks.{layer}.hook_resid_post"

    for s in tqdm(samples, desc=f"Calibrate v L{layer}"):
        prompt = format_prompt(s["question"], s["context"], dataset="triviaqa")
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]

        residual = {}

        def _hook(act, hook=None):
            residual["h"] = act[:, -1, :].detach()
            return act

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

        nid = int(logits[0, -1, :].argmax().item())
        gids = [nid]
        for _ in range(19):
            if nid == tokenizer.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")

        h_vec = residual["h"].float().cpu().numpy().flatten()
        if is_correct:
            h_correct.append(h_vec)
        else:
            h_incorrect.append(h_vec)

    n_correct = len(h_correct)
    n_incorrect = len(h_incorrect)

    if n_correct == 0 or n_incorrect == 0:
        raise RuntimeError(
            f"Need at least 1 correct and 1 incorrect sample, "
            f"got {n_correct}/{n_incorrect}"
        )

    h_correct = np.stack(h_correct, axis=0)
    h_incorrect = np.stack(h_incorrect, axis=0)
    v = h_correct.mean(axis=0) - h_incorrect.mean(axis=0)
    v_norm = np.linalg.norm(v)
    v = v / v_norm

    stats = {
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "v_norm_raw": float(v_norm),
    }

    return torch.from_numpy(v).float().to(device), stats


# ═════════════════════════════════════════════════════════════════════════════
# 3. Answer token ID extraction
# ═════════════════════════════════════════════════════════════════════════════


def get_first_answer_token_id(tokenizer, answers):
    """Return the first token ID of the first non-empty answer alias."""
    for ans in answers:
        tokens = tokenizer.encode(ans.lower().strip(), add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 4. Exact gradient g_L = ∇_{h_L} log P(y_true | h_L)
# ═════════════════════════════════════════════════════════════════════════════


def compute_g_L(h_L, y_true_token_id, W_U, b_U, ln_final):
    """Analytical gradient of log-prob of y_true w.r.t. last-layer hidden state.

    Uses the analytical formula (theory doc Section 3.2, without RMSNorm
    Jacobian) to avoid autograd memory overhead on 8GB GPU:

        g_L = W_U^T @ (e_ytrue - softmax(W_U @ RMSNorm(h_L)))

    This omits the RMSNorm Jacobian, which is an acceptable approximation
    for Phase A validation — the gradient direction is dominated by the
    unembedding projection, not the normalization.

    Args:
        h_L: [d_model] or [1, d_model] tensor, float16 on GPU (detached).
        y_true_token_id: int — first token ID of the ground-truth answer.
        W_U: [d_model, vocab_size] on device (float16).
        b_U: [vocab_size] on device (float16), or None.
        ln_final: RMSNorm module on device (float16).

    Returns:
        g_L: torch.Tensor [d_model] on CPU float32.
    """
    if h_L.ndim == 1:
        h_L = h_L.unsqueeze(0)  # [1, d_model]

    dtype = next(ln_final.parameters()).dtype  # model dtype (float16)
    device = h_L.device

    with torch.no_grad():
        # Forward through ln_final (no autograd needed for analytical g)
        h_f16 = h_L.to(dtype=dtype)
        h_norm = ln_final(h_f16)  # [1, d_model] float16

        # Cast to float32 for softmax precision (safely freeable temp)
        h_norm_f32 = h_norm.float()
        W_U_f32 = W_U.float()  # ~1.2 GB temporary

        logits = h_norm_f32 @ W_U_f32  # [1, vocab_size] float32
        if b_U is not None:
            logits = logits + b_U.float()

        probs = torch.softmax(logits, dim=-1)

        # Build one-hot error signal
        vocab_size = W_U.shape[1]
        target_oh = torch.zeros(1, vocab_size, device=device, dtype=torch.float32)
        target_oh[0, y_true_token_id] = 1.0

        error = target_oh - probs  # [1, vocab_size] float32

        # g_L = W_U @ error^T = [d_model, 1]
        g_L = (W_U_f32 @ error.t()).squeeze(1)  # [d_model] float32

        # Free large temporary tensors
        del W_U_f32, logits, probs, error

    return g_L.cpu()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Greedy generation with optional single-hook intervention
# ═════════════════════════════════════════════════════════════════════════════


def greedy_generate(
    model,
    tokenizer,
    prompt,
    device,
    fwd_hooks=None,
    max_new=20,
    return_full=False,
):
    """Greedy generation with optional forward hooks on first step.

    Args:
        model: HookedTransformer
        tokenizer: model tokenizer
        prompt: string
        device: "cuda" or "cpu"
        fwd_hooks: list of (hook_name, hook_fn) tuples, or None.
                   Applied on the FIRST forward pass only.
        max_new: max tokens to generate.
        return_full: if True, return (generated_text, initial_logits).

    Returns:
        generated_text: str — decoded generated tokens.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    hooks = fwd_hooks if fwd_hooks else []

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    initial_logits = logits.detach().clone()

    nid = int(logits[0, -1, :].argmax().item())
    gids = [nid]

    for _ in range(max_new - 1):
        if nid == tokenizer.eos_token_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        with torch.no_grad():
            logits = model(tokens)
        nid = int(logits[0, -1, :].argmax().item())
        gids.append(nid)

    ans = tokenizer.decode(gids).strip()

    if return_full:
        return ans, initial_logits
    return ans


# ═════════════════════════════════════════════════════════════════════════════
# 6. Hook extraction helper: capture h_l at last token position
# ═════════════════════════════════════════════════════════════════════════════


def extract_h_at_layer(model, tokenizer, prompt, device, layer):
    """Extract hidden state at a specific layer's resid_post, last token position.

    Returns:
        h: torch.Tensor [1, d_model] on device (float16, detached).
        logits: torch.Tensor [1, 1, vocab_size] on device.
        tokens: the token tensor used for the forward pass.
        last_pos: int — index of the last token position.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    last_pos = tokens.shape[1] - 1

    residual = {}
    hook_name = f"blocks.{layer}.hook_resid_post"

    def _hook(act, hook=None):
        residual["h"] = act[:, -1, :].detach()
        return act

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

    return residual["h"], logits, tokens, last_pos
