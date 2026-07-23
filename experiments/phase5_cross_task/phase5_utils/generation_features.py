"""Per-token feature extraction during greedy generation.

Core module for Phase 5 cross-task generalization. Adapts halluinacion detection
features from HellaSwag (4-choice MCQ, single forward pass) to TriviaQA (free-text
generation, per-token extraction).

The key challenge: in HellaSwag, we extract features from the LAST INPUT token's
logit-lens distribution over 4 fixed answer letters. In TriviaQA, the model
generates free text token-by-token, so we must extract features at EACH decode step.

Features extracted per token:
  - max_p[i]:  max softmax probability at layer i (via logit lens)
  - entropy[i]: logit-lens entropy at layer i
  - d2_js:      JS divergence between L_early and L_late full-vocab distributions

This is the analog of Phase 4's D2 JS (4-choice) but adapted to full vocabulary.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


def compute_js_vocab_full(
    probs_early: torch.Tensor,
    probs_late: torch.Tensor,
    eps: float = 1e-10,
) -> float:
    """JS divergence between two full-vocabulary softmax distributions.

    This is the TriviaQA analog of compute_d2_js_score() which operates on
    4-choice softmax. The formula is identical but the dimension changes from
    4 to vocab_size (~152K).

    Mathematical stability:
      - clamp(min=eps) prevents log(0) = -inf
      - renormalization after clamp preserves sum=1 property
      - all computation in float32 to avoid fp16 precision loss in log ops

    Args:
        probs_early: [vocab_size] softmax-normalized distribution at early layer.
        probs_late:  [vocab_size] softmax-normalized distribution at late layer.
        eps: clamp floor for numerical stability (default 1e-10).

    Returns:
        JS divergence (float), always >= 0.
    """
    # Clamp away zeros and renormalize
    p = probs_early.float().clamp(min=eps)
    q = probs_late.float().clamp(min=eps)
    p = p / p.sum()
    q = q / q.sum()

    # Midpoint distribution
    m = 0.5 * (p + q)

    # KL divergences: KL(P||M) and KL(Q||M)
    # Uses p * (log p - log m) = p * log(p/m) for numerical stability
    kl_p = (p * (p.log() - m.log())).sum()
    kl_q = (q * (q.log() - m.log())).sum()

    # JS = 0.5 * (KL(P||M) + KL(Q||M))
    return float(0.5 * (kl_p.detach() + kl_q.detach()))


def generate_with_per_token_features(
    model: HookedTransformer,
    prompt: str,
    W_U: torch.Tensor,
    b_U: torch.Tensor | None = None,
    max_new_tokens: int = 20,
    js_early_layer: int = 11,
    js_late_layer: int = 27,
    temperature: float = 1.0,
) -> dict:
    """Greedy decode while extracting per-token logit-lens features at every step.

    At each decode step, hooks all transformer layers to get their residual stream
    at the last sequence position. Projects each residual through the unembedding
    matrix (logit lens) to get a full-vocabulary softmax distribution. From this
    extracts:
      - max_p:   per-layer maximum probability
      - entropy: per-layer entropy
      - d2_js:   JS divergence between js_early_layer and js_late_layer

    The last decode step additionally saves FULL vocabulary distributions for
    all layers, enabling downstream all-pair JS scanning.

    Args:
        model: HookedTransformer instance in eval mode.
        prompt: Input prompt text (will be tokenized with prepend_bos=True).
        W_U: Unembedding matrix [d_model, vocab_size].
        b_U: Unembedding bias [vocab_size] or None.
        max_new_tokens: Max tokens to generate (default 20).
        js_early_layer: Early layer index for JS divergence (default 11).
        js_late_layer: Late layer index for JS divergence (default 27).
        temperature: Softmax temperature for logit lens (default 1.0).

    Returns:
        dict with keys:
            answer_text: str — generated answer (decoded, no prompt).
            answer_token_ids: list[int] — generated token IDs.
            n_tokens: int — number of generated tokens.
            per_token: list[dict] — one entry per generated token:
                {
                    "step": int (0-indexed),
                    "token_id": int,
                    "token_text": str,
                    "max_p": list[float],      # [n_layers]
                    "entropy": list[float],    # [n_layers]
                    "d2_js": float,            # JS(L_early, L_late)
                }
            last_token_vocab_probs: list[torch.Tensor] — [n_layers] full-vocab
                softmax distributions at the FINAL generated token position.
                Each tensor has shape [vocab_size]. None if no tokens generated.
    """
    n_layers = model.cfg.n_layers
    ln_final = model.ln_final
    device = next(model.parameters()).device

    # Tokenize prompt
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    per_token_features = []
    generated_ids = []

    for step in range(max_new_tokens):
        # ── Register hooks at all layers ──────────────────────────────
        residuals = {}

        def _make_hook(name):
            def hook(act, hook):
                # Capture residual at the LAST sequence position
                residuals[name] = act[:, -1, :].detach()

            return hook

        fwd_hooks = [
            (f"blocks.{i}.hook_resid_post", _make_hook(f"L{i}"))
            for i in range(n_layers)
        ]

        # ── Forward pass ──────────────────────────────────────────────
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        # ── Per-layer logit lens ──────────────────────────────────────
        max_p_vals = []
        entropy_vals = []
        probs_early = None
        probs_late = None

        for i in range(n_layers):
            h = residuals[f"L{i}"].to(device)  # [1, d_model]

            # Apply ln_final only to the final layer (standard logit lens convention)
            if i == n_layers - 1:
                h = ln_final(h)

            logits_L = h @ W_U  # [1, vocab_size]
            if b_U is not None:
                logits_L = logits_L + b_U

            # Softmax with temperature
            if temperature != 1.0:
                logits_L = logits_L / temperature

            probs_L = torch.softmax(logits_L, dim=-1)  # [1, vocab_size]
            log_probs_L = torch.log_softmax(logits_L, dim=-1)

            # Extract max_p and entropy
            max_p = probs_L.max(dim=-1).values.item()
            entropy = -(probs_L * log_probs_L).sum(dim=-1).item()

            max_p_vals.append(max_p)
            entropy_vals.append(entropy)

            # Save full-vocab probs for JS target layers
            if i == js_early_layer:
                probs_early = probs_L[0]  # [vocab_size]
            if i == js_late_layer:
                probs_late = probs_L[0]  # [vocab_size]

        # ── D2 JS (early vs late layer) ───────────────────────────────
        d2_js = compute_js_vocab_full(probs_early, probs_late)

        # ── Greedy next token ─────────────────────────────────────────
        next_id = logits[0, -1, :].argmax(dim=-1).item()
        token_text = model.tokenizer.decode(next_id)

        per_token_features.append({
            "step": step,
            "token_id": next_id,
            "token_text": token_text,
            "max_p": max_p_vals,
            "entropy": entropy_vals,
            "d2_js": d2_js,
        })
        generated_ids.append(next_id)

        # ── Check EOS ─────────────────────────────────────────────────
        if next_id == model.tokenizer.eos_token_id:
            # Extract full-vocab probs for ALL layers at this final position
            last_token_vocab_probs = _extract_all_layer_vocab_probs(
                residuals, ln_final, W_U, b_U, n_layers, device, temperature
            )
            break

        # Append token for next iteration
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=device)], dim=-1
        )
    else:
        # max_new_tokens reached (no EOS)
        # Extract full-vocab probs from the last decode step
        if per_token_features:
            last_token_vocab_probs = _extract_all_layer_vocab_probs(
                residuals, ln_final, W_U, b_U, n_layers, device, temperature
            )
        else:
            last_token_vocab_probs = None

    # ── Decode answer text ────────────────────────────────────────────
    if generated_ids:
        answer_text = model.tokenizer.decode(generated_ids).strip()
    else:
        answer_text = ""

    return {
        "answer_text": answer_text,
        "answer_token_ids": generated_ids,
        "n_tokens": len(generated_ids),
        "per_token": per_token_features,
        "last_token_vocab_probs": last_token_vocab_probs,
    }


def _extract_all_layer_vocab_probs(
    residuals: dict,
    ln_final,
    W_U: torch.Tensor,
    b_U: torch.Tensor | None,
    n_layers: int,
    device: torch.device,
    temperature: float = 1.0,
) -> list[torch.Tensor]:
    """Extract full-vocabulary softmax for ALL layers at the current position.

    Used once per sample (at the final generated token) to enable downstream
    all-pair JS scanning.

    Args:
        residuals: dict mapping "L{i}" to [1, d_model] tensors on device.
        ln_final: model.ln_final for the last layer.
        W_U: unembedding matrix.
        b_U: unembedding bias.
        n_layers: number of transformer layers.
        device: target device.
        temperature: softmax temperature.

    Returns:
        List of [vocab_size] float32 tensors on CPU, length n_layers.
    """
    vocab_probs = []
    for i in range(n_layers):
        h = residuals[f"L{i}"].to(device)
        if i == n_layers - 1:
            h = ln_final(h)

        logits_L = h @ W_U  # [1, vocab_size]
        if b_U is not None:
            logits_L = logits_L + b_U

        if temperature != 1.0:
            logits_L = logits_L / temperature

        probs_L = torch.softmax(logits_L, dim=-1)  # [1, vocab_size]
        vocab_probs.append(probs_L[0].float().cpu())

    return vocab_probs


def compute_all_pair_js(
    vocab_probs: list[torch.Tensor],
    n_layers: int,
    exclude_layer0: bool = True,
) -> dict:
    """Compute JS divergence for all layer pairs from full-vocab distributions.

    This is the full-vocab analog of compute_d2_js_topk(). Scans all (i, j)
    pairs where i < j and computes per-pair JS.

    Args:
        vocab_probs: list of [vocab_size] float32 tensors, length n_layers.
        n_layers: number of transformer layers.
        exclude_layer0: if True, skip L0 as early layer (default True).

    Returns:
        dict mapping "(early,late)" strings to float JS values.
        Example: {"(11,27)": 0.0456, "(0,15)": 0.0234, ...}
    """
    start_early = 1 if exclude_layer0 else 0
    pair_js = {}

    for early in range(start_early, n_layers):
        for late in range(early + 1, n_layers):
            js = compute_js_vocab_full(vocab_probs[early], vocab_probs[late])
            pair_js[f"({early},{late})"] = js

    return pair_js
