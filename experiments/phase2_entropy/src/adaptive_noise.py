"""Adaptive noise injection — detection-triggered intervention via TransformerLens hooks."""

import math

import torch
from transformer_lens import HookedTransformer


def hidden_state_index_to_hook(idx: int, n_layers: int) -> str:
    """Map hidden_states index to TransformerLens hook point.

    hidden_states layout (len = n_layers + 1):
      idx 0         = embedding → blocks.0.hook_resid_pre
      idx 1..n_layers-1 = after block i-1 → blocks.{i-1}.hook_resid_post
      idx n_layers  = ln_final(residual after last block) → blocks.{n_layers-1}.hook_resid_post
    """
    if idx == 0:
        return "blocks.0.hook_resid_pre"
    if idx <= n_layers - 1:
        return f"blocks.{idx - 1}.hook_resid_post"
    return f"blocks.{n_layers - 1}.hook_resid_post"


def make_noise_hook(std: float):
    """Return a hook that adds Gaussian noise N(0, std^2) to activations."""

    def hook(activation: torch.Tensor, hook) -> torch.Tensor:
        return activation + torch.randn_like(activation) * std

    return hook


def compute_adaptive_sigma(
    max_p: float, threshold: float, sigma_base: float, alpha: float
) -> float:
    """Compute adaptive noise std: σ = σ_base × exp(α × (max_p − threshold)).

    Only called when max_p > threshold (overconfident).
    """
    return sigma_base * math.exp(alpha * (max_p - threshold))


def detect_per_layer_max_p(
    model: HookedTransformer,
    prompt: str,
    W_U: torch.Tensor,
    b_U: torch.Tensor | None,
) -> list[float]:
    """Round 1: extract per-layer hidden states and compute max_p for each layer.

    Returns list of max_p values, length n_layers+1.
    """
    from .hidden_state import extract_hidden_states

    hidden_states, _, _, _ = extract_hidden_states(model, prompt)
    device = W_U.device
    max_probs = []
    for h in hidden_states:
        h_dev = h.to(device)
        logits = h_dev @ W_U
        if b_U is not None:
            logits = logits + b_U
        probs = torch.softmax(logits, dim=-1)
        max_probs.append(probs.max().item())
    return max_probs


def generate_with_hooks(
    model: HookedTransformer,
    prompt: str,
    fwd_hooks: list,
    max_new_tokens: int = 20,
) -> str:
    """Multi-token greedy decode with hooks active. Returns answer text (no prompt)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    prompt_len = tokens.shape[1]

    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(tokens)
                next_id = logits[0, -1, :].argmax(dim=-1)
                tokens = torch.cat([tokens, next_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                if next_id.item() == model.tokenizer.eos_token_id:
                    break

    new_ids = tokens[0, prompt_len:]
    return model.tokenizer.decode(new_ids).strip()


def generate_with_noise_hook(
    model: HookedTransformer,
    prompt: str,
    inject_idx: int,
    sigma: float,
    n_layers: int,
    max_new_tokens: int = 20,
) -> str:
    """Generate answer with Gaussian noise injected at a single layer."""
    hook_point = hidden_state_index_to_hook(inject_idx, n_layers)
    noise_hook = make_noise_hook(sigma)
    return generate_with_hooks(
        model, prompt, [(hook_point, noise_hook)], max_new_tokens
    )


def generate_with_per_layer_noise(
    model: HookedTransformer,
    prompt: str,
    per_layer_sigmas: dict[int, float],
    n_layers: int,
    max_new_tokens: int = 20,
) -> str:
    """Generate answer with per-layer adaptive noise. per_layer_sigmas maps idx→sigma."""
    hooks = []
    for idx, sigma in per_layer_sigmas.items():
        hook_point = hidden_state_index_to_hook(idx, n_layers)
        hooks.append((hook_point, make_noise_hook(sigma)))
    return generate_with_hooks(model, prompt, hooks, max_new_tokens)


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


def run_strategy_A(
    model,
    prompt,
    answers,
    dataset,
    W_U,
    b_U,
    n_layers,
    threshold,
    sigma_base,
    alpha,
) -> dict:
    """Strategy A: L28 detect → L28 inject (self-loop)."""
    detect_idx = n_layers  # last layer
    inject_idx = n_layers

    max_probs = detect_per_layer_max_p(model, prompt, W_U, b_U)
    max_p = max_probs[detect_idx]
    triggered = max_p > threshold

    if triggered:
        sigma = compute_adaptive_sigma(max_p, threshold, sigma_base, alpha)
        gen_text = generate_with_noise_hook(model, prompt, inject_idx, sigma, n_layers)
    else:
        gen_text = generate_with_hooks(model, prompt, [])

    from .data_loader import check_correct

    is_correct = check_correct(gen_text, answers, dataset=dataset)

    return {
        "triggered": triggered,
        "max_p_detect": max_p,
        "sigma": sigma if triggered else 0.0,
        "generated_text": gen_text,
        "is_correct": is_correct,
    }


def run_strategy_B(
    model,
    prompt,
    answers,
    dataset,
    W_U,
    b_U,
    n_layers,
    threshold,
    sigma_base,
    alpha,
) -> dict:
    """Strategy B: L28 detect → L11 inject."""
    detect_idx = n_layers  # L28
    inject_idx = 11  # L11 (wAUROC/entropy peak layer for 1.7B)

    max_probs = detect_per_layer_max_p(model, prompt, W_U, b_U)
    max_p = max_probs[detect_idx]
    triggered = max_p > threshold

    if triggered:
        sigma = compute_adaptive_sigma(max_p, threshold, sigma_base, alpha)
        gen_text = generate_with_noise_hook(model, prompt, inject_idx, sigma, n_layers)
    else:
        gen_text = generate_with_hooks(model, prompt, [])

    from .data_loader import check_correct

    is_correct = check_correct(gen_text, answers, dataset=dataset)

    return {
        "triggered": triggered,
        "max_p_detect": max_p,
        "sigma": sigma if triggered else 0.0,
        "generated_text": gen_text,
        "is_correct": is_correct,
    }


def run_strategy_C(
    model,
    prompt,
    answers,
    dataset,
    W_U,
    b_U,
    n_layers,
    threshold,
    sigma_base,
    alpha,
) -> dict:
    """Strategy C: per-layer adaptive — noise at each layer where max_p > threshold."""
    max_probs = detect_per_layer_max_p(model, prompt, W_U, b_U)

    per_layer_sigmas = {}
    for idx, max_p in enumerate(max_probs):
        if max_p > threshold:
            sigma = compute_adaptive_sigma(max_p, threshold, sigma_base, alpha)
            if sigma > 1e-8:
                per_layer_sigmas[idx] = sigma

    if per_layer_sigmas:
        gen_text = generate_with_per_layer_noise(
            model,
            prompt,
            per_layer_sigmas,
            n_layers,
        )
    else:
        gen_text = generate_with_hooks(model, prompt, [])

    from .data_loader import check_correct

    is_correct = check_correct(gen_text, answers, dataset=dataset)

    return {
        "triggered": len(per_layer_sigmas) > 0,
        "n_layers_triggered": len(per_layer_sigmas),
        "triggered_layers": list(per_layer_sigmas.keys()),
        "max_p_per_layer": max_probs,
        "generated_text": gen_text,
        "is_correct": is_correct,
    }


def run_fixed_sigma(
    model,
    prompt,
    answers,
    dataset,
    inject_idx,
    sigma,
    n_layers,
) -> dict:
    """Phase 5: fixed σ noise at a specific layer (no detection)."""
    gen_text = generate_with_noise_hook(model, prompt, inject_idx, sigma, n_layers)

    from .data_loader import check_correct

    is_correct = check_correct(gen_text, answers, dataset=dataset)

    return {"generated_text": gen_text, "is_correct": is_correct}
