"""Extended hidden state extraction: temperature sampling + Attn/FFN sub-layer hooks.

Extends phase2_entropy.src.hidden_state with two new capabilities not present
in the original module:
  1. generate_with_temperature() — non-greedy generation via torch.multinomial
  2. extract_all_sub_layer_states() — simultaneous hook of residual, attn, and ffn

Hook points used (TransformerLens convention):
  - blocks.{i}.hook_resid_post  — residual stream after block i
  - blocks.{i}.hook_attn_out     — attention sub-layer output (before residual add)
  - blocks.{i}.mlp.hook_post     — MLP sub-layer output (before residual add)
"""

import torch


def generate_with_temperature(
    model,
    prompt: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_new_tokens: int = 20,
    return_hidden_states: bool = False,
    hook_layer: int | None = None,
) -> str | tuple[str, list[torch.Tensor]]:
    """Multi-token generation with temperature sampling (NOT greedy).

    Uses torch.multinomial for non-deterministic sampling at each step.
    Supports optional top-p (nucleus) filtering and per-step hidden state capture.

    Args:
        model: HookedTransformer instance.
        prompt: Input text prompt.
        temperature: Softmax temperature (>0). Lower = more greedy.
        top_p: Nucleus sampling threshold (0 < top_p <= 1). 1.0 = no filtering.
        max_new_tokens: Maximum number of tokens to generate.
        return_hidden_states: If True, also capture hidden states at each decode step
            at the specified hook_layer.
        hook_layer: Layer index for hidden state capture (required if
            return_hidden_states=True).

    Returns:
        If return_hidden_states=False:
            answer_text: str — generated text (prompt excluded).
        If return_hidden_states=True:
            (answer_text, per_step_hidden_states) where per_step_hidden_states
            is a list of [d_model] tensors, one per decode step.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    prompt_len = tokens.shape[1]

    per_step_states = []
    storage = {}

    for _step in range(max_new_tokens):
        # Set up hook for hidden state capture if requested
        fwd_hooks = []
        if return_hidden_states and hook_layer is not None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

            def _capture_hook(act, hook, _name=hook_name):
                storage[_name] = act.detach()
                return act

            fwd_hooks = [(hook_name, _capture_hook)]

        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            else:
                logits = model(tokens)

        next_logits = logits[0, -1, :]  # [vocab_size]

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep at least one token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = -float("inf")

        # Sample via multinomial
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Capture hidden state if requested
        if return_hidden_states and hook_layer is not None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"
            if hook_name in storage:
                h = storage[hook_name][0, -1, :].cpu()
                per_step_states.append(h)

        tokens = torch.cat(
            [tokens, next_id.unsqueeze(0)], dim=-1
        )

        if next_id.item() == model.tokenizer.eos_token_id:
            break

    new_ids = tokens[0, prompt_len:]
    answer = model.tokenizer.decode(new_ids).strip()

    if return_hidden_states:
        return answer, per_step_states
    return answer


def extract_all_sub_layer_states(
    model,
    prompt: str,
    max_len: int = 1024,
) -> dict:
    """Extract residual, attention, and FFN sub-layer outputs in one forward pass.

    Hooks at:
      - blocks.{i}.hook_resid_post  → "hidden"  (residual after block i)
      - blocks.{i}.hook_attn_out    → "attn"    (attention output before residual)
      - blocks.{i}.mlp.hook_post    → "ffn"     (MLP output before residual)

    All values are at the last token position.

    Args:
        model: HookedTransformer instance.
        prompt: Input prompt text.
        max_len: Maximum token length (truncation).

    Returns:
        dict with keys:
            "hidden": list of [1, d_model] tensors (length n_layers), residual stream
                      after each transformer block.
            "attn":   list of [1, d_model] tensors (length n_layers), attention
                      sub-layer outputs.
            "ffn":    list of [1, d_model] tensors (length n_layers), MLP sub-layer
                      outputs.
            "logits": [vocab_size] tensor, raw logits at last position.
            "gen_token_id": int, greedy token id from final logits.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers

    storage = {}

    def _make_hook(key):
        def hook(act, hook=None):
            storage[key] = act[:, last_pos, :].detach()
            return act
        return hook

    # Register hooks at all three sub-layer types
    fwd_hooks = []
    for i in range(n_layers):
        fwd_hooks.append(
            (f"blocks.{i}.hook_resid_post", _make_hook(f"hidden_L{i}"))
        )
        fwd_hooks.append(
            (f"blocks.{i}.hook_attn_out", _make_hook(f"attn_L{i}"))
        )
        fwd_hooks.append(
            (f"blocks.{i}.mlp.hook_post", _make_hook(f"ffn_L{i}"))
        )

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    hidden_states = [
        storage[f"hidden_L{i}"].cpu() for i in range(n_layers)
    ]
    attn_states = [
        storage[f"attn_L{i}"].cpu() for i in range(n_layers)
    ]
    ffn_states = [
        storage[f"ffn_L{i}"].cpu() for i in range(n_layers)
    ]

    gen_token_id = logits[0, last_pos, :].argmax(dim=-1).item()

    return {
        "hidden": hidden_states,
        "attn": attn_states,
        "ffn": ffn_states,
        "logits": logits[0, last_pos, :].clone(),
        "gen_token_id": gen_token_id,
    }
