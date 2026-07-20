"""Forward pass and per-layer hidden state extraction via TransformerLens hooks."""

import torch
from transformer_lens import HookedTransformer


def extract_hidden_states(
    model: HookedTransformer,
    prompt: str,
    max_len: int = 1024,
) -> tuple[list[torch.Tensor], torch.Tensor, int, str]:
    """Run forward pass, extract per-layer hidden states at last token position.

    Returns:
        hidden_states: list of [1, d_model] tensors on CPU.
          idx 0           = embedding output (blocks.0.hook_resid_pre)
          idx 1..n_layers = residual after block i-1 (blocks.{i-1}.hook_resid_post)
          idx n_layers    = ln_final(residual after last block)
        logits_final: [1, vocab_size] raw logits at final position.
        gen_token_id: greedy token id from final logits.
        gen_text: decoded text of the generated token.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers
    ln_final = model.ln_final

    residuals = {}

    def _make_hook(name):
        def hook(resid, hook):
            residuals[name] = resid[:, last_pos, :].detach().cpu()

        return hook

    fwd_hooks = [("blocks.0.hook_resid_pre", _make_hook("embed"))] + [
        (f"blocks.{i}.hook_resid_post", _make_hook(f"L{i}")) for i in range(n_layers)
    ]

    with torch.no_grad():
        logits_final = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    hidden_states = []
    hidden_states.append(residuals["embed"])
    for i in range(n_layers - 1):
        hidden_states.append(residuals[f"L{i}"])
    # Last layer: apply ln_final
    hidden_states.append(ln_final(residuals[f"L{n_layers - 1}"]).cpu())

    gen_token_id = logits_final[0, last_pos, :].argmax(dim=-1).item()
    gen_text = model.tokenizer.decode(gen_token_id)

    return (
        hidden_states,
        logits_final[0, last_pos, :].clone(),
        gen_token_id,
        gen_text,
    )


def generate_answer(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 20,
    return_seq_prob: bool = False,
) -> str | tuple[str, float]:
    """Multi-token greedy decode. Returns answer text only (no prompt)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    prompt_len = tokens.shape[1]

    log_probs = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_id = logits[0, -1, :].argmax(dim=-1)
            if return_seq_prob:
                token_log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                log_probs.append(token_log_probs[next_id].item())
            tokens = torch.cat([tokens, next_id.unsqueeze(0).unsqueeze(0)], dim=-1)
            if next_id.item() == model.tokenizer.eos_token_id:
                break

    new_ids = tokens[0, prompt_len:]
    answer = model.tokenizer.decode(new_ids).strip()

    if return_seq_prob:
        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)
            return answer, float(torch.exp(torch.tensor(avg_log_prob)).item())
        return answer, 0.0
    return answer


def extract_post_mlp_states(
    model: HookedTransformer,
    prompt: str,
    max_len: int = 1024,
) -> list[torch.Tensor]:
    """Extract post-MLP activations at last token for each transformer layer.

    Uses blocks.{i}.mlp.hook_post hooks. These are the outputs of each
    layer's MLP sub-block before addition to the residual stream.

    Returns:
        post_mlp_states: list of [1, d_model] tensors on CPU, length n_layers.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers

    mlp_outputs = {}

    def _make_hook(name):
        def hook(act, hook):
            mlp_outputs[name] = act[:, last_pos, :].detach().cpu()

        return hook

    fwd_hooks = [
        (f"blocks.{i}.mlp.hook_post", _make_hook(f"L{i}")) for i in range(n_layers)
    ]

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    gen_token_id = logits[0, last_pos, :].argmax(dim=-1).item()
    return [mlp_outputs[f"L{i}"] for i in range(n_layers)], gen_token_id
