"""Load model, extract per-layer hidden states and LM head projections."""

import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
    get_pretrained_state_dict,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Project root is 3 levels up from this file
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints")
)


def _find_local_path(model_id: str) -> str | None:
    """Return the local directory path for *model_id* if it exists, else None."""
    local_path = os.path.join(
        CHECKPOINT_DIR, "models--" + model_id.replace("/", "--")
    )
    if os.path.isdir(local_path):
        return local_path
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_path = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))
    if os.path.isdir(hf_path):
        return hf_path
    return None


def load_model(
    device: str = "cuda",
    model_id: str = "Qwen/Qwen3-1.7B",
) -> HookedTransformer:
    """Load a model via TransformerLens (offline-compatible).

    When a local checkpoint directory is found, the model is loaded directly
    from disk and a HookedTransformer is constructed from the pre-loaded
    HuggingFace model, bypassing the cache-lookup path entirely.
    """
    local_path = _find_local_path(model_id)
    load_path = local_path if local_path else model_id

    hf_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        local_files_only=(local_path is None),
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
        local_files_only=(local_path is None),
    )

    if local_path is not None:
        # Build HookedTransformer directly from the loaded hf_model to skip
        # TransformerLens's internal AutoConfig lookup, which requires a
        # standard HF cache structure (snapshots/ + refs/).
        official_name = get_official_model_name(model_id)
        cfg_dict = convert_hf_model_config(local_path, trust_remote_code=True)
        cfg_dict["model_name"] = official_name.split("/")[-1]
        cfg_dict["init_weights"] = False
        if "original_architecture" not in cfg_dict:
            cfg_dict["original_architecture"] = hf_model.config.architectures[0]
        # Determine normalization type for fold_ln
        if cfg_dict.get("normalization_type") in ("LN", "LNPre", None):
            cfg_dict["normalization_type"] = "LNPre"
        cfg = HookedTransformerConfig(**cfg_dict)
        state_dict = get_pretrained_state_dict(
            official_name, cfg, hf_model, dtype=torch.float16,
        )
        model = HookedTransformer(cfg, tokenizer, move_to_device=False)
        model.load_and_process_state_dict(
            state_dict,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )
        if device not in (None, "cpu"):
            model.move_model_modules_to_device()
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            trust_remote_code=True,
            local_files_only=True,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )

    model.eval()
    return model


def get_per_layer_hidden_states(model: HookedTransformer, prompt: str):
    """Run forward pass and extract per-layer hidden states at last token position.

    Uses lightweight hooks that save only the last-position residual to CPU,
    avoiding the memory explosion of run_with_cache() on long contexts.

    Returns:
        hidden_states: list of tensors, each [1, d_model] on CPU.
          Index 0 = embedding output (resid_pre of block 0).
          Index i (1..n_layers-1) = after block i-1 (resid_post of block i-1).
          Index n_layers = final layer output after ln_final.
        final_logits: [1, vocab_size] raw logits at final layer.
        generated_token_id: the greedy token id from the final layer.
        generated_text: decoded text of the generated token.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    max_len = 1024
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]
    last_pos = tokens.shape[1] - 1
    n_layers = model.cfg.n_layers
    ln_final = model.ln_final

    residuals = {}

    def make_hook(name):
        def hook(resid, hook):
            residuals[name] = resid[:, last_pos, :].detach()

        return hook

    fwd_hooks = [("blocks.0.hook_resid_pre", make_hook("embed"))] + [
        (f"blocks.{i}.hook_resid_post", make_hook(f"L{i}")) for i in range(n_layers)
    ]

    with torch.no_grad():
        logits_final = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    hidden_states = []
    # Embedding output (raw, no ln_final)
    hidden_states.append(residuals["embed"])
    # Intermediate layers (raw residual, no ln_final)
    for i in range(n_layers - 1):
        hidden_states.append(residuals[f"L{i}"])
    # Only last layer gets ln_final
    hidden_states.append(ln_final(residuals[f"L{n_layers - 1}"]))

    generated_token_id = logits_final[0, last_pos, :].argmax(dim=-1).item()
    generated_text = model.tokenizer.decode(generated_token_id)

    return (
        hidden_states,
        logits_final[0, last_pos, :],
        generated_token_id,
        generated_text,
    )


def generate_token(
    model: HookedTransformer, prompt: str, temperature: float = 1.0
) -> tuple[int, str]:
    """Generate a single token with temperature sampling (no hidden state extraction).

    Returns (token_id, token_text).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    with torch.no_grad():
        logits = model(tokens)

    last_logits = logits[0, -1, :] / temperature
    probs = torch.softmax(last_logits, dim=-1)
    token_id = torch.multinomial(probs, 1).item()
    token_text = model.tokenizer.decode(token_id)
    return token_id, token_text


def get_confidence_dot(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    b_U: torch.Tensor,
    target_token_id: int,
    temperature: float = 1.0,
) -> list[float]:
    """Extract confidence for a target token using standard dot-product logits.

    logits = h @ W_U + b_U, then softmax with temperature.

    Returns:
        confidences: list of floats, one per layer.
    """
    confidences = []
    for h in hidden_states:
        logits = h @ W_U + b_U  # [1, vocab_size]
        probs = torch.softmax(logits / temperature, dim=-1)
        conf = probs[0, target_token_id].item()
        confidences.append(conf)
    return confidences


def get_confidence_cosine(
    hidden_states: list[torch.Tensor],
    W_U: torch.Tensor,
    target_token_id: int,
    temperature: float = 0.01,
    temperatures: list[float] | None = None,
) -> list[float]:
    """Extract confidence using cosine-similarity logits.

    logits[i] = cos(h, W_U[:,i]), then softmax with temperature.

    Args:
        hidden_states: list of [1, d_model] per layer
        W_U: unembedding matrix [d_model, vocab_size]
        target_token_id: target token
        temperature: default temperature for all layers
        temperatures: per-layer temperatures (overrides temperature)

    Returns:
        confidences: list of floats, one per layer.
    """
    confidences = []
    for layer_idx, h in enumerate(hidden_states):
        t = temperatures[layer_idx] if temperatures is not None else temperature
        h_norm = F.normalize(h, dim=-1)
        W_norm = F.normalize(W_U, dim=0)
        logits = h_norm @ W_norm
        probs = torch.softmax(logits / t, dim=-1)
        conf = probs[0, target_token_id].item()
        confidences.append(conf)
    return confidences


def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Partitions predictions into n_bins equal-width bins by confidence,
    then computes weighted sum of |accuracy - confidence| per bin.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (
            confidences < bin_boundaries[i + 1]
        )
        if in_bin.sum() == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_p_correct(logits: torch.Tensor, answers: list[str], tokenizer) -> float:
    """Approximate P(correct_answer | final_logits) as knowledge proxy.

    Tokenizes each answer alias and returns the max softmax probability
    across all single-token continuations. Used as sample weight in wAUROC.
    """
    probs = torch.softmax(logits, dim=-1)
    max_p = 0.0
    for ans in answers:
        tokens = tokenizer.encode(ans, add_special_tokens=False)
        if tokens:
            p = probs[tokens[0]].item()
            if p > max_p:
                max_p = p
    return max_p


def calibrate_temperatures(
    per_layer_h: list[list[torch.Tensor]],
    W_U: torch.Tensor,
    per_layer_targets: list[list[int]],
    per_layer_labels: list[list[int]],
    n_steps: int = 50,
) -> list[float]:
    """Find per-layer temperature T_ℓ minimizing ECE on calibration data.

    Batches all samples per layer on GPU for efficiency.

    Returns:
        temperatures: list of optimal T per layer
    """
    n_layers = len(per_layer_h)
    temperatures = []
    t_candidates = np.logspace(-5, 0, n_steps)  # 1e-5 to 1.0

    device = W_U.device
    W_norm = F.normalize(W_U, dim=0)  # [d_model, vocab_size]

    for layer_idx in range(n_layers):
        # Stack all hidden states for this layer → [n_samples, d_model]
        h_batch = torch.cat([h.to(device) for h in per_layer_h[layer_idx]], dim=0)
        h_norm = F.normalize(h_batch, dim=-1)  # [n_samples, d_model]
        targets = torch.tensor(per_layer_targets[layer_idx], device=device)
        labels = np.array(per_layer_labels[layer_idx])

        # Pre-compute cosine logits for all samples: [n_samples, vocab_size]
        all_logits = h_norm @ W_norm  # [n_samples, vocab_size]

        best_t = 1.0
        best_ece = float("inf")

        for t in t_candidates:
            probs = torch.softmax(all_logits / t, dim=-1)  # [n_samples, vocab_size]
            confs = probs[torch.arange(len(targets)), targets].detach().cpu().numpy()
            ece = compute_ece(confs, labels)

            if ece < best_ece:
                best_ece = ece
                best_t = t

        temperatures.append(float(best_t))

    return temperatures
