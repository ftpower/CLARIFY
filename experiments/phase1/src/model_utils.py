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
    """Return the local directory path for *model_id* if it exists, else None.

    Returns the directory that contains config.json (may be inside snapshots/).
    """
    for base in [
        CHECKPOINT_DIR,
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local_path = os.path.join(base, "models--" + model_id.replace("/", "--"))
        if not os.path.isdir(local_path):
            continue
        # If config.json is directly in local_path, use it
        if os.path.isfile(os.path.join(local_path, "config.json")):
            return local_path
        # Otherwise try snapshots/<hash>/ subdirectory
        snapshots_dir = os.path.join(local_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snap in sorted(os.listdir(snapshots_dir)):
                snap_path = os.path.join(snapshots_dir, snap)
                if os.path.isfile(os.path.join(snap_path, "config.json")):
                    return snap_path
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
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if local_path is not None:
        # Build HookedTransformer directly from the loaded hf_model to skip
        # TransformerLens's internal AutoConfig lookup, which requires a
        # standard HF cache structure (snapshots/ + refs/).
        official_name = get_official_model_name(model_id)
        cfg_dict = convert_hf_model_config(local_path, trust_remote_code=True)
        cfg_dict["model_name"] = official_name.split("/")[-1]
        cfg_dict["init_weights"] = False
        cfg_dict["device"] = "cpu"  # create on CPU, move to GPU after
        cfg_dict["dtype"] = torch.float16
        if "original_architecture" not in cfg_dict:
            cfg_dict["original_architecture"] = hf_model.config.architectures[0]
        # Determine normalization type for fold_ln
        if cfg_dict.get("normalization_type") in ("LN", "LNPre", None):
            cfg_dict["normalization_type"] = "LNPre"
        cfg = HookedTransformerConfig(**cfg_dict)
        state_dict = get_pretrained_state_dict(
            official_name, cfg, hf_model, dtype=torch.float16,
        )
        # Free hf_model *before* creating HookedTransformer to avoid holding
        # two full copies of weights in memory (hf_model + TL model = 32 GB).
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        model = HookedTransformer(cfg, tokenizer, move_to_device=False)
        # All processing disabled to avoid in-place copies that double peak RAM.
        model.load_and_process_state_dict(
            state_dict,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        del state_dict

        if device not in (None, "cpu"):
            model.cfg.device = device
            model.to(device)
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
        logits_final[0, last_pos, :].clone(),  # clone to free the full [1, seq, vocab] tensor
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


def generate_answer(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 20,
    return_seq_prob: bool = False,
) -> str | tuple[str, float]:
    """Generate a multi-token answer via greedy decoding (no hidden state extraction).

    Returns decoded answer text (does NOT include the prompt).
    If return_seq_prob, also returns exp(average log-prob of generated tokens).
    """
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
            return answer, float(np.exp(avg_log_prob))
        return answer, 0.0
    return answer


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


def compute_alias_seq_prob(
    model: HookedTransformer,
    prompt: str,
    aliases: list[str],
    tokenizer,
    logits_final: torch.Tensor,
) -> float:
    """Compute max sequence probability across correct answer aliases.

    For QA datasets: P(correct) = max_{alias} exp(mean log P(token | prefix)).
    Uses logits_final for the first token of each alias (no extra forward pass),
    then runs forward passes for subsequent tokens.

    Returns a float in [0, 1], comparable to the seq_prob from generate_answer.
    """
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    max_seq_len = 1024
    if prompt_tokens.shape[1] > max_seq_len:
        prompt_tokens = prompt_tokens[:, :max_seq_len]
    device = logits_final.device

    max_prob = 0.0
    for alias in aliases:
        alias_ids = tokenizer.encode(alias, add_special_tokens=False)
        if not alias_ids:
            continue

        log_prob_sum = 0.0
        current_tokens = prompt_tokens.clone()

        for i, tok_id in enumerate(alias_ids):
            if i == 0:
                # First token: use pre-computed logits_final (no forward pass needed)
                token_log_probs = torch.log_softmax(logits_final, dim=-1)
                log_prob_sum += token_log_probs[tok_id].item()
            else:
                # Subsequent tokens: run forward pass on extended sequence
                if current_tokens.shape[1] >= max_seq_len:
                    break  # stop extending if at length limit
                with torch.no_grad():
                    logits = model(current_tokens)
                token_log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                log_prob_sum += token_log_probs[tok_id].item()

            # Extend sequence for next iteration
            tok_tensor = torch.tensor([[tok_id]], device=device)
            current_tokens = torch.cat([current_tokens, tok_tensor], dim=-1)

        avg_log_prob = log_prob_sum / len(alias_ids)
        prob = float(np.exp(avg_log_prob))
        if prob > max_prob:
            max_prob = prob

    return max_prob


def compute_p_correct(
    logits: torch.Tensor, answers: list[str], tokenizer, dataset: str = "triviaqa"
) -> float:
    """Approximate P(correct_answer | final_logits) as knowledge proxy.

    HellaSwag: 4-way softmax over label letters A/B/C/D, returns P(correct_letter).
    QA datasets: max softmax probability across all tokens in all answer aliases.
    Used as sample weight in wAUROC.
    """
    probs = torch.softmax(logits, dim=-1)

    if dataset == "hellaswag":
        # answers[1] is the correct label letter (e.g., "A", "B", "C", "D")
        label_letters = ["A", "B", "C", "D"]
        correct_letter = answers[1].strip().upper()

        letter_ids = []
        for letter in label_letters:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                letter_ids.append(ids[0])

        # Softmax over just the 4 label letters (normalized 4-way choice)
        letter_logits = logits[letter_ids]  # [4]
        letter_probs = torch.softmax(letter_logits, dim=-1)
        correct_idx = label_letters.index(correct_letter)
        return letter_probs[correct_idx].item()

    # QA datasets: max P(token | prompt) across all tokens in all answer aliases
    max_p = 0.0
    for ans in answers:
        tokens = tokenizer.encode(ans, add_special_tokens=False)
        for tok in tokens:
            p = probs[tok].item()
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
