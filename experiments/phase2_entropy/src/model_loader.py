"""Standalone model loading via TransformerLens (offline-compatible)."""

import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
    get_pretrained_state_dict,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints")
)


def _find_local_path(model_id: str) -> str | None:
    for base in [
        CHECKPOINT_DIR,
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
        ),
    ]:
        local_path = os.path.join(base, "models--" + model_id.replace("/", "--"))
        if not os.path.isdir(local_path):
            continue
        if os.path.isfile(os.path.join(local_path, "config.json")):
            return local_path
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
        official_name = get_official_model_name(model_id)
        cfg_dict = convert_hf_model_config(local_path, trust_remote_code=True)
        cfg_dict["model_name"] = official_name.split("/")[-1]
        cfg_dict["init_weights"] = False
        cfg_dict["device"] = "cpu"
        cfg_dict["dtype"] = torch.float16
        if "original_architecture" not in cfg_dict:
            cfg_dict["original_architecture"] = hf_model.config.architectures[0]
        if cfg_dict.get("normalization_type") in ("LN", "LNPre", None):
            cfg_dict["normalization_type"] = "LNPre"
        cfg = HookedTransformerConfig(**cfg_dict)
        state_dict = get_pretrained_state_dict(
            official_name, cfg, hf_model, dtype=torch.float16,
        )
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        model = HookedTransformer(cfg, tokenizer, move_to_device=False)
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
