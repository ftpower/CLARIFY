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
        # 架构口径修正（2026-09-27，C3 模型级负例审查发现并 CPU 实测验证）：
        # 旧逻辑把 ("LN","LNPre",None) 一律强转为 "LNPre" 且 fold_ln=False——对 OPT/GPT-2 的
        # post-LN 双 LN 架构不成立（opt-125m 实测前向完全损坏：max|logits 差|≈15.75、top1 失配）；
        # 对 GPTNeoX（pythia）而言 convert 返回 "LN"，需转 "LNPre" 且必须 fold_ln=True
        # （fold_ln=False 实测 max|logits 差|≈14.34、top1 失配）。修正后的三族实测：
        #   Qwen3  → "RMS"、fold_ln=False（原样，主线 1.7B/8B 不受影响）
        #   OPT    → 保持 "LN"、fold_ln=False（max|logits 差| 0.0156＝fp16 舍入级）
        #   GPTNeoX→ "LNPre"、fold_ln=True（top1/top5 与 HF 逐位一致，差 0.289 属官方表征）
        if (cfg_dict.get("original_architecture") == "GPTNeoXForCausalLM"
                and cfg_dict.get("normalization_type") != "LNPre"):
            cfg_dict["normalization_type"] = "LNPre"
        fold_ln = bool(cfg_dict.get("normalization_type") == "LNPre")
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
            fold_ln=fold_ln,
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
