"""Phase 20.1: LoRA δ-Corrective Fine-tuning — reduce late-layer distractor amplification.

Theory: docs/theory-intervention-failure.md
Plan:   ~/.claude/plans/CLARIFY/phase20-training-intervention.md §20.1

Core idea:
  TLDC shows that the final layer over-amplifies distractor tokens relative to
  an earlier reference layer. Instead of post-hoc logit adjustment (TLDC), use
  LoRA to modify the last N layers' Q/V projections so the model doesn't
  over-amplify distractors in the first place.

  The reference layer is automatically set to the first of the last 8 layers
  (e.g., L20 for 28-layer 1.7B, L28 for 36-layer 8B). Channel gain:
    g(t|x) = y_last(t) - y_ref(t)
    distractor d = argmax(y_last)

  Loss = CE(y_true) + λ·max(0, g(d) - g(y_true) + m)

  LoRA: r=8, α=16, target=["q_proj","v_proj"], last 8 layers

Gates (see plan §20.1.6):
  P20.1.1: g(d) - g(t*) median decreases > 20%
  P20.1.2: KW exact match Δ > 0
  P20.1.3: KC exact match degradation ≤ 1 sample

Usage:
  # 1.7B (auto-detects L20-L27)
  python train_lora_delta.py --mode train --n_train 200 --n_test 100

  # 8B (auto-detects L28-L35)
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --kc_ce_only --lambda_delta 0.005
  python train_lora_delta.py --mode eval --model_path /path/to/Qwen3-8B

  # Multi-reference δ (Direction 3): L24/L26/L28/L30 weighted average
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --kc_ce_only --lambda_delta 0.0025 \\
      --multi_ref --ref_layers 24,26,28,30 --ref_weights uniform

  # v·h adaptive weighting (Direction 1): continuous weight instead of binary KC mask
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --vh_weight --vh_alpha 5.0 \\
      --lambda_delta 0.0025

  # Combined: multi-ref + v·h weighting
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --vh_weight --vh_alpha 5.0 \\
      --multi_ref --ref_layers 24,26,28,30 --ref_weights auroc --lambda_delta 0.0025

  # Direction B: Token-level δ (d* = argmax excluding y_true)
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --kc_ce_only --lambda_delta 0.0025 \\
      --token_level_delta --token_delta_margin 0.5

  # Direction D: λ fine sweep
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 200 --batch_size 1 --epochs 1 --kc_ce_only \\
      --lambda_values 0.0022,0.0023,0.0024,0.0025,0.0026,0.0027,0.0028

  # Direction A: large training set
  python train_lora_delta.py --mode train --model_path /path/to/Qwen3-8B \\
      --n_train 2000 --batch_size 1 --epochs 1 --kc_ce_only --lambda_delta 0.0025
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Environment ────────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_offline = os.environ.get("HF_ALLOW_ONLINE", "") != "1"
if _offline:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Path setup ─────────────────────────────────────────────────────────────────
_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_triviaqa, format_prompt, check_correct

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_DELTA_LAYERS = 8  # Number of late layers to target (last N layers)
RANK_THRESHOLD = 50
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "lin_theory"


def _get_delta_layers(model) -> tuple[int, int]:
    """Return (layer_early, layer_late) targeting the last NUM_DELTA_LAYERS layers."""
    n_layers = model.config.num_hidden_layers
    layer_late = n_layers - 1
    layer_early = max(0, n_layers - NUM_DELTA_LAYERS)
    return layer_early, layer_late


def _get_lora_dir(lambda_delta: float) -> Path:
    """Lambda-specific LoRA checkpoint directory."""
    return OUTPUT_DIR / f"s20_1_lambda{lambda_delta}"


def _get_results_path(lambda_delta: float) -> Path:
    """Lambda-specific results JSON path."""
    return OUTPUT_DIR / f"s20_1_lambda{lambda_delta}.json"


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U (no sklearn dependency)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # For each (pos, neg) pair: pos > neg → +1, pos == neg → +0.5
    # Vectorized: sum over neg for each pos
    n_pos, n_neg = len(pos), len(neg)
    # Concatenate and rank
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1  # 1-indexed ranks
    pos_ranks = ranks[:n_pos]
    U = pos_ranks.sum() - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def _compute_ref_layer_auroc(
    model, tokenizer, samples, ref_layers, device
) -> dict[int, float]:
    """Compute per-layer AUROC for truth direction on reference layers.

    Uses the same mean-diff v as C2_truth_direction.py but with HF hooks.
    Returns {layer: auroc} dict.
    """
    from src.data_loader import format_prompt, check_correct

    # Extract hidden states per layer
    h_correct = {li: [] for li in ref_layers}
    h_incorrect = {li: [] for li in ref_layers}

    # Navigate to layers
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        try:
            layers = model.model.model.layers
        except AttributeError:
            layers = model.model.layers

    for s in samples:
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        # Register hooks for all ref layers in one forward pass
        caches = {}
        handles = []

        def _make_hook(li):
            def _hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                caches[li] = hs[:, -1, :].detach()

            return _hook

        for li in ref_layers:
            handles.append(layers[li].register_forward_hook(_make_hook(li)))

        with torch.no_grad():
            outputs = model(**tokens)

        for h in handles:
            h.remove()

        # Generate answer to check correctness
        logits = outputs.logits[0, -1, :]
        nid = int(logits.argmax().item())
        gids = [nid]
        past_tokens = tokens.input_ids
        for _ in range(19):
            if nid == tokenizer.eos_token_id:
                break
            past_tokens = torch.cat(
                [past_tokens, torch.tensor([[nid]], device=device)], dim=1
            )
            with torch.no_grad():
                logits = model(past_tokens).logits
            nid = int(logits[0, -1, :].argmax().item())
            gids.append(nid)

        ans = tokenizer.decode(gids, skip_special_tokens=True).strip()
        is_correct = check_correct(ans, s["answers"], dataset="triviaqa")

        for li in ref_layers:
            if li in caches:
                h_vec = caches[li].float().cpu().numpy().flatten()
                if is_correct:
                    h_correct[li].append(h_vec)
                else:
                    h_incorrect[li].append(h_vec)

    # Compute AUROC per layer
    auroc_dict = {}
    for li in ref_layers:
        if len(h_correct[li]) == 0 or len(h_incorrect[li]) == 0:
            auroc_dict[li] = 0.5
            continue
        h_all = np.stack(h_correct[li] + h_incorrect[li])
        labels = np.array([1] * len(h_correct[li]) + [0] * len(h_incorrect[li]))
        v = h_all[labels == 1].mean(axis=0) - h_all[labels == 0].mean(axis=0)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
        scores = h_all @ v
        auroc_dict[li] = _compute_auroc(scores, labels)

    return auroc_dict


# Qwen3-1.7B model path (auto-detected or use cache)
MODEL_ID = "Qwen/Qwen3-1.7B"


def _find_model_path() -> str:
    """Auto-detect local HF model path."""
    import os as _os

    for base in [
        os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"),
        _os.path.join(
            _os.environ.get("HF_HOME", _os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    ]:
        local = _os.path.join(base, "models--" + MODEL_ID.replace("/", "--"))
        if _os.path.isdir(local):
            # Check for config.json directly
            if _os.path.isfile(_os.path.join(local, "config.json")):
                return local
            # Check snapshots
            snaps = _os.path.join(local, "snapshots")
            if _os.path.isdir(snaps):
                for s in sorted(_os.listdir(snaps)):
                    sp = _os.path.join(snaps, s)
                    if _os.path.isfile(_os.path.join(sp, "config.json")):
                        return sp
    return MODEL_ID  # fallback


MODEL_PATH = _find_model_path()


# ═══════════════════════════════════════════════════════════════════════════════
# Data: TriviaQA train split + prompt formatting
# ═══════════════════════════════════════════════════════════════════════════════


def load_triviaqa_train(n_samples: int, seed: int = 42) -> list[dict]:
    """Load TriviaQA TRAIN split samples."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=False)
    ds = ds.shuffle(seed=seed).select(range(n_samples))

    samples = []
    for item in ds:
        question = item["question"]
        answers = item["answer"]["aliases"]
        search_contexts = item["search_results"]["search_context"]
        context = "\n\n".join(ctx for ctx in search_contexts if ctx)
        samples.append({"question": question, "answers": answers, "context": context})
    return samples


def get_first_answer_token_id(tokenizer, answers: list[str]) -> int | None:
    """Return first token ID of the first non-empty answer alias (with leading space)."""
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        tokens = tokenizer.encode(" " + ans_clean, add_special_tokens=False)
        if tokens:
            return int(tokens[0])
    return None


class TriviaQADataset(Dataset):
    """Tokenized TriviaQA samples for LoRA training."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 768):
        self.data = []
        for s in samples:
            prompt = format_prompt(
                s["question"], s.get("context", ""), dataset="triviaqa"
            )
            y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
            if y_true_id is None:
                continue
            # Tokenize: prompt + answer (for CE loss on answer token)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) > max_length:
                prompt_ids = prompt_ids[-max_length:]
            # Full input = prompt + answer_token (we compute loss only on answer pos)
            input_ids = prompt_ids + [y_true_id]
            # Labels: -100 for prompt, y_true_id for answer position
            labels = [-100] * len(prompt_ids) + [y_true_id]
            self.data.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "y_true_id": y_true_id,
                    "question": s["question"],
                    "answers": s["answers"],
                    "prompt_len": len(prompt_ids),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_lora_batch(batch: list[dict], tokenizer) -> dict:
    """Pad sequences to max length in batch."""
    pad_id = tokenizer.pad_token_id or 0
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "y_true_ids": torch.tensor([b["y_true_id"] for b in batch], dtype=torch.long),
        "prompt_lens": [b["prompt_len"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_lora_delta(args):
    """Main training routine: LoRA δ-corrective fine-tuning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_dir = _get_lora_dir(args.lambda_delta)
    results_path = _get_results_path(args.lambda_delta)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(
        f"Phase 20.1: LoRA δ-Corrective Training | n_train={args.n_train} | λ={args.lambda_delta}"
    )

    # ── 0. Resolve target layers ────────────────────────────────────────────
    # Will be set after model loading; declared here for clarity
    layer_early = None
    layer_late = None

    # ── 1. Load tokenizer + base model ────────────────────────────────────
    print("\n[1/6] Loading model...")
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    model.eval()  # base model frozen
    for p in model.parameters():
        p.requires_grad = False
    layer_early, layer_late = _get_delta_layers(model)
    print(
        f"  Model: {model.config.num_hidden_layers} layers, "
        f"d_model={model.config.hidden_size}, "
        f"target L{layer_early}-L{layer_late}"
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── 2. Load training data ────────────────────────────────────────────
    print(f"\n[2/6] Loading {args.n_train} TriviaQA train samples...")
    train_samples = load_triviaqa_train(n_samples=args.n_train, seed=args.seed)
    train_dataset = TriviaQADataset(train_samples, tokenizer)
    print(f"  Valid samples: {len(train_dataset)}/{args.n_train}")

    # ── 2.5. Pre-compute v (truth direction) if v·h weighting enabled ──────
    v_direction = None
    v_median = None
    if getattr(args, "vh_weight", False):
        print(f"\n[2.5a] Computing v·h truth direction at L{layer_early}...")
        # Use a small subset of training data for calibration
        n_calib = min(100, len(train_samples))
        calib_samples = train_samples[:n_calib]

        # Navigate layers (model not yet wrapped in PeftModel)
        _layers = model.model.layers
        _norm = model.model.norm
        _lm_head = model.lm_head

        h_correct_list = []
        h_incorrect_list = []

        for s in tqdm(calib_samples, desc="  Calibrate v"):
            prompt = format_prompt(
                s["question"], s.get("context", ""), dataset="triviaqa"
            )
            tokens = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            # Hook at reference layer
            _cache = {}

            def _vh_hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                _cache["h"] = hs[:, -1, :].detach()

            _handle = _layers[layer_early].register_forward_hook(_vh_hook)

            with torch.no_grad():
                outputs = model(**tokens)
            _handle.remove()

            # Generate answer to check correctness
            logits = outputs.logits[0, -1, :]
            nid = int(logits.argmax().item())
            gids = [nid]
            past = tokens.input_ids
            for _ in range(19):
                if nid == tokenizer.eos_token_id:
                    break
                past = torch.cat([past, torch.tensor([[nid]], device=device)], dim=1)
                with torch.no_grad():
                    logits = model(past).logits
                nid = int(logits[0, -1, :].argmax().item())
                gids.append(nid)
            ans = tokenizer.decode(gids, skip_special_tokens=True).strip()
            is_correct = check_correct(ans, s["answers"], dataset="triviaqa")

            h_vec = _cache["h"].float().cpu().numpy().flatten()
            if is_correct:
                h_correct_list.append(h_vec)
            else:
                h_incorrect_list.append(h_vec)

        if len(h_correct_list) > 0 and len(h_incorrect_list) > 0:
            h_c = np.stack(h_correct_list)
            h_i = np.stack(h_incorrect_list)
            v_np = h_c.mean(axis=0) - h_i.mean(axis=0)
            v_norm = np.linalg.norm(v_np)
            if v_norm > 1e-10:
                v_np = v_np / v_norm
            v_direction = torch.from_numpy(v_np).float().to(device)

            # Compute s(x) = v·h for all calibration samples → median
            h_all = np.concatenate([h_c, h_i], axis=0)
            s_scores = h_all @ v_np
            v_median = float(np.median(s_scores))
            print(
                f"  v computed: n_correct={len(h_correct_list)}, "
                f"n_incorrect={len(h_incorrect_list)}, "
                f"v_norm={v_norm:.4f}, s_median={v_median:.4f}"
            )
        else:
            print("  WARNING: not enough correct/incorrect samples, v·h disabled")
            args.vh_weight = False

    # ── 2.6. Compute per-layer AUROC if multi-ref with auroc weights ──────
    ref_layers_list = None
    ref_alphas = None
    if getattr(args, "multi_ref", False):
        ref_layers_list = [int(x.strip()) for x in args.ref_layers.split(",")]
        # Validate: all ref layers must be < layer_early (before the last N layers)
        # In 8B mode: ref_layers are absolute indices like 24,26,28,30
        if getattr(args, "ref_weights", "uniform") == "auroc":
            print(
                f"\n[2.6] Computing per-layer AUROC for ref layers {ref_layers_list}..."
            )
            n_auroc = min(100, len(train_samples))
            _layers2 = model.model.layers
            auroc_dict = _compute_ref_layer_auroc(
                model, tokenizer, train_samples[:n_auroc], ref_layers_list, device
            )
            total = sum(auroc_dict.values())
            ref_alphas = [
                auroc_dict[li] / total if total > 0 else 1.0 / len(ref_layers_list)
                for li in ref_layers_list
            ]
            print(
                f"  AUROC: {dict(zip(ref_layers_list, [f'{a:.4f}' for a in ref_alphas]))}"
            )
        else:
            ref_alphas = [1.0 / len(ref_layers_list)] * len(ref_layers_list)
            print(f"\n[2.6] Multi-ref δ: layers={ref_layers_list}, weights=uniform")
        print(
            f"  Ref alphas: {dict(zip(ref_layers_list, [f'{a:.4f}' for a in ref_alphas]))}"
        )

    # ── 3. Apply LoRA ─────────────────────────────────────────────────────
    # Resolve target layers for LoRA (supports sparse via --target_layers)
    if getattr(args, "target_layers", None):
        lora_target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    else:
        lora_target_layers = list(range(layer_early, layer_late + 1))

    print(
        f"\n[3/6] Applying LoRA (r={args.lora_r}, α={args.lora_alpha}) "
        f"to layers {lora_target_layers}..."
    )
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=lora_target_layers,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # ── 4. Register hooks ────────────────────────────────────────────────────
    h_ref_caches = {}  # {layer_idx: {"h": tensor}}
    ref_handles = []

    # Navigate layers (post-PeftModel wrapping)
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        try:
            layers = model.model.model.layers
        except AttributeError:
            layers = model.model.layers

    use_multi_ref = getattr(args, "multi_ref", False) and ref_layers_list is not None

    if use_multi_ref:
        # Register hooks for all reference layers
        for li in ref_layers_list:
            cache = {}
            h_ref_caches[li] = cache

            def _make_multi_hook(module, input, output, _layer=li, _cache=cache):
                hs = output[0] if isinstance(output, tuple) else output
                _cache["h"] = hs.detach()  # [B, seq, d_model]

            ref_handles.append(layers[li].register_forward_hook(_make_multi_hook))
        print(f"  Multi-ref hooks registered: layers {ref_layers_list}")
    else:
        # Single reference layer (original behavior)
        h_ref_caches[layer_early] = {}

        def _capture_single(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            h_ref_caches[layer_early]["h"] = hs.detach()

        ref_handles.append(layers[layer_early].register_forward_hook(_capture_single))

    # Get norm and lm_head for logit computation
    try:
        norm = model.base_model.model.model.norm
        lm_head = model.base_model.model.lm_head
    except AttributeError:
        try:
            norm = model.model.model.norm
            lm_head = model.model.lm_head
        except AttributeError:
            norm = model.model.norm
            lm_head = model.lm_head

    # ── 5. Train ──────────────────────────────────────────────────────────
    mode_flags = []
    if getattr(args, "target_layers", None):
        mode_flags.append(
            f"sparse_lora({','.join(str(l) for l in lora_target_layers)})"
        )
    if use_multi_ref:
        mode_flags.append(f"multi_ref({','.join(str(l) for l in ref_layers_list)})")
    if getattr(args, "vh_weight", False):
        mode_flags.append(f"vh(α={getattr(args, 'vh_alpha', 5.0)})")
    if args.kc_ce_only and not getattr(args, "vh_weight", False):
        mode_flags.append("kc_ce_only")
    mode_str = " | " + " + ".join(mode_flags) if mode_flags else ""

    print(
        f"\n[5/6] Training | lr={args.lr} batch_size={args.batch_size} "
        f"epochs={args.epochs} lambda={args.lambda_delta} margin={args.margin}"
        f"{mode_str}"
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_lora_batch(b, tokenizer),
    )

    train_losses = []
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_ce = 0.0
        epoch_delta = 0.0
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            y_true_ids = batch["y_true_ids"].to(device)  # [B]
            B = input_ids.shape[0]

            # Clear all ref caches
            for cache in h_ref_caches.values():
                cache.clear()

            # Forward through full model (computes CE loss internally if labels given)
            outputs = model(input_ids=input_ids, labels=labels)
            ce_loss = outputs.loss  # scalar, mean over non-masked tokens

            # Get L27 logits at last position
            logits_L27 = outputs.logits.detach()  # [B, seq, vocab]

            # Find the last non-masked position for each sample
            last_positions = []
            for b_idx in range(B):
                non_mask = (labels[b_idx] != -100).nonzero(as_tuple=True)[0]
                if len(non_mask) > 0:
                    last_positions.append(non_mask[-1].item())
                else:
                    last_positions.append(labels.shape[1] - 1)

            # Gather L27 logits at answer positions
            g_L27 = torch.stack(
                [logits_L27[b, last_positions[b], :] for b in range(B)]
            ).float()  # [B, vocab]

            # Compute reference logits (single or multi-ref)
            has_ref_logits = False
            if use_multi_ref:
                # Multi-ref: g_ref = Σ α_ℓ · y_ℓ(t)
                g_ref = torch.zeros_like(g_L27)
                for i, li in enumerate(ref_layers_list):
                    h_li = h_ref_caches.get(li, {}).get("h")
                    if h_li is not None:
                        h_li_last = torch.stack(
                            [h_li[b, last_positions[b], :] for b in range(B)]
                        )
                        h_li_norm = norm(h_li_last.to(dtype=norm.weight.dtype))
                        logits_li = lm_head(h_li_norm).float()
                        g_ref = g_ref + ref_alphas[i] * logits_li
                        has_ref_logits = True
                if has_ref_logits:
                    h_vh = h_ref_caches.get(ref_layers_list[0], {}).get("h")
            else:
                # Single ref: original behavior
                h_early = h_ref_caches.get(layer_early, {}).get("h")
                if h_early is not None:
                    h_early_last = torch.stack(
                        [h_early[b, last_positions[b], :] for b in range(B)]
                    )
                    h_early_norm = norm(h_early_last.to(dtype=norm.weight.dtype))
                    g_ref = lm_head(h_early_norm).float()
                    h_vh = h_early
                    has_ref_logits = True

            # Compute δ penalty
            if not has_ref_logits:
                delta_loss = torch.tensor(0.0, device=device)
            else:
                if getattr(args, "token_level_delta", False):
                    # ── Direction B: Token-level δ ──────────────────────────────
                    # d* = argmax_{t ≠ y_true} — distractor excluding true answer.
                    # Penalise only the distractor token's channel gain, not a
                    # global scalar margin that confuses override with legitimate
                    # refinement.  KC samples (where argmax == y_true) still get
                    # penalised on their runner-up token, keeping the margin
                    # honest without the kc_ce_only binary gate.
                    g_masked = g_L27.clone()
                    g_masked[torch.arange(B, device=device), y_true_ids] = -float("inf")
                    d_ids = g_masked.argmax(dim=-1)  # d* = best WRONG token

                    g_d = (
                        g_L27[torch.arange(B, device=device), d_ids]
                        - g_ref[torch.arange(B, device=device), d_ids]
                    )
                    g_tstar = (
                        g_L27[torch.arange(B, device=device), y_true_ids]
                        - g_ref[torch.arange(B, device=device), y_true_ids]
                    )

                    # Same margin formula but d* excludes y_true by construction;
                    # kc_ce_only can still be used for ablation comparison.
                    margin_b = getattr(args, "token_delta_margin", 0.5)
                    penalty = F.relu(g_d - g_tstar + margin_b)

                    if args.kc_ce_only:
                        kc_mask_b = (y_true_ids == g_L27.argmax(dim=-1)).float()
                        penalty = penalty * (1 - kc_mask_b)
                else:
                    # ── Original: global-margin δ ──────────────────────────────
                    # Distractor: argmax of L27 logits
                    d_ids = g_L27.argmax(dim=-1)  # [B]

                    # g(d) - g(t*) for each sample
                    g_d = (
                        g_L27[torch.arange(B, device=device), d_ids]
                        - g_ref[torch.arange(B, device=device), d_ids]
                    )
                    g_tstar = (
                        g_L27[torch.arange(B, device=device), y_true_ids]
                        - g_ref[torch.arange(B, device=device), y_true_ids]
                    )

                    # δ penalty: max(0, g(d) - g(t*) + m)
                    penalty = F.relu(g_d - g_tstar + args.margin)

                    # KC mask (always available as fallback)
                    kc_mask = (y_true_ids == d_ids).float()  # 1=KC, 0=KW-like

                    if getattr(args, "vh_weight", False) and v_direction is not None:
                        # v·h continuous weighting
                        h_vh_last = torch.stack(
                            [h_vh[b, last_positions[b], :] for b in range(B)]
                        )  # [B, d_model]
                        s_score = torch.matmul(
                            h_vh_last.float(), v_direction
                        )  # [B], already normalized v
                        vh_alpha = getattr(args, "vh_alpha", 5.0)
                        w = 1.0 - torch.sigmoid(vh_alpha * (s_score - v_median))
                        penalty = penalty * w
                    elif args.kc_ce_only:
                        # Binary KC mask (original behavior)
                        penalty = penalty * (1 - kc_mask)

                delta_loss = penalty.mean()

            loss = ce_loss + args.lambda_delta * delta_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_delta += (
                delta_loss.item()
                if isinstance(delta_loss, torch.Tensor)
                else delta_loss
            )
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "delta": f"{delta_loss.item() if hasattr(delta_loss, 'item') else delta_loss:.4f}",
                }
            )

            del input_ids, labels, outputs, logits_L27
            if n_batches % 10 == 0:
                torch.cuda.empty_cache()

        avg_ce = epoch_ce / max(n_batches, 1)
        avg_delta = epoch_delta / max(n_batches, 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(
            {
                "epoch": epoch + 1,
                "ce": avg_ce,
                "delta": avg_delta,
                "total": avg_loss,
            }
        )
        print(
            f"    Epoch {epoch + 1}: ce={avg_ce:.4f} delta={avg_delta:.4f} total={avg_loss:.4f}"
        )

        # Save per-epoch checkpoint (for early stopping analysis)
        ep_dir = lora_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(str(ep_dir))
        print(f"    -> Saved LoRA adapter to {ep_dir}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    # ── 6. Cleanup & save metadata ────────────────────────────────────────
    for h in ref_handles:
        h.remove()
    print(f"\n[6/6] Saving metadata...")
    results = {
        "config": {
            "phase": "20.1",
            "n_train": args.n_train,
            "n_valid": len(train_dataset),
            "seed": args.seed,
            "layers": f"L{layer_early}-L{layer_late}",
            "target_modules": ["q_proj", "v_proj"],
            "lora_target_layers": lora_target_layers,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lambda_delta": args.lambda_delta,
            "margin": args.margin,
            "model_path": MODEL_PATH,
            "multi_ref": use_multi_ref,
            "ref_layers": ref_layers_list if use_multi_ref else None,
            "ref_weights": args.ref_weights if use_multi_ref else None,
            "vh_weight": getattr(args, "vh_weight", False),
            "vh_alpha": getattr(args, "vh_alpha", 5.0)
            if getattr(args, "vh_weight", False)
            else None,
            "kc_ce_only": args.kc_ce_only,
        },
        "train_losses": train_losses,
        "best_loss": best_loss,
        "n_trainable_params": n_trainable,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {results_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def classify_sample(
    logits: torch.Tensor, y_true_id: int | None, generated: str, answers: list[str]
) -> str:
    """KC/KW/DK classification."""
    is_correct = check_correct(generated, answers, dataset="triviaqa")
    if is_correct:
        return "KC"
    if y_true_id is None:
        return "DK"
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == y_true_id).nonzero(as_tuple=True)[0].item() + 1
    return "KW" if rank <= RANK_THRESHOLD else "DK"


@torch.no_grad()
def generate_with_model(
    model, tokenizer, prompt: str, device: str, max_new: int = 20
) -> tuple[str, int, torch.Tensor]:
    """Generate answer greedily. Returns (full_text, first_token_id, first_token_logits)."""
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :].float().cpu()  # [vocab_size]
    first_token_id = int(logits.argmax().item())

    # Continue generation
    gen_ids = [first_token_id]
    current_ids = input_ids
    for _ in range(max_new - 1):
        next_tok = torch.tensor([[gen_ids[-1]]], device=device)
        current_ids = torch.cat([current_ids, next_tok], dim=1)
        with torch.no_grad():
            out = model(input_ids=current_ids)
        nid = int(out.logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gen_ids.append(nid)

    full_text = tokenizer.decode(gen_ids).strip()
    return full_text, first_token_id, logits


def evaluate(args):
    """Evaluate LoRA model vs baseline on test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Allow caller to override model path (e.g., train_contrastive.py)
    _MODEL_PATH = getattr(args, "model_path", None) or MODEL_PATH
    lora_dir = (
        Path(args.lora_checkpoint)
        if getattr(args, "lora_checkpoint", None)
        else _get_lora_dir(getattr(args, "lambda_delta", 0.1))
    )
    results_path = _get_results_path(getattr(args, "lambda_delta", 0.1))
    print(
        f"Phase 20.1: LoRA δ-Corrective Evaluation | n_test={args.n_test} | λ={getattr(args, 'lambda_delta', 0.1)}"
    )

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_kwargs = dict(trust_remote_code=True, local_files_only=True)

    # ── 2. Evaluate baseline ──────────────────────────────────────────────
    print("\n[1/3] Evaluating baseline...")
    base_model = AutoModelForCausalLM.from_pretrained(
        _MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
    ).to(device)
    base_model.eval()

    test_samples = load_triviaqa(n_samples=args.n_test, seed=args.seed)
    bl_results = []
    for s in tqdm(test_samples, desc="  Baseline"):
        prompt = format_prompt(s["question"], s.get("context", ""), dataset="triviaqa")
        full_text, ft_id, logits = generate_with_model(
            base_model, tokenizer, prompt, device
        )
        y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
        ft_correct = y_true_id is not None and ft_id == y_true_id
        em_correct = check_correct(full_text, s["answers"], dataset="triviaqa")
        category = classify_sample(logits, y_true_id, full_text, s["answers"])
        bl_results.append(
            {
                "question": s["question"],
                "answers": s["answers"],
                "y_true_id": y_true_id,
                "ft_id": ft_id,
                "ft_correct": ft_correct,
                "em_correct": em_correct,
                "full_text": full_text,
                "category": category,
            }
        )

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Evaluate LoRA model ─────────────────────────────────────────────
    lora_results = []
    if getattr(args, "skip_lora", False):
        print("\n[2/3] Skipping LoRA evaluation (--skip_lora)")
    elif not lora_dir.exists():
        print(f"  WARNING: LoRA adapter not found at {lora_dir}, baseline-only")
    else:
        print("\n[2/3] Evaluating LoRA model...")
        base = AutoModelForCausalLM.from_pretrained(
            _MODEL_PATH, **hf_kwargs, torch_dtype=torch.float16
        ).to(device)
        # Load adapter, handling layers_to_transform across PEFT versions
        _cfg = json.loads((lora_dir / "adapter_config.json").read_text())
        _lt = _cfg.pop("layers_to_transform", None)
        from peft import LoraConfig as _LoraConfig

        _peft_cfg_fields = {
            "task_type",
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "layers_to_transform",
            "layers_pattern",
            "rank_pattern",
            "alpha_pattern",
            "fan_in_fan_out",
            "init_lora_weights",
            "use_dora",
            "use_rslora",
            "loftq_config",
        }
        _clean_cfg = {k: v for k, v in _cfg.items() if k in _peft_cfg_fields}
        if _lt is not None:
            _clean_cfg["layers_to_transform"] = _lt
        _lora_cfg = _LoraConfig(**_clean_cfg)
        lora_model = PeftModel.from_pretrained(base, str(lora_dir), config=_lora_cfg)
        lora_model.eval()

        for s in tqdm(test_samples, desc="  LoRA"):
            prompt = format_prompt(
                s["question"], s.get("context", ""), dataset="triviaqa"
            )
            full_text, ft_id, logits = generate_with_model(
                lora_model, tokenizer, prompt, device
            )
            y_true_id = get_first_answer_token_id(tokenizer, s["answers"])
            ft_correct = y_true_id is not None and ft_id == y_true_id
            em_correct = check_correct(full_text, s["answers"], dataset="triviaqa")
            category = classify_sample(logits, y_true_id, full_text, s["answers"])
            lora_results.append(
                {
                    "question": s["question"],
                    "answers": s["answers"],
                    "y_true_id": y_true_id,
                    "ft_id": ft_id,
                    "ft_correct": ft_correct,
                    "em_correct": em_correct,
                    "full_text": full_text,
                    "category": category,
                }
            )

        del base, lora_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary & Gates ────────────────────────────────────────────────────
    has_lora = len(lora_results) > 0
    print(f"\n[{'3/3' if has_lora else '2/2'}] Computing summary...")
    n = len(test_samples)
    bl_em = sum(1 for r in bl_results if r["em_correct"])
    lo_em = sum(1 for r in lora_results if r["em_correct"])
    bl_ft = sum(1 for r in bl_results if r["ft_correct"])
    lo_ft = sum(1 for r in lora_results if r["ft_correct"])

    # Per-category
    categories = ["KC", "KW", "DK"]
    cat_stats = {}
    for cat in categories:
        cat_stats[cat] = {
            "n": sum(1 for r in bl_results if r["category"] == cat),
            "bl_em": 0,
            "lo_em": 0,
            "bl_ft": 0,
            "lo_ft": 0,
        }
    for i in range(n):
        cat = bl_results[i]["category"]
        if bl_results[i]["em_correct"]:
            cat_stats[cat]["bl_em"] += 1
        if bl_results[i]["ft_correct"]:
            cat_stats[cat]["bl_ft"] += 1
        if has_lora:
            if lora_results[i]["em_correct"]:
                cat_stats[cat]["lo_em"] += 1
            if lora_results[i]["ft_correct"]:
                cat_stats[cat]["lo_ft"] += 1

    kw_n = cat_stats["KW"]["n"]
    kc_n = cat_stats["KC"]["n"]

    # Gate P20.1.2: KW exact match Δ > 0
    kw_bl_em = cat_stats["KW"]["bl_em"]
    kw_lo_em = cat_stats["KW"]["lo_em"]
    gate_p2012 = kw_lo_em > kw_bl_em

    # Gate P20.1.3: KC exact match degradation ≤ 1
    kc_bl_em = cat_stats["KC"]["bl_em"]
    kc_lo_em = cat_stats["KC"]["lo_em"]
    kc_degradation = kc_bl_em - kc_lo_em
    gate_p2013 = kc_degradation <= 1

    results = {
        "config": {
            "phase": "20.1",
            "n_test": n,
            "seed": args.seed,
            "rank_threshold": RANK_THRESHOLD,
        },
        "summary": {
            "n_total": n,
            "baseline_em_accuracy": bl_em / n,
            "lora_em_accuracy": lo_em / n,
            "em_delta": (lo_em - bl_em) / n,
            "baseline_ft_accuracy": bl_ft / n,
            "lora_ft_accuracy": lo_ft / n,
            "ft_delta": (lo_ft - bl_ft) / n,
            "per_category": cat_stats,
            "gates": {
                "P20.1.2": {
                    "description": "KW exact match Δ > 0",
                    "baseline_kw_em": kw_bl_em,
                    "lora_kw_em": kw_lo_em,
                    "kw_n": kw_n,
                    "delta": kw_lo_em - kw_bl_em,
                    "pass": gate_p2012,
                },
                "P20.1.3": {
                    "description": "KC exact match degradation ≤ 1",
                    "baseline_kc_em": kc_bl_em,
                    "lora_kc_em": kc_lo_em,
                    "kc_n": kc_n,
                    "degradation": kc_degradation,
                    "pass": gate_p2013,
                },
            },
        },
        "per_sample": [],
    }

    # Per-sample records
    for i in range(n):
        sample = {
            "question": bl_results[i]["question"],
            "category": bl_results[i]["category"],
            "baseline": {
                "ft_correct": bl_results[i]["ft_correct"],
                "em_correct": bl_results[i]["em_correct"],
                "full_text": bl_results[i]["full_text"],
            },
        }
        if has_lora:
            sample["lora"] = {
                "ft_correct": lora_results[i]["ft_correct"],
                "em_correct": lora_results[i]["em_correct"],
                "full_text": lora_results[i]["full_text"],
            }
        results["per_sample"].append(sample)

    # Print
    print(f"\n{'=' * 60}")
    print(f"RESULTS{' (baseline-only)' if not has_lora else ''}")
    print(f"{'=' * 60}")
    print(f"  N = {n} | KC={kc_n} KW={kw_n} DK={cat_stats['DK']['n']}")
    print(f"\n  Exact-match accuracy:")
    print(f"    Baseline: {bl_em}/{n} = {bl_em / n:.1%}")
    if has_lora:
        print(f"    LoRA:     {lo_em}/{n} = {lo_em / n:.1%}")
        print(f"    Delta:    {(lo_em - bl_em) / n:+.1%}")
    print(f"\n  First-token accuracy:")
    print(f"    Baseline: {bl_ft}/{n} = {bl_ft / n:.1%}")
    if has_lora:
        print(f"    LoRA:     {lo_ft}/{n} = {lo_ft / n:.1%}")
        print(f"    Delta:    {(lo_ft - bl_ft) / n:+.1%}")
    print(f"\n  Per-category EM:")
    for cat in categories:
        cs = cat_stats[cat]
        bl_a = cs["bl_em"] / max(cs["n"], 1)
        if has_lora:
            lo_a = cs["lo_em"] / max(cs["n"], 1)
            print(
                f"    {cat} (n={cs['n']}): baseline={bl_a:.1%} lora={lo_a:.1%} delta={lo_a - bl_a:+.1%}"
            )
        else:
            print(f"    {cat} (n={cs['n']}): baseline={bl_a:.1%}")
    if has_lora:
        print(f"\n  Gates:")
        for gname, ginfo in results["summary"]["gates"].items():
            status = "✅ PASS" if ginfo["pass"] else "❌ FAIL"
            print(f"    {gname}: {status} — {ginfo['description']}")
    else:
        print(f"\n  Gates: (run training first)")
    print(f"{'=' * 60}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {results_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 20.1: LoRA δ-Corrective Fine-tuning"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: LoRA δ-corrective | eval: evaluate vs baseline",
    )
    # Training
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument(
        "--lambda_values",
        type=str,
        default=None,
        help="Comma-separated lambda values for sweep (e.g. '0.005,0.01,0.05,0.1')",
    )
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--kc_ce_only",
        action="store_true",
        help="Apply CE-only loss to KC-like samples (y_true == argmax)",
    )
    # Token-level δ (Direction B)
    parser.add_argument(
        "--token_level_delta",
        action="store_true",
        help="Use token-level δ: d* = argmax_{t != y_true}, "
        "only penalise distractor token (not global margin). "
        "Direction B in phase20-8b-validation plan.",
    )
    parser.add_argument(
        "--token_delta_margin",
        type=float,
        default=0.5,
        help="Margin for token-level δ penalty (default 0.5, smaller than "
        "global margin because the signal is more targeted).",
    )
    # Multi-reference δ
    parser.add_argument(
        "--multi_ref",
        action="store_true",
        help="Use multiple reference layers for δ aggregation",
    )
    parser.add_argument(
        "--ref_layers",
        type=str,
        default="24,26,28,30",
        help="Comma-separated reference layer indices for multi-ref (relative to model layers)",
    )
    parser.add_argument(
        "--ref_weights",
        type=str,
        default="uniform",
        choices=["uniform", "auroc"],
        help="Weighting scheme: uniform or auroc (per-layer AUROC-based)",
    )
    # v·h adaptive weighting
    parser.add_argument(
        "--vh_weight",
        action="store_true",
        help="Use v·h truth direction for continuous δ penalty weighting",
    )
    parser.add_argument(
        "--vh_alpha",
        type=float,
        default=5.0,
        help="Sharpness of v·h sigmoid weighting (higher = sharper transition)",
    )
    # Sparse LoRA: only apply to specific layers
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for sparse LoRA (e.g. '32,33,34,35'). "
        "Default: all NUM_DELTA_LAYERS late layers.",
    )
    # Eval
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Explicit path to LoRA adapter directory for eval (overrides --lambda_delta)",
    )
    parser.add_argument(
        "--skip_lora",
        action="store_true",
        help="Evaluate baseline only, skip LoRA model loading",
    )
    # Shared
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path override (e.g. /root/.../Qwen3-1.7B)",
    )
    args = parser.parse_args()

    # Resolve model path before importing model
    if args.model_path:
        global MODEL_PATH
        MODEL_PATH = args.model_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        if args.lambda_values:
            # Lambda sweep: train + eval for each λ
            lambdas = [float(x.strip()) for x in args.lambda_values.split(",")]
            print(f"λ sweep: {lambdas}")
            all_results = {}
            for lam in lambdas:
                print(f"\n{'=' * 60}")
                print(f"  λ = {lam}")
                print(f"{'=' * 60}")
                args.lambda_delta = lam
                train_lora_delta(args)

                # Evaluate all epochs, pick best
                lora_dir = _get_lora_dir(lam)
                best_epoch = None
                best_kw_delta = -1
                for ep in range(1, args.epochs + 1):
                    ep_dir = lora_dir / f"epoch_{ep}"
                    if not ep_dir.exists():
                        continue
                    args.lora_checkpoint = str(ep_dir)
                    args.lambda_delta = lam
                    print(f"\n  --- Eval epoch {ep} ---")
                    evaluate(args)
                    # Read results to check KW delta
                    rp = _get_results_path(lam)
                    if rp.exists():
                        with open(rp) as f:
                            res = json.load(f)
                        kw_delta = res["summary"]["gates"]["P20.1.2"]["delta"]
                        kc_deg = res["summary"]["gates"]["P20.1.3"]["degradation"]
                        print(f"  λ={lam} ep={ep}: KW_Δ={kw_delta} KC_deg={kc_deg}")
                        all_results[f"λ={lam}_ep={ep}"] = {
                            "kw_delta": kw_delta,
                            "kc_degradation": kc_deg,
                            "em_delta": res["summary"]["em_delta"],
                        }
                        if kw_delta > best_kw_delta:
                            best_kw_delta = kw_delta
                            best_epoch = ep
                if best_epoch:
                    print(f"\n  Best: λ={lam} epoch={best_epoch} KW_Δ={best_kw_delta}")

            # Summary
            print(f"\n{'=' * 60}")
            print("SWEEP SUMMARY")
            print(f"{'=' * 60}")
            for k, v in sorted(all_results.items()):
                kw_mark = "✅" if v["kw_delta"] > 0 else "❌"
                kc_mark = "✅" if v["kc_degradation"] <= 1 else "❌"
                print(
                    f"  {k}: KW_Δ={v['kw_delta']} {kw_mark} | KC_deg={v['kc_degradation']} {kc_mark} | EM_Δ={v['em_delta']:+.1%}"
                )
            # Save sweep summary
            sweep_path = OUTPUT_DIR / "s20_1_sweep_summary.json"
            with open(sweep_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved: {sweep_path}")
        else:
            train_lora_delta(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
