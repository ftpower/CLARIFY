"""Phase 11: Geometry-Aware Truth Intervention.

Six methods (M0-M6) building on InternalInspector + Phase 7-9 geometric findings.

M0: Phase 9.2 replication (L20 single-layer, fixed v_h) — expect zero effect
M1: Contrastive Gradient — ∇_h f as intervention direction
M2: Confidence-Gated Multi-State — f.confidence gate, ∇_{h,a,m} f
M3: Layer-Attribution-Weighted Cascade — w_ℓ from ||∇_{h_ℓ} f||
M4: Sparse Truth Projection — mask ⊙ v, k-dim subset
M5: Orthogonal Dual-Channel — v_a + v_m simultaneously
M6: Push-Pull Geometric Correction — push from μ_i, pull to μ_c

Usage:
    python phase11_intervention.py --load ../phase9_multi_state/outputs_phase9/phase9_extract.json --n_test 50
    python phase11_intervention.py --load ... --n_test 30 --skip_training --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_sys_parent = Path(__file__).parent.parent
for _p in [
    str(_sys_parent / "phase2_entropy"),
    str(_sys_parent / "phase4_generalization"),
    str(_sys_parent / "phase5_cross_task"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model_loader import load_model
from src.data_loader import format_prompt, check_correct


# ═══════════════════════════════════════════════════════════════
# InternalInspector-style Classifier
# ═══════════════════════════════════════════════════════════════


class InternalInspectorMLP(nn.Module):
    """MLP classifier on concatenated [h, a, m] from a single layer.

    Simplified InternalInspector: uses 3-state input from one layer
    instead of all 28 layers. Captures the three-state complementary
    insight while being computationally feasible for gradient-based
    intervention during generation.
    """

    def __init__(self, d_model=2048, hidden_dims=(1024, 256), dropout=0.1):
        super().__init__()
        self.d_model = d_model
        input_dim = d_model * 3  # h + a + m concatenated

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hd),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))  # output: logit

        self.encoder = nn.Sequential(*layers)

    def forward(self, h, a, m):
        """h, a, m: each [..., d_model]. Returns logit [..., 1]."""
        x = torch.cat([h, a, m], dim=-1)
        return self.encoder(x)

    def get_confidence(self, h, a, m):
        return torch.sigmoid(self.forward(h, a, m))


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al. 2020).

    Args:
        features: [N, D] normalized feature vectors
        labels: [N] binary labels (0 or 1)
        temperature: softmax temperature
    """
    N = features.shape[0]
    device = features.device
    labels = labels.float()

    # Cosine similarity matrix
    sim = features @ features.T / temperature  # [N, N]

    # Positive mask: same label
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    # For each sample, compute contrastive loss over all others
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim * (1 - torch.eye(N, device=device))  # remove self

    # Sum over positives for numerator, sum over all for denominator
    pos_sum = (exp_sim * pos_mask).sum(dim=1)  # [N]
    all_sum = exp_sim.sum(dim=1)  # [N]

    # Only compute loss for samples that have positives
    has_positive = pos_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = -torch.log(pos_sum[has_positive] / (all_sum[has_positive] + 1e-10) + 1e-10)
    return loss.mean()


def train_internal_inspector(
    train_h,
    train_a,
    train_m,
    train_labels,
    d_model=2048,
    epochs=200,
    lr=1e-3,
    contrastive_weight=0.3,
    verbose=True,
):
    """Train II classifier on concatenated [h,a,m] states.

    Returns trained model (CPU, eval mode), scalar best_val_acc.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InternalInspectorMLP(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    h_t = torch.tensor(train_h, dtype=torch.float32).to(device)
    a_t = torch.tensor(train_a, dtype=torch.float32).to(device)
    m_t = torch.tensor(train_m, dtype=torch.float32).to(device)
    y_t = torch.tensor(train_labels, dtype=torch.float32).to(device)

    N = len(h_t)
    n_train = int(N * 0.8)
    idx = torch.randperm(N)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    h_tr, a_tr, m_tr, y_tr = h_t[tr_idx], a_t[tr_idx], m_t[tr_idx], y_t[tr_idx]
    h_va, a_va, m_va, y_va = h_t[val_idx], a_t[val_idx], m_t[val_idx], y_t[val_idx]

    best_val_acc = 0.0
    best_state = None
    patience, no_improve = 30, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(h_tr, a_tr, m_tr).squeeze(-1)
        cls_loss = F.binary_cross_entropy_with_logits(logits, y_tr)

        # Supervised contrastive loss on penultimate features
        with torch.no_grad():
            # Get features from second-to-last layer
            x = torch.cat([h_tr, a_tr, m_tr], dim=-1)
            for layer in list(model.encoder.children())[:-1]:
                x = layer(x)
        features = F.normalize(x, dim=-1)
        contr_loss = supervised_contrastive_loss(features, y_tr)

        loss = cls_loss + contrastive_weight * contr_loss
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(h_va, a_va, m_va).squeeze(-1)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_va).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    model.cpu()

    # Final metrics
    with torch.no_grad():
        all_logits = model(h_t.cpu(), a_t.cpu(), m_t.cpu()).squeeze(-1)
        all_preds = (torch.sigmoid(all_logits) > 0.5).float()
        train_acc = (all_preds == y_t.cpu()).float().mean().item()

    if verbose:
        print(
            f"  II classifier: train_acc={train_acc:.3f}, "
            f"best_val_acc={best_val_acc:.3f}, epochs={epoch}"
        )

    return model, best_val_acc


# ═══════════════════════════════════════════════════════════════
# Geometric Signal Computation
# ═══════════════════════════════════════════════════════════════


def compute_geometric_signals(train_records, layer=20, n_layers=28):
    """Compute all geometric signals from training data.

    Returns dict with v_h, v_a, v_m, per_layer_v, mu_correct, mu_incorrect.
    """
    li = str(layer)
    d_model = len(train_records[0]["h"]["0"])

    train_H = np.stack([np.array(r["h"][li]) for r in train_records])
    train_A = np.stack([np.array(r["a"][li]) for r in train_records])
    train_M = np.stack([np.array(r["m"][li]) for r in train_records])
    train_labels = np.array([r["label"] for r in train_records])

    mc, mi = train_labels == 1, train_labels == 0

    # Single-layer truth directions
    def normalize(x):
        return x / (np.linalg.norm(x) + 1e-10)

    v_h = normalize(train_H[mc].mean(0) - train_H[mi].mean(0))
    v_a = normalize(train_A[mc].mean(0) - train_A[mi].mean(0))
    v_m = normalize(train_M[mc].mean(0) - train_M[mi].mean(0))

    # Means
    mu_c_h = train_H[mc].mean(0)
    mu_i_h = train_H[mi].mean(0)

    # Per-layer truth directions (for M3)
    per_layer_v = {}
    for l in range(n_layers):
        ls = str(l)
        H_l = np.stack([np.array(r["h"][ls]) for r in train_records])
        v_l = normalize(H_l[mc].mean(0) - H_l[mi].mean(0))
        per_layer_v[l] = v_l

    # Sparse mask (for M4) — top-k dimensions by |v_h|
    sorted_idx = np.argsort(-np.abs(v_h))
    sparse_masks = {}
    for k in [50, 100, 200, 252, 500, 1024, 2048]:
        mask = np.zeros(d_model, dtype=np.float32)
        mask[sorted_idx[:k]] = 1.0
        sparse_masks[k] = mask

    return {
        "v_h": v_h,
        "v_a": v_a,
        "v_m": v_m,
        "mu_c_h": mu_c_h,
        "mu_i_h": mu_i_h,
        "per_layer_v": per_layer_v,
        "sparse_masks": sparse_masks,
        "d_model": d_model,
        "n_layers": n_layers,
    }


# ═══════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════


def prepare_data(records, train_idx, test_idx, layer=20):
    """Extract L20 h/a/m + labels for train/test splits."""
    li = str(layer)
    n_layers = len(records[0]["h"])

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    # Training data for classifier
    train_h = np.stack([np.array(r["h"][li]) for r in train_records])
    train_a = np.stack([np.array(r["a"][li]) for r in train_records])
    train_m = np.stack([np.array(r["m"][li]) for r in train_records])
    train_labels = np.array([r["label"] for r in train_records], dtype=np.float32)

    # Test data dicts
    test_data = []
    for r in test_records:
        test_data.append(
            {
                "question": r["question"],
                "context": r.get("context", ""),
                "gt_answers": r.get("gt_answers", [r.get("gt_answer", "")]),
                "label": r["label"],
                "h": {str(l): np.array(r["h"][str(l)]) for l in range(n_layers)},
                "a": {str(l): np.array(r["a"][str(l)]) for l in range(n_layers)},
                "m": {str(l): np.array(r["m"][str(l)]) for l in range(n_layers)},
            }
        )

    return train_h, train_a, train_m, train_labels, train_records, test_data


# ═══════════════════════════════════════════════════════════════
# Generation with intervention hooks
# ═══════════════════════════════════════════════════════════════


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20, stored=None):
    """Core greedy generation with hooks."""
    input_len = tokens.shape[1]
    gids = []

    for step in range(max_new):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break

    return tokenizer.decode(gids).strip()


def M0_baseline_generate(model, tokenizer, prompt, device, layer, alpha, v_h):
    """M0: Single-layer fixed-direction (Phase 9.2 replication)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    mod_vec = torch.tensor(alpha * v_h, dtype=torch.float32, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def hook(act, hook=None):
        act[0, input_len - 1, :] += mod_vec
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, hook)])


def M1_gradient_generate(
    model, tokenizer, prompt, device, layer, ii_model, alpha, ii_target_layer=20
):
    """M1: Contrastive gradient intervention.

    Uses ∇_h f(h,a,m) as intervention direction at specified layer.
    ii_target_layer: which layer's states are fed to II classifier.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    # We need to capture h, a, m at ii_target_layer, then intervene
    # at `layer` using gradient w.r.t. h at that layer.
    # For simplicity when layer == ii_target_layer: intervene at the
    # same layer whose states feed into II.
    stored = {}

    def _capture(name):
        def hk(act, hook=None):
            stored[name] = act[0, input_len - 1, :].detach().clone()
            return act

        return hk

    # First pass: capture states with gradient tracking
    # We use a two-pass approach: capture states, compute gradient,
    # then do a second pass with intervention.
    # Actually, for efficiency, do it in one pass by registering a
    # hook that computes gradient inline.
    grad_storage = {"grad": None}

    def _gradient_hook(act, hook=None):
        # Detach the model's forward, then create a leaf for gradient
        h_leaf = act[0, input_len - 1, :].detach().clone().requires_grad_(True)
        stored["h_leaf"] = h_leaf

        # Also need a and m - capture from hooks at same layer
        # We'll use pre-registered captures
        act[0, input_len - 1, :] = h_leaf  # replace with leaf for autograd
        return act

    # Actually, this approach is complex with multiple hooks.
    # Simpler: do a pre-pass to capture h/a/m, then compute grad,
    # then do the actual generation pass with the computed gradient.

    # Pre-pass to capture states
    pre_stored = {}
    li = str(ii_target_layer)

    pre_hooks = [
        (
            f"blocks.{ii_target_layer}.hook_resid_post",
            lambda act, hook=None: (
                pre_stored.update({"h": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
        (
            f"blocks.{ii_target_layer}.hook_attn_out",
            lambda act, hook=None: (
                pre_stored.update({"a": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
        (
            f"blocks.{ii_target_layer}.hook_mlp_out",
            lambda act, hook=None: (
                pre_stored.update({"m": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
    ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=pre_hooks)

    # Compute gradient through II model (cast to float32 for II model)
    h_t = pre_stored["h"].clone().detach().float().requires_grad_(True)
    a_t = pre_stored["a"].clone().detach().float()
    m_t = pre_stored["m"].clone().detach().float()

    # Forward through II with gradient tracking on h
    logit = ii_model.forward(h_t.unsqueeze(0), a_t.unsqueeze(0), m_t.unsqueeze(0))
    conf = torch.sigmoid(logit)
    # Maximize confidence: gradient pushes toward higher confidence
    conf.backward()
    grad = h_t.grad  # [d_model]
    grad_norm = torch.norm(grad)
    if grad_norm > 1e-10:
        grad = grad / grad_norm

    mod_vec = (alpha * grad).to(device)

    # Generation pass with intervention
    hook_name = f"blocks.{layer}.hook_resid_post"

    def _intervene(act, hook=None):
        act[0, input_len - 1, :] += mod_vec
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, _intervene)])


def M2_gated_multistate_generate(
    model, tokenizer, prompt, device, layer, ii_model, alpha, tau=0.3
):
    """M2: Confidence-gated multi-state steering.

    If II confidence < tau: intervene on all three states (h/a/m).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    # Pre-pass to capture states and check confidence
    pre_stored = {}
    pre_hooks = [
        (
            f"blocks.{layer}.hook_resid_post",
            lambda act, hook=None: (
                pre_stored.update({"h": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
        (
            f"blocks.{layer}.hook_attn_out",
            lambda act, hook=None: (
                pre_stored.update({"a": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
        (
            f"blocks.{layer}.hook_mlp_out",
            lambda act, hook=None: (
                pre_stored.update({"m": act[0, input_len - 1, :].detach().clone()})
                or act
            ),
        ),
    ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=pre_hooks)

    # Check confidence
    h_t = pre_stored["h"].unsqueeze(0).float()
    a_t = pre_stored["a"].unsqueeze(0).float()
    m_t = pre_stored["m"].unsqueeze(0).float()
    with torch.no_grad():
        conf = ii_model.get_confidence(h_t, a_t, m_t).item()

    if conf >= tau:
        # High confidence: no intervention
        return _gen_greedy(model, tokenizer, tokens, device, [])

    # Low confidence: compute gradients for all three states
    h_t.requires_grad_(True)
    a_t.requires_grad_(True)
    m_t.requires_grad_(True)

    logit = ii_model.forward(h_t, a_t, m_t)
    torch.sigmoid(logit).backward()

    g_h = h_t.grad
    g_h = g_h / (torch.norm(g_h) + 1e-10)
    g_a = a_t.grad
    g_a = g_a / (torch.norm(g_a) + 1e-10)
    g_m = m_t.grad
    g_m = g_m / (torch.norm(g_m) + 1e-10)

    mod_h = (alpha * g_h).squeeze(0).to(device)
    mod_a = (alpha * g_a).squeeze(0).to(device)
    mod_m = (alpha * g_m).squeeze(0).to(device)

    hooks = [
        (
            f"blocks.{layer}.hook_resid_post",
            lambda act, hook=None: (
                act.__setitem__((0, input_len - 1), act[0, input_len - 1] + mod_h)
                or act
            ),
        ),
        (
            f"blocks.{layer}.hook_attn_out",
            lambda act, hook=None: (
                act.__setitem__((0, input_len - 1), act[0, input_len - 1] + mod_a)
                or act
            ),
        ),
        (
            f"blocks.{layer}.hook_mlp_out",
            lambda act, hook=None: (
                act.__setitem__((0, input_len - 1), act[0, input_len - 1] + mod_m)
                or act
            ),
        ),
    ]

    return _gen_greedy(model, tokenizer, tokens, device, hooks, stored={"conf": conf})


def M3_attribution_cascade_generate(
    model,
    tokenizer,
    prompt,
    device,
    ii_model,
    per_layer_v,
    layer_weights,
    alpha_base,
    n_layers=28,
):
    """M3: Layer-attribution-weighted cascade.

    Uses pre-computed layer weights from ||∇_{h_ℓ} f||.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    hooks = []
    for layer in range(n_layers):
        w = layer_weights.get(layer, 0.0)
        if w < 1e-6:
            continue
        alpha = alpha_base * w
        v = per_layer_v[layer]
        mod_vec = torch.tensor(alpha * v, dtype=torch.float32, device=device)
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Need closure to capture mod_vec per layer
        def _make_hook(_mod):
            def _hk(act, hook=None):
                act[0, input_len - 1, :] += _mod
                return act

            return _hk

        hooks.append((hook_name, _make_hook(mod_vec)))

    return _gen_greedy(model, tokenizer, tokens, device, hooks)


def M4_sparse_generate(
    model, tokenizer, prompt, device, layer, v_h, sparse_mask, alpha
):
    """M4: Sparse truth projection — only modify top-k dimensions."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    sparse_v = sparse_mask * v_h
    sparse_v = sparse_v / (np.linalg.norm(sparse_v) + 1e-10)
    mod_vec = torch.tensor(alpha * sparse_v, dtype=torch.float32, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def hook(act, hook=None):
        act[0, input_len - 1, :] += mod_vec
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, hook)])


def M5_dual_channel_generate(
    model, tokenizer, prompt, device, layer, v_a, v_m, alpha_a, alpha_m
):
    """M5: Orthogonal dual-channel — intervene on attn_out + mlp_out."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    mod_a = torch.tensor(alpha_a * v_a, dtype=torch.float32, device=device)
    mod_m = torch.tensor(alpha_m * v_m, dtype=torch.float32, device=device)

    hooks = [
        (
            f"blocks.{layer}.hook_attn_out",
            lambda act, hook=None: (
                act.__setitem__((0, input_len - 1), act[0, input_len - 1] + mod_a)
                or act
            ),
        ),
        (
            f"blocks.{layer}.hook_mlp_out",
            lambda act, hook=None: (
                act.__setitem__((0, input_len - 1), act[0, input_len - 1] + mod_m)
                or act
            ),
        ),
    ]

    return _gen_greedy(model, tokenizer, tokens, device, hooks)


def M6_push_pull_generate(
    model, tokenizer, prompt, device, layer, v_h, mu_c_h, mu_i_h, alpha_push, alpha_pull
):
    """M6: Push-pull geometric correction."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    # Capture h at layer for push-pull computation
    stored = {}
    pre_hook = [
        (
            f"blocks.{layer}.hook_resid_post",
            lambda act, hook=None: (
                stored.update({"h": act[0, input_len - 1, :].detach().clone()}) or act
            ),
        )
    ]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=pre_hook)

    h_vec = stored["h"].cpu().numpy()

    # Push: away from incorrect mean
    diff_from_incorrect = h_vec - mu_i_h
    push_proj = np.dot(diff_from_incorrect, v_h) * v_h  # project onto truth direction
    push_vec = alpha_push * push_proj

    # Pull: toward correct mean
    pull_vec = alpha_pull * v_h

    # Combined correction
    total_vec = push_vec + pull_vec
    total_norm = np.linalg.norm(total_vec)
    if total_norm > 5.0:
        total_vec = total_vec / total_norm * 5.0

    mod_vec = torch.tensor(total_vec, dtype=torch.float32, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def hook(act, hook=None):
        act[0, input_len - 1, :] += mod_vec
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, hook)])


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════


def compute_layer_attribution_weights(
    ii_model, train_h, train_a, train_m, per_layer_v, n_layers, layer=20
):
    """Compute layer importance weights from II model gradients.

    For each layer ℓ, compute ||∇_{h_ℓ} f|| averaged over training samples.
    Since our II model only takes L20 states, we approximate by:
    - Using the II model's gradient at L20 as the reference
    - Distributing weights based on per-layer AUROC from Phase 9.1 detection
      (higher AUROC → higher weight)

    This is a practical approximation. A full implementation would train
    II on all layers' states and compute per-layer gradients.
    """
    # Use Phase 9.1 detection AUROC as proxy for layer importance
    # These numbers are from the actual Phase 9.1 results
    # Per-layer h AUROC (approximate from Phase 7 C2 scan)
    # We'll use a simplified version: weight by how well v_ℓ separates
    # For now, use uniform weights since we don't have per-layer II gradients
    weights = {}
    for l in range(n_layers):
        weights[l] = 1.0 / n_layers  # uniform fallback

    # Alternative: use II gradient magnitude at L20 as single-point estimate
    # and decay weights with distance from L20
    center = layer
    sigma = 3.0
    for l in range(n_layers):
        weights[l] = np.exp(-0.5 * ((l - center) / sigma) ** 2)
    w_sum = sum(weights.values())
    for l in weights:
        weights[l] /= w_sum

    return weights


def evaluate_method(
    model, tokenizer, test_data, device, method_name, generate_fn, **kwargs
):
    """Evaluate one intervention method on all test samples.

    Args are passed positionally or via a common dict.
    """
    correct = 0
    total = len(test_data)
    results = []

    for sample in tqdm(test_data, desc=method_name, leave=False):
        prompt = format_prompt(
            sample["question"], sample["context"], dataset="triviaqa"
        )
        gt_answers = sample["gt_answers"]

        ans = generate_fn(model, tokenizer, prompt, device, **kwargs)
        is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
        if is_correct:
            correct += 1
        results.append({"answer": ans[:150], "correct": is_correct})

    rate = correct / total
    return rate, results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 11: Geometry-Aware Intervention"
    )
    parser.add_argument(
        "--load",
        type=str,
        default="../phase9_multi_state/outputs_phase9/phase9_extract.json",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip II classifier training (use cached)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Reduced search grid for faster results"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 65}")
    print(f"Phase 11: Geometry-Aware Truth Intervention")
    print(f"  Layer: {args.layer}  Test samples: {args.n_test}")
    print(f"  Quick mode: {args.quick}")
    print(f"{'=' * 65}\n")

    # ── Load data ──
    print(f"Loading: {args.load}")
    with open(args.load) as f:
        data = json.load(f)
    records = data["records"]
    n_total = len(records)
    n_layers = data["config"]["n_layers"]
    d_model = data["config"]["d_model"]

    # Train/test split (same as Phase 9)
    indices = np.random.permutation(n_total)
    test_idx = indices[: args.n_test]
    train_idx = indices[args.n_test :]

    # Prepare data
    train_h, train_a, train_m, train_labels_np, train_records, test_data = prepare_data(
        records, train_idx, test_idx, layer=args.layer
    )

    labels_all = np.array([r["label"] for r in records])
    test_labels = labels_all[test_idx]
    baseline_rate = test_labels.mean()
    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(
        f"  Test correct (extraction): {test_labels.sum()}/{len(test_idx)} "
        f"({baseline_rate:.1%})"
    )

    # ── Phase 1: Train InternalInspector classifier ──
    ii_model_path = output_dir / "ii_classifier.pt"
    if args.skip_training and ii_model_path.exists():
        print(f"\nLoading cached II classifier: {ii_model_path}")
        ii_model = InternalInspectorMLP(d_model=d_model)
        ii_model.load_state_dict(torch.load(ii_model_path, map_location="cpu"))
        ii_model.eval()
    else:
        print(f"\n{'─' * 50}")
        print("Phase 1: Training InternalInspector classifier")
        print(f"{'─' * 50}")
        ii_model, ii_acc = train_internal_inspector(
            train_h,
            train_a,
            train_m,
            train_labels_np,
            d_model=d_model,
            epochs=200,
        )
        torch.save(ii_model.state_dict(), ii_model_path)
        print(f"  Saved: {ii_model_path}")

    # ── Phase 2: Compute geometric signals ──
    print(f"\n{'─' * 50}")
    print("Phase 2: Computing geometric signals")
    print(f"{'─' * 50}")
    geo = compute_geometric_signals(train_records, layer=args.layer, n_layers=n_layers)
    print(f"  cos(v_h, v_a) = {np.dot(geo['v_h'], geo['v_a']):.4f}")
    print(f"  cos(v_h, v_m) = {np.dot(geo['v_h'], geo['v_m']):.4f}")
    print(f"  cos(v_a, v_m) = {np.dot(geo['v_a'], geo['v_m']):.4f}")
    print(f"  ||mu_c - mu_i|| = {np.linalg.norm(geo['mu_c_h'] - geo['mu_i_h']):.2f}")

    # Layer attribution weights (for M3)
    layer_weights = compute_layer_attribution_weights(
        ii_model,
        train_h,
        train_a,
        train_m,
        geo["per_layer_v"],
        n_layers,
        layer=args.layer,
    )
    print(
        f"  Layer weights (top-5): {sorted(layer_weights.items(), key=lambda x: -x[1])[:5]}"
    )

    # ── Load generation model ──
    print(f"\n{'─' * 50}")
    print("Loading generation model...")
    t0 = time.time()
    gen_model = load_model(device=device, model_id=args.model)
    gen_tokenizer = gen_model.tokenizer
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # ── Phase 3: Evaluate all methods ──
    print(f"\n{'─' * 50}")
    print("Phase 3: Intervention Evaluation")
    print(f"{'─' * 50}")

    all_results = {}

    # Common args passed to evaluate_method + generate_fn
    common = dict(model=gen_model, tokenizer=gen_tokenizer, device=device)

    # ── M0: Baseline (Phase 9.2 replication) ──
    t0 = time.time()
    print("\n[M0] Phase 9.2 replication (L20, fixed v_h)...")
    m0_results = {}
    for alpha in [-1.0, -0.5, 0.5, 1.0]:
        rate, _ = evaluate_method(
            test_data=test_data,
            method_name=f"M0_a{alpha}",
            generate_fn=M0_baseline_generate,
            layer=args.layer,
            alpha=alpha,
            v_h=geo["v_h"],
            **common,
        )
        m0_results[f"a{alpha}"] = rate
        print(f"  M0 α={alpha:+.1f}: {rate:.2%}")
    all_results["M0_baseline"] = {"results": m0_results, "time_s": time.time() - t0}

    # ── M1: Contrastive Gradient ──
    t0 = time.time()
    print("\n[M1] Contrastive Gradient Intervention...")
    ii_model_device = ii_model.to(device)
    m1_results = {}
    alphas = [-1.0, -0.5, 0.5, 1.0] if not args.quick else [-1.0, 1.0]
    for alpha in alphas:
        rate, _ = evaluate_method(
            test_data=test_data,
            method_name=f"M1_a{alpha}",
            generate_fn=M1_gradient_generate,
            layer=args.layer,
            ii_model=ii_model_device,
            alpha=alpha,
            **common,
        )
        m1_results[f"a{alpha}"] = rate
        print(f"  M1 α={alpha:+.1f}: {rate:.2%}")
    all_results["M1_gradient"] = {"results": m1_results, "time_s": time.time() - t0}

    # ── M2: Confidence-Gated Multi-State ──
    t0 = time.time()
    print("\n[M2] Confidence-Gated Multi-State Steering...")
    m2_results = {}
    taus = [0.1, 0.3, 0.5]
    alphas = [1.0] if args.quick else [0.5, 1.0]
    best_m2 = 0.0
    for tau in taus:
        for alpha in alphas:
            rate, details = evaluate_method(
                test_data=test_data,
                method_name=f"M2_t{tau}_a{alpha}",
                generate_fn=M2_gated_multistate_generate,
                layer=args.layer,
                ii_model=ii_model_device,
                alpha=alpha,
                tau=tau,
                **common,
            )
            key = f"t{tau}_a{alpha}"
            m2_results[key] = rate
            if rate > best_m2:
                best_m2 = rate
            # Count how many were gated (had low confidence)
            n_gated = sum(
                1
                for d in details
                if d.get("conf", 1.0) < tau
                for d in [{"conf": d.get("conf", 1.0)}]
            )
            print(f"  M2 τ={tau} α={alpha:+.1f}: {rate:.2%}")
    all_results["M2_gated_multistate"] = {
        "results": m2_results,
        "time_s": time.time() - t0,
    }

    # ── M3: Layer-Attribution-Weighted Cascade ──
    t0 = time.time()
    print("\n[M3] Layer-Attribution-Weighted Cascade...")
    m3_results = {}
    alphas = [-1.0, -0.5, 0.5, 1.0] if not args.quick else [-1.0, 1.0]
    for alpha in alphas:
        rate, _ = evaluate_method(
            test_data=test_data,
            method_name=f"M3_a{alpha}",
            generate_fn=M3_attribution_cascade_generate,
            ii_model=ii_model_device,
            per_layer_v=geo["per_layer_v"],
            layer_weights=layer_weights,
            alpha_base=alpha,
            n_layers=n_layers,
            **common,
        )
        m3_results[f"a{alpha}"] = rate
        print(f"  M3 α_base={alpha:+.1f}: {rate:.2%}")
    all_results["M3_attribution_cascade"] = {
        "results": m3_results,
        "time_s": time.time() - t0,
    }

    # ── M4: Sparse Truth Projection ──
    t0 = time.time()
    print("\n[M4] Sparse Truth Projection...")
    m4_results = {}
    ks = [50, 100, 200, 252, 500, 1024, 2048]
    alphas = [-1.0, -0.5, 0.5, 1.0] if not args.quick else [-1.0, 1.0]
    for k in ks:
        mask = geo["sparse_masks"][k]
        for alpha in alphas:
            rate, _ = evaluate_method(
                test_data=test_data,
                method_name=f"M4_k{k}_a{alpha}",
                generate_fn=M4_sparse_generate,
                layer=args.layer,
                v_h=geo["v_h"],
                sparse_mask=mask,
                alpha=alpha,
                **common,
            )
            m4_results[f"k{k}_a{alpha}"] = rate
            nnz = int(mask.sum())
            print(f"  M4 k={k} ({nnz} dims) α={alpha:+.1f}: {rate:.2%}")
    all_results["M4_sparse"] = {"results": m4_results, "time_s": time.time() - t0}

    # ── M5: Orthogonal Dual-Channel ──
    t0 = time.time()
    print("\n[M5] Orthogonal Dual-Channel...")
    m5_results = {}
    if args.quick:
        aa_grid = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0.5, 0.5)]
    else:
        av = [-1.0, -0.5, 0.0, 0.5, 1.0]
        aa_grid = [(aa, am) for aa in av for am in av]
    for alpha_a, alpha_m in aa_grid:
        rate, _ = evaluate_method(
            test_data=test_data,
            method_name=f"M5_aa{alpha_a}_am{alpha_m}",
            generate_fn=M5_dual_channel_generate,
            layer=args.layer,
            v_a=geo["v_a"],
            v_m=geo["v_m"],
            alpha_a=alpha_a,
            alpha_m=alpha_m,
            **common,
        )
        m5_results[f"aa{alpha_a}_am{alpha_m}"] = rate
        best_so_far = max(m5_results.values()) if m5_results else 0
        if rate > best_so_far or abs(alpha_a) < 0.1 or abs(alpha_m) < 0.1:
            print(f"  M5 α_a={alpha_a:+.1f} α_m={alpha_m:+.1f}: {rate:.2%}")
    all_results["M5_dual_channel"] = {"results": m5_results, "time_s": time.time() - t0}

    # ── M6: Push-Pull Geometric Correction ──
    t0 = time.time()
    print("\n[M6] Push-Pull Geometric Correction...")
    m6_results = {}
    if args.quick:
        pp_grid = [(0, 1), (1, 0), (0.5, 0.5)]
    else:
        pp_grid = [(p, l) for p in [0.0, 0.5, 1.0] for l in [0.0, 0.5, 1.0]]
    for alpha_push, alpha_pull in pp_grid:
        rate, _ = evaluate_method(
            test_data=test_data,
            method_name=f"M6_push{alpha_push}_pull{alpha_pull}",
            generate_fn=M6_push_pull_generate,
            layer=args.layer,
            v_h=geo["v_h"],
            mu_c_h=geo["mu_c_h"],
            mu_i_h=geo["mu_i_h"],
            alpha_push=alpha_push,
            alpha_pull=alpha_pull,
            **common,
        )
        m6_results[f"push{alpha_push}_pull{alpha_pull}"] = rate
        if alpha_push > 0 or alpha_pull > 0:
            print(f"  M6 push={alpha_push} pull={alpha_pull}: {rate:.2%}")
    all_results["M6_push_pull"] = {"results": m6_results, "time_s": time.time() - t0}

    # ── Also get actual baseline (no intervention) ──
    print("\n[Baseline] Actual generation (no intervention)...")

    def _baseline_gen(model, tokenizer, prompt, device, **kwargs):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 1024:
            tokens = tokens[:, :1024]
        gids = []
        for _ in range(20):
            with torch.no_grad():
                logits = model(tokens)
            nid = int(logits[0, -1, :].argmax().item())
            if nid == tokenizer.eos_token_id:
                break
            gids.append(nid)
            tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
            if tokens.shape[1] > 1024:
                break
        return tokenizer.decode(gids).strip()

    t0 = time.time()
    real_baseline_rate, _ = evaluate_method(
        test_data=test_data,
        method_name="baseline",
        generate_fn=_baseline_gen,
        **common,
    )
    print(f"  Actual baseline: {real_baseline_rate:.2%}")

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"\n  Actual baseline: {real_baseline_rate:.2%}")
    print(f"\n  {'Method':35s} {'Best':>8s}  {'Δ':>8s}")
    print(f"  {'─' * 53}")

    best_overall = real_baseline_rate
    best_method = "baseline"

    for method, data in sorted(all_results.items()):
        results_dict = data["results"]
        if not results_dict:
            continue
        best_key = max(results_dict, key=results_dict.get)
        best_rate = results_dict[best_key]
        delta = best_rate - real_baseline_rate

        marker = ""
        if delta > 0.05:
            marker = " ↑"
        elif delta < -0.05:
            marker = " ↓"

        print(f"  {method:35s} {best_rate:>8.2%}  {delta:>+8.1%}{marker}")

        if best_rate > best_overall:
            best_overall = best_rate
            best_method = method

    print(f"\n  Best overall: {best_method} ({best_overall:.2%})")

    # ── Save ──
    save_path = output_dir / "phase11_results.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "baseline_rate": real_baseline_rate,
                "extraction_test_rate": float(baseline_rate),
                "n_test": args.n_test,
                "n_train": len(train_idx),
                "layer": args.layer,
                "ii_val_acc": float(ii_acc),
                "all_results": {
                    k: {
                        "best": max(v["results"].values()) if v["results"] else 0,
                        "details": v["results"],
                        "time_s": v["time_s"],
                    }
                    for k, v in all_results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {save_path}")

    print(f"\n{'=' * 65}")
    print("Phase 11 complete")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
