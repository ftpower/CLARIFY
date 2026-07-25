"""Phase 13: Full InternalInspector — CNN on ALL layers + contrastive learning.

Architecture matches InternalInspector paper: 2D-CNN on [L, d, 3] tensor,
supervised contrastive + classification loss, then ∇f for intervention.

Usage:
    python phase13_ii_full.py --load ../phase9_multi_state/outputs_phase9/phase9_extract.json --n_test 30 --quick
    python phase13_ii_full.py --load ... --n_test 50
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
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


# ═══════════════════════════════════════════════
# InternalInspector CNN (Full)
# ═══════════════════════════════════════════════


class InternalInspectorCNN(nn.Module):
    """2D-CNN on [L, d, 3] tensor. Processes all layers jointly."""

    def __init__(self, n_layers=28, d_model=2048, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        # Conv blocks: process [3, L, d] as 3-channel image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),  # → [16, L/2, d/4]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),  # → [32, L/4, d/16]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),  # → [64, 4, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [256]
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # logit
        )

    def forward(self, theta):
        """theta: [batch, 3, L, d] → logit [batch, 1]"""
        x = self.conv1(theta)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

    def get_confidence(self, theta):
        return torch.sigmoid(self.forward(theta))

    def get_features(self, theta):
        """Penultimate features for contrastive loss."""
        x = self.conv1(theta)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.shape[0], -1)  # [batch, 256]

    def get_gradient(self, theta):
        """Compute ∇_{theta} f, normalized."""
        theta.requires_grad_(True)
        logit = self.forward(theta)
        conf = torch.sigmoid(logit)
        conf.backward()
        grad = theta.grad.clone()
        theta.grad = None
        theta.requires_grad_(False)
        # Normalise per-layer per-state
        g_norm = torch.norm(
            grad.view(grad.shape[0], -1), dim=-1, keepdim=True
        ).clamp_min(1e-10)
        return grad / (g_norm.view(grad.shape[0], 1, 1, 1) + 1e-10)


# ═══════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Khosla et al. 2020."""
    N = features.shape[0]
    device = features.device
    labels = labels.float()
    features = F.normalize(features, dim=-1)
    sim = features @ features.T / temperature
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    exp_sim = torch.exp(sim) * (1 - torch.eye(N, device=device))
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1)
    has_pos = pos_mask.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device)
    loss = -torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-10) + 1e-10)
    return loss.mean()


def build_theta_tensor(records, layer_indices, device="cpu"):
    """Build [N, 3, L, d] tensor from records.

    Args:
        records: list of per-sample dicts with h/a/m per layer
        layer_indices: which layer indices to include
    Returns: torch tensor [N, 3, len(layer_indices), d_model]
    """
    n_layers = len(layer_indices)
    d_model = len(records[0]["h"]["0"])
    N = len(records)
    theta = torch.zeros(N, 3, n_layers, d_model, dtype=torch.float32)
    for i, r in enumerate(records):
        for j, li in enumerate(layer_indices):
            ls = str(li)
            theta[i, 0, j, :] = torch.tensor(r["h"][ls], dtype=torch.float32)
            theta[i, 1, j, :] = torch.tensor(r["a"][ls], dtype=torch.float32)
            theta[i, 2, j, :] = torch.tensor(r["m"][ls], dtype=torch.float32)
    return theta.to(device)


def train_ii_cnn(
    train_records, device="cuda", epochs=100, lr=1e-3, contr_weight=0.3, verbose=True
):
    """Train InternalInspector CNN classifier."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    n_layers = len(train_records[0]["h"])
    d_model = len(train_records[0]["h"]["0"])
    layer_indices = list(range(n_layers))

    # Build tensor (subsample layers if too large)
    if n_layers > 28:
        layer_indices = list(range(0, n_layers, n_layers // 28))[:28]

    theta = build_theta_tensor(train_records, layer_indices, device="cpu")
    y = torch.tensor([r["label"] for r in train_records], dtype=torch.float32)

    model = InternalInspectorCNN(
        n_layers=len(layer_indices),
        d_model=d_model,
    ).to(device)

    # Train/val split
    N = len(theta)
    n_train = int(N * 0.8)
    idx = torch.randperm(N)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    theta_tr, y_tr = theta[tr_idx].to(device), y[tr_idx].to(device)
    theta_va, y_va = theta[val_idx].to(device), y[val_idx].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0
    best_state = None
    patience, no_improve = 20, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(theta_tr).squeeze(-1)
        cls_loss = F.binary_cross_entropy_with_logits(logits, y_tr)

        features = model.get_features(theta_tr)
        contr_loss = supervised_contrastive_loss(features, y_tr)

        loss = cls_loss + contr_weight * contr_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(theta_va).squeeze(-1)
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

        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch}: loss={loss.item():.4f}, val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    model.cpu()

    # Final accuracy
    model.to(device)
    with torch.no_grad():
        all_preds = (torch.sigmoid(model(theta.to(device)).squeeze(-1)) > 0.5).float()
        train_acc = (all_preds == y.to(device)).float().mean().item()
    model.cpu()

    if verbose:
        print(
            f"  II-CNN: train_acc={train_acc:.3f}, best_val_acc={best_val_acc:.3f}, "
            f"epochs={epoch}"
        )

    return model, layer_indices, best_val_acc


# ═══════════════════════════════════════════════
# Generation with ∇f intervention
# ═══════════════════════════════════════════════


def generate_ii_gradient(
    model,
    tokenizer,
    prompt,
    device,
    intervention_layer,
    ii_model,
    layer_indices,
    alpha,
    n_layers=28,
    d_model=2048,
    max_new=20,
):
    """Intervene using ∇f from full II-CNN.

    1. Pre-pass: extract h/a/m at ALL layers
    2. Build theta tensor → compute ∇_{theta} f
    3. Extract gradient for target intervention layer
    4. Generate with gradient-based intervention
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    stored = {}

    def _make_hook(name):
        def hk(act, hook=None):
            stored[name] = act[:, :, :].detach().clone()
            return act

        return hk

    pre_hooks = []
    for li in layer_indices:
        pre_hooks.append((f"blocks.{li}.hook_resid_post", _make_hook(f"h_{li}")))
        pre_hooks.append((f"blocks.{li}.hook_attn_out", _make_hook(f"a_{li}")))
        pre_hooks.append((f"blocks.{li}.hook_mlp_out", _make_hook(f"m_{li}")))

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=pre_hooks)

    # Build theta [1, 3, n_li, d, seq] — use entire sequence, focus on last token
    n_li = len(layer_indices)
    theta_full = torch.zeros(1, 3, n_li, d_model, dtype=torch.float32, device=device)
    for j, li in enumerate(layer_indices):
        theta_full[0, 0, j, :] = stored[f"h_{li}"][0, input_len - 1, :].float()
        theta_full[0, 1, j, :] = stored[f"a_{li}"][0, input_len - 1, :].float()
        theta_full[0, 2, j, :] = stored[f"m_{li}"][0, input_len - 1, :].float()

    # Compute gradient through II-CNN
    ii_dev = next(ii_model.parameters()).device
    theta_full = theta_full.to(ii_dev)
    grad = ii_model.get_gradient(theta_full)  # [1, 3, n_li, d]

    # Extract gradient for intervention layer
    li_pos = layer_indices.index(intervention_layer)
    g_h = grad[0, 0, li_pos, :]  # resid_post gradient
    g_h /= torch.norm(g_h) + 1e-10

    mod_vec = (alpha * g_h).to(device)
    hook_name = f"blocks.{intervention_layer}.hook_resid_post"

    def hook(act, hook=None):
        act[0, input_len - 1, :] += mod_vec
        return act

    return _gen_greedy(model, tokenizer, tokens, device, [(hook_name, hook)], max_new)


def _gen_greedy(model, tokenizer, tokens, device, hooks, max_new=20):
    gids = []
    for _ in range(max_new):
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


def baseline_generate(model, tokenizer, prompt, device, max_new=20):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    gids = []
    for _ in range(max_new):
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


# ═══════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════


def evaluate(gen_model, tokenizer, test_data, device, gen_fn, **kwargs):
    correct = 0
    for sample in tqdm(test_data, desc="eval", leave=False):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        ans = gen_fn(gen_model, tokenizer, prompt, device, **kwargs)
        if check_correct(ans, sample["gt_answers"], dataset="triviaqa"):
            correct += 1
    return correct / len(test_data)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 13: Full InternalInspector")
    parser.add_argument(
        "--load",
        type=str,
        default="../phase9_multi_state/outputs_phase9/phase9_extract.json",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase13")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 65}")
    print(f"Phase 13: Full InternalInspector (CNN on all layers)")
    print(f"  Layer: {args.layer}  Test: {args.n_test}")
    print(f"{'=' * 65}\n")

    # Load data
    print(f"Loading: {args.load}")
    with open(args.load) as f:
        data = json.load(f)
    records = data["records"]
    n_total = len(records)
    n_layers = data["config"]["n_layers"]
    d_model = data["config"]["d_model"]

    indices = np.random.permutation(n_total)
    test_idx = indices[: args.n_test]
    train_idx = indices[args.n_test :]

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    test_labels = np.array([r["label"] for r in test_records])

    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(
        f"  Test correct: {test_labels.sum()}/{len(test_idx)} "
        f"({test_labels.mean():.1%})"
    )

    # Phase 1: Train II-CNN
    print(f"\n{'─' * 50}")
    print("Phase 13.1: Training InternalInspector CNN")
    print(f"{'─' * 50}")
    ii_model, layer_indices, ii_acc = train_ii_cnn(
        train_records,
        device=device,
        epochs=100,
    )
    torch.save(ii_model.state_dict(), output_dir / "ii_cnn.pt")

    # Phase 2: Compute geometric baseline (v_h for comparison)
    print(f"\n{'─' * 50}")
    print("Phase 13.2: Computing geometric baseline")
    print(f"{'─' * 50}")
    li = str(args.layer)
    train_h = np.stack([np.array(r["h"][li]) for r in train_records])
    train_labels_np = np.array([r["label"] for r in train_records])
    mc, mi = train_labels_np == 1, train_labels_np == 0
    v_h = train_h[mc].mean(0) - train_h[mi].mean(0)
    v_h /= np.linalg.norm(v_h) + 1e-10
    print(f"  v_h ready (||v_h||=1)")

    # Load generation model
    print(f"\n{'─' * 50}")
    print("Loading generation model...")
    t0 = time.time()
    gen_model = load_model(device=device, model_id=args.model)
    gen_tokenizer = gen_model.tokenizer
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # Prepare test data
    test_data = []
    for r in test_records:
        test_data.append(
            {
                "question": r["question"],
                "context": r.get("context", ""),
                "gt_answers": r.get("gt_answers", [r.get("gt_answer", "")]),
            }
        )

    # Phase 3: Evaluate
    print(f"\n{'─' * 50}")
    print("Phase 13.3: Intervention Evaluation")
    print(f"{'─' * 50}")

    # Baseline
    print("\n[Baseline]")
    bl_rate = evaluate(gen_model, gen_tokenizer, test_data, device, baseline_generate)
    print(f"  Baseline: {bl_rate:.2%}")

    # M0: Fixed v_h (Phase 9.2 replication)
    print("\n[M0] Fixed v_h...")
    alphas = [-1.0, -0.5, 0.5, 1.0] if not args.quick else [-1.0, 1.0]
    m0_best = bl_rate
    for alpha in alphas:

        def _m0_gen(m, tok, p, d, a=alpha):
            tokens = m.to_tokens(p, prepend_bos=True)
            if tokens.shape[1] > 1024:
                tokens = tokens[:, :1024]
            il = tokens.shape[1]
            mv = torch.tensor(a * v_h, dtype=torch.float32, device=d)
            hn = f"blocks.{args.layer}.hook_resid_post"

            def hk(act, hook=None):
                act[0, il - 1, :] += mv
                return act

            return _gen_greedy(m, tok, tokens, d, [(hn, hk)])

        rate = evaluate(gen_model, gen_tokenizer, test_data, device, _m0_gen)
        m0_best = max(m0_best, rate)
        print(f"    α={alpha:+.1f}: {rate:.2%}")
    print(f"  M0 best: {m0_best:.2%} (Δ={m0_best - bl_rate:+.1%})")

    # II: InternalInspector ∇f intervention
    print("\n[II-∇f] InternalInspector CNN gradient...")
    ii_model_dev = ii_model.to(device)
    ii_best = bl_rate
    for alpha in alphas:
        rate = evaluate(
            gen_model,
            gen_tokenizer,
            test_data,
            device,
            generate_ii_gradient,
            intervention_layer=args.layer,
            ii_model=ii_model_dev,
            layer_indices=layer_indices,
            alpha=alpha,
            n_layers=n_layers,
            d_model=d_model,
        )
        ii_best = max(ii_best, rate)
        print(f"    α={alpha:+.1f}: {rate:.2%}")
    print(f"  II-∇f best: {ii_best:.2%} (Δ={ii_best - bl_rate:+.1%})")

    # Summary
    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"  II-CNN val_acc: {ii_acc:.3f}")
    print(f"  Baseline:   {bl_rate:.2%}")
    print(f"  M0 (v_h):   {m0_best:.2%}  Δ={m0_best - bl_rate:+.1%}")
    print(f"  II-∇f:      {ii_best:.2%}  Δ={ii_best - bl_rate:+.1%}")
    print(f"\n{'=' * 65}\n")


if __name__ == "__main__":
    main()
