"""Phase 12: FactCheckmate Replication — Two-stage preemptive intervention.

f_θ: Detector (2-layer MLP, last-token hidden state → p(hallucination))
g_φ: Corrector (3-layer ReLU MLP, h → correction vector)

Three h* target variants for training g_φ:
  A: correct-only (identity target for correct, skip incorrect)
  B: nearest-correct-neighbor target for incorrect
  C: global-correct-mean target for incorrect

Usage:
    python phase12_factcheckmate.py --load ../phase9_multi_state/outputs_phase9/phase9_extract.json --n_test 50
    python phase12_factcheckmate.py --load ... --n_test 30 --quick
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
from sklearn.metrics import roc_auc_score
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
# FactCheckmate Detector f_θ
# ═══════════════════════════════════════════════════════════════


class HallucinationDetector(nn.Module):
    """2-layer MLP: hidden state → p(hallucination)."""

    def __init__(self, d_model=2048, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)  # logit


# ═══════════════════════════════════════════════════════════════
# FactCheckmate Corrector g_φ
# ═══════════════════════════════════════════════════════════════


class FactCheckmateCorrector(nn.Module):
    """3-layer ReLU MLP: hidden state → correction vector."""

    def __init__(self, d_model=2048, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, d_model),
        )

    def forward(self, h):
        return self.net(h)


class BottleneckAutoencoderCorrector(nn.Module):
    """Bottleneck autoencoder g_φ: h → bottleneck → correction.

    Key hypothesis (FactCheckmate Appendix C.2):
    Trained with identity target (h̃ = h + g_φ(h) ≈ h) on ALL samples.
    The bottleneck forces g_φ to learn the manifold of legal hidden states.
    For incorrect samples at inference: g_φ projects h onto the learned
    "correct manifold" — no explicit h* needed.
    """

    def __init__(self, d_model=2048, bottleneck=128, hidden=512, dropout=0.1):
        super().__init__()
        # Encoder: d → hidden → bottleneck (compressed representation)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )
        # Decoder: bottleneck → hidden → d (reconstruction → correction)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.bottleneck_dim = bottleneck

    def forward(self, h):
        z = self.encoder(h)
        return self.decoder(z)


# ═══════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════


def train_detector(
    train_h, train_labels, device="cuda", epochs=100, lr=1e-3, verbose=True
):
    """Train f_θ detector. Returns model (CPU), best val AUROC."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    d_model = train_h.shape[1]
    model = HallucinationDetector(d_model=d_model).to(device)

    h_t = torch.tensor(train_h, dtype=torch.float32).to(device)
    y_t = torch.tensor(train_labels, dtype=torch.float32).to(device)

    N = len(h_t)
    n_train = int(N * 0.8)
    idx = torch.randperm(N)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    h_tr, y_tr = h_t[tr_idx], y_t[tr_idx]
    h_va, y_va = h_t[val_idx], y_t[val_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auroc = 0.0
    best_state = None
    patience, no_improve = 20, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(h_tr)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(h_va)
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
            y_val_np = y_va.cpu().numpy()
            if y_val_np.std() > 0:
                auroc = roc_auc_score(y_val_np, val_scores)
                auroc = max(auroc, 1 - auroc)
            else:
                auroc = 0.5

        if auroc > best_auroc:
            best_auroc = auroc
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

    if verbose:
        print(f"  Detector: val_AUROC={best_auroc:.4f}, epochs={epoch}")

    return model, best_auroc


def train_corrector(
    train_h, train_labels, variant="A", device="cuda", epochs=200, lr=1e-4, verbose=True
):
    """Train g_φ corrector.

    variant A: correct-only (identity target), skip incorrect
    variant B: nearest-correct-neighbor target for incorrect
    variant C: global-correct-mean target for incorrect

    Returns model (CPU), best_val_loss.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    d_model = train_h.shape[1]
    if variant == "D":
        model = BottleneckAutoencoderCorrector(d_model=d_model).to(device)
    else:
        model = FactCheckmateCorrector(d_model=d_model).to(device)

    h_t = torch.tensor(train_h, dtype=torch.float32).to(device)
    y_t = torch.tensor(train_labels, dtype=torch.float32).to(device)

    # Pre-compute targets based on variant
    correct_mask = y_t > 0.5
    incorrect_mask = ~correct_mask

    if variant == "A":
        # Only train on correct samples, identity target
        train_mask = correct_mask
        targets = h_t.clone()  # identity
    elif variant == "B":
        # Correct: identity. Incorrect: nearest correct neighbor
        train_mask = torch.ones(len(h_t), dtype=torch.bool)
        targets = h_t.clone()
        h_correct = h_t[correct_mask].cpu().numpy()
        for i in range(len(h_t)):
            if incorrect_mask[i]:
                h_i = h_t[i].cpu().numpy()
                dists = np.linalg.norm(h_correct - h_i, axis=1)
                nearest = h_correct[dists.argmin()]
                targets[i] = torch.tensor(nearest, dtype=torch.float32).to(device)
    elif variant == "C":
        # Correct: identity. Incorrect: global correct mean
        train_mask = torch.ones(len(h_t), dtype=torch.bool)
        targets = h_t.clone()
        mean_correct = h_t[correct_mask].mean(dim=0)
        targets[incorrect_mask] = mean_correct
    elif variant == "D":
        # Bottleneck autoencoder: ALL samples, identity target
        train_mask = torch.ones(len(h_t), dtype=torch.bool)
        targets = h_t.clone()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Train/val split
    N = train_mask.sum().item()
    indices = torch.where(train_mask)[0]
    perm = torch.randperm(N)
    n_train = int(N * 0.8)
    tr_idx = indices[perm[:n_train]]
    val_idx = indices[perm[n_train:]]

    h_tr, t_tr = h_t[tr_idx], targets[tr_idx]
    h_va, t_va = h_t[val_idx], targets[val_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None
    patience, no_improve = 40, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        correction = model(h_tr)
        pred = h_tr + correction
        loss = nn.functional.mse_loss(pred, t_tr)
        if variant == "D":
            corr_norm = torch.norm(correction, dim=1).mean()
            loss = loss + 0.01 * corr_norm
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_correction = model(h_va)
            val_pred = h_va + val_correction
            val_loss = nn.functional.mse_loss(val_pred, t_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    if verbose:
        # Compute metrics
        with torch.no_grad():
            all_corr = model(h_t.cpu())
            corr_norms = torch.norm(all_corr, dim=1)
            # Cosine between correction and target direction
            target_dirs = targets.cpu() - h_t.cpu()
            cos_sims = nn.functional.cosine_similarity(all_corr, target_dirs, dim=1)
        print(
            f"  Corrector[{variant}]: val_MSE={best_val_loss:.6f}, "
            f"mean|corr|={corr_norms.mean():.3f}, "
            f"median|corr|={corr_norms.median():.3f}, "
            f"mean_cos={cos_sims.mean():.4f}, epochs={epoch}"
        )

    return model, best_val_loss


# ═══════════════════════════════════════════════════════════════
# Generation with FactCheckmate intervention
# ═══════════════════════════════════════════════════════════════


def generate_fcm_intervention(
    model,
    tokenizer,
    prompt,
    device,
    layer,
    detector,
    corrector,
    tau=0.3,
    max_new_tokens=20,
):
    """FactCheckmate-style intervention: detect + correct at first step."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]
    input_len = tokens.shape[1]

    # Step 1: Pre-pass to get hidden state for detection/correction
    stored = {}
    hook_name = f"blocks.{layer}.hook_resid_post"

    def _capture(act, hook=None):
        stored["h"] = act[0, input_len - 1, :].detach().clone()
        return act

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])

    # Step 2: Detection
    h_vec = stored["h"].float().unsqueeze(0).to(device)
    with torch.no_grad():
        logit = detector(h_vec)
        p_hallucination = torch.sigmoid(logit).item()

    # Step 3: Decide whether to intervene
    should_intervene = p_hallucination < tau

    # Step 4: Generation
    gids = []
    for step in range(max_new_tokens):
        if step == 0 and should_intervene:
            # Apply correction at first decoding step only
            with torch.no_grad():
                correction = corrector(h_vec).squeeze(0)
            mod_vec = stored["h"] + correction

            def _correct(act, hook=None):
                act[0, input_len - 1, :] = mod_vec
                return act

            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _correct)])
        else:
            with torch.no_grad():
                logits = model(tokens)

        nid = int(logits[0, -1, :].argmax().item())
        if nid == tokenizer.eos_token_id:
            break
        gids.append(nid)
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=device)], dim=1)
        if tokens.shape[1] > 1024:
            break

    return tokenizer.decode(gids).strip(), should_intervene, p_hallucination


def baseline_generate(model, tokenizer, prompt, device, max_new_tokens=20):
    """Normal generation without intervention."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > 1024:
        tokens = tokens[:, :1024]

    gids = []
    for _ in range(max_new_tokens):
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


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════


def evaluate_fcm(
    model,
    tokenizer,
    test_data,
    device,
    layer,
    detector,
    corrector,
    tau=0.3,
    verbose=False,
):
    """Evaluate FactCheckmate intervention on test set."""
    correct = 0
    n_intervened = 0
    total = len(test_data)
    results = []

    # Move models to device
    det = detector.to(device)
    cor = corrector.to(device)

    for sample in tqdm(test_data, desc=f"FCM_τ{tau}", leave=False):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        gt_answers = sample["gt_answers"]

        ans, intervened, p_hall = generate_fcm_intervention(
            model,
            tokenizer,
            prompt,
            device,
            layer,
            det,
            cor,
            tau=tau,
        )
        is_correct = check_correct(ans, gt_answers, dataset="triviaqa")
        if is_correct:
            correct += 1
        if intervened:
            n_intervened += 1
        results.append(
            {
                "correct": is_correct,
                "intervened": intervened,
                "p_hallucination": p_hall,
                "answer": ans[:150],
            }
        )

    det.cpu()
    cor.cpu()

    rate = correct / total
    if verbose:
        print(
            f"  Correct: {correct}/{total} ({rate:.2%}), "
            f"Intervened: {n_intervened}/{total}"
        )
    return rate, n_intervened, results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 12: FactCheckmate Replication")
    parser.add_argument(
        "--load",
        type=str,
        default="../phase9_multi_state/outputs_phase9/phase9_extract.json",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="outputs_phase12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="A,B,C",
        help="Comma-separated corrector variants to test",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    variants = args.variants.split(",")

    print(f"\n{'=' * 65}")
    print(f"Phase 12: FactCheckmate Replication")
    print(f"  Layer: {args.layer}  Test: {args.n_test}  Variants: {variants}")
    print(f"{'=' * 65}\n")

    # ── Load data ──
    print(f"Loading: {args.load}")
    with open(args.load) as f:
        data = json.load(f)
    records = data["records"]
    n_total = len(records)

    indices = np.random.permutation(n_total)
    test_idx = indices[: args.n_test]
    train_idx = indices[args.n_test :]

    li = str(args.layer)
    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    # Extract hidden states
    train_h = np.stack([np.array(r["h"][li]) for r in train_records])
    train_labels = np.array([r["label"] for r in train_records], dtype=np.float32)

    labels_all = np.array([r["label"] for r in records])
    baseline_rate = labels_all[test_idx].mean()

    print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")
    print(
        f"  Test correct: {labels_all[test_idx].sum()}/{len(test_idx)} "
        f"({baseline_rate:.1%})"
    )

    # ── Phase 1: Train detector ──
    print(f"\n{'─' * 50}")
    print("Phase 12.1: Training Detector f_θ")
    print(f"{'─' * 50}")
    detector, det_auroc = train_detector(train_h, train_labels, device=device)
    print(f"  Detector AUROC: {det_auroc:.4f}")
    torch.save(detector.state_dict(), output_dir / "detector.pt")

    # ── Phase 2: Train correctors ──
    print(f"\n{'─' * 50}")
    print("Phase 12.2: Training Correctors g_φ")
    print(f"{'─' * 50}")

    correctors = {}
    for variant in variants:
        cor, val_loss = train_corrector(
            train_h,
            train_labels,
            variant=variant,
            device=device,
        )
        correctors[variant] = cor
        torch.save(cor.state_dict(), output_dir / f"corrector_{variant}.pt")

    # ── Load generation model ──
    print(f"\n{'─' * 50}")
    print("Loading generation model...")
    t0 = time.time()
    gen_model = load_model(device=device, model_id=args.model)
    gen_tokenizer = gen_model.tokenizer
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # ── Phase 3: Evaluate baseline + interventions ──
    print(f"\n{'─' * 50}")
    print("Phase 12.3: Intervention Evaluation")
    print(f"{'─' * 50}")

    # Prepare test data
    test_data = []
    for r in test_records:
        test_data.append(
            {
                "question": r["question"],
                "context": r.get("context", ""),
                "gt_answers": r.get("gt_answers", [r.get("gt_answer", "")]),
                "label": r["label"],
            }
        )

    # Baseline
    print("\n[Baseline] No intervention...")
    t0 = time.time()
    correct_baseline = 0
    for sample in tqdm(test_data, desc="baseline", leave=False):
        prompt = format_prompt(
            sample["question"], sample.get("context", ""), dataset="triviaqa"
        )
        ans = baseline_generate(gen_model, gen_tokenizer, prompt, device)
        if check_correct(ans, sample["gt_answers"], dataset="triviaqa"):
            correct_baseline += 1
    baseline_rate_actual = correct_baseline / len(test_data)
    print(
        f"  Baseline: {correct_baseline}/{len(test_data)} ({baseline_rate_actual:.2%})"
    )

    # Interventions
    all_results = {}
    taus = [0.1, 0.3, 0.5] if not args.quick else [0.3]

    for variant, corrector in correctors.items():
        print(f"\n[FCM-{variant}] FactCheckmate intervention...")
        variant_results = {}
        for tau in taus:
            t0 = time.time()
            rate, n_int, details = evaluate_fcm(
                gen_model,
                gen_tokenizer,
                test_data,
                device,
                args.layer,
                detector,
                corrector,
                tau=tau,
                verbose=True,
            )
            key = f"FCM-{variant}_τ{tau}"
            variant_results[key] = {
                "rate": rate,
                "n_intervened": n_int,
                "time_s": time.time() - t0,
            }
            delta = rate - baseline_rate_actual
            marker = " ↑" if delta > 0.05 else (" ↓" if delta < -0.05 else "")
            print(
                f"    {variant} τ={tau}: {rate:.2%} (Δ={delta:+.1%}) "
                f"[intervened: {n_int}/{len(test_data)}]{marker}"
            )
        all_results.update(variant_results)

    # Also try "always intervene" mode (no gate)
    print(f"\n[FCM-A-always] No detection gate, always intervene...")
    for variant, corrector in correctors.items():
        t0 = time.time()
        rate, n_int, details = evaluate_fcm(
            gen_model,
            gen_tokenizer,
            test_data,
            device,
            args.layer,
            detector,
            corrector,
            tau=1.0,  # tau=1.0 means always < tau → always intervene
            verbose=False,
        )
        key = f"FCM-{variant}_always"
        all_results[key] = {
            "rate": rate,
            "n_intervened": n_int,
            "time_s": time.time() - t0,
        }
        delta = rate - baseline_rate_actual
        print(f"  {variant} always: {rate:.2%} (Δ={delta:+.1%})")

    # ── Summary ──
    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"\n  Detector AUROC: {det_auroc:.4f}")
    print(f"  Baseline: {baseline_rate_actual:.2%}")
    print(f"\n  {'Method':35s} {'Rate':>8s}  {'Δ':>8s}")
    print(f"  {'─' * 53}")

    best_rate = baseline_rate_actual
    best_method = "baseline"

    for method, info in sorted(all_results.items()):
        rate = info["rate"]
        delta = rate - baseline_rate_actual
        marker = " ↑" if delta > 0.05 else (" ↓" if delta < -0.05 else "")
        print(f"  {method:35s} {rate:>8.2%}  {delta:>+8.1%}{marker}")
        if rate > best_rate:
            best_rate = rate
            best_method = method

    print(f"\n  Best: {best_method} ({best_rate:.2%})")

    # ── Save ──
    save_path = output_dir / "phase12_results.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "detector_auroc": det_auroc,
                "baseline_rate": baseline_rate_actual,
                "n_test": len(test_data),
                "n_train": len(train_idx),
                "layer": args.layer,
                "variants": variants,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {save_path}")

    print(f"\n{'=' * 65}")
    print("Phase 12 complete")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
