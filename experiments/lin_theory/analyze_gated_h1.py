"""H1 decomposition + over-hype flip-capacity check for gated-TLDC stage 0.

Question asked (2026-08-27): does TLDC lack flip capacity on HIGH-confidence
hallucinations? If the detection score s = 1 - P(correct) is a proxy for
"confidently wrong" (over-hype), then:
  - H1 (theory): rescue rate INCREASES with s (high-conf wrong is over-hype,
    more flippable) -> gating would work.
  - User hypothesis: TLDC is a symmetric, correctness-blind perturbation; the
    most confidently wrong samples have the STABLEST trajectories -> rescue
    rate DECREASES with s. This would be a design flaw: the gate flags exactly
    the samples TLDC cannot flip, which explains gated <= ungated.

This script re-uses the simulate_gated_tldc.py join + out-of-fold probe, then
breaks rescue/break rates down by detection-score quintiles (wrong and correct
samples separately, KW/DK subsets).

Usage:
    python experiments/lin_theory/analyze_gated_h1.py \
      --pair probe_scores/probe_scores_seed123_Qwen3-8B.json experiments/outputs/lin_theory_8b/seed123_8b/s14_tldc_samples.json \
      --pair probe_scores/probe_scores_seed456_Qwen3-8B.json experiments/outputs/lin_theory_8b/seed456_8b/s14_tldc_samples.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from simulate_gated_tldc import BETAS, load_pair  # reuse join logic


def probe_oof_scores(samples, seed=42):
    """5-fold stratified CV logistic probe -> out-of-fold P(correct)."""
    X = np.stack([s["h"] for s in samples])
    y = np.array([s["y0"] for s in samples], dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(samples))
    for tr, va in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        p[va] = clf.predict_proba(X[va])[:, 1]
    return 1.0 - p  # s = P(wrong): high = confidently wrong


def quintile_table(smp, sel, label, betas, flip=False):
    """Rate per detection-score quintile for selected samples.

    flip=True -> break rate = 1 - P(y_b=1 | y0=1) (baseline-correct samples);
    flip=False -> rescue rate = P(y_b=1 | y0=0) (baseline-wrong samples).
    """
    if len(sel) < 10:
        print(f"  {label}: n={len(sel)} too small, skip")
        return
    s = np.array([smp[i]["score"] for i in sel])
    order = np.argsort(s)
    q = np.array_split(order, 5)
    print(f"\n  {label} (n={len(sel)}) — by score quintile (Q1=low s .. Q5=high s):")
    hdr = "    Q    n   s-range      " + "  ".join(f"b={b}" for b in betas)
    print(hdr)
    for qi, idx in enumerate(q, 1):
        sub = [smp[sel[i]] for i in idx]  # idx indexes into sel, not smp
        rates = [np.mean([x["yb"][b] for x in sub]) for b in betas]
        if flip:
            rates = [1.0 - r for r in rates]
        sr = f"{s[idx].min():.2f}-{s[idx].max():.2f}"
        print(f"    Q{qi}  {len(sub):3d}  {sr:12s}  " + "  ".join(f"{r:.3f}" for r in rates))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True, nargs=2,
                    metavar=("SCORES_JSON", "ARCHIVE_JSON"))
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(
        __file__).parent.parent / "outputs" / "lin_theory_8b"
    outdir.mkdir(exist_ok=True, parents=True)

    report = {"pairs": [], "per_seed": {}}
    for sc, ar in args.pair:
        name = Path(sc).stem
        report["pairs"].append(name)
        print("=" * 78)
        print(f"SEED: {name}")
        print("=" * 78)
        smp = load_pair(sc, ar)
        scores = probe_oof_scores(smp)
        for i, s in enumerate(smp):
            s["score"] = float(scores[i])

        wrong = [i for i, s in enumerate(smp) if not s["y0"]]
        correct = [i for i, s in enumerate(smp) if s["y0"]]
        kw = [i for i in wrong if smp[i]["subset"] == "know_wrong"]
        dk = [i for i in wrong if smp[i]["subset"] == "dont_know"]

        print(f"\n  probe AUROC (OOF) = "
              f"{np.mean([int(smp[i]['score'] > 0.5) == (not smp[i]['y0']) for i in range(len(smp))]):.3f} "
              f"(proxy only; real AUROC from detect_lr_probe_cv)")

        # --- H1: rescue rate vs score among baseline-wrong ---
        print("\n>>> H1: RESCUE rate (P(y_b=1 | y0=0)) by score quintile")
        quintile_table(smp, wrong, "ALL baseline-wrong", BETAS)
        quintile_table(smp, kw, "KW only", BETAS)
        quintile_table(smp, dk, "DK only", BETAS)

        # --- H2: break rate vs score among baseline-correct ---
        print("\n>>> H2: BREAK rate (P(y_b=0 | y0=1)) by score quintile")
        quintile_table(smp, correct, "ALL baseline-correct", BETAS, flip=True)

        # --- correlation: score vs flip outcome (delta) ---
        print("\n>>> Spearman corr(s, y_b - y0) over baseline-wrong samples:")
        seed_corr = {}
        for b in BETAS:
            ys = np.array([smp[i]["yb"][b] for i in wrong], dtype=float)
            ss = np.array([smp[i]["score"] for i in wrong])
            rho, p = spearmanr(ss, ys)
            seed_corr[b] = {"rho": float(rho), "p": float(p)}
            print(f"    b={b}: rho={rho:+.3f} p={p:.3f} (n={len(wrong)})")

        report["per_seed"][name] = {
            "n": len(smp), "n_wrong": len(wrong), "n_correct": len(correct),
            "n_kw": len(kw), "n_dk": len(dk),
            "spearman_rescue_vs_score": seed_corr,
        }

    out = outdir / "gated_h1_quintile_8b.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
