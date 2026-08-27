"""Post-hoc simulation of gated TLDC (detect-then-intervene) on 8B.

Theory: docs/theory-gated-tldc.md §5.2 (stage 0). Read-only; no model needed.

Pipeline simulated:
    decode baseline a0 (outcome y0 = baseline_correct)
    gate score s = 1 - P(correct | h_L28) from a 5-fold CV logistic probe
    if s >= tau: use TLDC outcome y_b (correct_betaX) else keep y0
    y_G = y_b * [s >= tau] + y0 * [s < tau]

Inputs (one --pair per seed, repeatable):
    probe_scores_seed{123,456}_Qwen3-8B.json   (from extract_tldc_probe_scores.py)
    s14_tldc_samples.json                      (per-sample TLDC archive)

Join key: (sample_id, question). The archive's baseline_correct is
cross-checked against the scores file's is_correct.

Outputs: console tables + <outdir>/gated_simulation_8b.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BETAS = ["0.01", "0.03", "0.05", "0.08", "0.10", "0.15", "0.20"]
TAUS = np.arange(0.30, 0.81, 0.05)


def cp95(k, n):
    """Exact Clopper-Pearson 95% CI of a rate k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def load_pair(scores_path, archive_path):
    sc = json.load(open(scores_path, encoding="utf-8"))
    ar = json.load(open(archive_path, encoding="utf-8"))
    by_key = {}
    for e in sc["entries"]:
        by_key[(e["sample_id"], e["question"])] = e
    samples = []
    mism = 0
    for sid, rec in ar["samples"].items():
        key = (int(sid), rec["question"])
        if key not in by_key:
            raise KeyError(f"archive entry {key} missing in scores file")
        e = by_key[key]
        if bool(e["is_correct"]) != bool(rec["baseline_correct"]):
            mism += 1
        samples.append({
            "sample_id": int(sid),
            "question": rec["question"],
            "subset": rec["subset"],
            "y0": bool(rec["baseline_correct"]),
            "h": np.array(e["h"], dtype=np.float32),
            "yb": {b: bool(rec[f"correct_beta{b}"]) for b in BETAS},
        })
    print(f"  joined {len(samples)} samples | baseline-label mismatches vs archive: {mism}")
    return samples


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
    return 1.0 - p  # s = P(wrong)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True, nargs=2,
                    metavar=("SCORES_JSON", "ARCHIVE_JSON"),
                    help="repeatable: probe scores file + TLDC per-sample archive")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(
        __file__).parent.parent / "outputs" / "lin_theory_8b"
    outdir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("Gated-TLDC simulation (theory-gated-tldc.md §5.2)")
    print("=" * 70)

    all_samples = []
    pair_names = []
    for sc, ar in args.pair:
        name = f"{Path(sc).stem}|{Path(ar).parent.name}"
        pair_names.append(name)
        print(f"[load] {name}")
        smp = load_pair(sc, ar)
        scores = probe_oof_scores(smp)
        for i, s in enumerate(smp):
            s["score"] = float(scores[i])
        all_samples.append(smp)

    # ---------- per-pair and pooled ----------
    groups = [(pair_names[i], all_samples[i]) for i in range(len(all_samples))]
    if len(all_samples) > 1:
        groups.append(("POOLED", [s for smp in all_samples for s in smp]))

    summary = {"pairs": pair_names, "groups": {}}
    for gname, smp in groups:
        scores = np.array([s["score"] for s in smp])
        n = len(smp)
        base_correct = int(sum(s["y0"] for s in smp))
        rows = []
        print(f"\n=== group {gname} (n={n}, baseline correct {base_correct}/{n} = {100*base_correct/n:.1f}%) ===")

        for b in BETAS:
            yb = np.array([s["yb"][b] for s in smp], dtype=bool)
            y0 = np.array([s["y0"] for s in smp], dtype=bool)
            # ungated = all flagged
            ung = int((yb & ~y0).sum()) - int((~yb & y0).sum())
            # oracle = flag exactly baseline-wrong
            yG_oracle = np.where(~y0, yb, y0)
            ora = int(yG_oracle.sum()) - int(y0.sum())
            # tau sweep
            best = None
            for tau in TAUS:
                flagged = scores >= tau
                yG = np.where(flagged, yb, y0)
                net = int(yG.sum()) - int(y0.sum())
                if best is None or net > best[1]:
                    best = (tau, net)
            tau_star, net_best = best
            # breakdown at tau_star
            flagged = scores >= tau_star
            yG = np.where(flagged, yb, y0)
            kw_idx = [i for i, s in enumerate(smp) if s["subset"] == "know_wrong"]
            kc_idx = [i for i, s in enumerate(smp) if s["subset"] == "know_correct"]
            kw_resc = int((yG[kw_idx] & ~y0[kw_idx]).sum()) if kw_idx else 0
            kc_brk = int((~yG[kc_idx] & y0[kc_idx]).sum()) if kc_idx else 0
            n_kw, n_kc = len(kw_idx), len(kc_idx)
            recall = float((~y0[flagged]).sum() / max(1, (~y0).sum()))
            spec = float((y0[~flagged]).sum() / max(1, y0.sum()))
            rows.append(dict(beta=b, ungated_net=ung, oracle_net=ora,
                             tau_star=float(tau_star), gated_net=int(net_best),
                             kw_rescued=int(kw_resc), kw_n=n_kw,
                             kc_broken=int(kc_brk), kc_n=n_kc,
                             recall=recall, spec=spec))
            print(f"  b={b}: ungated {ung:+d} | oracle {ora:+d} | "
                  f"gated {net_best:+d} @tau={tau_star:.2f} "
                  f"(KW {kw_resc}/{n_kw}, KC -{kc_brk}/{n_kc}, recall {recall:.2f}, spec {spec:.2f})")
        # H1 decomposition: baseline-wrong samples by score tercile
        wrong = [s for s in smp if not s["y0"]]
        if len(wrong) >= 15:
            wrong = sorted(wrong, key=lambda s: s["score"])
            t = np.array_split(np.arange(len(wrong)), 3)
            print("  H1 check (baseline-wrong, score terciles -> rescue rate per b):")
            line = "    tercile  n   " + "  ".join(f"b={b}" for b in BETAS)
            print(line)
            for ti, lab in zip(t, ("low", "mid", "high")):
                sub = [wrong[i] for i in ti]
                rates = []
                for b in BETAS:
                    r = np.mean([s["yb"][b] for s in sub])
                    rates.append(f"{r:.2f}")
                print(f"    {lab:7s} {len(sub):3d}   " + "   ".join(rates))
        summary["groups"][gname] = {"n": n, "baseline_correct": base_correct, "rows": rows}

    out = outdir / "gated_simulation_8b.json"
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
