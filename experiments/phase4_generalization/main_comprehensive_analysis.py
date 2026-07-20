"""P3: Comprehensive Analysis — aggregate all Phase 4 results into evaluation matrices.

Automatically discovers result JSON files from the outputs/ directory, parses
experiment results, builds the full cross-dataset cross-model evaluation matrix,
checks each experiment against predefined success/failure criteria, and generates
a structured analysis report in both JSON and Markdown formats.

Usage:
    python main_comprehensive_analysis.py                          # auto-discover
    python main_comprehensive_analysis.py --output_dir outputs     # specific dir
    python main_comprehensive_analysis.py --include_phase3         # include Phase 3 baselines
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Constants: success / failure criteria from Phase 4 plan
# ═══════════════════════════════════════════════════════════════════════════════

CRITERIA = {
    "part_a_detection": {
        "name": "Part A: Generalization Detection Features",
        "success": "Joint AUROC > 0.94 AND ≥1 new feature AUROC > 0.85",
        "abandon": "All new features AUROC < 0.80 AND joint AUROC ≤ max_p alone",
        "check_fn": "check_part_a",
    },
    "part_b1_subspace": {
        "name": "Part B1: Subspace Alignment Intervention",
        "success": "S1 test Δ > +2pp",
        "abandon": "Max principal angle > 45° OR all Δ < 0",
        "check_fn": "check_part_b1",
    },
    "part_b2_js_decoding": {
        "name": "Part B2: Adaptive JS Decoding",
        "success": "Accuracy Δ > +3pp over greedy baseline",
        "abandon": "All τ/α configurations Δ < +1pp",
        "check_fn": "check_part_b2",
    },
    "part_c_preemptive": {
        "name": "Part C: Preemptive Detection",
        "success": "Preemptive AUROC > 0.70",
        "abandon": "Preemptive AUROC < 0.65 (gap to post-hoc too large)",
        "check_fn": "check_part_c",
    },
    "part_d_8b": {
        "name": "Part D: 8B Cross-Model Validation",
        "success": "1.7B→8B AUROC drop < 10pp",
        "abandon": "1.7B→8B AUROC drop > 20pp (features not transferable)",
        "check_fn": "check_part_d",
    },
}

# Phase 2/3 known baselines (from memory/phase2_entropy_progress.md)
KNOWN_BASELINES = {
    "1.7B": {
        "HellaSwag": {
            "max_p_filtered": 0.905,
            "d2_js_filtered": 0.767,
            "d2_joint_filtered": 0.936,
            "i1_best_delta": 5.56,
            "s1_delta": -1.67,
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-discovery of result files
# ═══════════════════════════════════════════════════════════════════════════════


def discover_results(output_dir: Path) -> dict[str, dict]:
    """Scan output_dir for known result JSON files and load them.

    Returns:
        dict keyed by experiment name, each containing the parsed JSON data
        plus metadata about when the file was generated.
    """
    discovered = {}

    file_patterns = {
        "generalization_features": "generalization_features_results.json",
        "preemptive_detection": "preemptive_detection_results.json",
        "js_adaptive_decoding": "js_adaptive_decoding_results.json",
        "subspace_intervention": "subspace_intervention_results.json",
        "d2_consistency": "d2_consistency_results.json",
        "i1_directions": "i1_directions_results.json",
        "s1_pipeline": "s1_pipeline_results.json",
    }

    # Also look in phase2_entropy outputs for Phase 3 baselines
    phase2_outputs = output_dir.parent / "phase2_entropy" / "outputs"

    search_dirs = [output_dir, phase2_outputs]

    for exp_name, filename in file_patterns.items():
        for search_dir in search_dirs:
            filepath = search_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"  WARNING: Could not parse {filepath}: {e}")
                    continue

                stat = filepath.stat()
                discovered[exp_name] = {
                    "data": data,
                    "path": str(filepath),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "source_dir": search_dir.name,
                }
                break  # Found it, don't check other dirs

    return discovered


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation matrix builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_evaluation_matrix(
    discovered: dict[str, dict],
) -> dict:
    """Build the cross-dataset cross-model evaluation matrix.

    Matrix structure:
        rows = features (max_p, D2 JS, EigenScore, HaloScope ζ, Attn/FFN, Joint LR)
        cols = datasets × models (1.7B HellaSwag, 1.7B TriviaQA, 8B HellaSwag, 8B TriviaQA)
    """
    matrix = {
        "rows": [
            "max_p",
            "d2_js",
            "eigenscore",
            "haloscope_zeta",
            "attn_ffn_ratio",
            "entropy",
            "top5_mass",
            "joint_lr",
        ],
        "columns": [
            {"model": "1.7B", "dataset": "HellaSwag", "mode": "train+CV"},
            {"model": "1.7B", "dataset": "TriviaQA", "mode": "zero-shot"},
            {"model": "1.7B", "dataset": "SQuAD", "mode": "zero-shot"},
            {"model": "8B", "dataset": "HellaSwag", "mode": "zero-shot"},
            {"model": "8B", "dataset": "TriviaQA", "mode": "zero-shot"},
        ],
        "values": {},
        "source": {},
    }

    # Fill from generalization_features results (Part A)
    if "generalization_features" in discovered:
        gf = discovered["generalization_features"]["data"]

        # Per-feature AUROC on HellaSwag filtered
        per_feat = gf.get("per_feature_auroc", {})

        # Map feature names to matrix rows
        feat_map = {
            "max_p_best": "max_p",
            "d2_js_top5": "d2_js",
            "eigenscore": "eigenscore",
            "haloscope_zeta": "haloscope_zeta",
            "attn_ffn_ratio": "attn_ffn_ratio",
            "entropy_best": "entropy",
            "top5_mass_best": "top5_mass",
        }

        for feat_col, mat_row in feat_map.items():
            if feat_col in per_feat:
                key = ("max_p" if mat_row == "max_p" else mat_row, 0)
                if per_feat[feat_col] is not None:
                    matrix["values"][(mat_row, 0)] = per_feat[feat_col]
                    matrix["source"][(mat_row, 0)] = "Part A CV"

        # Joint LR
        joint = gf.get("joint_auroc_cv")
        if joint is not None:
            matrix["values"][("joint_lr", 0)] = joint
            matrix["source"][("joint_lr", 0)] = "Part A 5-fold CV"

        # Cross-dataset zero-shot results
        cross = gf.get("cross_dataset_zero_shot", [])
        dataset_to_col = {"triviaqa": 1, "squad": 2}
        for cr in cross:
            ds_name = cr.get("dataset", "").lower()
            if ds_name in dataset_to_col:
                col = dataset_to_col[ds_name]
                if cr.get("auroc") is not None:
                    matrix["values"][("joint_lr", col)] = cr["auroc"]
                    matrix["source"][("joint_lr", col)] = f"Part A {ds_name} zero-shot"

    # Fill known baselines into column 0
    for feat_key, col_idx in [
        ("max_p_filtered", "max_p"),
        ("d2_js_filtered", "d2_js"),
        ("d2_joint_filtered", "joint_lr"),
    ]:
        baseline_val = KNOWN_BASELINES["1.7B"]["HellaSwag"].get(feat_key)
        if baseline_val is not None:
            # Only use known baseline if Part A didn't already provide a value
            mat_key = (col_idx, 0)
            if mat_key not in matrix["values"]:
                matrix["values"][mat_key] = baseline_val
                matrix["source"][mat_key] = "Phase 3 known baseline"

    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Criteria checking
# ═══════════════════════════════════════════════════════════════════════════════


def check_part_a(discovered: dict, matrix: dict) -> dict:
    """Check Part A success criteria."""
    result = {"status": "unknown", "details": []}

    gf = discovered.get("generalization_features", {}).get("data", {})

    joint_auroc = gf.get("joint_auroc_cv")
    per_feat = gf.get("per_feature_auroc", {})

    if joint_auroc is None:
        result["status"] = "no_data"
        result["details"].append("No joint AUROC found — Part A not yet run?")
        return result

    # Check: ≥1 new feature AUROC > 0.85
    new_features = ["d2_js_top5", "eigenscore", "haloscope_zeta", "attn_ffn_ratio"]
    new_feat_above_085 = [
        f for f in new_features if per_feat.get(f, 0) and per_feat[f] > 0.85
    ]

    result["details"].append(f"Joint AUROC = {joint_auroc:.4f}")
    result["details"].append(
        f"New features > 0.85: {new_feat_above_085 if new_feat_above_085 else 'none'}"
    )
    result["joint_auroc"] = joint_auroc
    result["new_features_above_085"] = new_feat_above_085

    if joint_auroc > 0.94 and len(new_feat_above_085) >= 1:
        result["status"] = "success"
    elif all(
        per_feat.get(f, 0) is None or per_feat.get(f, 0) < 0.80
        for f in new_features
    ):
        max_p_auroc = per_feat.get("max_p_best", 0)
        if joint_auroc <= (max_p_auroc or 0):
            result["status"] = "abandon"
        else:
            result["status"] = "marginal"
    else:
        result["status"] = "marginal"

    return result


def check_part_b1(discovered: dict, matrix: dict) -> dict:
    """Check Part B1 (subspace alignment) criteria."""
    result = {"status": "unknown", "details": []}

    si = discovered.get("subspace_intervention", {}).get("data", {})

    alignment = si.get("alignment", {})
    evaluation = si.get("evaluation", [])
    decision = si.get("decision", {})

    if not alignment and not evaluation:
        result["status"] = "no_data"
        result["details"].append("No subspace intervention results found.")
        return result

    # Check principal angles
    any_above_45 = decision.get("any_max_angle_above_45", False)
    max_angles = {}
    for layer_str, info in alignment.items():
        max_angle = info.get("max_angle_deg", 0)
        max_angles[layer_str] = max_angle
        result["details"].append(f"L{layer_str}: max principal angle = {max_angle:.1f}°")
        if max_angle > 45.0:
            result["details"].append(f"  ⚠ L{layer_str} exceeds 45° threshold")

    result["max_angles"] = max_angles

    # Check best delta
    best_delta = -float("inf")
    for r in evaluation:
        if r.get("direction_type") in ("raw_mean_diff", "pca_aligned"):
            df = r.get("delta_filtered")
            if df is not None and df > best_delta:
                best_delta = df
                result["best_config"] = {
                    "type": r["direction_type"],
                    "layer": r["layer"],
                    "lam": r["lam"],
                    "mode": r["mode"],
                    "delta_filtered": df,
                }

    result["best_delta_filtered"] = best_delta if best_delta > -float("inf") else None
    if result["best_delta_filtered"] is not None:
        result["details"].append(
            f"Best intervention Δf = {result['best_delta_filtered']:+.2f}pp"
        )

    if any_above_45:
        result["status"] = "abandon"
    elif best_delta > 2.0:
        result["status"] = "success"
    elif best_delta < 0:
        result["status"] = "abandon"
    else:
        result["status"] = "marginal"

    return result


def check_part_b2(discovered: dict, matrix: dict) -> dict:
    """Check Part B2 (JS adaptive decoding) criteria."""
    result = {"status": "unknown", "details": []}

    js = discovered.get("js_adaptive_decoding", {}).get("data", {})

    baseline = js.get("baseline", {})
    best = js.get("best_config", {})
    sweep = js.get("full_sweep", [])

    if not sweep:
        result["status"] = "no_data"
        result["details"].append("No JS decoding results found.")
        return result

    baseline_acc = baseline.get("accuracy", 0)
    best_delta = best.get("delta", 0)

    result["baseline_acc"] = baseline_acc
    result["best_delta"] = best_delta
    result["best_config"] = best

    result["details"].append(f"Greedy baseline accuracy: {baseline_acc:.4f}")
    result["details"].append(
        f"Best config: τ={best.get('tau')}, α1={best.get('alpha1')}, "
        f"α2={best.get('alpha2')}, Δ={best_delta:+.4f}"
    )

    if best_delta > 3.0:
        result["status"] = "success"
    elif all(r.get("delta", 0) < 1.0 for r in sweep):
        result["status"] = "abandon"
    else:
        result["status"] = "marginal"

    return result


def check_part_c(discovered: dict, matrix: dict) -> dict:
    """Check Part C (preemptive detection) criteria."""
    result = {"status": "unknown", "details": []}

    pd_data = discovered.get("preemptive_detection", {}).get("data", {})

    preemptive = pd_data.get("preemptive", {})
    post_hoc = pd_data.get("post_hoc", {})
    joint_all = pd_data.get("joint_all_auroc")

    if not preemptive:
        result["status"] = "no_data"
        result["details"].append("No preemptive detection results found.")
        return result

    preemptive_auroc = preemptive.get("auroc_cv_mean")
    post_hoc_auroc = post_hoc.get("joint_post_hoc_auroc")

    result["preemptive_auroc"] = preemptive_auroc
    result["post_hoc_auroc"] = post_hoc_auroc
    result["joint_all_auroc"] = joint_all

    if preemptive_auroc is not None:
        result["details"].append(f"Preemptive MLP AUROC = {preemptive_auroc:.4f}")
    if post_hoc_auroc is not None:
        result["details"].append(f"Post-hoc joint AUROC = {post_hoc_auroc:.4f}")
    if joint_all is not None:
        gain = joint_all - post_hoc_auroc if post_hoc_auroc else float("nan")
        result["details"].append(f"Joint (pre+post) AUROC = {joint_all:.4f} (Δ={gain:+.4f})")

    if preemptive_auroc is None:
        result["status"] = "no_data"
    elif preemptive_auroc > 0.70:
        result["status"] = "success"
    elif preemptive_auroc < 0.65:
        result["status"] = "abandon"
    else:
        result["status"] = "marginal"

    return result


def check_part_d(discovered: dict, matrix: dict) -> dict:
    """Check Part D (8B cross-model) criteria.

    Part D requires AutoDL 5090 — will be NO_DATA until executed.
    """
    result = {"status": "no_data", "details": []}

    # Check for 8B validation results
    d2_8b = discovered.get("d2_consistency", {})

    if not d2_8b:
        result["details"].append("8B validation not yet run (requires AutoDL 5090).")
        result["details"].append("Part D pending — run main_8b_validation.py on AutoDL.")
        return result

    result["details"].append("8B results found but cross-model comparison pending.")
    result["details"].append("Need 1.7B-trained LR → 8B zero-shot evaluation.")
    result["status"] = "pending"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(
    discovered: dict,
    matrix: dict,
    criteria_results: dict,
    output_dir: Path,
) -> str:
    """Generate comprehensive analysis report in Markdown format."""

    lines = []
    lines.append("# Phase 4 Plan 1 — Comprehensive Analysis Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results directory**: `{output_dir}`")
    lines.append(f"**Experiments found**: {len(discovered)}/{len(CRITERIA) + 2}")

    # ── Section 1: Data Availability ──
    lines.append("\n---")
    lines.append("\n## 1. Data Availability\n")
    lines.append("| Experiment | Status | Source | Last Modified |")
    lines.append("|-----------|--------|--------|--------------|")

    all_experiments = [
        ("generalization_features", "Part A: Generalization Features"),
        ("preemptive_detection", "Part C: Preemptive Detection"),
        ("js_adaptive_decoding", "Part B2: JS Adaptive Decoding"),
        ("subspace_intervention", "Part B1: Subspace Intervention"),
        ("d2_consistency", "Phase 3: D2 Consistency (baseline)"),
        ("i1_directions", "Phase 3: I1 Directions (baseline)"),
        ("s1_pipeline", "Phase 3: S1 Pipeline (baseline)"),
    ]

    for key, label in all_experiments:
        if key in discovered:
            info = discovered[key]
            mod = info.get("modified", "?")[:16]
            lines.append(f"| {label} | ✅ Found | `{info['source_dir']}/` | {mod} |")
        else:
            lines.append(f"| {label} | ❌ Not found | — | — |")

    # ── Section 2: Evaluation Matrix ──
    lines.append("\n---")
    lines.append("\n## 2. Evaluation Matrix\n")

    # Header
    col_headers = [f"{c['model']} {c['dataset']}<br>({c['mode']})" for c in matrix["columns"]]
    lines.append("| Feature | " + " | ".join(col_headers) + " |")
    lines.append("|---------|" + "|".join(["---------" for _ in col_headers]) + "|")

    for row_name in matrix["rows"]:
        cells = []
        for col_idx in range(len(matrix["columns"])):
            key = (row_name, col_idx)
            val = matrix["values"].get(key)
            src = matrix["source"].get(key, "")

            if val is not None:
                # Mark known baselines differently
                if "known baseline" in src:
                    cells.append(f"*{val:.4f}*")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")

    # ── Section 3: Success Criteria Check ──
    lines.append("\n---")
    lines.append("\n## 3. Success Criteria\n")
    lines.append("| Experiment | Status | Key Metric | Verdict |")
    lines.append("|-----------|--------|------------|---------|")

    status_icons = {
        "success": "✅ PASS",
        "abandon": "🛑 ABANDON",
        "marginal": "⚠️ MARGINAL",
        "no_data": "⬜ NO DATA",
        "pending": "⏳ PENDING",
        "unknown": "❓ UNKNOWN",
    }

    part_labels = {
        "part_a_detection": ("Part A: Detection Features", "joint_auroc"),
        "part_b1_subspace": ("Part B1: Subspace Alignment", "best_delta_filtered"),
        "part_b2_js_decoding": ("Part B2: JS Decoding", "best_delta"),
        "part_c_preemptive": ("Part C: Preemptive Detection", "preemptive_auroc"),
        "part_d_8b": ("Part D: 8B Validation", None),
    }

    for crit_key, (label, metric_key) in part_labels.items():
        cr = criteria_results.get(crit_key, {})
        status = cr.get("status", "unknown")
        icon = status_icons.get(status, "❓")

        if metric_key and cr.get(metric_key) is not None:
            metric_str = f"{cr[metric_key]:.4f}"
        elif metric_key:
            metric_str = "N/A"
        else:
            metric_str = "—"

        lines.append(f"| {label} | {icon} | {metric_str} | {CRITERIA[crit_key]['success']} |")

    # ── Section 4: Detailed Diagnostics ──
    lines.append("\n---")
    lines.append("\n## 4. Detailed Diagnostics\n")

    for crit_key, cr in criteria_results.items():
        lines.append(f"\n### {CRITERIA[crit_key]['name']}\n")
        lines.append(f"**Status**: {status_icons.get(cr.get('status', 'unknown'), '❓')}")
        lines.append(f"**Success**: {CRITERIA[crit_key]['success']}")
        lines.append(f"**Abandon if**: {CRITERIA[crit_key]['abandon']}")

        details = cr.get("details", [])
        if details:
            lines.append("")
            for d in details:
                lines.append(f"- {d}")

    # ── Section 5: Recommendations ──
    lines.append("\n---")
    lines.append("\n## 5. Actionable Recommendations\n")

    recommendations = []

    for crit_key, cr in criteria_results.items():
        status = cr.get("status", "unknown")
        name = CRITERIA[crit_key]["name"]

        if status == "success":
            recommendations.append(
                f"✅ **{name}**: Criteria met. Proceed with confidence. "
                f"Integrate into Phase 4 final pipeline."
            )
        elif status == "abandon":
            recommendations.append(
                f"🛑 **{name}**: Abandon criteria triggered. "
                f"Stop further experiments on this track. Document negative result."
            )
        elif status == "marginal":
            recommendations.append(
                f"⚠️ **{name}**: Marginal — neither clear success nor clear failure. "
                f"Consider: increase sample size, test on 8B for stronger signal, "
                f"or run additional ablation to isolate the bottleneck."
            )
        elif status == "no_data":
            recommendations.append(
                f"⬜ **{name}**: No data yet. Run the corresponding experiment script."
            )
        elif status == "pending":
            recommendations.append(
                f"⏳ **{name}**: Partially complete. Finish remaining steps."
            )

    for rec in recommendations:
        lines.append(f"- {rec}")

    # ── Section 6: Next Steps ──
    lines.append("\n---")
    lines.append("\n## 6. Next Steps\n")

    # Determine which phases are complete
    p0_complete = all(
        criteria_results.get(k, {}).get("status") != "no_data"
        for k in ["part_a_detection", "part_c_preemptive", "part_b2_js_decoding"]
    )
    p1_complete = criteria_results.get("part_b1_subspace", {}).get("status") != "no_data"

    if not p0_complete:
        lines.append("1. **Finish P0**: Run remaining P0 experiments on local RTX 5060")
    if p0_complete and not p1_complete:
        lines.append("1. **Run P1**: Execute `main_subspace_intervention.py` locally")
    if p0_complete and p1_complete:
        lines.append("1. **P0/P1 Complete** ✅ — Move to P2 on AutoDL RTX 5090")
        lines.append("2. Run `main_8b_validation.py` with new features on 8B model")
        lines.append("3. Re-run P3 analysis to include 8B cross-model results")

    lines.append("\n---")
    lines.append(f"\n*Report auto-generated by main_comprehensive_analysis.py*\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="P3: Comprehensive Analysis — aggregate Phase 4 results"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--include_phase3", action="store_true",
        help="Also search phase2_entropy/outputs for Phase 3 baselines",
    )
    parser.add_argument(
        "--report_format", type=str, default="both",
        choices=["json", "markdown", "both"],
        help="Output format(s) for the analysis report",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3_dir = output_dir.parent / "phase2_entropy" / "outputs"

    print("=" * 60)
    print("P3: Comprehensive Analysis — Phase 4 Results Aggregation")
    print("=" * 60)

    # ── Step 1: Discover results ──
    print(f"\nScanning for result files in:")
    print(f"  {output_dir.resolve()}")
    if args.include_phase3:
        print(f"  {phase3_dir.resolve()}")
    else:
        print(f"  (Phase 3 baselines from {phase3_dir.resolve()} — skipped, use --include_phase3)")

    discovered = discover_results(output_dir)
    print(f"\nFound {len(discovered)} result files:")

    for exp_name, info in sorted(discovered.items()):
        print(f"  ✅ {exp_name}: {info['path']}")
        print(f"     ({info['source_dir']}, modified {info['modified'][:16]})")

    # List missing
    all_expected = [
        "generalization_features", "preemptive_detection",
        "js_adaptive_decoding", "subspace_intervention",
    ]
    missing = [e for e in all_expected if e not in discovered]
    if missing:
        print(f"\nMissing ({len(missing)}):")
        for m in missing:
            print(f"  ❌ {m} — run the corresponding main_*.py script first")

    # ── Step 2: Build evaluation matrix ──
    print(f"\n{'=' * 60}")
    print("Building Evaluation Matrix")
    print("=" * 60)

    matrix = build_evaluation_matrix(discovered)

    # Print matrix to console
    col_headers = [f"{c['model']}\n{c['dataset']}" for c in matrix["columns"]]
    print(f"\n{'Feature':<16}", end="")
    for ch in col_headers:
        print(f"{ch:>16}", end="")
    print()
    print("-" * (16 + 16 * len(col_headers)))

    for row_name in matrix["rows"]:
        print(f"{row_name:<16}", end="")
        for col_idx in range(len(matrix["columns"])):
            val = matrix["values"].get((row_name, col_idx))
            if val is not None:
                print(f"{val:>16.4f}", end="")
            else:
                print(f"{'—':>16}", end="")
        print()

    # ── Step 3: Check criteria ──
    print(f"\n{'=' * 60}")
    print("Checking Success Criteria")
    print("=" * 60)

    criteria_results = {}
    for crit_key, crit_info in CRITERIA.items():
        check_fn_name = crit_info["check_fn"]
        check_fn = globals()[check_fn_name]
        cr = check_fn(discovered, matrix)
        criteria_results[crit_key] = cr

        icon = {"success": "✅", "abandon": "🛑", "marginal": "⚠️",
                "no_data": "⬜", "pending": "⏳", "unknown": "❓"}.get(cr["status"], "❓")
        print(f"  {icon} {crit_info['name']}: {cr['status']}")

    # ── Step 4: Generate report ──
    print(f"\n{'=' * 60}")
    print("Generating Report")
    print("=" * 60)

    if args.report_format in ("json", "both"):
        json_report = {
            "generated": datetime.now().isoformat(),
            "output_dir": str(output_dir.resolve()),
            "experiments_found": len(discovered),
            "experiments_expected": len(all_expected),
            "missing_experiments": missing,
            "evaluation_matrix": {
                "rows": matrix["rows"],
                "columns": [
                    {"model": c["model"], "dataset": c["dataset"], "mode": c["mode"]}
                    for c in matrix["columns"]
                ],
                "values": {
                    f"{row},{col}": val
                    for (row, col), val in matrix["values"].items()
                },
                "sources": {
                    f"{row},{col}": src
                    for (row, col), src in matrix["source"].items()
                },
            },
            "criteria": {
                key: {
                    "status": cr["status"],
                    "details": cr.get("details", []),
                    "metrics": {
                        k: v for k, v in cr.items()
                        if k not in ("status", "details")
                    },
                }
                for key, cr in criteria_results.items()
            },
        }

        json_path = output_dir / "comprehensive_analysis.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"  ✅ JSON report: {json_path}")

    if args.report_format in ("markdown", "both"):
        md_report = generate_report(discovered, matrix, criteria_results, output_dir)
        md_path = output_dir / "comprehensive_analysis.md"
        with open(md_path, "w") as f:
            f.write(md_report)
        print(f"  ✅ Markdown report: {md_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)

    status_counts = {}
    for cr in criteria_results.values():
        s = cr.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    for status, icon in [
        ("success", "✅"), ("marginal", "⚠️"), ("abandon", "🛑"),
        ("no_data", "⬜"), ("pending", "⏳"),
    ]:
        count = status_counts.get(status, 0)
        if count > 0:
            print(f"  {icon} {status}: {count} experiment(s)")

    n_abandon = status_counts.get("abandon", 0)
    n_success = status_counts.get("success", 0)
    n_no_data = status_counts.get("no_data", 0)

    if n_abandon > 0:
        print(f"\n⚠ {n_abandon} experiment(s) triggered abandon criteria.")
        print("  Review detailed diagnostics in the report before proceeding.")

    if n_no_data > 0:
        print(f"\n⬜ {n_no_data} experiment(s) have no data yet.")
        print("  Run the corresponding scripts before re-running P3 analysis.")

    print(f"\nReports saved to: {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
