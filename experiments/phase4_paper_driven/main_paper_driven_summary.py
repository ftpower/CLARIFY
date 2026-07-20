"""Plan 2 Comprehensive Summary — aggregate all paper-driven experiment results.

Similar to Plan 1's main_comprehensive_analysis.py, but focused on Plan 2's
unique contributions: orthogonality, two-stage detection, functional
differentiation, and cross-scale alignment.

Auto-discovers result JSONs, checks against Plan 2's success/abandon criteria,
generates structured analysis report.

Usage:
    python main_paper_driven_summary.py
    python main_paper_driven_summary.py --include_plan1  # cross-reference Plan 1
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase2_entropy"))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "phase4_generalization"))


# ═══════════════════════════════════════════════════════════════════════════════
# Plan 2 specific criteria
# ═══════════════════════════════════════════════════════════════════════════════

PLAN2_CRITERIA = {
    "orthogonality": {
        "name": "Phase 4.1: Multi-Signal Orthogonality",
        "files": ["orthogonality_analysis_results.json"],
        "success": "Pairwise r < 0.3 for all 3 pairs AND joint > best_single + 2pp",
        "abandon": "Pairwise r > 0.5 (signals redundant) OR joint ≤ best_single",
    },
    "enhanced_js": {
        "name": "Phase 4.2: Enhanced JS Decoding",
        "files": ["enhanced_js_decoding_results.json"],
        "success": "Accuracy Δ > +3pp over greedy baseline",
        "abandon": "All τ/α configurations Δ < +1pp",
    },
    "two_stage": {
        "name": "Phase 4.3: Two-Stage Detection",
        "files": ["two_stage_detection_results.json"],
        "success": "Skip > 50% AND False Skip < 5% AND Joint > Stage2 + 1pp",
        "abandon": "No config meets Skip > 50% with False Skip < 5%",
    },
    "attn_ffn": {
        "name": "Innovation 5: Attn/FFN Differentiation",
        "files": ["attn_ffn_task_results.json"],
        "success": "TriviaQA FFN>Attn AND SQuAD Attn>FFN (both confirmed)",
        "abandon": "Neither hypothesis confirmed (random-level AUROC for both)",
    },
    "cross_scale": {
        "name": "Phase 4.4: Cross-Scale Alignment",
        "files": ["cross_scale_1_7b_bases.npz", "detector_1_7b_for_8b_transfer.json"],
        "success": "1.7B LR → 8B AUROC drop < 10pp",
        "abandon": "Drop > 20pp (features not transferable)",
    },
}


def discover_results(output_dir: Path) -> dict:
    """Discover Plan 2 result files."""
    discovered = {}

    all_files = [
        "orthogonality_analysis_results.json",
        "enhanced_js_decoding_results.json",
        "two_stage_detection_results.json",
        "attn_ffn_task_results.json",
        "cross_scale_1_7b_bases.npz",
        "detector_1_7b_for_8b_transfer.json",
    ]

    for filename in all_files:
        filepath = output_dir / filename
        if filepath.exists():
            stat = filepath.stat()
            try:
                if filename.endswith(".json"):
                    with open(filepath) as f:
                        data = json.load(f)
                else:
                    data = {"file_exists": True, "size_mb": stat.st_size / (1024 * 1024)}
            except (json.JSONDecodeError, IOError):
                data = {"error": "Could not parse"}
            discovered[filename] = {
                "data": data,
                "path": str(filepath),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()[:16],
            }

    return discovered


def check_plan2_criteria(discovered: dict) -> dict:
    """Check each Plan 2 experiment against its criteria."""
    results = {}

    # Orthogonality
    ortho = discovered.get("orthogonality_analysis_results.json", {}).get("data", {})
    ortho_data = ortho.get("orthogonality", {})
    interp = ortho.get("interpretation", {})
    results["orthogonality"] = {
        "status": "success" if interp.get("orthogonal") else
                  ("marginal" if ortho_data else "no_data"),
        "details": [
            f"Orthogonal: {interp.get('orthogonal', 'N/A')}",
            f"Claim 1 supported: {interp.get('claim1_supported', 'N/A')}",
            f"Joint AUROC: {ortho_data.get('joint_all_auroc', 'N/A')}",
        ],
    }

    # Enhanced JS
    js = discovered.get("enhanced_js_decoding_results.json", {}).get("data", {})
    best_js = js.get("best_config", {}) or {}
    results["enhanced_js"] = {
        "status": (
            "success" if best_js.get("delta", 0) > 3.0 else
            "abandon" if js.get("full_sweep") and all(
                r.get("delta", 0) < 1.0 for r in js.get("full_sweep", [])
            ) else
            "marginal" if js else "no_data"
        ),
        "details": [
            f"Best Δ: {best_js.get('delta', 'N/A')}",
            f"Baselines: greedy={js.get('baselines', {}).get('greedy', {}).get('accuracy', 'N/A')}, "
            f"DoLa={js.get('baselines', {}).get('dola_standard', {}).get('accuracy', 'N/A')}",
        ],
    }

    # Two-stage
    ts = discovered.get("two_stage_detection_results.json", {}).get("data", {})
    best_ts = ts.get("best_config") or {}
    criteria_met = ts.get("criteria_met") or {}
    results["two_stage"] = {
        "status": (
            "success" if criteria_met.get("skip_rate_above_50") and
                         criteria_met.get("false_skip_below_5") else
            "marginal" if best_ts else "no_data"
        ),
        "details": [
            f"Skip Rate: {best_ts.get('skip_rate', 'N/A')}",
            f"False Skip: {best_ts.get('false_skip_rate', 'N/A')}",
            f"Joint AUROC: {best_ts.get('joint_auroc', 'N/A')}",
            f"Stage 1 AUROC: {ts.get('stage1', {}).get('auroc_oof', 'N/A')}",
            f"Stage 2 AUROC: {ts.get('stage2', {}).get('auroc_cv', 'N/A')}",
        ],
    }

    # Attn/FFN
    aff = discovered.get("attn_ffn_task_results.json", {}).get("data", {})
    aff_results = aff.get("results", {})
    aff_tests = aff_results.get("hypothesis_tests", {})
    results["attn_ffn"] = {
        "status": (
            "success" if aff_results.get("claim5_supported") else
            "marginal" if aff_tests else "no_data"
        ),
        "details": [
            f"Claim 5 supported: {aff_results.get('claim5_supported', 'N/A')}",
            f"TriviaQA FFN>Attn: {aff_tests.get('triviaqa_ffn_gt_attn', {}).get('supported', 'N/A')} "
            f"(Δ={aff_tests.get('triviaqa_ffn_gt_attn', {}).get('delta', 'N/A')})",
            f"SQuAD Attn>FFN: {aff_tests.get('squad_attn_gt_ffn', {}).get('supported', 'N/A')} "
            f"(Δ={aff_tests.get('squad_attn_gt_ffn', {}).get('delta', 'N/A')})",
        ],
    }

    # Cross-scale
    cs_bases = discovered.get("cross_scale_1_7b_bases.npz", {})
    cs_detector = discovered.get("detector_1_7b_for_8b_transfer.json", {})
    results["cross_scale"] = {
        "status": (
            "pending" if cs_bases and cs_detector else "no_data"
        ),
        "details": [
            f"1.7B bases saved: {bool(cs_bases)}",
            f"1.7B detector saved: {bool(cs_detector)}",
            "8B validation pending — run on AutoDL RTX 5090",
        ],
    }

    return results


def generate_markdown_report(discovered: dict, criteria: dict) -> str:
    """Generate Plan 2 comprehensive analysis report."""

    lines = []
    lines.append("# Phase 4 Plan 2 — Comprehensive Analysis Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results found**: {len(discovered)}/{len(PLAN2_CRITERIA)}")

    # Data availability
    lines.append("\n---")
    lines.append("\n## 1. Data Availability\n")
    lines.append("| Experiment | Status | Modified |")
    lines.append("|-----------|--------|----------|")
    for key, info in PLAN2_CRITERIA.items():
        found = any(f in discovered for f in info["files"])
        mod = "—"
        for f in info["files"]:
            if f in discovered:
                mod = discovered[f].get("modified", "?")
                break
        lines.append(f"| {info['name']} | {'✅' if found else '❌'} | {mod} |")

    # Criteria check
    lines.append("\n---")
    lines.append("\n## 2. Success Criteria\n")
    lines.append("| Experiment | Status | Verdict |")
    lines.append("|-----------|--------|---------|")

    icons = {"success": "✅ PASS", "abandon": "🛑 ABANDON", "marginal": "⚠️ MARGINAL",
             "no_data": "⬜ NO DATA", "pending": "⏳ PENDING"}

    for key, cr in criteria.items():
        status = cr.get("status", "unknown")
        lines.append(f"| {PLAN2_CRITERIA[key]['name']} | {icons.get(status, '❓')} | "
                     f"{PLAN2_CRITERIA[key]['success']} |")

    # Detailed diagnostics
    lines.append("\n---")
    lines.append("\n## 3. Detailed Diagnostics\n")
    for key, cr in criteria.items():
        lines.append(f"\n### {PLAN2_CRITERIA[key]['name']}\n")
        lines.append(f"**Status**: {icons.get(cr['status'], '❓')}")
        lines.append(f"**Success**: {PLAN2_CRITERIA[key]['success']}")
        lines.append(f"**Abandon if**: {PLAN2_CRITERIA[key]['abandon']}")
        for d in cr.get("details", []):
            lines.append(f"- {d}")

    # Claims
    lines.append("\n---")
    lines.append("\n## 4. Paper Claims Status\n")
    claims = [
        ("Claim 1: Multi-Signal Orthogonality", "orthogonality"),
        ("Claim 2: Self-Contained Layer-Pair Decoding", "enhanced_js"),
        ("Claim 3: Two-Stage Detection Framework", "two_stage"),
        ("Claim 4: Cross-Scale Generalization", "cross_scale"),
        ("Claim 5: Attn/FFN Functional Differentiation", "attn_ffn"),
    ]
    for claim_name, crit_key in claims:
        status = criteria.get(crit_key, {}).get("status", "no_data")
        icon = {"success": "✅", "marginal": "⚠️", "no_data": "⬜",
                "pending": "⏳"}.get(status, "❓")
        lines.append(f"- {icon} **{claim_name}**: {status}")

    # Next steps
    lines.append("\n---")
    lines.append("\n## 5. Next Steps\n")

    n_complete = sum(
        1 for cr in criteria.values()
        if cr["status"] in ("success", "abandon")
    )
    n_pending = sum(
        1 for cr in criteria.values()
        if cr["status"] in ("no_data", "pending")
    )

    if n_pending > 0:
        lines.append(f"1. **Run pending experiments** ({n_pending} remaining):")
        for key, cr in criteria.items():
            if cr["status"] in ("no_data", "pending"):
                lines.append(f"   - {PLAN2_CRITERIA[key]['name']}")
    if n_complete > 0:
        lines.append(f"2. **{n_complete} experiments complete** — consolidate findings")
    lines.append("3. **Cross-reference with Plan 1 results** — identify converging/diverging signals")
    lines.append("4. **Draft manuscript** using confirmed claims as evidence backbone")

    lines.append(f"\n---\n*Report auto-generated by main_paper_driven_summary.py*\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Plan 2 Comprehensive Summary"
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--include_plan1", action="store_true",
                        help="Cross-reference Plan 1 results")
    parser.add_argument("--format", type=str, default="both",
                        choices=["json", "markdown", "both"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Plan 2 — Comprehensive Analysis")
    print("=" * 60)

    # Discover
    discovered = discover_results(output_dir)
    print(f"\nFound {len(discovered)} result files:")
    for fname, info in discovered.items():
        print(f"  ✅ {fname} ({info.get('modified', '?')})")

    missing = [f for f in PLAN2_CRITERIA if not any(
        p in discovered for p in PLAN2_CRITERIA[f]["files"]
    )]
    if missing:
        print(f"\nMissing experiments ({len(missing)}):")
        for m in missing:
            print(f"  ❌ {PLAN2_CRITERIA[m]['name']}")

    # Check criteria
    criteria = check_plan2_criteria(discovered)
    print(f"\n{'=' * 60}")
    print("Criteria Check")
    print(f"{'=' * 60}")
    icons = {"success": "✅", "abandon": "🛑", "marginal": "⚠️",
             "no_data": "⬜", "pending": "⏳"}
    for key, cr in criteria.items():
        print(f"  {icons.get(cr['status'], '❓')} {PLAN2_CRITERIA[key]['name']}: "
              f"{cr['status']}")

    # Generate reports
    if args.format in ("json", "both"):
        json_report = {
            "generated": datetime.now().isoformat(),
            "experiments_found": len(discovered),
            "criteria": {
                key: {"status": cr["status"], "details": cr["details"]}
                for key, cr in criteria.items()
            },
        }
        with open(output_dir / "paper_driven_summary.json", "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"\n✅ JSON: {output_dir / 'paper_driven_summary.json'}")

    if args.format in ("markdown", "both"):
        md = generate_markdown_report(discovered, criteria)
        with open(output_dir / "paper_driven_summary.md", "w") as f:
            f.write(md)
        print(f"✅ Markdown: {output_dir / 'paper_driven_summary.md'}")


if __name__ == "__main__":
    main()
