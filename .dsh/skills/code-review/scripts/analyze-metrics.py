#!/usr/bin/env python3
"""Analyze code for general and DL-specific metrics."""

import re
import sys


def analyze_code_metrics(code):
    """Analyze code for common and deep learning specific metrics."""

    # --- General metrics ---
    functions = len(re.findall(r"^def\s+\w+", code, re.MULTILINE))
    classes = len(re.findall(r"^class\s+\w+", code, re.MULTILINE))

    lines = code.split("\n")
    avg_length = sum(len(l) for l in lines) / len(lines) if lines else 0

    complexity = len(re.findall(r"\b(if|elif|else|for|while|and|or)\b", code))

    # --- DL-specific metrics ---
    cuda_calls = len(re.findall(r"\.cuda\(\)", code))
    to_device = len(re.findall(r"\.to\(\s*['\"]cuda", code))

    detach_calls = len(re.findall(r"\.detach\(\)", code))

    no_grad_ctx = len(re.findall(r"torch\.no_grad\(\)", code))
    autocast_ctx = len(re.findall(r"torch\.(cuda\.)?amp\.autocast", code))
    autocast_decorator = len(re.findall(r"@torch\.(cuda\.)?amp\.autocast", code))

    inplace_ops = len(re.findall(r"\.\w+_\(\)", code))

    shape_asserts = len(re.findall(r"assert\s+\S+\.(shape|size|dim)\b", code))
    nan_checks = len(re.findall(r"torch\.isnan|torch\.isfinite|torch\.isinf", code))
    all_reduce = len(re.findall(r"dist\.all_reduce\(|dist\.all_gather\(|dist\.broadcast\(", code))

    # --- Summary ---
    total_lines = len(lines)
    dl_indicator = bool(
        re.search(r"\b(import torch|from torch|nn\.Module|torch\.nn)\b", code)
    )

    return {
        # General
        "functions": functions,
        "classes": classes,
        "total_lines": total_lines,
        "avg_line_length": avg_length,
        "complexity_score": complexity,
        # DL-specific
        "is_dl_code": dl_indicator,
        "cuda_calls": cuda_calls,
        "to_device_calls": to_device,
        "detach_calls": detach_calls,
        "no_grad_contexts": no_grad_ctx,
        "autocast_contexts": autocast_ctx + autocast_decorator,
        "inplace_ops": inplace_ops,
        "shape_asserts": shape_asserts,
        "nan_checks": nan_checks,
        "dist_collectives": all_reduce,
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        code = f.read()

    metrics = analyze_code_metrics(code)

    print("=" * 50)
    print("  CODE METRICS ANALYSIS")
    print("=" * 50)
    print(f"  File: {sys.argv[1]}")
    print(f"  DL Code: {'Yes' if metrics.pop('is_dl_code') else 'No/Unknown'}")
    print()

    print("--- General ---")
    for key in ("functions", "classes", "total_lines", "avg_line_length", "complexity_score"):
        print(f"  {key}: {metrics[key]:.2f}" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")

    print()
    print("--- DL-Specific ---")
    for key in ("cuda_calls", "to_device_calls", "detach_calls",
                "no_grad_contexts", "autocast_contexts", "inplace_ops",
                "shape_asserts", "nan_checks", "dist_collectives"):
        val = metrics[key]
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
    print("=" * 50)
