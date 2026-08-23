---
name: code-review
description: Comprehensive code review for deep learning projects (PyTorch, training scripts, model implementations). Covers numerical stability, GPU memory, distributed training, data pipeline, model correctness, reproducibility, training engineering, code quality, and testing. Use when users ask to review code, analyze code quality, evaluate pull requests, or mention code review, security analysis, or performance optimization.
user-invocable: true
---
# Code Review Skill (Deep Learning)

This skill provides comprehensive code review capabilities tailored for deep learning projects, focusing on:

1. **Numerical Stability**
   - NaN/Inf detection and propagation paths
   - Softmax / log_softmax numerical safety (log-sum-exp trick)
   - Mixed precision (FP16/BF16) underflow/overflow risks
   - Epsilon values in normalization layers (LayerNorm, BatchNorm)
   - Loss computation edge cases (e.g., chunked loss paths)

2. **GPU Memory & Compute Efficiency**
   - Tensor lifecycle management (detach, retain_graph)
   - In-place operation compatibility with autograd
   - Gradient accumulation logic (accumulation steps, loss scaling)
   - CPU-GPU data transfer minimization
   - Memory fragmentation from dynamic shapes

3. **Distributed Training Correctness**
   - all_reduce / all_gather / broadcast usage correctness
   - Global batch size calculation
   - Rank-dependent logic (no race conditions)
   - Gradient synchronization timing (DDP find_unused_parameters)
   - FSDP / DeepSpeed configuration

4. **Data Pipeline**
   - DataLoader num_workers and multi-process safety
   - Shuffle randomness and seed control
   - Transform consistency across ranks
   - Dataset __getitem__ implementation correctness
   - Data prefetching and caching strategy

5. **Model Correctness**
   - Loss function implementation correctness
   - Gradient clipping (max_norm value, clip timing)
   - Weight initialization method selection
   - Tensor shape assertions / checks
   - model.train() / model.eval() switching

6. **Reproducibility**
   - Random seed completeness (torch, numpy, random, cuda)
   - cudnn deterministic / benchmark settings
   - Checkpoint save/load integrity (optimizer, scheduler, scaler)
   - Data ordering determinism

7. **Training Loop Engineering**
   - Checkpoint save and resume logic
   - EMA update timing and decay factor
   - LR scheduler step timing (per-step vs per-epoch)
   - autocast scope and GradScaler usage
   - Logging and metrics computation (avoid duplicate logs across ranks)

8. **Code Quality**
   - Function length (< 50 lines)
   - Clear variable naming
   - No duplicate code
   - Type annotations (Python typing)
   - Comments explain WHY, not WHAT

9. **Testing**
   - Shape correctness tests
   - Numerical gradient checks (torch.autograd.gradcheck)
   - Loss value sanity checks (known input → expected output)
   - Overfitting test on small batch (model can learn)
   - Determinism test (same seed → same result)

## Reference Files

This skill includes supporting files that you should read when performing reviews:

- **`templates/review-checklist.md`** — Structured checklist covering numerical stability, GPU memory, distributed training, data pipeline, model correctness, reproducibility, training engineering, code quality, and testing. Read this file and use it as a guide to ensure no category is missed during review.
- **`templates/finding-template.md`** — Standard template for documenting individual findings with severity, location, code examples, and impact analysis. Read this file and use its format when reporting issues.
- **`scripts/analyze-metrics.py`** — Python script that calculates both general code metrics (function count, class count, complexity score) and DL-specific metrics (.cuda() calls, .detach() usage, autocast/no_grad contexts, in-place ops, tensor shape asserts). Run this on the file under review to gather quantitative data.
- **`scripts/compare-complexity.py`** — Python script that compares cyclomatic and cognitive complexity between two versions of a file. Run this with the before and after versions when reviewing refactoring changes.

## Review Template

For each piece of code reviewed, provide:

### Summary
- Overall quality assessment (1-5)
- Key findings count
- Recommended priority areas

### Critical Issues (if any)
- **Issue**: Clear description
- **Location**: File and line number
- **Impact**: Why this matters
- **Severity**: Critical/High/Medium
- **Fix**: Code example

### Findings by Category

#### Numerical Stability (if issues found)
NaN/Inf risks, mixed precision concerns, eps values, loss edge cases

#### GPU Memory & Compute (if issues found)
Tensor lifecycle issues, in-place op problems, gradient accumulation bugs, CPU-GPU transfer waste

#### Distributed Training (if issues found)
Collective communication errors, global batch size miscalculation, rank races, sync issues

#### Data Pipeline (if issues found)
DataLoader problems, shuffle/seed issues, transform inconsistencies, dataset bugs

#### Model Correctness (if issues found)
Loss function errors, gradient clipping issues, initialization problems, shape mismatches

#### Reproducibility (if issues found)
Missing seeds, non-deterministic ops, checkpoint corruption risks

#### Training Engineering (if issues found)
Resume bugs, EMA errors, LR schedule mistakes, autocast/scaler misuse, logging issues

#### Code Quality (if issues found)
Naming, duplication, type annotations, documentation

#### Testing (if issues found)
Missing shape/gradient/loss/overfit/determinism tests

## Version History

- v2.0.0 (2026-05-07): Rewritten for deep learning projects — replaced web-focused Security/Performance dimensions with Numerical Stability, GPU Memory, Distributed Training, Data Pipeline, Model Correctness, Reproducibility, and Training Engineering
- v1.0.0 (2024-12-10): Initial release with security, performance, quality, and maintainability analysis
