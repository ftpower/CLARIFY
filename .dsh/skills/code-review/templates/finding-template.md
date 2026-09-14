# Code Review Finding Template

Use this template when documenting each issue found during code review.

---

## Issue: [TITLE]

### Severity
- [ ] Critical (blocks deployment)
- [ ] High (should fix before merge)
- [ ] Medium (should fix soon)
- [ ] Low (nice to have)

### Category
- [ ] Numerical Stability
- [ ] GPU Memory & Compute
- [ ] Distributed Training
- [ ] Data Pipeline
- [ ] Model Correctness
- [ ] Reproducibility
- [ ] Training Engineering
- [ ] Code Quality
- [ ] Testing

### Location
**File:** `src/training/loss.py`

**Lines:** 45-52

**Function/Method:** `forward()`

### Issue Description

**What:** Describe what the issue is.

**Why it matters:** Explain the impact and why this needs to be fixed.

**Current behavior:** Show the problematic code or behavior.

**Expected behavior:** Describe what should happen instead.

### Code Example

#### Current (Problematic)

```python
# Loss computation without eps guard
loss = -torch.log(probabilities).mean()
```

#### Suggested Fix

```python
# Numerically stable loss with eps clamping
eps = 1e-7
loss = -torch.log(probabilities.clamp(min=eps)).mean()
```

### Impact Analysis

| Aspect | Impact | Severity |
|--------|--------|----------|
| Numerical Stability | NaN propagation under FP16 | Critical |
| Training Correctness | Silent wrong loss values | High |
| Reproducibility | Non-deterministic when NaN hits | Medium |

### Related Issues

- Similar issue in `src/models/contrastive.py` line 120
- Related PR: #456
- Related issue: #789

### Additional Resources

- [PyTorch Numerical Stability Best Practices](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [Mixed Precision Training Guide](https://pytorch.org/docs/stable/amp.html)

### Reviewer Notes

- Check all log-softmax/log-sigmoid usage across the codebase
- Consider adding a NaN monitor callback during training
- PyTorch 2.0+ autocast handles some of this automatically

### Author Response (for feedback)

*To be filled by the code author:*

- [ ] Fix implemented in commit: `abc123`
- [ ] Fix status: Complete / In Progress / Needs Discussion
- [ ] Questions or concerns: (describe)

---

## Finding Statistics (for Reviewer)

When reviewing multiple findings, track:

- **Total Issues Found:** X
- **Critical:** X
- **High:** X
- **Medium:** X
- **Low:** X

**Recommendation:** ✅ Approve / ⚠️ Request Changes / 🔄 Needs Discussion

**Overall Code Quality:** 1-5 stars
