---
name: refactor
description: Systematic code refactoring for deep learning projects based on Martin Fowler's methodology. Covers DL-specific code smells (Training Loop Bloat, Missing Mode Switch, Checkpoint Fragility, Autograd Leak, etc.) and refactoring techniques (Extract Loss Class, Parameterize Model Config, Add Shape Guards, etc.). Use when users ask to refactor code, improve code structure, reduce technical debt, clean up legacy code, eliminate code smells, or improve code maintainability.
user-invocable: true
---
# Code Refactoring Skill

A systematic approach to refactoring code based on Martin Fowler's *Refactoring: Improving the Design of Existing Code* (2nd Edition). This skill emphasizes safe, incremental changes backed by tests.

> "Refactoring is the process of changing a software system in such a way that it does not alter the external behavior of the code yet improves its internal structure." — Martin Fowler

## Core Principles

1. **Behavior Preservation**: External behavior must remain unchanged
2. **Small Steps**: Make tiny, testable changes
3. **Test-Driven**: Tests are the safety net
4. **Continuous**: Refactoring is ongoing, not a one-time event
5. **Collaborative**: User approval required at each phase

## Workflow Overview

```
Phase 1: Research & Analysis
    ↓
Phase 2: Test Coverage Assessment
    ↓
Phase 3: Code Smell Identification
    ↓
Phase 4: Refactoring Plan Creation
    ↓
Phase 5: Incremental Implementation
    ↓
Phase 6: Review & Iteration
```

---

## Phase 1: Research & Analysis

### Objectives
- Understand the codebase structure and purpose
- Identify the scope of refactoring
- Gather context about business requirements

### Questions to Ask User
Before starting, clarify:

1. **Scope**: Which files/modules/functions need refactoring?
2. **Goals**: What problems are you trying to solve? (readability, performance, maintainability)
3. **Constraints**: Are there any areas that should NOT be changed?
4. **Timeline pressure**: Is this blocking other work?
5. **Test status**: Do tests exist? Are they passing?

### Actions
- [ ] Read and understand the target code
- [ ] Identify dependencies and integrations
- [ ] Document current architecture
- [ ] Note any existing technical debt markers (TODOs, FIXMEs)

### Output
Present findings to user:
- Code structure summary
- Identified problem areas
- Initial recommendations
- **Request approval to proceed**

---

## Phase 2: Test Coverage Assessment

### Why Tests Matter
> "Refactoring without tests is like driving without a seatbelt." — Martin Fowler

Tests are the **key enabler** of safe refactoring. Without them, you risk introducing bugs.

### Assessment Steps

1. **Check for existing tests**
   ```bash
   # Look for test files
   find . -name "*test*" -o -name "*spec*" | head -20
   ```

2. **Run existing tests**
   ```bash
   # JavaScript/TypeScript
   npm test

   # Python
   pytest -v

   # Java
   mvn test
   ```

3. **Check coverage (if available)**
   ```bash
   # JavaScript
   npm run test:coverage

   # Python
   pytest --cov=.
   ```

### Decision Point: Ask User

**If tests exist and pass:**
- Proceed to Phase 3

**If tests are missing or incomplete:**
Present options:
1. Write tests first (recommended)
2. Add tests incrementally during refactoring
3. Proceed without tests (risky - requires user acknowledgment)

**If tests are failing:**
- STOP. Fix failing tests before refactoring
- Ask user: Should we fix tests first?

### Test Writing Guidelines (if needed)

For each function being refactored, ensure tests cover:
- Happy path (normal operation)
- Edge cases (empty inputs, null, boundaries)
- Error scenarios (invalid inputs, exceptions)

Use the "red-green-refactor" cycle:
1. Write failing test (red)
2. Make it pass (green)
3. Refactor

---

## Phase 3: Code Smell Identification

### What Are Code Smells?
Symptoms of deeper problems in code. They're not bugs, but indicators that the code could be improved.

### Common Code Smells to Check

See [references/code-smells.md](references/code-smells.md) for the complete catalog.

#### Quick Reference

| Smell | Signs | Impact |
|-------|-------|--------|
| **Long Method** | Methods > 30-50 lines | Hard to understand, test, maintain |
| **Duplicated Code** | Same logic in multiple places | Bug fixes needed in multiple places |
| **Large Class** | Class with too many responsibilities | Violates Single Responsibility |
| **Feature Envy** | Method uses another class's data more | Poor encapsulation |
| **Primitive Obsession** | Overuse of primitives instead of objects | Missing domain concepts |
| **Long Parameter List** | Methods with 4+ parameters | Hard to call correctly |
| **Data Clumps** | Same data items appearing together | Missing abstraction |
| **Switch Statements** | Complex switch/if-else chains | Hard to extend |
| **Speculative Generality** | Code "just in case" | Unnecessary complexity |
| **Dead Code** | Unused code | Confusion, maintenance burden |

### Analysis Steps

1. **Automated Analysis** (if scripts available)
   ```bash
   python scripts/detect-smells.py <file>
   ```

2. **Manual Review**
   - Walk through code systematically
   - Note each smell with location and severity
   - Categorize by impact (Critical/High/Medium/Low)

3. **Prioritization**
   Focus on smells that:
   - Block current development
   - Cause bugs or confusion
   - Affect most-changed code paths

### Output: Smell Report

Present to user:
- List of identified smells with locations
- Severity assessment for each
- Recommended priority order
- **Request approval on priorities**

---

## Phase 4: Refactoring Plan Creation

### Selecting Refactorings

For each smell, select an appropriate refactoring from the catalog.

See [references/refactoring-catalog.md](references/refactoring-catalog.md) for the complete list.

#### Smell-to-Refactoring Mapping

| Code Smell | Recommended Refactoring(s) |
|------------|---------------------------|
| Long Method | Extract Method, Replace Temp with Query |
| Duplicated Code | Extract Method, Pull Up Method, Form Template Method |
| Large Class | Extract Class, Extract Subclass |
| Feature Envy | Move Method, Move Field |
| Primitive Obsession | Replace Primitive with Object, Replace Type Code with Class |
| Long Parameter List | Introduce Parameter Object, Preserve Whole Object |
| Data Clumps | Extract Class, Introduce Parameter Object |
| Switch Statements | Replace Conditional with Polymorphism |
| Speculative Generality | Collapse Hierarchy, Inline Class, Remove Dead Code |
| Dead Code | Remove Dead Code |

### Plan Structure

Use the template at [templates/refactoring-plan.md](templates/refactoring-plan.md).

For each refactoring:
1. **Target**: What code will change
2. **Smell**: What problem it addresses
3. **Refactoring**: Which technique to apply
4. **Steps**: Detailed micro-steps
5. **Risks**: What could go wrong
6. **Rollback**: How to undo if needed

### Phased Approach

**CRITICAL**: Introduce refactoring gradually in phases.

**Phase A: Quick Wins** (Low risk, high value)
- Rename variables for clarity
- Extract obvious duplicate code
- Remove dead code

**Phase B: Structural Improvements** (Medium risk)
- Extract methods from long functions
- Introduce parameter objects
- Move methods to appropriate classes

**Phase C: Architectural Changes** (Higher risk)
- Replace conditionals with polymorphism
- Extract classes
- Introduce design patterns

### Decision Point: Present Plan to User

Before implementation:
- Show complete refactoring plan
- Explain each phase and its risks
- Get explicit approval for each phase
- **Ask**: "Should I proceed with Phase A?"

---

## Phase 5: Incremental Implementation

### The Golden Rule
> "Change → Test → Green? → Commit → Next step"

### Implementation Rhythm

For each refactoring step:

1. **Pre-check**
   - Tests are passing (green)
   - Code compiles

2. **Make ONE small change**
   - Follow the mechanics from the catalog
   - Keep changes minimal

3. **Verify**
   - Run tests immediately
   - Check for compilation errors

4. **If tests pass (green)**
   - Commit with descriptive message
   - Move to next step

5. **If tests fail (red)**
   - STOP immediately
   - Undo the change
   - Analyze what went wrong
   - Ask user if unclear

### Commit Strategy

Each commit should be:
- **Atomic**: One logical change
- **Reversible**: Easy to revert
- **Descriptive**: Clear commit message

Example commit messages:
```
refactor: Extract calculateTotal() from processOrder()
refactor: Rename 'x' to 'customerCount' for clarity
refactor: Remove unused validateOldFormat() method
```

### Progress Reporting

After each sub-phase, report to user:
- Changes made
- Tests still passing?
- Any issues encountered
- **Ask**: "Continue with next batch?"

---

## Phase 6: Review & Iteration

### Post-Refactoring Checklist

- [ ] All tests passing
- [ ] No new warnings/errors
- [ ] Code compiles successfully
- [ ] Behavior unchanged (manual verification)
- [ ] Documentation updated if needed
- [ ] Commit history is clean

### Metrics Comparison

Run complexity analysis before and after:
```bash
python scripts/analyze-complexity.py <file>
```

Present improvements:
- Lines of code change
- Cyclomatic complexity change
- Maintainability index change

### User Review

Present final results:
- Summary of all changes
- Before/after code comparison
- Metrics improvements
- Remaining technical debt
- **Ask**: "Are you satisfied with these changes?"

### Next Steps

Discuss with user:
- Additional smells to address?
- Schedule follow-up refactoring?
- Apply similar changes elsewhere?

---

## Important Guidelines

### When to STOP and Ask

Always pause and consult user when:
- Unsure about business logic
- Change might affect external APIs
- Test coverage is inadequate
- Significant architectural decision needed
- Risk level increases
- You encounter unexpected complexity

### Safety Rules

1. **Never refactor without tests** (unless user explicitly acknowledges risk)
2. **Never make big changes** - break into tiny steps
3. **Never skip the test run** after each change
4. **Never continue if tests fail** - fix or rollback first
5. **Never assume** - when in doubt, ask
6. **For DL code**: Always validate numerical equivalence (loss values match within tolerance)
7. **For DL code**: Run an overfit test after structural refactoring (model should still learn)
8. **For DL code**: Check GPU memory after refactoring — equivalent code should use similar memory

### DL-Specific Safety Rules

When refactoring deep learning code, additional checks are required beyond standard unit tests:

1. **Numerical equivalence**: Verify outputs match within `atol=1e-5` (FP32) or `atol=1e-3` (FP16)
2. **Gradient flow**: Run `torch.autograd.gradcheck` on custom autograd functions after changes
3. **Memory parity**: Use `torch.cuda.max_memory_allocated()` to ensure refactoring doesn't increase memory
4. **Determinism**: Run twice with same seed — outputs should be identical
5. **Overfit test**: Train on a small batch for 10 steps — loss must decrease

### Test Writing Guidelines (DL-specific)

For each function being refactored, ensure tests cover:

- **Shape correctness**: Known input → expected output shape
- **Numerical sanity**: Known input → expected output range (e.g., loss > 0, logits in [-1, 1])
- **Gradient existence**: Parameters receive gradients after backward
- **Edge cases**: Empty batch, single sample, extreme values
- **FP16 compatibility**: Run with autocast — no NaN/Inf

Use the DL red-green-refactor cycle:
1. Write a failing deterministic test (red)
2. Make it pass with the refactored code (green)
3. Run overfit test to confirm learning capability
4. Refactor further

### What NOT to Do

- Don't combine refactoring with feature additions
- Don't refactor during production emergencies
- Don't refactor code you don't understand
- Don't over-engineer - keep it simple
- Don't refactor everything at once

---

## Quick Start Example

### Scenario: Training Loop Bloat

**Before:**
```python
def train(model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, texts) in enumerate(dataloader):
            images, texts = images.to(device), texts.to(device)
            # Forward
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Logit scale (hardcoded temperature)
            logit_scale = model.logit_scale.exp().clamp(max=100)
            # Contrastive loss (inline computation)
            logits = logit_scale * image_features @ text_features.T
            labels = torch.arange(len(images), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Logging inline
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
        # Manual checkpoint
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()}, f"ckpt_{epoch}.pt")
```

**Refactoring Steps:**

1. **Ensure tests exist** for the training function (overfit test on small batch)
2. **Extract** loss computation into `ClipLoss(nn.Module)` class
3. **Test** - loss values should match original (within tolerance)
4. **Extract** checkpoint logic into `save_checkpoint()` helper
5. **Test** - checkpoint save/load roundtrip
6. **Extract** logit computation and normalization into model's `forward()`
7. **Test** - all outputs match
8. **Review** - `train()` now orchestrates clear components

**After:**
```python
def train(model, dataloader, epochs, lr, device):
    loss_fn = ClipLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, texts) in enumerate(dataloader):
            images, texts = images.to(device), texts.to(device)
            image_features, text_features, logit_scale = model(images, texts)
            loss = loss_fn(image_features, text_features, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_metrics(epoch, batch_idx, loss)
        if should_checkpoint(epoch):
            save_checkpoint(model, optimizer, epoch, f"ckpt_{epoch}.pt")
```

---

## References

- [Code Smells Catalog](references/code-smells.md) - Complete list of code smells
- [Refactoring Catalog](references/refactoring-catalog.md) - Refactoring techniques
- [Refactoring Plan Template](templates/refactoring-plan.md) - Planning template

## Scripts

- `scripts/analyze-complexity.py` - Analyze code complexity metrics
- `scripts/detect-smells.py` - Automated smell detection

## Version History

- v2.0.0 (2026-05-07): Adapted for deep learning projects — added 10 DL-specific code smells (Training Loop Bloat, Missing Mode Switch, Autograd Leak, Checkpoint Fragility, etc.) and 10 DL-specific refactoring techniques (Extract Loss Class, Parameterize Model Config, Add Shape Guards, etc.). All examples converted to PyTorch. Added DL-specific safety rules and test guidelines. Added 6 DL detection rules to detect-smells.py.
- v1.0.0 (2025-01-15): Initial release with Fowler methodology, phased approach, user consultation points
