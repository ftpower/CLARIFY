# Code Smells Catalog

A comprehensive reference of code smells based on Martin Fowler's *Refactoring* (2nd Edition). Code smells are symptoms of deeper problems—they indicate that something might be wrong with your code's design.

> "A code smell is a surface indication that usually corresponds to a deeper problem in the system." — Martin Fowler

---

## Bloaters

Code smells representing something that has grown too large to be handled effectively.

### Long Method

**Signs:**
- Method exceeds 30-50 lines
- Need to scroll to see the whole method
- Multiple levels of nesting
- Comments explaining what sections do

**Why it's bad:**
- Hard to understand
- Difficult to test in isolation
- Changes have unintended consequences
- Duplicate logic hides inside

**Refactorings:**
- Extract Method
- Replace Temp with Query
- Introduce Parameter Object
- Replace Method with Method Object
- Decompose Conditional

**Example (Before):**
```javascript
function processOrder(order) {
  // Validate order (20 lines)
  if (!order.items) throw new Error('No items');
  if (order.items.length === 0) throw new Error('Empty order');
  // ... more validation

  // Calculate totals (30 lines)
  let subtotal = 0;
  for (const item of order.items) {
    subtotal += item.price * item.quantity;
  }
  // ... tax, shipping, discounts

  // Send notifications (20 lines)
  // ... email logic
}
```

**Example (After):**
```javascript
function processOrder(order) {
  validateOrder(order);
  const totals = calculateOrderTotals(order);
  sendOrderNotifications(order, totals);
  return { order, totals };
}
```

---

### Large Class

**Signs:**
- Class has many instance variables (>7-10)
- Class has many methods (>15-20)
- Class name is vague (Manager, Handler, Processor)
- Methods don't use all instance variables

**Why it's bad:**
- Violates Single Responsibility Principle
- Hard to test
- Changes ripple through unrelated features
- Difficult to reuse parts

**Refactorings:**
- Extract Class
- Extract Subclass
- Extract Interface

**Detection:**
```
Lines of code > 300
Number of methods > 15
Number of fields > 10
```

---

### Primitive Obsession

**Signs:**
- Using primitives for domain concepts (string for email, int for money)
- Arrays of primitives instead of objects
- String constants for type codes
- Magic numbers/strings

**Why it's bad:**
- No validation at type level
- Logic scattered across codebase
- Easy to pass wrong values
- Missing domain concepts

**Refactorings:**
- Replace Primitive with Object
- Replace Type Code with Class
- Replace Type Code with Subclasses
- Replace Type Code with State/Strategy

**Example (Before):**
```javascript
const user = {
  email: 'john@example.com',     // Just a string
  phone: '1234567890',           // Just a string
  status: 'active',              // Magic string
  balance: 10050                 // Cents as integer
};
```

**Example (After):**
```javascript
const user = {
  email: new Email('john@example.com'),
  phone: new PhoneNumber('1234567890'),
  status: UserStatus.ACTIVE,
  balance: Money.cents(10050)
};
```

---

### Long Parameter List

**Signs:**
- Methods with 4+ parameters
- Parameters that always appear together
- Boolean flags changing method behavior
- Null/undefined passed frequently

**Why it's bad:**
- Hard to call correctly
- Parameter order confusion
- Indicates method doing too much
- Hard to add new parameters

**Refactorings:**
- Introduce Parameter Object
- Preserve Whole Object
- Replace Parameter with Method Call
- Remove Flag Argument

**Example (Before):**
```javascript
function createUser(firstName, lastName, email, phone,
                    street, city, state, zip,
                    isAdmin, isActive, createdBy) {
  // ...
}
```

**Example (After):**
```javascript
function createUser(personalInfo, address, options) {
  // personalInfo: { firstName, lastName, email, phone }
  // address: { street, city, state, zip }
  // options: { isAdmin, isActive, createdBy }
}
```

---

### Data Clumps

**Signs:**
- Same 3+ fields appear together repeatedly
- Parameters that always travel together
- Classes with field subsets belonging together

**Why it's bad:**
- Duplicate handling logic
- Missing abstraction
- Harder to extend
- Indicates hidden class

**Refactorings:**
- Extract Class
- Introduce Parameter Object
- Preserve Whole Object

**Example:**
```javascript
// Data clump: (x, y, z) coordinates
function movePoint(x, y, z, dx, dy, dz) { }
function scalePoint(x, y, z, factor) { }
function distanceBetween(x1, y1, z1, x2, y2, z2) { }

// Extract Point3D class
class Point3D {
  constructor(x, y, z) { }
  move(delta) { }
  scale(factor) { }
  distanceTo(other) { }
}
```

---

## Object-Orientation Abusers

Smells indicating incomplete or incorrect use of OOP principles.

### Switch Statements

**Signs:**
- Long switch/case or if/else chains
- Same switch in multiple places
- Switch on type codes
- Adding new cases requires changes everywhere

**Why it's bad:**
- Violates Open/Closed Principle
- Changes ripple to all switch locations
- Hard to extend
- Often indicates missing polymorphism

**Refactorings:**
- Replace Conditional with Polymorphism
- Replace Type Code with Subclasses
- Replace Type Code with State/Strategy

**Example (Before):**
```javascript
function calculatePay(employee) {
  switch (employee.type) {
    case 'hourly':
      return employee.hours * employee.rate;
    case 'salaried':
      return employee.salary / 12;
    case 'commissioned':
      return employee.sales * employee.commission;
  }
}
```

**Example (After):**
```javascript
class HourlyEmployee {
  calculatePay() {
    return this.hours * this.rate;
  }
}

class SalariedEmployee {
  calculatePay() {
    return this.salary / 12;
  }
}
```

---

### Temporary Field

**Signs:**
- Instance variables only used in some methods
- Fields set conditionally
- Complex initialization for certain cases

**Why it's bad:**
- Confusing—field exists but might be null
- Hard to understand object state
- Indicates conditional logic hiding

**Refactorings:**
- Extract Class
- Introduce Null Object
- Replace Temp Field with Local

---

### Refused Bequest

**Signs:**
- Subclass doesn't use inherited methods/data
- Subclass overrides to do nothing
- Inheritance used for code reuse, not IS-A relationship

**Why it's bad:**
- Wrong abstraction
- Violates Liskov Substitution Principle
- Misleading hierarchy

**Refactorings:**
- Push Down Method/Field
- Replace Subclass with Delegate
- Replace Inheritance with Delegation

---

### Alternative Classes with Different Interfaces

**Signs:**
- Two classes that do similar things
- Different method names for same concept
- Could be used interchangeably

**Why it's bad:**
- Duplicate implementations
- No common interface
- Hard to switch between

**Refactorings:**
- Rename Method
- Move Method
- Extract Superclass
- Extract Interface

---

## Change Preventers

Smells that make changes difficult—changing one thing requires changing many others.

### Divergent Change

**Signs:**
- One class changed for multiple different reasons
- Changes in different areas trigger same class edits
- Class is a "God class"

**Why it's bad:**
- Violates Single Responsibility
- High change frequency
- Merge conflicts

**Refactorings:**
- Extract Class
- Extract Superclass
- Extract Subclass

**Example:**
A `User` class changes for:
- Authentication changes
- Profile changes
- Billing changes
- Notification changes

→ Extract: `AuthService`, `ProfileService`, `BillingService`, `NotificationService`

---

### Shotgun Surgery

**Signs:**
- One change requires edits in many classes
- Small feature needs touching 10+ files
- Changes are scattered, hard to find all

**Why it's bad:**
- Easy to miss a spot
- High coupling
- Changes are error-prone

**Refactorings:**
- Move Method
- Move Field
- Inline Class

**Detection:**
Look for: adding one field requires changes in >5 files.

---

### Parallel Inheritance Hierarchies

**Signs:**
- Creating subclass in one hierarchy requires subclass in another
- Class prefixes match (e.g., `DatabaseOrder`, `DatabaseProduct`)

**Why it's bad:**
- Double the maintenance
- Coupling between hierarchies
- Easy to forget one side

**Refactorings:**
- Move Method
- Move Field
- Eliminate one hierarchy

---

## Dispensables

Something unnecessary that should be removed.

### Comments (Excessive)

**Signs:**
- Comments explaining what code does
- Commented-out code
- TODO/FIXME that linger forever
- Apologies in comments

**Why it's bad:**
- Comments lie (get out of sync)
- Code should be self-documenting
- Dead code causes confusion

**Refactorings:**
- Extract Method (name explains what)
- Rename (clarity without comments)
- Remove commented code
- Introduce Assertion

**Good vs Bad Comments:**
```javascript
// BAD: Explaining what
// Loop through users and check if active
for (const user of users) {
  if (user.status === 'active') { }
}

// GOOD: Explaining why
// Active users only - inactive are handled by cleanup job
const activeUsers = users.filter(u => u.isActive);
```

---

### Duplicate Code

**Signs:**
- Same code in multiple places
- Similar code with small variations
- Copy-paste patterns

**Why it's bad:**
- Bug fixes needed in multiple places
- Inconsistency risk
- Bloated codebase

**Refactorings:**
- Extract Method
- Extract Class
- Pull Up Method (in hierarchies)
- Form Template Method

**Detection Rule:**
Any code duplicated 3+ times should be extracted.

---

### Lazy Class

**Signs:**
- Class doesn't do enough to justify existence
- Wrapper with no added value
- Result of over-engineering

**Why it's bad:**
- Maintenance overhead
- Unnecessary indirection
- Complexity without benefit

**Refactorings:**
- Inline Class
- Collapse Hierarchy

---

### Dead Code

**Signs:**
- Unreachable code
- Unused variables/methods/classes
- Commented-out code
- Code behind impossible conditions

**Why it's bad:**
- Confusion
- Maintenance burden
- Slows down understanding

**Refactorings:**
- Remove Dead Code
- Safe Delete

**Detection:**
```bash
# Look for unused exports
# Look for unreferenced functions
# IDE "unused" warnings
```

---

### Speculative Generality

**Signs:**
- Abstract classes with one subclass
- Unused parameters "for future use"
- Methods that only delegate
- "Framework" for one use case

**Why it's bad:**
- Complexity without benefit
- YAGNI (You Ain't Gonna Need It)
- Harder to understand

**Refactorings:**
- Collapse Hierarchy
- Inline Class
- Remove Parameter
- Rename Method

---

## Couplers

Smells that represent excessive coupling between classes.

### Feature Envy

**Signs:**
- Method uses more data from another class than its own
- Many getter calls to another object
- Data and behavior are separated

**Why it's bad:**
- Wrong location for behavior
- Poor encapsulation
- Hard to maintain

**Refactorings:**
- Move Method
- Move Field
- Extract Method (then move)

**Example (Before):**
```javascript
class Order {
  getDiscountedPrice(customer) {
    // Uses customer data heavily
    if (customer.loyaltyYears > 5) {
      return this.price * customer.discountRate;
    }
    return this.price;
  }
}
```

**Example (After):**
```javascript
class Customer {
  getDiscountedPriceFor(price) {
    if (this.loyaltyYears > 5) {
      return price * this.discountRate;
    }
    return price;
  }
}
```

---

### Inappropriate Intimacy

**Signs:**
- Classes access each other's private parts
- Bidirectional references
- Subclasses know too much about parents

**Why it's bad:**
- High coupling
- Changes cascade
- Hard to modify one without other

**Refactorings:**
- Move Method
- Move Field
- Change Bidirectional to Unidirectional
- Extract Class
- Hide Delegate

---

### Message Chains

**Signs:**
- Long chains of method calls: `a.getB().getC().getD().getValue()`
- Client depends on navigation structure
- "Train wreck" code

**Why it's bad:**
- Fragile—any change breaks chain
- Violates Law of Demeter
- Coupling to structure

**Refactorings:**
- Hide Delegate
- Extract Method
- Move Method

**Example:**
```javascript
// Bad: Message chain
const managerName = employee.getDepartment().getManager().getName();

// Better: Hide delegation
const managerName = employee.getManagerName();
```

---

### Middle Man

**Signs:**
- Class that only delegates to another
- Half the methods are delegations
- No added value

**Why it's bad:**
- Unnecessary indirection
- Maintenance overhead
- Confusing architecture

**Refactorings:**
- Remove Middle Man
- Inline Method

---

## Smell Severity Guide

| Severity | Description | Action |
|----------|-------------|--------|
| **Critical** | Blocks development, causes bugs | Fix immediately |
| **High** | Significant maintenance burden | Fix in current sprint |
| **Medium** | Noticeable but manageable | Plan for near future |
| **Low** | Minor inconvenience | Fix opportunistically |

---

## Quick Detection Checklist

Use this checklist when scanning code:

- [ ] Any method > 30 lines?
- [ ] Any class > 300 lines?
- [ ] Any method with > 4 parameters?
- [ ] Any duplicated code blocks?
- [ ] Any switch/case on type codes?
- [ ] Any unused code?
- [ ] Any methods using another class's data heavily?
- [ ] Any long chains of method calls?
- [ ] Any comments explaining "what" not "why"?
- [ ] Any primitives that should be objects?

---

---

## Deep Learning Specific Smells

Smells unique to PyTorch / deep learning projects.

### Training Loop Bloat

**Signs:**
- `train()` function exceeds 100 lines
- Forward pass, loss computation, backward, logging, checkpointing all in one function
- Adding a new feature (e.g., EMA) requires modifying the monolithic function

**Why it's bad:**
- Hard to test individual components (loss, logging, checkpoint)
- Changes to logging break training logic
- Can't reuse components in different training scripts

**Refactorings:**
- Extract Training Step
- Extract Loss Class
- Separate Metrics from Training
- Introduce Checkpoint Manager

**Example (Before):**
```python
def train(model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, texts in dataloader:
            images, texts = images.to(device), texts.to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T
            labels = torch.arange(len(images), device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Loss: {loss.item()}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"ckpt_{epoch}.pt")
```

**Example (After):**
```python
def train(model, dataloader, epochs, lr, device):
    loss_fn = ClipLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for images, texts in dataloader:
            loss = training_step(model, loss_fn, optimizer, images, texts, device)
            log_metrics(epoch, batch_idx, loss)
        checkpoint_if_needed(model, optimizer, epoch)
```

---

### Missing Mode Switch

**Signs:**
- `model.train()` or `model.eval()` missing before training/evaluation loops
- BatchNorm/Dropout behavior silently wrong
- Inference runs with dropout active

**Why it's bad:**
- Silent correctness bugs — no error, just wrong results
- BatchNorm accumulates wrong statistics in eval mode
- Dropout applied during inference, causing non-deterministic outputs

**Refactorings:**
- Add explicit `.train()` / `.eval()` calls at loop boundaries
- Use `torch.set_grad_enabled()` context manager

**Example (Before):**
```python
# Inference without eval mode — DROPOUT STILL ACTIVE!
outputs = model(images)
predictions = outputs.argmax(dim=-1)
```

**Example (After):**
```python
model.eval()
with torch.no_grad():
    outputs = model(images)
predictions = outputs.argmax(dim=-1)
```

---

### Autograd Leak

**Signs:**
- Inference/evaluation code without `torch.no_grad()`
- Tensors retained in computation graph unnecessarily
- GPU memory grows during evaluation loop
- `@torch.no_grad()` decorator missing on eval functions

**Why it's bad:**
- GPU memory exhaustion during evaluation
- Slower inference (gradient bookkeeping overhead)
- Silent memory leak — accumulates over evaluation steps

**Detection:**
```python
# Functions with "eval", "inference", "test" in name should have @torch.no_grad()
```

**Refactorings:**
- Wrap with no_grad
- Add `@torch.no_grad()` decorator to eval methods

**Example (Before):**
```python
def evaluate(model, dataloader):
    results = []
    for images in dataloader:
        outputs = model(images)  # Gradients tracked!
        results.append(outputs.cpu())
    return results
```

**Example (After):**
```python
@torch.no_grad()
def evaluate(model, dataloader):
    results = []
    for images in dataloader:
        outputs = model(images)  # No gradients
        results.append(outputs.cpu())
    return results
```

---

### Inline Loss Computation

**Signs:**
- Loss formula directly in training loop
- Numerical stability handling (eps, clamping) duplicated
- Same loss logic copied between training scripts

**Why it's bad:**
- Hard to test loss computation in isolation
- Duplicated numerical stability fixes
- Can't swap loss functions easily

**Refactorings:**
- Extract Loss Class
- Parameterize loss function selection

**Example (Before):**
```python
# Loss inline in training loop
logits = logit_scale * image_features @ text_features.T
labels = torch.arange(len(images), device=device)
loss_i = F.cross_entropy(logits, labels)
loss_t = F.cross_entropy(logits.T, labels)
loss = (loss_i + loss_t) / 2
```

**Example (After):**
```python
loss = loss_fn(image_features, text_features, logit_scale)
# loss_fn is ClipLoss(nn.Module) — testable, reusable, swappable
```

---

### Config Scatter

**Signs:**
- Hyperparameters hardcoded as literals across multiple files
- Same value (e.g., `embed_dim=512`) repeated in model.py, train.py, eval.py
- Changing a hyperparameter requires edits in 3+ places

**Why it's bad:**
- Inconsistent values across the project
- Easy to forget one location
- Hard to experiment with different configs

**Refactorings:**
- Parameterize Model Config
- Replace Magic Numbers with Config

**Example (Before):**
```python
# model.py
vision = VisionTransformer(embed_dim=512, layers=12, width=768)

# train.py
model = CLIP(embed_dim=512, vision_cfg=CLIPVisionCfg(layers=12, width=768))
```

**Example (After):**
```python
# config.py
@dataclass
class ModelConfig:
    embed_dim: int = 512
    vision_layers: int = 12
    vision_width: int = 768

# model.py
vision = VisionTransformer(embed_dim=cfg.embed_dim, layers=cfg.vision_layers, width=cfg.vision_width)

# train.py
model = CLIP(embed_dim=cfg.embed_dim, vision_cfg=CLIPVisionCfg(layers=cfg.vision_layers, width=cfg.vision_width))
```

---

### Shape Ambiguity

**Signs:**
- `forward()` method has no docstring
- Tensor shapes not documented in code or comments
- Batch dimension handling unclear (B, N, L, D — which is which?)

**Why it's bad:**
- Must trace code to understand expected shapes
- Shape errors produce cryptic PyTorch error messages
- Hard to reuse modules in different contexts

**Refactorings:**
- Add Shape Guards
- Document shapes in docstring

**Example (Before):**
```python
class VisionTransformer(nn.Module):
    def forward(self, x):
        # What shape is x? What does it return?
        x = self.conv1(x)
        # ... 50 lines
        return pooled
```

**Example (After):**
```python
class VisionTransformer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VisionTransformer forward pass.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            pooled: [B, output_dim] global image features
        """
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        x = self.conv1(x)  # [B, width, H/P, W/P]
        # ...
        return pooled
```

---

### Architecture Duplication

**Signs:**
- VisionTransformer and TextTransformer share 60%+ identical code
- Same attention mechanism copied between modules
- Bug fix in one module not applied to the other

**Why it's bad:**
- Fix applied to one variant, not the other
- Bloated codebase
- Inconsistent behavior between similar modules

**Refactorings:**
- Extract Common Architecture
- Pull Up Method (to shared base class)
- Extract shared Transformer/Attention base

---

### Mixed Responsibility

**Signs:**
- `nn.Module` subclass handles data loading, preprocessing, metrics
- Model class stores training state (epoch, optimizer)
- `forward()` method has side effects (logging, checkpoint saving)

**Why it's bad:**
- Can't reuse model for inference without training baggage
- Violates Single Responsibility Principle
- Hard to test model in isolation

**Refactorings:**
- Extract Data Pipeline
- Separate Metrics from Training
- Move training state to Trainer class

---

### Magic DL Numbers

**Signs:**
- Hardcoded `eps=1e-5` in LayerNorm/BatchNorm
- Hardcoded `temperature=0.07` / `logit_scale=ln(1/0.07)`
- Hardcoded `dim=512`, `heads=8`, `layers=12` appearing as literals
- Missing justification in comments

**Why it's bad:**
- Origin of values unclear (was it tuned? copied from paper?)
- Can't experiment with different values easily
- Values might be wrong for new use cases

**Refactorings:**
- Replace Magic Numbers with Config
- Reference original paper/source in comment

---

### Checkpoint Fragility

**Signs:**
- `torch.save(model.state_dict(), ...)` — only saves model weights
- Missing optimizer state, scheduler state, GradScaler state
- Resume training produces different results than uninterrupted run
- No version/epoch metadata in checkpoint

**Why it's bad:**
- Cannot resume training correctly
- Losing optimizer momentum/Adam moments
- FP16 training breaks on resume (scaler state lost)

**Refactorings:**
- Introduce Checkpoint Manager

**Example (Before):**
```python
# Incomplete checkpoint
torch.save(model.state_dict(), "checkpoint.pt")
```

**Example (After):**
```python
checkpoint = {
    "epoch": epoch,
    "step": global_step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict() if use_amp else None,
}
torch.save(checkpoint, f"checkpoint_epoch{epoch}.pt")
```

---

## Further Reading

- Fowler, M. (2018). *Refactoring: Improving the Design of Existing Code* (2nd ed.)
- Kerievsky, J. (2004). *Refactoring to Patterns*
- Feathers, M. (2004). *Working Effectively with Legacy Code*
