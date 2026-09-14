# Refactoring Catalog

A curated catalog of refactoring techniques from Martin Fowler's *Refactoring* (2nd Edition). Each refactoring includes motivation, step-by-step mechanics, and examples.

> "A refactoring is defined by its mechanics—the precise sequence of steps that you follow to carry out the change." — Martin Fowler

---

## How to Use This Catalog

1. **Identify the smell** using the code smells reference
2. **Find the matching refactoring** in this catalog
3. **Follow the mechanics** step by step
4. **Test after each step** to ensure behavior is preserved

**Golden Rule**: If any step takes more than 10 minutes, break it into smaller steps.

---

## Most Common Refactorings

### Extract Method

**When to use**: Long method, duplicate code, need to name a concept

**Motivation**: Turn a code fragment into a method whose name explains the purpose.

**Mechanics**:
1. Create a new method named for what it does (not how)
2. Copy the code fragment into the new method
3. Scan for local variables used in the fragment
4. Pass local variables as parameters (or declare in method)
5. Handle return values appropriately
6. Replace the original fragment with a call to the new method
7. Test

**Before**:
```javascript
function printOwing(invoice) {
  let outstanding = 0;

  console.log("***********************");
  console.log("**** Customer Owes ****");
  console.log("***********************");

  // Calculate outstanding
  for (const order of invoice.orders) {
    outstanding += order.amount;
  }

  // Print details
  console.log(`name: ${invoice.customer}`);
  console.log(`amount: ${outstanding}`);
}
```

**After**:
```javascript
function printOwing(invoice) {
  printBanner();
  const outstanding = calculateOutstanding(invoice);
  printDetails(invoice, outstanding);
}

function printBanner() {
  console.log("***********************");
  console.log("**** Customer Owes ****");
  console.log("***********************");
}

function calculateOutstanding(invoice) {
  return invoice.orders.reduce((sum, order) => sum + order.amount, 0);
}

function printDetails(invoice, outstanding) {
  console.log(`name: ${invoice.customer}`);
  console.log(`amount: ${outstanding}`);
}
```

---

### Inline Method

**When to use**: Method body is as clear as its name, excessive delegation

**Motivation**: Remove needless indirection when the method doesn't add value.

**Mechanics**:
1. Check that the method isn't polymorphic
2. Find all calls to the method
3. Replace each call with the method body
4. Test after each replacement
5. Remove the method definition

**Before**:
```javascript
function getRating(driver) {
  return moreThanFiveLateDeliveries(driver) ? 2 : 1;
}

function moreThanFiveLateDeliveries(driver) {
  return driver.numberOfLateDeliveries > 5;
}
```

**After**:
```javascript
function getRating(driver) {
  return driver.numberOfLateDeliveries > 5 ? 2 : 1;
}
```

---

### Extract Variable

**When to use**: Complex expression that is hard to understand

**Motivation**: Give a name to a piece of a complex expression.

**Mechanics**:
1. Ensure the expression has no side effects
2. Declare an immutable variable
3. Set it to the result of the expression (or part)
4. Replace the original expression with the variable
5. Test

**Before**:
```javascript
return order.quantity * order.itemPrice -
  Math.max(0, order.quantity - 500) * order.itemPrice * 0.05 +
  Math.min(order.quantity * order.itemPrice * 0.1, 100);
```

**After**:
```javascript
const basePrice = order.quantity * order.itemPrice;
const quantityDiscount = Math.max(0, order.quantity - 500) * order.itemPrice * 0.05;
const shipping = Math.min(basePrice * 0.1, 100);
return basePrice - quantityDiscount + shipping;
```

---

### Inline Variable

**When to use**: Variable name doesn't communicate more than the expression

**Motivation**: Remove unnecessary indirection.

**Mechanics**:
1. Check that the right-hand side has no side effects
2. If variable isn't immutable, make it so and test
3. Find the first reference and replace with the expression
4. Test
5. Repeat for all references
6. Remove the declaration and assignment
7. Test

---

### Rename Variable

**When to use**: Name doesn't clearly communicate purpose

**Motivation**: Good names are crucial for clean code.

**Mechanics**:
1. If variable is widely used, consider encapsulating
2. Find all references
3. Change each reference
4. Test

**Tips**:
- Use intention-revealing names
- Avoid abbreviations
- Use domain terminology

```javascript
// Bad
const d = 30;
const x = users.filter(u => u.a);

// Good
const daysSinceLastLogin = 30;
const activeUsers = users.filter(user => user.isActive);
```

---

### Change Function Declaration

**When to use**: Function name doesn't explain purpose, parameters need change

**Motivation**: Good function names make code self-documenting.

**Mechanics (Simple)**:
1. Remove parameters not needed
2. Change the name
3. Add parameters needed
4. Test

**Mechanics (Migration - for complex changes)**:
1. If removing parameter, make sure it's not used
2. Create new function with desired declaration
3. Have old function call new function
4. Test
5. Change callers to use new function
6. Test after each
7. Remove old function

**Before**:
```javascript
function circum(radius) {
  return 2 * Math.PI * radius;
}
```

**After**:
```javascript
function circumference(radius) {
  return 2 * Math.PI * radius;
}
```

---

### Encapsulate Variable

**When to use**: Direct access to data from multiple places

**Motivation**: Provide a clear access point for data manipulation.

**Mechanics**:
1. Create getter and setter functions
2. Find all references
3. Replace reads with getter
4. Replace writes with setter
5. Test after each change
6. Restrict visibility of the variable

**Before**:
```javascript
let defaultOwner = { firstName: "Martin", lastName: "Fowler" };

// Used in many places
spaceship.owner = defaultOwner;
```

**After**:
```javascript
let defaultOwnerData = { firstName: "Martin", lastName: "Fowler" };

function defaultOwner() { return defaultOwnerData; }
function setDefaultOwner(arg) { defaultOwnerData = arg; }

spaceship.owner = defaultOwner();
```

---

### Introduce Parameter Object

**When to use**: Several parameters that frequently go together

**Motivation**: Group data that naturally belongs together.

**Mechanics**:
1. Create a new class/structure for the grouped parameters
2. Test
3. Use Change Function Declaration to add the new object
4. Test
5. For each parameter in the group, remove it from the function and use the new object
6. Test after each

**Before**:
```javascript
function amountInvoiced(startDate, endDate) { ... }
function amountReceived(startDate, endDate) { ... }
function amountOverdue(startDate, endDate) { ... }
```

**After**:
```javascript
class DateRange {
  constructor(start, end) {
    this.start = start;
    this.end = end;
  }
}

function amountInvoiced(dateRange) { ... }
function amountReceived(dateRange) { ... }
function amountOverdue(dateRange) { ... }
```

---

### Combine Functions into Class

**When to use**: Several functions operate on the same data

**Motivation**: Group functions with the data they operate on.

**Mechanics**:
1. Apply Encapsulate Record to the common data
2. Move each function into the class
3. Test after each move
4. Replace data arguments with uses of class fields

**Before**:
```javascript
function base(reading) { ... }
function taxableCharge(reading) { ... }
function calculateBaseCharge(reading) { ... }
```

**After**:
```javascript
class Reading {
  constructor(data) { this._data = data; }

  get base() { ... }
  get taxableCharge() { ... }
  get calculateBaseCharge() { ... }
}
```

---

### Split Phase

**When to use**: Code deals with two different things

**Motivation**: Separate code into distinct phases with clear boundaries.

**Mechanics**:
1. Create a second function for the second phase
2. Test
3. Introduce an intermediate data structure between phases
4. Test
5. Extract first phase into its own function
6. Test

**Before**:
```javascript
function priceOrder(product, quantity, shippingMethod) {
  const basePrice = product.basePrice * quantity;
  const discount = Math.max(quantity - product.discountThreshold, 0)
    * product.basePrice * product.discountRate;
  const shippingPerCase = (basePrice > shippingMethod.discountThreshold)
    ? shippingMethod.discountedFee : shippingMethod.feePerCase;
  const shippingCost = quantity * shippingPerCase;
  return basePrice - discount + shippingCost;
}
```

**After**:
```javascript
function priceOrder(product, quantity, shippingMethod) {
  const priceData = calculatePricingData(product, quantity);
  return applyShipping(priceData, shippingMethod);
}

function calculatePricingData(product, quantity) {
  const basePrice = product.basePrice * quantity;
  const discount = Math.max(quantity - product.discountThreshold, 0)
    * product.basePrice * product.discountRate;
  return { basePrice, quantity, discount };
}

function applyShipping(priceData, shippingMethod) {
  const shippingPerCase = (priceData.basePrice > shippingMethod.discountThreshold)
    ? shippingMethod.discountedFee : shippingMethod.feePerCase;
  const shippingCost = priceData.quantity * shippingPerCase;
  return priceData.basePrice - priceData.discount + shippingCost;
}
```

---

## Moving Features

### Move Method

**When to use**: Method uses more features of another class than its own

**Motivation**: Put functions with the data they use most.

**Mechanics**:
1. Examine all program elements used by method in its class
2. Check if method is polymorphic
3. Copy method to target class
4. Adjust for new context
5. Make original method delegate to target
6. Test
7. Consider removing original method

---

### Move Field

**When to use**: Field is used more by another class

**Motivation**: Keep data with the functions that use it.

**Mechanics**:
1. Encapsulate the field if not already
2. Test
3. Create field in target
4. Update references to use target field
5. Test
6. Remove original field

---

### Move Statements into Function

**When to use**: Same code always appears with a function call

**Motivation**: Remove duplication by moving repeated code into the function.

**Mechanics**:
1. Extract the repeated code into a function if not already
2. Move statements into that function
3. Test
4. If callers no longer need standalone statements, remove them

---

### Move Statements to Callers

**When to use**: Common behavior varies between callers

**Motivation**: When behavior needs to differ, move it out of the function.

**Mechanics**:
1. Use Extract Method on the code to move
2. Use Inline Method on the original function
3. Remove the now-inlined call
4. Move extracted code to each caller
5. Test

---

## Organizing Data

### Replace Primitive with Object

**When to use**: Data item needs more behavior than simple value

**Motivation**: Encapsulate data with its behavior.

**Mechanics**:
1. Apply Encapsulate Variable
2. Create a simple value class
3. Change the setter to create a new instance
4. Change the getter to return the value
5. Test
6. Add richer behavior to the new class

**Before**:
```javascript
class Order {
  constructor(data) {
    this.priority = data.priority; // string: "high", "rush", etc.
  }
}

// Usage
if (order.priority === "high" || order.priority === "rush") { ... }
```

**After**:
```javascript
class Priority {
  constructor(value) {
    if (!Priority.legalValues().includes(value))
      throw new Error(`Invalid priority: ${value}`);
    this._value = value;
  }

  static legalValues() { return ['low', 'normal', 'high', 'rush']; }
  get value() { return this._value; }

  higherThan(other) {
    return Priority.legalValues().indexOf(this._value) >
           Priority.legalValues().indexOf(other._value);
  }
}

// Usage
if (order.priority.higherThan(new Priority("normal"))) { ... }
```

---

### Replace Temp with Query

**When to use**: Temporary variable holds result of an expression

**Motivation**: Make the code clearer by extracting the expression into a function.

**Mechanics**:
1. Check that the variable is assigned only once
2. Extract the assignment's right-hand side into a method
3. Replace references to the temp with the method call
4. Test
5. Remove the temp declaration and assignment

**Before**:
```javascript
const basePrice = this._quantity * this._itemPrice;
if (basePrice > 1000) {
  return basePrice * 0.95;
} else {
  return basePrice * 0.98;
}
```

**After**:
```javascript
get basePrice() {
  return this._quantity * this._itemPrice;
}

// In the method
if (this.basePrice > 1000) {
  return this.basePrice * 0.95;
} else {
  return this.basePrice * 0.98;
}
```

---

## Simplifying Conditional Logic

### Decompose Conditional

**When to use**: Complex conditional (if-then-else) statement

**Motivation**: Make the intention clear by extracting conditions and actions.

**Mechanics**:
1. Apply Extract Method on the condition
2. Apply Extract Method on the then-branch
3. Apply Extract Method on the else-branch (if present)

**Before**:
```javascript
if (!aDate.isBefore(plan.summerStart) && !aDate.isAfter(plan.summerEnd)) {
  charge = quantity * plan.summerRate;
} else {
  charge = quantity * plan.regularRate + plan.regularServiceCharge;
}
```

**After**:
```javascript
if (isSummer(aDate, plan)) {
  charge = summerCharge(quantity, plan);
} else {
  charge = regularCharge(quantity, plan);
}

function isSummer(date, plan) {
  return !date.isBefore(plan.summerStart) && !date.isAfter(plan.summerEnd);
}

function summerCharge(quantity, plan) {
  return quantity * plan.summerRate;
}

function regularCharge(quantity, plan) {
  return quantity * plan.regularRate + plan.regularServiceCharge;
}
```

---

### Consolidate Conditional Expression

**When to use**: Multiple conditions with the same result

**Motivation**: Make it clear that conditions are a single check.

**Mechanics**:
1. Verify no side effects in conditions
2. Combine conditions using `and` or `or`
3. Consider Extract Method on the combined condition

**Before**:
```javascript
if (employee.seniority < 2) return 0;
if (employee.monthsDisabled > 12) return 0;
if (employee.isPartTime) return 0;
```

**After**:
```javascript
if (isNotEligibleForDisability(employee)) return 0;

function isNotEligibleForDisability(employee) {
  return employee.seniority < 2 ||
         employee.monthsDisabled > 12 ||
         employee.isPartTime;
}
```

---

### Replace Nested Conditional with Guard Clauses

**When to use**: Deeply nested conditionals making flow hard to follow

**Motivation**: Use guard clauses for special cases, keeping normal flow clear.

**Mechanics**:
1. Find the special case conditions
2. Replace them with guard clauses that return early
3. Test after each change

**Before**:
```javascript
function payAmount(employee) {
  let result;
  if (employee.isSeparated) {
    result = { amount: 0, reasonCode: "SEP" };
  } else {
    if (employee.isRetired) {
      result = { amount: 0, reasonCode: "RET" };
    } else {
      result = calculateNormalPay(employee);
    }
  }
  return result;
}
```

**After**:
```javascript
function payAmount(employee) {
  if (employee.isSeparated) return { amount: 0, reasonCode: "SEP" };
  if (employee.isRetired) return { amount: 0, reasonCode: "RET" };
  return calculateNormalPay(employee);
}
```

---

### Replace Conditional with Polymorphism

**When to use**: Switch/case based on type, conditional logic varying by type

**Motivation**: Let objects handle their own behavior.

**Mechanics**:
1. Create class hierarchy (if not exists)
2. Use Factory Function for object creation
3. Move conditional logic into superclass method
4. Create subclass method for each case
5. Remove original conditional

**Before**:
```javascript
function plumages(birds) {
  return birds.map(b => plumage(b));
}

function plumage(bird) {
  switch (bird.type) {
    case 'EuropeanSwallow':
      return "average";
    case 'AfricanSwallow':
      return (bird.numberOfCoconuts > 2) ? "tired" : "average";
    case 'NorwegianBlueParrot':
      return (bird.voltage > 100) ? "scorched" : "beautiful";
    default:
      return "unknown";
  }
}
```

**After**:
```javascript
class Bird {
  get plumage() { return "unknown"; }
}

class EuropeanSwallow extends Bird {
  get plumage() { return "average"; }
}

class AfricanSwallow extends Bird {
  get plumage() {
    return (this.numberOfCoconuts > 2) ? "tired" : "average";
  }
}

class NorwegianBlueParrot extends Bird {
  get plumage() {
    return (this.voltage > 100) ? "scorched" : "beautiful";
  }
}

function createBird(data) {
  switch (data.type) {
    case 'EuropeanSwallow': return new EuropeanSwallow(data);
    case 'AfricanSwallow': return new AfricanSwallow(data);
    case 'NorwegianBlueParrot': return new NorwegianBlueParrot(data);
    default: return new Bird(data);
  }
}
```

---

### Introduce Special Case (Null Object)

**When to use**: Repeated null checks for special cases

**Motivation**: Return a special object that handles the special case.

**Mechanics**:
1. Create special case class with expected interface
2. Add isSpecialCase check
3. Introduce factory method
4. Replace null checks with special case object usage
5. Test

**Before**:
```javascript
const customer = site.customer;
// ... many places checking
if (customer === "unknown") {
  customerName = "occupant";
} else {
  customerName = customer.name;
}
```

**After**:
```javascript
class UnknownCustomer {
  get name() { return "occupant"; }
  get billingPlan() { return registry.defaultPlan; }
}

// Factory method
function customer(site) {
  return site.customer === "unknown"
    ? new UnknownCustomer()
    : site.customer;
}

// Usage - no null checks needed
const customerName = customer.name;
```

---

## Refactoring APIs

### Separate Query from Modifier

**When to use**: Function both returns a value and has side effects

**Motivation**: Make it clear which operations have side effects.

**Mechanics**:
1. Create a new query function
2. Copy original function's return logic
3. Modify original to return void
4. Replace calls that use return value
5. Test

**Before**:
```javascript
function alertForMiscreant(people) {
  for (const p of people) {
    if (p === "Don") {
      setOffAlarms();
      return "Don";
    }
    if (p === "John") {
      setOffAlarms();
      return "John";
    }
  }
  return "";
}
```

**After**:
```javascript
function findMiscreant(people) {
  for (const p of people) {
    if (p === "Don") return "Don";
    if (p === "John") return "John";
  }
  return "";
}

function alertForMiscreant(people) {
  if (findMiscreant(people) !== "") setOffAlarms();
}
```

---

### Parameterize Function

**When to use**: Several functions doing similar things with different values

**Motivation**: Remove duplication by adding a parameter.

**Mechanics**:
1. Select one function
2. Add parameter for the varying literal
3. Change body to use the parameter
4. Test
5. Change callers to use the parameterized version
6. Remove now-unused functions

**Before**:
```javascript
function tenPercentRaise(person) {
  person.salary = person.salary * 1.10;
}

function fivePercentRaise(person) {
  person.salary = person.salary * 1.05;
}
```

**After**:
```javascript
function raise(person, factor) {
  person.salary = person.salary * (1 + factor);
}

// Usage
raise(person, 0.10);
raise(person, 0.05);
```

---

### Remove Flag Argument

**When to use**: Boolean parameter that changes function behavior

**Motivation**: Make the behavior explicit through separate functions.

**Mechanics**:
1. Create explicit function for each flag value
2. Replace each call with appropriate new function
3. Test after each change
4. Remove original function

**Before**:
```javascript
function bookConcert(customer, isPremium) {
  if (isPremium) {
    // premium booking logic
  } else {
    // regular booking logic
  }
}

bookConcert(customer, true);
bookConcert(customer, false);
```

**After**:
```javascript
function bookPremiumConcert(customer) {
  // premium booking logic
}

function bookRegularConcert(customer) {
  // regular booking logic
}

bookPremiumConcert(customer);
bookRegularConcert(customer);
```

---

## Dealing with Inheritance

### Pull Up Method

**When to use**: Same method in multiple subclasses

**Motivation**: Remove duplication in class hierarchy.

**Mechanics**:
1. Inspect methods to ensure they are identical
2. Check signatures are the same
3. Create new method in superclass
4. Copy body from one subclass
5. Delete one subclass method, test
6. Delete other subclass methods, test each

---

### Push Down Method

**When to use**: Behavior relevant only to a subset of subclasses

**Motivation**: Put method where it's used.

**Mechanics**:
1. Copy method to each subclass that needs it
2. Remove method from superclass
3. Test
4. Remove from subclasses that don't need it
5. Test

---

### Replace Subclass with Delegate

**When to use**: Inheritance is being used incorrectly, need more flexibility

**Motivation**: Prefer composition over inheritance when appropriate.

**Mechanics**:
1. Create empty class for delegate
2. Add field to host class holding delegate
3. Create constructor for delegate, called from host
4. Move features to delegate
5. Test after each move
6. Replace inheritance with delegation

---

## Extract Class

**When to use**: Large class with multiple responsibilities

**Motivation**: Split class to maintain single responsibility.

**Mechanics**:
1. Decide how to split responsibilities
2. Create new class
3. Move field from original to new class
4. Test
5. Move methods from original to new class
6. Test after each move
7. Review and rename both classes
8. Decide how to expose new class

**Before**:
```javascript
class Person {
  get name() { return this._name; }
  set name(arg) { this._name = arg; }
  get officeAreaCode() { return this._officeAreaCode; }
  set officeAreaCode(arg) { this._officeAreaCode = arg; }
  get officeNumber() { return this._officeNumber; }
  set officeNumber(arg) { this._officeNumber = arg; }

  get telephoneNumber() {
    return `(${this._officeAreaCode}) ${this._officeNumber}`;
  }
}
```

**After**:
```javascript
class Person {
  constructor() {
    this._telephoneNumber = new TelephoneNumber();
  }
  get name() { return this._name; }
  set name(arg) { this._name = arg; }
  get telephoneNumber() { return this._telephoneNumber.toString(); }
  get officeAreaCode() { return this._telephoneNumber.areaCode; }
  set officeAreaCode(arg) { this._telephoneNumber.areaCode = arg; }
}

class TelephoneNumber {
  get areaCode() { return this._areaCode; }
  set areaCode(arg) { this._areaCode = arg; }
  get number() { return this._number; }
  set number(arg) { this._number = arg; }
  toString() { return `(${this._areaCode}) ${this._number}`; }
}
```

---

---

## Deep Learning Specific Refactorings

Refactorings tailored for PyTorch deep learning projects.

### Extract Loss Class

**When to use**: Loss computation is inline in training loop or scattered in model code

**Motivation**: Make loss independently testable, swappable, and reusable.

**Mechanics**:
1. Create a class inheriting from `nn.Module`
2. Move loss formula into `forward(self, image_features, text_features, logit_scale)`
3. Handle numerical stability (eps clamping, softplus) inside the class
4. Replace inline loss code with `loss = self.loss_fn(...)`
5. Test: verify loss values match original within tolerance

**Before**:
```python
# Inline in training loop
logits = logit_scale * image_features @ text_features.T
labels = torch.arange(len(images), device=device)
loss_i = F.cross_entropy(logits, labels)
loss_t = F.cross_entropy(logits.T, labels)
loss = (loss_i + loss_t) / 2
```

**After**:
```python
class ClipLoss(nn.Module):
    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        logits = logit_scale * image_features @ text_features.T
        labels = torch.arange(len(logits), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        return {"loss": loss, "logits": logits} if output_dict else loss

# In training loop
loss = loss_fn(image_features, text_features, logit_scale)
```

---

### Extract Training Step

**When to use**: `train()` function > 100 lines, mixing forward/backward/optimizer/logging

**Motivation**: Make the core training step independently testable and composable.

**Mechanics**:
1. Identify the per-batch logic: forward → loss → backward → optimizer step
2. Extract into `training_step(model, loss_fn, optimizer, batch, device)` function
3. Return loss value for logging
4. Test: one step should decrease loss on a small batch
5. Extract logging and checkpointing separately

**Before**:
```python
def train(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for batch in dataloader:
            images, texts = batch
            images, texts = images.cuda(), texts.cuda()
            # 30 lines of forward + loss + backward + step + log
```

**After**:
```python
def training_step(model, loss_fn, optimizer, scaler, batch, device):
    images, texts = batch[0].to(device), batch[1].to(device)
    with torch.cuda.amp.autocast():
        outputs = model(images, texts)
        loss = loss_fn(*outputs[:3])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.detach()

def train(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = ClipLoss()
    for epoch in range(epochs):
        for batch in dataloader:
            loss = training_step(model, loss_fn, optimizer, scaler, batch, "cuda")
            log_metrics(loss)
```

---

### Parameterize Model Config

**When to use**: Hyperparameters hardcoded across multiple files

**Motivation**: Single source of truth for all model/training hyperparameters.

**Mechanics**:
1. Create `@dataclass` classes for model config, training config
2. Move all hardcoded defaults into dataclass fields
3. Use dataclass instance everywhere instead of literals
4. Add `from_json()` / `to_json()` methods for serialization
5. Test: same config → same model

**Before**:
```python
# model.py
model = VisionTransformer(image_size=224, patch_size=32, width=768, layers=12, heads=12)

# train.py
model = VisionTransformer(image_size=224, patch_size=32, width=768, layers=12, heads=12)
```

**After**:
```python
@dataclass
class CLIPVisionCfg:
    image_size: int = 224
    patch_size: int = 32
    width: int = 768
    layers: int = 12
    heads: int = 12

# model.py and train.py
cfg = CLIPVisionCfg()
model = VisionTransformer(
    image_size=cfg.image_size, patch_size=cfg.patch_size,
    width=cfg.width, layers=cfg.layers, heads=cfg.heads
)
```

---

### Introduce Checkpoint Manager

**When to use**: `torch.save()` only stores model weights, missing optimizer/scheduler/scaler

**Motivation**: Enable correct training resume from any checkpoint.

**Mechanics**:
1. Create `save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step)` function
2. Create `load_checkpoint(path, model, optimizer, scheduler, scaler)` function
3. Always save the full training state
4. Test: save → load → train 1 step → compare with uninterrupted training

**Before**:
```python
# Loses optimizer momentum, scheduler state, AMP scaler
torch.save(model.state_dict(), "checkpoint.pt")
model.load_state_dict(torch.load("checkpoint.pt"))
```

**After**:
```python
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, **meta):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "meta": meta,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt["scheduler"] and scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt["scaler"] and scaler:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["global_step"]
```

---

### Extract Data Pipeline

**When to use**: Dataset logic mixed into training script or model code

**Motivation**: Cleanly separate data loading from training logic.

**Mechanics**:
1. Create a `Dataset` subclass for raw data
2. Define transforms as composable callables (use `torchvision.transforms`)
3. Create `DataLoader` in a factory function
4. Test: iterate one epoch, verify shapes and value ranges

**Before**:
```python
def train(model):
    images = load_images("data/")  # Custom loading inline
    images = [resize(img, 224) for img in images]  # Transform inline
    # ... training loop
```

**After**:
```python
class ImageTextDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = load_samples(root)
        self.transform = transform or default_transform()

    def __getitem__(self, idx):
        image, text = self.samples[idx]
        return self.transform(image), text

def create_dataloader(root, batch_size, num_workers=4):
    dataset = ImageTextDataset(root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=True, pin_memory=True)
```

---

### Add Shape Guards

**When to use**: `forward()` has no tensor shape documentation or assertions

**Motivation**: Fail fast with clear error messages when shapes are wrong.

**Mechanics**:
1. Add type annotations to `forward()` signature
2. Add `assert x.dim() == expected` at entry
3. Document expected shapes in docstring
4. Test: wrong shape input → clear error message, not cryptic PyTorch error

**Before**:
```python
class Attention(nn.Module):
    def forward(self, x, k_x=None, v_x=None, attn_mask=None):
        # No shape info — what is x? B,L,D or B,H,L,D?
        q = self.q_proj(x)
        k = self.k_proj(k_x if k_x is not None else x)
        # ...
```

**After**:
```python
class Attention(nn.Module):
    def forward(self,
                x: torch.Tensor,              # [B, L, dim]
                k_x: Optional[torch.Tensor] = None,  # [B, L_k, dim]
                v_x: Optional[torch.Tensor] = None,  # [B, L_k, dim]
                attn_mask: Optional[torch.Tensor] = None,  # [B, L, L]
                ) -> torch.Tensor:            # [B, L, dim]
        assert x.dim() == 3, f"Expected 3D input [B,L,dim], got {x.shape}"
        # ...
```

---

### Wrap with no_grad

**When to use**: Inference/evaluation code lacks `torch.no_grad()` context

**Motivation**: Eliminate unnecessary gradient tracking, reduce memory usage.

**Mechanics**:
1. Add `@torch.no_grad()` decorator to eval/inference methods
2. Or wrap eval loop body with `with torch.no_grad():`
3. Verify with `torch.cuda.max_memory_allocated()` — eval memory should drop significantly
4. Test: run eval twice, memory usage should be stable (not growing)

**Before**:
```python
def evaluate(model, dataloader):
    results = []
    for batch in dataloader:
        output = model(batch["image"].cuda())
        results.append(output.cpu())
    return torch.cat(results)
```

**After**:
```python
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    results = []
    for batch in dataloader:
        output = model(batch["image"].cuda())
        results.append(output.cpu())
    return torch.cat(results)
```

---

### Extract Common Architecture

**When to use**: VisionTransformer and TextTransformer share duplicated attention/FFN patterns

**Motivation**: Eliminate copy-pasted architecture blocks, single fix applies everywhere.

**Mechanics**:
1. Identify the shared patterns (e.g., attention block, MLP, layer norm + residual)
2. Extract into a shared base `Transformer` or `ResidualAttentionBlock`
3. Make variants configurable via parameters (not inheritance)
4. Test: both vision and text paths produce identical outputs after refactoring

**Before**:
```python
# VisionTransformer (200 lines)
class VisionTransformer(nn.Module):
    def __init__(self, ...):
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(d_model, n_head) for _ in range(layers)
        ])

# TextTransformer (200 lines — nearly identical block logic)
class TextTransformer(nn.Module):
    def __init__(self, ...):
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(d_model, n_head) for _ in range(layers)
        ])
```

**After**:
```python
# Shared Transformer backbone
class Transformer(nn.Module):
    def __init__(self, width, layers, heads, block_type="residual", **kwargs):
        block_cls = get_block_class(block_type)
        self.resblocks = nn.ModuleList([
            block_cls(width, heads, **kwargs) for _ in range(layers)
        ])

# Vision and text towers use the same Transformer
class VisionTransformer(nn.Module):
    def __init__(self, ...):
        self.transformer = Transformer(width, layers, heads, **block_kwargs)

class TextTransformer(nn.Module):
    def __init__(self, ...):
        self.transformer = Transformer(width, layers, heads, **block_kwargs)
```

---

### Separate Metrics from Training

**When to use**: Loss aggregation, accuracy calculation, logging mixed into training loop

**Motivation**: Keep training loop focused on optimization, not metrics.

**Mechanics**:
1. Extract metric computation into standalone functions or callable objects
2. Use moving averages for smoother logging
3. Handle distributed aggregation (reduce across ranks)
4. Test: metrics output matches original inline calculation

**Before**:
```python
for batch in dataloader:
    loss = training_step(...)
    total_loss += loss.item()
    batch_count += 1
    if step % 100 == 0:
        avg_loss = total_loss / batch_count
        print(f"Step {step}: loss={avg_loss:.4f}")
        total_loss = 0
        batch_count = 0
```

**After**:
```python
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

loss_meter = AverageMeter()
for batch in dataloader:
    loss = training_step(...)
    loss_meter.update(loss.item())
    if step % log_every == 0:
        log_metrics({"loss": loss_meter.avg, "step": step})
        loss_meter.reset()
```

---

### Replace Magic Numbers with Config

**When to use**: DL-specific magic numbers scattered in code (eps, temperature, dims)

**Motivation**: Make hyperparameter values explicit and centralized.

**Mechanics**:
1. Identify all numeric literals that are hyperparameters
2. Move them to a config dataclass or named module-level constants
3. Add a comment with the source (paper reference, tuning result)
4. Test: behavior unchanged after migration

**Before**:
```python
# Scattered magic numbers
logit_scale = model.logit_scale.exp().clamp(max=100)
eps = 1e-5
temperature = 0.07
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

**After**:
```python
# Named constants with sources
LOGIT_SCALE_CLAMP_MAX = 100  # CLIP paper: prevents overflow
EPS = 1e-5  # Numerical stability for FP16
INIT_TEMPERATURE = 0.07  # CLIP paper: corresponds to logit_scale = ln(1/0.07) ≈ 2.659

logit_scale = model.logit_scale.exp().clamp(max=LOGIT_SCALE_CLAMP_MAX)
image_features = F.normalize(image_features, dim=-1, eps=EPS)
```

---

## Quick Reference: DL Smell to Refactoring

| DL Code Smell | Primary Refactoring | Alternative |
|---------------|-------------------|-------------|
| Training Loop Bloat | Extract Training Step | Separate Metrics from Training |
| Missing Mode Switch | Explicit .train()/.eval() at boundary | Context manager wrapper |
| Autograd Leak | Wrap with no_grad | @torch.no_grad() decorator |
| Inline Loss Computation | Extract Loss Class | Parameterize loss selection |
| Config Scatter | Parameterize Model Config | Replace Magic Numbers with Config |
| Shape Ambiguity | Add Shape Guards | Document shapes in docstring |
| Architecture Duplication | Extract Common Architecture | Pull Up Method |
| Mixed Responsibility | Extract Data Pipeline | Separate Metrics from Training |
| Magic DL Numbers | Replace Magic Numbers with Config | Parameterize Model Config |
| Checkpoint Fragility | Introduce Checkpoint Manager | — |

---

---

## Further Reading

- Fowler, M. (2018). *Refactoring: Improving the Design of Existing Code* (2nd ed.)
- Online catalog: https://refactoring.com/catalog/
