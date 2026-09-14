# Code Review Checklist (Deep Learning)

## Numerical Stability Checklist
- [ ] NaN/Inf values checked after key operations (loss, gradients, activations)
- [ ] Softmax / log_softmax use numerically stable implementations (log-sum-exp)
- [ ] Mixed precision (FP16) ops have adequate gradient scaling / loss scaling
- [ ] Normalization layer eps values are appropriate (not too small for FP16)
- [ ] Log operations have input clamping (e.g., log(0) guarded with eps)
- [ ] Division operations have zero-denominator guards
- [ ] Sigmoid / tanh saturation regions considered for gradient flow
- [ ] Large exponent operations (exp, pow) bounded to prevent overflow

## GPU Memory & Compute Checklist
- [ ] Tensors detached from computation graph when no longer needed
- [ ] retain_graph=False (default) unless specifically required
- [ ] In-place operations (_ variants) verified compatible with autograd
- [ ] Gradient accumulation: loss divided by accumulation steps before backward
- [ ] .item() / .cpu() called only when necessary (minimize GPU→CPU transfers)
- [ ] Tensor allocations reused instead of creating new tensors in loops
- [ ] torch.no_grad() used for inference / evaluation
- [ ] Empty cache (torch.cuda.empty_cache()) not called in training loop

## Distributed Training Checklist
- [ ] all_reduce / all_gather calls use correct op (sum vs mean vs max)
- [ ] Global batch size = per_device_batch_size × num_devices × accumulation_steps
- [ ] Rank 0 guarded operations: logging, checkpoint save, print
- [ ] DistributedSampler used with correct seed and shuffle settings
- [ ] Barrier (dist.barrier()) used before operations that require sync
- [ ] DDP find_unused_parameters set appropriately for the model
- [ ] Broadcast of model state / config done before training starts
- [ ] Gradient sync not broken by conditional backward paths

## Data Pipeline Checklist
- [ ] DataLoader num_workers > 0 properly tested (no pickling errors)
- [ ] Worker init function sets seeds for reproducibility
- [ ] Shuffle enabled for training, disabled for validation
- [ ] DistributedSampler sets epoch each epoch for correct shuffling
- [ ] Dataset __getitem__ returns consistent tensor shapes
- [ ] Data augmentation consistency across train/val split
- [ ] pin_memory=True set for GPU training
- [ ] Prefetch factor and persistent_workers tuned for throughput

## Model Correctness Checklist
- [ ] Loss function matches the task (cross-entropy, contrastive, etc.)
- [ ] Loss reduction (mean vs sum) consistent with gradient accumulation
- [ ] Gradient clipping applied after backward, before optimizer.step()
- [ ] Weight initialization matches model architecture (e.g., trunc_normal for ViT)
- [ ] Tensor shapes verified with assertions at key computation points
- [ ] model.train() called before training, model.eval() before evaluation
- [ ] No gradients computed for frozen / eval-only parameters
- [ ] Output range checked (e.g., logits vs probabilities)

## Reproducibility Checklist
- [ ] torch.manual_seed() set
- [ ] np.random.seed() set
- [ ] random.seed() set
- [ ] torch.cuda.manual_seed_all() set
- [ ] torch.backends.cudnn.deterministic = True (if full reproducibility needed)
- [ ] torch.backends.cudnn.benchmark = False (if full reproducibility needed)
- [ ] DataLoader worker_init_fn sets seeds
- [ ] Checkpoint includes: model state, optimizer state, scheduler state, scaler state, epoch/step

## Training Engineering Checklist
- [ ] Checkpoint save path unique and versioned (not overwriting best by accident)
- [ ] Resume correctly restores epoch, step, optimizer, scheduler, scaler
- [ ] EMA model updated with correct decay and timing (after optimizer step)
- [ ] LR scheduler step called at correct frequency (per-step vs per-epoch)
- [ ] autocast scope covers forward pass, not backward/optimizer step
- [ ] GradScaler used for FP16, GradScaler.step() wrapping optimizer.step()
- [ ] scaler.update() called after optimizer.step()
- [ ] Logging on rank 0 only (or aggregated across ranks)
- [ ] Metrics (loss, accuracy) tracked with moving average, not raw per-step values

## Code Quality Checklist
- [ ] Functions < 50 lines
- [ ] Clear variable naming (no single-letter except loop indices)
- [ ] No duplicate code blocks — extract shared logic
- [ ] Type annotations on function signatures
- [ ] Comments explain WHY, not WHAT
- [ ] No debug print / breakpoint / pdb left in code
- [ ] Config values not hardcoded (use argparse / config dict)
- [ ] Imports organized (stdlib → third-party → local)

## Testing Checklist
- [ ] Shape tests: verify output shapes for key modules given known input shapes
- [ ] Gradient tests: torch.autograd.gradcheck for custom autograd functions
- [ ] Loss sanity: known input pair → expected loss value (within tolerance)
- [ ] Overfit test: model can overfit a small batch (verifies learning capability)
- [ ] Determinism test: same seed → identical outputs
- [ ] Mixed precision test: model trains without NaN/Inf under FP16
- [ ] Resume test: training from checkpoint matches uninterrupted training
- [ ] Edge case tests: empty batch, single sample, max sequence length
