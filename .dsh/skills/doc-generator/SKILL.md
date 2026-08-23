---
name: doc-generator
description: Generate comprehensive module/API documentation from PyTorch deep learning source code. Extracts nn.Module classes, loss functions, factory functions, and config dataclasses with parameter tables, forward signatures, and tensor shapes. Use when creating or updating documentation for PyTorch projects, or when users mention API docs, module docs, or documentation generation.
user-invocable: true
---
# Deep Learning Module Documentation Generator

Generates structured documentation from PyTorch project source code, covering:

1. **nn.Module Classes** — Inheritance chain, `__init__` parameters, `forward` signature, tensor shapes
2. **Loss Functions** — Input/output shapes, reduction behavior
3. **Factory Functions** — Parameter tables, return types, usage
4. **Config Dataclasses** — Field tables with types and defaults

## Usage

```bash
# Extract all module classes from a file
python generate-docs.py src/model.py --type module

# Extract only factory/builder functions
python generate-docs.py src/factory.py --type factory

# Extract config dataclasses
python generate-docs.py src/model.py --type config

# Extract everything
python generate-docs.py src/model.py --type all

# Plain text output (default is markdown)
python generate-docs.py src/model.py --format plain
```

## Documentation Template

### For nn.Module Classes

```markdown
## ClassName (nn.Module)
**File:** src/path/to/file.py:123
**Inherits:** nn.Module

Class description from docstring or inline comments.

### `__init__` Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| embed_dim | int | — | Embedding dimension |
| ... | ... | ... | ... |

### `forward`
```python
forward(self, x: torch.Tensor) -> torch.Tensor
```

**Input:** x `[B, C, H, W]` input image tensor
**Output:** `[B, output_dim]` output features

### Submodules
- `visual` — VisionTransformer (nn.Module)
- `logit_scale` — nn.Parameter
```

### For Factory Functions

```markdown
## function_name
**File:** src/path/to/file.py:262
**Signature:** 
```python
def create_model(model_name: str, pretrained: str = '') -> CLIP
```

Description extracted from docstring.

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| model_name | str | — | Name of model architecture |
| pretrained | str | '' | Pretrained tag or path |

### Returns
`CLIP` — Instantiated model with loaded weights.
```

### For Config Dataclasses

```markdown
## CLIPVisionCfg (dataclass)
**File:** src/path/to/model.py:37

Vision tower configuration.

### Fields
| Name | Type | Default | Description |
|------|------|---------|-------------|
| image_size | int | 224 | Input image resolution |
| layers | int | 12 | Number of transformer layers |
| width | int | 768 | Hidden dimension |
| patch_size | int | 32 | Patch size for patch embedding |
| ... | ... | ... | ... |
```

## Reference Files

- **`generate-docs.py`** — AST-based extraction script. Reads Python source files, walks the AST, and generates Markdown documentation for nn.Module classes, factory functions, and config dataclasses. Supports `--type` (module/factory/config/all) and `--format` (markdown/plain) options.

## Version History

- v2.0.0 (2026-05-07): Rewritten for deep learning projects — replaced REST API / OpenAPI focus with nn.Module class, loss function, factory function, and config dataclass documentation extraction
- v1.0.0 (2024-12-10): Initial release with REST API endpoint documentation (OpenAPI/Swagger, HTTP methods, JSON responses)
