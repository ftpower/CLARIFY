#!/usr/bin/env python3
"""
Deep Learning Module Documentation Generator.

Extracts structured documentation from PyTorch source code using AST analysis.
Detects nn.Module subclasses, factory functions, and config dataclasses.
"""

import ast
import argparse
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# AST node helpers
# ---------------------------------------------------------------------------

def _is_nn_module_base(base: ast.expr) -> bool:
    """Check if a base class expression is nn.Module or torch.nn.Module."""
    if isinstance(base, ast.Attribute):
        # nn.Module or torch.nn.Module
        return base.attr == "Module"
    if isinstance(base, ast.Name):
        # Module (if directly imported)
        return base.id == "Module"
    return False


def _is_dataclass_decorator(dec: ast.expr) -> bool:
    """Check if a decorator is @dataclass."""
    if isinstance(dec, ast.Name):
        return dec.id == "dataclass"
    if isinstance(dec, ast.Attribute):
        return dec.attr == "dataclass"
    if isinstance(dec, ast.Call):
        func = dec.func
        if isinstance(func, ast.Name):
            return func.id == "dataclass"
        if isinstance(func, ast.Attribute):
            return func.attr == "dataclass"
    return False


def _unparse_annotation(node: Optional[ast.expr]) -> str:
    """Convert an AST annotation node to a string."""
    if node is None:
        return ""
    return ast.unparse(node)


def _unparse_default(node: Optional[ast.expr]) -> str:
    """Convert an AST default value node to a string."""
    if node is None:
        return "—"
    return ast.unparse(node)


def _extract_docstring(node) -> str:
    """Extract docstring from a node, return empty string if none."""
    doc = ast.get_docstring(node)
    return doc.strip() if doc else ""


def _get_first_sentence(text: str) -> str:
    """Get the first sentence or line of a docstring."""
    if not text:
        return ""
    # Take up to first period followed by space/newline, or first newline
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 >= len(text) or text[i + 1] in (" ", "\n")):
            return text[: i + 1]
        if ch == "\n":
            return text[:i]
    return text


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_modules(tree: ast.AST) -> list[dict]:
    """Extract all nn.Module subclasses from an AST."""
    modules = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if any base class is nn.Module
        if not any(_is_nn_module_base(b) for b in node.bases):
            continue

        info = {
            "name": node.name,
            "lineno": node.lineno,
            "docstring": _extract_docstring(node),
            "init_params": [],
            "forward_sig": None,
            "forward_doc": "",
            "submodules": [],
        }

        # Extract __init__ parameters
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                info["init_params"] = _extract_init_params(item)

        # Extract forward method
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                info["forward_sig"] = _format_forward_signature(item)
                info["forward_doc"] = _extract_docstring(item)

        # Extract submodule assignments
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                info["submodules"] = _extract_submodule_assigns(item)

        modules.append(info)

    return modules


def _extract_init_params(func: ast.FunctionDef) -> list[dict]:
    """Extract parameters from an __init__ method, skipping self."""
    params = []
    args = func.args

    # Combine all argument lists, skip self
    all_args = []
    if args.args:
        all_args.extend(args.args[1:])  # Skip self
    if args.posonlyargs:
        all_args.extend(args.posonlyargs[1:])  # Skip self if it's posonly
    if args.kwonlyargs:
        all_args.extend(args.kwonlyargs)

    defaults = args.defaults or []
    kw_defaults = args.kw_defaults or []
    # Pad defaults so they align with the last N positional args
    num_no_default = len(all_args) - len(defaults) - len(kw_defaults)

    for i, arg in enumerate(all_args):
        annotation = _unparse_annotation(arg.annotation)
        # Determine default value
        if i < num_no_default or i - num_no_default >= len(defaults):
            default = "—"
        else:
            default = _unparse_default(defaults[i - num_no_default])

        params.append({
            "name": arg.arg,
            "type": annotation or "—",
            "default": default,
        })

    return params


def _format_forward_signature(func: ast.FunctionDef) -> str:
    """Format a forward method signature as a string."""
    args = []
    for arg in func.args.args:
        a = arg.arg
        if arg.annotation:
            a += f": {ast.unparse(arg.annotation)}"
        args.append(a)

    return_ann = ""
    if func.returns:
        return_ann = f" -> {ast.unparse(func.returns)}"

    params = ", ".join(args)
    return f"forward({params}){return_ann}"


def _is_likely_submodule(node: ast.expr) -> bool:
    """Check if an AST value node looks like an nn.Module, not a plain value.

    Returns True for: Call (constructor), Attribute (e.g. text.transformer),
    BinOp, some comprehensions.
    Returns False for: Name (variable pass-through), Constant, List, Dict,
    Tuple, Set, Lambda.
    """
    if isinstance(node, ast.Call):
        return True
    if isinstance(node, ast.Attribute):
        return True
    if isinstance(node, ast.BinOp):
        return True
    if isinstance(node, (ast.Constant, ast.Name, ast.List, ast.Dict,
                         ast.Tuple, ast.Set, ast.Lambda)):
        return False
    # For anything else (ListComp, IfExp, etc.), be conservative
    return False


def _extract_submodule_assigns(func: ast.FunctionDef) -> list[dict]:
    """Extract self.X = <likely Module>(...) assignments from __init__."""
    submodules = []

    class SubmoduleVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    name = target.attr
                    if name.startswith("_"):
                        continue
                    if not _is_likely_submodule(node.value):
                        continue
                    value_str = ast.unparse(node.value)
                    if len(value_str) > 80:
                        value_str = value_str[:77] + "..."
                    submodules.append({"name": name, "init": value_str})

    SubmoduleVisitor().visit(func)
    return submodules


def extract_factories(tree: ast.AST) -> list[dict]:
    """Extract module-level (non-private) functions."""
    factories = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue

        sig_parts = []
        for arg in node.args.args:
            a = arg.arg
            if arg.annotation:
                a += f": {ast.unparse(arg.annotation)}"
            sig_parts.append(a)

        return_ann = ""
        if node.returns:
            return_ann = f" -> {ast.unparse(node.returns)}"

        params = _extract_func_params(node)

        factories.append({
            "name": node.name,
            "lineno": node.lineno,
            "docstring": _extract_docstring(node),
            "signature": f"def {node.name}({', '.join(sig_parts)}){return_ann}",
            "params": params,
            "return_type": _unparse_annotation(node.returns) or "—",
        })

    return factories


def _extract_func_params(func: ast.FunctionDef) -> list[dict]:
    """Extract parameters from a regular function."""
    params = []
    args = func.args
    all_args = list(args.args)
    defaults = args.defaults or []
    num_no_default = len(all_args) - len(defaults)

    for i, arg in enumerate(all_args):
        annotation = _unparse_annotation(arg.annotation)
        if i < num_no_default:
            default = "—"
        else:
            default = _unparse_default(defaults[i - num_no_default])
        params.append({
            "name": arg.arg,
            "type": annotation or "—",
            "default": default,
        })

    return params


def extract_configs(tree: ast.AST) -> list[dict]:
    """Extract @dataclass decorated classes (config structures)."""
    configs = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check for @dataclass decorator
        has_dataclass = any(_is_dataclass_decorator(d) for d in node.decorator_list)
        if not has_dataclass:
            continue

        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Annotated assignment: name: Type = default
                name = item.target.id
                if name.startswith("_"):
                    continue
                annotation = _unparse_annotation(item.annotation)
                default = _unparse_default(item.value) if item.value else "—"
                fields.append({
                    "name": name,
                    "type": annotation or "—",
                    "default": default,
                })
            elif isinstance(item, ast.Assign):
                # Plain assignment (no type annotation)
                for target in item.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        fields.append({
                            "name": target.id,
                            "type": "—",
                            "default": ast.unparse(item.value),
                        })

        configs.append({
            "name": node.name,
            "lineno": node.lineno,
            "docstring": _extract_docstring(node),
            "fields": fields,
        })

    return configs


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_all_markdown(tree: ast.AST, filepath: str) -> str:
    """Format all extracted documentation as Markdown."""
    output = []
    output.append(f"# Documentation for `{filepath}`")
    output.append("")

    modules = extract_modules(tree)
    factories = extract_factories(tree)
    configs = extract_configs(tree)

    if modules:
        output.append("---")
        output.append("## Module Classes (nn.Module)")
        output.append("")
        for m in modules:
            output.append(_format_module_markdown(m))

    if factories:
        output.append("---")
        output.append("## Functions")
        output.append("")
        for f in factories:
            output.append(_format_factory_markdown(f))

    if configs:
        output.append("---")
        output.append("## Config Dataclasses")
        output.append("")
        for c in configs:
            output.append(_format_config_markdown(c))

    if not (modules or factories or configs):
        output.append("*No documented elements found.*")

    return "\n".join(output)


def _format_module_markdown(m: dict) -> str:
    lines = []
    lines.append(f"### `{m['name']}` (nn.Module)")
    lines.append(f"**Line:** {m['lineno']}")
    lines.append("")

    if m["docstring"]:
        lines.append(_get_first_sentence(m["docstring"]))
        lines.append("")

    # __init__ parameters
    if m["init_params"]:
        lines.append("#### `__init__` Parameters")
        lines.append("")
        lines.append("| Name | Type | Default |")
        lines.append("|------|------|---------|")
        for p in m["init_params"]:
            lines.append(f"| `{p['name']}` | {p['type']} | {p['default']} |")
        lines.append("")

    # forward
    if m["forward_sig"]:
        lines.append("#### `forward`")
        lines.append(f"```python\n{m['forward_sig']}\n```")
        if m["forward_doc"]:
            lines.append("")
            lines.append(m["forward_doc"])
        lines.append("")

    # Submodules
    if m["submodules"]:
        lines.append("#### Submodules")
        lines.append("")
        for s in m["submodules"]:
            lines.append(f"- `{s['name']}` — `{s['init']}`")
        lines.append("")

    return "\n".join(lines)


def _format_factory_markdown(f: dict) -> str:
    lines = []
    lines.append(f"### `{f['name']}`()")
    lines.append(f"**Line:** {f['lineno']}")
    lines.append("")
    lines.append(f"```python\n{f['signature']}\n```")
    lines.append("")

    if f["docstring"]:
        lines.append(_get_first_sentence(f["docstring"]))
        lines.append("")

    if f["params"]:
        lines.append("#### Parameters")
        lines.append("")
        lines.append("| Name | Type | Default |")
        lines.append("|------|------|---------|")
        for p in f["params"]:
            lines.append(f"| `{p['name']}` | {p['type']} | {p['default']} |")
        lines.append("")

    lines.append(f"**Returns:** `{f['return_type']}`")
    lines.append("")
    return "\n".join(lines)


def _format_config_markdown(c: dict) -> str:
    lines = []
    lines.append(f"### `{c['name']}` (dataclass)")
    lines.append(f"**Line:** {c['lineno']}")
    lines.append("")

    if c["docstring"]:
        lines.append(_get_first_sentence(c["docstring"]))
        lines.append("")

    if c["fields"]:
        lines.append("#### Fields")
        lines.append("")
        lines.append("| Name | Type | Default |")
        lines.append("|------|------|---------|")
        for f in c["fields"]:
            lines.append(f"| `{f['name']}` | {f['type']} | {f['default']} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain text formatters (compact, no markdown)
# ---------------------------------------------------------------------------

def format_all_plain(tree: ast.AST, filepath: str) -> str:
    """Format as plain text."""
    output = []
    output.append(f"Documentation for: {filepath}")
    output.append("=" * 60)

    for m in extract_modules(tree):
        output.append(f"\n[MODULE] {m['name']} (nn.Module)  — line {m['lineno']}")
        if m["docstring"]:
            output.append(f"  {_get_first_sentence(m['docstring'])}")
        if m["init_params"]:
            output.append("  __init__ params:")
            for p in m["init_params"]:
                output.append(f"    {p['name']}: {p['type']} = {p['default']}")
        if m["forward_sig"]:
            output.append(f"  forward: {m['forward_sig']}")

    for f in extract_factories(tree):
        output.append(f"\n[FUNC] {f['name']}()  — line {f['lineno']}")
        output.append(f"  {f['signature']}")
        if f["docstring"]:
            output.append(f"  {_get_first_sentence(f['docstring'])}")

    for c in extract_configs(tree):
        output.append(f"\n[CONFIG] {c['name']} (dataclass)  — line {c['lineno']}")
        for f in c["fields"]:
            output.append(f"  {f['name']}: {f['type']} = {f['default']}")

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation from PyTorch source code"
    )
    parser.add_argument("file", help="Python source file to analyze")
    parser.add_argument(
        "--type",
        choices=["module", "factory", "config", "all"],
        default="all",
        help="Type of elements to document (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    args = parser.parse_args()

    with open(args.file) as f:
        source = f.read()

    tree = ast.parse(source)

    # Build output based on type
    if args.format == "markdown":
        if args.type == "all":
            output = format_all_markdown(tree, args.file)
        else:
            output = _format_single_type_markdown(tree, args.file, args.type)
    else:
        if args.type == "all":
            output = format_all_plain(tree, args.file)
        else:
            output = _format_single_type_plain(tree, args.file, args.type)

    print(output)


def _format_single_type_markdown(tree, filepath, doc_type):
    header = f"# `{filepath}` — {doc_type}s\n"
    if doc_type == "module":
        items = extract_modules(tree)
        fmt = _format_module_markdown
    elif doc_type == "factory":
        items = extract_factories(tree)
        fmt = _format_factory_markdown
    else:
        items = extract_configs(tree)
        fmt = _format_config_markdown

    if not items:
        return header + "\n*No {doc_type}s found.*"
    return header + "\n" + "\n---\n".join(fmt(i) for i in items)


def _format_single_type_plain(tree, filepath, doc_type):
    header = f"{filepath} — {doc_type}s\n" + "=" * 60
    if doc_type == "module":
        items = extract_modules(tree)
    elif doc_type == "factory":
        items = extract_factories(tree)
    else:
        items = extract_configs(tree)

    if not items:
        return header + f"\nNo {doc_type}s found."

    lines = [header]
    for item in items:
        lines.append(f"\n{'='*40}")
        for k, v in item.items():
            if v:
                lines.append(f"{k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
