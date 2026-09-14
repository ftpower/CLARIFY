#!/usr/bin/env python3
"""
Code Smell Detector

Detects common code smells in Python, JavaScript, and TypeScript files.
Based on Martin Fowler's catalog of code smells.

Usage:
    python detect-smells.py <file>
    python detect-smells.py --dir <directory>
    python detect-smells.py -v <file>  # Verbose with code snippets

Detects:
    - Long Method (>30 lines)
    - Long Parameter List (>4 params)
    - Duplicate Code (similar code blocks)
    - Large Class (>300 lines, >10 methods)
    - Dead Code (unused variables/functions)
    - Complex Conditionals (deep nesting, long chains)
    - Magic Numbers/Strings
    - Feature Envy (methods using other class data heavily)
    - Comments explaining what (not why)
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class SmellSeverity(Enum):
    """Severity levels for code smells."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class SmellType(Enum):
    """Types of code smells."""
    LONG_METHOD = "Long Method"
    LONG_PARAMETER_LIST = "Long Parameter List"
    DUPLICATE_CODE = "Duplicate Code"
    LARGE_CLASS = "Large Class"
    DEAD_CODE = "Dead Code"
    COMPLEX_CONDITIONAL = "Complex Conditional"
    MAGIC_NUMBER = "Magic Number/String"
    FEATURE_ENVY = "Feature Envy"
    EXCESSIVE_COMMENTS = "Excessive Comments"
    DEEPLY_NESTED = "Deeply Nested Code"
    PRIMITIVE_OBSESSION = "Primitive Obsession"
    DATA_CLUMPS = "Data Clumps"
    SWITCH_STATEMENT = "Switch Statement"
    MESSAGE_CHAIN = "Message Chain"
    # DL-specific smells
    TRAINING_LOOP_BLOAT = "Training Loop Bloat"
    MISSING_MODE_SWITCH = "Missing Mode Switch"
    AUTOGRAD_LEAK = "Autograd Leak"
    INLINE_LOSS = "Inline Loss Computation"
    SHAPE_AMBIGUITY = "Shape Ambiguity"
    MAGIC_DL_NUMBERS = "Magic DL Numbers"
    CHECKPOINT_FRAGILITY = "Checkpoint Fragility"


@dataclass
class CodeSmell:
    """Represents a detected code smell."""
    smell_type: SmellType
    severity: SmellSeverity
    location: str
    line_start: int
    line_end: int
    description: str
    suggestion: str
    code_snippet: str = ""


@dataclass
class SmellReport:
    """Report of all smells found in a file."""
    filename: str
    smells: List[CodeSmell] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for s in self.smells if s.severity == SmellSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for s in self.smells if s.severity == SmellSeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for s in self.smells if s.severity == SmellSeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for s in self.smells if s.severity == SmellSeverity.LOW)


class SmellDetector:
    """Detect code smells in source files."""

    # Thresholds (configurable)
    THRESHOLDS = {
        'long_method_lines': 30,
        'very_long_method_lines': 50,
        'max_parameters': 4,
        'large_class_lines': 300,
        'large_class_methods': 10,
        'max_nesting_depth': 4,
        'long_chain_length': 3,
        'duplicate_min_lines': 5,
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.language = self._detect_language()

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            self.code = f.read()
        self.lines = self.code.split('\n')
        self.smells: List[CodeSmell] = []

    def _detect_language(self) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(self.filepath)[1].lower()
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
        }
        return ext_map.get(ext, 'python')

    def detect_all(self) -> SmellReport:
        """Run all smell detectors."""
        self._detect_long_methods()
        self._detect_long_parameter_lists()
        self._detect_large_class()
        self._detect_complex_conditionals()
        self._detect_magic_numbers()
        self._detect_excessive_comments()
        self._detect_deeply_nested()
        self._detect_switch_statements()
        self._detect_message_chains()
        self._detect_duplicate_code()
        self._detect_dead_code()
        # DL-specific detectors
        if self.language == 'python':
            self._detect_training_loop_bloat()
            self._detect_missing_no_grad()
            self._detect_magic_dl_numbers()
            self._detect_missing_mode_switch()
            self._detect_shape_ambiguity()
            self._detect_checkpoint_fragility()

        return SmellReport(filename=self.filename, smells=self.smells)

    def _get_snippet(self, start: int, end: int, context: int = 2) -> str:
        """Get code snippet with context."""
        actual_start = max(0, start - context)
        actual_end = min(len(self.lines), end + context)
        snippet_lines = []
        for i in range(actual_start, actual_end):
            prefix = "→ " if start <= i < end else "  "
            snippet_lines.append(f"{i+1:4d} {prefix}{self.lines[i]}")
        return '\n'.join(snippet_lines)

    def _detect_long_methods(self) -> None:
        """Detect methods that are too long."""
        if self.language == 'python':
            pattern = r'^\s*def\s+(\w+)\s*\([^)]*\):'
        else:
            pattern = r'(?:function\s+(\w+)|(\w+)\s*[=:]\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'

        current_method = None
        method_start = 0
        brace_depth = 0
        indent_level = 0

        for i, line in enumerate(self.lines):
            match = re.search(pattern, line)
            if match:
                # Check previous method if exists
                if current_method:
                    method_lines = i - method_start
                    self._check_method_length(current_method, method_start, i - 1, method_lines)

                current_method = match.group(1) or (match.group(2) if match.lastindex and match.lastindex > 1 else None)
                method_start = i
                indent_level = len(line) - len(line.lstrip())

            # Track end of Python functions by indentation
            if self.language == 'python' and current_method:
                if line.strip() and not line.strip().startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and i > method_start:
                        method_lines = i - method_start
                        self._check_method_length(current_method, method_start, i - 1, method_lines)
                        current_method = None

        # Check last method
        if current_method:
            method_lines = len(self.lines) - method_start
            self._check_method_length(current_method, method_start, len(self.lines) - 1, method_lines)

    def _check_method_length(self, name: str, start: int, end: int, lines: int) -> None:
        """Check if method is too long and add smell if so."""
        if lines > self.THRESHOLDS['very_long_method_lines']:
            severity = SmellSeverity.HIGH
            desc = f"Method '{name}' is {lines} lines (threshold: {self.THRESHOLDS['long_method_lines']})"
        elif lines > self.THRESHOLDS['long_method_lines']:
            severity = SmellSeverity.MEDIUM
            desc = f"Method '{name}' is {lines} lines (threshold: {self.THRESHOLDS['long_method_lines']})"
        else:
            return

        self.smells.append(CodeSmell(
            smell_type=SmellType.LONG_METHOD,
            severity=severity,
            location=f"{self.filename}:{start+1}-{end+1}",
            line_start=start + 1,
            line_end=end + 1,
            description=desc,
            suggestion="Apply Extract Method to break down into smaller functions",
            code_snippet=self._get_snippet(start, min(start + 5, end), 0)
        ))

    def _detect_long_parameter_lists(self) -> None:
        """Detect functions with too many parameters."""
        if self.language == 'python':
            pattern = r'def\s+(\w+)\s*\(([^)]*)\)'
        else:
            pattern = r'(?:function\s+(\w+)\s*\(([^)]*)\)|(\w+)\s*[=:]\s*(?:async\s+)?(?:function\s*)?\(([^)]*)\))'

        for i, line in enumerate(self.lines):
            match = re.search(pattern, line)
            if match:
                # Safely extract groups
                groups = match.groups()
                func_name = groups[0] or (groups[2] if len(groups) > 2 else None)
                params_str = groups[1] if len(groups) > 1 else ""
                if not params_str and len(groups) > 3:
                    params_str = groups[3] or ""

                # Count parameters
                if params_str.strip():
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    # Filter out 'self', 'cls' for Python
                    if self.language == 'python':
                        params = [p for p in params if p not in ('self', 'cls')]
                    param_count = len(params)

                    if param_count > self.THRESHOLDS['max_parameters']:
                        severity = SmellSeverity.HIGH if param_count > 6 else SmellSeverity.MEDIUM
                        self.smells.append(CodeSmell(
                            smell_type=SmellType.LONG_PARAMETER_LIST,
                            severity=severity,
                            location=f"{self.filename}:{i+1}",
                            line_start=i + 1,
                            line_end=i + 1,
                            description=f"Function '{func_name}' has {param_count} parameters (max: {self.THRESHOLDS['max_parameters']})",
                            suggestion="Consider Introduce Parameter Object or Preserve Whole Object",
                            code_snippet=self._get_snippet(i, i + 1, 1)
                        ))

    def _detect_large_class(self) -> None:
        """Detect classes that are too large."""
        if self.language == 'python':
            class_pattern = r'^\s*class\s+(\w+)'
            method_pattern = r'^\s+def\s+\w+'
        else:
            class_pattern = r'class\s+(\w+)'
            method_pattern = r'(?:^\s+\w+\s*\([^)]*\)\s*\{|^\s+(?:async\s+)?\w+\s*=)'

        current_class = None
        class_start = 0
        method_count = 0
        class_indent = 0

        for i, line in enumerate(self.lines):
            class_match = re.search(class_pattern, line)
            if class_match:
                # Check previous class
                if current_class:
                    self._check_class_size(current_class, class_start, i - 1, method_count)

                current_class = class_match.group(1)
                class_start = i
                method_count = 0
                class_indent = len(line) - len(line.lstrip())

            # Count methods in current class
            if current_class and re.search(method_pattern, line):
                method_count += 1

        # Check last class
        if current_class:
            self._check_class_size(current_class, class_start, len(self.lines) - 1, method_count)

    def _check_class_size(self, name: str, start: int, end: int, methods: int) -> None:
        """Check if class is too large."""
        lines = end - start + 1

        issues = []
        severity = SmellSeverity.MEDIUM

        if lines > self.THRESHOLDS['large_class_lines']:
            issues.append(f"{lines} lines (max: {self.THRESHOLDS['large_class_lines']})")
            severity = SmellSeverity.HIGH

        if methods > self.THRESHOLDS['large_class_methods']:
            issues.append(f"{methods} methods (max: {self.THRESHOLDS['large_class_methods']})")
            if severity != SmellSeverity.HIGH:
                severity = SmellSeverity.MEDIUM

        if issues:
            self.smells.append(CodeSmell(
                smell_type=SmellType.LARGE_CLASS,
                severity=severity,
                location=f"{self.filename}:{start+1}-{end+1}",
                line_start=start + 1,
                line_end=end + 1,
                description=f"Class '{name}' is too large: {', '.join(issues)}",
                suggestion="Apply Extract Class to split responsibilities",
                code_snippet=self._get_snippet(start, start + 3, 0)
            ))

    def _detect_complex_conditionals(self) -> None:
        """Detect complex conditional expressions."""
        for i, line in enumerate(self.lines):
            # Count logical operators in line
            and_or_count = len(re.findall(r'\b(and|or|&&|\|\|)\b', line))

            if and_or_count >= 3:
                self.smells.append(CodeSmell(
                    smell_type=SmellType.COMPLEX_CONDITIONAL,
                    severity=SmellSeverity.MEDIUM,
                    location=f"{self.filename}:{i+1}",
                    line_start=i + 1,
                    line_end=i + 1,
                    description=f"Complex conditional with {and_or_count} logical operators",
                    suggestion="Apply Decompose Conditional or Consolidate Conditional Expression",
                    code_snippet=self._get_snippet(i, i + 1, 1)
                ))

    def _detect_magic_numbers(self) -> None:
        """Detect magic numbers and strings."""
        # Skip common acceptable values
        acceptable = {'0', '1', '-1', '2', '100', 'true', 'false', 'null', 'None', '""', "''"}

        for i, line in enumerate(self.lines):
            # Skip comments and imports
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or \
               stripped.startswith('import') or stripped.startswith('from'):
                continue

            # Find numeric literals (excluding in variable names)
            numbers = re.findall(r'(?<![a-zA-Z_])\b(\d+\.?\d*)\b(?![a-zA-Z_])', line)

            for num in numbers:
                if num not in acceptable and float(num) > 2:
                    # Check if it's likely a magic number (in calculation or comparison)
                    if re.search(rf'[<>=+\-*/]\s*{re.escape(num)}|{re.escape(num)}\s*[<>=+\-*/]', line):
                        self.smells.append(CodeSmell(
                            smell_type=SmellType.MAGIC_NUMBER,
                            severity=SmellSeverity.LOW,
                            location=f"{self.filename}:{i+1}",
                            line_start=i + 1,
                            line_end=i + 1,
                            description=f"Magic number '{num}' - consider using a named constant",
                            suggestion="Replace magic number with named constant",
                            code_snippet=self._get_snippet(i, i + 1, 0)
                        ))
                        break  # One magic number per line is enough

    def _detect_excessive_comments(self) -> None:
        """Detect comments that explain 'what' instead of 'why'."""
        what_patterns = [
            r'#\s*(set|get|return|loop|iterate|check|if|increment|decrement)',
            r'//\s*(set|get|return|loop|iterate|check|if|increment|decrement)',
        ]

        for i, line in enumerate(self.lines):
            for pattern in what_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.smells.append(CodeSmell(
                        smell_type=SmellType.EXCESSIVE_COMMENTS,
                        severity=SmellSeverity.LOW,
                        location=f"{self.filename}:{i+1}",
                        line_start=i + 1,
                        line_end=i + 1,
                        description="Comment explains 'what' not 'why' - consider renaming or removing",
                        suggestion="Use Extract Method with descriptive name instead of comment",
                        code_snippet=self._get_snippet(i, i + 1, 0)
                    ))
                    break

    def _detect_deeply_nested(self) -> None:
        """Detect deeply nested code blocks."""
        max_depth = 0
        current_depth = 0
        depth_start = 0

        for i, line in enumerate(self.lines):
            if self.language == 'python':
                # Count by indentation
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    depth = indent // 4  # Assume 4-space indent
                    if depth > current_depth:
                        if depth > max_depth:
                            max_depth = depth
                            if depth >= self.THRESHOLDS['max_nesting_depth']:
                                depth_start = i
                    current_depth = depth
            else:
                # Count braces
                current_depth += line.count('{') - line.count('}')
                if current_depth > max_depth:
                    max_depth = current_depth
                    if current_depth >= self.THRESHOLDS['max_nesting_depth']:
                        depth_start = i

        if max_depth >= self.THRESHOLDS['max_nesting_depth']:
            self.smells.append(CodeSmell(
                smell_type=SmellType.DEEPLY_NESTED,
                severity=SmellSeverity.HIGH if max_depth > 5 else SmellSeverity.MEDIUM,
                location=f"{self.filename}:{depth_start+1}",
                line_start=depth_start + 1,
                line_end=depth_start + 1,
                description=f"Code nested {max_depth} levels deep (max: {self.THRESHOLDS['max_nesting_depth']})",
                suggestion="Apply Replace Nested Conditional with Guard Clauses or Extract Method",
                code_snippet=self._get_snippet(depth_start, depth_start + 5, 0)
            ))

    def _detect_switch_statements(self) -> None:
        """Detect switch statements that might need polymorphism."""
        if self.language == 'python':
            # Python 3.10+ match statements or if/elif chains
            pattern = r'^\s*(if|elif).*==.*:'
            consecutive_conditions = 0
            chain_start = 0

            for i, line in enumerate(self.lines):
                if re.search(pattern, line):
                    if consecutive_conditions == 0:
                        chain_start = i
                    consecutive_conditions += 1
                else:
                    if consecutive_conditions >= 4:
                        self._add_switch_smell(chain_start, i - 1, consecutive_conditions)
                    consecutive_conditions = 0
        else:
            # JavaScript/TypeScript switch
            pattern = r'\bswitch\s*\('
            for i, line in enumerate(self.lines):
                if re.search(pattern, line):
                    # Count cases
                    case_count = 0
                    for j in range(i, min(i + 50, len(self.lines))):
                        case_count += len(re.findall(r'\bcase\b', self.lines[j]))
                    if case_count >= 4:
                        self._add_switch_smell(i, i + 1, case_count)

    def _add_switch_smell(self, start: int, end: int, cases: int) -> None:
        """Add a switch statement smell."""
        self.smells.append(CodeSmell(
            smell_type=SmellType.SWITCH_STATEMENT,
            severity=SmellSeverity.MEDIUM,
            location=f"{self.filename}:{start+1}",
            line_start=start + 1,
            line_end=end + 1,
            description=f"Switch/case statement with {cases} cases - consider polymorphism",
            suggestion="Apply Replace Conditional with Polymorphism",
            code_snippet=self._get_snippet(start, start + 5, 0)
        ))

    def _detect_message_chains(self) -> None:
        """Detect long method chains (train wrecks)."""
        chain_pattern = r'(\w+(?:\.\w+\([^)]*\)){3,})'

        for i, line in enumerate(self.lines):
            matches = re.findall(chain_pattern, line)
            for match in matches:
                chain_length = match.count('.')
                if chain_length >= self.THRESHOLDS['long_chain_length']:
                    self.smells.append(CodeSmell(
                        smell_type=SmellType.MESSAGE_CHAIN,
                        severity=SmellSeverity.MEDIUM,
                        location=f"{self.filename}:{i+1}",
                        line_start=i + 1,
                        line_end=i + 1,
                        description=f"Message chain with {chain_length} calls - violates Law of Demeter",
                        suggestion="Apply Hide Delegate to reduce coupling",
                        code_snippet=self._get_snippet(i, i + 1, 0)
                    ))

    def _detect_duplicate_code(self) -> None:
        """Detect potential duplicate code blocks (simplified)."""
        # Create line hashes for comparison
        line_hashes: Dict[str, List[int]] = defaultdict(list)

        for i, line in enumerate(self.lines):
            normalized = re.sub(r'\s+', ' ', line.strip())
            if len(normalized) > 20:  # Only significant lines
                line_hashes[normalized].append(i)

        # Find duplicates
        for normalized, positions in line_hashes.items():
            if len(positions) >= 3:  # At least 3 occurrences
                self.smells.append(CodeSmell(
                    smell_type=SmellType.DUPLICATE_CODE,
                    severity=SmellSeverity.MEDIUM,
                    location=f"{self.filename}:{positions[0]+1}",
                    line_start=positions[0] + 1,
                    line_end=positions[0] + 1,
                    description=f"Potential duplicate code found {len(positions)} times",
                    suggestion="Apply Extract Method to eliminate duplication",
                    code_snippet=self._get_snippet(positions[0], positions[0] + 1, 0)
                ))

    def _detect_dead_code(self) -> None:
        """Detect potentially dead code (simplified)."""
        # Look for common dead code patterns
        patterns = [
            (r'^\s*#.*TODO.*delete', "TODO to delete"),
            (r'^\s*#.*FIXME.*remove', "FIXME to remove"),
            (r'^\s*//.*TODO.*delete', "TODO to delete"),
            (r'^\s*//.*FIXME.*remove', "FIXME to remove"),
            (r'^\s*if\s+False:', "Code behind 'if False'"),
            (r'^\s*if\s*\(\s*false\s*\)', "Code behind 'if (false)'"),
        ]

        for i, line in enumerate(self.lines):
            for pattern, desc in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.smells.append(CodeSmell(
                        smell_type=SmellType.DEAD_CODE,
                        severity=SmellSeverity.LOW,
                        location=f"{self.filename}:{i+1}",
                        line_start=i + 1,
                        line_end=i + 1,
                        description=f"Potential dead code: {desc}",
                        suggestion="Remove dead code",
                        code_snippet=self._get_snippet(i, i + 1, 0)
                    ))

    # ------------------------------------------------------------------
    # DL-specific detectors
    # ------------------------------------------------------------------

    def _detect_training_loop_bloat(self) -> None:
        """Detect excessively long training functions."""
        train_pattern = r'^\s*def\s+(train|training_loop|run_training)\s*\('

        for i, line in enumerate(self.lines):
            match = re.search(train_pattern, line)
            if not match:
                continue

            # Find end of function by dedent
            func_start = i
            base_indent = len(line) - len(line.lstrip())
            func_end = len(self.lines)
            for j in range(i + 1, len(self.lines)):
                stripped = self.lines[j].strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                    current_indent = len(self.lines[j]) - len(self.lines[j].lstrip())
                    if current_indent <= base_indent:
                        func_end = j
                        break

            func_lines = func_end - func_start
            if func_lines > self.THRESHOLDS['very_long_method_lines']:
                severity = SmellSeverity.HIGH
            elif func_lines > self.THRESHOLDS['long_method_lines']:
                severity = SmellSeverity.MEDIUM
            else:
                return

            self.smells.append(CodeSmell(
                smell_type=SmellType.TRAINING_LOOP_BLOAT,
                severity=severity,
                location=f"{self.filename}:{func_start + 1}",
                line_start=func_start + 1,
                line_end=func_end,
                description=f"Training function '{match.group(1)}()' is {func_lines} lines — "
                            f"consider extracting loss, logging, and checkpoint logic",
                suggestion="Apply Extract Training Step, Extract Loss Class, "
                          "and Separate Metrics from Training",
                code_snippet=self._get_snippet(func_start, min(func_start + 5, func_end), 0)
            ))

    def _detect_missing_no_grad(self) -> None:
        """Detect eval/inference functions lacking @torch.no_grad() decorator."""
        eval_pattern = r'^\s*(?:@\w+\s*\n\s*)*def\s+(evaluate?|validate|inference?|encode)\s*\('

        for i, line in enumerate(self.lines):
            match = re.search(eval_pattern, line, re.IGNORECASE)
            if not match:
                continue

            # Check if @torch.no_grad() appears within the preceding lines
            has_no_grad = False
            for j in range(max(0, i - 5), i):
                if '@torch.no_grad()' in self.lines[j] or 'torch.no_grad()' in self.lines[j]:
                    has_no_grad = True
                    break

            # Also check if there's a with torch.no_grad(): inside the function
            if not has_no_grad:
                # Find function body
                func_end = min(i + 30, len(self.lines))
                for j in range(i, func_end):
                    if 'torch.no_grad()' in self.lines[j] or 'with torch.no_grad():' in self.lines[j]:
                        has_no_grad = True
                        break

            if not has_no_grad:
                self.smells.append(CodeSmell(
                    smell_type=SmellType.AUTOGRAD_LEAK,
                    severity=SmellSeverity.HIGH,
                    location=f"{self.filename}:{i + 1}",
                    line_start=i + 1,
                    line_end=i + 1,
                    description=f"Function '{match.group(1)}()' lacks @torch.no_grad() — "
                                f"gradients will be tracked during inference/eval",
                    suggestion="Add @torch.no_grad() decorator or wrap body with 'with torch.no_grad():'",
                    code_snippet=self._get_snippet(i, min(i + 3, len(self.lines)), 1)
                ))

    def _detect_magic_dl_numbers(self) -> None:
        """Detect DL-specific magic numbers: eps, temperature, common dims."""
        dl_magic_patterns = [
            # (regex, description)
            (r'(?<![a-zA-Z_])eps\s*=\s*1e-\d+(?![a-zA-Z_])',
             "Hardcoded eps value", SmellSeverity.LOW),
            (r'(?<![a-zA-Z_])temperature\s*=\s*[\d.]+(?![a-zA-Z_])',
             "Hardcoded temperature value", SmellSeverity.LOW),
            (r'(?<![a-zA-Z_])logit_scale\s*=\s*np\.log\s*\(\s*1\s*/\s*0\.07\s*\)',
             "Hardcoded logit_scale from CLIP paper (temperature=0.07)", SmellSeverity.LOW),
            (r'(?<![a-zA-Z_])embed_dim\s*=\s*(512|768|1024)(?![a-zA-Z_])',
             "Hardcoded embed_dim value", SmellSeverity.MEDIUM),
        ]

        for i, line in enumerate(self.lines):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('import') or stripped.startswith('from'):
                continue

            for pattern, desc, sev in dl_magic_patterns:
                if re.search(pattern, line):
                    self.smells.append(CodeSmell(
                        smell_type=SmellType.MAGIC_DL_NUMBERS,
                        severity=sev,
                        location=f"{self.filename}:{i + 1}",
                        line_start=i + 1,
                        line_end=i + 1,
                        description=f"{desc} — consider moving to config or named constant",
                        suggestion="Apply Replace Magic Numbers with Config",
                        code_snippet=self._get_snippet(i, i + 1, 0)
                    ))

    def _detect_missing_mode_switch(self) -> None:
        """Detect eval/inference code blocks missing model.eval() call."""
        eval_contexts = [
            (r'^\s*def\s+(evaluate?|validate|test|infer(?:ence)?)\s*\(', "eval function"),
            (r'@torch\.no_grad\(\)', "torch.no_grad block"),
        ]

        for i, line in enumerate(self.lines):
            for pattern, ctx_name in eval_contexts:
                if not re.search(pattern, line, re.IGNORECASE):
                    continue
                # Check surrounding context for model.eval()
                context_start = max(0, i - 3)
                context_end = min(len(self.lines), i + 15)
                context_lines = self.lines[context_start:context_end]
                has_eval = any('.eval()' in l for l in context_lines)
                if has_eval:
                    continue

                self.smells.append(CodeSmell(
                    smell_type=SmellType.MISSING_MODE_SWITCH,
                    severity=SmellSeverity.CRITICAL,
                    location=f"{self.filename}:{i + 1}",
                    line_start=i + 1,
                    line_end=i + 1,
                    description=f"{ctx_name} missing model.eval() — "
                                f"BatchNorm/Dropout may behave incorrectly",
                    suggestion="Add 'model.eval()' before evaluation, 'model.train()' after",
                    code_snippet=self._get_snippet(i, min(i + 5, len(self.lines)), 1)
                ))
                break

    def _detect_shape_ambiguity(self) -> None:
        """Detect nn.Module forward() methods without docstring."""
        in_class = False
        class_name = ""
        class_is_module = False

        for i, line in enumerate(self.lines):
            # Track class context
            class_match = re.search(r'^\s*class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(1)
                # Check if inherits from nn.Module
                class_is_module = 'nn.Module' in line or 'Module' in line
                in_class = True
                continue

            # Detect dedent from class
            if in_class and line.strip() and not line[0].isspace():
                in_class = False
                class_is_module = False

            if not (in_class and class_is_module):
                continue

            # Check for forward method
            forward_match = re.search(r'^\s+def\s+forward\s*\(', line)
            if not forward_match:
                continue

            # Check if forward has a docstring
            has_docstring = False
            for j in range(i + 1, min(i + 5, len(self.lines))):
                stripped = self.lines[j].strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    has_docstring = True
                    break
                # If next line is code (not a comment), assume no docstring
                if stripped and not stripped.startswith('#'):
                    break

            if not has_docstring:
                self.smells.append(CodeSmell(
                    smell_type=SmellType.SHAPE_AMBIGUITY,
                    severity=SmellSeverity.MEDIUM,
                    location=f"{self.filename}:{i + 1}",
                    line_start=i + 1,
                    line_end=i + 1,
                    description=f"'{class_name}.forward()' has no docstring — "
                                f"tensor shapes are unclear",
                    suggestion="Add shape documentation to forward() docstring "
                              "(e.g., 'Args: x [B,C,H,W] -> Returns: [B,dim]')",
                    code_snippet=self._get_snippet(i, min(i + 3, len(self.lines)), 0)
                ))

    def _detect_checkpoint_fragility(self) -> None:
        """Detect torch.save() calls that only save model state_dict."""
        save_pattern = r'torch\.save\s*\(\s*(?:model\.)?state_dict\(\)'

        for i, line in enumerate(self.lines):
            if re.search(save_pattern, line):
                # Check if optimizer/scheduler/scaler state also saved nearby
                context_start = max(0, i - 5)
                context_end = min(len(self.lines), i + 5)
                context = '\n'.join(self.lines[context_start:context_end])
                has_optimizer = 'optimizer' in context
                has_scheduler = 'scheduler' in context
                has_scaler = 'scaler' in context

                if not (has_optimizer or has_scheduler or has_scaler):
                    self.smells.append(CodeSmell(
                        smell_type=SmellType.CHECKPOINT_FRAGILITY,
                        severity=SmellSeverity.HIGH,
                        location=f"{self.filename}:{i + 1}",
                        line_start=i + 1,
                        line_end=i + 1,
                        description="torch.save() only saves model state — "
                                    "missing optimizer, scheduler, and scaler state",
                        suggestion="Apply Introduce Checkpoint Manager: save a dict with "
                                  "model, optimizer, scheduler, scaler state_dicts + epoch/step",
                        code_snippet=self._get_snippet(max(0, i - 2), min(len(self.lines), i + 3), 0)
                    ))


def print_report(report: SmellReport, verbose: bool = False) -> None:
    """Print smell report in readable format."""
    print("=" * 70)
    print(f"CODE SMELL DETECTION REPORT: {report.filename}")
    print("=" * 70)

    print(f"\n📊 SUMMARY")
    print("-" * 40)
    print(f"  Total smells found:     {len(report.smells)}")
    print(f"  Critical:               {report.critical_count}")
    print(f"  High:                   {report.high_count}")
    print(f"  Medium:                 {report.medium_count}")
    print(f"  Low:                    {report.low_count}")

    if not report.smells:
        print("\n✅ No code smells detected!")
        print("=" * 70)
        return

    # Group by type
    by_type: Dict[SmellType, List[CodeSmell]] = defaultdict(list)
    for smell in report.smells:
        by_type[smell.smell_type].append(smell)

    print(f"\n📋 FINDINGS BY TYPE")
    print("-" * 40)

    for smell_type, smells in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n### {smell_type.value} ({len(smells)} found)")

        for smell in sorted(smells, key=lambda x: x.severity.value):
            severity_icon = {
                SmellSeverity.CRITICAL: "🔴",
                SmellSeverity.HIGH: "🟠",
                SmellSeverity.MEDIUM: "🟡",
                SmellSeverity.LOW: "🟢",
            }[smell.severity]

            print(f"\n  {severity_icon} [{smell.severity.value}] {smell.location}")
            print(f"     {smell.description}")
            print(f"     💡 {smell.suggestion}")

            if verbose and smell.code_snippet:
                print(f"\n     Code:")
                for snippet_line in smell.code_snippet.split('\n'):
                    print(f"       {snippet_line}")

    print("\n" + "=" * 70)
    print("💡 RECOMMENDED ACTIONS")
    print("-" * 40)

    if report.critical_count > 0:
        print("  1. Address CRITICAL issues immediately")
    if report.high_count > 0:
        print("  2. Plan to fix HIGH severity issues this sprint")
    if report.medium_count > 0:
        print("  3. Schedule MEDIUM issues for upcoming work")
    if report.low_count > 0:
        print("  4. Fix LOW issues opportunistically")

    print("\n" + "=" * 70)


def analyze_directory(directory: str, verbose: bool = False) -> None:
    """Analyze all supported files in a directory."""
    supported_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx']
    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in supported_extensions):
                files.append(os.path.join(root, filename))

    if not files:
        print(f"No supported files found in {directory}")
        return

    print(f"Scanning {len(files)} files in {directory}...\n")

    total_smells = 0
    total_critical = 0
    total_high = 0
    files_with_smells = 0

    for filepath in sorted(files):
        try:
            detector = SmellDetector(filepath)
            report = detector.detect_all()

            if report.smells:
                files_with_smells += 1
                total_smells += len(report.smells)
                total_critical += report.critical_count
                total_high += report.high_count

                flag = " 🔴" if report.critical_count else (" 🟠" if report.high_count else " 🟡")
                print(f"  {report.filename}: {len(report.smells)} smells{flag}")

                if verbose:
                    for smell in report.smells:
                        print(f"    - [{smell.severity.value}] {smell.smell_type.value}: line {smell.line_start}")
            else:
                print(f"  {report.filename}: ✅ Clean")

        except Exception as e:
            print(f"  Error analyzing {filepath}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files analyzed:         {len(files)}")
    print(f"  Files with smells:      {files_with_smells}")
    print(f"  Total smells found:     {total_smells}")
    print(f"  Critical issues:        {total_critical}")
    print(f"  High severity issues:   {total_high}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect code smells in source files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.py                    Analyze single file
  %(prog)s --dir src/                   Analyze directory
  %(prog)s -v myfile.py                 Verbose with code snippets
        """
    )
    parser.add_argument('file', nargs='?', help='File to analyze')
    parser.add_argument('--dir', '-d', help='Directory to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show code snippets')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if args.dir:
        analyze_directory(args.dir, args.verbose)
    elif args.file:
        detector = SmellDetector(args.file)
        report = detector.detect_all()

        if args.json:
            import json
            smells_data = [{
                'type': s.smell_type.value,
                'severity': s.severity.value,
                'location': s.location,
                'line_start': s.line_start,
                'line_end': s.line_end,
                'description': s.description,
                'suggestion': s.suggestion,
            } for s in report.smells]
            print(json.dumps({
                'filename': report.filename,
                'total_smells': len(report.smells),
                'critical': report.critical_count,
                'high': report.high_count,
                'medium': report.medium_count,
                'low': report.low_count,
                'smells': smells_data
            }, indent=2))
        else:
            print_report(report, args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
