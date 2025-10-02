"""
Code smell detection validator implementing Martin Fowler's taxonomy.

Detects common code smells like long methods, large classes, magic numbers,
and other patterns that indicate design issues.

vibelint/src/vibelint/validators/code_smells.py
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ...validators.types import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["CodeSmellValidator"]


class CodeSmellValidator(BaseValidator):
    """Detects common code smells based on Martin Fowler's taxonomy."""

    rule_id = "CODE-SMELLS"
    name = "Code Smell Detector"
    description = "Detects long methods, large classes, magic numbers, and other code smells"
    default_severity = Severity.INFO

    def validate(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Single-pass AST analysis for code smell detection."""

        # Check file length first (doesn't require AST parsing)
        yield from self._check_file_length(file_path, content, config)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        # Single AST walk detecting all smell categories
        for node in ast.walk(tree):
            yield from self._check_bloaters(node, file_path)
            yield from self._check_lexical_abusers(node, file_path)
            yield from self._check_couplers(node, file_path)
            yield from self._check_obfuscators(node, file_path)

    def _check_bloaters(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Bloater code smells: Large Class, Long Method, Long Parameter List."""
        # Long Method (>20 lines suspicious, >50 bad)
        if isinstance(node, ast.FunctionDef):
            method_length = self._count_logical_lines(node)
            if method_length > 50:
                yield self.create_finding(
                    message=f"Method '{node.name}' is too long ({method_length} lines)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break method into smaller, focused functions",
                )
            elif method_length > 20:
                yield self.create_finding(
                    message=f"Method '{node.name}' is getting long ({method_length} lines)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider breaking into smaller functions",
                )

        # Large Class (>500 lines or >20 methods)
        elif isinstance(node, ast.ClassDef):
            class_length = self._count_logical_lines(node)
            method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
            if class_length > 500 or method_count > 20:
                yield self.create_finding(
                    message=f"Class '{node.name}' is too large ({class_length} lines, {method_count} methods)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Split class responsibilities using Single Responsibility Principle",
                )

        # Long Parameter List (>3 suspicious, >5 bad)
        if isinstance(node, ast.FunctionDef):
            param_count = len(node.args.args)
            if param_count > 5:
                yield self.create_finding(
                    message=f"Function '{node.name}' has too many parameters ({param_count})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider parameter object or builder pattern",
                )

    def _check_lexical_abusers(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Lexical Abuser code smells: Magic Numbers, Uncommunicative Names."""
        # Magic Numbers
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in [0, 1, -1, 2] and not self._is_in_test_context(node):
                yield self.create_finding(
                    message=f"Magic number '{node.value}' should be named constant",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion=f"Replace with named constant: MEANINGFUL_NAME = {node.value}",
                )

        # Uncommunicative Names
        name_patterns = [
            (ast.FunctionDef, "function", "name"),
            (ast.ClassDef, "class", "name"),
            (ast.arg, "parameter", "arg"),
        ]
        for node_type, context, attr in name_patterns:
            if isinstance(node, node_type):
                if hasattr(node, attr):
                    name = getattr(node, attr)
                    if name and self._is_uncommunicative_name(name):
                        yield self.create_finding(
                            message=f"Uncommunicative {context} name '{name}'",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion=f"Use descriptive name that explains {context} purpose",
                        )

    def _check_couplers(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Coupler code smells: Message Chain, Feature Envy."""
        # Message Chain (a.b.c.d.method())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain_length = self._count_attribute_chain(node.func)
            if chain_length > 3:
                yield self.create_finding(
                    message=f"Long message chain detected ({chain_length} levels)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider introducing intermediate methods to reduce coupling",
                )

    def _check_obfuscators(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Obfuscator code smells: Complicated Boolean Expression, Clever Code."""
        # Complicated Boolean Expression
        if isinstance(node, ast.BoolOp):
            complexity = self._calculate_boolean_complexity(node)
            if complexity > 4:
                yield self.create_finding(
                    message=f"Complex boolean expression (complexity: {complexity})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break into intermediate boolean variables with descriptive names",
                )

        # Clever Code (nested comprehensions)
        if isinstance(node, ast.ListComp):
            nesting_level = self._count_comprehension_nesting(node)
            if nesting_level > 2:
                yield self.create_finding(
                    message=f"Overly complex list comprehension (nesting level: {nesting_level})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break into multiple steps or use traditional loops for clarity",
                )

    # Helper methods
    def _count_logical_lines(self, node: ast.AST) -> int:
        """Count logical lines of code (excluding comments/blank lines)."""
        lines = set()
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                lines.add(child.lineno)
        return len(lines)

    def _is_uncommunicative_name(self, name: str) -> bool:
        """Check if name is uncommunicative."""
        if len(name) == 1 and name not in ["i", "j", "k", "x", "y", "z"]:
            return True
        if len(name) > 2 and not any(c in "aeiou" for c in name.lower()):
            return True
        return False

    def _count_attribute_chain(self, node: ast.Attribute) -> int:
        """Count depth of attribute chain."""
        count = 1
        current = node.value
        while isinstance(current, ast.Attribute):
            count += 1
            current = current.value
        return count

    def _calculate_boolean_complexity(self, node: ast.BoolOp) -> int:
        """Calculate complexity of boolean expression."""
        complexity = 1
        for value in node.values:
            if isinstance(value, ast.BoolOp):
                complexity += self._calculate_boolean_complexity(value)
            else:
                complexity += 1
        return complexity

    def _count_comprehension_nesting(self, node: ast.ListComp) -> int:
        """Count nesting level of comprehensions."""
        max_nesting = 1
        for generator in node.generators:
            for comp in ast.walk(generator.iter):
                if isinstance(comp, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    max_nesting = max(max_nesting, 1 + self._count_comprehension_nesting(comp))
        return max_nesting

    def _is_in_test_context(self, node: ast.AST) -> bool:
        """Check if node is in test context where magic numbers are more acceptable."""
        return False  # Simplified for now

    def _check_file_length(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Check if file exceeds recommended line count limits (FILE-TOO-LONG smell)."""
        # Default thresholds can be overridden in config
        warning_threshold = 500
        error_threshold = 1000
        exclude_patterns = ["test_*.py", "*_test.py", "conftest.py", "__init__.py"]

        if config:
            warning_threshold = config.get("line_count_warning", warning_threshold)
            error_threshold = config.get("line_count_error", error_threshold)
            exclude_patterns = config.get("line_count_exclude", exclude_patterns)

        # Skip excluded files
        filename = file_path.name
        for pattern in exclude_patterns:
            if (
                filename == pattern
                or (pattern.startswith("*") and filename.endswith(pattern[1:]))
                or (pattern.endswith("*") and filename.startswith(pattern[:-1]))
            ):
                return

        # Count non-empty lines (excluding pure whitespace)
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        line_count = len(non_empty_lines)

        # Check thresholds
        if line_count >= error_threshold:
            yield self.create_finding(
                message=f"File is too long ({line_count} lines, threshold: {error_threshold})",
                file_path=file_path,
                line=1,
                suggestion=self._get_file_split_suggestion(file_path, line_count, content),
            )
        elif line_count >= warning_threshold:
            yield self.create_finding(
                message=f"File is getting long ({line_count} lines, threshold: {warning_threshold})",
                file_path=file_path,
                line=1,
                suggestion=self._get_file_split_suggestion(file_path, line_count, content),
            )

    def _get_file_split_suggestion(self, file_path: Path, line_count: int, content: str) -> str:
        """Generate context-appropriate file splitting suggestions."""
        try:
            tree = ast.parse(content)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if len(classes) > 1:
                return f"Split {len(classes)} classes into separate files"
            elif len(functions) > 10:
                return f"Group {len(functions)} functions into classes or separate modules"
            elif "cli" in str(file_path).lower():
                return "Break CLI into command modules (validation, analysis, etc.)"
            elif "config" in str(file_path).lower():
                return "Separate config loading, validation, and defaults"
            else:
                return "Extract classes or functions into separate modules"
        except (SyntaxError, UnicodeDecodeError):
            return f"Break this {line_count}-line file into smaller, focused modules"
