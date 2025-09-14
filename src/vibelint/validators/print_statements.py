"""
Print statement validator using BaseValidator plugin system.

Detects print statements that should be replaced with proper logging
for better maintainability, configurability, and production readiness.

vibelint/validators/print_statements.py
"""

import ast
import fnmatch
from pathlib import Path
from typing import Iterator

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["PrintStatementValidator"]


class PrintStatementValidator(BaseValidator):
    """Validator for detecting print statements."""

    rule_id = "PRINT-STATEMENT"
    name = "Print Statement Checker"
    description = "Detects print() calls that should be replaced with logging"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate print statement usage in a Python file."""
        # Check if file should be excluded based on configuration
        if self._should_exclude_file(file_path):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        visitor = _PrintVisitor()
        visitor.visit(tree)

        for line_num, context in visitor.print_calls:
            message = (
                f"Print statement found{context}. Replace with logging for better maintainability."
            )
            suggestion = "Use logger.info(), logger.debug(), or logger.error() instead"

            yield self.create_finding(
                message=message, file_path=file_path, line=line_num, suggestion=suggestion
            )

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from print statement validation."""
        # Get exclude patterns from configuration
        print_config = self.config.get("print_validation", {})
        exclude_globs = print_config.get(
            "exclude_globs",
            [
                # Default patterns if no configuration is provided
                "test_*.py",
                "*_test.py",
                "conftest.py",
                "tests/**/*.py",
                "cli.py",
                "main.py",
                "__main__.py",
                "*_cli.py",
                "*_cmd.py",
            ],
        )

        # Check if file matches any exclude pattern
        for pattern in exclude_globs:
            # Check against file name
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

            # Check against relative path pattern
            relative_path = str(file_path).replace("\\", "/")  # Normalize path separators
            if fnmatch.fnmatch(relative_path, pattern):
                return True

            # Check against path from parent directories
            for parent in file_path.parents:
                parent_relative = str(file_path.relative_to(parent)).replace("\\", "/")
                if fnmatch.fnmatch(parent_relative, pattern):
                    return True

        return False


class _PrintVisitor(ast.NodeVisitor):
    """AST visitor to detect print statements."""

    def __init__(self):
        self.print_calls = []
        self.current_function = None

    def visit_Call(self, node):
        """Visit Call nodes to detect print() function calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            context = f" in function {self.current_function}" if self.current_function else ""
            self.print_calls.append((node.lineno, context))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit FunctionDef nodes to track current function context for print detection."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
