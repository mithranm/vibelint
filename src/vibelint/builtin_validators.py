"""
Built-in validators migrated to the plugin system.

These are the core validators that ship with vibelint, converted to use
the new BaseValidator interface.
"""

import ast
from pathlib import Path
from typing import Iterator

from .plugin_system import BaseValidator, Finding, Severity

__all__ = ["PrintStatementValidator", "MissingAllValidator", "MissingDocstringValidator", "BUILTIN_VALIDATORS"]


class PrintStatementValidator(BaseValidator):
    """Validator for detecting print statements."""

    rule_id = "VBL701"
    name = "Print Statement Checker"
    description = "Detects print() calls that should be replaced with logging"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate print statement usage in a Python file."""
        # Skip test files
        if self._is_test_file(file_path):
            return

        # Skip CLI files
        if self._is_cli_file(file_path):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        visitor = _PrintVisitor()
        visitor.visit(tree)

        for line_num, context in visitor.print_calls:
            message = f"Print statement found{context}. Replace with logging for better maintainability."
            suggestion = "Use logger.info(), logger.debug(), or logger.error() instead"

            yield self.create_finding(
                message=message,
                file_path=file_path,
                line=line_num,
                suggestion=suggestion
            )

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file appears to be a test file."""
        name = file_path.name.lower()
        parent_names = [p.name.lower() for p in file_path.parents]

        return (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
            or "test" in parent_names
            or "tests" in parent_names
        )

    def _is_cli_file(self, file_path: Path) -> bool:
        """Check if file appears to be a CLI script."""
        name = file_path.name.lower()
        cli_indicators = ["cli.py", "main.py", "__main__.py", "command.py", "cmd.py"]
        return name in cli_indicators or name.endswith("_cli.py") or name.endswith("_cmd.py")


class MissingAllValidator(BaseValidator):
    """Validator for missing __all__ definitions."""

    rule_id = "VBL301"
    name = "Missing __all__ Checker"
    description = "Checks for missing __all__ definitions in modules"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for missing __all__ definition."""
        # Skip if it's __init__.py or private module
        if file_path.name.startswith("_"):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Look for __all__ definition
        has_all = False
        has_exports = False

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        break
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                has_exports = True

        if has_exports and not has_all:
            yield self.create_finding(
                message="Module has public functions/classes but no __all__ definition",
                file_path=file_path,
                line=1,
                suggestion="Add __all__ = [...] to explicitly define public API"
            )


class MissingDocstringValidator(BaseValidator):
    """Validator for missing docstrings."""

    rule_id = "VBL101"
    name = "Missing Docstring Checker"
    description = "Checks for missing docstrings in modules, classes, and functions"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for missing docstrings."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Check module docstring
        if not ast.get_docstring(tree):
            yield self.create_finding(
                message="Module is missing docstring",
                file_path=file_path,
                line=1,
                suggestion="Add a module-level docstring explaining the module's purpose"
            )

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith("_") and not ast.get_docstring(node):
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    yield self.create_finding(
                        message=f"{node_type} '{node.name}' is missing docstring",
                        file_path=file_path,
                        line=node.lineno,
                        suggestion=f"Add docstring to {node.name}() explaining its purpose"
                    )


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


# Registry of built-in validators
BUILTIN_VALIDATORS = [
    PrintStatementValidator,
    MissingAllValidator,
    MissingDocstringValidator
]
