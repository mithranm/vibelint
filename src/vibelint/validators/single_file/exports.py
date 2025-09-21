"""
__all__ exports validator using BaseValidator plugin system.

Checks for presence and correct format of __all__ definitions in Python modules.

vibelint/src/vibelint/validators/exports.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["MissingAllValidator"]


class MissingAllValidator(BaseValidator):
    """Validator for missing __all__ definitions."""

    rule_id = "EXPORTS-MISSING-ALL"
    name = "Missing __all__ Checker"
    description = "Checks for missing __all__ definitions in modules"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing __all__ definition."""
        # Skip if it's __init__.py or private module
        if file_path.name.startswith("_"):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            yield self.create_finding(
                message="SyntaxError parsing file during __all__ validation",
                file_path=file_path,
                line=1,
                suggestion="Fix Python syntax errors in the file",
            )
            return

        # Look for __all__ definition
        has_all = False
        has_exports = False

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        # Check if __all__ is properly formatted
                        if not self._is_valid_all_format(node.value):
                            yield self.create_finding(
                                message="__all__ is not assigned a list or tuple value",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion='Ensure __all__ = ["item1", "item2"] or __all__ = ("item1", "item2")',
                            )
                        break
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith(
                "_"
            ):
                has_exports = True

        if has_exports and not has_all:
            yield self.create_finding(
                message="Module has public functions/classes but no __all__ definition",
                file_path=file_path,
                line=1,
                suggestion="Add __all__ = [...] to explicitly define public API",
            )

    def _is_valid_all_format(self, node: ast.AST) -> bool:
        """Check if __all__ assignment value is a valid list or tuple."""
        if isinstance(node, (ast.List, ast.Tuple)):
            # Check that all elements are strings
            return all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.elts
            )
        return False


class InitAllValidator(BaseValidator):
    """Validator for missing __all__ in __init__.py files."""

    rule_id = "EXPORTS-MISSING-ALL-INIT"
    name = "Missing __all__ in __init__.py"
    description = "__init__.py file is missing __all__ definition (optional based on config)"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing __all__ in __init__.py files."""
        # Only check __init__.py files
        if file_path.name != "__init__.py":
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Look for __all__ definition
        has_all = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        break

        if not has_all:
            yield self.create_finding(
                message="__init__.py file is missing __all__ definition",
                file_path=file_path,
                line=1,
                suggestion="Add __all__ = [...] to control package imports",
            )
