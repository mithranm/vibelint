"""
Docstring validator using BaseValidator plugin system.

Checks for missing docstrings and proper path references in modules,
classes, and functions.

vibelint/src/vibelint/validators/docstring.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["MissingDocstringValidator", "DocstringPathValidator"]


class MissingDocstringValidator(BaseValidator):
    """Validator for missing docstrings."""

    rule_id = "DOCSTRING-MISSING"
    name = "Missing Docstring Checker"
    description = "Checks for missing docstrings in modules, classes, and functions"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
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
                suggestion="Add a module-level docstring explaining the module's purpose",
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
                        suggestion=f"Add docstring to {node.name}() explaining its purpose",
                    )


class DocstringPathValidator(BaseValidator):
    """Validator for missing path references in docstrings."""

    rule_id = "DOCSTRING-PATH-REFERENCE"
    name = "Missing Path Reference in Docstring"
    description = "Checks that docstrings end with the expected relative file path reference"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing path references in docstrings."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Get expected path reference (relative to project root)
        expected_path = str(file_path).replace(str(file_path.parents[3]), "").lstrip("/")
        if expected_path.startswith("src/"):
            expected_path = expected_path[4:]  # Remove src/ prefix

        # Check module docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring and not module_docstring.strip().endswith(expected_path):
            yield self.create_finding(
                message=f"Module docstring missing/incorrect path reference (expected '{expected_path}')",
                file_path=file_path,
                line=1,
                suggestion=f"Add '{expected_path}' at the end of the module docstring for LLM context",
            )

        # Check function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring and not docstring.strip().endswith(expected_path):
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    yield self.create_finding(
                        message=f"{node_type} '{node.name}' docstring missing/incorrect path reference (expected '{expected_path}')",
                        file_path=file_path,
                        line=node.lineno,
                        suggestion=f"Add '{expected_path}' at the end of the docstring for LLM context",
                    )
