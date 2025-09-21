"""
Docstring validator using BaseValidator plugin system.

Checks for missing docstrings and proper path references in modules,
classes, and functions.

vibelint/src/vibelint/validators/docstring.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["MissingDocstringValidator", "DocstringPathValidator"]


class MissingDocstringValidator(BaseValidator):
    """Validator for missing docstrings."""

    rule_id = "DOCSTRING-MISSING"
    name = "Missing Docstring Checker"
    description = "Checks for missing docstrings in modules, classes, and functions"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing docstrings based on configuration."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Get docstring configuration
        docstring_config = (config and config.get("docstring", {})) or {}
        require_module = docstring_config.get("require_module_docstrings", True)
        require_class = docstring_config.get("require_class_docstrings", True)
        require_function = docstring_config.get("require_function_docstrings", False)
        include_private = docstring_config.get("include_private_functions", False)

        # Check module docstring
        if require_module and not ast.get_docstring(tree):
            yield self.create_finding(
                message="Module is missing docstring",
                file_path=file_path,
                line=1,
                suggestion="Add a module-level docstring explaining the module's purpose",
            )

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check class docstrings
                if require_class and (include_private or not node.name.startswith("_")):
                    if not ast.get_docstring(node):
                        yield self.create_finding(
                            message=f"Class '{node.name}' is missing docstring",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion=f"Add docstring to {node.name} explaining its purpose",
                        )
            elif isinstance(node, ast.FunctionDef):
                # Check function docstrings
                if require_function and (include_private or not node.name.startswith("_")):
                    if not ast.get_docstring(node):
                        yield self.create_finding(
                            message=f"Function '{node.name}' is missing docstring",
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
        """Check for missing path references in docstrings based on configuration."""
        # Get docstring configuration
        docstring_config = (config and config.get("docstring", {})) or {}
        require_path_references = docstring_config.get("require_path_references", False)

        # Skip validation if path references are not required
        if not require_path_references:
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Get expected path reference based on format configuration
        path_format = docstring_config.get("path_reference_format", "relative")
        expected_path = self._get_expected_path(file_path, path_format)

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

    def _get_expected_path(self, file_path: Path, path_format: str) -> str:
        """Get expected path reference based on format configuration."""
        if path_format == "absolute":
            return str(file_path)
        elif path_format == "module_path":
            # Convert to Python module path (e.g., vibelint.validators.docstring)
            parts = file_path.parts
            if "src" in parts:
                src_idx = parts.index("src")
                module_parts = parts[src_idx + 1 :]
            else:
                module_parts = parts

            # Remove .py extension and convert to module path
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

            return ".".join(module_parts)
        else:  # relative format (default)
            # Get relative path, removing project root and src/ prefix
            relative_path = str(file_path)
            try:
                # Try to find project root by looking for common markers
                current = file_path.parent
                while current.parent != current:
                    if any(
                        (current / marker).exists()
                        for marker in ["pyproject.toml", "setup.py", ".git"]
                    ):
                        relative_path = str(file_path.relative_to(current))
                        break
                    current = current.parent
            except ValueError:
                pass

            # Remove src/ prefix if present
            if relative_path.startswith("src/"):
                relative_path = relative_path[4:]

            return relative_path
