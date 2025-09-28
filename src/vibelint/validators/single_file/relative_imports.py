"""
Validator for detecting and suggesting fixes for relative imports.

Relative imports can cause issues in larger codebases and make modules less portable.
This validator detects relative imports and suggests absolute import alternatives.
"""

import ast
import logging
from pathlib import Path
from typing import Iterator, List, Optional

from vibelint.plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class RelativeImportValidator(BaseValidator):
    """Validates and suggests fixes for relative imports."""

    rule_id = "RELATIVE-IMPORTS"
    description = "Detect relative imports and suggest absolute alternatives"

    def __init__(self, config=None, severity=None):
        super().__init__(config)
        self.config = config or {}
        self.severity = severity

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate Python file for relative imports.

        Args:
            file_path: Path to the Python file
            content: File content as string
            config: Optional configuration

        Yields:
            Finding objects for relative imports found
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            # If file doesn't parse, skip validation
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        # Extract package structure information
        package_info = self._analyze_package_structure(file_path)

        # Find all import nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                findings = self._check_import_node(node, file_path, package_info)
                yield from findings

    def _analyze_package_structure(self, file_path: Path) -> dict:
        """Analyze the package structure to understand how to convert relative imports."""
        package_info = {
            "file_path": file_path,
            "package_parts": [],
            "is_package": False,
            "project_root": None
        }

        # Find project root (look for pyproject.toml, setup.py, or .git)
        current = file_path.parent
        while current != current.parent:
            if any((current / name).exists() for name in ["pyproject.toml", "setup.py", ".git"]):
                package_info["project_root"] = current
                break
            current = current.parent

        if not package_info["project_root"]:
            package_info["project_root"] = file_path.parent

        # Determine package path relative to project root
        try:
            relative_path = file_path.relative_to(package_info["project_root"])
            package_info["package_parts"] = list(relative_path.parent.parts)

            # Remove common non-package directories
            if package_info["package_parts"] and package_info["package_parts"][0] in ["src", "lib"]:
                package_info["package_parts"] = package_info["package_parts"][1:]

            package_info["is_package"] = file_path.name == "__init__.py"
        except ValueError:
            # File is not under project root
            package_info["package_parts"] = []

        return package_info

    def _check_import_node(self, node: ast.AST, file_path: Path, package_info: dict) -> List[Finding]:
        """Check an import node for relative import issues."""
        findings = []

        if isinstance(node, ast.ImportFrom):
            # Check for relative imports (those starting with . or ..)
            if node.level > 0:  # Relative import detected
                absolute_suggestion = self._suggest_absolute_import(node, package_info)

                if absolute_suggestion:
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        message=f"Relative import detected: {'.' * node.level}{node.module or ''}",
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity=Severity.WARN,
                        suggestion=f"Replace with absolute import: {absolute_suggestion}"
                    ))

        return findings

    def _suggest_absolute_import(self, node: ast.ImportFrom, package_info: dict) -> Optional[str]:
        """Suggest an absolute import to replace the relative import."""
        if not package_info["package_parts"]:
            return None

        # Calculate the absolute module path
        current_package = package_info["package_parts"].copy()

        # Handle different levels of relative imports
        if node.level == 1:  # from .module import something
            # Same package level
            target_package = current_package
        elif node.level > 1:  # from ..module import something
            # Go up the package hierarchy
            levels_up = node.level - 1
            if levels_up >= len(current_package):
                return None  # Can't go up that many levels
            target_package = current_package[:-levels_up] if levels_up > 0 else current_package
        else:
            return None

        # Build the absolute import
        if node.module:
            absolute_module = ".".join(target_package + [node.module])
        else:
            absolute_module = ".".join(target_package)

        # Format the import statement
        if node.names:
            if len(node.names) == 1 and node.names[0].name == "*":
                return f"from {absolute_module} import *"
            else:
                imports = []
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"{alias.name} as {alias.asname}")
                    else:
                        imports.append(alias.name)
                return f"from {absolute_module} import {', '.join(imports)}"
        else:
            return f"import {absolute_module}"

    def can_fix(self) -> bool:
        """Returns True if this validator can automatically fix issues."""
        return True

    def fix_finding(self, file_path: Path, content: str, finding: Finding) -> str:
        """
        Automatically fix a relative import finding.

        Args:
            file_path: Path to the file
            content: Current file content
            finding: The finding to fix

        Returns:
            Updated file content with the fix applied
        """
        if "Replace with absolute import:" not in finding.suggestion:
            return content

        # Extract the suggested absolute import
        suggestion_parts = finding.suggestion.split("Replace with absolute import: ", 1)
        if len(suggestion_parts) != 2:
            return content

        absolute_import = suggestion_parts[1]

        # Find and replace the relative import on the specific line
        lines = content.split('\n')
        if finding.line <= len(lines):
            line_idx = finding.line - 1  # Convert to 0-based index
            original_line = lines[line_idx]

            # Try to identify and replace the relative import
            # This is a simplified replacement - in practice, you might want more sophisticated parsing
            if original_line.strip().startswith('from .'):
                # Find the indentation and replace the import
                indent = len(original_line) - len(original_line.lstrip())
                lines[line_idx] = ' ' * indent + absolute_import

                return '\n'.join(lines)

        return content