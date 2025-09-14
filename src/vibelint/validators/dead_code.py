"""
Dead code detection validator.

Identifies unused imports, unreferenced functions, duplicate implementations,
and other forms of dead code that can be safely removed.

vibelint/validators/dead_code.py
"""

import ast
from pathlib import Path
from typing import Dict, Iterator, Set

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["DeadCodeValidator"]


class DeadCodeValidator(BaseValidator):
    """Detects various forms of dead code."""

    rule_id = "DEAD-CODE-FOUND"
    name = "Dead Code Detector"
    description = "Identifies unused imports, unreferenced functions, and other dead code"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Analyze file for dead code patterns."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Analyze the AST for various dead code patterns
        yield from self._check_unused_imports(file_path, tree, content)
        yield from self._check_unreferenced_definitions(file_path, tree)
        yield from self._check_duplicate_patterns(file_path, content)
        yield from self._check_legacy_patterns(file_path, content)

    def _check_unused_imports(
        self, file_path: Path, tree: ast.AST, content: str
    ) -> Iterator[Finding]:
        """Check for imported names that are never used."""
        imported_names: Dict[str, int] = {}  # name -> line number
        used_names: Set[str] = set()

        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names[name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue  # Skip wildcard imports
                    name = alias.asname if alias.asname else alias.name
                    imported_names[name] = node.lineno

        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For attribute access like `os.path`, record `os` as used
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused imports
        for name, line_num in imported_names.items():
            if name not in used_names:
                # Check if it might be used in strings (common in dynamic imports)
                if f"'{name}'" not in content and f'"{name}"' not in content:
                    yield self.create_finding(
                        message=f"Imported '{name}' is never used",
                        file_path=file_path,
                        line=line_num,
                        suggestion=f"Remove unused import: {name}",
                        severity=Severity.INFO,
                    )

    def _check_unreferenced_definitions(self, file_path: Path, tree: ast.AST) -> Iterator[Finding]:
        """Check for functions/classes that are defined but never referenced."""
        # Skip this check for __init__.py files and test files
        if file_path.name == "__init__.py" or "test" in file_path.name.lower():
            return

        defined_names: Dict[str, int] = {}  # name -> line number
        referenced_names: Set[str] = set()

        # Collect all function and class definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # Skip private/dunder methods and main blocks
                if not node.name.startswith("_") and node.name != "main":
                    defined_names[node.name] = node.lineno

        # Collect all references
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced_names.add(node.attr)

        # Find unreferenced definitions
        for name, line_num in defined_names.items():
            if name not in referenced_names:
                yield self.create_finding(
                    message=f"Function/class '{name}' is defined but never referenced",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider removing unused definition or adding to __all__",
                    severity=Severity.INFO,
                )

    def _check_duplicate_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for duplicate code patterns that suggest redundancy."""
        lines = content.splitlines()

        # Check for duplicate validation result classes
        validation_classes = []
        for line_num, line in enumerate(lines, 1):
            if "ValidationResult" in line and "class " in line:
                validation_classes.append((line_num, line.strip()))

        if len(validation_classes) > 1:
            for line_num, class_def in validation_classes:
                yield self.create_finding(
                    message="Validation result class found - may be duplicating plugin system",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider using plugin system's Finding class instead",
                    severity=Severity.INFO,
                )

        # Check for duplicate validation functions
        validation_functions = []
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith("def validate_") and not line.strip().startswith(
                "def validate("
            ):
                validation_functions.append((line_num, line.strip()))

        if len(validation_functions) > 0:
            for line_num, func_def in validation_functions:
                yield self.create_finding(
                    message="Legacy validation function found - may duplicate BaseValidator",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider migrating to BaseValidator plugin system",
                    severity=Severity.INFO,
                )

    def _check_legacy_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for legacy code patterns that might be dead."""
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Legacy pattern detection removed - no legacy patterns exist to detect

            # Check for manual console instantiation (except in console_utils.py which creates the shared instance)
            if "= Console()" in stripped and not file_path.name == "console_utils.py":
                yield self.create_finding(
                    message="Manual Console instantiation - use shared console_utils instead",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with: from .console_utils import console",
                    severity=Severity.INFO,
                )
