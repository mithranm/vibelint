"""
Dead code detection validator.

Identifies unused imports, unreferenced functions, duplicate implementations,
and other forms of dead code that can be safely removed.

vibelint/src/vibelint/validators/dead_code.py
"""

import ast
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set

from ..plugin_system import BaseValidator, Finding, Severity

__all__ = ["DeadCodeValidator"]


class DeadCodeValidator(BaseValidator):
    """Detects various forms of dead code."""

    rule_id = "DEAD-CODE-FOUND"
    name = "Dead Code Detector"
    description = "Identifies unused imports, unreferenced functions, and other dead code"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        self._project_files_cache = None
        self._all_exports_cache = None

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze file for dead code patterns."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Build project context for dynamic analysis
        project_root = self._get_project_root(file_path)

        # Analyze the AST for various dead code patterns
        yield from self._check_unused_imports_dynamic(file_path, tree, content, project_root)
        yield from self._check_unreferenced_definitions_dynamic(file_path, tree, project_root)
        yield from self._check_duplicate_patterns(file_path, content)
        yield from self._check_legacy_patterns(file_path, content)

    def _get_project_root(self, file_path: Path) -> Path:
        """Find project root by looking for pyproject.toml or setup.py."""
        current = file_path.parent if file_path.is_file() else file_path
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return file_path.parent

    def _get_project_files(self, project_root: Path) -> List[Path]:
        """Get all Python files in the project."""
        if self._project_files_cache is None:
            self._project_files_cache = []
            for py_file in project_root.rglob("*.py"):
                if not any(part.startswith(".") for part in py_file.parts):
                    self._project_files_cache.append(py_file)
        return self._project_files_cache

    def _get_all_exports(self, project_root: Path) -> Dict[str, Set[str]]:
        """Get all __all__ exports across the project."""
        if self._all_exports_cache is None:
            self._all_exports_cache = {}
            for py_file in self._get_project_files(project_root):
                try:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content)
                    exports = self._extract_all_exports(tree)
                    if exports:
                        self._all_exports_cache[str(py_file)] = exports
                except (UnicodeDecodeError, SyntaxError):
                    continue
        return self._all_exports_cache

    def _extract_all_exports(self, tree: ast.AST) -> Set[str]:
        """Extract names from __all__ assignments."""
        exports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports.add(elt.value)
        return exports

    def _scan_string_references(self, content: str, name: str) -> bool:
        """Check if name is referenced in strings (getattr, importlib, etc.)."""
        patterns = [
            rf"getattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"hasattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"importlib\.import_module\(['\"].*{re.escape(name)}.*['\"]",
            rf"__import__\(['\"].*{re.escape(name)}.*['\"]",
            rf"['\"]({re.escape(name)})['\"]",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    def _is_used_in_tests(self, name: str, project_root: Path) -> bool:
        """Check if name is used in test files."""
        test_files = [f for f in self._get_project_files(project_root) if "test" in str(f).lower()]
        for test_file in test_files:
            try:
                content = test_file.read_text(encoding="utf-8")
                if name in content:
                    return True
            except UnicodeDecodeError:
                continue
        return False

    def _check_unused_imports_dynamic(
        self, file_path: Path, tree: ast.AST, content: str, project_root: Path
    ) -> Iterator[Finding]:
        """Check for imported names that are never used with dynamic analysis."""
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

        # Get __all__ exports for this file
        all_exports = self._extract_all_exports(tree)

        # Find unused imports with dynamic checks
        for name, line_num in imported_names.items():
            if name not in used_names:
                # Dynamic analysis checks
                is_exported = name in all_exports
                is_string_referenced = self._scan_string_references(content, name)
                is_test_used = self._is_used_in_tests(name, project_root)

                # Skip if used dynamically
                if is_exported or is_string_referenced or is_test_used:
                    continue

                yield self.create_finding(
                    message=f"Imported '{name}' is never used",
                    file_path=file_path,
                    line=line_num,
                    suggestion=f"Remove unused import: {name}",
                )

    def _check_unreferenced_definitions_dynamic(self, file_path: Path, tree: ast.AST, project_root: Path) -> Iterator[Finding]:
        """Check for functions/classes that are defined but never referenced with dynamic analysis."""
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

        # Get __all__ exports for this file
        all_exports = self._extract_all_exports(tree)

        # Check cross-file usage
        def _is_used_in_other_files(name: str) -> bool:
            module_name = self._get_module_name(file_path, project_root)
            if not module_name:
                return False

            for py_file in self._get_project_files(project_root):
                if py_file == file_path:
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    # Check for direct imports
                    if f"from {module_name} import" in content and name in content:
                        return True
                    # Check for module imports
                    if f"import {module_name}" in content and f"{module_name}.{name}" in content:
                        return True
                except UnicodeDecodeError:
                    continue
            return False

        # Find unreferenced definitions with dynamic checks
        for name, line_num in defined_names.items():
            if name not in referenced_names:
                # Dynamic analysis checks
                is_exported = name in all_exports
                is_string_referenced = self._scan_string_references(file_path.read_text(encoding="utf-8"), name)
                is_test_used = self._is_used_in_tests(name, project_root)
                is_cross_file_used = _is_used_in_other_files(name)

                # Skip if used dynamically
                if is_exported or is_string_referenced or is_test_used or is_cross_file_used:
                    continue

                yield self.create_finding(
                    message=f"Function/class '{name}' is defined but never referenced",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider removing unused definition or adding to __all__",
                )

    def _get_module_name(self, file_path: Path, project_root: Path) -> str:
        """Convert file path to Python module name."""
        try:
            rel_path = file_path.relative_to(project_root)
            if rel_path.name == "__init__.py":
                module_parts = rel_path.parent.parts
            else:
                module_parts = rel_path.with_suffix("").parts
            return ".".join(module_parts)
        except ValueError:
            return ""

    def _check_duplicate_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for duplicate code patterns that suggest redundancy."""
        lines = content.splitlines()

        # Check for duplicate validation result classes
        validation_classes = []
        for line_num, line in enumerate(lines, 1):
            if "ValidationResult" in line and "class " in line:
                validation_classes.append((line_num, line.strip()))

        if len(validation_classes) > 1:
            for line_num, _ in validation_classes:
                yield self.create_finding(
                    message="Validation result class found - may be duplicating plugin system",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider using plugin system's Finding class instead",
                )

        # Check for duplicate validation functions
        validation_functions = []
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith("def validate_") and not line.strip().startswith(
                "def validate("
            ):
                validation_functions.append((line_num, line.strip()))

        if len(validation_functions) > 0:
            for line_num, _ in validation_functions:
                yield self.create_finding(
                    message="Legacy validation function found - may duplicate BaseValidator",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider migrating to BaseValidator plugin system",
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
                )
