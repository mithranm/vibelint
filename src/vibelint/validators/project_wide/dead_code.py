"""
Dead code detection validator with project-wide call graph analysis.

Identifies unused imports, unreferenced functions, duplicate implementations,
and other forms of dead code that can be safely removed.

Performance optimized: builds a single project-wide call graph and uses
graph traversal to identify unreachable code.

vibelint/src/vibelint/validators/dead_code.py
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from ...validators.types import BaseValidator, Finding, Severity
from ...filesystem import find_files_by_extension, find_project_root

__all__ = ["DeadCodeValidator"]


class CallGraph:
    """Project-wide call graph for efficient dead code detection."""

    def __init__(self):
        self.definitions: Dict[str, Tuple[Path, int]] = {}  # name -> (file, line)
        self.calls: Dict[str, Set[str]] = defaultdict(set)  # caller -> set of callees
        self.exports: Dict[str, Set[str]] = {}  # file -> set of exported names
        self.imports: Dict[str, Set[str]] = defaultdict(set)  # file -> set of imports
        self.file_modules: Dict[Path, str] = {}  # file -> module name

    def add_definition(self, name: str, file_path: Path, line: int):
        """Record a function/class definition."""
        self.definitions[name] = (file_path, line)

    def add_call(self, caller: str, callee: str):
        """Record a function call."""
        self.calls[caller].add(callee)

    def add_export(self, file_path: Path, name: str):
        """Record an exported name from __all__."""
        file_key = str(file_path)
        if file_key not in self.exports:
            self.exports[file_key] = set()
        self.exports[file_key].add(name)

    def add_import(self, file_path: Path, name: str):
        """Record an imported name."""
        self.imports[str(file_path)].add(name)

    def is_exported(self, name: str, file_path: Path) -> bool:
        """Check if name is exported from file."""
        return name in self.exports.get(str(file_path), set())

    def get_reachable_from_exports(self) -> Set[str]:
        """Get all names reachable from exported functions."""
        reachable = set()
        queue = []

        # Start with all exported names
        for exports in self.exports.values():
            queue.extend(exports)
            reachable.update(exports)

        # BFS traversal of call graph
        while queue:
            current = queue.pop(0)
            for callee in self.calls.get(current, []):
                if callee not in reachable:
                    reachable.add(callee)
                    queue.append(callee)

        return reachable


class DeadCodeValidator(BaseValidator):
    """Detects various forms of dead code using project-wide call graph analysis."""

    rule_id = "DEAD-CODE-FOUND"
    name = "Dead Code Detector"
    description = "Identifies unused imports, unreferenced functions, and other dead code"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        self._call_graph: CallGraph | None = None
        self._project_root: Path | None = None
        self._analyzed_files: Set[Path] = set()

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze file for dead code patterns."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Build project context on first file
        if self._project_root is None:
            self._project_root = find_project_root(file_path) or file_path.parent

        # Build call graph if this is the first analysis
        if self._call_graph is None:
            self._call_graph = self._build_call_graph(self._project_root)

        # Mark this file as analyzed
        self._analyzed_files.add(file_path)

        # Analyze the file using the call graph
        yield from self._check_unused_imports(file_path, tree, content)
        yield from self._check_unreferenced_definitions(file_path, tree)
        yield from self._check_duplicate_patterns(file_path, content)
        yield from self._check_legacy_patterns(file_path, content)

    def _build_call_graph(self, project_root: Path) -> CallGraph:
        """Build project-wide call graph for all Python files."""
        graph = CallGraph()
        exclude_patterns = ["*/__pycache__/*", "*/.pytest_cache/*", "*/build/*", "*/dist/*"]
        project_files = find_files_by_extension(
            project_root, extension=".py", exclude_globs=exclude_patterns
        )

        for py_file in project_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                # Extract module name
                module_name = self._get_module_name(py_file, project_root)
                graph.file_modules[py_file] = module_name

                # Extract definitions, calls, and exports
                self._extract_from_ast(py_file, tree, graph)

            except (UnicodeDecodeError, SyntaxError):
                continue

        return graph

    def _extract_from_ast(self, file_path: Path, tree: ast.AST, graph: CallGraph):
        """Extract definitions, calls, and exports from an AST."""
        # Extract __all__ exports
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    graph.add_export(file_path, elt.value)

        # Extract function/class definitions and their internal calls
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                name = node.name
                graph.add_definition(name, file_path, node.lineno)

                # Extract calls within this function/class
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name):
                            graph.add_call(name, inner_node.func.id)
                        elif isinstance(inner_node.func, ast.Attribute):
                            # Record method calls
                            graph.add_call(name, inner_node.func.attr)

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
                # Check if exported
                if self._call_graph.is_exported(name, file_path):
                    continue

                # Check for dynamic string references
                if self._scan_string_references(content, name):
                    continue

                yield self.create_finding(
                    message=f"Imported '{name}' is never used",
                    file_path=file_path,
                    line=line_num,
                    suggestion=f"Remove unused import: {name}",
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

        # Collect all references within this file
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced_names.add(node.attr)

        # Get reachable names from call graph
        reachable_names = self._call_graph.get_reachable_from_exports()

        # Find unreferenced definitions
        for name, line_num in defined_names.items():
            # Skip if used locally
            if name in referenced_names:
                continue

            # Skip if exported
            if self._call_graph.is_exported(name, file_path):
                continue

            # Skip if reachable from any exported function
            if name in reachable_names:
                continue

            # Check for dynamic string references
            content = file_path.read_text(encoding="utf-8")
            if self._scan_string_references(content, name):
                continue

            yield self.create_finding(
                message=f"Function/class '{name}' is defined but never referenced",
                file_path=file_path,
                line=line_num,
                suggestion="Consider removing unused definition or adding to __all__",
            )

    def _scan_string_references(self, content: str, name: str) -> bool:
        """Check if name is referenced in strings (getattr, importlib, etc.)."""
        patterns = [
            rf"getattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"hasattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"importlib\.import_module\(['\"].*{re.escape(name)}.*['\"]",
            rf"__import__\(['\"].*{re.escape(name)}.*['\"]",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    def _get_module_name(self, file_path: Path, project_root: Path) -> str:
        """Convert file path to Python module name."""
        try:
            rel_path = file_path.relative_to(project_root)
            # Handle src layout
            if rel_path.parts[0] == "src" and len(rel_path.parts) > 1:
                rel_path = Path(*rel_path.parts[1:])

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

            # Check for manual console instantiation (except in ui.py which creates the shared instance)
            if "= Console()" in stripped and file_path.name not in ["utils.py", "ui.py"]:
                yield self.create_finding(
                    message="Manual Console instantiation - use shared utils instead",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with: from vibelint.ui import console",
                )
