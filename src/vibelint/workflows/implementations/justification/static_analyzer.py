"""
Static analysis module for justification workflow.

Handles AST parsing, dependency analysis, structural analysis without LLM calls.
Provides deterministic analysis that can be cached and reused.
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class StaticAnalyzer:
    """Performs static analysis on Python code without external dependencies."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def analyze_file_structure(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze file structure using AST parsing."""
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return {
                "error": f"Syntax error: {e}",
                "functions": [],
                "classes": [],
                "imports": [],
                "module_docstring": None
            }

        # Extract components
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        imports = self._extract_imports(tree)
        module_docstring = self._extract_module_docstring(tree)

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "module_docstring": module_docstring,
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
            "has_main": self._has_main_block(content),
            "is_test_file": self._is_test_file(file_path, content),
            "is_init_file": file_path.name == "__init__.py"
        }

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions with metadata."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "is_private": node.name.startswith('_'),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorator_count": len(node.decorator_list)
                }
                functions.append(func_info)
        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions with metadata."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "bases": [self._get_node_name(base) for base in node.bases],
                    "docstring": ast.get_docstring(node),
                    "methods": [m.name for m in methods],
                    "method_count": len(methods),
                    "is_private": node.name.startswith('_'),
                    "decorator_count": len(node.decorator_list)
                }
                classes.append(class_info)
        return classes

    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements with metadata."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line_number": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "level": node.level,
                        "line_number": node.lineno
                    })
        return imports

    def _extract_module_docstring(self, tree: ast.AST) -> Optional[str]:
        """Extract module-level docstring."""
        return ast.get_docstring(tree)

    def _get_node_name(self, node: ast.AST) -> str:
        """Get name from AST node safely."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def _has_main_block(self, content: str) -> bool:
        """Check if file has main block."""
        return 'if __name__ == "__main__"' in content

    def _is_test_file(self, file_path: Path, content: str) -> bool:
        """Determine if this is a test file."""
        return (
            file_path.name.startswith('test_') or
            file_path.name.endswith('_test.py') or
            'test' in file_path.parts or
            'def test_' in content
        )

    def build_dependency_graph(self, file_analyses: Dict[Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Build dependency graph from analyzed files."""
        dependencies = {}

        for file_path, analysis in file_analyses.items():
            file_deps = []

            # Process imports to build dependencies
            for import_info in analysis.get("imports", []):
                if import_info["type"] == "from":
                    module = import_info["module"]
                    if module and not module.startswith('.'):  # Skip relative imports for now
                        file_deps.append(module)
                elif import_info["type"] == "import":
                    module = import_info["module"]
                    file_deps.append(module)

            dependencies[str(file_path)] = {
                "imports": file_deps,
                "internal_deps": [],  # Will be filled by resolve_internal_dependencies
                "external_deps": []   # Will be filled by categorize_dependencies
            }

        return dependencies

    def detect_circular_imports(self, dependency_graph: Dict[str, Any]) -> List[List[str]]:
        """Detect circular import dependencies."""
        circles = []
        visited = set()

        def find_cycles_from_node(node: str, path: List[str], visited_in_path: Set[str]) -> None:
            if node in visited_in_path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) > 2:  # Ignore self-references
                    circles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            visited_in_path.add(node)

            # Follow dependencies
            node_data = dependency_graph.get(node) if node in dependency_graph else None
            deps = node_data.get("internal_deps") if node_data and "internal_deps" in node_data else []
            for dep in deps:
                find_cycles_from_node(dep, path + [node], visited_in_path.copy())

        for node in dependency_graph:
            if node not in visited:
                find_cycles_from_node(node, [], set())

        return circles

    def calculate_complexity_metrics(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Calculate basic complexity metrics."""
        metrics = {
            "total_functions": analysis.get("function_count", 0),
            "total_classes": analysis.get("class_count", 0),
            "total_imports": analysis.get("import_count", 0),
            "private_functions": 0,
            "public_functions": 0,
            "methods_per_class": 0
        }

        # Count private vs public functions
        for func in analysis.get("functions", []):
            if func.get("is_private", False):
                metrics["private_functions"] += 1
            else:
                metrics["public_functions"] += 1

        # Calculate average methods per class
        classes = analysis.get("classes", [])
        if classes:
            total_methods = sum(cls.get("method_count", 0) for cls in classes)
            metrics["methods_per_class"] = total_methods // len(classes)

        return metrics

    def find_structural_issues(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find basic structural issues without LLM."""
        issues = []

        # Large function count
        if analysis.get("function_count", 0) > 20:
            issues.append({
                "type": "complexity",
                "issue": f"High function count: {analysis['function_count']}",
                "suggestion": "Consider breaking into multiple modules"
            })

        # No docstring
        if not analysis.get("module_docstring"):
            issues.append({
                "type": "documentation",
                "issue": "Missing module docstring",
                "suggestion": "Add module-level documentation"
            })

        # Many imports
        if analysis.get("import_count", 0) > 15:
            issues.append({
                "type": "coupling",
                "issue": f"High import count: {analysis['import_count']}",
                "suggestion": "Review dependencies, consider refactoring"
            })

        return issues

    def detect_misplaced_files(self, file_analyses: Dict[Path, Dict[str, Any]],
                              project_root: Path) -> List[Dict[str, str]]:
        """Detect files that are placed in wrong directories based on their content and imports."""
        misplaced_files = []

        for file_path, analysis in file_analyses.items():
            relative_path = file_path.relative_to(project_root)
            path_parts = relative_path.parts
            file_name = file_path.name

            # Check for common misplacement patterns
            misplacement = self._analyze_file_placement(file_path, analysis, path_parts)
            if misplacement:
                misplaced_files.append(misplacement)

        return misplaced_files

    def _analyze_file_placement(self, file_path: Path, analysis: Dict[str, Any],
                               path_parts: tuple) -> Optional[Dict[str, str]]:
        """Analyze if a single file is properly placed."""
        file_name = file_path.name
        imports = analysis.get("imports", [])
        functions = analysis.get("functions", [])
        classes = analysis.get("classes", [])

        # Script files in wrong locations
        if (analysis.get("has_main") or "if __name__" in str(analysis)) and "scripts" not in path_parts and "tools" in path_parts:
            # Script in tools/ should be in scripts/ or have proper entry point
            return {
                "file": str(file_path),
                "issue": "Executable script in tools/ directory",
                "suggestion": "Move to scripts/ or create console script entry point",
                "severity": "medium"
            }

        # Package-specific utilities in wrong location
        if any("vibelint" in imp.get("module", "") for imp in imports):
            # File imports vibelint but is outside src/vibelint/
            if "src" not in path_parts or "vibelint" not in path_parts:
                return {
                    "file": str(file_path),
                    "issue": "vibelint-specific code outside main package",
                    "suggestion": "Move to src/vibelint/ or make generic",
                    "severity": "high"
                }

        # CLI-related files outside cli/
        if any("cli" in func.get("name", "").lower() or "command" in func.get("name", "").lower() for func in functions):
            if "cli" not in path_parts and not analysis.get("is_test_file"):
                return {
                    "file": str(file_path),
                    "issue": "CLI functionality outside cli/ directory",
                    "suggestion": "Move to src/vibelint/cli/ or extract CLI parts",
                    "severity": "medium"
                }

        # Test files in wrong location
        if analysis.get("is_test_file") and "test" not in path_parts:
            return {
                "file": str(file_path),
                "issue": "Test file outside tests/ directory",
                "suggestion": "Move to tests/ directory",
                "severity": "medium"
            }

        # Workflow files outside workflows/
        if any("workflow" in cls.get("name", "").lower() for cls in classes):
            if "workflow" not in path_parts and not analysis.get("is_test_file"):
                return {
                    "file": str(file_path),
                    "issue": "Workflow implementation outside workflows/ directory",
                    "suggestion": "Move to src/vibelint/workflows/",
                    "severity": "medium"
                }

        # Validator files outside validators/
        if any("validator" in cls.get("name", "").lower() or "validate" in func.get("name", "").lower() for cls in classes for func in functions):
            if "validator" not in path_parts and not analysis.get("is_test_file"):
                return {
                    "file": str(file_path),
                    "issue": "Validation logic outside validators/ directory",
                    "suggestion": "Move to src/vibelint/validators/",
                    "severity": "medium"
                }

        return None