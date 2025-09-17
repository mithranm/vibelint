"""
Module cohesion validator for detecting scattered related modules.

Identifies when related modules should be grouped together in subpackages
based on naming patterns, import relationships, and functional cohesion.

vibelint/src/vibelint/validators/module_cohesion.py
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set

from ..plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["ModuleCohesionValidator"]


class ModuleCohesionValidator(BaseValidator):
    """Detects module organization issues and unjustified file existence."""

    rule_id = "MODULE-COHESION"
    name = "Module Cohesion & File Justification Analyzer"
    description = (
        "Identifies scattered related modules and files without clear purpose justification"
    )
    default_severity = Severity.INFO

    def __init__(self, severity=None, config=None):
        super().__init__(severity=severity, config=config)
        # Common patterns that suggest related modules
        self.cohesion_patterns = [
            # Prefixed modules (llm_, api_, db_, etc.)
            r"^([a-z]+)_[a-z_]+\.py$",
            # Service/handler patterns
            r"^([a-z]+)_(service|handler|manager|client|adapter)\.py$",
            # Model/schema patterns
            r"^([a-z]+)_(model|schema|entity|dto)\.py$",
            # Test patterns
            r"^test_([a-z]+)_.*\.py$",
            # Utils patterns
            r"^([a-z]+)_(utils|helpers|tools)\.py$",
        ]

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze project structure for module cohesion issues."""
        # Only analyze from project root to avoid duplicates
        project_root = self._get_project_root(file_path)
        if file_path.parent != project_root / "src" / "vibelint":
            return

        # Get all Python files in the project
        src_dir = project_root / "src" / "vibelint"
        python_files = list(src_dir.glob("*.py"))

        if len(python_files) < 3:  # Need multiple files to detect patterns
            return

        # Group files by naming patterns
        pattern_groups = self._group_files_by_patterns(python_files)

        # Check for scattered modules that should be grouped
        for pattern, files in pattern_groups.items():
            if len(files) >= 2:  # 2+ files with same prefix suggest a module group
                yield from self._suggest_module_grouping(pattern, files, src_dir)

        # Check for functional cohesion based on imports
        yield from self._analyze_import_cohesion(python_files, src_dir)

        # Check for unjustified files
        yield from self._check_file_justification(project_root)

    def _get_project_root(self, file_path: Path) -> Path:
        """Find project root by looking for pyproject.toml."""
        current = file_path.parent if file_path.is_file() else file_path
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return file_path.parent

    def _group_files_by_patterns(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by common naming patterns."""
        import re

        groups = defaultdict(list)

        for file_path in files:
            filename = file_path.name

            # Skip special files
            if filename in ["__init__.py", "cli.py", "main.py"]:
                continue

            # Check each pattern
            for pattern in self.cohesion_patterns:
                match = re.match(pattern, filename)
                if match:
                    prefix = match.group(1)
                    groups[prefix].append(file_path)
                    break

        return groups

    def _suggest_module_grouping(
        self, prefix: str, files: List[Path], src_dir: Path
    ) -> Iterator[Finding]:
        """Suggest grouping related files into a submodule."""
        if len(files) < 2:
            return

        # Check if they're already in a submodule
        if any(len(f.relative_to(src_dir).parts) > 1 for f in files):
            return  # Already organized

        file_names = [f.name for f in files]

        yield Finding(
            rule_id=self.rule_id,
            message=f"Related modules with '{prefix}_' prefix should be grouped: {', '.join(file_names)}",
            file_path=files[0],  # Report on first file
            line=1,
            severity=self.default_severity,
            suggestion=f"Create 'src/vibelint/{prefix}/' subpackage and move related modules:\n"
            f"  mkdir src/vibelint/{prefix}/\n"
            f"  mv {' '.join(file_names)} src/vibelint/{prefix}/\n"
            f"  # Rename files to remove prefix: {prefix}_manager.py -> manager.py",
        )

    def _analyze_import_cohesion(self, files: List[Path], src_dir: Path) -> Iterator[Finding]:
        """Analyze import patterns to suggest functional grouping."""
        # Build import graph
        import_graph = defaultdict(set)

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)

                imports = self._extract_local_imports(tree, src_dir)
                module_name = file_path.stem
                import_graph[module_name].update(imports)

            except (UnicodeDecodeError, SyntaxError):
                continue

        # Find tightly coupled modules (import each other frequently)
        coupled_groups = self._find_coupled_modules(import_graph)

        for group in coupled_groups:
            if len(group) >= 3:  # Suggest grouping for 3+ tightly coupled modules
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Tightly coupled modules should be grouped: {', '.join(group)}",
                    file_path=src_dir / f"{list(group)[0]}.py",
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Consider grouping these functionally related modules into a subpackage",
                )

    def _extract_local_imports(self, tree: ast.AST, src_dir: Path) -> Set[str]:
        """Extract imports that reference local modules."""
        local_imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("vibelint"):
                    # Extract module name
                    parts = node.module.split(".")
                    if len(parts) >= 2:
                        local_imports.add(parts[-1])

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("vibelint"):
                        parts = alias.name.split(".")
                        if len(parts) >= 2:
                            local_imports.add(parts[-1])

        return local_imports

    def _find_coupled_modules(self, import_graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find groups of modules that frequently import each other."""
        # Simple clustering based on mutual imports
        coupled_groups = []
        processed = set()

        for module, imports in import_graph.items():
            if module in processed:
                continue

            # Find modules that import this one and vice versa
            mutual_imports = set()
            for imported in imports:
                if imported in import_graph and module in import_graph[imported]:
                    mutual_imports.add(imported)

            if mutual_imports:
                group = {module} | mutual_imports
                coupled_groups.append(group)
                processed.update(group)

        return coupled_groups

    def _check_file_justification(self, project_root: Path) -> Iterator[Finding]:
        """Check that every file has clear justification for existence."""
        # Files that are automatically justified
        auto_justified = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "README.md",
            "README.rst",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            ".gitignore",
            ".gitattributes",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
            "tox.ini",
            ".pre-commit-config.yaml",
            "__init__.py",
            "conftest.py",
            "pytest.ini",
        }

        # Check all files in project
        for file_path in project_root.rglob("*"):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue

            # Skip auto-justified files
            if file_path.name in auto_justified:
                continue

            # Skip files in build/cache directories
            if any(
                part
                in [
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "build",
                    "dist",
                    ".tox",
                    ".mypy_cache",
                    "node_modules",
                ]
                for part in file_path.parts
            ):
                continue

            # Check file justification based on type
            if file_path.suffix == ".py":
                yield from self._check_python_file_justification(file_path)
            elif file_path.suffix in [".md", ".rst", ".txt"]:
                yield from self._check_documentation_justification(file_path)
            elif file_path.suffix in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]:
                yield from self._check_config_file_justification(file_path)
            elif file_path.suffix in [".sh", ".bat", ".ps1"]:
                yield from self._check_script_justification(file_path)
            else:
                # Unknown file type - requires explicit justification
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Unknown file type without clear justification: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comment/documentation explaining file purpose or remove if unnecessary",
                )

    def _check_python_file_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that Python files have module docstrings."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Check for module docstring
            has_module_docstring = False
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
            ):
                if (
                    isinstance(tree.body[0].value.value, str)
                    and len(tree.body[0].value.value.strip()) > 10
                ):
                    has_module_docstring = True

            if not has_module_docstring:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Python file lacks module docstring explaining its purpose: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.WARN,
                    suggestion='Add module docstring: """Brief description of module purpose."""',
                )

        except (UnicodeDecodeError, SyntaxError):
            yield Finding(
                rule_id=self.rule_id,
                message=f"Python file has syntax errors or encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix syntax errors or encoding issues, or remove if unused",
            )

    def _check_documentation_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that documentation files have meaningful content."""
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            # Check for minimal content
            if len(content) < 50:  # Very short docs
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Documentation file is too short to be useful: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add meaningful content or remove if unnecessary",
                )

            # Check for placeholder/template content
            placeholder_indicators = ["TODO", "FIXME", "PLACEHOLDER", "Lorem ipsum", "Example text"]
            if any(indicator in content for indicator in placeholder_indicators):
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Documentation file contains placeholder content: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Replace placeholder content with actual documentation or remove file",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Documentation file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )

    def _check_config_file_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that config files have clear purpose."""
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            # Check for empty config files
            if len(content) < 10:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Config file is nearly empty: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add configuration content or remove if unnecessary",
                )

            # Check for comments explaining purpose
            has_explanatory_comments = any(
                line.strip().startswith("#") and len(line.strip()) > 10
                for line in content.splitlines()[:5]  # Check first 5 lines
            )

            if not has_explanatory_comments and file_path.suffix in [".json", ".yaml", ".yml"]:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Config file lacks explanatory comments: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comments explaining configuration purpose and usage",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Config file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )

    def _check_script_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that script files have clear purpose."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for shebang and comments
            lines = content.splitlines()

            # Look for explanatory comments in first 10 lines
            has_explanation = any(
                line.strip().startswith("#") and len(line.strip()) > 20 for line in lines[:10]
            )

            if not has_explanation:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Script file lacks explanatory comments: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comments explaining script purpose, usage, and requirements",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Script file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )
