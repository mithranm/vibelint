"""
Multi-level context analyzer for catching organizational violations.

Provides different "zoom levels" of codebase analysis:
1. Tree Level: File organization, naming patterns, structure violations
2. Content Level: Code structure, imports, dependencies
3. Deep Level: Full LLM-powered semantic analysis

vibelint/src/vibelint/context_analyzer.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

__all__ = ["ContextAnalyzer", "TreeViolation", "ContentViolation"]


@dataclass
class TreeViolation:
    """File tree organization violation."""

    violation_type: str
    file_path: Path
    message: str
    suggestion: str
    severity: str = "WARN"


@dataclass
class ContentViolation:
    """File content structure violation."""

    violation_type: str
    file_path: Path
    line: int
    message: str
    suggestion: str
    severity: str = "WARN"


class ContextAnalyzer:
    """Multi-level context analyzer for organizational violations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.file_tree_cache = None

    def analyze_tree_level(self) -> List[TreeViolation]:
        """
        Lightweight analysis of file organization without reading content.

        Catches issues like:
        - Files in wrong locations
        - Missing organization
        - Naming violations
        """
        violations = []

        # Get project file tree
        tree = self._get_file_tree()

        # Check root directory clutter
        violations.extend(self._check_root_clutter(tree))

        # Check for scattered related files
        violations.extend(self._check_scattered_modules(tree))

        # Check directory structure appropriateness
        violations.extend(self._check_directory_structure(tree))

        return violations

    def _get_file_tree(self) -> Dict[str, Any]:
        """Get lightweight file tree structure."""
        if self.file_tree_cache:
            return self.file_tree_cache

        tree = {"files": [], "dirs": {}}

        for path in self.project_root.iterdir():
            if path.is_file():
                tree["files"].append(
                    {
                        "name": path.name,
                        "path": path,
                        "size": path.stat().st_size,
                        "suffix": path.suffix,
                    }
                )
            elif path.is_dir() and not path.name.startswith("."):
                tree["dirs"][path.name] = self._get_dir_tree(path)

        self.file_tree_cache = tree
        return tree

    def _get_dir_tree(self, dir_path: Path) -> Dict[str, Any]:
        """Get tree for subdirectory."""
        tree = {"files": [], "dirs": {}}

        try:
            for path in dir_path.iterdir():
                if path.is_file():
                    tree["files"].append({"name": path.name, "path": path, "suffix": path.suffix})
                elif path.is_dir() and not path.name.startswith("."):
                    tree["dirs"][path.name] = self._get_dir_tree(path)
        except PermissionError:
            pass

        return tree

    def _check_root_clutter(self, tree: Dict[str, Any]) -> List[TreeViolation]:
        """Check for inappropriate files in project root."""
        violations = []

        # Files that belong in project root
        allowed_root_files = {
            "README.md",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
            ".gitignore",
            ".gitattributes",
            "tox.ini",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
        }

        # Files that should be in docs/
        doc_suffixes = {".md", ".rst", ".txt"}
        doc_keywords = {"plan", "design", "spec", "guide", "tutorial", "example"}

        # Files that should be in scripts/
        script_suffixes = {".sh", ".bat", ".ps1", ".py"}
        script_keywords = {"script", "build", "deploy", "test", "setup", "install"}

        for file_info in tree["files"]:
            name = file_info["name"].lower()
            path = file_info["path"]

            # Skip allowed files
            if file_info["name"] in allowed_root_files:
                continue

            # Check for documentation files
            if file_info["suffix"] in doc_suffixes and any(
                keyword in name for keyword in doc_keywords
            ):

                violations.append(
                    TreeViolation(
                        violation_type="ROOT_CLUTTER",
                        file_path=path,
                        message=f"Documentation file in project root: {file_info['name']}",
                        suggestion=f"Move to docs/ directory: mv {file_info['name']} docs/",
                    )
                )

            # Check for script files
            elif (
                file_info["suffix"] in script_suffixes
                and any(keyword in name for keyword in script_keywords)
                and file_info["name"] != "setup.py"
            ):

                violations.append(
                    TreeViolation(
                        violation_type="ROOT_CLUTTER",
                        file_path=path,
                        message=f"Script file in project root: {file_info['name']}",
                        suggestion=f"Move to scripts/ directory: mkdir -p scripts && mv {file_info['name']} scripts/",
                    )
                )

            # Check for random Python files (not entry points)
            elif file_info["suffix"] == ".py" and file_info["name"] not in {
                "setup.py",
                "conftest.py",
            }:

                violations.append(
                    TreeViolation(
                        violation_type="ROOT_CLUTTER",
                        file_path=path,
                        message=f"Python file in project root: {file_info['name']}",
                        suggestion="Move to appropriate src/ subdirectory or remove if temporary",
                    )
                )

            # Check for config files that should be in config/
            elif file_info["suffix"] in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
                if not any(
                    allowed in file_info["name"]
                    for allowed in ["pyproject.toml", "setup.cfg", "tox.ini"]
                ):
                    violations.append(
                        TreeViolation(
                            violation_type="ROOT_CLUTTER",
                            file_path=path,
                            message=f"Config file in project root: {file_info['name']}",
                            suggestion="Move to config/ directory or src/package/config/",
                        )
                    )

        return violations

    def _check_scattered_modules(self, tree: Dict[str, Any]) -> List[TreeViolation]:
        """Check for related modules scattered across directories."""
        violations = []

        # Look in src directory for Python files
        if "src" in tree["dirs"]:
            src_tree = tree["dirs"]["src"]

            # Check for package-level scattered files
            for pkg_name, pkg_tree in src_tree["dirs"].items():
                python_files = [f for f in pkg_tree["files"] if f["suffix"] == ".py"]

                if len(python_files) > 10:  # Too many files in one directory
                    # Look for common prefixes
                    prefixes = self._find_common_prefixes([f["name"] for f in python_files])

                    for prefix in prefixes:
                        matching_files = [f for f in python_files if f["name"].startswith(prefix)]

                        if len(matching_files) >= 3:
                            violations.append(
                                TreeViolation(
                                    violation_type="SCATTERED_MODULES",
                                    file_path=matching_files[0]["path"].parent,
                                    message=f"Related modules with '{prefix}' prefix should be grouped: {len(matching_files)} files",
                                    suggestion=f"Create {prefix}/ subdirectory and move related files",
                                )
                            )

        return violations

    def _find_common_prefixes(self, filenames: List[str]) -> List[str]:
        """Find common prefixes in filenames (before first underscore)."""
        prefixes = {}

        for filename in filenames:
            if "_" in filename:
                prefix = filename.split("_")[0]
                if len(prefix) > 2:  # Meaningful prefix
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1

        # Return prefixes that appear 3+ times
        return [prefix for prefix, count in prefixes.items() if count >= 3]

    def _check_directory_structure(self, tree: Dict[str, Any]) -> List[TreeViolation]:
        """Check if directory structure is appropriate for project size."""
        violations = []

        # Count total Python files
        total_py_files = self._count_python_files(tree)

        # Check if structure is too flat for project size
        if total_py_files > 15:
            # Should have some organization
            src_structure = tree["dirs"].get("src", {})

            if src_structure:
                main_package = None
                for pkg_name, pkg_tree in src_structure["dirs"].items():
                    pkg_py_files = len([f for f in pkg_tree["files"] if f["suffix"] == ".py"])
                    if pkg_py_files > 10:
                        main_package = pkg_name
                        break

                if main_package:
                    violations.append(
                        TreeViolation(
                            violation_type="FLAT_STRUCTURE",
                            file_path=self.project_root / "src" / main_package,
                            message=f"Package '{main_package}' has {pkg_py_files} Python files - too flat",
                            suggestion="Group related functionality into subpackages",
                        )
                    )

        return violations

    def _count_python_files(self, tree: Dict[str, Any]) -> int:
        """Recursively count Python files in tree."""
        count = len([f for f in tree["files"] if f["suffix"] == ".py"])

        for subdir in tree["dirs"].values():
            count += self._count_python_files(subdir)

        return count

    def quick_check(self, file_path: Path) -> List[TreeViolation]:
        """
        Quick check for immediate violations when creating/modifying files.

        This should be called before file operations to catch issues early.
        """
        violations = []

        # Check if file is being created in project root inappropriately
        if file_path.parent == self.project_root:
            violations.extend(self._check_single_file_root_placement(file_path))

        return violations

    def _check_single_file_root_placement(self, file_path: Path) -> List[TreeViolation]:
        """Check if a single file should be in project root."""
        violations = []

        allowed_root_files = {
            "README.md",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
            ".gitignore",
            ".gitattributes",
            "tox.ini",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
        }

        if file_path.name not in allowed_root_files:
            # Documentation files
            if file_path.suffix in {".md", ".rst", ".txt"}:
                violations.append(
                    TreeViolation(
                        violation_type="ROOT_CLUTTER",
                        file_path=file_path,
                        message=f"Documentation file should not be in project root: {file_path.name}",
                        suggestion="Move to docs/ directory",
                    )
                )

            # Python files (except setup.py)
            elif file_path.suffix == ".py" and file_path.name != "setup.py":
                violations.append(
                    TreeViolation(
                        violation_type="ROOT_CLUTTER",
                        file_path=file_path,
                        message=f"Python file should not be in project root: {file_path.name}",
                        suggestion="Move to appropriate src/ subdirectory",
                    )
                )

        return violations
