"""
Validator for orphaned Python scripts outside of include globs.

Detects Python scripts that exist outside the configured include patterns.
This is an OPINIONATED validator that enforces a specific project structure
preference - not all projects/teams will consider scripts outside include
patterns to be violations.

This validator is useful for teams that prefer to keep all Python code within
well-defined directories (like src/, tests/, scripts/) and want to catch
one-off scripts that might be created during development but forgotten.

vibelint/validators/orphaned_scripts.py
"""

import fnmatch
from pathlib import Path
from typing import List, Tuple, Set, Optional

from ..error_codes import VBL801, VBL802

__all__ = [
    "OrphanedScriptValidationResult",
    "validate_orphaned_scripts",
]

ValidationIssue = Tuple[str, str]


class OrphanedScriptValidationResult:
    """
    Result of orphaned script validation.

    vibelint/validators/orphaned_scripts.py
    """

    def __init__(self) -> None:
        """Initialize an empty orphaned script validation result."""
        self.issues: List[ValidationIssue] = []
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.orphaned_count: int = 0
        self.orphaned_locations: List[Tuple[str, str]] = []  # (file_path, reason)

    def add_orphaned_script(self, code: str, message: str, file_path: str, reason: str) -> None:
        """Add an orphaned script issue to the result."""
        self.issues.append((code, message))

        # All orphaned scripts are warnings (not errors) since they might be intentional
        self.warnings.append((code, message))

        self.orphaned_locations.append((file_path, reason))
        self.orphaned_count += 1


def validate_orphaned_scripts(
    project_root: Path,
    include_globs: List[str],
    exclude_globs: Optional[List[str]] = None
) -> OrphanedScriptValidationResult:
    """
    Find Python scripts that exist outside the configured include patterns.

    Args:
        project_root: The root directory of the project
        include_globs: List of glob patterns that define included files
        exclude_globs: Optional list of glob patterns to exclude

    Returns:
        OrphanedScriptValidationResult containing any orphaned scripts found
    """
    result = OrphanedScriptValidationResult()

    if not project_root.exists() or not project_root.is_dir():
        result.add_orphaned_script(
            VBL801,
            f"Project root {project_root} does not exist or is not a directory",
            str(project_root), "invalid_project_root"
        )
        return result

    exclude_globs = exclude_globs or []

    # Find all Python files in the project
    all_python_files = list(project_root.rglob("*.py"))

    # Convert globs to be relative to project root for matching
    included_files = set()
    orphaned_files = []

    for py_file in all_python_files:
        try:
            relative_path = py_file.relative_to(project_root)
            relative_path_str = str(relative_path).replace("\\", "/")  # Normalize path separators

            # Check if file should be excluded
            should_exclude = False
            for exclude_pattern in exclude_globs:
                if fnmatch.fnmatch(relative_path_str, exclude_pattern):
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Check if file matches any include pattern
            matches_include = False
            for include_pattern in include_globs:
                if fnmatch.fnmatch(relative_path_str, include_pattern):
                    matches_include = True
                    included_files.add(relative_path_str)
                    break

            if not matches_include:
                orphaned_files.append((py_file, relative_path_str))

        except ValueError:
            # File is outside project root somehow
            result.add_orphaned_script(
                VBL802,
                f"Python file {py_file} is outside project root {project_root}",
                str(py_file), "outside_project_root"
            )

    # Report orphaned files
    for py_file, relative_path_str in orphaned_files:
        # Determine the likely reason why it's orphaned
        reason = _determine_orphan_reason(py_file, relative_path_str, include_globs)

        message = (
            f"Orphaned Python script '{relative_path_str}' found outside include patterns. "
            f"Reason: {reason}. Consider moving to appropriate location or adding to include patterns. "
            f"(Note: This is an opinionated check - not all projects require strict organization.)"
        )

        result.add_orphaned_script(VBL801, message, relative_path_str, reason)

    return result


def _determine_orphan_reason(py_file: Path, relative_path: str, include_globs: List[str]) -> str:
    """Determine why a Python file might be considered orphaned."""

    # Check if it's a common one-off script pattern
    if py_file.name.startswith("test_") and "test" not in " ".join(include_globs):
        return "looks like a test script but tests not in include patterns"

    if py_file.name.startswith("debug_") or py_file.name.startswith("tmp_"):
        return "appears to be a debug/temporary script"

    if py_file.name in ["scratch.py", "temp.py", "test.py", "debug.py", "example.py"]:
        return "common one-off script name"

    if py_file.parent.name in ["scratch", "temp", "debug", "examples"] and "examples" not in " ".join(include_globs):
        return f"in '{py_file.parent.name}' directory not covered by include patterns"

    # Check if it's in the project root (often indicates a one-off script)
    if len(py_file.parts) == 2:  # project_root/script.py
        return "script in project root (consider organizing into appropriate subdirectory)"

    return "does not match any include pattern"