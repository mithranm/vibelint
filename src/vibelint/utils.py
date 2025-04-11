"""
Utility functions for vibelint.

vibelint/utils.py
"""

from pathlib import Path
from typing import Dict, Any
import fnmatch


def count_python_files(
    directory: Path, config: Dict[str, Any], include_vcs_hooks: bool = False
) -> int:
    """
    Count the number of Python files in a directory that match the configuration.

    vibelint/utils.py
    """
    count = 0

    for include_glob in config["include_globs"]:
        for file_path in directory.glob(include_glob):
            # Skip if it's not a file or not a Python file
            if not file_path.is_file() or file_path.suffix != ".py":
                continue

            # Skip VCS directories unless explicitly included
            if not include_vcs_hooks and any(
                part.startswith(".") and part in {".git", ".hg", ".svn"}
                for part in file_path.parts
            ):
                continue

            # Check exclude patterns
            if any(
                fnmatch.fnmatch(str(file_path), str(directory / exclude_glob))
                for exclude_glob in config["exclude_globs"]
            ):
                continue

            count += 1

    return count


def find_package_root(directory: Path) -> Path:
    """
    Find the Python package root by looking for __init__.py files.

    vibelint/utils.py
    """
    current = directory.absolute()
    while current != current.parent:
        # If this directory has an __init__.py, check if its parent has one too
        if (current / "__init__.py").exists():
            parent = current.parent
            if not (parent / "__init__.py").exists():
                return current
        current = current.parent

    # If no package root found, return the original directory
    return directory
