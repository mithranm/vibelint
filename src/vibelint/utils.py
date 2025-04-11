"""
Utility functions for vibelint.

vibelint/utils.py
"""

from pathlib import Path
from typing import Dict, Any, Optional
import fnmatch


def find_package_root(path: Path) -> Optional[Path]:
    """
    Find the package root directory by looking for a setup.py, pyproject.toml, or __init__.py file.
    
    Args:
        path: Path to start searching from
        
    Returns:
        The package root directory, or None if not found
    """
    if path.is_file():
        path = path.parent
    
    current = path
    
    # First try to find setup.py or pyproject.toml
    while current.parent != current:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # If not found, try looking for the top-level __init__.py
    current = path
    while current.parent != current:
        if not (current / "__init__.py").exists() and (current.parent / "__init__.py").exists():
            return current
        if not (current.parent / "__init__.py").exists():
            # Return the last directory that contained __init__.py
            return current
        current = current.parent
    
    # If no package structure found, return the original directory
    return path


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