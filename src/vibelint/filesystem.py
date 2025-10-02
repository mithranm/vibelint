"""
Filesystem and path utility functions for vibelint.

vibelint/src/vibelint/fs.py
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_directory",
    "find_files_by_extension",
    "find_package_root",
    "find_project_root",
    "get_import_path",
    "get_module_name",
    "get_relative_path",
    "is_binary",
    "is_python_file",
    "read_file_safe",
    "walk_up_for_config",
    "walk_up_for_project_root",
    "write_file_safe",
]


def walk_up_for_project_root(start_path: Path) -> Path | None:
    """
    Walk up directory tree to find project root markers.

    Project root markers (in order of precedence):
    1. .git directory (definitive project boundary)
    2. pyproject.toml file (Python project config)
    3. dev.pyproject.toml file (development/parent project config)

    Args:
        start_path: Path to start walking up from

    Returns:
        Path to project root, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    while True:
        # Check for git repo (strongest indicator of project root)
        if (current_path / ".git").is_dir():
            return current_path
        # Check for standard Python project config
        if (current_path / "pyproject.toml").is_file():
            return current_path
        # Check for development/parent project config (e.g., kaia's dev.pyproject.toml)
        if (current_path / "dev.pyproject.toml").is_file():
            return current_path
        # Stop at filesystem root
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def walk_up_for_config(start_path: Path) -> Path | None:
    """
    Walk up directory tree to find vibelint configuration.

    Searches for configuration files in this order:
    1. pyproject.toml with [tool.vibelint] section
    2. dev.pyproject.toml with [tool.vibelint] section
    3. .git directory (fallback to git repo root)

    Args:
        start_path: Path to start walking up from

    Returns:
        Path containing viable configuration, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        # Check for standard pyproject.toml with vibelint config
        pyproject_path = current_path / "pyproject.toml"
        if pyproject_path.is_file():
            if _has_vibelint_config(pyproject_path):
                return current_path

        # Check for dev.pyproject.toml with vibelint config (kaia pattern)
        dev_pyproject_path = current_path / "dev.pyproject.toml"
        if dev_pyproject_path.is_file():
            if _has_vibelint_config(dev_pyproject_path):
                return current_path

        # Fallback to git repo root
        if (current_path / ".git").is_dir():
            return current_path

        # Stop at filesystem root
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def _has_vibelint_config(toml_path: Path) -> bool:
    """Check if a TOML file contains vibelint configuration."""
    try:
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            return "tool" in data and "vibelint" in data.get("tool", {})
    except Exception:
        return False


# Backward compatibility alias
find_project_root = walk_up_for_project_root


def find_package_root(start_path: Path) -> Path | None:
    """
    Find the root directory of a Python package containing the given path.

    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A setup.py file
    3. An __init__.py file at the top level with no parent

    Args:
        start_path: Path to start the search from

    Returns:
        Path to package root, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        if (current_path / "__init__.py").is_file():
            project_root_marker = walk_up_for_project_root(current_path)
            if project_root_marker and current_path.is_relative_to(project_root_marker):
                pass

        if (current_path / "pyproject.toml").is_file() or (current_path / ".git").is_dir():
            src_dir = current_path / "src"
            if src_dir.is_dir():
                if start_path.resolve().is_relative_to(src_dir):
                    for item in src_dir.iterdir():
                        if item.is_dir() and (item / "__init__.py").is_file():
                            return item
                    return src_dir
                else:
                    if (current_path / "__init__.py").is_file():
                        return current_path

            if (current_path / "__init__.py").is_file():
                return current_path
            return current_path

        if current_path.parent == current_path:
            return start_path.parent if start_path.is_file() else start_path

        current_path = current_path.parent


def is_python_file(path: Path) -> bool:
    """
    Check if a path represents a Python file.

    Args:
        path: Path to check

    Returns:
        True if the path is a Python file, False otherwise

    vibelint/src/vibelint/fs.py
    """
    return path.is_file() and path.suffix == ".py"


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Safely compute a relative path, falling back to the original path.

    vibelint/src/vibelint/fs.py
    """
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError as e:
        logger.debug(f"Path {path} is not relative to {base}: {e}")
        return path.resolve()


def get_import_path(file_path: Path, package_root: Path | None = None) -> str:
    """
    Get the import path for a Python file.

    Args:
        file_path: Path to the Python file
        package_root: Optional path to the package root

    Returns:
        Import path (e.g., "vibelint.utils")

    vibelint/src/vibelint/fs.py
    """
    if package_root is None:
        package_root = find_package_root(file_path)

    if package_root is None:
        return file_path.stem

    try:
        rel_path = file_path.relative_to(package_root)
        import_path = str(rel_path).replace(os.sep, ".").replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]
        return import_path
    except ValueError as e:
        logger.debug(f"Could not determine import path for {file_path}: {e}")
        return file_path.stem


def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.

    Args:
        file_path: Path to a Python file

    Returns:
        Module name

    vibelint/src/vibelint/fs.py
    """
    return file_path.stem


def find_files_by_extension(
    root_path: Path,
    extension: str = ".py",
    exclude_globs: list[str] = [],
    include_vcs_hooks: bool = False,
) -> list[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.

    Args:
        root_path: Root path to search in
        extension: File extension to look for (including the dot)
        exclude_globs: Glob patterns to exclude
        include_vcs_hooks: Whether to include version control directories

    Returns:
        List of paths to files with the specified extension

    vibelint/src/vibelint/fs.py
    """
    if exclude_globs is None:
        exclude_globs = []

    result = []

    for file_path in root_path.glob(f"**/*{extension}"):
        if not include_vcs_hooks:
            if any(
                part.startswith(".") and part in {".git", ".hg", ".svn"} for part in file_path.parts
            ):
                continue

        if any(fnmatch.fnmatch(str(file_path), pattern) for pattern in exclude_globs):
            continue

        result.append(file_path)

    return result


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path to the directory

    vibelint/src/vibelint/fs.py
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> str | None:
    """
    Safely read a file, returning None if any errors occur.

    Args:
        file_path: Path to file
        encoding: File encoding

    Returns:
        File contents or None if error

    vibelint/src/vibelint/fs.py
    """
    try:
        return file_path.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not read file {file_path}: {e}")
        return None


def write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> bool:
    """
    Safely write content to a file, returning success status.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding

    Returns:
        True if successful, False otherwise

    vibelint/src/vibelint/fs.py
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except (OSError, UnicodeEncodeError) as e:
        logger.debug(f"Could not write file {file_path}: {e}")
        return False


def is_binary(file_path: Path, chunk_size: int = 1024) -> bool:
    """
    Check if a file appears to be binary by looking for null bytes
    or a high proportion of non-text bytes in the first chunk.

    Args:
        file_path: The path to the file.
        chunk_size: The number of bytes to read from the beginning.

    Returns:
        True if the file seems binary, False otherwise.

    vibelint/src/vibelint/fs.py
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
        if not chunk:
            return False

        if b"\x00" in chunk:
            return True

        text_characters = bytes(range(32, 127)) + b"\n\r\t\f\b"
        non_text_count = sum(1 for byte in chunk if bytes([byte]) not in text_characters)

        if len(chunk) > 0 and (non_text_count / len(chunk)) > 0.3:
            return True

        return False
    except OSError:
        return True
    except (TypeError, AttributeError) as e:
        logger.debug(f"Error checking if {file_path} is binary: {e}")
        return True
