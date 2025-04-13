"""
Utility functions for vibelint.

vibelint/utils.py
"""

import os
from pathlib import Path
from typing import List, Optional

__all__ = [
    "ensure_directory",
    "find_files_by_extension",
    "find_package_root",
    "get_import_path",
    "get_module_name",
    "get_relative_path",
    "is_python_file",
    "read_file_safe",
    "write_file_safe",
    "find_project_root",
    "is_binary",
]


def find_project_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a project containing the given path.

    A project root is identified by containing either:
    1. A pyproject.toml file
    2. A .git directory

    Args:
    start_path: Path to start the search from

    Returns:
    Path to project root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    while True:
        if (current_path / "pyproject.toml").is_file():
            return current_path
        if (current_path / ".git").is_dir():
            return current_path
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def find_package_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a Python package containing the given path.

    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A  file
    3. An  file at the top level with no parent

    Args:
    start_path: Path to start the search from

    Returns:
    Path to package root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        if (current_path / "__init__.py").is_file():
            project_root_marker = find_project_root(current_path)
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

    vibelint/utils.py
    """

    return path.is_file() and path.suffix == ".py"


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Safely compute a relative path, falling back to the original path.

    vibelint/utils.py
    """

    try:

        return path.resolve().relative_to(base.resolve())
    except ValueError:

        return path.resolve()


def get_import_path(file_path: Path, package_root: Optional[Path] = None) -> str:
    """
    Get the import path for a Python file.

    Args:
    file_path: Path to the Python file
    package_root: Optional path to the package root

    Returns:
    Import path (e.g., "vibelint.utils")

    vibelint/utils.py
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
    except ValueError:
        return file_path.stem


def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.

    Args:
    file_path: Path to a Python file

    Returns:
    Module name

    vibelint/utils.py
    """

    return file_path.stem


def find_files_by_extension(
    root_path: Path,
    extension: str = ".py",
    exclude_globs: List[str] = [],
    include_vcs_hooks: bool = False,
) -> List[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.

    Args:
    root_path: Root path to search in
    extension: File extension to look for (including the dot)
    exclude_globs: Glob patterns to exclude
    include_vcs_hooks: Whether to include version control directories

    Returns:
    List of paths to files with the specified extension

    vibelint/utils.py
    """

    import fnmatch

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

    vibelint/utils.py
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
    """
    Safely read a file, returning None if any errors occur.

    Args:
    file_path: Path to file
    encoding: File encoding

    Returns:
    File contents or None if error

    vibelint/utils.py
    """

    try:
        return file_path.read_text(encoding=encoding)
    except Exception:
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

    vibelint/utils.py
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception:
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

    vibelint/utils.py
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
    except Exception:

        return True
