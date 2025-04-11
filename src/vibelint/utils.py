"""
Utility functions for vibelint.

src/vibelint/utils.py
"""

import os
from pathlib import Path
from typing import Optional, List

def find_package_root(path: Path) -> Optional[Path]:
    """
    Find the root directory of a Python package containing the given path.
    
    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A setup.py file
    3. An __init__.py file at the top level with no parent __init__.py
    
    Args:
        path: Path to start the search from
        
    Returns:
        Path to package root, or None if not found
    
    src/vibelint/utils.py
    """
    # Convert to absolute path if it isn't already
    path = path.resolve()
    
    # If path is a file, start with its parent directory
    if path.is_file():
        path = path.parent
    
    # Start from the given directory and move upwards
    current = path
    
    # Limit the traversal to avoid infinite loops
    max_depth = 10
    depth = 0
    
    while current != current.parent and depth < max_depth:
        # Check for package indicators
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        
        # Also, check for __init__.py in all subdirectories to identify a package
        if (current / "__init__.py").exists():
            # If we find __init__.py, we need to keep going up until no more __init__.py
            # to find the actual package root
            package_parent = current.parent
            if not (package_parent / "__init__.py").exists():
                return current
        
        # Move one directory up
        current = current.parent
        depth += 1
    
    # No package root found
    return None

def is_python_file(path: Path) -> bool:
    """
    Check if a path represents a Python file.
    
    Args:
        path: Path to check
        
    Returns:
        True if the path is a Python file, False otherwise
    
    src/vibelint/utils.py
    """
    return path.is_file() and path.suffix == ".py"

def get_relative_path(file_path: Path, base_paths: List[Path]) -> str:
    """
    Get the shortest relative path from a list of base paths.
    
    Args:
        file_path: Path to get the relative path for
        base_paths: List of potential base paths
        
    Returns:
        Shortest relative path as string
    
    src/vibelint/utils.py
    """
    shortest = None
    
    for base_path in base_paths:
        try:
            rel_path = file_path.relative_to(base_path)
            if shortest is None or len(str(rel_path)) < len(str(shortest)):
                shortest = rel_path
        except ValueError:
            continue
    
    return str(shortest) if shortest else str(file_path)

def get_import_path(file_path: Path, package_root: Optional[Path] = None) -> str:
    """
    Get the import path for a Python file.
    
    Args:
        file_path: Path to the Python file
        package_root: Optional path to the package root
        
    Returns:
        Import path (e.g., "vibelint.utils")
        
    src/vibelint/utils.py
    """
    if package_root is None:
        package_root = find_package_root(file_path)
    
    if package_root is None:
        # Fall back to just the file name without extension
        return file_path.stem
    
    try:
        rel_path = file_path.relative_to(package_root)
        # Convert path separators to dots and remove the .py extension
        import_path = str(rel_path).replace(os.sep, ".").replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]
        return import_path
    except ValueError:
        # If the file is not within the package root
        return file_path.stem

def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.
    
    Args:
        file_path: Path to a Python file
        
    Returns:
        Module name
    
    src/vibelint/utils.py
    """
    return file_path.stem

def find_files_by_extension(root_path: Path, extension: str = ".py", 
                           exclude_globs: List[str] = [], 
                           include_vcs_hooks: bool = False) -> List[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.
    
    Args:
        root_path: Root path to search in
        extension: File extension to look for (including the dot)
        exclude_globs: Glob patterns to exclude
        include_vcs_hooks: Whether to include version control directories
        
    Returns:
        List of paths to files with the specified extension
    
    src/vibelint/utils.py
    """
    import fnmatch
    
    if exclude_globs is None:
        exclude_globs = []
    
    result = []
    
    for file_path in root_path.glob(f"**/*{extension}"):
        # Skip VCS directories if not included
        if not include_vcs_hooks:
            if any(part.startswith(".") and part in {".git", ".hg", ".svn"} 
                  for part in file_path.parts):
                continue
        
        # Skip excluded paths
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
    
    src/vibelint/utils.py
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
    
    src/vibelint/utils.py
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
    
    src/vibelint/utils.py
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False