"""
AST parsing utilities for vibelint validators.

Provides common AST parsing functionality with consistent error handling,
reducing code duplication across validators.

vibelint/src/vibelint/ast_utils.py
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "safe_parse",
    "parse_or_none",
]


def safe_parse(content: str, filename: str | Path = "<unknown>") -> ast.AST:
    """
    Parse Python source code into an AST.

    Args:
        content: Python source code as string
        filename: Optional filename for error messages

    Returns:
        Parsed AST

    Raises:
        SyntaxError: If the code has syntax errors
    """
    return ast.parse(content, filename=str(filename))


def parse_or_none(content: str, filename: str | Path = "<unknown>") -> Optional[ast.AST]:
    """
    Parse Python source code into an AST, returning None on syntax errors.

    This is the recommended function for validators to use - it handles
    syntax errors gracefully and logs them appropriately.

    Args:
        content: Python source code as string
        filename: Optional filename for error messages

    Returns:
        Parsed AST or None if parsing failed

    Example:
        >>> tree = parse_or_none(file_content, file_path)
        >>> if tree is None:
        >>>     return  # Skip validation for files with syntax errors
    """
    try:
        return ast.parse(content, filename=str(filename))
    except SyntaxError as e:
        logger.debug(f"Syntax error in {filename}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing {filename}: {e}")
        return None


def get_docstring(node: ast.AST) -> Optional[str]:
    """
    Extract docstring from an AST node (module, function, or class).

    Args:
        node: AST node (Module, FunctionDef, ClassDef, or AsyncFunctionDef)

    Returns:
        Docstring text or None if no docstring found
    """
    if not isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return None

    body = node.body if hasattr(node, "body") else []
    if not body:
        return None

    first_stmt = body[0]
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        value = first_stmt.value.value
        if isinstance(value, str):
            return value

    return None


def get_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """
    Get argument names from a function definition.

    Args:
        node: Function definition node

    Returns:
        List of argument names
    """
    args = []

    # Regular arguments
    for arg in node.args.args:
        args.append(arg.arg)

    # *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")

    # **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return args


def is_private_name(name: str) -> bool:
    """
    Check if a name is private (starts with underscore).

    Args:
        name: Name to check

    Returns:
        True if name is private (starts with _ but not __)
    """
    return name.startswith("_") and not name.startswith("__")


def is_dunder_name(name: str) -> bool:
    """
    Check if a name is a dunder/magic method (starts and ends with __).

    Args:
        name: Name to check

    Returns:
        True if name is a dunder method
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4
