"""
Validator for __all__ exports in Python modules.

Checks for presence (optional for __init__.py) and correct format.

vibelint/validators/exports.py
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Config
from ..error_codes import VBL301, VBL302, VBL303, VBL304

__all__ = ["ExportValidationResult", "validate_exports"]

ValidationIssue = Tuple[str, str]


class ExportValidationResult:
    """
    Stores the result of __all__ validation for a single file.

    vibelint/validators/exports.py
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes ExportValidationResult.

        vibelint/validators/exports.py
        """
        self.file_path = file_path
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.has_all: bool = False

        self.all_lineno: Optional[int] = None

    def has_issues(self) -> bool:
        """
        Returns True if there are any errors or warnings.

        vibelint/validators/exports.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/exports.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/exports.py
        """
        self.warnings.append((code, message))


def validate_exports(
    source_code: str, file_path_str: str, config: Config
) -> ExportValidationResult:
    """
    Validates the presence and format of __all__ in the source code.

    Args:
    source_code: The source code of the Python file as a string.
    file_path_str: The path to the file (used for context, e.g., __init__.py).
    config: The loaded vibelint configuration.

    Returns:
    An ExportValidationResult object.

    vibelint/validators/exports.py
    """
    result = ExportValidationResult(file_path_str)
    file_path = Path(file_path_str)
    is_init_py = file_path.name == "__init__.py"

    try:
        tree = ast.parse(source_code, filename=file_path_str)
    except SyntaxError as e:
        result.add_error(VBL304, f"SyntaxError parsing file: {e}")
        return result

    found_all = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                found_all = True
                result.has_all = True
                result.all_lineno = node.lineno

                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    msg = f"__all__ is not assigned a List or Tuple (assigned {type(node.value).__name__}) near line {node.lineno}."
                    result.add_error(VBL303, msg)

                break

    if not found_all:
        error_on_init = config.get("error_on_missing_all_in_init", False)
        if is_init_py and not error_on_init:
            msg = f"Optional: __all__ definition not found in {file_path.name}."
            result.add_warning(VBL302, msg)
        elif not is_init_py:
            msg = f"__all__ definition not found in {file_path.name}."
            result.add_error(VBL301, msg)
        elif is_init_py and error_on_init:
            msg = f"__all__ definition not found in {file_path.name} (required by config)."
            result.add_error(VBL301, msg)

    return result
