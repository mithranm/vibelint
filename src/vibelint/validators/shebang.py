"""
Validator for Python shebang lines.

vibelint/validators/shebang.py
"""

import ast
from pathlib import Path
from typing import List, Tuple

from ..error_codes import VBL401, VBL402, VBL403

__all__ = [
    "ShebangValidationResult",
    "validate_shebang",
    "file_contains_top_level_main_block",
]

ValidationIssue = Tuple[str, str]


class ShebangValidationResult:
    """
    Result of a shebang validation.

    vibelint/validators/shebang.py
    """

    def __init__(self) -> None:
        """
        Initializes ShebangValidationResult.

        vibelint/validators/shebang.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = 0

    def has_issues(self) -> bool:
        """
        Check if any issues were found.

        vibelint/validators/shebang.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/shebang.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/shebang.py
        """
        self.warnings.append((code, message))


def validate_shebang(
    content: str, is_script: bool, allowed_shebangs: List[str]
) -> ShebangValidationResult:
    """
    Validate the shebang line if present; ensure it's correct for scripts with __main__.

    vibelint/validators/shebang.py
    """
    res = ShebangValidationResult()
    lines = content.splitlines()

    if lines and lines[0].startswith("#!"):
        sb = lines[0]
        res.line_number = 0
        if not is_script:
            msg = f"File has shebang '{sb}' but no 'if __name__ == \"__main__\":' block found."
            res.add_error(VBL401, msg)
        elif sb not in allowed_shebangs:
            allowed_str = ", ".join(f"'{s}'" for s in allowed_shebangs)
            msg = f"Invalid shebang '{sb}'. Allowed: {allowed_str}."
            res.add_error(VBL402, msg)
    else:
        if is_script:
            res.line_number = 0
            msg = "Script contains 'if __name__ == \"__main__\":' block but lacks a shebang line ('#!...')."
            res.add_warning(VBL403, msg)

    return res


def file_contains_top_level_main_block(file_path: Path, content: str) -> bool:
    """
    Checks if a Python file contains a top-level 'if __name__ == "__main__":' block using AST.
    Returns True if found, False otherwise (including syntax errors).

    vibelint/validators/shebang.py
    """

    try:
        tree = ast.parse(content, filename=str(file_path))
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and (
                        (
                            isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value == "__main__"
                        )
                        or (
                            isinstance(test.comparators[0], ast.Str)
                            and test.comparators[0].s == "__main__"
                        )
                    )
                ):
                    return True
    except (SyntaxError, Exception):

        return False
    return False
