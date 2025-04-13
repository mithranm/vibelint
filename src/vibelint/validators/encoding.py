"""
Validator for Python encoding cookies.

vibelint/validators/encoding.py
"""

import re
from typing import List, Tuple

from ..error_codes import VBL201

__all__ = [
    "EncodingValidationResult",
    "validate_encoding_cookie",
]

ValidationIssue = Tuple[str, str]


class EncodingValidationResult:
    """
    Result of a validation for encoding cookies.

    vibelint/validators/encoding.py
    """

    def __init__(self) -> None:
        """
        Initializes EncodingValidationResult.

        vibelint/validators/encoding.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = -1

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        vibelint/validators/encoding.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/encoding.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/encoding.py
        """
        self.warnings.append((code, message))


def validate_encoding_cookie(content: str) -> EncodingValidationResult:
    """
    Validate the encoding cookie in a Python file.

    vibelint/validators/encoding.py
    """
    result = EncodingValidationResult()
    lines = content.splitlines()
    pattern = r"^# -\*- coding: (.+) -\*-$"

    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1

    if idx < len(lines):
        m = re.match(pattern, lines[idx])
        if m:
            enc = m.group(1).lower()
            result.line_number = idx
            if enc != "utf-8":
                msg = f"Invalid encoding cookie: '{enc}' found on line {idx + 1}, must be 'utf-8'."
                result.add_error(VBL201, msg)

    return result
