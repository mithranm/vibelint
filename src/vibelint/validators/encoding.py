"""
Validator for Python encoding cookies.

vibelint/validators/encoding.py
"""

import re
from typing import List


class ValidationResult:
    """
    Class to store the result of a validation.

    vibelint/validators/encoding.py
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.errors) > 0 or len(self.warnings) > 0


def validate_encoding_cookie(content: str) -> ValidationResult:
    """
    Validate the encoding cookie in a Python file.

    vibelint/validators/encoding.py
    """
    result = ValidationResult()
    lines = content.splitlines()

    # Check for encoding cookie pattern
    encoding_pattern = r"^# -\*- coding: (.+) -\*-$"

    # Determine where to look for the encoding cookie
    start_line = 0
    # If there's a shebang, look for the encoding cookie on the second line
    if len(lines) > 0 and lines[0].startswith("#!"):
        start_line = 1

    # Check the encoding cookie
    if start_line < len(lines):
        match = re.match(encoding_pattern, lines[start_line])
        if match:
            encoding = match.group(1)
            result.line_number = start_line

            # Check if the encoding is utf-8
            if encoding.lower() != "utf-8":
                result.errors.append(
                    f"Invalid encoding cookie: {encoding}. Use 'utf-8' instead."
                )
                result.needs_fix = True
        else:
            # No encoding cookie, but we don't require one
            pass

    return result


def fix_encoding_cookie(content: str, result: ValidationResult) -> str:
    """
    Fix encoding cookie issues in a Python file.

    vibelint/validators/encoding.py
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()

    # If there's an invalid encoding cookie, replace it
    if result.line_number >= 0:
        lines[result.line_number] = "# -*- coding: utf-8 -*-"

    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")
