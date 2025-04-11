"""
Validator for Python encoding cookies.

src/vibelint/validators/encoding.py
"""

import re
from typing import List


class EncodingValidationResult:
    """
    Result of a validation for encoding cookies.

    src/vibelint/validators/encoding.py
    """
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = -1
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        src/vibelint/validators/encoding.py
        """
        return bool(self.errors or self.warnings)


def validate_encoding_cookie(content: str) -> EncodingValidationResult:
    """
    Validate the encoding cookie in a Python file.

    src/vibelint/validators/encoding.py
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
                result.errors.append(f"Invalid encoding cookie: {enc}, must be utf-8.")
                result.needs_fix = True
    return result


def fix_encoding_cookie(content: str, result: EncodingValidationResult) -> str:
    """
    Fix encoding cookie issues in a Python file.

    src/vibelint/validators/encoding.py
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()
    if 0 <= result.line_number < len(lines):
        lines[result.line_number] = "# -*- coding: utf-8 -*-"
    text = "\n".join(lines)
    if content.endswith("\n"):
        text += "\n"
    return text
