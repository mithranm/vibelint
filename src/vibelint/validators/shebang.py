"""
Validator for Python shebangs.

vibelint/validators/shebang.py
"""

from typing import List


class ValidationResult:
    """
    Class to store the result of a validation.

    vibelint/validators/shebang.py
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = 0
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.errors) > 0 or len(self.warnings) > 0


def validate_shebang(
    content: str, is_script: bool, allowed_shebangs: List[str]
) -> ValidationResult:
    """
    Validate the shebang in a Python file.

    vibelint/validators/shebang.py
    """
    result = ValidationResult()
    lines = content.splitlines()

    # Check if there's a shebang line
    has_shebang = len(lines) > 0 and lines[0].startswith("#!")

    if has_shebang:
        result.line_number = 0
        shebang_line = lines[0]

        # Check if script has __main__ block
        if not is_script:
            result.errors.append(
                f"File has a shebang ({shebang_line}) but no '__main__' block. "
                "Shebangs should only be used in executable scripts."
            )
            result.needs_fix = True
        # Check if shebang is in the allowed list
        elif shebang_line not in allowed_shebangs:
            result.errors.append(
                f"Invalid shebang: {shebang_line}. "
                f"Allowed shebangs: {', '.join(allowed_shebangs)}"
            )
            result.needs_fix = True
    else:
        # If the file is a script, it should have a shebang
        if is_script:
            result.warnings.append(
                "Script with '__main__' block should have a shebang line."
            )
            result.needs_fix = True
            result.line_number = 0  # Insert at the beginning

    return result


def fix_shebang(
    content: str, result: ValidationResult, is_script: bool, preferred_shebang: str
) -> str:
    """
    Fix shebang issues in a Python file.

    vibelint/validators/shebang.py
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()

    # If there's already a shebang line
    if result.line_number == 0 and len(lines) > 0 and lines[0].startswith("#!"):
        # Remove it if the file is not a script
        if not is_script:
            lines.pop(0)
        # Replace it with the preferred shebang if it's invalid
        else:
            lines[0] = preferred_shebang
    # Add a shebang if the file is a script and doesn't have one
    elif is_script and (len(lines) == 0 or not lines[0].startswith("#!")):
        lines.insert(0, preferred_shebang)

    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")
