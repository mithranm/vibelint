"""
Validator for Python shebang lines.

src/vibelint/validators/shebang.py
"""

from typing import List


class ShebangValidationResult:
    """
    Result of a shebang validation.

    src/vibelint/validators/shebang.py
    """
    def __init__(self) -> None:
        """
        Docstring for method 'ShebangValidationResult.__init__'.
        
        vibelint/validators/shebang.py
        """
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = 0
        self.needs_fix: bool = False

    def has_issues(self) -> bool:
        """
        Check if any issues were found.

        src/vibelint/validators/shebang.py
        """
        return bool(self.errors or self.warnings)


def validate_shebang(
    content: str,
    is_script: bool,
    allowed_shebangs: List[str]
) -> ShebangValidationResult:
    """
    Validate the shebang line if present; ensure it's correct for scripts with __main__.

    src/vibelint/validators/shebang.py
    """
    res = ShebangValidationResult()
    lines = content.splitlines()

    if lines and lines[0].startswith("#!"):
        sb = lines[0]
        res.line_number = 0
        if not is_script:
            res.errors.append(
                f"File has shebang {sb} but no '__main__' block."
            )
            res.needs_fix = True
        elif sb not in allowed_shebangs:
            res.errors.append(
                f"Invalid shebang {sb}. Allowed: {', '.join(allowed_shebangs)}"
            )
            res.needs_fix = True
    else:
        # If is_script but no shebang
        if is_script:
            res.warnings.append("Script has '__main__' but lacks a shebang.")
            res.needs_fix = True
            res.line_number = 0

    return res


def fix_shebang(
    content: str,
    result: ShebangValidationResult,
    is_script: bool,
    preferred_shebang: str
) -> str:
    """
    Fix the shebang line if needed.

    src/vibelint/validators/shebang.py
    """
    if not result.needs_fix:
        return content

    lines = content.splitlines()
    if lines and lines[0].startswith("#!"):
        if not is_script:
            # remove it
            lines.pop(0)
        else:
            lines[0] = preferred_shebang
    else:
        # add
        if is_script:
            lines.insert(0, preferred_shebang)
    text = "\n".join(lines)
    if content.endswith("\n"):
        text += "\n"
    return text
