"""
Line count validator for individual files.

Flags files that exceed configured line count thresholds to encourage
module decomposition and maintainability.

vibelint/src/vibelint/validators/single_file/line_count.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["LineCountValidator"]


class LineCountValidator(BaseValidator):
    """Validates that files don't exceed reasonable line count limits."""

    rule_id = "FILE-TOO-LONG"
    name = "Line Count Validator"
    description = "File exceeds recommended line count limit"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)

        # Default thresholds can be overridden in project config
        self.warning_threshold = self.config.get("warning_threshold", 500)
        self.error_threshold = self.config.get("error_threshold", 1000)
        self.exclude_patterns = self.config.get(
            "exclude_patterns", ["test_*.py", "*_test.py", "conftest.py", "__init__.py"]
        )

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate that file doesn't exceed line count thresholds."""

        # Skip excluded files
        filename = file_path.name
        for pattern in self.exclude_patterns:
            if (
                filename == pattern
                or (pattern.startswith("*") and filename.endswith(pattern[1:]))
                or (pattern.endswith("*") and filename.startswith(pattern[:-1]))
            ):
                return

        # Count non-empty lines (excluding pure whitespace)
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        line_count = len(non_empty_lines)

        # Check thresholds
        if line_count >= self.error_threshold:
            yield self.create_finding(
                message=f"File has {line_count} lines, exceeding error threshold of {self.error_threshold}",
                file_path=file_path,
                line=1,
                column=1,
                suggestion=self._get_suggestion(file_path, line_count),
            )
        elif line_count >= self.warning_threshold:
            yield self.create_finding(
                message=f"File has {line_count} lines, exceeding warning threshold of {self.warning_threshold}",
                file_path=file_path,
                line=1,
                column=1,
                suggestion=self._get_suggestion(file_path, line_count),
            )

    def _get_suggestion(self, file_path: Path, line_count: int) -> str:
        """Generate context-appropriate refactoring suggestions."""

        # Try to analyze file structure for better suggestions
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            # Count classes and functions
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if len(classes) > 1:
                return f"Consider splitting {len(classes)} classes into separate files"
            elif len(functions) > 10:
                return (
                    f"Consider grouping {len(functions)} functions into classes or separate modules"
                )
            elif "cli" in str(file_path).lower():
                return "Consider breaking CLI into command modules (validation, analysis, etc.)"
            elif "config" in str(file_path).lower():
                return "Consider separating config loading, validation, and defaults"
            else:
                return "Consider extracting classes or functions into separate modules"

        except (SyntaxError, UnicodeDecodeError):
            # Fallback for unparseable files
            return f"Consider breaking this {line_count}-line file into smaller, focused modules"


# Entry point registration
def get_validators():
    """Return list of validators provided by this module."""
    return [LineCountValidator]
