"""Single-file validators for vibelint.

These validators analyze individual Python files in isolation.
They should not require knowledge of other files in the project.

vibelint/src/vibelint/validators/single_file/__init__.py
"""

from pathlib import Path
from typing import Iterator, List

from ...validators.types import BaseValidator, Finding


class SingleFileValidator(BaseValidator):
    """Base class for validators that analyze individual files."""

    def validate_file(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate a single file in isolation.

        Args:
            file_path: Path to the file being validated
            content: File content as string
            config: Configuration object

        Yields:
            Finding objects for any issues found

        """
        # Default implementation delegates to validate method
        yield from self.validate(file_path, content, config)

    def requires_project_context(self) -> bool:
        """Single-file validators do not require project context."""
        return False


def get_single_file_validators() -> List[str]:
    """Get list of single-file validator names."""
    return [
        "DOCSTRING-MISSING",
        "DOCSTRING-PATH-REFERENCE",
        "PRINT-STATEMENT",
        "EMOJI-IN-STRING",
        "TYPING-POOR-PRACTICE",
        "EXPORTS-MISSING-ALL",
        "EXPORTS-MISSING-ALL-INIT",
    ]
