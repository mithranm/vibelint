"""Project-wide validators for vibelint.

These validators analyze entire projects and require knowledge
of multiple files to identify issues like:
- Dead code across modules
- API consistency violations
- Architecture pattern violations
- Semantic similarity between modules

vibelint/src/vibelint/validators/project_wide/__init__.py
"""

from pathlib import Path
from typing import Dict, Iterator, List

from ...validators.types import BaseValidator, Finding


class ProjectWideValidator(BaseValidator):
    """Base class for validators that analyze entire projects."""

    def validate_project(self, project_files: Dict[Path, str], config=None) -> Iterator[Finding]:
        """Validate entire project with knowledge of all files.

        Args:
            project_files: Dictionary mapping file paths to their content
            config: Configuration object

        Yields:
            Finding objects for any issues found

        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Project-wide validators must implement validate_project")

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Single-file validate method for project-wide validators.

        Project-wide validators should not be called on individual files.
        This method raises an error to prevent misuse.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a project-wide validator. "
            "Use validate_project() instead of validate()."
        )

    def requires_project_context(self) -> bool:
        """Project-wide validators require full project context."""
        return True


def get_project_wide_validators() -> List[str]:
    """Get list of project-wide validator names."""
    return [
        "DEAD-CODE-FOUND",
        "ARCHITECTURE-INCONSISTENT",
        "ARCHITECTURE-LLM",
        "SEMANTIC-SIMILARITY",
        "FALLBACK-SILENT-FAILURE",
        "API-CONSISTENCY",
        "CODE-SMELLS",
        "MODULE-COHESION",
    ]
