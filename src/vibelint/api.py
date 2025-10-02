"""Public API for vibelint that returns results instead of calling sys.exit().

This module provides a clean library interface for programmatic usage of vibelint,
allowing integration with other tools without subprocess overhead.

The CLI commands wrap these API functions and handle exit codes/formatting.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from vibelint.config import load_config
from vibelint.filesystem import walk_up_for_config
from vibelint.validation_engine import PluginValidationRunner


@dataclass
class FindingDict:
    """Serializable representation of a Finding for API responses."""

    rule: str
    level: str
    path: str
    line: int
    column: int
    msg: str
    context: str = ""
    suggestion: str = ""


@dataclass
class FindingSummary:
    """Summary of findings by severity level."""

    INFO: int = 0
    WARN: int = 0
    BLOCK: int = 0


@dataclass
class CheckResults:
    """Results from a check operation."""

    findings: List[FindingDict]
    summary: FindingSummary
    total_files_checked: int


@dataclass
class VibelintResult:
    """Container for vibelint operation results."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class VibelintAPI:
    """Main API interface for vibelint operations."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the vibelint API.

        Args:
            config_path: Path to vibelint config file (optional)
            working_dir: Working directory for operations (optional, defaults to current dir)

        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

        # Find project root and load config
        self.project_root = walk_up_for_config(self.working_dir)
        self.config = load_config(self.project_root or self.working_dir)

        # Set up logging to capture errors without outputting to console
        self.logger = logging.getLogger(__name__)

    def check(
        self,
        targets: Optional[List[str]] = None,
        exclude_ai: bool = False,
        rules: Optional[List[str]] = None,
    ) -> VibelintResult:
        """Run vibelint validation checks.

        Args:
            targets: List of files/directories to check (defaults to current directory)
            exclude_ai: Skip AI-powered validators for faster execution
            rules: Specific rules to run (comma-separated list)

        Returns:
            VibelintResult with validation results

        """
        try:
            targets = targets or ["."]

            # Convert string targets to Path objects and discover files
            from vibelint.discovery import discover_files_from_paths

            target_paths = [Path(t) for t in targets]
            file_paths = discover_files_from_paths(target_paths, self.config)

            # Create runner with config
            runner = PluginValidationRunner(self.config, self.project_root or self.working_dir)

            # Run validation
            findings = runner.run_validation(file_paths)

            # Convert findings to our format and create summary
            all_findings = []
            summary = FindingSummary()

            for finding in findings:
                finding_dict = FindingDict(
                    rule=finding.rule_id,
                    level=finding.severity.name,
                    path=str(finding.file_path),
                    line=finding.line,
                    column=finding.column,
                    msg=finding.message,
                    context=finding.context or "",
                    suggestion=finding.suggestion or "",
                )
                all_findings.append(finding_dict)

                # Update summary
                level = finding.severity.name
                if level == "INFO":
                    summary.INFO += 1
                elif level == "WARN":
                    summary.WARN += 1
                elif level == "BLOCK":
                    summary.BLOCK += 1

            check_results = CheckResults(
                findings=all_findings, summary=summary, total_files_checked=len(file_paths)
            )
            return VibelintResult(True, asdict(check_results))

        except Exception as e:
            self.logger.error(f"Check operation failed: {e}")
            return VibelintResult(False, errors=[str(e)])

    def validate_file(self, file_path: Union[str, Path]) -> VibelintResult:
        """Validate a single file.

        Args:
            file_path: Path to file to validate

        Returns:
            VibelintResult with validation results for the file

        """
        try:
            path = Path(file_path)
            if not path.exists():
                return VibelintResult(False, errors=[f"File does not exist: {path}"])

            if not path.is_file():
                return VibelintResult(False, errors=[f"Path is not a file: {path}"])

            # Create runner with config
            runner = PluginValidationRunner(self.config, self.project_root or self.working_dir)

            # Run validation on single file
            findings = runner.run_validation([path])

            # Convert findings to our format and create summary
            all_findings = []
            summary = {"INFO": 0, "WARN": 0, "BLOCK": 0}

            for finding in findings:
                finding_dict = {
                    "rule": finding.rule_id,
                    "level": finding.severity.name,
                    "path": str(finding.file_path),
                    "line": finding.line,
                    "column": finding.column,
                    "msg": finding.message,
                    "context": finding.context or "",
                    "suggestion": finding.suggestion or "",
                }
                all_findings.append(finding_dict)

                # Update summary
                level = finding.severity.name
                summary[level] = summary.get(level, 0) + 1

            return VibelintResult(
                True, {"file": str(path), "findings": all_findings, "summary": summary}
            )

        except Exception as e:
            return VibelintResult(False, errors=[str(e)])

    def run_justification(self, target_dir: Optional[str] = None) -> VibelintResult:
        """Run justification workflow for architectural analysis.

        Args:
            target_dir: Directory to analyze (defaults to current directory)

        Returns:
            VibelintResult with justification analysis

        """
        try:
            target_path = Path(target_dir) if target_dir else self.working_dir

            if not target_path.exists():
                return VibelintResult(
                    False, errors=[f"Target directory does not exist: {target_path}"]
                )

            # For now, return a simplified justification result
            # The full justification engine requires more complex setup
            return VibelintResult(
                True,
                {
                    "analysis": {"message": "Justification analysis not yet implemented in API"},
                    "target_directory": str(target_path),
                },
            )

        except Exception as e:
            self.logger.error(f"Justification workflow failed: {e}")
            return VibelintResult(False, errors=[str(e)])


# Convenience functions for common operations
def check_files(
    targets: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    exclude_ai: bool = False,
    rules: Optional[List[str]] = None,
) -> VibelintResult:
    """Convenience function to check files/directories.

    Args:
        targets: Files/directories to check
        config_path: Path to config file
        exclude_ai: Skip AI validators
        rules: Specific rules to run

    Returns:
        VibelintResult with validation results

    """
    api = VibelintAPI(config_path)
    return api.check(targets, exclude_ai, rules)


def validate_single_file(
    file_path: Union[str, Path], config_path: Optional[str] = None
) -> VibelintResult:
    """Convenience function to validate a single file.

    Args:
        file_path: Path to file to validate
        config_path: Path to config file

    Returns:
        VibelintResult with validation results

    """
    api = VibelintAPI(config_path)
    return api.validate_file(file_path)


def run_project_justification(
    target_dir: Optional[str] = None, config_path: Optional[str] = None
) -> VibelintResult:
    """Convenience function to run justification analysis.

    Args:
        target_dir: Directory to analyze
        config_path: Path to config file

    Returns:
        VibelintResult with justification analysis

    """
    api = VibelintAPI(config_path)
    return api.run_justification(target_dir)
