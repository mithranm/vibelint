"""
Plugin-aware validation runner for vibelint.

This module provides the PluginValidationRunner that uses the new plugin system
to run validators and format output according to user configuration.
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

from .plugin_system import Finding, Severity, plugin_manager
from .rules import RuleEngine
from .formatters import BUILTIN_FORMATTERS
from .discovery import discover_files

# Note: No longer importing BUILTIN_VALIDATORS - using plugin discovery instead

__all__ = ["PluginValidationRunner", "run_plugin_validation"]


class PluginValidationRunner:
    """Runs validation using the plugin system."""

    def __init__(self, config_dict: Dict[str, Any], project_root: Path):
        """Initialize the plugin validation runner."""
        self.project_root = project_root
        self.config_dict = config_dict
        self.config = config_dict  # Add config property for formatters
        self.rule_engine = RuleEngine(config_dict)
        self.findings: List[Finding] = []

        # Register built-in validators with plugin manager
        self._register_builtin_validators()

    def _register_builtin_validators(self):
        """Register built-in validators with the plugin manager via entry point discovery."""
        # Built-in validators are now discovered automatically via entry points
        plugin_manager.load_plugins()

    def run_validation(self, file_paths: List[Path]) -> List[Finding]:
        """Run validation on the specified files."""
        self.findings = []

        # Get enabled validators
        validators = self.rule_engine.get_enabled_validators()

        for file_path in file_paths:
            if not file_path.exists() or not file_path.is_file():
                continue

            if file_path.suffix != ".py":
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            # Run all validators on this file
            for validator in validators:
                try:
                    for finding in validator.validate(file_path, content):
                        # Make path relative to project root
                        relative_path = file_path.relative_to(self.project_root)
                        finding.file_path = relative_path
                        self.findings.append(finding)
                except Exception:
                    # Skip validator if it fails
                    continue

        return self.findings

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts by severity level."""
        summary = defaultdict(int)
        for finding in self.findings:
            summary[finding.severity.value] += 1
        return dict(summary)

    def format_output(self, output_format: str = "human") -> str:
        """Format the validation results."""
        # Get formatter
        if output_format in BUILTIN_FORMATTERS:
            formatter_class = BUILTIN_FORMATTERS[output_format]
            formatter = formatter_class()
        else:
            # Try plugin formatters
            formatter_class = plugin_manager.get_formatter(output_format)
            if formatter_class:
                formatter = formatter_class()
            else:
                # Fallback to human format
                formatter = BUILTIN_FORMATTERS["human"]()

        summary = self.get_summary()
        return formatter.format_results(self.findings, summary, self.config)

    def has_blocking_issues(self) -> bool:
        """Check if any findings are blocking (BLOCK severity)."""
        return any(finding.severity == Severity.BLOCK for finding in self.findings)

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on findings."""
        if self.has_blocking_issues():
            return 1
        return 0


def run_plugin_validation(
    config_dict: Dict[str, Any], project_root: Path
) -> PluginValidationRunner:
    """
    Run validation using the plugin system.

    Args:
        config_dict: Configuration dictionary from pyproject.toml
        project_root: Project root path

    Returns:
        PluginValidationRunner with results
    """
    from .config import Config

    runner = PluginValidationRunner(config_dict, project_root)

    # Create a fake config object for discovery
    fake_config = Config(project_root, config_dict)

    # Use discovery API properly
    files = discover_files(paths=[project_root], config=fake_config, explicit_exclude_paths=set())

    # Run validation
    runner.run_validation(files)

    return runner
