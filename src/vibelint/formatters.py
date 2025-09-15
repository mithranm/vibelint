"""
Report formatters for vibelint validation results.

Provides pluggable report formatters for different output formats including
human-readable, JSON, SARIF, and LLM-optimized formats.

vibelint/src/vibelint/formatters.py
"""

import json
from typing import Any, Dict, List, Optional

from .plugin_system import BaseFormatter, Finding, Severity

__all__ = [
    "NaturalLanguageFormatter",
    "JsonFormatter",
    "SarifFormatter",
    "BUILTIN_FORMATTERS",
    "FORMAT_CHOICES",
    "DEFAULT_FORMAT",
]


class NaturalLanguageFormatter(BaseFormatter):
    """Natural language output formatter for humans and AI agents."""

    name = "natural"
    description = "Natural language output format optimized for human and AI agent consumption"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results for human reading."""
        if not findings:
            return "All checks passed!"

        # Get max display limit from config
        max_displayed = 50  # Default to 50 issues for readability (0 means no limit)
        if config and hasattr(config, "get"):
            max_displayed = config.get("max_displayed_issues", 50)
        elif config and hasattr(config, "__getitem__"):
            max_displayed = config.get("max_displayed_issues", 50)

        # Group findings by severity
        by_severity = {Severity.BLOCK: [], Severity.WARN: [], Severity.INFO: []}

        for finding in findings:
            if finding.severity in by_severity:
                by_severity[finding.severity].append(finding)

        lines = []
        displayed_count = 0
        total_count = len(findings)

        # Add findings by severity (highest first)
        for severity in [Severity.BLOCK, Severity.WARN, Severity.INFO]:
            if by_severity[severity]:
                lines.append(f"\n{severity.value}:")

                severity_findings = by_severity[severity]
                for _, finding in enumerate(severity_findings):
                    if max_displayed > 0 and displayed_count >= max_displayed:
                        remaining_total = total_count - displayed_count
                        lines.append("")
                        lines.append(
                            f"  WARNING: Showing first {max_displayed} issues. {remaining_total} more found."
                        )
                        lines.append(
                            "  TIP: Set max_displayed_issues = 0 in pyproject.toml to show all issues."
                        )
                        break

                    location = (
                        f"{finding.file_path}:{finding.line}"
                        if finding.line > 0
                        else str(finding.file_path)
                    )
                    lines.append(f"  {finding.rule_id}: {finding.message} ({location})")
                    if finding.suggestion:
                        lines.append(f"    → {finding.suggestion}")

                    displayed_count += 1

                if max_displayed > 0 and displayed_count >= max_displayed:
                    break

        # Add summary with full counts
        total_errors = sum(1 for f in findings if f.severity == Severity.BLOCK)
        total_warnings = sum(1 for f in findings if f.severity == Severity.WARN)
        total_info = sum(1 for f in findings if f.severity == Severity.INFO)

        summary_line = (
            f"\nSummary: {total_errors} errors, {total_warnings} warnings, {total_info} info"
        )
        if max_displayed > 0 and total_count > max_displayed:
            summary_line += f" (showing first {min(max_displayed, total_count)} of {total_count})"

        lines.append(summary_line)

        return "\n".join(lines)


class JsonFormatter(BaseFormatter):
    """JSON output formatter for machine processing."""

    name = "json"
    description = "JSON output format for CI/tooling integration"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results as JSON."""
        result = {"summary": summary, "findings": [finding.to_dict() for finding in findings]}
        return json.dumps(result, indent=2, default=str)


class SarifFormatter(BaseFormatter):
    """SARIF output formatter for GitHub integration."""

    name = "sarif"
    description = "SARIF format for GitHub code scanning"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results as SARIF JSON."""
        rules = {}
        results = []

        for finding in findings:
            # Collect unique rules
            if finding.rule_id not in rules:
                rules[finding.rule_id] = {
                    "id": finding.rule_id,
                    "name": finding.rule_id,
                    "shortDescription": {"text": finding.message},
                    "defaultConfiguration": {
                        "level": self._severity_to_sarif_level(finding.severity)
                    },
                }

            # Add result
            result = {
                "ruleId": finding.rule_id,
                "level": self._severity_to_sarif_level(finding.severity),
                "message": {"text": finding.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": str(finding.file_path)},
                            "region": {
                                "startLine": max(1, finding.line),
                                "startColumn": max(1, finding.column),
                            },
                        }
                    }
                ],
            }

            if finding.suggestion:
                result["fixes"] = [{"description": {"text": finding.suggestion}}]

            results.append(result)

        sarif_output = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "vibelint",
                            "version": "0.1.2",
                            "informationUri": "https://github.com/mithranm/vibelint",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

        return json.dumps(sarif_output, separators=(",", ":"))

    def _severity_to_sarif_level(self, severity: Severity) -> str:
        """Convert vibelint severity to SARIF level."""
        mapping = {
            Severity.BLOCK: "error",
            Severity.WARN: "warning",
            Severity.INFO: "note",
            Severity.OFF: "none",
        }
        return mapping.get(severity, "warning")


class LLMFormatter(BaseFormatter):
    """LLM-optimized formatter for AI analysis."""

    name = "llm"
    description = "LLM-optimized format for AI analysis"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results for LLM analysis."""
        if not findings:
            return "No issues found."

        output = []
        for finding in findings:
            output.append(
                f"{finding.rule_id}: {finding.message} " f"({finding.file_path}:{finding.line})"
            )

        return "\n".join(output)


class HumanFormatter(NaturalLanguageFormatter):
    """Human-readable formatter (alias for NaturalLanguageFormatter)."""

    name = "human"
    description = "Human-readable format with colors and styling"


# Built-in report formatters
BUILTIN_FORMATTERS = {
    "natural": NaturalLanguageFormatter,
    "human": HumanFormatter,  # Separate class for plugin system compatibility
    "json": JsonFormatter,
    "sarif": SarifFormatter,
    "llm": LLMFormatter,
}

# Format choices for CLI - single source of truth
FORMAT_CHOICES = list(BUILTIN_FORMATTERS.keys())
DEFAULT_FORMAT = "natural"
