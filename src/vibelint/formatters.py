"""
Output formatters for vibelint results.

Provides pluggable formatters for different output formats including
human-readable, JSON, and extensible custom formats.
"""

import json
from typing import Dict, List
from .plugin_system import BaseFormatter, Finding, Severity

__all__ = ["HumanFormatter", "JsonFormatter", "SarifFormatter", "BUILTIN_FORMATTERS"]


class HumanFormatter(BaseFormatter):
    """Human-readable output formatter (default)."""

    name = "human"
    description = "Human-readable output format"

    def format_results(self, findings: List[Finding], summary: Dict[str, int]) -> str:
        """Format results for human reading."""
        if not findings:
            return "All checks passed! ✨"

        # Group findings by severity
        by_severity = {
            Severity.BLOCK: [],
            Severity.WARN: [],
            Severity.INFO: []
        }

        for finding in findings:
            if finding.severity in by_severity:
                by_severity[finding.severity].append(finding)

        lines = []

        # Add findings by severity (highest first)
        for severity in [Severity.BLOCK, Severity.WARN, Severity.INFO]:
            if by_severity[severity]:
                lines.append(f"\n{severity.value}:")
                for finding in by_severity[severity]:
                    location = f"{finding.file_path}:{finding.line}" if finding.line > 0 else str(finding.file_path)
                    lines.append(f"  {finding.rule_id}: {finding.message} ({location})")
                    if finding.suggestion:
                        lines.append(f"    → {finding.suggestion}")

        # Add summary
        lines.append(f"\nSummary: {summary.get('BLOCK', 0)} errors, {summary.get('WARN', 0)} warnings, {summary.get('INFO', 0)} info")

        return "\n".join(lines)


class JsonFormatter(BaseFormatter):
    """JSON output formatter for machine processing."""

    name = "json"
    description = "JSON output format for CI/tooling integration"

    def format_results(self, findings: List[Finding], summary: Dict[str, int]) -> str:
        """Format results as JSON."""
        result = {
            "summary": summary,
            "findings": [finding.to_dict() for finding in findings]
        }
        return json.dumps(result, indent=2, default=str)


class SarifFormatter(BaseFormatter):
    """SARIF output formatter for GitHub integration."""

    name = "sarif"
    description = "SARIF format for GitHub code scanning"

    def format_results(self, findings: List[Finding], summary: Dict[str, int]) -> str:
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
                    }
                }

            # Add result
            result = {
                "ruleId": finding.rule_id,
                "level": self._severity_to_sarif_level(finding.severity),
                "message": {"text": finding.message},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(finding.file_path)},
                        "region": {
                            "startLine": max(1, finding.line),
                            "startColumn": max(1, finding.column)
                        }
                    }
                }]
            }

            if finding.suggestion:
                result["fixes"] = [{
                    "description": {"text": finding.suggestion}
                }]

            results.append(result)

        sarif_output = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "vibelint",
                        "version": "0.1.2",
                        "informationUri": "https://github.com/mithranm/vibelint",
                        "rules": list(rules.values())
                    }
                },
                "results": results
            }]
        }

        return json.dumps(sarif_output, separators=(',', ':'))

    def _severity_to_sarif_level(self, severity: Severity) -> str:
        """Convert vibelint severity to SARIF level."""
        mapping = {
            Severity.BLOCK: "error",
            Severity.WARN: "warning",
            Severity.INFO: "note",
            Severity.OFF: "none"
        }
        return mapping.get(severity, "warning")


# Built-in formatters
BUILTIN_FORMATTERS = {
    "human": HumanFormatter,
    "json": JsonFormatter,
    "sarif": SarifFormatter
}
