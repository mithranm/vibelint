"""Comprehensive reporting system for vibelint analysis results.

Provides structured report generation with granular verbosity levels,
artifact management, hyperlinked reports, and multiple output formatters
for different consumers (humans, CI/CD, GitHub, LLMs).

vibelint/src/vibelint/reporting.py
"""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from vibelint.validators import BaseFormatter, Finding, Severity

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "VerbosityLevel",
    "NaturalLanguageFormatter",
    "JsonFormatter",
    "SarifFormatter",
    "LLMFormatter",
    "HumanFormatter",
    "BUILTIN_FORMATTERS",
    "FORMAT_CHOICES",
    "DEFAULT_FORMAT",
    "ExecutiveSummary",
    "PriorityAction",
    "TreeViolation",
    "Synthesis",
    "AnalysisResults",
    "FileAnalysisEntry",
    "ContentAnalysis",
    "TreeAnalysis",
]


class VerbosityLevel(Enum):
    """Report verbosity levels for different use cases."""

    EXECUTIVE = "executive"  # High-level summary for planning
    TACTICAL = "tactical"  # Actionable items for development
    DETAILED = "detailed"  # Comprehensive analysis with context
    FORENSIC = "forensic"  # Complete diagnostic information


@dataclass
class ExecutiveSummary:
    """High-level summary of analysis results."""

    overall_health: str
    critical_issues: int
    improvement_opportunities: int
    estimated_effort: str


@dataclass
class PriorityAction:
    """Priority action item from analysis."""

    title: str
    priority: str
    description: str
    effort_hours: str
    risk_if_ignored: str


@dataclass
class TreeViolation:
    """Organizational/tree structure violation."""

    violation_type: str
    message: str


@dataclass
class Synthesis:
    """Synthesis of all analysis results."""

    executive_summary: ExecutiveSummary
    priority_actions: List[PriorityAction]
    quick_wins: List[str] = field(default_factory=list)


@dataclass
class TreeAnalysis:
    """Tree/organizational analysis results."""

    quick_violations: List[TreeViolation] = field(default_factory=list)


@dataclass
class FileAnalysisEntry:
    """Single file analysis entry with findings."""

    file_path: str
    findings: List[Finding] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Content/structural analysis results."""

    file_analyses: List[FileAnalysisEntry] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Complete analysis results container."""

    synthesis: Synthesis
    tree_analysis: Optional[TreeAnalysis] = None
    content_analysis: Optional[ContentAnalysis] = None
    deep_analysis: Optional[Dict[str, Any]] = None  # Keep as dict for flexibility


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Output settings
    output_directory: Path
    report_name: str = "vibelint_analysis"
    verbosity_level: VerbosityLevel = VerbosityLevel.TACTICAL

    # Format settings
    formats: List[str] = None  # ["markdown", "json", "html"]
    include_artifacts: bool = True
    create_index: bool = True

    # Content settings
    max_findings_per_category: int = 20
    include_raw_llm_responses: bool = False
    include_performance_metrics: bool = True

    # Navigation settings
    generate_hyperlinks: bool = True
    create_quick_nav: bool = True

    def __post_init__(self):
        if self.formats is None:
            self.formats = ["markdown", "html"]


class ReportGenerator:
    """Generates structured analysis reports with granular verbosity control."""

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_comprehensive_report(
        self, analysis_results: AnalysisResults, timestamp: Optional[str] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive report with all artifacts."""
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create report directory structure
        report_dir = self.config.output_directory / f"{self.config.report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Generate main report
        main_report_path = self._generate_main_report(analysis_results, report_dir, timestamp)
        generated_files["main_report"] = main_report_path

        # Generate artifacts if enabled
        if self.config.include_artifacts:
            artifact_paths = self._generate_artifacts(analysis_results, report_dir, timestamp)
            generated_files.update(artifact_paths)

        # Generate index/navigation if enabled
        if self.config.create_index:
            index_path = self._generate_index(report_dir, generated_files, timestamp)
            generated_files["index"] = index_path

        # Generate quick action plan
        quick_plan_path = self._generate_quick_action_plan(analysis_results, report_dir, timestamp)
        generated_files["quick_plan"] = quick_plan_path

        return generated_files

    def _generate_main_report(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Path:
        """Generate main analysis report."""
        # Filter content based on verbosity level
        filtered_results = self._filter_by_verbosity(analysis_results)

        # Generate markdown content
        content = self._format_main_report_markdown(filtered_results, timestamp)

        # Save to file
        report_path = report_dir / "main_report.md"
        report_path.write_text(content, encoding="utf-8")

        return report_path

    def _format_main_report_markdown(
        self, analysis_results: AnalysisResults, timestamp: str
    ) -> str:
        """Format main report as markdown."""
        executive = analysis_results.synthesis.executive_summary
        priority_actions = analysis_results.synthesis.priority_actions

        content = f"""# Vibelint Analysis Report

Generated: {timestamp}
Verbosity Level: {self.config.verbosity_level.value}

## Executive Summary

- **Overall Health**: {executive.overall_health}
- **Critical Issues**: {executive.critical_issues}
- **Improvement Opportunities**: {executive.improvement_opportunities}
- **Estimated Effort**: {executive.estimated_effort}

## Priority Actions

"""

        for i, action in enumerate(priority_actions[:5], 1):
            content += f"""### {i}. {action.title} ({action.priority})

{action.description}

**Effort**: {action.effort_hours} hours
**Risk if ignored**: {action.risk_if_ignored}

"""

        # Add findings summary based on verbosity
        if self.config.verbosity_level != VerbosityLevel.EXECUTIVE:
            content += self._add_findings_section(analysis_results)

        return content

    def _add_findings_section(self, analysis_results: AnalysisResults) -> str:
        """Add findings section based on verbosity level."""
        content = "\n## Findings Summary\n\n"

        # Tree violations
        if analysis_results.tree_analysis:
            tree_violations = analysis_results.tree_analysis.quick_violations
            if tree_violations:
                content += f"### Organizational Issues ({len(tree_violations)})\n\n"
                for violation in tree_violations[: self.config.max_findings_per_category]:
                    content += f"- **{violation.violation_type}**: {violation.message}\n"
                content += "\n"

        # Content findings
        if analysis_results.content_analysis:
            content_findings = []
            for file_analysis in analysis_results.content_analysis.file_analyses:
                content_findings.extend(file_analysis.findings)

            if content_findings:
                content += f"### Structural Issues ({len(content_findings)})\n\n"
                for finding in content_findings[: self.config.max_findings_per_category]:
                    content += f"- **{finding.rule_id}**: {finding.message}\n"
                content += "\n"

        return content

    def _generate_artifacts(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Dict[str, Path]:
        """Generate detailed artifacts for different analysis aspects."""
        artifacts_dir = report_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        artifact_paths = {}

        # Generate JSON artifacts for each analysis level
        if analysis_results.tree_analysis:
            tree_path = artifacts_dir / "organizational_analysis.json"
            # Convert dataclass to dict for JSON serialization using asdict()
            tree_data = {
                "quick_violations": [
                    asdict(v) for v in analysis_results.tree_analysis.quick_violations
                ]
            }
            tree_path.write_text(
                json.dumps(tree_data, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["organizational"] = tree_path

        if analysis_results.content_analysis:
            content_path = artifacts_dir / "structural_analysis.json"
            # Convert dataclass to dict for JSON serialization using asdict()
            content_data = {
                "file_analyses": [
                    {
                        "file_path": fa.file_path,
                        "findings": [asdict(f) for f in fa.findings],
                    }
                    for fa in analysis_results.content_analysis.file_analyses
                ]
            }
            content_path.write_text(
                json.dumps(content_data, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["structural"] = content_path

        if analysis_results.deep_analysis:
            arch_path = artifacts_dir / "architectural_analysis.json"
            arch_path.write_text(
                json.dumps(analysis_results.deep_analysis, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["architectural"] = arch_path

        return artifact_paths

    def _generate_quick_action_plan(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Path:
        """Generate quick action plan for immediate development focus."""
        synthesis = analysis_results.synthesis

        quick_plan = f"""# Quick Action Plan
Generated: {timestamp}

## Immediate Actions (< 1 hour each)

"""

        # Add quick wins
        quick_wins = synthesis.quick_wins
        for i, win in enumerate(quick_wins[:5], 1):
            quick_plan += f"{i}. {win}\n"

        quick_plan += "\n## Priority Issues (requires planning)\n\n"

        # Add priority actions
        priority_actions = synthesis.priority_actions
        for action in priority_actions[:3]:
            quick_plan += f"### {action.title} ({action.priority})\n"
            quick_plan += f"**Effort**: {action.effort_hours} hours\n"
            quick_plan += f"**Description**: {action.description}\n\n"

        quick_plan_path = report_dir / "QUICK_ACTION_PLAN.md"
        quick_plan_path.write_text(quick_plan, encoding="utf-8")

        return quick_plan_path

    def _generate_index(
        self, report_dir: Path, generated_files: Dict[str, Path], timestamp: str
    ) -> Path:
        """Generate navigation index for the report."""
        index_content = f"""# Vibelint Analysis Report Index
Generated: {timestamp}

## Main Reports

- [[REPORT] Main Analysis Report](main_report.md)
- [[ROCKET] Quick Action Plan](QUICK_ACTION_PLAN.md)

## Detailed Artifacts

"""

        # Add artifact links
        artifact_types = {
            "organizational": "[BUILD] Organizational Analysis",
            "structural": "[TOOL] Structural Analysis",
            "architectural": "[ARCH] Architectural Analysis",
        }

        for artifact_key, description in artifact_types.items():
            if artifact_key in generated_files:
                artifact_path = generated_files[artifact_key]
                relative_path = f"artifacts/{artifact_path.name}"
                index_content += f"- [{description}]({relative_path})\n"

        index_content += f"\n---\n\nReport generated by vibelint at {timestamp}\n"

        index_path = report_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")

        return index_path

    def _filter_by_verbosity(self, analysis_results: AnalysisResults) -> AnalysisResults:
        """Filter analysis results based on configured verbosity level."""
        if self.config.verbosity_level == VerbosityLevel.EXECUTIVE:
            # Only high-level summary and critical issues
            critical_actions = self._extract_critical_issues(analysis_results)
            return AnalysisResults(
                synthesis=Synthesis(
                    executive_summary=analysis_results.synthesis.executive_summary,
                    priority_actions=critical_actions,
                    quick_wins=[],
                )
            )

        elif self.config.verbosity_level == VerbosityLevel.TACTICAL:
            # Actionable items and priority information
            # Limit findings per category
            limited_content = None
            if analysis_results.content_analysis:
                limited_content = self._limit_content_findings(analysis_results.content_analysis)

            return AnalysisResults(
                synthesis=analysis_results.synthesis,
                tree_analysis=analysis_results.tree_analysis,
                content_analysis=limited_content,
                deep_analysis=analysis_results.deep_analysis,
            )

        else:  # DETAILED or FORENSIC
            # Most or all information
            return analysis_results

    def _extract_critical_issues(self, analysis_results: AnalysisResults) -> List[PriorityAction]:
        """Extract only critical/blocking issues for executive summary."""
        critical_issues = []

        # Check synthesis for critical items
        for action in analysis_results.synthesis.priority_actions:
            if action.priority in ["P0", "P1"]:
                critical_issues.append(action)

        return critical_issues

    def _limit_content_findings(self, content_analysis: ContentAnalysis) -> ContentAnalysis:
        """Limit content findings for tactical verbosity."""
        limited_file_analyses = []

        for file_analysis in content_analysis.file_analyses:
            # Keep only high-severity findings for tactical view
            high_severity = [
                f for f in file_analysis.findings if f.severity in [Severity.BLOCK, Severity.WARN]
            ]
            limited_file_analyses.append(
                FileAnalysisEntry(
                    file_path=file_analysis.file_path, findings=high_severity[:5]  # Limit to 5
                )
            )

        return ContentAnalysis(file_analyses=limited_file_analyses)


# ===== FORMATTERS =====
# Consolidated from formatters.py


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
        # Config can be a dict (external API data) - .get() is acceptable here
        if config and isinstance(config, dict):
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
        # Mapping dict - .get() is legitimate for safe enum->string conversion
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
