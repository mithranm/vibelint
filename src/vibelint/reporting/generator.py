"""
Report generator with granular verbosity levels and artifact management.

Creates structured analysis reports with hyperlinked artifacts that allow
developers to focus on specific issues and plan development work effectively.

vibelint/src/vibelint/reporting/generator.py
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from .artifacts import ArtifactManager, ArtifactType
from .formats import MarkdownFormatter, JSONFormatter, HTMLFormatter

__all__ = ["ReportGenerator", "ReportConfig", "VerbosityLevel"]


class VerbosityLevel(Enum):
    """Report verbosity levels for different use cases."""

    EXECUTIVE = "executive"    # High-level summary for planning
    TACTICAL = "tactical"      # Actionable items for development
    DETAILED = "detailed"      # Comprehensive analysis with context
    FORENSIC = "forensic"      # Complete diagnostic information


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
        self.artifact_manager = ArtifactManager(config.output_directory)

        # Initialize formatters
        self.formatters = {
            "markdown": MarkdownFormatter(),
            "json": JSONFormatter(),
            "html": HTMLFormatter()
        }

    def generate_comprehensive_report(
        self,
        analysis_results: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive report with all artifacts."""

        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create report directory structure
        report_dir = self.config.output_directory / f"{self.config.report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Generate main report in each requested format
        for format_name in self.config.formats:
            if format_name in self.formatters:
                formatter = self.formatters[format_name]
                main_report_path = self._generate_main_report(
                    analysis_results, report_dir, formatter, timestamp
                )
                generated_files[f"main_{format_name}"] = main_report_path

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
        self,
        analysis_results: Dict[str, Any],
        report_dir: Path,
        formatter,
        timestamp: str
    ) -> Path:
        """Generate main analysis report."""

        # Filter content based on verbosity level
        filtered_results = self._filter_by_verbosity(analysis_results)

        # Generate report content
        content = formatter.format_main_report(
            filtered_results,
            self.config,
            timestamp
        )

        # Save to file
        extension = formatter.get_file_extension()
        report_path = report_dir / f"main_report.{extension}"
        report_path.write_text(content, encoding="utf-8")

        return report_path

    def _generate_artifacts(
        self,
        analysis_results: Dict[str, Any],
        report_dir: Path,
        timestamp: str
    ) -> Dict[str, Path]:
        """Generate detailed artifacts for different analysis aspects."""

        artifacts_dir = report_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        artifact_paths = {}

        # Tree-level organizational artifacts
        if "tree_analysis" in analysis_results:
            tree_artifact = self.artifact_manager.create_artifact(
                ArtifactType.ORGANIZATIONAL,
                analysis_results["tree_analysis"],
                artifacts_dir / "organizational_analysis.json"
            )
            artifact_paths["organizational"] = tree_artifact

        # Content-level structural artifacts
        if "content_analysis" in analysis_results:
            content_artifact = self.artifact_manager.create_artifact(
                ArtifactType.STRUCTURAL,
                analysis_results["content_analysis"],
                artifacts_dir / "structural_analysis.json"
            )
            artifact_paths["structural"] = content_artifact

        # Deep architectural artifacts
        if "deep_analysis" in analysis_results:
            arch_artifact = self.artifact_manager.create_artifact(
                ArtifactType.ARCHITECTURAL,
                analysis_results["deep_analysis"],
                artifacts_dir / "architectural_analysis.json"
            )
            artifact_paths["architectural"] = arch_artifact

        # Performance metrics artifacts
        if self.config.include_performance_metrics and "performance_data" in analysis_results:
            perf_artifact = self.artifact_manager.create_artifact(
                ArtifactType.METRICS,
                analysis_results["performance_data"],
                artifacts_dir / "performance_metrics.json"
            )
            artifact_paths["performance"] = perf_artifact

        # Raw LLM responses if requested
        if self.config.include_raw_llm_responses and "llm_responses" in analysis_results:
            llm_artifact = self.artifact_manager.create_artifact(
                ArtifactType.RAW_DATA,
                analysis_results["llm_responses"],
                artifacts_dir / "llm_responses.json"
            )
            artifact_paths["llm_responses"] = llm_artifact

        # Findings by category
        self._generate_findings_artifacts(analysis_results, artifacts_dir, artifact_paths)

        return artifact_paths

    def _generate_findings_artifacts(
        self,
        analysis_results: Dict[str, Any],
        artifacts_dir: Path,
        artifact_paths: Dict[str, Path]
    ):
        """Generate separate artifacts for different finding categories."""

        findings_dir = artifacts_dir / "findings"
        findings_dir.mkdir(exist_ok=True)

        # Collect all findings by category
        findings_by_category = {}

        # Tree violations
        tree_violations = analysis_results.get("tree_analysis", {}).get("quick_violations", [])
        if tree_violations:
            findings_by_category["organizational_violations"] = tree_violations

        # Content findings
        content_findings = []
        for file_analysis in analysis_results.get("content_analysis", {}).get("file_analyses", []):
            findings = file_analysis.get("analysis", {}).get("findings", [])
            content_findings.extend(findings)
        if content_findings:
            findings_by_category["structural_issues"] = content_findings

        # Architectural findings
        arch_findings = analysis_results.get("deep_analysis", {}).get("architectural_analysis", {})
        if arch_findings.get("architectural_findings"):
            findings_by_category["architectural_issues"] = arch_findings["architectural_findings"]
        if arch_findings.get("code_smells"):
            findings_by_category["code_smells"] = arch_findings["code_smells"]

        # Generate artifact for each category
        for category, findings in findings_by_category.items():
            if findings:
                # Limit findings per category if configured
                limited_findings = findings[:self.config.max_findings_per_category]

                artifact_path = self.artifact_manager.create_artifact(
                    ArtifactType.FINDINGS,
                    {
                        "category": category,
                        "findings": limited_findings,
                        "total_count": len(findings),
                        "truncated": len(findings) > self.config.max_findings_per_category
                    },
                    findings_dir / f"{category}.json"
                )
                artifact_paths[f"findings_{category}"] = artifact_path

    def _generate_index(
        self,
        report_dir: Path,
        generated_files: Dict[str, Path],
        timestamp: str
    ) -> Path:
        """Generate navigation index for the report."""

        index_content = self._create_index_content(generated_files, timestamp)
        index_path = report_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")

        return index_path

    def _generate_quick_action_plan(
        self,
        analysis_results: Dict[str, Any],
        report_dir: Path,
        timestamp: str
    ) -> Path:
        """Generate quick action plan for immediate development focus."""

        synthesis = analysis_results.get("synthesis", {})

        quick_plan = f"""# Quick Action Plan
Generated: {timestamp}

## Immediate Actions (< 1 hour each)

"""

        # Add quick wins
        quick_wins = synthesis.get("quick_wins", [])
        for i, win in enumerate(quick_wins[:5], 1):
            quick_plan += f"{i}. {win}\n"

        quick_plan += "\n## Priority Issues (requires planning)\n\n"

        # Add priority actions
        priority_actions = synthesis.get("priority_actions", [])
        for action in priority_actions[:3]:
            title = action.get("title", "Unknown action")
            priority = action.get("priority", "P?")
            effort = action.get("effort_hours", "?")

            quick_plan += f"### {title} ({priority})\n"
            quick_plan += f"**Effort**: {effort} hours\n"
            quick_plan += f"**Description**: {action.get('description', 'No description')}\n\n"

        quick_plan += "\n## Next Review Triggers\n\n"

        # Add review triggers
        triggers = analysis_results.get("next_review_triggers", [])
        for trigger in triggers:
            quick_plan += f"- {trigger}\n"

        quick_plan_path = report_dir / "QUICK_ACTION_PLAN.md"
        quick_plan_path.write_text(quick_plan, encoding="utf-8")

        return quick_plan_path

    def _filter_by_verbosity(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter analysis results based on configured verbosity level."""

        if self.config.verbosity_level == VerbosityLevel.EXECUTIVE:
            # Only high-level summary and critical issues
            return {
                "executive_summary": analysis_results.get("synthesis", {}).get("executive_summary", {}),
                "critical_issues": self._extract_critical_issues(analysis_results),
                "strategic_recommendations": analysis_results.get("synthesis", {}).get("strategic_initiatives", [])
            }

        elif self.config.verbosity_level == VerbosityLevel.TACTICAL:
            # Actionable items and priority information
            filtered = analysis_results.copy()

            # Limit findings per category
            if "content_analysis" in filtered:
                filtered["content_analysis"] = self._limit_content_findings(
                    filtered["content_analysis"]
                )

            return filtered

        elif self.config.verbosity_level == VerbosityLevel.DETAILED:
            # Most information but still organized
            return analysis_results

        else:  # FORENSIC
            # Everything including raw data
            return analysis_results

    def _extract_critical_issues(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract only critical/blocking issues for executive summary."""
        critical_issues = []

        # Check synthesis for critical items
        synthesis = analysis_results.get("synthesis", {})
        for action in synthesis.get("priority_actions", []):
            if action.get("priority") in ["P0", "P1"]:
                critical_issues.append(action)

        return critical_issues

    def _limit_content_findings(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Limit content findings for tactical verbosity."""
        limited = content_analysis.copy()

        if "file_analyses" in limited:
            for file_analysis in limited["file_analyses"]:
                if "analysis" in file_analysis and "findings" in file_analysis["analysis"]:
                    findings = file_analysis["analysis"]["findings"]
                    # Keep only high-severity findings for tactical view
                    high_severity = [f for f in findings if f.get("severity") in ["BLOCK", "WARN"]]
                    file_analysis["analysis"]["findings"] = high_severity[:5]  # Limit to 5 per file

        return limited

    def _create_index_content(self, generated_files: Dict[str, Path], timestamp: str) -> str:
        """Create navigation index content."""

        index = f"""# Vibelint Analysis Report Index
Generated: {timestamp}

## Main Reports

"""

        # Link to main reports
        for file_type, file_path in generated_files.items():
            if file_type.startswith("main_"):
                format_name = file_type.replace("main_", "")
                relative_path = file_path.name
                index += f"- [{format_name.upper()} Report]({relative_path})\n"

        index += "\n## Quick Navigation\n\n"

        # Quick action plan
        if "quick_plan" in generated_files:
            index += f"- [🚀 Quick Action Plan]({generated_files['quick_plan'].name})\n"

        index += "\n## Detailed Artifacts\n\n"

        # Artifacts
        artifact_types = {
            "organizational": "🏗️ Organizational Analysis",
            "structural": "🔧 Structural Analysis",
            "architectural": "🏛️ Architectural Analysis",
            "performance": "📊 Performance Metrics"
        }

        for artifact_key, description in artifact_types.items():
            if artifact_key in generated_files:
                relative_path = f"artifacts/{generated_files[artifact_key].name}"
                index += f"- [{description}]({relative_path})\n"

        index += "\n## Findings by Category\n\n"

        # Findings artifacts
        findings_categories = {
            "findings_organizational_violations": "📋 Organizational Violations",
            "findings_structural_issues": "⚙️ Structural Issues",
            "findings_architectural_issues": "🏗️ Architectural Issues",
            "findings_code_smells": "👃 Code Smells"
        }

        for finding_key, description in findings_categories.items():
            if finding_key in generated_files:
                relative_path = f"artifacts/findings/{generated_files[finding_key].name}"
                index += f"- [{description}]({relative_path})\n"

        index += "\n---\n\n"
        index += f"Report generated by vibelint at {timestamp}\n"

        return index