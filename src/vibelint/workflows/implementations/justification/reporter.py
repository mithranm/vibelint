"""
Reporting and quality gate module for justification workflow.

Handles report generation, file output, and quality gate enforcement.
Separated for cleaner testing and configuration management.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JustificationReporter:
    """Handles reporting and quality gate enforcement for justification analysis."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.reports_base_dir = Path(".vibelint-reports")

    def generate_justification_report(self, enhanced_tree: str, dependency_analysis: Dict[str, Any],
                                    directory_path: Path, llm_logs: List[Dict] = None) -> str:
        """Generate the comprehensive justification report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Project Justification Analysis
**Project:** {directory_path.name}
**Generated:** {timestamp}
**Analysis Type:** Comprehensive (Static + Dependency + LLM)

---

## Project Structure with Analysis

{enhanced_tree}

---

## Dependency Analysis Summary

**Total Files Analyzed:** {len(dependency_analysis.get('files_analyzed', []))}
**Entry Points Found:** {len(dependency_analysis.get('entry_points', []))}
**Circular Dependencies:** {len(dependency_analysis.get('circular_imports', []))}

### Entry Points
{self._format_entry_points(dependency_analysis.get('entry_points', []))}

### Circular Dependencies
{self._format_circular_dependencies(dependency_analysis.get('circular_imports', []))}

---

## Analysis Methodology

- **Static Analysis:** ✅ AST parsing for structure, imports, complexity
- **Dependency Tracing:** ✅ From entry points using import resolution
- **LLM Enhancement:** ✅ File purpose analysis via fast LLM calls
- **Backup Detection:** ✅ Intelligent file pattern recognition
- **Architectural Analysis:** ✅ Via LLM orchestrator

## Methodology

This analysis combines:
1. **File system discovery** for project structure
2. **Fast LLM** (750 token limit) for file purpose summaries
3. **AST parsing** for deterministic import analysis
4. **Graph analysis** for circular dependency detection
All analysis is logged and reproducible.
"""

        if llm_logs:
            report += f"\n\n---\n\n## LLM Analysis Logs\n\n"
            report += f"**Total LLM Calls:** {len(llm_logs)}\n\n"
            for log in llm_logs[-5:]:  # Show last 5 calls
                operation = log.get('operation') if 'operation' in log else 'unknown'
                timestamp = log.get('timestamp') if 'timestamp' in log else 'unknown'
                report += f"- **{operation}** at {timestamp}\n"

        return report

    def _format_entry_points(self, entry_points: List[str]) -> str:
        """Format entry points list for report."""
        if not entry_points:
            return "No entry points detected."

        formatted = ""
        for ep in entry_points[:10]:  # Limit to first 10
            formatted += f"- `{ep}`\n"

        if len(entry_points) > 10:
            formatted += f"- ... and {len(entry_points) - 10} more\n"

        return formatted

    def _format_circular_dependencies(self, circular_deps: List[List[str]]) -> str:
        """Format circular dependencies for report."""
        if not circular_deps:
            return "✅ No circular dependencies detected."

        formatted = "⚠️ **Circular dependencies found:**\n\n"
        for i, cycle in enumerate(circular_deps[:5], 1):  # Limit to first 5
            formatted += f"{i}. {' → '.join(cycle)}\n"

        if len(circular_deps) > 5:
            formatted += f"\n... and {len(circular_deps) - 5} more cycles\n"

        return formatted

    def save_justification_report(self, report: str, directory_path: Path, analysis_id: str = None) -> Path:
        """Save the justification workflow results to file."""
        if not analysis_id:
            analysis_id = str(int(time.time()))

        # Create reports directory
        reports_dir = self._get_reports_directory("justification_workflow")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save main report
        report_file = reports_dir / f"justification_workflow_{analysis_id}.md"
        report_file.write_text(report)
        logger.info(f"Saved justification report: {report_file}")

        return report_file

    def save_final_analysis(self, final_analysis: str, analysis_id: str = None) -> Path:
        """Save the final LLM analysis to a separate file."""
        if not analysis_id:
            analysis_id = str(int(time.time()))

        reports_dir = self._get_reports_directory("justification_workflow")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp-based filename for final analysis
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_file = reports_dir / f"architectural_analysis__{timestamp}_FINAL_ANALYSIS.md"

        # Create structured final analysis
        structured_analysis = f"""# Vibelint Final Analysis Report

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Type:** Orchestrator LLM Comprehensive Review
**Initial Report:** justification_workflow_{analysis_id}.md

---

{final_analysis}

---

## Analysis Metadata

**LLM Calls Made:** Multiple (see logs for details)
**Analysis Depth:** Comprehensive (Static + Dependency + LLM)
**Report Generation:** Automated via JustificationEngineV2

For detailed file-by-file analysis, see the initial report: `justification_workflow_{analysis_id}.md`
"""

        final_file.write_text(structured_analysis)
        logger.info(f"Saved final analysis: {final_file}")

        return final_file

    def save_logs(self, llm_logs: List[Dict], analysis_id: str = None) -> Path:
        """Save LLM interaction logs as JSON."""
        if not analysis_id:
            analysis_id = str(int(time.time()))

        reports_dir = self._get_reports_directory("justification_workflow")
        reports_dir.mkdir(parents=True, exist_ok=True)

        log_file = reports_dir / f"justification_workflow_{analysis_id}_logs.json"

        log_data = {
            "analysis_id": analysis_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_calls": len(llm_logs),
            "calls": llm_logs
        }

        log_file.write_text(json.dumps(log_data, indent=2))
        logger.debug(f"Saved LLM logs: {log_file}")

        return log_file

    def check_quality_gate(self, final_analysis: str) -> bool:
        """Check if quality gate passes based on final analysis."""
        if not final_analysis:
            return False

        # Simple quality gate: look for LGTM in the final analysis
        lgtm_indicators = [
            "LGTM",
            "no significant issues",
            "well-structured",
            "no major problems",
            "architecture is sound"
        ]

        analysis_lower = final_analysis.lower()
        return any(indicator.lower() in analysis_lower for indicator in lgtm_indicators)

    def enforce_quality_gate(self, final_analysis: str, safe_mode: bool = False) -> Dict[str, Any]:
        """Enforce quality gate and return result with exit code."""
        gate_passed = self.check_quality_gate(final_analysis)

        result = {
            "gate_passed": gate_passed,
            "exit_code": 0 if gate_passed else 2,  # 2 for quality failure, not runtime error
            "message": "Quality gate PASSED" if gate_passed else "Quality gate FAILED: No LGTM found in final analysis"
        }

        if not safe_mode:
            # In normal mode, log the result
            if gate_passed:
                logger.info("✅ Quality gate PASSED")
            else:
                logger.warning("❌ Quality gate FAILED: No LGTM found in final analysis")

        return result

    def _get_reports_directory(self, subdirectory: str = None) -> Path:
        """Get the reports directory, creating if necessary."""
        if subdirectory:
            return self.reports_base_dir / subdirectory
        return self.reports_base_dir

    def create_quality_assessment_report(self, final_analysis: str, analysis_id: str = None) -> Path:
        """Create a separate quality assessment report with structured findings."""
        if not analysis_id:
            analysis_id = str(int(time.time()))

        reports_dir = self._get_reports_directory("justification_workflow")
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        assessment_file = reports_dir / f"architectural_analysis__{timestamp}_QUALITY_ASSESSMENT.md"

        assessment_content = final_analysis

        assessment_file.write_text(assessment_content)
        logger.info(f"Saved quality assessment: {assessment_file}")

        return assessment_file

    def generate_summary_report(self, total_files: int, analysis_time: float,
                               gate_result: Dict[str, Any], report_files: List[Path]) -> str:
        """Generate a summary of the analysis run."""
        return f"""# Justification Analysis Summary

**Files Analyzed:** {total_files}
**Analysis Time:** {analysis_time:.1f} seconds
**Quality Gate:** {"✅ PASSED" if gate_result.get("gate_passed") else "❌ FAILED"}

## Generated Reports

{chr(10).join(f"- {f.name}" for f in report_files)}

## Next Steps

{"Review and address any issues found in the reports above." if not gate_result.get("gate_passed") else "No critical issues found. Consider reviewing recommendations for further improvements."}
"""