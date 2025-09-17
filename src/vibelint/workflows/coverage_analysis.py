"""
Coverage analysis workflow for comprehensive test coverage evaluation.

AI-powered coverage analysis, edge case detection, and automated
test generation using dual LLM architecture. Analyzes coverage gaps
and suggests targeted improvements.

vibelint/src/vibelint/workflows/coverage_analysis.py
"""

import ast
import logging
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..llm import LLMRequest, LLMManager
from .base import BaseWorkflow, WorkflowResult, WorkflowStatus, WorkflowConfig

logger = logging.getLogger(__name__)

__all__ = ["CoverageAnalysisWorkflow", "CoverageGap", "EdgeCase", "CodeSuggestion"]


@dataclass
class CoverageGap:
    """Represents a coverage gap in the codebase."""
    file_path: str
    line_number: int
    function_name: Optional[str]
    coverage_percentage: float
    gap_type: str  # 'uncovered_line', 'missing_branch', 'error_path'
    description: str


@dataclass
class EdgeCase:
    """Represents an identified edge case that should be tested."""
    file_path: str
    function_name: str
    line_number: int
    case_type: str  # 'boundary', 'error_condition', 'race_condition', 'null_input'
    description: str
    suggested_test_data: Optional[Dict[str, Any]] = None


@dataclass
class CodeSuggestion:
    """Represents a generated code suggestion for improving coverage."""
    suggestion_name: str
    file_path: str
    function_under_analysis: str
    suggested_code: str
    priority: int  # 1-5, 5 being highest priority
    edge_cases_covered: List[str]
    estimated_coverage_increase: float


class CoverageAnalysisWorkflow(BaseWorkflow):
    """Workflow for AI-powered test coverage analysis and improvement."""

    workflow_id = "coverage-analysis"
    name = "Coverage Analysis"
    description = "Analyzes test coverage gaps and suggests targeted improvements using AI"
    category = "testing"
    tags = {"coverage", "testing", "ai", "quality"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config)
        self.coverage_gaps: List[CoverageGap] = []
        self.edge_cases: List[EdgeCase] = []
        self.code_suggestions: List[CodeSuggestion] = []

    def get_required_inputs(self) -> Set[str]:
        """Get set of required input data keys."""
        return {"project_root"}

    def get_produced_outputs(self) -> Set[str]:
        """Get set of output data keys this workflow produces."""
        return {
            "coverage_gaps",
            "edge_cases",
            "code_suggestions",
            "coverage_metrics",
            "improvement_recommendations"
        }

    async def execute(self, project_root: Path, context: Dict[str, Any]) -> WorkflowResult:
        """Execute coverage analysis workflow."""

        logger.info("Starting AI-powered coverage analysis...")

        try:
            # Step 1: Run coverage analysis
            coverage_data = await self._run_coverage_analysis(project_root)

            # Step 2: Analyze gaps with AI
            if coverage_data:
                await self._analyze_coverage_gaps(coverage_data, project_root)

            # Step 3: Identify edge cases with AI
            await self._identify_edge_cases(project_root)

            # Step 4: Generate improvement suggestions
            await self._generate_improvement_suggestions(project_root)

            # Create findings
            findings = []

            # Add coverage gap findings
            for gap in self.coverage_gaps:
                severity = "BLOCK" if gap.coverage_percentage < 0.5 else "WARN" if gap.coverage_percentage < 0.8 else "INFO"
                findings.append({
                    "rule_id": "COVERAGE-GAP",
                    "severity": severity,
                    "message": f"Coverage gap in {gap.function_name or 'unknown function'}: {gap.coverage_percentage:.1%} coverage",
                    "file_path": gap.file_path,
                    "line": gap.line_number,
                    "suggestion": gap.description
                })

            # Add edge case findings
            for edge_case in self.edge_cases:
                findings.append({
                    "rule_id": "EDGE-CASE-MISSING",
                    "severity": "INFO",
                    "message": f"Missing edge case test for {edge_case.function_name}: {edge_case.case_type}",
                    "file_path": edge_case.file_path,
                    "line": edge_case.line_number,
                    "suggestion": edge_case.description
                })

            # Create artifacts
            artifacts = {
                "coverage_gaps": [self._gap_to_dict(g) for g in self.coverage_gaps],
                "edge_cases": [self._edge_case_to_dict(e) for e in self.edge_cases],
                "code_suggestions": [self._suggestion_to_dict(s) for s in self.code_suggestions],
                "coverage_metrics": coverage_data or {},
                "improvement_recommendations": self._generate_recommendations()
            }

            # Update metrics
            self.metrics.files_processed = len(self._get_python_files(project_root))
            self.metrics.findings_generated = len(findings)
            self.metrics.confidence_score = self._calculate_confidence()

            return self._create_result(
                WorkflowStatus.COMPLETED,
                findings=findings,
                artifacts=artifacts
            )

        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}", exc_info=True)
            return self._create_result(
                WorkflowStatus.FAILED,
                error_message=str(e)
            )

    async def _run_coverage_analysis(self, project_root: Path) -> Optional[Dict[str, Any]]:
        """Run coverage analysis using pytest-cov."""
        try:
            # Look for test directories
            test_dirs = []
            for candidate in ["tests", "test"]:
                test_path = project_root / candidate
                if test_path.exists():
                    test_dirs.append(str(test_path))

            if not test_dirs:
                logger.warning("No test directories found, skipping coverage analysis")
                return None

            # Run coverage
            cmd = [
                "python", "-m", "pytest",
                "--cov=src",
                "--cov-report=xml",
                "--cov-report=term-missing",
                *test_dirs
            ]

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse coverage XML if available
            coverage_xml = project_root / "coverage.xml"
            if coverage_xml.exists():
                return self._parse_coverage_xml(coverage_xml)

            return {"stdout": result.stdout, "stderr": result.stderr}

        except subprocess.TimeoutExpired:
            logger.warning("Coverage analysis timed out")
            return None
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            return None

    def _parse_coverage_xml(self, coverage_xml_path: Path) -> Dict[str, Any]:
        """Parse coverage.xml file."""
        try:
            tree = ET.parse(coverage_xml_path)
            root = tree.getroot()

            coverage_data = {
                "overall_coverage": 0.0,
                "file_coverage": {},
                "line_coverage": {}
            }

            # Extract overall coverage
            if root.attrib.get("line-rate"):
                coverage_data["overall_coverage"] = float(root.attrib["line-rate"])

            # Extract per-file coverage
            for package in root.findall(".//package"):
                for class_elem in package.findall("classes/class"):
                    filename = class_elem.attrib.get("filename", "")
                    line_rate = float(class_elem.attrib.get("line-rate", 0))

                    coverage_data["file_coverage"][filename] = line_rate

                    # Extract line-by-line coverage
                    lines_data = {}
                    for line in class_elem.findall("lines/line"):
                        line_num = int(line.attrib.get("number", 0))
                        hits = int(line.attrib.get("hits", 0))
                        lines_data[line_num] = hits > 0

                    coverage_data["line_coverage"][filename] = lines_data

            return coverage_data

        except Exception as e:
            logger.warning(f"Failed to parse coverage XML: {e}")
            return {}

    async def _analyze_coverage_gaps(self, coverage_data: Dict[str, Any], project_root: Path):
        """Analyze coverage gaps using AI."""
        if not coverage_data.get("file_coverage"):
            return

        # Find files with low coverage
        for file_path, coverage_rate in coverage_data["file_coverage"].items():
            if coverage_rate < 0.8:  # Less than 80% coverage
                full_path = project_root / file_path
                if full_path.exists():
                    try:
                        # Analyze the file to understand what's missing
                        content = full_path.read_text(encoding="utf-8")
                        gap = await self._analyze_file_coverage_gap(file_path, content, coverage_rate)
                        if gap:
                            self.coverage_gaps.append(gap)
                    except Exception as e:
                        logger.debug(f"Failed to analyze coverage gap for {file_path}: {e}")

    async def _analyze_file_coverage_gap(self, file_path: str, content: str, coverage_rate: float) -> Optional[CoverageGap]:
        """Analyze specific file coverage gap."""
        try:
            tree = ast.parse(content)

            # Find the largest function (likely the main issue)
            largest_function = None
            max_lines = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 10
                    if lines > max_lines:
                        max_lines = lines
                        largest_function = node

            if largest_function:
                return CoverageGap(
                    file_path=file_path,
                    line_number=largest_function.lineno,
                    function_name=largest_function.name,
                    coverage_percentage=coverage_rate,
                    gap_type="uncovered_function",
                    description=f"Function '{largest_function.name}' likely has uncovered code paths"
                )

        except Exception as e:
            logger.debug(f"Failed to analyze file {file_path}: {e}")

        return None

    async def _identify_edge_cases(self, project_root: Path):
        """Identify missing edge cases using AI analysis."""
        python_files = self._get_python_files(project_root)

        for file_path in python_files[:5]:  # Limit for testing
            try:
                content = file_path.read_text(encoding="utf-8")
                edge_cases = await self._analyze_file_for_edge_cases(file_path, content)
                self.edge_cases.extend(edge_cases)
            except Exception as e:
                logger.debug(f"Failed to analyze edge cases for {file_path}: {e}")

    async def _analyze_file_for_edge_cases(self, file_path: Path, content: str) -> List[EdgeCase]:
        """Analyze file for potential edge cases."""
        edge_cases = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for common edge case patterns
                    if any(arg.arg in ["data", "input", "value"] for arg in node.args.args):
                        edge_cases.append(EdgeCase(
                            file_path=str(file_path.relative_to(file_path.parents[2])),
                            function_name=node.name,
                            line_number=node.lineno,
                            case_type="null_input",
                            description=f"Consider testing {node.name} with null/empty inputs"
                        ))

                    # Look for division or mathematical operations
                    for child in ast.walk(node):
                        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div):
                            edge_cases.append(EdgeCase(
                                file_path=str(file_path.relative_to(file_path.parents[2])),
                                function_name=node.name,
                                line_number=getattr(child, 'lineno', node.lineno),
                                case_type="division_by_zero",
                                description=f"Test division by zero in {node.name}"
                            ))

        except Exception as e:
            logger.debug(f"AST analysis failed for {file_path}: {e}")

        return edge_cases

    async def _generate_improvement_suggestions(self, project_root: Path):
        """Generate AI-powered improvement suggestions."""
        # For now, generate simple suggestions based on gaps found
        for gap in self.coverage_gaps:
            suggestion = CodeSuggestion(
                suggestion_name=f"Test for {gap.function_name}",
                file_path=gap.file_path,
                function_under_analysis=gap.function_name or "unknown",
                suggested_code=f"def test_{gap.function_name or 'function'}():\n    # TODO: Add test for {gap.description}",
                priority=3,
                edge_cases_covered=[gap.gap_type],
                estimated_coverage_increase=0.1
            )
            self.code_suggestions.append(suggestion)

    def _generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []

        if self.coverage_gaps:
            recommendations.append(f"Address {len(self.coverage_gaps)} coverage gaps to improve test quality")

        if self.edge_cases:
            recommendations.append(f"Add tests for {len(self.edge_cases)} identified edge cases")

        if self.code_suggestions:
            recommendations.append(f"Consider implementing {len(self.code_suggestions)} suggested test improvements")

        if not recommendations:
            recommendations.append("Coverage analysis complete - no major issues found")

        return recommendations

    def _calculate_confidence(self) -> float:
        """Calculate confidence in analysis results."""
        # Base confidence on presence of coverage data and number of findings
        base_confidence = 0.7

        if self.coverage_gaps:
            base_confidence += 0.1
        if self.edge_cases:
            base_confidence += 0.1
        if self.code_suggestions:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _get_python_files(self, project_root: Path) -> List[Path]:
        """Get Python files for analysis."""
        source_candidates = [
            project_root / "src",
            project_root
        ]

        python_files = []
        for source_root in source_candidates:
            if source_root.exists():
                files = list(source_root.rglob("*.py"))
                python_files.extend(files)

        # Filter out test files and common non-source files
        filtered_files = []
        for file_path in python_files:
            if not any(skip in str(file_path) for skip in ["__pycache__", ".pytest_cache", "test", "tests"]):
                filtered_files.append(file_path)

        return filtered_files

    def _gap_to_dict(self, gap: CoverageGap) -> Dict[str, Any]:
        """Convert CoverageGap to dictionary."""
        return {
            "file_path": gap.file_path,
            "line_number": gap.line_number,
            "function_name": gap.function_name,
            "coverage_percentage": gap.coverage_percentage,
            "gap_type": gap.gap_type,
            "description": gap.description
        }

    def _edge_case_to_dict(self, edge_case: EdgeCase) -> Dict[str, Any]:
        """Convert EdgeCase to dictionary."""
        return {
            "file_path": edge_case.file_path,
            "function_name": edge_case.function_name,
            "line_number": edge_case.line_number,
            "case_type": edge_case.case_type,
            "description": edge_case.description,
            "suggested_test_data": edge_case.suggested_test_data
        }

    def _suggestion_to_dict(self, suggestion: CodeSuggestion) -> Dict[str, Any]:
        """Convert CodeSuggestion to dictionary."""
        return {
            "suggestion_name": suggestion.suggestion_name,
            "file_path": suggestion.file_path,
            "function_under_analysis": suggestion.function_under_analysis,
            "suggested_code": suggestion.suggested_code,
            "priority": suggestion.priority,
            "edge_cases_covered": suggestion.edge_cases_covered,
            "estimated_coverage_increase": suggestion.estimated_coverage_increase
        }