"""
Agentic coverage vibe checking for vibelint.

AI-powered coverage analysis, edge case detection, and automated
code generation using dual LLM architecture. Checks the "vibe" of your
coverage and suggests improvements.

tools/vibelint/src/vibelint/coverage_analysis.py
"""

import ast
import logging
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .llm_manager import LLMRequest, create_llm_manager

logger = logging.getLogger(__name__)

__all__ = [
    "CoverageVibeAnalyzer",
    "CoverageGap",
    "EdgeCase",
    "CodeSuggestion",
    "run_coverage_vibe_check",
]


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


@dataclass
class TestAssessment:
    """Assessment of test quality and behavior validation."""

    test_name: str
    file_path: str
    function_under_test: str
    input_validation_score: float  # 0-1, how well test inputs match expected behavior
    output_validation_score: float  # 0-1, how well test outputs match requirements
    requirement_coverage: List[str]  # Which requirements this test covers
    missing_requirements: List[str]  # Requirements not covered by any test
    behavioral_issues: List[str]  # Potential behavioral inconsistencies
    improvement_suggestions: List[str]  # AI-generated suggestions for better testing
    confidence_score: float  # 0-1, AI confidence in assessment


class CoverageVibeAnalyzer:
    """Agentic coverage vibe analyzer using dual LLM architecture."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with vibelint configuration."""
        self.config = config
        self.llm_manager = create_llm_manager(config)
        self.coverage_gaps: List[CoverageGap] = []
        self.edge_cases: List[EdgeCase] = []
        self.code_suggestions: List[CodeSuggestion] = []
        self.test_assessments: List[TestAssessment] = []

    async def analyze_test_coverage(
        self,
        source_paths: List[str],
        test_paths: List[str],
        coverage_threshold: float = 80.0,
        max_suggestions: int = 20,
    ) -> Dict[str, Any]:
        """Run comprehensive agentic test analysis."""

        if not self.llm_manager:
            logger.warning("No LLM configuration found - skipping AI analysis")
            return {"error": "LLM configuration required for test analysis"}

        print("=== Agentic Coverage Vibe Analysis ===")
        print("• Coverage gap detection")
        print("• Edge case discovery")
        print("• AI-powered code generation")
        print("• Self-validating test assessment")
        print("• Requirements compliance checking")
        print()

        # Step 1: Parse existing coverage data
        coverage_data = await self._parse_coverage_data()
        if not coverage_data:
            # Generate fresh coverage data
            coverage_data = await self._generate_coverage_data(source_paths)

        # Step 2: Identify coverage gaps
        self.coverage_gaps = await self._identify_coverage_gaps(
            coverage_data, source_paths, coverage_threshold
        )

        # Step 3: AI-powered edge case discovery
        self.edge_cases = await self._discover_edge_cases(source_paths)

        # Step 4: Generate code suggestions
        self.code_suggestions = await self._generate_code_suggestions(source_paths, max_suggestions)

        # Step 5: Self-validating test assessment
        self.test_assessments = await self._assess_existing_tests(test_paths)

        # Step 6: Requirements compliance checking
        await self._check_requirements_compliance()

        # Step 7: Compile results
        results = {
            "coverage_data": coverage_data,
            "coverage_gaps": self.coverage_gaps,
            "edge_cases": self.edge_cases,
            "code_suggestions": self.code_suggestions,
            "test_assessments": self.test_assessments,
            "analysis_summary": self._create_analysis_summary(coverage_threshold),
        }

        return results

    async def _parse_coverage_data(self) -> Optional[Dict[str, Any]]:
        """Parse existing pytest-cov XML coverage data."""
        coverage_file = Path("coverage.xml")
        if not coverage_file.exists():
            return None

        try:
            tree = ET.parse(coverage_file)
            root = tree.getroot()

            coverage_data = {
                "overall_coverage": float(root.attrib.get("line-rate", 0)) * 100,
                "files": {},
            }

            for package in root.findall(".//package"):
                for class_elem in package.findall("classes/class"):
                    filename = class_elem.attrib["filename"]
                    line_rate = float(class_elem.attrib.get("line-rate", 0)) * 100

                    # Parse line coverage
                    covered_lines = set()
                    uncovered_lines = set()

                    for line in class_elem.findall("lines/line"):
                        line_num = int(line.attrib["number"])
                        hits = int(line.attrib["hits"])

                        if hits > 0:
                            covered_lines.add(line_num)
                        else:
                            uncovered_lines.add(line_num)

                    coverage_data["files"][filename] = {
                        "line_rate": line_rate,
                        "covered_lines": covered_lines,
                        "uncovered_lines": uncovered_lines,
                    }

            return coverage_data

        except ET.ParseError as e:
            logger.error(f"Failed to parse coverage.xml: {e}")
            return None

    async def _generate_coverage_data(self, source_paths: List[str]) -> Dict[str, Any]:
        """Generate fresh coverage data by running pytest."""
        print("Generating fresh coverage data...")

        try:
            # Run pytest with coverage
            cmd = [
                "python",
                "-m",
                "pytest",
                "--cov=" + ",".join(source_paths),
                "--cov-report=xml",
                "--cov-report=term-missing",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return await self._parse_coverage_data() or {}
            else:
                logger.error(f"pytest failed: {result.stderr}")
                return {}

        except subprocess.TimeoutExpired:
            logger.error("pytest timeout - tests taking too long")
            return {}
        except Exception as e:
            logger.error(f"Failed to generate coverage: {e}")
            return {}

    async def _identify_coverage_gaps(
        self, coverage_data: Dict[str, Any], source_paths: List[str], threshold: float
    ) -> List[CoverageGap]:
        """Identify significant coverage gaps using fast LLM."""
        gaps = []

        for file_path, file_data in coverage_data.get("files", {}).items():
            if file_data["line_rate"] < threshold:
                # Use fast LLM to analyze uncovered lines
                uncovered_lines = file_data["uncovered_lines"]
                if uncovered_lines:
                    gaps.extend(await self._analyze_uncovered_lines(file_path, uncovered_lines))

        return gaps

    async def _analyze_uncovered_lines(
        self, file_path: str, uncovered_lines: Set[int]
    ) -> List[CoverageGap]:
        """Analyze uncovered lines using fast LLM."""
        gaps = []

        try:
            # Read the source file
            source_code = Path(file_path).read_text(encoding="utf-8")
            lines = source_code.splitlines()

            # Create context for LLM analysis
            context_lines = []
            for line_num in sorted(uncovered_lines):
                if 1 <= line_num <= len(lines):
                    # Add some context around uncovered line
                    start = max(0, line_num - 3)
                    end = min(len(lines), line_num + 2)
                    context = "\n".join(f"{i+1:3d}: {lines[i]}" for i in range(start, end))
                    context_lines.append(f"Uncovered line {line_num}:\n{context}")

            if not context_lines:
                return gaps

            # Fast LLM request to categorize gaps
            request = LLMRequest(
                content=f"""Analyze these uncovered code lines from {file_path}:

{chr(10).join(context_lines[:5])}  # Limit to first 5 for speed

For each uncovered line, identify:
1. Function name (if applicable)
2. Gap type: 'error_path', 'edge_case', 'normal_flow', 'unreachable'
3. Brief description of what should be tested

Respond in format:
Line X: function_name | gap_type | description""",
                task_type="coverage_analysis",
                max_tokens=512,
                temperature=0.1,
            )

            response = await self.llm_manager.process_request(request)

            if response["success"]:
                # Parse LLM response to create CoverageGap objects
                gaps.extend(
                    self._parse_coverage_gaps_response(
                        response["content"], file_path, uncovered_lines
                    )
                )

        except Exception as e:
            logger.error(f"Failed to analyze uncovered lines in {file_path}: {e}")

        return gaps

    def _parse_coverage_gaps_response(
        self, response: str, file_path: str, uncovered_lines: Set[int]
    ) -> List[CoverageGap]:
        """Parse LLM response into CoverageGap objects."""
        gaps = []

        for line in response.strip().split("\n"):
            try:
                if ":" in line and "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) >= 3:
                        line_part = parts[0].strip()
                        gap_type = parts[1].strip()
                        description = parts[2].strip()

                        # Extract line number
                        line_num = None
                        if "Line" in line_part:
                            try:
                                line_num = int(line_part.split("Line")[1].split(":")[0].strip())
                            except (ValueError, IndexError):
                                continue

                        if line_num and line_num in uncovered_lines:
                            # Extract function name (simplified)
                            function_name = None
                            if "|" in line_part:
                                func_part = line_part.split("|")[0].strip()
                                if func_part and func_part != f"Line {line_num}":
                                    function_name = func_part

                            gaps.append(
                                CoverageGap(
                                    file_path=file_path,
                                    line_number=line_num,
                                    function_name=function_name,
                                    coverage_percentage=0.0,  # Uncovered
                                    gap_type=gap_type,
                                    description=description,
                                )
                            )

            except Exception as e:
                logger.debug(f"Failed to parse gap line '{line}': {e}")
                continue

        return gaps

    async def _discover_edge_cases(self, source_paths: List[str]) -> List[EdgeCase]:
        """Discover edge cases using orchestrator LLM for complex reasoning."""
        edge_cases = []

        for source_path in source_paths:
            path = Path(source_path)
            if path.is_file() and path.suffix == ".py":
                edge_cases.extend(await self._analyze_file_edge_cases(str(path)))
            elif path.is_dir():
                # Analyze Python files in directory
                for py_file in path.rglob("*.py"):
                    edge_cases.extend(await self._analyze_file_edge_cases(str(py_file)))

                    # Limit analysis to avoid overwhelming LLM
                    if len(edge_cases) > 50:
                        break

        return edge_cases

    async def _analyze_file_edge_cases(self, file_path: str) -> List[EdgeCase]:
        """Analyze a single file for edge cases using orchestrator LLM."""
        try:
            source_code = Path(file_path).read_text(encoding="utf-8")

            # Parse AST to extract function signatures
            tree = ast.parse(source_code)
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            ]

            if not functions:
                return []

            # Create context for orchestrator LLM
            func_signatures = []
            for func in functions[:5]:  # Limit to first 5 functions
                args = [arg.arg for arg in func.args.args]
                func_signatures.append(f"{func.name}({', '.join(args)}) - line {func.lineno}")

            request = LLMRequest(
                content=f"""Analyze this Python file for potential edge cases that should be tested:

File: {file_path}
Functions:
{chr(10).join(func_signatures)}

Source code (relevant sections):
{source_code[:2000]}...

Identify edge cases for each function:
1. Boundary conditions (empty inputs, max values, None values)
2. Error conditions (invalid types, network failures, file not found)
3. Race conditions (concurrent access, state changes)
4. Integration issues (external API failures, database errors)

For each edge case, provide:
- Function name
- Line number (approximate)
- Case type: boundary/error_condition/race_condition/integration
- Description of what should be tested
- Suggested test data (if applicable)

Format: function_name|line_num|case_type|description|test_data""",
                task_type="edge_case_analysis",
                max_tokens=1024,
                temperature=0.2,
            )

            response = await self.llm_manager.process_request(request)

            if response["success"]:
                return self._parse_edge_cases_response(response["content"], file_path)

        except Exception as e:
            logger.error(f"Failed to analyze edge cases in {file_path}: {e}")

        return []

    def _parse_edge_cases_response(self, response: str, file_path: str) -> List[EdgeCase]:
        """Parse LLM response into EdgeCase objects."""
        edge_cases = []

        for line in response.strip().split("\n"):
            try:
                if "|" in line and line.count("|") >= 3:
                    parts = line.split("|", 4)
                    function_name = parts[0].strip()
                    line_number = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                    case_type = parts[2].strip()
                    description = parts[3].strip()
                    test_data = parts[4].strip() if len(parts) > 4 else None

                    # Parse test data if provided
                    suggested_test_data = None
                    if test_data and test_data != "None":
                        try:
                            # Simple parsing - could be enhanced
                            suggested_test_data = {"input": test_data}
                        except Exception:
                            suggested_test_data = None

                    edge_cases.append(
                        EdgeCase(
                            file_path=file_path,
                            function_name=function_name,
                            line_number=line_number,
                            case_type=case_type,
                            description=description,
                            suggested_test_data=suggested_test_data,
                        )
                    )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse edge case line '{line}': {e}")
                continue

        return edge_cases

    async def _generate_code_suggestions(
        self, source_paths: List[str], max_suggestions: int
    ) -> List[CodeSuggestion]:
        """Generate pytest-compatible test suggestions using orchestrator LLM."""
        suggestions = []

        # Combine coverage gaps and edge cases for test generation
        test_targets = []

        # High-priority targets from coverage gaps
        for gap in self.coverage_gaps[:10]:  # Top 10 gaps
            test_targets.append(
                {
                    "type": "coverage_gap",
                    "file_path": gap.file_path,
                    "function": gap.function_name,
                    "line": gap.line_number,
                    "description": gap.description,
                    "priority": 4 if gap.gap_type == "error_path" else 3,
                }
            )

        # Edge cases as test targets
        for edge_case in self.edge_cases[:10]:  # Top 10 edge cases
            test_targets.append(
                {
                    "type": "edge_case",
                    "file_path": edge_case.file_path,
                    "function": edge_case.function_name,
                    "line": edge_case.line_number,
                    "description": edge_case.description,
                    "priority": 5 if edge_case.case_type == "error_condition" else 3,
                }
            )

        # Sort by priority and generate tests
        test_targets.sort(key=lambda x: x["priority"], reverse=True)

        for target in test_targets[:max_suggestions]:
            suggestion = await self._generate_single_suggestion(target)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    async def _generate_single_suggestion(self, target: Dict[str, Any]) -> Optional[CodeSuggestion]:
        """Generate a single test case using orchestrator LLM."""
        try:
            # Read source file for context
            source_code = Path(target["file_path"]).read_text(encoding="utf-8")

            # Extract function code if possible
            function_context = self._extract_function_context(
                source_code, target["function"], target["line"]
            )

            request = LLMRequest(
                content=f"""Generate a pytest test case for this {target['type']}:

File: {target['file_path']}
Function: {target['function']}
Issue: {target['description']}

Function context:
{function_context}

Requirements:
1. Create a complete pytest test function
2. Use proper assertions and test data
3. Handle mocking if needed for external dependencies
4. Follow pytest best practices
5. Include docstring explaining what is being tested

Generate ONLY the test function code, properly formatted for pytest.
Test function name should be descriptive and start with 'test_'.""",
                task_type="test_generation",
                max_tokens=768,
                temperature=0.3,
            )

            response = await self.llm_manager.process_request(request)

            if response["success"] and response["content"]:
                # Extract clean test code
                test_code = self._clean_generated_test_code(response["content"])

                if test_code:
                    return CodeSuggestion(
                        test_name=self._extract_test_name(test_code),
                        file_path=target["file_path"],
                        function_under_test=target["function"] or "unknown",
                        test_code=test_code,
                        priority=target["priority"],
                        edge_cases_covered=[target["description"]],
                        estimated_coverage_increase=5.0,  # Rough estimate
                    )

        except Exception as e:
            logger.error(f"Failed to generate test for {target}: {e}")

        return None

    def _extract_function_context(
        self, source_code: str, function_name: Optional[str], line_number: int
    ) -> str:
        """Extract relevant function context from source code."""
        lines = source_code.splitlines()

        if not function_name or line_number <= 0:
            # Return lines around the target line
            start = max(0, line_number - 5)
            end = min(len(lines), line_number + 5)
            return "\n".join(f"{i+1:3d}: {lines[i]}" for i in range(start, end))

        # Try to find the function definition
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract function lines
                    start_line = node.lineno - 1
                    end_line = node.end_lineno or (start_line + 10)

                    func_lines = lines[start_line:end_line]
                    return "\n".join(
                        f"{start_line + i + 1:3d}: {line}" for i, line in enumerate(func_lines)
                    )
        except Exception:
            pass

        # Fallback to line context
        start = max(0, line_number - 5)
        end = min(len(lines), line_number + 5)
        return "\n".join(f"{i+1:3d}: {lines[i]}" for i in range(start, end))

    def _clean_generated_test_code(self, raw_code: str) -> str:
        """Clean and validate generated test code."""
        # Remove markdown code blocks if present
        if "```python" in raw_code:
            lines = raw_code.split("\n")
            start_idx = None
            end_idx = None

            for i, line in enumerate(lines):
                if "```python" in line and start_idx is None:
                    start_idx = i + 1
                elif "```" in line and start_idx is not None:
                    end_idx = i
                    break

            if start_idx is not None:
                end_idx = end_idx or len(lines)
                raw_code = "\n".join(lines[start_idx:end_idx])

        # Basic validation - should contain def test_
        if "def test_" not in raw_code:
            return ""

        # Clean up indentation
        lines = raw_code.strip().split("\n")
        if lines:
            # Remove extra leading whitespace
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            if min_indent > 0:
                lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]

        return "\n".join(lines)

    def _extract_test_name(self, test_code: str) -> str:
        """Extract test function name from generated code."""
        for line in test_code.split("\n"):
            if "def test_" in line:
                try:
                    return line.split("def ")[1].split("(")[0]
                except IndexError:
                    pass
        return "test_generated"

    def _create_analysis_summary(self, threshold: float) -> Dict[str, Any]:
        """Create summary of analysis results."""
        return {
            "coverage_gaps_found": len(self.coverage_gaps),
            "edge_cases_discovered": len(self.edge_cases),
            "test_suggestions_generated": len(self.test_suggestions),
            "high_priority_suggestions": len([s for s in self.test_suggestions if s.priority >= 4]),
            "estimated_coverage_increase": sum(
                s.estimated_coverage_increase for s in self.test_suggestions
            ),
            "coverage_threshold": threshold,
        }


async def run_coverage_vibe_check(
    config: Optional[Dict[str, Any]] = None,
    source_paths: Optional[List[str]] = None,
    test_paths: Optional[List[str]] = None,
    coverage_threshold: float = 80.0,
    max_suggestions: int = 20,
    generate_tests: bool = False,
) -> Dict[str, Any]:
    """Run comprehensive agentic coverage analysis."""
    # Load config from pyproject.toml if not provided
    if config is None:
        from .config import load_config

        config = load_config().settings

    # Default paths if not provided
    if source_paths is None:
        source_paths = ["src/vibelint"]
    if test_paths is None:
        test_paths = ["tests"]

    analyzer = CoverageVibeAnalyzer(config)

    results = await analyzer.analyze_test_coverage(
        source_paths, test_paths, coverage_threshold, max_suggestions
    )

    if generate_tests and results.get("code_suggestions"):
        # Optionally write generated tests to files
        await _write_generated_tests(results["code_suggestions"])

    return results


async def _write_generated_tests(suggestions: List[CodeSuggestion]) -> None:
    """Write generated test suggestions to test files."""
    test_dir = Path("tests/generated")
    test_dir.mkdir(exist_ok=True)

    for i, suggestion in enumerate(suggestions):
        test_file = test_dir / f"test_generated_{i+1:02d}_{suggestion.test_name}.py"

        test_content = f'''"""
Generated test case for {suggestion.function_under_test}

Priority: {suggestion.priority}/5
Estimated coverage increase: {suggestion.estimated_coverage_increase:.1f}%
Edge cases covered: {', '.join(suggestion.edge_cases_covered)}

This test was generated by vibelint agentic analysis.
Review and modify as needed before using in production.
"""

import pytest
from unittest.mock import Mock, patch

{suggestion.test_code}
'''

        test_file.write_text(test_content, encoding="utf-8")
        print(f"Generated test file: {test_file}")

    async def _assess_existing_tests(self, test_paths: List[str]) -> List[TestAssessment]:
        """Use dual LLM system to assess existing test quality and behavior validation."""
        assessments = []

        print("🧠 Assessing existing tests with dual LLM validation...")

        for test_path in test_paths:
            path = Path(test_path)
            if path.is_file() and path.suffix == ".py":
                assessments.extend(await self._assess_test_file(str(path)))
            elif path.is_dir():
                # Analyze test files in directory
                for test_file in path.rglob("test_*.py"):
                    assessments.extend(await self._assess_test_file(str(test_file)))

                    # Limit to avoid overwhelming LLM
                    if len(assessments) > 20:
                        break

        return assessments

    async def _assess_test_file(self, test_file_path: str) -> List[TestAssessment]:
        """Assess a single test file using orchestrator LLM for deep analysis."""
        assessments = []

        try:
            test_code = Path(test_file_path).read_text(encoding="utf-8")

            # Parse test functions
            tree = ast.parse(test_code)
            test_functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ]

            if not test_functions:
                return assessments

            # Analyze up to 5 test functions per file to avoid token limits
            for test_func in test_functions[:5]:
                assessment = await self._assess_single_test(test_func, test_file_path, test_code)
                if assessment:
                    assessments.append(assessment)

        except Exception as e:
            logger.error(f"Failed to assess test file {test_file_path}: {e}")

        return assessments

    async def _assess_single_test(
        self, test_func: ast.FunctionDef, test_file_path: str, full_test_code: str
    ) -> Optional[TestAssessment]:
        """Deep assessment of a single test function using orchestrator LLM."""
        try:
            # Extract test function code
            lines = full_test_code.splitlines()
            start_line = test_func.lineno - 1
            end_line = test_func.end_lineno or (start_line + 20)

            test_function_code = "\n".join(lines[start_line:end_line])

            # Try to identify the function being tested
            function_under_test = self._identify_function_under_test(test_func.name)

            request = LLMRequest(
                content=f"""Analyze this test function for behavioral correctness and requirement coverage:

Test File: {test_file_path}
Test Function: {test_func.name}
Likely Testing: {function_under_test}

Test Code:
{test_function_code}

Assess the following aspects and provide scores (0.0-1.0):

1. INPUT VALIDATION SCORE: How well do the test inputs represent realistic, edge, and boundary cases?
2. OUTPUT VALIDATION SCORE: How thoroughly does the test validate outputs and side effects?
3. REQUIREMENT COVERAGE: What specific requirements/behaviors does this test validate?
4. MISSING REQUIREMENTS: What important behaviors/requirements are NOT tested?
5. BEHAVIORAL ISSUES: Any potential issues with test logic or assumptions?
6. IMPROVEMENT SUGGESTIONS: Specific suggestions for better testing?
7. CONFIDENCE SCORE: Your confidence in this assessment?

Format your response as:
INPUT_VALIDATION: 0.X
OUTPUT_VALIDATION: 0.X
REQUIREMENTS_COVERED: requirement1, requirement2, requirement3
MISSING_REQUIREMENTS: missing1, missing2, missing3
BEHAVIORAL_ISSUES: issue1, issue2
IMPROVEMENTS: suggestion1, suggestion2, suggestion3
CONFIDENCE: 0.X""",
                task_type="test_assessment",
                max_tokens=1024,
                temperature=0.2,
            )

            response = await self.llm_manager.process_request(request)

            if response["success"] and response["content"]:
                return self._parse_test_assessment_response(
                    response["content"], test_func.name, test_file_path, function_under_test
                )

        except Exception as e:
            logger.error(f"Failed to assess test {test_func.name}: {e}")

        return None

    def _identify_function_under_test(self, test_name: str) -> str:
        """Try to identify what function is being tested based on test name."""
        # Remove test_ prefix and common suffixes
        cleaned_name = test_name.replace("test_", "")

        # Handle common test naming patterns
        if "_should_" in cleaned_name:
            cleaned_name = cleaned_name.split("_should_")[0]
        elif "_when_" in cleaned_name:
            cleaned_name = cleaned_name.split("_when_")[0]
        elif "_with_" in cleaned_name:
            cleaned_name = cleaned_name.split("_with_")[0]

        return cleaned_name

    def _parse_test_assessment_response(
        self, response: str, test_name: str, test_file: str, function_under_test: str
    ) -> Optional[TestAssessment]:
        """Parse LLM response into TestAssessment object."""
        try:
            lines = response.strip().split("\n")

            input_score = 0.5
            output_score = 0.5
            requirements_covered = []
            missing_requirements = []
            behavioral_issues = []
            improvements = []
            confidence = 0.5

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().upper()
                    value = value.strip()

                    if key == "INPUT_VALIDATION":
                        try:
                            input_score = float(value)
                        except ValueError:
                            pass
                    elif key == "OUTPUT_VALIDATION":
                        try:
                            output_score = float(value)
                        except ValueError:
                            pass
                    elif key == "REQUIREMENTS_COVERED":
                        requirements_covered = [
                            req.strip() for req in value.split(",") if req.strip()
                        ]
                    elif key == "MISSING_REQUIREMENTS":
                        missing_requirements = [
                            req.strip() for req in value.split(",") if req.strip()
                        ]
                    elif key == "BEHAVIORAL_ISSUES":
                        behavioral_issues = [
                            issue.strip() for issue in value.split(",") if issue.strip()
                        ]
                    elif key == "IMPROVEMENTS":
                        improvements = [imp.strip() for imp in value.split(",") if imp.strip()]
                    elif key == "CONFIDENCE":
                        try:
                            confidence = float(value)
                        except ValueError:
                            pass

            return TestAssessment(
                test_name=test_name,
                file_path=test_file,
                function_under_test=function_under_test,
                input_validation_score=input_score,
                output_validation_score=output_score,
                requirement_coverage=requirements_covered,
                missing_requirements=missing_requirements,
                behavioral_issues=behavioral_issues,
                improvement_suggestions=improvements,
                confidence_score=confidence,
            )

        except Exception as e:
            logger.debug(f"Failed to parse test assessment response: {e}")
            return None

    async def _check_requirements_compliance(self):
        """Interactive requirements discovery and compliance checking."""
        if not self.test_assessments:
            return

        print("🔍 Analyzing requirements coverage across all tests...")

        # Aggregate all covered and missing requirements
        all_covered = set()
        all_missing = set()

        for assessment in self.test_assessments:
            all_covered.update(assessment.requirement_coverage)
            all_missing.update(assessment.missing_requirements)

        # Find gaps - requirements mentioned as missing but never covered
        uncovered_requirements = all_missing - all_covered

        if uncovered_requirements:
            print("\n⚠️  Potentially uncovered requirements detected:")
            for req in sorted(uncovered_requirements):
                print(f"   • {req}")

            # Could prompt user for clarification here
            print(
                "\n💡 Consider adding tests for these requirements or clarifying if they're needed."
            )
