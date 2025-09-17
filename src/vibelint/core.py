"""
Core analysis engine for vibelint.

Dynamic LLM-powered code quality analysis that generates validators on-demand
instead of using static rule files. Draws from Martin Fowler's refactoring
catalog and adapts to specific codebase contexts.

vibelint/src/vibelint/core.py
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from .llm import LLMManager, LLMRequest
from .plugin_system import Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["DynamicAnalyzer", "AnalysisRequest", "AnalysisResult"]


@dataclass
class AnalysisRequest:
    """Request for dynamic code analysis."""
    file_path: Path
    content: str
    analysis_types: List[str]  # e.g., ["architecture", "code_smells", "dead_code"]
    context: Optional[str] = None
    severity_threshold: str = "INFO"


@dataclass
class AnalysisResult:
    """Result of dynamic analysis."""
    file_path: Path
    findings: List[Finding]
    analysis_duration: float
    llm_calls_made: int
    confidence_score: float


class DynamicAnalyzer:
    """
    LLM-powered dynamic code analyzer.

    Replaces 15+ static validators with intelligent on-demand analysis.
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        self.analysis_cache = {}  # Cache for repeated analyses

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform dynamic analysis on code using LLMs."""

        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.analysis_cache:
            logger.debug(f"Using cached analysis for {request.file_path}")
            return self.analysis_cache[cache_key]

        start_time = time.time()
        all_findings = []
        llm_calls = 0

        # Route analysis types to appropriate LLM
        for analysis_type in request.analysis_types:
            findings, calls = self._analyze_with_llm(request, analysis_type)
            all_findings.extend(findings)
            llm_calls += calls

        duration = time.time() - start_time
        confidence = self._calculate_confidence_score(all_findings)

        result = AnalysisResult(
            file_path=request.file_path,
            findings=all_findings,
            analysis_duration=duration,
            llm_calls_made=llm_calls,
            confidence_score=confidence
        )

        # Cache result
        self.analysis_cache[cache_key] = result
        return result

    def _analyze_with_llm(self, request: AnalysisRequest, analysis_type: str) -> tuple[List[Finding], int]:
        """Perform specific analysis type using appropriate LLM."""

        prompt = self._generate_analysis_prompt(request, analysis_type)

        # Route to appropriate LLM based on complexity
        llm_request = LLMRequest(
            content=prompt,
            task_type=analysis_type,
            max_tokens=2048,
            temperature=0.1
        )

        try:
            response = self.llm.process_request(llm_request)
            findings = self._parse_llm_findings(response["content"], request.file_path)
            return findings, 1

        except Exception as e:
            logger.error(f"LLM analysis failed for {analysis_type}: {e}")
            return [], 0

    def _generate_analysis_prompt(self, request: AnalysisRequest, analysis_type: str) -> str:
        """Generate analysis prompt based on type and context."""

        base_context = f"""
File: {request.file_path}
Content length: {len(request.content)} characters
Context: {request.context or "General code analysis"}

Code to analyze:
```python
{request.content}
```
"""

        if analysis_type == "architecture":
            return f"""
{base_context}

Analyze this code for architectural issues based on SOLID principles and Martin Fowler's refactoring catalog:

1. Single Responsibility Principle violations
2. Dependencies and coupling issues
3. Code organization and module cohesion
4. Design patterns misuse or opportunities

Return findings in this JSON format:
{{
  "findings": [
    {{
      "rule_id": "ARCHITECTURE-SRP",
      "severity": "WARN|INFO|ERROR",
      "line": 123,
      "message": "Brief description of issue",
      "suggestion": "Specific actionable fix"
    }}
  ]
}}
"""

        elif analysis_type == "code_smells":
            return f"""
{base_context}

Analyze this code for common code smells from Martin Fowler's catalog:

1. Long Method (>20 lines)
2. Large Class (>300 lines or >20 methods)
3. Long Parameter List (>3 parameters)
4. Magic Numbers (unexplained constants)
5. Duplicated Code
6. Dead Code
7. Feature Envy
8. Message Chains

Return findings in JSON format with line numbers and specific suggestions.
"""

        elif analysis_type == "naming":
            return f"""
{base_context}

Analyze naming conventions and clarity:

1. Uncommunicative variable/function names
2. Inconsistent naming patterns
3. Misleading names
4. Abbreviations that reduce clarity

Focus on making code self-documenting through better names.
"""

        elif analysis_type == "complexity":
            return f"""
{base_context}

Analyze code complexity issues:

1. Cyclomatic complexity (nested conditions)
2. Cognitive complexity (hard to understand)
3. Boolean expression complexity
4. Nested loop/comprehension complexity

Suggest simplifications that improve readability.
"""

        else:
            return f"""
{base_context}

Perform general code quality analysis covering:
- Code smells and anti-patterns
- Naming and clarity issues
- Structural problems
- Maintainability concerns

Be specific about line numbers and provide actionable suggestions.
"""

    def _parse_llm_findings(self, llm_response: str, file_path: Path) -> List[Finding]:
        """Parse LLM response into Finding objects."""
        findings = []

        try:
            # Try to parse as JSON first
            import json

            # Extract JSON from response (LLM might add extra text)
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = llm_response[start:end]
                data = json.loads(json_str)

                for finding_data in data.get("findings", []):
                    severity_map = {
                        "ERROR": Severity.BLOCK,
                        "WARN": Severity.WARN,
                        "INFO": Severity.INFO
                    }

                    finding = Finding(
                        rule_id=finding_data.get("rule_id", "LLM-ANALYSIS"),
                        message=finding_data.get("message", "LLM identified issue"),
                        file_path=file_path,
                        line=finding_data.get("line", 1),
                        severity=severity_map.get(finding_data.get("severity", "INFO"), Severity.INFO),
                        suggestion=finding_data.get("suggestion", "See LLM analysis")
                    )
                    findings.append(finding)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")

            # Fallback: Parse as unstructured text
            findings.append(Finding(
                rule_id="LLM-ANALYSIS",
                message="LLM identified code quality issues",
                file_path=file_path,
                line=1,
                severity=Severity.INFO,
                suggestion=f"LLM Analysis: {llm_response[:200]}..."
            ))

        return findings

    def _generate_cache_key(self, request: AnalysisRequest) -> str:
        """Generate cache key for analysis request."""
        import hashlib

        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        analysis_types = "+".join(sorted(request.analysis_types))

        return f"{request.file_path}:{content_hash}:{analysis_types}"

    def _calculate_confidence_score(self, findings: List[Finding]) -> float:
        """Calculate confidence score for analysis results."""
        if not findings:
            return 1.0  # High confidence in "no issues"

        # Score based on finding specificity (line numbers, detailed messages)
        specific_findings = sum(1 for f in findings if f.line > 1 and len(f.message) > 20)

        return min(specific_findings / len(findings), 1.0)

    def generate_validator_code(self, rule_description: str) -> str:
        """Generate Python validator code for a custom rule."""

        prompt = f"""
Generate a Python validator class for this rule:

Rule: {rule_description}

The validator should:
1. Inherit from BaseValidator
2. Have appropriate rule_id, name, description
3. Implement validate() method that yields Finding objects
4. Include proper error handling
5. Follow the existing vibelint plugin system patterns

Return only the Python code, no explanations.

Example structure:
```python
class CustomValidator(BaseValidator):
    rule_id = "CUSTOM-RULE"
    name = "Rule Name"
    description = "Rule description"

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        # Implementation
        pass
```
"""

        llm_request = LLMRequest(
            content=prompt,
            task_type="code_generation",
            max_tokens=1000,
            temperature=0.1
        )

        try:
            response = self.llm.process_request(llm_request)
            return response["content"]
        except Exception as e:
            logger.error(f"Failed to generate validator code: {e}")
            return f"# Error generating validator: {e}"


def create_dynamic_analyzer(config: Dict[str, Any]) -> Optional[DynamicAnalyzer]:
    """Create dynamic analyzer with LLM configuration."""
    from .llm import create_llm_manager

    llm_manager = create_llm_manager(config)
    if not llm_manager:
        logger.warning("No LLM configured - dynamic analysis unavailable")
        return None

    return DynamicAnalyzer(llm_manager)
