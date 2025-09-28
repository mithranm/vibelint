"""
Core data models for justification workflow.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class CodeAnalysis:
    """Factual analysis of a code element."""

    file_path: str
    element_type: str  # 'file', 'method', 'class'
    element_name: str
    line_number: int
    description: str  # What we can factually observe
    analysis_method: str  # 'static_analysis', 'embedding_similarity', 'llm_comparison'
    has_documentation: bool
    dependencies: List[str]
    complexity_metrics: Dict[str, int]
    llm_reasoning: Optional[str] = None  # Only when LLM was actually used


@dataclass
class AnalysisResult:
    """Result of code analysis."""

    target_path: str
    analyses: List[CodeAnalysis]
    structural_issues: List[str]
    recommendations: List[str]
    analysis_summary: Dict[str, Any]
    llm_calls_made: List[Dict[str, Any]]  # Full log of any LLM interactions