"""
Justification workflow implementation.

A modular workflow for comprehensive code justification analysis including
dependency analysis, static issue detection, and LLM-powered insights.
"""

from .engine import JustificationEngine
from .models import CodeAnalysis, AnalysisResult

__all__ = ['JustificationEngine', 'CodeAnalysis', 'AnalysisResult']