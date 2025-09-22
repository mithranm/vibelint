"""
Context management system for vibelint.

Provides multi-level context analysis, LLM context probing, and organizational
violation detection across different granularities.

vibelint/src/vibelint/context/__init__.py
"""

from .analyzer import ContentViolation, ContextAnalyzer, TreeViolation
from .probing import (ContextProber, InferenceEngine, ProbeConfig, ProbeResult,
                      run_context_probing)

__all__ = [
    "ContextAnalyzer",
    "TreeViolation",
    "ContentViolation",
    "ContextProber",
    "ProbeResult",
    "ProbeConfig",
    "InferenceEngine",
    "run_context_probing",
]
