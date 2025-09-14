"""
Architecture analysis module for vibelint.

This module consolidates all architecture-related validators:
- Basic architectural patterns (consistency, naming)
- LLM-powered semantic analysis
- Embedding-based similarity detection
- Fallback pattern analysis for silent failures

vibelint/src/vibelint/validators/architecture/__init__.py
"""

from .basic_patterns import ArchitectureValidator
from .llm_analysis import ArchitectureLLMValidator
from .semantic_similarity import SemanticSimilarityValidator
from .fallback_patterns import FallbackAnalyzer

__all__ = [
    "ArchitectureValidator",
    "ArchitectureLLMValidator",
    "SemanticSimilarityValidator",
    "FallbackAnalyzer"
]