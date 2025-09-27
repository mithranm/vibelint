"""
LLM subsystem for vibelint.

Provides orchestrated LLM capabilities with fast/orchestrator role separation
and configuration management.

vibelint/src/vibelint/llm/__init__.py
"""

from .llm_config import get_llm_config
from .llm_orchestrator import (
    LLMOrchestrator,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMBackend,
    create_llm_orchestrator
)
from .manager import create_llm_manager

__all__ = [
    # Configuration
    "get_llm_config",
    # Core orchestrator
    "LLMOrchestrator",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "LLMBackend",
    "create_llm_orchestrator",
    # Manager
    "create_llm_manager"
]