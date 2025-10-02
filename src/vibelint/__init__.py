"""Vibelint: Code Quality and Style Validator

A focused code analysis tool for Python projects.
"""

from vibelint.config import (
    Config,
    EmbeddingConfig,
    LLMConfig,
    get_embedding_config,
    get_llm_config,
    load_config,
)
from vibelint.llm_client import (
    LLMManager,
    LLMRequest,
    LLMResponse,
    LLMRole,
    create_llm_manager,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Config",
    "load_config",
    "LLMConfig",
    "EmbeddingConfig",
    "get_llm_config",
    "get_embedding_config",
    # LLM Client
    "LLMManager",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "create_llm_manager",
]
