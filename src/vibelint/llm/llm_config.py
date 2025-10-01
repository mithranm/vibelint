"""
Vibelint Configuration Management

Handles configuration loading for vibelint with support for multiple config sources
and environment overrides.

Configuration priority order:
1. Environment variables
2. dev.pyproject.toml (development overrides)
3. pyproject.toml (production config)
4. Default values

This is the authoritative config implementation - kaia imports from here.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Handle TOML library imports - support Python 3.10 (tomli) and 3.11+ (tomllib)
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10 fallback

from dotenv import load_dotenv

# Configuration constants
DEFAULT_FAST_TEMPERATURE = 0.1
DEFAULT_FAST_MAX_TOKENS = 2048
DEFAULT_ORCHESTRATOR_TEMPERATURE = 0.2
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 8192
DEFAULT_CONTEXT_THRESHOLD = 3000


@dataclass
class LLMConfig:
    """Typed LLM configuration with explicit validation."""

    # Required fields first
    fast_api_url: str
    fast_model: str
    orchestrator_api_url: str
    orchestrator_model: str

    # Optional fields with defaults
    fast_backend: str = "vllm"
    fast_temperature: float = DEFAULT_FAST_TEMPERATURE
    fast_max_tokens: int = DEFAULT_FAST_MAX_TOKENS
    fast_max_context_tokens: Optional[int] = None
    fast_api_key: Optional[str] = None

    orchestrator_backend: str = "llamacpp"
    orchestrator_temperature: float = DEFAULT_ORCHESTRATOR_TEMPERATURE
    orchestrator_max_tokens: int = DEFAULT_ORCHESTRATOR_MAX_TOKENS
    orchestrator_max_context_tokens: Optional[int] = None
    orchestrator_api_key: Optional[str] = None

    context_threshold: int = DEFAULT_CONTEXT_THRESHOLD
    enable_context_probing: bool = True
    enable_fallback: bool = False

    def __post_init__(self):
        """Validate required configuration."""
        if not self.fast_api_url:
            raise ValueError("fast_api_url is required - configure in [tool.vibelint.llm]")
        if not self.fast_model:
            raise ValueError("fast_model is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_api_url:
            raise ValueError("orchestrator_api_url is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_model:
            raise ValueError("orchestrator_model is required - configure in [tool.vibelint.llm]")


@dataclass
class EmbeddingConfig:
    """Typed embedding configuration with explicit validation."""

    # Required fields first
    code_api_url: str
    natural_api_url: str

    # Optional fields with defaults
    code_model: str = "text-embedding-ada-002"
    natural_model: str = "text-embedding-ada-002"
    use_specialized_embeddings: bool = True

    def __post_init__(self):
        """Validate required configuration."""
        if not self.code_api_url:
            raise ValueError("code_api_url is required - configure in [tool.vibelint.embeddings]")
        if not self.natural_api_url:
            raise ValueError("natural_api_url is required - configure in [tool.vibelint.embeddings]")


def load_env_files(project_root: Optional[Path] = None):
    """Load environment variables from multiple possible locations."""
    if project_root is None:
        # Auto-detect from vibelint location
        project_root = Path(__file__).parent.parent.parent.parent.parent

    env_paths = [
        Path.cwd() / ".env",  # Current directory
        project_root / ".env",  # Project root
        project_root / "tools" / "vibelint" / ".env",  # vibelint directory
        Path.home() / ".vibelint.env",  # User home directory
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)


def load_toml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {config_path}: {e}")
        return {}


def find_project_root() -> Path:
    """Find the project root (killeraiagent) from vibelint's location."""
    current = Path(__file__).parent

    # Look for the main killeraiagent project, not vibelint's own project
    while current.parent != current:
        # Check for killeraiagent project indicators
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            try:
                with open(pyproject, "rb") as f:
                    config = tomllib.load(f)
                # Look for killeraiagent project name or our specific LLM config
                if (
                    config.get("project", {}).get("name") == "killeraiagent"
                    or ("tool" in config
                        and "vibelint" in config["tool"]
                        and "llm" in config["tool"]["vibelint"]
                        and config["tool"]["vibelint"]["llm"].get("fast_api_url"))
                ):
                    return current
            except Exception:
                pass
        current = current.parent

    # Fallback to 5 levels up from this file (should reach killeraiagent root)
    return Path(__file__).parent.parent.parent.parent.parent


def get_vibelint_config() -> Dict[str, Any]:
    """
    Load vibelint configuration with proper priority order.

    Priority:
    1. Environment variables
    2. dev.pyproject.toml (development overrides)
    3. pyproject.toml (production config) 
    4. Default values
    """
    project_root = find_project_root()

    # Load environment files first
    load_env_files(project_root)

    # Load base configuration from project root
    base_config = load_toml_config(project_root / "pyproject.toml")

    # Load development overrides from project root
    dev_config = load_toml_config(project_root / "dev.pyproject.toml")

    # Load vibelint-specific config
    vibelint_config = load_toml_config(Path(__file__).parent.parent.parent / "pyproject.toml")

    # Merge configurations (dev overrides base)
    merged_config: Dict[str, Any] = {}

    # Start with vibelint's own config
    if "tool" in vibelint_config and "vibelint" in vibelint_config["tool"]:
        merged_config = vibelint_config["tool"]["vibelint"].copy()

    # Apply base config from project root
    if "tool" in base_config and "vibelint" in base_config["tool"]:
        merged_config.update(base_config["tool"]["vibelint"])

    # Apply dev overrides from project root
    if "tool" in dev_config and "vibelint" in dev_config["tool"]:
        merged_config.update(dev_config["tool"]["vibelint"])

    return merged_config


def get_llm_config() -> LLMConfig:
    """
    Get typed LLM configuration for vibelint.

    Returns:
    Validated LLMConfig object with environment variable overrides
    """
    config = get_vibelint_config()
    llm_dict = config.get("llm", {})

    # Build kwargs with environment overrides
    kwargs = {
        # Fast LLM
        "fast_api_url": (
            os.getenv("VIBELINT_FAST_LLM_API_URL") or
            os.getenv("FAST_LLM_API_URL") or
            llm_dict.get("fast_api_url")
        ),
        "fast_model": (
            os.getenv("VIBELINT_FAST_LLM_MODEL") or
            os.getenv("FAST_LLM_MODEL") or
            llm_dict.get("fast_model")
        ),
        "fast_backend": (
            os.getenv("VIBELINT_FAST_LLM_BACKEND") or
            os.getenv("FAST_LLM_BACKEND") or
            llm_dict.get("fast_backend", "vllm")
        ),
        "fast_api_key": (
            os.getenv("VIBELINT_FAST_LLM_API_KEY") or
            os.getenv("FAST_LLM_API_KEY") or
            llm_dict.get("fast_api_key")
        ),
        "fast_temperature": (
            _get_env_float("VIBELINT_FAST_LLM_TEMPERATURE") or
            _get_env_float("FAST_LLM_TEMPERATURE") or
            llm_dict.get("fast_temperature", DEFAULT_FAST_TEMPERATURE)
        ),
        "fast_max_tokens": (
            _get_env_int("VIBELINT_FAST_LLM_MAX_TOKENS") or
            _get_env_int("FAST_LLM_MAX_TOKENS") or
            llm_dict.get("fast_max_tokens", DEFAULT_FAST_MAX_TOKENS)
        ),
        "fast_max_context_tokens": llm_dict.get("fast_max_context_tokens"),

        # Orchestrator LLM
        "orchestrator_api_url": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_URL") or
            os.getenv("ORCHESTRATOR_LLM_API_URL") or
            llm_dict.get("orchestrator_api_url")
        ),
        "orchestrator_model": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_MODEL") or
            os.getenv("ORCHESTRATOR_LLM_MODEL") or
            llm_dict.get("orchestrator_model")
        ),
        "orchestrator_backend": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_BACKEND") or
            os.getenv("ORCHESTRATOR_LLM_BACKEND") or
            llm_dict.get("orchestrator_backend", "llamacpp")
        ),
        "orchestrator_api_key": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_KEY") or
            os.getenv("ORCHESTRATOR_LLM_API_KEY") or
            llm_dict.get("orchestrator_api_key")
        ),
        "orchestrator_temperature": (
            _get_env_float("VIBELINT_ORCHESTRATOR_LLM_TEMPERATURE") or
            _get_env_float("ORCHESTRATOR_LLM_TEMPERATURE") or
            llm_dict.get("orchestrator_temperature", DEFAULT_ORCHESTRATOR_TEMPERATURE)
        ),
        "orchestrator_max_tokens": (
            _get_env_int("VIBELINT_ORCHESTRATOR_LLM_MAX_TOKENS") or
            _get_env_int("ORCHESTRATOR_LLM_MAX_TOKENS") or
            llm_dict.get("orchestrator_max_tokens", DEFAULT_ORCHESTRATOR_MAX_TOKENS)
        ),
        "orchestrator_max_context_tokens": llm_dict.get("orchestrator_max_context_tokens"),

        # Routing configuration
        "context_threshold": (
            _get_env_int("VIBELINT_LLM_CONTEXT_THRESHOLD") or
            _get_env_int("LLM_CONTEXT_THRESHOLD") or
            llm_dict.get("context_threshold", DEFAULT_CONTEXT_THRESHOLD)
        ),
        "enable_context_probing": llm_dict.get("enable_context_probing", True),
        "enable_fallback": llm_dict.get("enable_fallback", False),
    }

    return LLMConfig(**kwargs)


def get_embedding_config() -> EmbeddingConfig:
    """
    Get typed embedding configuration for vibelint.

    Returns:
    Validated EmbeddingConfig object
    """
    config = get_vibelint_config()
    embedding_dict = config.get("embeddings", {})

    kwargs = {
        "code_api_url": embedding_dict.get("code_api_url"),
        "natural_api_url": embedding_dict.get("natural_api_url"),
        "code_model": embedding_dict.get("code_model", "text-embedding-ada-002"),
        "natural_model": embedding_dict.get("natural_model", "text-embedding-ada-002"),
        "use_specialized_embeddings": embedding_dict.get("use_specialized_embeddings", True),
    }

    return EmbeddingConfig(**kwargs)


# All functions now use typed configuration - no legacy dict functions


def _get_env_float(key: str) -> Optional[float]:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_env_int(key: str) -> Optional[int]:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _get_env_bool(key: str) -> Optional[bool]:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        return value.lower() in ("true", "1", "yes", "on")
    return None


# Factory function for creating orchestrator
def create_llm_orchestrator_from_config():
    """Create LLM orchestrator from vibelint configuration."""
    from .llm_orchestrator import LLMOrchestrator

    config = {"llm": get_llm_config()}
    return LLMOrchestrator.from_config(config)


# Export main functions
__all__ = [
    "get_vibelint_config",
    "get_llm_config",
    "load_env_files",
    "find_project_root",
    "create_llm_orchestrator_from_config",
]
