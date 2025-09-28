"""
Consolidated LLM system for vibelint.

Manages dual LLMs, tracing, and dynamic validator generation:
- Fast: High-speed inference for quick tasks
- Orchestrator: Large context for complex reasoning
- Dynamic: On-demand validator generation from prompts

vibelint/src/vibelint/llm.py
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
# Load environment variables from .env files in order of preference:
# 1. Current working directory (.env)
# 2. User's home directory (~/.vibelint.env)
# 3. Project root directory (.env)
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


def _load_env_files():
    """Load environment variables from multiple possible locations."""
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path.home() / ".vibelint.env",  # User home directory
        Path(__file__).parent.parent.parent / ".env",  # Project root (fallback)
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


_load_env_files()

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_CONTEXT_THRESHOLD = 3000
DEFAULT_FAST_TEMPERATURE = 0.1
DEFAULT_FAST_MAX_TOKENS = 2048
DEFAULT_ORCHESTRATOR_TEMPERATURE = 0.2
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 8192
ORCHESTRATOR_TIMEOUT_SECONDS = 600
FAST_TIMEOUT_SECONDS = 30
TOKEN_ESTIMATION_DIVISOR = 4

__all__ = ["LLMRole", "LLMManager", "LLMRequest", "create_llm_manager"]


class LLMRole(Enum):
    """LLM roles for different types of tasks."""

    FAST = "fast"  # High-speed inference, small context
    ORCHESTRATOR = "orchestrator"  # Large context, complex reasoning


@dataclass
class LLMRequest:
    """Simple request specification for LLM processing."""

    content: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    structured_output: Optional[Dict[str, Any]] = None  # JSON schema for structured responses
    # If structured_output is None, expects unstructured natural language response


class LLMManager:
    """Simple manager for dual LLM setup."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with vibelint configuration.

        Args:
            config: Vibelint configuration dictionary with [tool.vibelint.llm] section
        """
        llm_config = config.get("llm", {})

        # Fast LLM configuration with fallback logic
        self.fast_config = self._build_llm_config(
            llm_config,
            "fast",
            {
                "temperature": DEFAULT_FAST_TEMPERATURE,
                "max_tokens": DEFAULT_FAST_MAX_TOKENS,
                "api_key_env": "FAST_LLM_API_KEY",
            },
        )

        # Orchestrator LLM configuration with fallback logic
        self.orchestrator_config = self._build_llm_config(
            llm_config,
            "orchestrator",
            {
                "temperature": DEFAULT_ORCHESTRATOR_TEMPERATURE,
                "max_tokens": DEFAULT_ORCHESTRATOR_MAX_TOKENS,
                "api_key_env": "ORCHESTRATOR_LLM_API_KEY",
            },
        )

        # Routing configuration
        self.context_threshold = llm_config.get("context_threshold", DEFAULT_CONTEXT_THRESHOLD)
        self.enable_fallback = llm_config.get("enable_fallback", False)

        # Session for HTTP requests
        self.session = requests.Session()
        # Note: timeout is set per-request rather than on session

    def _build_llm_config(
        self, llm_config: Dict[str, Any], role: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build LLM configuration with intelligent fallback logic.

        Supports multiple deployment scenarios:
        1. Two separate endpoints (fast_api_url + orchestrator_api_url)
        2. Single endpoint, two models (api_url + fast_model + orchestrator_model)
        3. Single endpoint, single model (api_url + model)
        4. Mixed scenarios with role-specific overrides
        """
        config = {}

        # Priority order for API URL:
        # 1. Role-specific URL (fast_api_url/orchestrator_api_url)
        # 2. Generic URL (api_url)
        config["api_url"] = llm_config.get(f"{role}_api_url") or llm_config.get("api_url")

        # Priority order for model:
        # 1. Role-specific model (fast_model/orchestrator_model)
        # 2. Generic model (model)
        config["model"] = llm_config.get(f"{role}_model") or llm_config.get("model")

        # Priority order for API key:
        # 1. Role-specific env var (FAST_LLM_API_KEY/ORCHESTRATOR_LLM_API_KEY)
        # 2. Generic env var (LLM_API_KEY)
        # 3. OpenAI fallback (OPENAI_API_KEY)
        config["api_key"] = (
            os.getenv(defaults["api_key_env"])
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        # Role-specific settings with fallbacks
        config["temperature"] = (
            llm_config.get(f"{role}_temperature")
            or llm_config.get("temperature")
            or defaults["temperature"]
        )

        config["max_tokens"] = (
            llm_config.get(f"{role}_max_tokens")
            or llm_config.get("max_tokens")
            or defaults["max_tokens"]
        )

        return config

    def select_llm(self, request: LLMRequest) -> LLMRole:
        """Select appropriate LLM based on hard constraints.

        Routes based on actual LLM hard limits:
        - Context window size (input tokens)
        - Max output tokens
        - Use cheapest/fastest LLM that can handle the request
        """
        content_size = len(request.content)
        max_tokens = request.max_tokens or 50

        # Get LLM hard limits from configs
        fast_max_tokens = self.fast_config.get("max_tokens", DEFAULT_FAST_MAX_TOKENS)
        fast_max_context = self.fast_config.get("fast_max_context_tokens", 1000)  # Fast LLMs typically have small context

        orchestrator_max_tokens = self.orchestrator_config.get("max_tokens", DEFAULT_ORCHESTRATOR_MAX_TOKENS)
        # Orchestrator typically has much larger context window

        fast_available = bool(self.fast_config.get("api_url"))
        orchestrator_available = bool(self.orchestrator_config.get("api_url"))

        # Estimate input tokens (rough approximation: 4 chars per token)
        estimated_input_tokens = content_size // 4

        # Hard constraint: If request exceeds fast LLM's output token limit
        if max_tokens > fast_max_tokens:
            if orchestrator_available:
                logger.debug(f"Routing to orchestrator: output tokens ({max_tokens}) > fast limit ({fast_max_tokens})")
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(f"Request needs {max_tokens} tokens but orchestrator unavailable, truncating to fast LLM limit")
                return LLMRole.FAST

        # Hard constraint: If input exceeds fast LLM's context window
        if estimated_input_tokens > fast_max_context:
            if orchestrator_available:
                logger.debug(f"Routing to orchestrator: input tokens (~{estimated_input_tokens}) > fast context ({fast_max_context})")
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(f"Large input (~{estimated_input_tokens} tokens) but orchestrator unavailable, truncating to fast LLM")
                return LLMRole.FAST

        # No hard constraints violated - use fast LLM (cheaper/faster)
        if fast_available:
            logger.debug(f"Routing to fast: within limits (input~{estimated_input_tokens}, output={max_tokens})")
            return LLMRole.FAST
        elif orchestrator_available:
            logger.debug(f"Routing to orchestrator: fast LLM unavailable")
            return LLMRole.ORCHESTRATOR
        else:
            raise ValueError("No LLMs configured - need either fast_api_url or orchestrator_api_url in config")

    async def process_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process request using the appropriate LLM."""
        selected_llm = self.select_llm(request)

        # Check if required LLM is configured
        if selected_llm == LLMRole.FAST and not self.fast_config.get("api_url"):
            raise ValueError(
                f"Fast LLM required but not configured. "
                f"Add [tool.vibelint.llm] fast_api_url and fast_model to pyproject.toml"
            )
        elif selected_llm == LLMRole.ORCHESTRATOR and not self.orchestrator_config.get("api_url"):
            raise ValueError(
                f"Orchestrator LLM required but not configured. "
                f"Add [tool.vibelint.llm] orchestrator_api_url and orchestrator_model to pyproject.toml"
            )

        try:
            if selected_llm == LLMRole.FAST:
                return await self._call_fast_llm(request)
            else:
                return await self._call_orchestrator_llm(request)

        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            # Configured but unavailable - fail fast
            logger.error(
                f"{selected_llm.value} LLM configured but unavailable - aborting analysis: {e}"
            )
            raise RuntimeError(
                f"LLM analysis failed: {selected_llm.value} model configured but unavailable. "
                f"Check model server status and network connectivity. Error: {e}"
            ) from e

    async def _call_fast_llm(self, request: LLMRequest) -> Dict[str, Any]:
        """Call the fast LLM."""
        return await self._make_api_call(self.fast_config, request, LLMRole.FAST)

    async def _call_orchestrator_llm(self, request: LLMRequest) -> Dict[str, Any]:
        """Call the orchestrator (large context) LLM."""
        return await self._make_api_call(self.orchestrator_config, request, LLMRole.ORCHESTRATOR)

    async def _make_api_call(
        self, llm_config: Dict[str, Any], request: LLMRequest, role: LLMRole
    ) -> Dict[str, Any]:
        """Make API call to specified LLM."""
        start_time = time.time()

        if not llm_config.get("api_url"):
            raise ValueError(f"No API URL configured for {role.value} LLM")

        url = f"{llm_config['api_url'].rstrip('/')}/v1/chat/completions"

        headers = {}
        if llm_config.get("api_key"):
            headers["Authorization"] = f"Bearer {llm_config['api_key']}"

        payload = {
            "model": llm_config["model"],
            "messages": [{"role": "user", "content": request.content}],
            "temperature": request.temperature or llm_config["temperature"],
            "max_tokens": request.max_tokens or llm_config["max_tokens"],
            "stream": False,
        }

        # Add structured output if requested
        if request.structured_output:
            if "json_schema" in request.structured_output:
                # OpenAI-style structured output with schema
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.structured_output["json_schema"]
                }
            else:
                # Simple JSON mode
                payload["response_format"] = {"type": "json_object"}

        # Debug: Log the request details
        logger.info(f"LLM Request: {role.value} to {url}")
        logger.info(f"Model: {payload['model']}, Max tokens: {payload['max_tokens']}")

        # Set timeout based on LLM role - orchestrator needs more time for large prompts
        timeout_seconds = (
            ORCHESTRATOR_TIMEOUT_SECONDS if role == LLMRole.ORCHESTRATOR else FAST_TIMEOUT_SECONDS
        )

        response = self.session.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()

        data = response.json()
        duration = time.time() - start_time

        if "choices" not in data or not data["choices"]:
            raise ValueError("Invalid LLM response format")

        return {
            "content": data["choices"][0]["message"]["content"],
            "llm_used": role.value,
            "duration_seconds": duration,
            "input_tokens": len(request.content) // TOKEN_ESTIMATION_DIVISOR,  # Rough estimate
            "success": True,
        }

    def is_llm_available(self, role: LLMRole) -> bool:
        """Check if a specific LLM is configured and available."""
        if role == LLMRole.FAST:
            return bool(self.fast_config.get("api_url") and self.fast_config.get("model"))
        else:
            return bool(
                self.orchestrator_config.get("api_url") and self.orchestrator_config.get("model")
            )

    def get_available_features(self) -> Dict[str, bool]:
        """Get which AI features are available based on LLM configuration."""
        fast_available = self.is_llm_available(LLMRole.FAST)
        orchestrator_available = self.is_llm_available(LLMRole.ORCHESTRATOR)
        any_llm_available = fast_available or orchestrator_available

        return {
            # LLM-powered features
            "architecture_analysis": orchestrator_available,  # Requires orchestrator LLM
            "docstring_generation": fast_available,  # Can use fast LLM
            "code_smell_detection": fast_available,  # Can use fast LLM
            "coverage_assessment": orchestrator_available,  # Requires orchestrator LLM
            "llm_validation": any_llm_available,  # Any LLM works
            # Embedding-only features (no LLM required)
            "semantic_similarity": True,  # Always available (uses local embeddings)
            "embedding_clustering": True,  # Always available (uses local embeddings)
            "duplicate_detection": True,  # Always available (uses local embeddings)
        }

    def get_status(self) -> Dict[str, bool | int | Dict[str, bool]]:
        """Get status of both LLMs."""
        return {
            "fast_configured": self.is_llm_available(LLMRole.FAST),
            "orchestrator_configured": self.is_llm_available(LLMRole.ORCHESTRATOR),
            "context_threshold": self.context_threshold,
            "fallback_enabled": False,  # Always disabled for predictable behavior
            "available_features": self.get_available_features(),
        }

    def process_request_sync(self, request: LLMRequest) -> Dict[str, Any]:
        """Synchronous wrapper for process_request to support legacy code."""
        import asyncio
        try:
            # Always create a new event loop for sync calls
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.process_request(request))
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error(f"Sync LLM request failed: {e}")
            return {
                "content": f"LLM sync call failed: {e}",
                "llm_used": "error",
                "duration_seconds": 0,
                "input_tokens": 0,
                "success": False,
                "error": str(e)
            }


def create_llm_manager(config: Dict[str, Any]) -> Optional[LLMManager]:
    """Create LLM manager from vibelint configuration.

    Always returns an LLMManager instance, even if no LLMs are configured.
    This allows embedding-only analysis to work without LLM endpoints.
    """
    if "llm" not in config:
        logger.info("No LLM configuration found - embedding-only analysis will be available")
        # Create empty config for embedding-only mode
        config = {"llm": {}}

    return LLMManager(config)
