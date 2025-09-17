"""
Consolidated LLM system for vibelint.

Manages dual LLMs, tracing, and dynamic validator generation:
- Fast: High-speed inference for quick tasks
- Orchestrator: Large context for complex reasoning
- Dynamic: On-demand validator generation from prompts

vibelint/src/vibelint/llm.py
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

__all__ = ["LLMRole", "LLMManager", "LLMRequest", "create_llm_manager"]


class LLMRole(Enum):
    """LLM roles for different types of tasks."""

    FAST = "fast"  # High-speed inference, small context
    ORCHESTRATOR = "orchestrator"  # Large context, complex reasoning


@dataclass
class LLMRequest:
    """Simple request specification for LLM processing."""

    content: str
    task_type: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class LLMManager:
    """Simple manager for dual LLM setup."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with vibelint configuration.

        Args:
            config: Vibelint configuration dictionary with [tool.vibelint.llm] section
        """
        llm_config = config.get("llm", {})

        # Fast LLM configuration
        self.fast_config = {
            "api_url": llm_config.get("fast_api_url"),
            "model": llm_config.get("fast_model"),
            "temperature": llm_config.get("fast_temperature", 0.1),
            "max_tokens": llm_config.get("fast_max_tokens", 2048),
        }

        # Orchestrator LLM configuration
        self.orchestrator_config = {
            "api_url": llm_config.get("orchestrator_api_url"),
            "model": llm_config.get("orchestrator_model"),
            "temperature": llm_config.get("orchestrator_temperature", 0.2),
            "max_tokens": llm_config.get("orchestrator_max_tokens", 8192),
        }

        # Routing configuration
        self.context_threshold = llm_config.get("context_threshold", 3000)
        self.enable_fallback = llm_config.get("enable_fallback", False)

        # Session for HTTP requests
        self.session = requests.Session()
        # Note: timeout is set per-request rather than on session

    def select_llm(self, request: LLMRequest) -> LLMRole:
        """Select appropriate LLM based on request characteristics.

        Simple routing logic:
        - Use orchestrator for large content (>context_threshold)
        - Use orchestrator for complex tasks (architecture, planning)
        - Use fast for everything else
        """
        content_size = len(request.content)

        # Size-based routing
        if content_size > self.context_threshold:
            return LLMRole.ORCHESTRATOR

        # Task-based routing
        complex_tasks = ["architecture", "planning", "summarization", "multi_file"]
        if any(task in request.task_type.lower() for task in complex_tasks):
            return LLMRole.ORCHESTRATOR

        return LLMRole.FAST

    async def process_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process request using the appropriate LLM."""
        selected_llm = self.select_llm(request)

        # Check if required LLM is configured
        if selected_llm == LLMRole.FAST and not self.fast_config.get("api_url"):
            raise ValueError(
                f"Fast LLM required for task '{request.task_type}' but not configured. "
                f"Add [tool.vibelint.llm] fast_api_url and fast_model to pyproject.toml"
            )
        elif selected_llm == LLMRole.ORCHESTRATOR and not self.orchestrator_config.get("api_url"):
            raise ValueError(
                f"Orchestrator LLM required for task '{request.task_type}' but not configured. "
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

        payload = {
            "model": llm_config["model"],
            "messages": [{"role": "user", "content": request.content}],
            "temperature": request.temperature or llm_config["temperature"],
            "max_tokens": request.max_tokens or llm_config["max_tokens"],
            "stream": False,
        }

        response = self.session.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        duration = time.time() - start_time

        if "choices" not in data or not data["choices"]:
            raise ValueError("Invalid LLM response format")

        return {
            "content": data["choices"][0]["message"]["content"],
            "llm_used": role.value,
            "duration_seconds": duration,
            "input_tokens": len(request.content) // 4,  # Rough estimate
            "success": True,
        }

    def is_llm_available(self, role: LLMRole) -> bool:
        """Check if a specific LLM is configured and available."""
        if role == LLMRole.FAST:
            return bool(self.fast_config.get("api_url"))
        else:
            return bool(self.orchestrator_config.get("api_url"))

    def get_available_features(self) -> Dict[str, bool]:
        """Get which AI features are available based on LLM configuration."""
        fast_available = self.is_llm_available(LLMRole.FAST)
        orchestrator_available = self.is_llm_available(LLMRole.ORCHESTRATOR)

        return {
            "architecture_analysis": orchestrator_available,  # Requires orchestrator
            "docstring_generation": fast_available,  # Can use fast
            "code_smell_detection": fast_available,  # Can use fast
            "coverage_assessment": orchestrator_available,  # Requires orchestrator
            "semantic_analysis": orchestrator_available,  # Requires orchestrator
            "simple_validation": fast_available or orchestrator_available,  # Either works
        }

    def get_status(self) -> Dict[str, Any]:
        """Get status of both LLMs."""
        return {
            "fast_configured": self.is_llm_available(LLMRole.FAST),
            "orchestrator_configured": self.is_llm_available(LLMRole.ORCHESTRATOR),
            "context_threshold": self.context_threshold,
            "fallback_enabled": False,  # Always disabled for predictable behavior
            "available_features": self.get_available_features(),
        }


def create_llm_manager(config: Dict[str, Any]) -> Optional[LLMManager]:
    """Create LLM manager from vibelint configuration.

    Returns None if no LLM configuration found.
    """
    if "llm" not in config:
        logger.warning("No LLM configuration found")
        return None

    llm_config = config["llm"]

    # Check if at least one LLM is configured
    has_fast = bool(llm_config.get("fast_api_url"))
    has_orchestrator = bool(llm_config.get("orchestrator_api_url"))

    if not has_fast and not has_orchestrator:
        logger.warning("No LLM endpoints configured")
        return None

    return LLMManager(config)
