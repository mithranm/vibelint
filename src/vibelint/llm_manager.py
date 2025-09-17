"""
Simple dual LLM manager for vibelint.

Manages two LLMs with different roles:
- Fast: High-speed inference for quick tasks (docstrings, simple analysis)
- Orchestrator: Large context for complex reasoning (architecture, summarization)

vibelint/src/vibelint/llm_manager.py
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

__all__ = ["LLMRole", "LLMManager", "LLMRequest"]


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
        self.enable_fallback = llm_config.get("enable_fallback", True)

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

        try:
            if selected_llm == LLMRole.FAST:
                return await self._call_fast_llm(request)
            else:
                return await self._call_orchestrator_llm(request)

        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            if self.enable_fallback:
                logger.warning(f"{selected_llm.value} LLM failed, trying fallback: {e}")

                # Try the other LLM
                fallback_role = (
                    LLMRole.ORCHESTRATOR if selected_llm == LLMRole.FAST else LLMRole.FAST
                )

                try:
                    if fallback_role == LLMRole.FAST:
                        result = await self._call_fast_llm(request)
                    else:
                        result = await self._call_orchestrator_llm(request)

                    result["fallback_used"] = True
                    result["original_llm"] = selected_llm.value
                    return result

                except (requests.exceptions.RequestException, ValueError, KeyError) as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            raise e

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

    def get_status(self) -> Dict[str, Union[bool, int]]:
        """Get status of both LLMs."""
        return {
            "fast_configured": bool(self.fast_config.get("api_url")),
            "orchestrator_configured": bool(self.orchestrator_config.get("api_url")),
            "context_threshold": self.context_threshold,
            "fallback_enabled": self.enable_fallback,
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
