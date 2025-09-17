"""
LLM Orchestrator for intelligent dual-LLM usage in vibelint.

This module provides agentic coordination between fast and slow LLMs,
automatically selecting the appropriate model based on task complexity,
context size, and performance requirements.

vibelint/src/vibelint/llm_orchestrator.py
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["LLMType", "TaskComplexity", "LegacyLLMRequest", "LLMOrchestrator", "OrchestratorConfig"]


class LLMType(Enum):
    """Available LLM types for different use cases."""

    FAST = "fast"  # Small context, fast responses (docstrings, quick analysis)
    ORCHESTRATOR = "orchestrator"  # Large context, complex reasoning (architecture, summarization)


class TaskComplexity(Enum):
    """Task complexity levels for LLM selection."""

    SIMPLE = "simple"  # Single function/class analysis, docstring generation
    MODERATE = "moderate"  # Multi-function analysis, pattern detection
    COMPLEX = "complex"  # Multi-file analysis, architectural decisions
    ORCHESTRATION = "orchestration"  # Planning, coordination, large context summarization


@dataclass
class LegacyLLMRequest:
    """Legacy request specification for LLM processing (deprecated, use llm_manager.LLMRequest)."""

    task_type: str
    content: str
    context_size: int
    complexity: TaskComplexity
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    requires_reasoning: bool = False
    requires_planning: bool = False
    timeout_seconds: int = 60


@dataclass
class OrchestratorConfig:
    """Configuration for the LLM orchestrator."""

    # Fast LLM configuration
    fast_api_base: str
    fast_model: str
    fast_max_context: int
    fast_max_tokens: int

    # Orchestrator LLM configuration
    orchestrator_api_base: str
    orchestrator_model: str
    orchestrator_max_context: int
    orchestrator_max_tokens: int

    # Selection thresholds
    context_threshold: int = 3000  # Switch to orchestrator above this context size
    complexity_threshold: TaskComplexity = TaskComplexity.MODERATE

    # Performance settings
    enable_caching: bool = True
    enable_fallback: bool = True
    max_retries: int = 2


class LLMOrchestrator:
    """Intelligent orchestrator for dual LLM usage."""

    def __init__(self, config: OrchestratorConfig):
        """Initialize orchestrator with dual LLM configuration.

        Args:
            config: Orchestrator configuration with both LLM settings
        """
        self.config = config
        self.performance_metrics = {
            LLMType.FAST: {"avg_latency": 0.0, "success_rate": 1.0, "call_count": 0},
            LLMType.ORCHESTRATOR: {"avg_latency": 0.0, "success_rate": 1.0, "call_count": 0},
        }

    def select_llm(self, request: LegacyLLMRequest) -> LLMType:
        """Intelligently select the appropriate LLM for a request.

        Selection criteria:
        - Context size: Use orchestrator for large contexts
        - Task complexity: Use orchestrator for complex reasoning
        - Performance history: Consider past success rates
        - Resource constraints: Prefer fast LLM when possible

        Args:
            request: LLM request specification

        Returns:
            Selected LLM type
        """
        # Rule 1: Context size threshold
        if request.context_size > self.config.context_threshold:
            logger.debug(
                f"Selecting orchestrator LLM: context size {request.context_size} > {self.config.context_threshold}"
            )
            return LLMType.ORCHESTRATOR

        # Rule 2: Task complexity threshold
        if request.complexity.value in [
            TaskComplexity.COMPLEX.value,
            TaskComplexity.ORCHESTRATION.value,
        ]:
            logger.debug(f"Selecting orchestrator LLM: high complexity task ({request.complexity})")
            return LLMType.ORCHESTRATOR

        # Rule 3: Reasoning/planning requirements
        if request.requires_reasoning or request.requires_planning:
            logger.debug("Selecting orchestrator LLM: requires advanced reasoning/planning")
            return LLMType.ORCHESTRATOR

        # Rule 4: Performance-based selection
        fast_metrics = self.performance_metrics[LLMType.FAST]
        orchestrator_metrics = self.performance_metrics[LLMType.ORCHESTRATOR]

        # If fast LLM has poor success rate, consider orchestrator
        if fast_metrics["success_rate"] < 0.8 and orchestrator_metrics["success_rate"] > 0.9:
            logger.debug("Selecting orchestrator LLM: fast LLM has poor success rate")
            return LLMType.ORCHESTRATOR

        # Default: Use fast LLM for simple tasks
        logger.debug("Selecting fast LLM: simple task with small context")
        return LLMType.FAST

    async def process_request(self, request: LegacyLLMRequest) -> Dict[str, Any]:
        """Process an LLM request using the most appropriate model.

        Args:
            request: LLM request specification

        Returns:
            Response dictionary with content and metadata
        """
        selected_llm = self.select_llm(request)

        start_time = time.time()
        try:
            if selected_llm == LLMType.FAST:
                response = await self._call_fast_llm(request)
            else:
                response = await self._call_orchestrator_llm(request)

            duration = time.time() - start_time
            self._update_metrics(selected_llm, duration, success=True)

            return {
                "content": response,
                "llm_used": selected_llm.value,
                "duration": duration,
                "context_size": request.context_size,
                "success": True,
            }

        except Exception as e:
            duration = time.time() - start_time
            self._update_metrics(selected_llm, duration, success=False)

            # Fallback strategy
            if self.config.enable_fallback and selected_llm == LLMType.FAST:
                logger.warning(f"Fast LLM failed ({e}), falling back to orchestrator")
                try:
                    response = await self._call_orchestrator_llm(request)
                    fallback_duration = time.time() - start_time

                    return {
                        "content": response,
                        "llm_used": f"{selected_llm.value}_fallback_to_{LLMType.ORCHESTRATOR.value}",
                        "duration": fallback_duration,
                        "context_size": request.context_size,
                        "success": True,
                        "fallback_used": True,
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            return {
                "content": None,
                "llm_used": selected_llm.value,
                "duration": duration,
                "context_size": request.context_size,
                "success": False,
                "error": str(e),
            }

    async def _call_fast_llm(self, request: LegacyLLMRequest) -> str:
        """Call the fast LLM for quick analysis tasks."""
        # Implementation would use langchain with fast LLM config
        # This is a placeholder for the actual LLM call
        raise NotImplementedError("Fast LLM implementation pending")

    async def _call_orchestrator_llm(self, request: LegacyLLMRequest) -> str:
        """Call the orchestrator LLM for complex analysis tasks."""
        # Implementation would use langchain with orchestrator config
        # This is a placeholder for the actual LLM call
        raise NotImplementedError("Orchestrator LLM implementation pending")

    def _update_metrics(self, llm_type: LLMType, duration: float, success: bool):
        """Update performance metrics for an LLM."""
        metrics = self.performance_metrics[llm_type]

        # Update call count
        metrics["call_count"] += 1

        # Update average latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if metrics["avg_latency"] == 0.0:
            metrics["avg_latency"] = duration
        else:
            metrics["avg_latency"] = alpha * duration + (1 - alpha) * metrics["avg_latency"]

        # Update success rate (exponential moving average)
        success_value = 1.0 if success else 0.0
        metrics["success_rate"] = alpha * success_value + (1 - alpha) * metrics["success_rate"]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for both LLMs."""
        return {
            "fast_llm": self.performance_metrics[LLMType.FAST].copy(),
            "orchestrator_llm": self.performance_metrics[LLMType.ORCHESTRATOR].copy(),
            "selection_stats": {
                "context_threshold": self.config.context_threshold,
                "complexity_threshold": self.config.complexity_threshold.value,
                "fallback_enabled": self.config.enable_fallback,
            },
        }


def create_orchestrator_from_config(vibelint_config: Dict[str, Any]) -> Optional[LLMOrchestrator]:
    """Create an orchestrator from vibelint configuration.

    Args:
        vibelint_config: Vibelint configuration dictionary

    Returns:
        Configured orchestrator or None if missing config
    """
    llm_config = vibelint_config.get("llm_analysis", {})
    orchestrator_config = vibelint_config.get("llm_orchestrator", {})

    if not llm_config.get("api_base_url") or not orchestrator_config.get("api_base_url"):
        logger.warning("Dual LLM configuration incomplete, orchestrator disabled")
        return None

    config = OrchestratorConfig(
        fast_api_base=llm_config["api_base_url"],
        fast_model=llm_config.get("model", "gpt-3.5-turbo"),
        fast_max_context=llm_config.get("max_context_tokens", 4000),
        fast_max_tokens=llm_config.get("max_tokens", 2048),
        orchestrator_api_base=orchestrator_config["api_base_url"],
        orchestrator_model=orchestrator_config.get("model", "llama3.2:latest"),
        orchestrator_max_context=orchestrator_config.get("max_context_tokens", 32000),
        orchestrator_max_tokens=orchestrator_config.get("max_tokens", 8192),
        context_threshold=llm_config.get("max_context_tokens", 4000) - 500,  # Leave buffer
        enable_caching=True,
        enable_fallback=True,
    )

    return LLMOrchestrator(config)
