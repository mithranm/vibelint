"""
Vibelint LLM Orchestrator

Configurable LLM system supporting multiple inference backends (vLLM, llama.cpp, OpenAI)
with dual LLM architecture (fast + orchestrator) for code analysis tasks.

This is the authoritative LLM implementation - kaia guardrails imports from here.

Features:
- Backend abstraction (vLLM, llama.cpp, OpenAI)
- Structured JSON generation
- Intelligent routing between fast/orchestrator LLMs
- Configuration management
- Request/response standardization
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Supported LLM inference backends."""

    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"


class LLMRole(Enum):
    """LLM roles for different types of tasks."""

    FAST = "fast"  # High-speed inference, small context
    ORCHESTRATOR = "orchestrator"  # Large context, complex reasoning


@dataclass
class LLMRequest:
    """Request specification for LLM processing."""

    content: str
    task_type: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    require_json: bool = False  # Request structured JSON response
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM processing."""

    content: str
    role_used: LLMRole
    backend_used: LLMBackend
    tokens_used: Optional[int] = None
    processing_time: float = 0.0
    parsed_json: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None


class LLMBackendClient(ABC):
    """Abstract base class for LLM backend clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get("api_url")
        self.model = config.get("model")
        self.api_key = config.get("api_key")

    @abstractmethod
    def make_request(self, request: LLMRequest) -> LLMResponse:
        """Make a request to the LLM backend."""
        pass

    @abstractmethod
    def supports_structured_json(self) -> bool:
        """Whether this backend supports structured JSON generation."""
        pass


class VLLMClient(LLMBackendClient):
    """vLLM backend client with guided JSON support."""

    def supports_structured_json(self) -> bool:
        return True

    def make_request(self, request: LLMRequest) -> LLMResponse:
        """Make request to vLLM server with optional structured output."""
        start_time = time.time()

        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.content})

        # Base request data
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature or 0.1,
            "max_tokens": request.max_tokens or 2048,
        }

        # Add structured JSON if requested (vLLM format with json_schema)
        if request.require_json:
            # Define flexible schema for any JSON response
            json_schema = {
                "type": "object",
                "properties": {},  # Allow any JSON structure
                "additionalProperties": True
            }

            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "flexible_response",
                    "schema": json_schema
                }
            }

            # Use system message approach for better JSON generation
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": "Reason briefly step-by-step, then output only JSON matching the schema."
                })
            else:
                # Enhance existing system message
                for msg in messages:
                    if msg["role"] == "system":
                        msg["content"] += " Output only JSON matching the schema."
                        break

        try:
            # Make HTTP request
            req = Request(
                f"{self.api_url}/v1/chat/completions",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]

                # Parse JSON if requested
                parsed_json = None
                if request.require_json:
                    try:
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from vLLM: {e}")

                return LLMResponse(
                    content=content,
                    role_used=LLMRole.FAST,  # Will be corrected by orchestrator
                    backend_used=LLMBackend.VLLM,
                    processing_time=time.time() - start_time,
                    parsed_json=parsed_json,
                    raw_response=result,
                )

        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise


class LlamaCppClient(LLMBackendClient):
    """llama.cpp backend client with structured JSON support."""

    def supports_structured_json(self) -> bool:
        return True

    def make_request(self, request: LLMRequest) -> LLMResponse:
        """Make request to llama.cpp server with optional JSON mode."""
        start_time = time.time()

        # Build messages for chat completions API
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.content})

        # Base request data
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature or 0.1,
            "max_tokens": request.max_tokens or 2048,
        }

        # Add JSON mode if requested (llama.cpp format)
        if request.require_json:
            data["response_format"] = {"type": "json_object"}

            # Use system message approach for better JSON generation
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": "Reason briefly step-by-step with low effort, then output only valid JSON."
                })
            else:
                # Enhance existing system message
                for msg in messages:
                    if msg["role"] == "system":
                        msg["content"] += " Output only valid JSON format."
                        break

        try:
            # Make HTTP request
            req = Request(
                f"{self.api_url}/v1/chat/completions",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]

                # Parse JSON if requested
                parsed_json = None
                if request.require_json:
                    try:
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from llama.cpp: {e}")

                return LLMResponse(
                    content=content,
                    role_used=LLMRole.FAST,  # Will be corrected by orchestrator
                    backend_used=LLMBackend.LLAMACPP,
                    processing_time=time.time() - start_time,
                    parsed_json=parsed_json,
                    raw_response=result,
                )

        except Exception as e:
            logger.error(f"llama.cpp request failed: {e}")
            raise


class OpenAIClient(LLMBackendClient):
    """OpenAI backend client with structured outputs."""

    def supports_structured_json(self) -> bool:
        return True

    def make_request(self, request: LLMRequest) -> LLMResponse:
        """OpenAI client not implemented - use vLLM or llama.cpp instead."""
        raise NotImplementedError("OpenAI client not implemented - configure vLLM or llama.cpp backends instead")


class LLMOrchestrator:
    """Main orchestrator for managing dual LLM setup with configurable backends."""

    def __init__(
        self,
        fast_client: Optional[LLMBackendClient] = None,
        orchestrator_client: Optional[LLMBackendClient] = None,
        context_threshold: int = 3000,
    ):
        self.fast_client = fast_client
        self.orchestrator_client = orchestrator_client
        self.context_threshold = context_threshold

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMOrchestrator":
        """Create orchestrator from configuration dict."""
        llm_config = config.get("llm", {})

        # Extract context threshold
        context_threshold = llm_config.get("context_threshold", 3000)

        # Build fast LLM client
        fast_client = None
        if llm_config.get("fast_api_url"):
            fast_config = {
                "api_url": llm_config["fast_api_url"],
                "model": llm_config.get("fast_model"),
                "api_key": llm_config.get("fast_api_key") or os.getenv("FAST_LLM_API_KEY"),
                "backend": llm_config.get("fast_backend", "vllm"),
            }
            fast_client = cls._create_client(fast_config)

        # Build orchestrator LLM client
        orchestrator_client = None
        if llm_config.get("orchestrator_api_url"):
            orchestrator_config = {
                "api_url": llm_config["orchestrator_api_url"],
                "model": llm_config.get("orchestrator_model"),
                "api_key": llm_config.get("orchestrator_api_key")
                or os.getenv("ORCHESTRATOR_LLM_API_KEY"),
                "backend": llm_config.get("orchestrator_backend", "llamacpp"),
            }
            orchestrator_client = cls._create_client(orchestrator_config)

        return cls(fast_client, orchestrator_client, context_threshold)

    @staticmethod
    def _create_client(config: Dict[str, Any]) -> LLMBackendClient:
        """Create appropriate backend client based on configuration."""
        backend = config.get("backend", "vllm").lower()

        if backend == "vllm":
            return VLLMClient(config)
        elif backend == "llamacpp":
            return LlamaCppClient(config)
        elif backend == "openai":
            return OpenAIClient(config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def select_llm_role(self, request: LLMRequest) -> LLMRole:
        """Select appropriate LLM role based on request characteristics."""
        content_size = len(request.content)

        # Size-based routing
        if content_size > self.context_threshold:
            return LLMRole.ORCHESTRATOR

        # Task-based routing
        complex_tasks = [
            "architecture",
            "planning",
            "summarization",
            "multi_file",
            "compliance",
            "analysis",
            "refactoring",
            "validation",
        ]
        if any(task in request.task_type.lower() for task in complex_tasks):
            return LLMRole.ORCHESTRATOR

        return LLMRole.FAST

    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process request using appropriate LLM with intelligent fallback."""
        selected_role = self.select_llm_role(request)

        # Determine primary and fallback clients
        primary_client = None
        fallback_client = None
        primary_role = selected_role
        fallback_role = None

        if selected_role == LLMRole.FAST:
            if self.fast_client:
                primary_client = self.fast_client
                primary_role = LLMRole.FAST
            if self.orchestrator_client:
                fallback_client = self.orchestrator_client
                fallback_role = LLMRole.ORCHESTRATOR
        else:  # ORCHESTRATOR
            if self.orchestrator_client:
                primary_client = self.orchestrator_client
                primary_role = LLMRole.ORCHESTRATOR
            if self.fast_client:
                fallback_client = self.fast_client
                fallback_role = LLMRole.FAST

        # Check if we have any clients configured
        if not primary_client and not fallback_client:
            raise ValueError("No LLM clients configured")

        # If no primary, use fallback as primary
        if not primary_client:
            primary_client = fallback_client
            primary_role = fallback_role
            fallback_client = None
            fallback_role = None

        # Attempt primary request
        last_exception = None
        try:
            logger.debug(f"Attempting primary request with {primary_role.value if primary_role else 'unknown'} LLM")
            response = primary_client.make_request(request)
            if primary_role:
                response.role_used = primary_role

            # Validate the response for structured JSON requests
            if request.require_json:
                if response.parsed_json is None:
                    raise ValueError("Primary LLM failed to produce valid JSON")
                # Check if JSON has required compliance fields
                if request.task_type == "compliance_assessment":
                    required_fields = {"score", "compliant", "violations", "reasoning"}
                    if not all(field in response.parsed_json for field in required_fields):
                        raise ValueError("Primary LLM JSON missing required compliance fields")

            logger.debug(f"Primary {primary_role.value if primary_role else 'unknown'} LLM request successful")
            return response

        except Exception as e:
            logger.warning(f"Primary {primary_role.value if primary_role else 'unknown'} LLM failed: {e}")
            last_exception = e

        # Try fallback if available
        if fallback_client:
            try:
                logger.info(f"Falling back to {fallback_role.value if fallback_role else 'unknown'} LLM")
                response = fallback_client.make_request(request)
                if fallback_role:
                    response.role_used = fallback_role

                # Validate fallback response for structured JSON requests
                if request.require_json:
                    if response.parsed_json is None:
                        raise ValueError("Fallback LLM failed to produce valid JSON")
                    # Check if JSON has required compliance fields
                    if request.task_type == "compliance_assessment":
                        required_fields = {"score", "compliant", "violations", "reasoning"}
                        if not all(field in response.parsed_json for field in required_fields):
                            raise ValueError("Fallback LLM JSON missing required compliance fields")

                logger.info(f"Fallback {fallback_role.value if fallback_role else 'unknown'} LLM request successful")
                return response

            except Exception as fallback_error:
                logger.error(f"Fallback {fallback_role.value if fallback_role else 'unknown'} LLM also failed: {fallback_error}")
                # Create composite error message
                error_msg = (
                    f"Both LLMs failed. Primary ({primary_role.value if primary_role else 'unknown'}): {last_exception}. "
                    f"Fallback ({fallback_role.value if fallback_role else 'unknown'}): {fallback_error}"
                )
                raise RuntimeError(error_msg) from fallback_error
        else:
            # No fallback available, re-raise original error
            logger.error(
                f"No fallback available, primary {primary_role.value if primary_role else 'unknown'} LLM failed: {last_exception}"
            )
            raise RuntimeError(
                f"Primary {primary_role.value if primary_role else 'unknown'} LLM failed and no fallback configured: {last_exception}"
            ) from last_exception


# Factory function
def create_llm_orchestrator(config: Dict[str, Any]) -> LLMOrchestrator:
    """Factory function to create LLM orchestrator from configuration."""
    return LLMOrchestrator.from_config(config)


# Export all public interfaces
__all__ = [
    "LLMOrchestrator",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "LLMBackend",
    "LLMBackendClient",
    "VLLMClient",
    "LlamaCppClient",
    "OpenAIClient",
    "create_llm_orchestrator",
]
