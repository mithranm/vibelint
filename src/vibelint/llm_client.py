"""Consolidated LLM system for vibelint.

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

__all__ = [
    "LLMRole",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMBackendConfig",
    "LogEntry",
    "APIPayload",
    "FeatureAvailability",
    "LLMStatus",
    "create_llm_manager",
]


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
    system_prompt: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None  # JSON schema for structured responses
    # If structured_output is None, expects unstructured natural language response


@dataclass
class LLMResponse:
    """Typed response from LLM processing."""

    content: str
    success: bool
    llm_used: str
    duration_seconds: float
    input_tokens: int
    reasoning_content: str = ""
    error: Optional[str] = None


@dataclass
class LLMBackendConfig:
    """Configuration for a single LLM backend."""

    backend: str
    api_url: str
    model: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    max_context_tokens: int


@dataclass
class LogEntry:
    """Log entry for LLM request/response pairs."""

    type: str
    timestamp: str
    llm_used: str
    request_content_length: int
    request_content_preview: str
    request_max_tokens: Optional[int]
    request_temperature: Optional[float]
    response_success: bool
    response_content_length: int
    response_content_preview: str
    response_duration_seconds: float
    response_error: Optional[str]


@dataclass
class APIPayload:
    """API payload for LLM requests."""

    model: str
    messages: list[Dict[str, str]]
    temperature: float
    max_tokens: int
    stream: bool
    response_format: Optional[Dict[str, Any]] = None
    grammar: Optional[str] = None


@dataclass
class FeatureAvailability:
    """Feature availability based on LLM configuration."""

    architecture_analysis: bool
    docstring_generation: bool
    code_smell_detection: bool
    coverage_assessment: bool
    llm_validation: bool
    semantic_similarity: bool
    embedding_clustering: bool
    duplicate_detection: bool


@dataclass
class LLMStatus:
    """Status of LLM manager."""

    fast_configured: bool
    orchestrator_configured: bool
    context_threshold: int
    fallback_enabled: bool
    available_features: FeatureAvailability


class LLMClient:
    """Simple manager for dual LLM setup."""

    def __init__(self, config: Optional["LLMConfig"] = None):
        """Initialize with vibelint configuration.

        Args:
            config: Optional typed LLMConfig - if None, loads from config files

        """
        from vibelint.config import get_llm_config

        # Load typed configuration
        self.llm_config = config if config is not None else get_llm_config()

        # Build configs for fast and orchestrator using typed config
        self.fast_config = self._build_fast_config(self.llm_config)
        self.orchestrator_config = self._build_orchestrator_config(self.llm_config)

        # Routing configuration from typed config
        self.context_threshold = self.llm_config.context_threshold
        self.enable_fallback = self.llm_config.enable_fallback

        # Session for HTTP requests
        self.session = requests.Session()
        # Note: timeout is set per-request rather than on session

        # Optional logging callback for external logging (e.g., JSONL workflow logs)
        self.log_callback = None

    def set_log_callback(self, callback):
        """Set a callback function for logging LLM requests/responses.

        Callback signature: callback(log_entry: LogEntry) -> None
        """
        self.log_callback = callback

    def _log_request_response(self, request: LLMRequest, response: LLMResponse, llm_used: str):
        """Log request/response pair if callback is registered."""
        if self.log_callback:
            try:
                log_entry = LogEntry(
                    type="llm_call",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    llm_used=llm_used,
                    request_content_length=len(request.content),
                    request_content_preview=(
                        request.content[:500] + "..."
                        if len(request.content) > 500
                        else request.content
                    ),
                    request_max_tokens=request.max_tokens,
                    request_temperature=request.temperature,
                    response_success=response.success,
                    response_content_length=len(response.content),
                    response_content_preview=(
                        response.content[:500] + "..."
                        if len(response.content) > 500
                        else response.content
                    ),
                    response_duration_seconds=response.duration_seconds,
                    response_error=response.error,
                )
                self.log_callback(log_entry)
            except Exception as e:
                logger.debug(f"Log callback failed: {e}")

    def _build_fast_config(self, llm_config) -> LLMBackendConfig:
        """Build configuration for fast LLM from typed config."""
        return LLMBackendConfig(
            backend=llm_config.fast_backend,
            api_url=llm_config.fast_api_url,
            model=llm_config.fast_model,
            api_key=llm_config.fast_api_key,
            temperature=llm_config.fast_temperature,
            max_tokens=llm_config.fast_max_tokens,
            max_context_tokens=llm_config.fast_max_context_tokens,
        )

    def _build_orchestrator_config(self, llm_config) -> LLMBackendConfig:
        """Build configuration for orchestrator LLM from typed config."""
        return LLMBackendConfig(
            backend=llm_config.orchestrator_backend,
            api_url=llm_config.orchestrator_api_url,
            model=llm_config.orchestrator_model,
            api_key=llm_config.orchestrator_api_key,
            temperature=llm_config.orchestrator_temperature,
            max_tokens=llm_config.orchestrator_max_tokens,
            max_context_tokens=llm_config.orchestrator_max_context_tokens,
        )

    def select_llm(self, request: LLMRequest) -> LLMRole:
        """Select appropriate LLM based on hard constraints.

        Routes based on actual LLM hard limits:
        - Context window size (input tokens)
        - Max output tokens
        - Use cheapest/fastest LLM that can handle the request
        """
        content_size = len(request.content)
        max_tokens = request.max_tokens or 50

        # Get LLM hard limits from typed config
        fast_max_tokens = self.llm_config.fast_max_tokens
        fast_max_context = (
            self.llm_config.fast_max_context_tokens or 1000
        )  # Default if not specified

        orchestrator_max_tokens = self.llm_config.orchestrator_max_tokens
        # Orchestrator typically has much larger context window

        fast_available = bool(self.fast_config.api_url)
        orchestrator_available = bool(self.orchestrator_config.api_url)

        # Estimate input tokens (conservative: 3 chars per token)
        estimated_input_tokens = content_size // 3

        # Hard constraint: If request exceeds fast LLM's output token limit
        if max_tokens > fast_max_tokens:
            if orchestrator_available:
                logger.debug(
                    f"Routing to orchestrator: output tokens ({max_tokens}) > fast limit ({fast_max_tokens})"
                )
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(
                    f"Request needs {max_tokens} tokens but orchestrator unavailable, truncating to fast LLM limit"
                )
                return LLMRole.FAST

        # Hard constraint: If input exceeds fast LLM's context window
        if estimated_input_tokens > fast_max_context:
            if orchestrator_available:
                logger.debug(
                    f"Routing to orchestrator: input tokens (~{estimated_input_tokens}) > fast context ({fast_max_context})"
                )
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(
                    f"Large input (~{estimated_input_tokens} tokens) but orchestrator unavailable, truncating to fast LLM"
                )
                return LLMRole.FAST

        # No hard constraints violated - use fast LLM (cheaper/faster)
        if fast_available:
            logger.debug(
                f"Routing to fast: within limits (input~{estimated_input_tokens}, output={max_tokens})"
            )
            return LLMRole.FAST
        elif orchestrator_available:
            logger.debug("Routing to orchestrator: fast LLM unavailable")
            return LLMRole.ORCHESTRATOR
        else:
            raise ValueError(
                "No LLMs configured - need either fast_api_url or orchestrator_api_url in config"
            )

    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process request using intelligent routing with fallback."""
        # Check if at least one LLM is configured
        fast_available = bool(self.llm_config.fast_api_url)
        orchestrator_available = bool(self.llm_config.orchestrator_api_url)

        if not fast_available and not orchestrator_available:
            raise ValueError(
                "No LLM configured. Add fast_api_url or orchestrator_api_url to pyproject.toml"
            )

        # Handle oversized content with truncation warning
        if len(request.content) > 50000:  # ~50k chars for huge log files
            logger.warning(
                f"Content too large ({len(request.content)} chars), truncating to 50k chars"
            )
            request.content = request.content[:50000] + "\n[...content truncated...]"

        # Use intelligent routing to select appropriate LLM
        selected_llm = self.select_llm(request)

        # Try primary LLM based on routing decision
        primary_failed = False
        if selected_llm == LLMRole.FAST and fast_available:
            try:
                logger.debug("Attempting fast LLM (selected by routing)")
                result = await self._call_fast_llm(request)
                if result.content and result.content.strip():
                    self._log_request_response(request, result, "fast")
                    return result
                else:
                    logger.warning("Fast LLM returned empty content")
                    primary_failed = True
            except Exception as e:
                logger.warning(f"Fast LLM failed: {e}")
                primary_failed = True
        elif selected_llm == LLMRole.ORCHESTRATOR and orchestrator_available:
            try:
                logger.debug("Attempting orchestrator LLM (selected by routing)")
                # Ensure minimum tokens for orchestrator
                orchestrator_request = LLMRequest(
                    content=request.content,
                    max_tokens=max(request.max_tokens or 0, 1000),
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                    structured_output=request.structured_output,
                )
                result = await self._call_orchestrator_llm(orchestrator_request)
                if result.content and result.content.strip():
                    self._log_request_response(request, result, "orchestrator")
                    return result
                else:
                    logger.warning("Orchestrator LLM returned empty content")
                    primary_failed = True
            except Exception as e:
                logger.warning(f"Orchestrator LLM failed: {e}")
                primary_failed = True

        # Fallback to other LLM if primary failed
        if primary_failed:
            if selected_llm == LLMRole.FAST and orchestrator_available:
                try:
                    logger.info("Falling back to orchestrator LLM")
                    orchestrator_request = LLMRequest(
                        content=request.content,
                        max_tokens=max(request.max_tokens or 0, 1000),
                        temperature=request.temperature,
                        system_prompt=request.system_prompt,
                        structured_output=request.structured_output,
                    )
                    result = await self._call_orchestrator_llm(orchestrator_request)
                    if result.content and result.content.strip():
                        self._log_request_response(request, result, "orchestrator_fallback")
                        return result
                except Exception as e:
                    logger.warning(f"Orchestrator fallback failed: {e}")
            elif selected_llm == LLMRole.ORCHESTRATOR and fast_available:
                try:
                    logger.info("Falling back to fast LLM")
                    result = await self._call_fast_llm(request)
                    if result.content and result.content.strip():
                        self._log_request_response(request, result, "fast_fallback")
                        return result
                except Exception as e:
                    logger.warning(f"Fast LLM fallback failed: {e}")

        # All attempts failed - return graceful failure
        return LLMResponse(
            content="[LLM analysis unavailable: All configured LLMs failed or returned empty content]",
            llm_used="none",
            duration_seconds=0,
            input_tokens=0,
            success=False,
            error="All LLM attempts failed or returned empty content",
        )

    async def _call_fast_llm(self, request: LLMRequest) -> LLMResponse:
        """Call the fast LLM."""
        return await self._make_api_call(self.fast_config, request, LLMRole.FAST)

    async def _call_orchestrator_llm(self, request: LLMRequest) -> LLMResponse:
        """Call the orchestrator (large context) LLM."""
        return await self._make_api_call(self.orchestrator_config, request, LLMRole.ORCHESTRATOR)

    async def _make_api_call(
        self, llm_config: LLMBackendConfig, request: LLMRequest, role: LLMRole
    ) -> LLMResponse:
        """Make API call to specified LLM."""
        start_time = time.time()

        if not llm_config.api_url:
            raise ValueError(f"No API URL configured for {role.value} LLM")

        url = f"{llm_config.api_url.rstrip('/')}/v1/chat/completions"

        headers = {}
        if llm_config.api_key:
            headers["Authorization"] = f"Bearer {llm_config.api_key}"

        # Build messages with optional system prompt
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.content})

        payload = APIPayload(
            model=llm_config.model,
            messages=messages,
            temperature=request.temperature or llm_config.temperature,
            max_tokens=request.max_tokens or llm_config.max_tokens,
            stream=False,
        )

        # Add structured output if requested
        if request.structured_output:
            backend_type = self._get_backend_type_for_role(role)

            if backend_type == "vllm":
                # vLLM now uses OpenAI-compatible response_format (updated API)
                if "json_schema" in request.structured_output:
                    payload.response_format = {
                        "type": "json_schema",
                        "json_schema": request.structured_output["json_schema"],
                    }
                else:
                    payload.response_format = {"type": "json_object"}
            elif backend_type == "llamacpp":
                # llama.cpp uses grammar constraints (simplified JSON grammar)
                json_grammar = self._create_json_grammar(request.structured_output)
                payload.grammar = json_grammar
            else:
                # OpenAI and other backends use response_format
                if "json_schema" in request.structured_output:
                    payload.response_format = {
                        "type": "json_schema",
                        "json_schema": request.structured_output["json_schema"],
                    }
                else:
                    payload.response_format = {"type": "json_object"}

        # Convert payload to dict for requests library
        payload_dict = {
            "model": payload.model,
            "messages": payload.messages,
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "stream": payload.stream,
        }
        if payload.response_format:
            payload_dict["response_format"] = payload.response_format
        if payload.grammar:
            payload_dict["grammar"] = payload.grammar

        # Debug: Log the request details
        logger.info(f"LLM Request: {role.value} to {url}")
        logger.info(f"Model: {payload.model}, Max tokens: {payload.max_tokens}")
        logger.debug(f"Request payload: {payload_dict}")

        # Set timeout based on LLM role - orchestrator needs more time for large prompts
        timeout_seconds = (
            ORCHESTRATOR_TIMEOUT_SECONDS if role == LLMRole.ORCHESTRATOR else FAST_TIMEOUT_SECONDS
        )

        logger.debug(f"Making HTTP request to {url} with timeout {timeout_seconds}s")
        response = self.session.post(
            url, json=payload_dict, headers=headers, timeout=timeout_seconds
        )
        logger.debug(f"HTTP response status: {response.status_code}")
        logger.debug(f"HTTP response headers: {dict(response.headers)}")

        response.raise_for_status()

        logger.debug(f"Raw response text length: {len(response.text)} chars")
        logger.debug(f"Raw response preview: {response.text[:500]}...")

        data = response.json()
        duration = time.time() - start_time
        logger.debug(f"Response parsed successfully, duration: {duration:.2f}s")

        if "choices" not in data or not data["choices"]:
            logger.error(f"Invalid LLM response format. Response keys: {list(data.keys())}")
            logger.error(f"Full response: {data}")
            raise ValueError("Invalid LLM response format")

        message = data["choices"][0]["message"]
        logger.debug(f"Message keys: {list(message.keys())}")

        # Extract content and reasoning content separately (vLLM/llama.cpp format)
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        logger.debug(f"Extracted content length: {len(content)} chars")
        logger.debug(f"Content preview: {content[:200]}...")

        if not content:
            logger.error(f"LLM response content is empty. Message: {message}")
            raise ValueError("LLM response content is empty")

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            llm_used=role.value,
            duration_seconds=duration,
            input_tokens=len(request.content) // TOKEN_ESTIMATION_DIVISOR,
            success=True,
        )

    def _get_backend_type_for_role(self, role: LLMRole) -> str:
        """Get backend type for the specified LLM role."""
        if role == LLMRole.FAST:
            return getattr(self.llm_config, "fast_backend", "vllm")
        elif role == LLMRole.ORCHESTRATOR:
            return getattr(self.llm_config, "orchestrator_backend", "llamacpp")
        else:
            return "openai"  # Default fallback

    def _create_json_grammar(self, json_schema: Dict[str, Any]) -> str:
        """Create a GBNF JSON grammar for llama.cpp from JSON schema."""
        if json_schema.get("type") == "object":
            properties = json_schema.get("properties", {})
            required = json_schema.get("required", [])
            defs = json_schema.get("$defs", {})

            # Handle single-property objects (common case)
            if len(properties) == 1:
                prop_name = list(properties.keys())[0]
                prop_schema = list(properties.values())[0]

                # Handle $ref to enum definitions (Pydantic style)
                if "$ref" in prop_schema:
                    ref_path = prop_schema["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            ref_def = defs[ref_name]
                            if ref_def.get("type") == "string" and "enum" in ref_def:
                                enum_values = ref_def["enum"]
                                enum_choices = " | ".join(f'"{value}"' for value in enum_values)
                                return f'root ::= "{{" ws ""{prop_name}"" ws ":" ws ({enum_choices}) ws "}}" ws ::= [ \\t\\n\\r]*'

                # Handle direct string enums (like "yes"/"no")
                elif prop_schema.get("type") == "string" and "enum" in prop_schema:
                    enum_values = prop_schema["enum"]
                    enum_choices = " | ".join(f'"{value}"' for value in enum_values)
                    return f'root ::= "{{" ws ""{prop_name}"" ws ":" ws ({enum_choices}) ws "}}" ws ::= [ \\t\\n\\r]*'

                # Handle boolean fields
                elif prop_schema.get("type") == "boolean":
                    return f'root ::= "{{" ws ""{prop_name}"" ws ":" ws ("true" | "false") ws "}}" ws ::= [ \\t\\n\\r]*'

        # Fallback to basic JSON object grammar
        return '''root ::= "{" ws "}" | "{" ws object-item (ws "," ws object-item)* ws "}"
ws ::= [ \\t\\n\\r]*
object-item ::= "\\"" [a-zA-Z_][a-zA-Z0-9_]* "\\"" ws ":" ws value
value ::= "true" | "false" | "\\"" [^"]* "\\""'''

    def is_llm_available(self, role: LLMRole) -> bool:
        """Check if a specific LLM is configured and available."""
        if role == LLMRole.FAST:
            return bool(self.fast_config.api_url and self.fast_config.model)
        else:
            return bool(self.orchestrator_config.api_url and self.orchestrator_config.model)

    def get_available_features(self) -> FeatureAvailability:
        """Get which AI features are available based on LLM configuration."""
        fast_available = self.is_llm_available(LLMRole.FAST)
        orchestrator_available = self.is_llm_available(LLMRole.ORCHESTRATOR)
        any_llm_available = fast_available or orchestrator_available

        return FeatureAvailability(
            # LLM-powered features
            architecture_analysis=orchestrator_available,  # Requires orchestrator LLM
            docstring_generation=fast_available,  # Can use fast LLM
            code_smell_detection=fast_available,  # Can use fast LLM
            coverage_assessment=orchestrator_available,  # Requires orchestrator LLM
            llm_validation=any_llm_available,  # Any LLM works
            # Embedding-only features (no LLM required)
            semantic_similarity=True,  # Always available (uses local embeddings)
            embedding_clustering=True,  # Always available (uses local embeddings)
            duplicate_detection=True,  # Always available (uses local embeddings)
        )

    def get_status(self) -> LLMStatus:
        """Get status of both LLMs."""
        return LLMStatus(
            fast_configured=self.is_llm_available(LLMRole.FAST),
            orchestrator_configured=self.is_llm_available(LLMRole.ORCHESTRATOR),
            context_threshold=self.context_threshold,
            fallback_enabled=False,  # Always disabled for predictable behavior
            available_features=self.get_available_features(),
        )

    def process_request_sync(self, request: LLMRequest) -> LLMResponse:
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
            return LLMResponse(
                content=f"LLM sync call failed: {e}",
                llm_used="error",
                duration_seconds=0,
                input_tokens=0,
                success=False,
                error=str(e),
            )


def create_llm_manager(config: Optional["LLMConfig"] = None) -> Optional[LLMClient]:
    """Create LLM manager from vibelint configuration.

    Always returns an LLMClient instance, even if no LLMs are configured.
    This allows embedding-only analysis to work without LLM endpoints.

    Args:
        config: Optional typed LLMConfig - if None, loads from config files

    """
    return LLMClient(config)
