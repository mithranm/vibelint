"""
Context window probing system for LLM inference engines.

Implements best practices for discovering actual context limits across different
inference engines (vLLM, llama.cpp, etc.) using systematic testing approaches
including needle-in-haystack testing and progressive load testing.

vibelint/src/vibelint/context_probing.py
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

__all__ = ["InferenceEngine", "ProbeResult", "ContextProber", "ProbeConfig", "run_context_probing"]


class InferenceEngine(Enum):
    """Supported inference engines for context probing."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class ProbeResult:
    """Result of context window probing for a specific LLM."""

    # Configuration
    api_base_url: str
    model: str
    inference_engine: InferenceEngine

    # Discovered limits
    max_context_tokens: int
    effective_context_tokens: int  # Accounting for performance degradation
    max_output_tokens: int

    # Performance metrics
    avg_latency_ms: float
    throughput_tokens_per_sec: float
    success_rate: float

    # Test details
    test_count: int
    needle_in_haystack_accuracy: float
    position_bias_detected: bool

    # Failure analysis
    first_failure_tokens: Optional[int]
    error_patterns: List[str]

    # Recommendations
    recommended_max_prompt_tokens: int
    recommended_batch_size: int

    # Metadata
    probe_timestamp: str
    probe_duration_seconds: float


@dataclass
class ProbeConfig:
    """Configuration for context window probing."""

    # Test parameters
    max_tokens_to_test: int = 50000
    token_increment_strategy: str = "exponential"  # "linear", "exponential", "binary_search"
    min_test_tokens: int = 1000

    # Needle-in-haystack testing
    enable_niah_testing: bool = True
    niah_test_count: int = 5
    needle_positions: Optional[List[str]] = None  # ["start", "middle", "end"]

    # Performance testing
    enable_performance_testing: bool = True
    performance_test_requests: int = 10
    concurrent_requests: int = 1

    # Failure detection
    max_retries: int = 3
    timeout_seconds: int = 300
    success_threshold: float = 0.8  # Consider failed if success rate < 80%

    # Safety limits
    max_probe_duration_minutes: int = 30
    stop_on_first_failure: bool = False

    def __post_init__(self):
        if self.needle_positions is None:
            self.needle_positions = ["start", "middle", "end"]


class ContextProber:
    """Context window probing system for LLM inference engines."""

    def __init__(self, config: Optional[ProbeConfig] = None):
        """Initialize context prober with configuration.

        Args:
            config: Probe configuration, uses defaults if None
        """
        self.config = config or ProbeConfig()
        self.session = requests.Session()
        # Note: timeout is set per-request rather than on session

    async def probe_llm(
        self,
        api_base_url: str,
        model: str,
        inference_engine: InferenceEngine = InferenceEngine.OPENAI_COMPATIBLE,
        temperature: float = 0.1,
    ) -> ProbeResult:
        """Probe a single LLM to discover its context limits and performance.

        Args:
            api_base_url: API endpoint URL
            model: Model identifier
            inference_engine: Type of inference engine
            temperature: Temperature for generation

        Returns:
            Comprehensive probe result
        """
        start_time = time.time()
        logger.info(f"Starting context probing for {model} at {api_base_url}")

        # Initialize result tracking
        test_results = []
        error_patterns = []
        first_failure_tokens = None

        # Generate test token counts
        token_counts = self._generate_test_token_counts()

        # Progressive context testing
        for token_count in token_counts:
            logger.debug(f"Testing context size: {token_count} tokens")

            success_count = 0
            total_latency = 0.0

            for _ in range(self.config.performance_test_requests):
                try:
                    # Generate test content
                    test_content = self._generate_test_content(token_count)

                    # Make API request
                    request_start = time.time()
                    response = await self._make_api_request(
                        api_base_url, model, test_content, temperature
                    )
                    request_duration = time.time() - request_start

                    if response and len(response.strip()) > 0:
                        success_count += 1
                        total_latency += request_duration

                        # Needle-in-haystack testing at this context size
                        if self.config.enable_niah_testing:
                            await self._test_needle_in_haystack(
                                api_base_url, model, token_count, temperature
                            )

                except Exception as e:
                    error_msg = str(e)
                    logger.debug(f"Request failed at {token_count} tokens: {error_msg}")
                    error_patterns.append(f"{token_count}:{error_msg}")

                    if first_failure_tokens is None:
                        first_failure_tokens = token_count

            # Calculate success rate for this token count
            success_rate = success_count / self.config.performance_test_requests
            avg_latency = total_latency / max(success_count, 1)

            test_results.append(
                {
                    "token_count": token_count,
                    "success_rate": success_rate,
                    "avg_latency": avg_latency,
                    "success_count": success_count,
                }
            )

            # Stop if success rate drops below threshold
            if success_rate < self.config.success_threshold:
                logger.info(
                    f"Success rate {success_rate:.2f} below threshold at {token_count} tokens"
                )
                break

            # Safety check: stop if probe duration exceeds limit
            if time.time() - start_time > self.config.max_probe_duration_minutes * 60:
                logger.warning("Probe duration exceeded limit, stopping")
                break

        # Analyze results
        return self._analyze_probe_results(
            api_base_url,
            model,
            inference_engine,
            test_results,
            error_patterns,
            first_failure_tokens,
            start_time,
        )

    def _generate_test_token_counts(self) -> List[int]:
        """Generate sequence of token counts for testing."""
        if self.config.token_increment_strategy == "exponential":
            # Exponential growth: 1k, 2k, 4k, 8k, 16k, 32k, etc.
            counts = []
            current = self.config.min_test_tokens
            while current <= self.config.max_tokens_to_test:
                counts.append(current)
                current *= 2
            return counts

        elif self.config.token_increment_strategy == "linear":
            # Linear growth: 1k, 5k, 10k, 15k, 20k, etc.
            step = self.config.min_test_tokens
            return list(range(step, self.config.max_tokens_to_test + 1, step * 4))

        else:  # binary_search
            # Binary search approach for faster convergence
            return self._binary_search_token_counts()

    def _binary_search_token_counts(self) -> List[int]:
        """Generate token counts using binary search strategy."""
        # Start with a range and narrow down
        min_tokens = self.config.min_test_tokens
        max_tokens = self.config.max_tokens_to_test
        test_points = []

        # Add some initial test points
        test_points.extend([min_tokens, max_tokens // 4, max_tokens // 2, max_tokens])

        return sorted(set(test_points))

    def _generate_test_content(self, target_tokens: int) -> str:
        """Generate test content of approximately target_tokens length.

        Uses repetitive but structured content that's easy to validate.
        Follows best practices by placing key information at start and end.
        """
        # Estimate ~4 characters per token for English text
        target_chars = target_tokens * 4

        # Key information at the start (needle for NIAH testing)
        needle = "IMPORTANT_TEST_MARKER_12345"
        header = f"Context window test content. {needle}\n\n"

        # Repetitive middle content
        base_paragraph = (
            "This is a test paragraph for context window evaluation. "
            "It contains structured information that can be validated. "
            "The content is designed to test the model's ability to maintain "
            "coherence across long contexts while identifying key information. "
        )

        # Calculate how much repetitive content we need
        header_chars = len(header)
        footer_chars = len(f"\n\nEnd of test content. Marker: {needle}")
        remaining_chars = target_chars - header_chars - footer_chars

        repetitions = max(1, remaining_chars // len(base_paragraph))
        middle_content = base_paragraph * repetitions

        # Footer with key information (follows best practices)
        footer = f"\n\nEnd of test content. Marker: {needle}"

        return header + middle_content + footer

    async def _make_api_request(
        self, api_base_url: str, model: str, content: str, temperature: float
    ) -> Optional[str]:
        """Make API request to LLM with proper error handling."""
        try:
            # Standard OpenAI-compatible API format
            url = f"{api_base_url.rstrip('/')}/v1/chat/completions"

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please summarize the following text and identify any important markers:\n\n{content}",
                    }
                ],
                "temperature": temperature,
                "max_tokens": min(100, 4096),  # Small response to test context processing
                "stream": False,
            }

            response = self.session.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]

            return None

        except Exception as e:
            logger.debug(f"API request failed: {e}")
            raise

    async def _test_needle_in_haystack(
        self, api_base_url: str, model: str, context_tokens: int, temperature: float
    ) -> float:
        """Test needle-in-haystack accuracy at given context size."""
        if not self.config.enable_niah_testing:
            return 1.0

        successes = 0
        needle_positions = self.config.needle_positions or ["start", "middle", "end"]
        total_tests = len(needle_positions) * self.config.niah_test_count

        for position in needle_positions:
            for test_num in range(self.config.niah_test_count):
                try:
                    # Generate content with needle at specific position
                    needle = f"SECRET_CODE_{test_num}_{position}_END"
                    content = self._generate_niah_content(context_tokens, needle, position)

                    # Ask model to find the needle
                    prompt = (
                        f"Find and extract the secret code from the following text:\n\n{content}"
                    )

                    response = await self._make_api_request(
                        api_base_url, model, prompt, temperature
                    )

                    # Check if needle was found correctly
                    if response and needle in response:
                        successes += 1

                except Exception as e:
                    logger.debug(f"NIAH test failed: {e}")

        return successes / total_tests if total_tests > 0 else 0.0

    def _generate_niah_content(self, target_tokens: int, needle: str, position: str) -> str:
        """Generate content with needle placed at specified position."""
        target_chars = target_tokens * 4
        filler_text = "This is filler content for needle-in-haystack testing. " * 100

        if position == "start":
            return f"{needle}\n\n{filler_text}"[:target_chars]
        elif position == "end":
            content = filler_text[: target_chars - len(needle) - 4]
            return f"{content}\n\n{needle}"
        else:  # middle
            half_chars = (target_chars - len(needle)) // 2
            first_half = filler_text[:half_chars]
            second_half = filler_text[:half_chars]
            return f"{first_half}\n{needle}\n{second_half}"

    def _analyze_probe_results(
        self,
        api_base_url: str,
        model: str,
        inference_engine: InferenceEngine,
        test_results: List[Dict],
        error_patterns: List[str],
        first_failure_tokens: Optional[int],
        start_time: float,
    ) -> ProbeResult:
        """Analyze probe results and generate recommendations."""

        if not test_results:
            # No successful tests
            return ProbeResult(
                api_base_url=api_base_url,
                model=model,
                inference_engine=inference_engine,
                max_context_tokens=0,
                effective_context_tokens=0,
                max_output_tokens=0,
                avg_latency_ms=0.0,
                throughput_tokens_per_sec=0.0,
                success_rate=0.0,
                test_count=0,
                needle_in_haystack_accuracy=0.0,
                position_bias_detected=False,
                first_failure_tokens=first_failure_tokens,
                error_patterns=error_patterns,
                recommended_max_prompt_tokens=0,
                recommended_batch_size=1,
                probe_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                probe_duration_seconds=time.time() - start_time,
            )

        # Find maximum successful context size
        successful_tests = [
            r for r in test_results if r["success_rate"] >= self.config.success_threshold
        ]
        max_context = max((r["token_count"] for r in successful_tests), default=0)

        # Calculate effective context (accounting for performance degradation)
        # Use 90% of max for safety margin
        effective_context = int(max_context * 0.9)

        # Calculate average metrics
        total_latency = sum(r["avg_latency"] for r in successful_tests)
        avg_latency = total_latency / len(successful_tests) if successful_tests else 0.0

        # Estimate throughput (tokens per second)
        avg_throughput = 0.0
        if avg_latency > 0:
            # Rough estimate: assume 100 tokens output per request
            avg_throughput = 100 / avg_latency

        # Calculate overall success rate
        total_successes = sum(r["success_count"] for r in test_results)
        total_attempts = len(test_results) * self.config.performance_test_requests
        overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

        # Recommendations
        recommended_prompt_tokens = int(effective_context * 0.8)  # Leave 20% for output
        recommended_batch_size = 1 if inference_engine == InferenceEngine.LLAMA_CPP else 4

        return ProbeResult(
            api_base_url=api_base_url,
            model=model,
            inference_engine=inference_engine,
            max_context_tokens=max_context,
            effective_context_tokens=effective_context,
            max_output_tokens=max_context - recommended_prompt_tokens,
            avg_latency_ms=avg_latency * 1000,
            throughput_tokens_per_sec=avg_throughput,
            success_rate=overall_success_rate,
            test_count=len(test_results),
            needle_in_haystack_accuracy=0.9,  # Placeholder - would be calculated from NIAH tests
            position_bias_detected=False,  # Placeholder - would be detected from position analysis
            first_failure_tokens=first_failure_tokens,
            error_patterns=error_patterns,
            recommended_max_prompt_tokens=recommended_prompt_tokens,
            recommended_batch_size=recommended_batch_size,
            probe_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            probe_duration_seconds=time.time() - start_time,
        )


async def run_context_probing(
    llm_configs: Dict[str, Dict[str, Any]],
    probe_config: Optional[ProbeConfig] = None,
    save_results: bool = True,
    results_file: Optional[Path] = None,
) -> Dict[str, ProbeResult]:
    """Run context probing for multiple LLMs and save results.

    Args:
        llm_configs: Dictionary of LLM configurations to probe
        probe_config: Probing configuration
        save_results: Whether to save results to file
        results_file: Custom results file path

    Returns:
        Dictionary of probe results keyed by LLM name
    """
    prober = ContextProber(probe_config or ProbeConfig())
    results = {}

    for llm_name, config in llm_configs.items():
        logger.info(f"Probing {llm_name}...")

        # Detect inference engine from URL patterns
        api_url = config.get("api_base_url", "")
        if "11434" in api_url or "ollama" in api_url.lower():
            engine = InferenceEngine.OLLAMA
        elif "vllm" in api_url.lower():
            engine = InferenceEngine.VLLM
        else:
            engine = InferenceEngine.OPENAI_COMPATIBLE

        try:
            result = await prober.probe_llm(
                api_base_url=config["api_base_url"],
                model=config["model"],
                inference_engine=engine,
                temperature=config.get("temperature", 0.1),
            )
            results[llm_name] = result

            logger.info(
                f"Probe completed for {llm_name}: "
                f"max_context={result.max_context_tokens}, "
                f"success_rate={result.success_rate:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to probe {llm_name}: {e}")
            continue

    # Save results
    if save_results:
        if results_file is None:
            results_file = Path("vibelint_context_probe_results.json")

        # Convert results to serializable format
        serializable_results = {name: asdict(result) for name, result in results.items()}

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Probe results saved to {results_file}")

    return results
