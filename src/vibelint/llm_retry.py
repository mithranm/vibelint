"""
Generalized LLM retry pattern with few-shot learning.

Provides reusable components for robust LLM interactions that can recover
from parsing failures using few-shot examples and progressive retry strategies.

vibelint/src/vibelint/llm_retry.py
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type of items being processed
R = TypeVar('R')  # Type of parsed results


class RetryStrategy(Enum):
    """Available retry strategies."""
    NONE = "none"
    FEW_SHOT = "few_shot"
    PROGRESSIVE = "progressive"
    ADAPTIVE = "adaptive"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 2
    strategy: RetryStrategy = RetryStrategy.FEW_SHOT
    retry_threshold: int = 3  # Only retry if failures <= this number
    enable_logging: bool = True


@dataclass
class ParseResult(Generic[T, R]):
    """Result of a parsing attempt."""
    successful_items: List[T]
    failed_items: List[T]
    parsed_data: Dict[T, R]
    success_rate: float
    error_message: Optional[str] = None


class LLMRetryHandler(ABC, Generic[T, R]):
    """Abstract base class for LLM retry handling with few-shot learning."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._llm_call_count = 0
        self._total_retries = 0

    @abstractmethod
    def create_initial_prompt(self, items: List[T]) -> str:
        """Create the initial prompt for processing items."""
        pass

    @abstractmethod
    def parse_response(self, items: List[T], response: str) -> ParseResult[T, R]:
        """Parse LLM response and return success/failure breakdown."""
        pass

    @abstractmethod
    def create_few_shot_examples(self) -> str:
        """Create few-shot examples showing correct format."""
        pass

    @abstractmethod
    def invoke_llm(self, prompt: str) -> str:
        """Make the actual LLM call."""
        pass

    def create_retry_prompt(self, failed_items: List[T], original_response: str, retry_attempt: int) -> str:
        """Create retry prompt with few-shot learning."""
        examples = self.create_few_shot_examples()

        strategy_message = {
            RetryStrategy.FEW_SHOT: "Here are examples of the correct format:",
            RetryStrategy.PROGRESSIVE: f"Attempt {retry_attempt + 1}: Let's try a simpler approach.",
            RetryStrategy.ADAPTIVE: "The previous response had formatting issues. Let's be more specific:"
        }.get(self.config.strategy, "Please try again with the correct format:")

        item_details = self.format_items_for_retry(failed_items)

        return f"""The previous response didn't follow the expected format. {strategy_message}

{examples}

Now please re-analyze these specific items using the EXACT format shown above:

{item_details}

Remember: Follow the format exactly as shown in the examples."""

    def format_items_for_retry(self, items: List[T]) -> str:
        """Format items for inclusion in retry prompt. Override for custom formatting."""
        return "\n".join(f"Item {i+1}: {item}" for i, item in enumerate(items))

    def should_retry(self, failed_items: List[T], retry_attempt: int) -> bool:
        """Determine if retry should be attempted."""
        if retry_attempt >= self.config.max_retries:
            return False
        if len(failed_items) > self.config.retry_threshold:
            return False
        if self.config.strategy == RetryStrategy.NONE:
            return False
        return True

    def process_with_retry(self, items: List[T]) -> ParseResult[T, R]:
        """Process items with automatic retry on parsing failures."""

        if self.config.enable_logging:
            logger.info(f"Processing {len(items)} items with retry strategy: {self.config.strategy.value}")

        # Initial attempt
        initial_prompt = self.create_initial_prompt(items)
        initial_response = self.invoke_llm(initial_prompt)
        self._llm_call_count += 1

        result = self.parse_response(items, initial_response)

        # Track overall results
        all_successful = list(result.successful_items)
        all_parsed_data = dict(result.parsed_data)

        # Retry loop for failed items
        retry_attempt = 0
        current_failed = result.failed_items

        while self.should_retry(current_failed, retry_attempt):
            if self.config.enable_logging:
                logger.info(f"Retry attempt {retry_attempt + 1} for {len(current_failed)} failed items")

            retry_prompt = self.create_retry_prompt(current_failed, initial_response, retry_attempt)
            retry_response = self.invoke_llm(retry_prompt)
            self._llm_call_count += 1
            self._total_retries += 1

            retry_result = self.parse_response(current_failed, retry_response)

            # Update overall results
            all_successful.extend(retry_result.successful_items)
            all_parsed_data.update(retry_result.parsed_data)

            # Update for next iteration
            current_failed = retry_result.failed_items
            retry_attempt += 1

            if self.config.enable_logging and retry_result.successful_items:
                success_count = len(retry_result.successful_items)
                total_retry_items = len(current_failed) + success_count
                logger.info(f"Retry succeeded for {success_count}/{total_retry_items} items")

        # Final result
        final_success_rate = len(all_successful) / len(items) if items else 0.0

        if self.config.enable_logging:
            logger.info(f"Processing complete: {len(all_successful)}/{len(items)} successful "
                       f"({final_success_rate:.1%}), {self._total_retries} retries used")

        return ParseResult(
            successful_items=all_successful,
            failed_items=current_failed,
            parsed_data=all_parsed_data,
            success_rate=final_success_rate
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "llm_calls": self._llm_call_count,
            "total_retries": self._total_retries,
            "config": {
                "max_retries": self.config.max_retries,
                "strategy": self.config.strategy.value,
                "retry_threshold": self.config.retry_threshold
            }
        }


class SimpleTextRetryHandler(LLMRetryHandler[str, str]):
    """Example implementation for simple text processing."""

    def __init__(self, llm_callable: Callable[[str], str], format_instructions: str,
                 few_shot_examples: str, config: Optional[RetryConfig] = None):
        super().__init__(config)
        self.llm_callable = llm_callable
        self.format_instructions = format_instructions
        self.few_shot_examples = few_shot_examples

    def create_initial_prompt(self, items: List[str]) -> str:
        return f"""{self.format_instructions}

Items to process:
{chr(10).join(f"- {item}" for item in items)}"""

    def parse_response(self, items: List[str], response: str) -> ParseResult[str, str]:
        # Simple implementation - assumes one response line per item
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        successful = []
        parsed = {}

        for i, item in enumerate(items):
            if i < len(lines) and lines[i]:
                successful.append(item)
                parsed[item] = lines[i]

        failed = [item for item in items if item not in successful]
        success_rate = len(successful) / len(items) if items else 0.0

        return ParseResult(
            successful_items=successful,
            failed_items=failed,
            parsed_data=parsed,
            success_rate=success_rate
        )

    def create_few_shot_examples(self) -> str:
        return self.few_shot_examples

    def invoke_llm(self, prompt: str) -> str:
        return self.llm_callable(prompt)


@dataclass
class FileSummary:
    """Structured file summary result."""
    purpose: str
    exports: str
    dependencies: str
    concerns: str


class FileAnalysisRetryHandler(LLMRetryHandler[Path, FileSummary]):
    """Specialized retry handler for file analysis tasks."""

    def __init__(self, llm_callable: Callable[[str], str], config: Optional[RetryConfig] = None):
        super().__init__(config)
        self.llm_callable = llm_callable

    def create_initial_prompt(self, files: List[Path]) -> str:
        """Create initial file analysis prompt."""
        batch_content = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                batch_content.append(f"FILE: {file_path}\n{content[:1500]}")
            except Exception as e:
                batch_content.append(f"FILE: {file_path}\nERROR: Could not read file - {e}")

        return f"""Analyze these Python files and provide concise summaries for each:

{chr(10).join(batch_content)}

For each file, provide a summary in this format:
FILE: [filename]
PURPOSE: What this file does
EXPORTS: Key classes/functions it exports
DEPENDENCIES: What it imports/depends on
CONCERNS: Any potential issues
---
"""

    def parse_response(self, files: List[Path], response: str) -> ParseResult[Path, FileSummary]:
        """Parse file analysis response."""
        import re
        from pathlib import Path

        # Enhanced parsing with multiple strategies
        file_sections = re.split(r"FILE:\s*([^\n]+)", response)
        parsed_files = {}
        successful_files = []

        if len(file_sections) > 1:
            for i in range(1, len(file_sections), 2):
                if i + 1 < len(file_sections):
                    filename_part = file_sections[i].strip()
                    summary_content = file_sections[i + 1].split("---")[0].strip()

                    # Find matching file by name
                    for file_path in files:
                        if (file_path.name in filename_part or
                            str(file_path) in filename_part or
                            str(file_path).replace("src/", "") in filename_part):

                            # Extract structured data
                            summary = self._extract_file_summary(summary_content)
                            if summary:
                                parsed_files[file_path] = summary
                                successful_files.append(file_path)
                            break

        failed_files = [f for f in files if f not in successful_files]
        success_rate = len(successful_files) / len(files) if files else 0.0

        return ParseResult(
            successful_items=successful_files,
            failed_items=failed_files,
            parsed_data=parsed_files,
            success_rate=success_rate
        )

    def _extract_file_summary(self, content: str) -> Optional[FileSummary]:
        """Extract structured summary from content."""
        import re

        patterns = {
            'purpose': r'PURPOSE:\s*([^\n]+)',
            'exports': r'EXPORTS:\s*([^\n]+)',
            'dependencies': r'DEPENDENCIES:\s*([^\n]+)',
            'concerns': r'CONCERNS:\s*([^\n]+)'
        }

        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            extracted[key] = match.group(1).strip() if match else "Not specified"

        # Only return if we got at least purpose
        if extracted['purpose'] != "Not specified":
            return FileSummary(**extracted)
        return None

    def create_few_shot_examples(self) -> str:
        """Create few-shot examples for file analysis."""
        return """EXAMPLE 1:
FILE: src/example/config.py
PURPOSE: Configuration loading and validation for the application
EXPORTS: ConfigLoader class, load_config() function
DEPENDENCIES: pathlib, yaml, logging
CONCERNS: No input validation on config file paths
---

EXAMPLE 2:
FILE: src/example/utils.py
PURPOSE: Utility functions for string manipulation and file operations
EXPORTS: sanitize_string(), read_file_safe(), write_file_atomic()
DEPENDENCIES: re, pathlib, tempfile
CONCERNS: Limited error handling in file operations
---"""

    def format_items_for_retry(self, files: List[Path]) -> str:
        """Format files for retry prompt."""
        formatted = []
        for file_path in files:
            try:
                content_preview = file_path.read_text(encoding='utf-8')[:500]
                formatted.append(f"FILE: {file_path}\nContent preview: {content_preview[:200]}...")
            except Exception:
                formatted.append(f"FILE: {file_path}\nERROR: Could not read file")

        return chr(10).join(formatted)

    def invoke_llm(self, prompt: str) -> str:
        return self.llm_callable(prompt)