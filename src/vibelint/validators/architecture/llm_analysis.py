"""
LLM-powered architectural analysis validator using OpenAI-compatible APIs.

Provides intelligent architectural analysis using Large Language Models
to detect design issues, inconsistencies, and improvement opportunities.

vibelint/validators/architecture/llm_analysis.py
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from ...plugin_system import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["LLMAnalysisValidator"]


class ArchitecturalFinding(BaseModel):
    """Schema for architectural analysis findings."""

    file: str = Field(description="File path where issue was found")
    line: int = Field(default=1, description="Line number of the issue")
    severity: str = Field(description="Severity level: error, warning, info")
    category: str = Field(description="Category of architectural issue")
    message: str = Field(description="Description of the architectural issue")
    suggestion: str = Field(default="", description="Suggested improvement")


class ArchitecturalAnalysis(BaseModel):
    """Schema for complete architectural analysis response."""

    findings: List[ArchitecturalFinding] = Field(description="List of architectural issues found")
    summary: str = Field(description="Overall assessment of the codebase architecture")


class LLMAnalysisValidator(BaseValidator):
    """LLM-powered architectural analysis using OpenAI-compatible API for structured workflow."""

    rule_id = "ARCHITECTURE-LLM"
    name = "LLM Architectural Analysis"
    description = "AI-powered analysis of code architecture and design patterns"
    default_severity = Severity.INFO

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(severity, config)

        # Track analysis state
        self._llm_setup_attempted = False
        self._llm_available = False
        self._analysis_completed = False
        self._analyzed_files: set[Path] = set()  # Track which files we've seen
        self._global_analysis_done: bool = False  # Track if global analysis has run

        # Multi-phase analysis state
        self._file_summaries: dict[Path, str] = {}  # File path -> summary
        self._file_embeddings: dict[Path, list[float]] = {}  # File path -> embedding vector
        self._similarity_clusters: list[list[Path]] = []  # Groups of similar files
        self._pairwise_analyses: dict[tuple[Path, Path], str] = {}  # Pair -> analysis

        # API configuration from config with environment variable fallbacks
        llm_config = self.config.get("llm_analysis", {})

        self.api_url = os.getenv(
            "OPENAI_BASE_URL", llm_config.get("api_base_url", "http://localhost:11434")
        )
        self.model = os.getenv("OPENAI_MODEL", llm_config.get("model", "llama2"))
        self.api_key = os.getenv("OPENAI_API_KEY", "not-needed")

        # Generation settings from config with sensible defaults
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.temperature = llm_config.get("temperature", 0.1)

        # Advanced settings with defaults (most users won't need to change these)
        self.max_context_tokens = llm_config.get("max_context_tokens", 32768)
        self.max_prompt_tokens = llm_config.get("max_prompt_tokens", 28672)
        self.max_file_lines = llm_config.get("max_file_lines", 100)
        self.max_files = llm_config.get("max_files", 10)
        self.remove_thinking_tokens = llm_config.get("remove_thinking_tokens", True)
        self.thinking_format = llm_config.get("thinking_format", "harmony")
        self.custom_thinking_patterns = llm_config.get("custom_thinking_patterns", [])

        # Diagnostics for optimal LLM utilization
        self.enable_token_diagnostics = llm_config.get("enable_token_diagnostics", True)
        self.token_usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "calls": 0,
            "context_efficiency": [],
        }

        # Dynamic context discovery
        self.enable_context_probing = llm_config.get("enable_context_probing", True)
        self.discovered_context_limit = None
        self.context_probe_cache = {}

        # Enable all compression strategies by default
        self.enable_import_summary = True
        self.enable_hierarchy_extraction = True
        self.enable_pattern_detection = True
        self.enable_complexity_analysis = True
        self.max_signature_lines = 20
        self.max_hierarchy_depth = 15

        # Initialize langchain components for OpenAI-compatible API
        self.llm = ChatOpenAI(
            base_url=self.api_url + "/v1",
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            api_key=SecretStr(self.api_key),
        )

        # Global architecture analysis prompt
        self.global_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert software architect analyzing Python code for structural quality.

Analyze the provided codebase for architectural issues including:
- Module organization and coupling problems
- Violation of design principles (SOLID, DRY, etc.)
- Inconsistent patterns and abstractions
- Poor separation of concerns
- Code duplication and redundancy
- Naming convention violations
- API design issues

Focus on structural problems that affect maintainability, not syntax errors.

Return your response as valid JSON with this structure:
{{
  "findings": [
    {{
      "file": "path/to/file.py",
      "line": 1,
      "severity": "warning",
      "category": "architectural",
      "message": "Description of issue",
      "suggestion": "Suggested improvement"
    }}
  ]
}}""",
                ),
                ("user", "Analyze this Python codebase structure:\n\n{code_content}"),
            ]
        )

        # Create analysis chain
        self.global_chain = self.global_prompt | self.llm

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Run LLM architectural analysis on the specified files.

        Analyzes the actual files being processed, not the entire project.
        """
        # Skip if LLM setup hasn't been attempted yet
        if not self._llm_setup_attempted:
            self._llm_setup_attempted = True
            self._llm_available = self._test_connectivity()

        if not self._llm_available:
            logger.debug("LLM service not available, skipping analysis")
            return

        # Skip if we've already analyzed this file
        if file_path in self._analyzed_files:
            return

        self._analyzed_files.add(file_path)

        # Skip if global analysis has already run
        if self._global_analysis_done:
            return

        self._global_analysis_done = True

        # Get the actual files being analyzed from the validation engine
        analysis_files = config.get("_analysis_files", [file_path]) if config else [file_path]

        # Estimate analysis time (rough estimate: ~30-60 seconds per file for LLM analysis)
        estimated_minutes = max(1, (len(analysis_files) * 45) // 60)  # 45 seconds per file average
        estimated_range = f"{max(1, estimated_minutes - 1)}-{estimated_minutes + 2}"

        logger.info(f"Starting LLM architectural analysis on {len(analysis_files)} files")
        logger.info(f"ESTIMATED TIME: {estimated_range} minutes (depends on LLM response speed)")
        logger.info(
            "TIP: For faster analysis, use path override: vibelint check src/specific_dir/ --exclude-ai"
        )

        if estimated_minutes > 5:
            logger.warning(f"TIMEOUT RISK: Analysis may take {estimated_range} minutes")
            logger.warning(
                "If using AI coding tools or CI systems, consider analyzing smaller chunks:"
            )
            logger.warning("   vibelint check src/module1/ --rule ARCHITECTURE-LLM")
            logger.warning("   vibelint check src/module2/ --rule ARCHITECTURE-LLM")

        # Run architectural analysis on the specified files
        yield from self._analyze_global_structure(file_path, analysis_files)

    def _test_connectivity(self) -> bool:
        """Test if OpenAI-compatible API service is available."""
        try:
            # Simple connectivity test with ChatOpenAI
            self.llm.invoke("Test connectivity")

            # If context probing is enabled, discover real limits
            if self.enable_context_probing:
                self._discover_context_limits()

            return True
        except Exception as e:
            logger.debug(f"LLM connectivity test failed: {e}")
            return False

    def _discover_context_limits(self) -> None:
        """Dynamically discover the LLM's actual context window limits."""
        if self.discovered_context_limit is not None:
            logger.debug(f"Using cached context limit: {self.discovered_context_limit}")
            return

        logger.info("Probing LLM for actual context window limits...")

        # Binary search for maximum context size
        min_tokens = 1000
        max_tokens = 200000  # Start with a high upper bound
        working_limit = min_tokens

        # Quick test sizes to find rough range
        test_sizes = [4000, 8000, 16000, 32000, 64000, 128000]

        for test_size in test_sizes:
            if self._test_context_size(test_size):
                working_limit = test_size
                logger.debug(f"PASS - Context size {test_size} tokens: PASSED")
            else:
                logger.debug(f"FAIL - Context size {test_size} tokens: FAILED")
                max_tokens = test_size
                break

        # Binary search within the working range
        min_tokens = working_limit
        attempts = 0
        max_attempts = 8  # Limit binary search iterations

        while min_tokens < max_tokens - 1000 and attempts < max_attempts:
            mid_tokens = (min_tokens + max_tokens) // 2

            if self._test_context_size(mid_tokens):
                min_tokens = mid_tokens
                working_limit = mid_tokens
                logger.debug(f"PASS - Binary search {mid_tokens} tokens: PASSED")
            else:
                max_tokens = mid_tokens
                logger.debug(f"FAIL - Binary search {mid_tokens} tokens: FAILED")

            attempts += 1

        self.discovered_context_limit = working_limit
        efficiency_gain = (working_limit / self.max_context_tokens) * 100

        if working_limit > self.max_context_tokens:
            logger.info(
                f"SUCCESS - Discovered larger context window: {working_limit:,} tokens (configured: {self.max_context_tokens:,})"
            )
            logger.info(
                f"TIP - OPTIMIZATION: Increase max_context_tokens to {working_limit} for {efficiency_gain:.0f}% better utilization"
            )
        elif working_limit < self.max_context_tokens:
            logger.warning(
                f"WARNING - Actual context limit lower than configured: {working_limit:,} tokens vs {self.max_context_tokens:,}"
            )
            logger.warning("TIP - RECOMMENDATION: Reduce max_context_tokens to avoid errors")
        else:
            logger.info(f"OK - Context configuration optimal: {working_limit:,} tokens")

    def _test_context_size(self, token_count: int) -> bool:
        """Test if the LLM can handle a specific context size."""
        # Use cache to avoid repeated tests
        if token_count in self.context_probe_cache:
            return self.context_probe_cache[token_count]

        # Generate test content roughly matching token count
        # Approximate 4 characters per token for English text
        char_count = token_count * 4
        test_content = "def test_function():\n    pass\n\n" * (char_count // 30)

        test_prompt = f"""Analyze this code for basic issues. Respond with just "OK" if analysis completes successfully.

CODE TO ANALYZE:
{test_content[:char_count]}

Respond with exactly: OK"""

        try:
            response = self.llm.invoke(test_prompt)
            success = "OK" in str(response).upper()
            self.context_probe_cache[token_count] = success
            return success

        except Exception as e:
            error_msg = str(e).lower()
            # Check for context-related errors
            context_errors = [
                "context length",
                "token limit",
                "max tokens",
                "context window",
                "input too long",
                "maximum context",
                "context exceeded",
            ]

            is_context_error = any(err in error_msg for err in context_errors)
            if is_context_error:
                logger.debug(f"Context limit reached at {token_count} tokens: {e}")
            else:
                logger.debug(f"Non-context error at {token_count} tokens: {e}")

            self.context_probe_cache[token_count] = False
            return False

    def _get_effective_context_limit(self) -> int:
        """Get the actual usable context limit (discovered or configured)."""
        if self.enable_context_probing and self.discovered_context_limit:
            return self.discovered_context_limit
        return self.max_context_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token for English)."""
        return max(1, len(text) // 4)

    def _optimize_context_usage(self, prompt: str) -> tuple[str, dict]:
        """Optimize prompt for maximum context utilization while tracking metrics."""
        estimated_input_tokens = self._estimate_tokens(prompt)

        # Use discovered context limit if available
        effective_context_limit = self._get_effective_context_limit()

        # Calculate optimal generation tokens based on actual context limits
        available_tokens = effective_context_limit - estimated_input_tokens
        optimal_generation_tokens = min(
            self.max_tokens, available_tokens - 500
        )  # 500 token safety buffer

        context_efficiency = estimated_input_tokens / effective_context_limit

        metrics = {
            "estimated_input_tokens": estimated_input_tokens,
            "optimal_generation_tokens": optimal_generation_tokens,
            "context_efficiency": context_efficiency,
            "context_utilization_percent": context_efficiency * 100,
            "effective_context_limit": effective_context_limit,
            "using_discovered_limit": self.discovered_context_limit is not None,
        }

        if self.enable_token_diagnostics:
            limit_source = "discovered" if self.discovered_context_limit else "configured"
            logger.info(
                f"Context utilization: {context_efficiency:.1%} ({estimated_input_tokens:,}/{effective_context_limit:,} tokens, {limit_source})"
            )
            if context_efficiency > 0.8:
                logger.warning(
                    f"High context usage ({context_efficiency:.1%}) - consider reducing batch size"
                )
            elif context_efficiency < 0.3:
                logger.info(
                    f"Low context usage ({context_efficiency:.1%}) - could increase batch size for efficiency"
                )

        return prompt, metrics

    def _invoke_with_diagnostics(self, prompt: str) -> str:
        """Invoke LLM with optimal context usage and diagnostics."""
        optimized_prompt, metrics = self._optimize_context_usage(prompt)

        try:
            response = self.global_chain.invoke({"code_content": optimized_prompt})
            cleaned_response = self._clean_response(response)

            # Track usage statistics
            if self.enable_token_diagnostics:
                estimated_output_tokens = self._estimate_tokens(cleaned_response)
                self.token_usage_stats["total_input_tokens"] += metrics["estimated_input_tokens"]
                self.token_usage_stats["total_output_tokens"] += estimated_output_tokens
                self.token_usage_stats["calls"] += 1
                self.token_usage_stats["context_efficiency"].append(metrics["context_efficiency"])

                logger.debug(
                    f"Call {self.token_usage_stats['calls']}: {metrics['estimated_input_tokens']}→{estimated_output_tokens} tokens"
                )

            return cleaned_response

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            if self.enable_token_diagnostics:
                self.token_usage_stats["calls"] += 1
            return f"Error: LLM analysis failed - {e}"

    def _log_final_diagnostics(self):
        """Log final token usage diagnostics."""
        if not self.enable_token_diagnostics or self.token_usage_stats["calls"] == 0:
            return

        stats = self.token_usage_stats
        avg_efficiency = (
            sum(stats["context_efficiency"]) / len(stats["context_efficiency"])
            if stats["context_efficiency"]
            else 0
        )

        effective_limit = self._get_effective_context_limit()
        limit_source = "discovered" if self.discovered_context_limit else "configured"

        logger.info("=== LLM Usage Diagnostics ===")
        logger.info(f"Context window: {effective_limit:,} tokens ({limit_source})")
        logger.info(f"Total LLM calls: {stats['calls']}")
        logger.info(f"Total input tokens: {stats['total_input_tokens']:,}")
        logger.info(f"Total output tokens: {stats['total_output_tokens']:,}")
        logger.info(f"Average context efficiency: {avg_efficiency:.1%}")
        logger.info(
            f"Estimated cost efficiency: {(stats['total_input_tokens'] + stats['total_output_tokens']) / (stats['calls'] * effective_limit):.1%}"
        )

        if (
            self.discovered_context_limit
            and self.discovered_context_limit != self.max_context_tokens
        ):
            improvement = (self.discovered_context_limit / self.max_context_tokens - 1) * 100
            if improvement > 0:
                logger.info(
                    f"SUCCESS - Context discovery found {improvement:.0f}% larger window than configured!"
                )
            else:
                logger.warning(
                    f"WARNING - Context discovery found {abs(improvement):.0f}% smaller window than configured"
                )

        if avg_efficiency < 0.5:
            logger.info(
                "TIP: Context usage is low - consider increasing batch sizes for better LLM utilization"
            )
        elif avg_efficiency > 0.9:
            logger.warning(
                "WARNING: Context usage is very high - consider reducing batch sizes or file content"
            )
        else:
            logger.info("OK - Context usage is well-optimized")

    def _remove_thinking_tokens(self, text: str) -> str:
        """Remove thinking tokens from LLM response based on configured format."""
        if not self.remove_thinking_tokens:
            return text

        # Built-in format patterns
        format_patterns = {
            "harmony": [
                r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|<\|end\|>|$)",
                r"<\|channel\|>commentary<\|message\|>.*?(?=<\|channel\|>|<\|end\|>|$)",
                r"<\|[^|]*\|>",
                r"<think>.*?</think>",
                r"<thinking>.*?</thinking>",
                r"\[Thought:.*?\]",
                r"\[Internal:.*?\]",
            ],
            "qwen": [
                r"<think>.*?</think>",
                r"<思考>.*?</思考>",
                r"\[思考\].*?\[/思考\]",
                r"思考：.*?(?=\n|\r|\r\n|$)",
            ],
        }

        # Get patterns for the configured format
        if self.thinking_format in format_patterns:
            patterns = format_patterns[self.thinking_format]
        else:
            # Use custom patterns if format not recognized
            patterns = self.custom_thinking_patterns

        # Special handling for Harmony format - try to extract final channel first
        if self.thinking_format == "harmony":
            import re

            final_pattern = r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"
            final_matches = re.findall(final_pattern, text, re.DOTALL)

            if final_matches:
                # Found explicit final channel content
                result = final_matches[-1].strip()
                logger.debug(f"Extracted final channel content ({len(result)} chars)")
                return result

        # Remove all patterns
        import re

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        # Log the cleaning if significant content was removed
        original_length = len(text)
        cleaned_length = len(cleaned_text)
        if original_length - cleaned_length > 50:
            logger.debug(
                f"Removed {original_length - cleaned_length} characters of thinking tokens"
            )

        return cleaned_text

    def _detect_unremoved_thinking_tokens(self, text: str) -> list[str]:
        """Detect potential thinking tokens that weren't removed."""
        common_patterns = {
            "<think>": r"<think>.*?</think>",
            "<reasoning>": r"<reasoning>.*?</reasoning>",
            "[THINKING]": r"\[THINKING\].*?\[/THINKING\]",
            "```thinking": r"```thinking.*?```",
            "<thought>": r"<thought>.*?</thought>",
            "# Thinking:": r"# Thinking:.*?(?=\n|\r|\r\n|$)",
            "## Analysis": r"## Analysis.*?(?=\n#|\n\n|$)",
        }

        detected = []
        for name, pattern in common_patterns.items():
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                detected.append(name)

        return detected

    def _warn_about_unremoved_tokens(self, text: str) -> None:
        """Warn user about potential unremoved thinking tokens and suggest configuration."""
        detected_patterns = self._detect_unremoved_thinking_tokens(text)

        if detected_patterns and len(detected_patterns) > 0:
            logger.warning(
                f"Detected potential unremoved thinking tokens: {', '.join(detected_patterns)}. "
                f"Consider updating your vibelint configuration:"
            )
            logger.warning(
                "Set thinking_format='custom' and add custom_thinking_patterns to your pyproject.toml:"
            )

            # Suggest specific patterns based on what was detected
            suggestions = []
            for pattern_name in detected_patterns:
                if pattern_name == "<think>":
                    suggestions.append("r'<think>.*?</think>'")
                elif pattern_name == "<reasoning>":
                    suggestions.append("r'<reasoning>.*?</reasoning>'")
                elif pattern_name == "[THINKING]":
                    suggestions.append("r'\\[THINKING\\].*?\\[/THINKING\\]'")
                elif pattern_name == "```thinking":
                    suggestions.append("r'```thinking.*?```'")
                elif pattern_name == "<thought>":
                    suggestions.append("r'<thought>.*?</thought>'")
                elif pattern_name == "# Thinking:":
                    suggestions.append("r'# Thinking:.*?(?=\\n|\\r|\\r\\n|$)'")
                elif pattern_name == "## Analysis":
                    suggestions.append("r'## Analysis.*?(?=\\n#|\\n\\n|$)'")

            if suggestions:
                logger.warning(
                    f"Add to [tool.vibelint.llm_analysis]: custom_thinking_patterns = {suggestions}"
                )

    def _analyze_global_structure(
        self, current_file: Path, analysis_files: list[Path]
    ) -> Iterator[Finding]:
        """Multi-phase intelligent analysis: file summaries → embedding similarity → pairwise comparison → global synthesis."""

        if not analysis_files:
            logger.warning("No analysis files provided")
            return

        logger.info(f"Starting multi-phase architectural analysis on {len(analysis_files)} files")

        # Phase 1: Generate individual file summaries using DFS traversal
        yield from self._phase1_generate_summaries(analysis_files)

        # Phase 2: Compute embeddings and find semantic similarities
        yield from self._phase2_semantic_clustering(analysis_files)

        # Phase 3: Pairwise analysis of similar files
        yield from self._phase3_pairwise_analysis()

        # Phase 4: Global synthesis of all findings
        yield from self._phase4_global_synthesis(current_file)

        # Log final token usage diagnostics
        self._log_final_diagnostics()

        logger.info(f"Multi-phase architectural analysis COMPLETED on {len(analysis_files)} files")

    def _phase1_generate_summaries(self, analysis_files: list[Path]) -> Iterator[Finding]:
        """Phase 1: Generate file summaries in efficient batches."""
        logger.info(f"Phase 1: Generating summaries for {len(analysis_files)} files")

        # Sort files by depth for DFS-like processing (deeper files first)
        sorted_files = sorted(analysis_files, key=lambda p: len(p.parts), reverse=True)

        # Batch files for efficiency (process 8 files per LLM call)
        batch_size = 8
        batches = [
            sorted_files[i : i + batch_size] for i in range(0, len(sorted_files), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            try:
                # Create batch prompt with multiple files
                batch_content = []
                for file_path in batch:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        batch_content.append(f"FILE: {file_path}\n{content[:1500]}")
                    except Exception as e:
                        batch_content.append(f"FILE: {file_path}\nERROR: Could not read file - {e}")

                batch_prompt = f"""Analyze these Python files and provide concise summaries for each:

{chr(10).join(batch_content)}

For each file, provide a summary in this format:
FILE: [filename]
PURPOSE: What this file does
EXPORTS: Key classes/functions it exports
DEPENDENCIES: What it imports/depends on
CONCERNS: Any potential issues
---
"""

                batch_summaries = self._invoke_with_diagnostics(batch_prompt)

                # Parse batch response and assign to individual files
                self._parse_batch_summaries(batch, batch_summaries)
                logger.debug(f"Processed batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)")

            except Exception as e:
                logger.warning(f"Failed to process batch {batch_idx + 1}: {e}")
                # Fall back to individual error summaries
                for file_path in batch:
                    self._file_summaries[file_path] = f"ERROR: Batch processing failed - {e}"

        logger.info(f"Phase 1 complete: Generated {len(self._file_summaries)} summaries")
        return iter([])  # No findings yet, just building state

    def _parse_batch_summaries(self, batch_files: list[Path], batch_response: str) -> None:
        """Parse batch summary response and assign to individual files."""
        # Split by file markers
        file_sections = re.split(r"FILE:\s*([^\n]+)", batch_response)

        # file_sections[0] is empty, then alternates between filename and content

        for i in range(1, len(file_sections), 2):
            if i + 1 < len(file_sections):
                filename_part = file_sections[i].strip()
                summary_content = file_sections[i + 1].split("---")[0].strip()

                # Find matching file by name
                for file_path in batch_files:
                    if file_path.name in filename_part or str(file_path) in filename_part:
                        self._file_summaries[file_path] = summary_content
                        break

        # Ensure all files have summaries (fallback)
        for file_path in batch_files:
            if file_path not in self._file_summaries:
                self._file_summaries[file_path] = (
                    "ERROR: Could not parse summary from batch response"
                )

    def _parse_batch_pairwise(
        self, batch_pairs: list[tuple[Path, Path]], batch_response: str
    ) -> None:
        """Parse batch pairwise response and assign to individual pairs."""
        # Split by comparison markers
        comparison_sections = re.split(r"COMPARISON\s*(\d+):", batch_response)

        # comparison_sections[0] is empty, then alternates between comparison number and content
        for i in range(1, len(comparison_sections), 2):
            if i + 1 < len(comparison_sections):
                comparison_num = int(comparison_sections[i].strip()) - 1  # Convert to 0-based index
                analysis_content = comparison_sections[i + 1].split("---")[0].strip()

                # Assign to corresponding pair
                if 0 <= comparison_num < len(batch_pairs):
                    file1, file2 = batch_pairs[comparison_num]
                    self._pairwise_analyses[(file1, file2)] = analysis_content

        # Ensure all pairs have analyses (fallback)
        for file1, file2 in batch_pairs:
            if (file1, file2) not in self._pairwise_analyses:
                self._pairwise_analyses[(file1, file2)] = (
                    "ERROR: Could not parse pairwise analysis from batch response"
                )

    def _phase2_semantic_clustering(self, analysis_files: list[Path]) -> Iterator[Finding]:
        """Phase 2: Compute embeddings and find semantically similar files."""
        logger.info(f"Phase 2: Computing semantic similarities for {len(analysis_files)} files")

        try:
            # Get the shared embedding model
            embedding_model = self._get_embedding_model()
            if not embedding_model:
                logger.warning("Embedding model not available, skipping semantic clustering")
                return iter([])

            # Extract embeddings from summaries
            for file_path, summary in self._file_summaries.items():
                if summary and not summary.startswith("ERROR:"):
                    # Use the embedding model to get vector representation
                    embedding = embedding_model.encode([summary])[0].tolist()
                    self._file_embeddings[file_path] = embedding

            # Find similar file pairs using cosine similarity
            from itertools import combinations

            import numpy as np

            similar_pairs = []
            files_with_embeddings = list(self._file_embeddings.keys())

            for file1, file2 in combinations(files_with_embeddings, 2):
                emb1 = np.array(self._file_embeddings[file1])
                emb2 = np.array(self._file_embeddings[file2])

                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

                # Group files with >70% similarity for pairwise analysis
                if similarity > 0.7:
                    similar_pairs.append((file1, file2, similarity))
                    logger.debug(
                        f"Found similar files: {file1.name} <-> {file2.name} ({similarity:.3f})"
                    )

            # Create clusters from similar pairs
            clusters = []
            for file1, file2, _sim in similar_pairs:
                # Find existing cluster or create new one
                placed = False
                for cluster in clusters:
                    if file1 in cluster or file2 in cluster:
                        cluster.update([file1, file2])
                        placed = True
                        break
                if not placed:
                    clusters.append({file1, file2})

            self._similarity_clusters = [list(cluster) for cluster in clusters]
            logger.info(
                f"Phase 2 complete: Found {len(self._similarity_clusters)} similarity clusters"
            )

        except Exception as e:
            logger.warning(f"Phase 2 semantic clustering failed: {e}")

        return iter([])  # No findings yet

    def _phase3_pairwise_analysis(self) -> Iterator[Finding]:
        """Phase 3: Deep pairwise analysis of semantically similar files (batched for efficiency)."""
        logger.info(f"Phase 3: Analyzing {len(self._similarity_clusters)} similarity clusters")

        # Collect all pairs from all clusters
        all_pairs = []
        for cluster in self._similarity_clusters:
            if len(cluster) < 2:
                continue

            from itertools import combinations

            cluster_pairs = list(combinations(cluster, 2))
            all_pairs.extend(cluster_pairs)

        if not all_pairs:
            logger.info("Phase 3 complete: No similarity pairs found")
            return iter([])

        logger.info(f"Phase 3: Analyzing {len(all_pairs)} similarity pairs")

        # Batch pairs for efficiency (process 4 pairs per LLM call)
        batch_size = 4
        pair_batches = [all_pairs[i : i + batch_size] for i in range(0, len(all_pairs), batch_size)]

        for batch_idx, batch in enumerate(pair_batches):
            try:
                # Create batch prompt with multiple pairs
                batch_comparisons = []
                for file1, file2 in batch:
                    summary1 = self._file_summaries.get(file1, "No summary")
                    summary2 = self._file_summaries.get(file2, "No summary")

                    comparison = f"""COMPARISON {len(batch_comparisons) + 1}:
FILE 1: {file1}
{summary1[:400]}

FILE 2: {file2}
{summary2[:400]}
"""
                    batch_comparisons.append(comparison)

                batch_prompt = f"""Analyze these file pairs for duplication and refactoring opportunities:

{chr(10).join(batch_comparisons)}

For each comparison, analyze:
- Code duplication opportunities
- Potential for shared abstractions
- Redundant functionality
- Refactoring suggestions

Format for each:
COMPARISON [N]:
DUPLICATION: What's duplicated
ABSTRACTION: What could be abstracted
REFACTOR: How to refactor
---
"""

                batch_analyses = self._invoke_with_diagnostics(batch_prompt)

                # Parse batch response and assign to individual pairs
                self._parse_batch_pairwise(batch, batch_analyses)
                logger.debug(
                    f"Processed pair batch {batch_idx + 1}/{len(pair_batches)} ({len(batch)} pairs)"
                )

            except Exception as e:
                logger.warning(f"Failed to process pair batch {batch_idx + 1}: {e}")
                # Fall back to individual error analyses
                for file1, file2 in batch:
                    self._pairwise_analyses[(file1, file2)] = f"Error analyzing pair: {e}"

        logger.info(f"Phase 3 complete: Analyzed {len(self._pairwise_analyses)} file pairs")
        return iter([])  # No findings yet

    def _phase4_global_synthesis(self, current_file: Path) -> Iterator[Finding]:
        """Phase 4: Synthesize all findings into global architectural insights."""
        logger.info("Phase 4: Synthesizing global architectural findings")

        # Combine all summaries and pairwise analyses
        all_summaries = "\n\n".join(
            [f"{path}: {summary}" for path, summary in self._file_summaries.items()]
        )
        all_pairwise = "\n\n".join(
            [f"{p1} <-> {p2}: {analysis}" for (p1, p2), analysis in self._pairwise_analyses.items()]
        )

        synthesis_prompt = f"""Based on individual file summaries and pairwise comparisons, identify architectural issues:

FILE SUMMARIES:
{all_summaries[:3000]}

PAIRWISE ANALYSES:
{all_pairwise[:2000]}

Identify systemic issues in this format:

ISSUE: Architectural problem description
SUGGESTION: How to fix it
---
ISSUE: Another architectural problem
SUGGESTION: Another solution
---

Focus on:
- Widespread code duplication patterns
- Missing architectural layers
- Inconsistent patterns across the codebase
- Opportunities for shared abstractions
"""

        try:
            cleaned_response = self._invoke_with_diagnostics(synthesis_prompt)

            # Parse findings using regex
            import re

            issue_pattern = r"ISSUE:\s*(.*?)\s*SUGGESTION:\s*(.*?)(?=---|\s*$)"
            findings = re.findall(issue_pattern, cleaned_response, re.DOTALL | re.IGNORECASE)

            if findings:
                for issue, suggestion in findings:
                    issue = issue.strip()
                    suggestion = suggestion.strip()
                    if issue and suggestion:
                        yield self.create_finding(
                            message=f"[ARCHITECTURAL] {issue} - {suggestion}",
                            file_path=current_file,
                            line=1,
                        )
            else:
                # Fallback
                yield self.create_finding(
                    message=f"Global architectural analysis: {cleaned_response[:400]}...",
                    file_path=current_file,
                    line=1,
                )

        except Exception as e:
            logger.warning(f"Phase 4 synthesis failed: {e}")

        logger.info("Phase 4 complete: Global synthesis finished")

    def _clean_response(self, response) -> str:
        """Clean and extract response content."""
        response_text = response.content if hasattr(response, "content") else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        cleaned = self._remove_thinking_tokens(response_text)
        self._warn_about_unremoved_tokens(cleaned)
        return cleaned

    def _get_embedding_model(self):
        """Get the shared embedding model from the rules engine."""
        try:
            # Check if the model was passed in config
            if self.config and hasattr(self.config, "get"):
                shared_model = self.config.get("_shared_model")
                if shared_model:
                    return shared_model

            # Try to load the model directly
            from sentence_transformers import SentenceTransformer

            # Use the default EmbeddingGemma model
            model_name = "google/embeddinggemma-300m"
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            return model

        except Exception as e:
            logger.warning(f"Could not access embedding model: {e}")
            return None

    def _create_structured_summary(self, file_paths: list[Path]) -> str:
        """Create a structured summary optimized for LLM analysis with advanced compression strategies."""
        summary_parts = []
        total_tokens_estimate = 0

        # Rough token estimation: ~4 chars per token
        max_content_tokens = self.max_prompt_tokens - 2000  # Reserve space for prompt template

        # Limit files to prevent token overflow
        limited_files = file_paths[: self.max_files]

        # First pass: collect and prioritize content
        file_contents = []
        for path in limited_files:
            try:
                content = path.read_text(encoding="utf-8")
                compressed_content = self._compress_file_content(path, content)
                if compressed_content:
                    file_contents.append((path, compressed_content))
            except Exception as e:
                logger.debug(f"Could not analyze {path}: {e}")

        # Second pass: fit content within token budget
        for path, compressed_content in file_contents:
            try:
                rel_path = path.relative_to(path.parents[2])
            except (ValueError, IndexError):
                rel_path = path

            # Create file summary with compression indicators
            file_summary = f"=== {rel_path} ===\n{compressed_content}"

            # Estimate tokens for this file summary
            file_tokens = len(file_summary) // 4

            # Check if adding this file would exceed context window
            if total_tokens_estimate + file_tokens > max_content_tokens:
                logger.info(
                    f"Context window limit reached, truncating at {len(summary_parts)} files"
                )
                break

            summary_parts.append(file_summary)
            total_tokens_estimate += file_tokens

        result = "\n\n".join(summary_parts)
        logger.info(
            f"Generated compressed summary with ~{total_tokens_estimate} tokens for {len(summary_parts)} files"
        )
        return result

    def _compress_file_content(self, file_path: Path, content: str) -> str:
        """Apply multiple compression strategies to file content."""
        lines = content.split("\n")

        # Strategy 1: Extract architectural signatures
        signatures = self._extract_architectural_signatures(lines)

        # Strategy 2: Summarize imports and dependencies
        import_summary = self._summarize_imports(lines) if self.enable_import_summary else None

        # Strategy 3: Extract class/function hierarchy
        hierarchy = (
            self._extract_code_hierarchy(lines) if self.enable_hierarchy_extraction else None
        )

        # Strategy 4: Find architectural patterns and anti-patterns
        patterns = self._identify_patterns(lines) if self.enable_pattern_detection else None

        # Strategy 5: Extract complexity indicators
        complexity = (
            self._analyze_complexity_indicators(lines, file_path)
            if self.enable_complexity_analysis
            else None
        )

        # Combine all compressed information
        compressed_parts = []

        if import_summary:
            compressed_parts.append(f"IMPORTS: {import_summary}")

        if hierarchy:
            compressed_parts.append(f"STRUCTURE:\n{hierarchy}")

        if signatures:
            compressed_parts.append(
                f"KEY_CODE:\n{chr(10).join(signatures[:self.max_signature_lines])}"
            )

        if patterns:
            compressed_parts.append(f"PATTERNS: {patterns}")

        if complexity:
            compressed_parts.append(f"COMPLEXITY: {complexity}")

        return "\n".join(compressed_parts)

    def _extract_architectural_signatures(self, lines: list[str]) -> list[str]:
        """Extract the most architecturally significant lines."""
        signatures = []

        for line in lines[: self.max_file_lines]:
            stripped = line.strip()

            # High priority: classes, functions, decorators
            if (
                stripped.startswith(("class ", "def ", "async def ", "@"))
                or
                # Medium priority: control flow and error handling
                any(keyword in stripped for keyword in ["raise", "except", "finally", "with"])
                or
                # Architecture indicators
                any(pattern in stripped for pattern in ["TODO", "FIXME", "XXX", "HACK", "NOTE"])
                or
                # Design patterns
                any(
                    pattern in stripped
                    for pattern in ["factory", "singleton", "observer", "strategy", "adapter"]
                )
                or
                # Type hints and contracts
                "->" in stripped
                or ": " in stripped
                and "=" not in stripped
            ):

                signatures.append(line)

        return signatures

    def _summarize_imports(self, lines: list[str]) -> str:
        """Summarize import structure to understand dependencies."""
        imports = {"stdlib": set(), "third_party": set(), "local": set()}

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                if stripped.startswith("from .") or stripped.startswith("from .."):
                    imports["local"].add(stripped)
                elif any(
                    stdlib in stripped
                    for stdlib in ["os", "sys", "json", "logging", "pathlib", "typing"]
                ):
                    imports["stdlib"].add(stripped)
                else:
                    imports["third_party"].add(stripped)

        summary_parts = []
        if imports["stdlib"]:
            summary_parts.append(f"stdlib({len(imports['stdlib'])})")
        if imports["third_party"]:
            summary_parts.append(f"3rd_party({len(imports['third_party'])})")
        if imports["local"]:
            summary_parts.append(f"local({len(imports['local'])})")

        return ", ".join(summary_parts)

    def _extract_code_hierarchy(self, lines: list[str]) -> str:
        """Extract class and function hierarchy structure."""
        hierarchy = []
        current_class = None
        indent_level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Calculate indentation
            leading_spaces = len(line) - len(line.lstrip())

            if stripped.startswith("class "):
                class_name = stripped.split("(")[0].replace("class ", "").strip(":")
                current_class = class_name
                hierarchy.append(f"class {class_name}")
                indent_level = leading_spaces

            elif stripped.startswith(("def ", "async def ")) and current_class:
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                if leading_spaces > indent_level:  # Method inside class
                    hierarchy.append(f"  └─ {func_name}()")
                else:  # New top-level function
                    current_class = None
                    hierarchy.append(f"def {func_name}()")

            elif stripped.startswith(("def ", "async def ")):
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "").strip()
                hierarchy.append(f"def {func_name}()")
                current_class = None

        return "\n".join(hierarchy[: self.max_hierarchy_depth])  # Use configured hierarchy depth

    def _identify_patterns(self, lines: list[str]) -> str:
        """Identify architectural patterns and anti-patterns."""
        patterns = []

        # Count pattern indicators
        class_count = sum(1 for line in lines if line.strip().startswith("class "))
        function_count = sum(1 for line in lines if line.strip().startswith(("def ", "async def ")))
        import_count = sum(1 for line in lines if line.strip().startswith(("import ", "from ")))

        # Identify specific patterns
        has_inheritance = any("(" in line and line.strip().startswith("class ") for line in lines)
        has_decorators = any(line.strip().startswith("@") for line in lines)
        has_async = any("async " in line for line in lines)
        has_context_managers = any("with " in line for line in lines)
        has_exceptions = any(word in line for line in lines for word in ["raise", "except", "try:"])

        # Anti-pattern detection
        long_functions = []
        for i, line in enumerate(lines):
            if line.strip().startswith(("def ", "async def ")):
                # Count lines until next function/class or end
                func_lines = 0
                for j in range(i + 1, min(i + 100, len(lines))):
                    if lines[j].strip().startswith(("def ", "class ", "async def ")):
                        break
                    if lines[j].strip():  # Non-empty line
                        func_lines += 1
                if func_lines > 50:  # Arbitrary threshold
                    func_name = line.split("(")[0].replace("def ", "").replace("async ", "").strip()
                    long_functions.append(f"{func_name}({func_lines}L)")

        # Build pattern summary
        if class_count:
            patterns.append(f"classes({class_count})")
        if function_count:
            patterns.append(f"functions({function_count})")
        if import_count > 10:
            patterns.append(f"heavy_imports({import_count})")
        if has_inheritance:
            patterns.append("inheritance")
        if has_decorators:
            patterns.append("decorators")
        if has_async:
            patterns.append("async")
        if has_context_managers:
            patterns.append("context_mgmt")
        if has_exceptions:
            patterns.append("error_handling")
        if long_functions:
            patterns.append(f"long_funcs[{','.join(long_functions[:3])}]")

        return ", ".join(patterns)

    def _analyze_complexity_indicators(self, lines: list[str], file_path: Path) -> str:
        """Analyze complexity and quality indicators."""
        indicators = []

        # File size indicators
        total_lines = len(lines)
        code_lines = len(
            [line for line in lines if line.strip() and not line.strip().startswith("#")]
        )

        # Complexity indicators
        nested_blocks = 0
        max_nesting = 0
        current_nesting = 0

        for line in lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in ["if ", "for ", "while ", "with ", "try:"]):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
                nested_blocks += 1
            # Simple heuristic for block end (not perfect but gives indication)
            elif stripped and not line.startswith(" ") and current_nesting > 0:
                current_nesting = max(0, current_nesting - 1)

        # Quality indicators
        todo_count = sum(
            1
            for line in lines
            if any(marker in line.upper() for marker in ["TODO", "FIXME", "XXX", "HACK"])
        )
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))

        # Build complexity summary
        if total_lines > 200:
            indicators.append(f"large_file({total_lines}L)")
        if max_nesting > 3:
            indicators.append(f"deep_nesting({max_nesting})")
        if todo_count > 0:
            indicators.append(f"todos({todo_count})")
        if comment_lines / max(code_lines, 1) < 0.1:
            indicators.append("low_comments")
        elif comment_lines / max(code_lines, 1) > 0.3:
            indicators.append("well_documented")

        return ", ".join(indicators) if indicators else "moderate_complexity"
