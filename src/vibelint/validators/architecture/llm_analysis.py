"""
LLM-powered architectural analysis validator using OpenAI-compatible APIs.

Provides intelligent architectural analysis using Large Language Models
to detect design issues, inconsistencies, and improvement opportunities.

vibelint/validators/architecture/llm_analysis.py
"""

import json
import logging
import os
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

        # Get the actual files being analyzed from the validation engine
        analysis_files = config.get("_analysis_files", [file_path]) if config else [file_path]

        logger.info(f"Starting LLM architectural analysis on {len(analysis_files)} files")

        # Run architectural analysis on the specified files
        yield from self._analyze_global_structure(file_path, analysis_files)

    def _test_connectivity(self) -> bool:
        """Test if OpenAI-compatible API service is available."""
        try:
            # Simple connectivity test with ChatOpenAI
            self.llm.invoke("Test connectivity")
            return True
        except Exception as e:
            logger.debug(f"LLM connectivity test failed: {e}")
            return False

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

    def _analyze_global_structure(
        self, current_file: Path, analysis_files: list[Path]
    ) -> Iterator[Finding]:
        """Analyze overall codebase architecture using intelligent multi-phase langchain approach."""

        if not analysis_files:
            logger.warning("No analysis files provided")
            return

        logger.info(f"Phase 1: Running global structure analysis on {len(analysis_files)} files")

        # Phase 1: Create code context for LLM
        code_content = self._create_structured_summary(analysis_files)

        # Phase 1: Global analysis with raw channel response (not JSON)
        phase1_prompt = f"""Analyze this Python codebase structure for potential architectural issues:

{code_content}

Look for patterns that indicate:
1. Code duplication across files
2. Overly similar functionality suggesting redundant code
3. Poor separation of concerns
4. Missing abstractions where there should be shared code
5. Architectural anti-patterns

This is Phase 1 of a multi-phase analysis. Provide your analysis as natural language commentary about what you observe."""

        try:
            # Phase 1: Use langchain for initial analysis (expects channel/message format)
            phase1_response = self.global_chain.invoke({"code_content": phase1_prompt})

            # Extract and clean raw response content
            response_text = (
                phase1_response.content
                if hasattr(phase1_response, "content")
                else str(phase1_response)
            )
            if not isinstance(response_text, str):
                response_text = str(response_text)
            cleaned_response = self._remove_thinking_tokens(response_text)
            logger.info(f"Phase 1 complete: {cleaned_response[:200]}...")

            # Phase 2: Follow up with structured JSON request
            phase2_prompt = f"""Based on your previous analysis:

{cleaned_response}

Now provide specific architectural findings as valid JSON with this structure:
{{
    "findings": [
        {{
            "message": "Description of the issue",
            "suggestion": "How to fix it",
            "line": 1,
            "severity": "info"
        }}
    ]
}}

Return ONLY the JSON, no other text."""

            # Phase 2: Get structured JSON response
            phase2_response = self.global_chain.invoke({"code_content": phase2_prompt})

            json_text = (
                phase2_response.content
                if hasattr(phase2_response, "content")
                else str(phase2_response)
            )

            # Ensure we have a string for JSON parsing and clean thinking tokens
            if not isinstance(json_text, str):
                json_text = str(json_text)
            cleaned_json_text = self._remove_thinking_tokens(json_text)

            try:
                analysis_data = json.loads(cleaned_json_text)
                findings_list = analysis_data.get("findings", [])

                # Convert findings to plugin system findings
                for finding_data in findings_list:
                    yield self.create_finding(
                        message=f"{finding_data.get('message', '')} {finding_data.get('suggestion', '')}".strip(),
                        file_path=current_file,  # Report on current file being processed
                        line=finding_data.get("line", 1),
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Phase 2 JSON parsing failed: {e}")
                # Fallback: create finding from Phase 1 analysis
                yield self.create_finding(
                    message=f"Architectural analysis: {cleaned_response[:300]}...",
                    file_path=current_file,
                    line=1,
                )

        except Exception as e:
            logger.warning(f"Global LLM analysis failed: {e}")

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
