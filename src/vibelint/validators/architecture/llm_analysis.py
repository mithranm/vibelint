"""
LLM-powered architectural analysis validator for vibelint.

Detects over-engineering patterns and architectural redundancies that traditional
rule-based linting cannot identify, using an OpenAI-compatible API for semantic analysis.

vibelint/validators/architecture_llm.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...config import Config
from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["ArchitectureLLMValidator"]
logger = logging.getLogger(__name__)


class ArchitectureLLMValidator(BaseValidator):
    """
    Validates architectural patterns using LLM analysis to detect over-engineering,
    unnecessary abstractions, and semantic redundancies that escape traditional linting.
    """

    rule_id = "ARCHITECTURE-LLM"

    def __init__(self, severity: Optional[Severity] = None, config: Optional[Dict] = None):
        super().__init__(severity, config)
        self._api_base_url: Optional[str] = None
        self._model_name: Optional[str] = None
        self._session: Optional[requests.Session] = None

    def _setup_llm_client(self, config: Config) -> bool:
        """
        Initialize LLM client with configuration from vibelint config.

        Returns:
            True if LLM API is available and configured, False otherwise.
        """
        # Check for LLM configuration in vibelint config
        llm_config = config.get("llm_analysis", {})

        if not isinstance(llm_config, dict):
            logger.debug("LLM analysis disabled: no llm_analysis config section found")
            return False

        self._api_base_url = llm_config.get("api_base_url")
        self._model_name = llm_config.get("model")
        self._api_key = llm_config.get("api_key")
        self._max_tokens = llm_config.get("max_tokens", 2048)
        self._temperature = llm_config.get("temperature", 0.3)
        self._top_p = llm_config.get("top_p", 0.9)
        self._top_k = llm_config.get("top_k", 40)
        self._frequency_penalty = llm_config.get("frequency_penalty", 0.1)
        self._presence_penalty = llm_config.get("presence_penalty", 0.1)
        self._timeout_seconds = llm_config.get("timeout_seconds", 120)

        if not self._api_base_url:
            logger.debug("LLM analysis disabled: no api_base_url configured")
            return False

        if not self._model_name:
            logger.debug("LLM analysis disabled: no model specified")
            return False

        # Check if we're running in CI and if LLM analysis is explicitly enabled
        is_ci = any(
            ci_var in os.environ
            for ci_var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "TRAVIS", "CIRCLECI"]
        )

        if is_ci and not llm_config.get("enable_in_ci", False):
            logger.debug(
                "LLM analysis disabled: running in CI environment and enable_in_ci is not set"
            )
            return False

        # Setup requests session with retry strategy
        self._session = requests.Session()
        retry_strategy = Retry(
            total=2, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Set authorization header if API key is provided
        if self._api_key:
            self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        # Test API connectivity
        try:
            response = self._session.get(f"{self._api_base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info(f"LLM analysis enabled using API at {self._api_base_url}")
                return True
            else:
                logger.warning(f"LLM API test failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.debug(f"LLM API connectivity test failed: {e}")
            return False

    def _analyze_codebase_structure(self, project_root: Path) -> dict:
        """
        Analyze codebase structure to identify potential architectural issues.

        Args:
            project_root: Root directory of the project

        Returns:
            Dictionary containing structural analysis data
        """
        structure = {"files": [], "modules": {}, "potential_issues": []}

        # Collect Python files and their basic info
        for py_file in project_root.rglob("*.py"):
            try:
                rel_path = py_file.relative_to(project_root)
                file_info = {
                    "path": str(rel_path),
                    "size": py_file.stat().st_size,
                    "lines": len(py_file.read_text(encoding="utf-8", errors="ignore").splitlines()),
                }

                # Analyze imports and basic structure
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                file_info["imports"] = self._extract_imports(content)
                file_info["classes"] = self._extract_classes(content)
                file_info["functions"] = self._extract_functions(content)

                structure["files"].append(file_info)

            except Exception as e:
                logger.debug(f"Error analyzing {py_file}: {e}")

        return structure

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements from file content."""
        imports = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(("import ", "from ")):
                imports.append(line)
        return imports

    def _extract_classes(self, content: str) -> list[str]:
        """Extract class definitions from file content."""
        classes = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("class "):
                class_name = line.split()[1].split("(")[0].rstrip(":")
                classes.append(class_name)
        return classes

    def _extract_functions(self, content: str) -> list[str]:
        """Extract function definitions from file content."""
        functions = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("def "):
                func_name = line.split()[1].split("(")[0]
                functions.append(func_name)
        return functions

    def _query_llm_for_analysis(self, codebase_summary: str) -> Optional[dict]:
        """
        Query the LLM API for architectural analysis.

        Args:
            codebase_summary: Summarized codebase structure for analysis

        Returns:
            Analysis results from LLM or None if API call fails
        """
        if not self._session or not self._api_base_url:
            return None

        prompt = f"""Analyze this Python file for architectural over-engineering patterns.

{codebase_summary}

Respond with a JSON object containing your analysis. Use this exact format:
{{
    "has_issues": true/false,
    "summary": "Brief 1-2 sentence summary of findings",
    "details": "Detailed explanation if issues found, or empty string if none"
}}

Look for: thin wrapper classes, unnecessary middle layers, over-complex data classes, fake plugin systems, premature abstractions."""

        try:
            payload = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
                "top_k": self._top_k,
                "frequency_penalty": self._frequency_penalty,
                "presence_penalty": self._presence_penalty,
                "response_format": {"type": "json_object"},
            }

            response = self._session.post(
                f"{self._api_base_url}/v1/chat/completions",
                json=payload,
                timeout=self._timeout_seconds,
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract content from harmony format if present
                if "<|message|>" in response_text:
                    # Look for final channel content
                    if "<|channel|>final<|message|>" in response_text:
                        final_start = response_text.find("<|channel|>final<|message|>") + len(
                            "<|channel|>final<|message|>"
                        )
                        content = response_text[final_start:].split("<|end|>")[0].strip()
                    else:
                        # Fallback to any message content
                        message_start = response_text.find("<|message|>") + len("<|message|>")
                        content = response_text[message_start:].split("<|end|>")[0].strip()
                else:
                    content = response_text.strip()

                # Parse JSON response
                try:
                    analysis_json = json.loads(content)
                    if analysis_json.get("has_issues", False):
                        return analysis_json
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from LLM response: {content[:200]}...")
                    # Fallback to text analysis
                    if content and "no significant" not in content.lower():
                        return {"analysis": content}

            else:
                logger.warning(f"LLM API request failed with status {response.status_code}")
                logger.debug(f"Response: {response.text[:200]}...")

        except Exception as e:
            logger.debug(f"LLM analysis request failed: {e}")

        return None

    def validate(self, file_path: Path, content: str, config: Config) -> Iterator[Finding]:
        """
        Perform LLM-powered architectural validation on individual files.

        Analyzes files that are likely to contain architectural issues based on patterns:
        - Files with "runner", "manager", "system" in the name
        - Files with many classes but few methods per class
        - Files that are thin wrappers around other modules
        """
        # Skip if LLM is not available or not configured
        if not hasattr(self, "_llm_setup_attempted"):
            self._llm_setup_attempted = True
            self._llm_available = self._setup_llm_client(config)

        if not self._llm_available:
            return

        # Check if this file is worth analyzing for architectural issues
        if not self._should_analyze_file(file_path, content):
            return

        logger.info(f"Running LLM architectural analysis on {file_path}")

        # Create focused analysis prompt for this specific file
        analysis_prompt = self._create_file_analysis_prompt(file_path, content)

        # Query LLM for analysis
        analysis_result = self._query_llm_for_analysis(analysis_prompt)

        if analysis_result:
            if "summary" in analysis_result and "details" in analysis_result:
                # Structured JSON response
                summary = analysis_result["summary"]
                details = analysis_result.get("details", "")
                message = f"{summary}"
                if details:
                    message += f" Details: {details}"
            elif "analysis" in analysis_result:
                # Fallback text response
                message = analysis_result["analysis"]
            else:
                return

            yield Finding(
                rule_id=f"{self.rule_id}-ANALYSIS",
                message=message,
                file_path=file_path,
                line=1,
                severity=self.severity,
            )

    def _should_analyze_file(self, file_path: Path, content: str) -> bool:
        """
        Determine if a file is worth analyzing for architectural issues.

        Returns True for files that commonly contain architectural issues:
        - Files with suspicious names (runner, manager, system, wrapper, facade)
        - Files that are short but have many imports
        - Files with many small classes
        """
        filename = file_path.name.lower()

        # Check for suspicious filename patterns
        suspicious_patterns = [
            "runner",
            "manager",
            "system",
            "wrapper",
            "facade",
            "handler",
            "controller",
        ]
        if any(pattern in filename for pattern in suspicious_patterns):
            logger.debug(f"Analyzing {file_path} due to suspicious filename pattern")
            return True

        # Analyze content patterns
        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines < 20:
            return False  # Too small to have architectural issues

        # Count imports, classes, functions
        import_count = len(
            [line for line in lines if line.strip().startswith(("import ", "from "))]
        )
        class_count = len([line for line in lines if line.strip().startswith("class ")])
        len([line for line in lines if line.strip().startswith("def ")])

        # High import-to-code ratio might indicate a thin wrapper
        if import_count > 5 and total_lines < 150 and import_count / total_lines > 0.1:
            logger.debug(f"Analyzing {file_path} due to high import-to-code ratio")
            return True

        # Many small classes might indicate over-engineering
        if class_count > 3 and total_lines / class_count < 50:
            logger.debug(f"Analyzing {file_path} due to many small classes")
            return True

        return False

    def _create_file_analysis_prompt(self, file_path: Path, content: str) -> str:
        """Create a focused prompt for analyzing a specific file."""
        # Truncate content if too long
        lines = content.splitlines()
        if len(lines) > 200:
            content_preview = (
                "\n".join(lines[:100]) + "\n\n... [truncated] ...\n\n" + "\n".join(lines[-50:])
            )
        else:
            content_preview = content

        return f"""Analyze this Python file for architectural over-engineering patterns:

File: {file_path.name}

{content_preview}

Respond with a JSON object using this exact format:
{{
    "has_issues": true/false,
    "summary": "Brief 1-2 sentence summary of findings",
    "details": "Detailed explanation if issues found, or empty string if none"
}}

Look for: thin wrapper classes, unnecessary middle layers, over-complex data classes, fake plugin systems, premature abstractions."""
