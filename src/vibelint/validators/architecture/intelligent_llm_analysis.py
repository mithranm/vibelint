"""
Intelligent LLM-powered architectural analysis validator for vibelint.

Uses a multi-phase approach:
1. Global project structure analysis to identify potentially problematic areas
2. Pairwise file comparison for identified files
3. Structured JSON output with specific architectural issues

vibelint/validators/architecture/intelligent_llm_analysis.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...plugin_system import BaseValidator, Finding, Severity

__all__ = ["IntelligentLLMValidator"]
logger = logging.getLogger(__name__)


class IntelligentLLMValidator(BaseValidator):
    """
    Intelligent LLM-powered architectural analysis using multi-phase approach.

    Phase 1: Analyzes entire project structure to identify potentially problematic files
    Phase 2: Performs pairwise comparison of flagged files
    Phase 3: Generates specific architectural findings
    """

    rule_id = "ARCHITECTURE-INTELLIGENT-LLM"
    default_severity = Severity.INFO

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(severity, config)
        self._api_base_url: Optional[str] = None
        self._model_name: Optional[str] = None
        self._session: Optional[requests.Session] = None
        self._project_files: List[Path] = []
        self._analysis_cache: Dict[str, Any] = {}

    def _setup_llm_client(self, config: Dict[str, Any]) -> bool:
        """Initialize LLM client with configuration."""
        llm_config = config.get("llm_analysis", {})

        if not isinstance(llm_config, dict):
            logger.debug("Intelligent LLM analysis disabled: no llm_analysis config section found")
            return False

        self._api_base_url = llm_config.get("api_base_url")
        self._model_name = llm_config.get("model")
        self._api_key = llm_config.get("api_key")
        self._max_tokens = llm_config.get("max_tokens", 4096)
        self._temperature = llm_config.get("temperature", 0.3)

        if not self._api_base_url or not self._model_name:
            logger.debug("Intelligent LLM analysis disabled: missing api_base_url or model")
            return False

        # Setup requests session
        self._session = requests.Session()
        retry_strategy = Retry(
            total=2, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        if self._api_key:
            self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        # Test connectivity
        try:
            response = self._session.get(f"{self._api_base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info(f"Intelligent LLM analysis enabled using API at {self._api_base_url}")
                return True
            else:
                logger.warning(f"LLM API test failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.debug(f"LLM API connectivity test failed: {e}")
            return False

    def _query_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Query the LLM API with a prompt and return JSON response."""
        if not self._session or not self._api_base_url:
            return None

        try:
            payload = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "response_format": {"type": "json_object"},
            }

            response = self._session.post(
                f"{self._api_base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Handle different response formats
                if "<|message|>" in response_text:
                    if "<|channel|>final<|message|>" in response_text:
                        final_start = response_text.find("<|channel|>final<|message|>") + len(
                            "<|channel|>final<|message|>"
                        )
                        content = response_text[final_start:].split("<|end|>")[0].strip()
                    else:
                        message_start = response_text.find("<|message|>") + len("<|message|>")
                        content = response_text[message_start:].split("<|end|>")[0].strip()
                else:
                    content = response_text.strip()

                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from LLM response: {content[:200]}...")
                    return None
            else:
                logger.warning(f"LLM API request failed with status {response.status_code}")
                return None

        except Exception as e:
            logger.debug(f"LLM analysis request failed: {e}")
            return None

    def _collect_project_structure(self, project_root: Path, file_paths: List[Path]) -> str:
        """Create a project structure overview for LLM analysis."""
        structure_lines = ["# Project Structure Overview\n"]

        # Group files by directory
        dirs_files: Dict[str, List[Path]] = {}
        for file_path in file_paths:
            try:
                rel_path = file_path.relative_to(project_root)
                dir_path = str(rel_path.parent)
                if dir_path not in dirs_files:
                    dirs_files[dir_path] = []
                dirs_files[dir_path].append(rel_path)
            except ValueError:
                continue

        # Create tree representation
        for dir_path in sorted(dirs_files.keys()):
            structure_lines.append(f"\n## Directory: {dir_path}")
            files = sorted(dirs_files[dir_path])

            for file_path in files:
                # Add basic file info
                try:
                    full_path = project_root / file_path
                    if full_path.exists():
                        size = full_path.stat().st_size
                        lines = len(
                            full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        )
                        structure_lines.append(f"- {file_path.name} ({lines} lines, {size} bytes)")
                except Exception:
                    structure_lines.append(f"- {file_path.name}")

        return "\n".join(structure_lines)

    def _phase1_global_analysis(
        self, project_root: Path, file_paths: List[Path]
    ) -> Optional[List[str]]:
        """Phase 1: Global project structure analysis to identify potentially problematic files."""
        logger.info("Phase 1: Running global project structure analysis")

        structure_overview = self._collect_project_structure(project_root, file_paths)

        prompt = f"""Analyze this Python project structure for potential architectural issues.

{structure_overview}

Look for patterns that indicate:
1. Code duplication across files
2. Overly similar file names suggesting redundant functionality
3. Thin wrapper files that might be unnecessary abstractions
4. Files that appear to be doing too many things (poor separation of concerns)
5. Missing abstractions where there should be shared code

Respond with a JSON object containing:
{{
    "potentially_problematic_files": [
        "path/to/file1.py",
        "path/to/file2.py"
    ],
    "reasoning": "Brief explanation of why these files were flagged",
    "comparison_pairs": [
        ["file1.py", "file2.py"],
        ["file3.py", "file4.py"]
    ]
}}

Focus on files that likely contain architectural issues, not every file in the project."""

        response = self._query_llm(prompt)
        if response and "potentially_problematic_files" in response:
            flagged_files = response["potentially_problematic_files"]
            logger.info(
                f"Phase 1 complete: Flagged {len(flagged_files)} potentially problematic files"
            )
            logger.debug(f"Reasoning: {response.get('reasoning', 'No reasoning provided')}")

            # Store comparison pairs for phase 2
            self._analysis_cache["comparison_pairs"] = response.get("comparison_pairs", [])
            return flagged_files

        logger.warning("Phase 1 failed: No valid response from LLM")
        return None

    def _phase2_pairwise_comparison(
        self, project_root: Path, comparison_pairs: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Phase 2: Pairwise comparison of flagged files."""
        logger.info(f"Phase 2: Running pairwise comparison of {len(comparison_pairs)} file pairs")

        issues = []

        for pair in comparison_pairs:
            if len(pair) != 2:
                continue

            file1_path = project_root / pair[0]
            file2_path = project_root / pair[1]

            if not (file1_path.exists() and file2_path.exists()):
                continue

            try:
                file1_content = file1_path.read_text(encoding="utf-8", errors="ignore")
                file2_content = file2_path.read_text(encoding="utf-8", errors="ignore")

                # Limit content size for LLM
                if len(file1_content) > 3000:
                    file1_content = file1_content[:3000] + "\n... [truncated]"
                if len(file2_content) > 3000:
                    file2_content = file2_content[:3000] + "\n... [truncated]"

                prompt = f"""Compare these two Python files for architectural redundancy and issues:

## File 1: {pair[0]}
```python
{file1_content}
```

## File 2: {pair[1]}
```python
{file2_content}
```

Analyze for:
1. Duplicate or very similar functionality
2. One file being a thin wrapper around the other
3. Opportunities for consolidation or better abstraction
4. Poor separation of concerns

Respond with JSON:
{{
    "has_issues": true/false,
    "issue_type": "duplication|thin_wrapper|poor_separation|other",
    "severity": "low|medium|high",
    "description": "Specific description of the architectural issue",
    "recommendation": "Specific recommendation for improvement"
}}"""

                response = self._query_llm(prompt)
                if response and response.get("has_issues"):
                    issues.append(
                        {
                            "files": pair,
                            "issue_type": response.get("issue_type", "unknown"),
                            "severity": response.get("severity", "medium"),
                            "description": response.get("description", ""),
                            "recommendation": response.get("recommendation", ""),
                        }
                    )

            except Exception as e:
                logger.debug(f"Failed to compare {pair[0]} and {pair[1]}: {e}")
                continue

        logger.info(f"Phase 2 complete: Found {len(issues)} architectural issues")
        return issues

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Run intelligent multi-phase LLM architectural analysis.

        Only runs on the first file encountered, then analyzes the entire project.
        """
        # Skip if LLM is not available or already analyzed
        if not hasattr(self, "_llm_setup_attempted"):
            self._llm_setup_attempted = True
            self._llm_available = config is not None and self._setup_llm_client(config)

        if not self._llm_available:
            return

        # Only run analysis once per project (on first file)
        if hasattr(self, "_analysis_completed"):
            return

        self._analysis_completed = True

        # Collect all project files
        project_root = config.get("project_root", file_path.parent) if config else file_path.parent
        self._project_files = list(project_root.rglob("*.py"))

        logger.info(
            f"Starting intelligent LLM architectural analysis on {len(self._project_files)} files"
        )

        # Phase 1: Global analysis
        flagged_files = self._phase1_global_analysis(project_root, self._project_files)
        if not flagged_files:
            return

        # Phase 2: Pairwise comparison
        comparison_pairs = self._analysis_cache.get("comparison_pairs", [])
        if comparison_pairs:
            issues = self._phase2_pairwise_comparison(project_root, comparison_pairs)

            # Generate findings
            for issue in issues:
                files_str = " and ".join(issue["files"])
                message = f"Architectural issue in {files_str}: {issue['description']}"

                yield self.create_finding(
                    message=message,
                    file_path=file_path,  # Report on current file
                    line=1,
                )
