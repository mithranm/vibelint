"""
LLM orchestration for justification workflow.

Handles all LLM interactions including file analysis, backup detection,
and final architectural analysis. Separated for better testability and caching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JustificationLLMOrchestrator:
    """Orchestrates LLM calls for the justification workflow."""

    def __init__(self, llm_manager, config: Optional[Dict] = None):
        self.llm_manager = llm_manager
        self.config = config or {}
        self._llm_call_logs = []

    def analyze_file_purpose(self, file_path: Path, analysis: Dict[str, Any], content: str) -> str:
        """Use LLM to analyze file purpose and generate summary."""
        if not self.llm_manager:
            return self._generate_pattern_based_justification(file_path, analysis)

        # Create concise prompt for fast LLM
        functions_list = analysis.get("functions") if "functions" in analysis else []
        functions = [f["name"] for f in functions_list]
        classes_list = analysis.get("classes") if "classes" in analysis else []
        classes = [c["name"] for c in classes_list]
        module_doc = analysis.get("module_docstring") if "module_docstring" in analysis else ""

        prompt = f"""File: {file_path.name}
Functions: {', '.join(functions[:5])}
Classes: {', '.join(classes[:3])}
Docstring: {module_doc[:100] if module_doc else 'None'}

Provide a concise 1-line architectural justification for why this file exists in the codebase."""

        try:
            from vibelint.llm.manager import LLMRequest

            llm_request = LLMRequest(
                content=prompt,
                max_tokens=300,  # Fast LLM route
                temperature=0.1
            )

            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success") and response.get("content"):
                summary = response["content"].strip()
                self._llm_call_logs.append({
                    "operation": "file_purpose_analysis",
                    "file": str(file_path),
                    "response": summary,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                return summary
            else:
                logger.warning(f"LLM failed for {file_path}, using pattern-based justification")
                return self._generate_pattern_based_justification(file_path, analysis)

        except Exception as e:
            logger.error(f"LLM analysis failed for {file_path}: {e}")
            return self._generate_pattern_based_justification(file_path, analysis)

    def _generate_pattern_based_justification(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate justification based on code patterns when LLM unavailable."""
        functions_list = analysis.get("functions") if "functions" in analysis else []
        functions = [f["name"] for f in functions_list]
        classes_list = analysis.get("classes") if "classes" in analysis else []
        classes = [c["name"] for c in classes_list]

        file_name = file_path.name
        path_parts = file_path.parts

        # Pattern-based analysis
        if "cli" in path_parts:
            return f"CLI command interface providing {', '.join(functions[:2])} commands"
        elif "validators" in path_parts:
            return f"Validation module with {len(functions)} validation functions"
        elif "workflows" in path_parts:
            return f"Workflow implementation with {len(classes)} classes"
        elif "test" in file_name or "test" in path_parts:
            return f"Test module with {len(functions)} test functions"
        elif file_name == "__init__.py":
            return "Package initialization and exports"
        elif classes and functions:
            return f"Module with {len(classes)} classes and {len(functions)} functions"
        elif classes:
            return f"Class definitions module with {len(classes)} classes"
        elif functions:
            return f"Utility module with {len(functions)} functions"
        else:
            return "Configuration or data module"

    def detect_backup_files(self, directory_path: Path) -> List[Dict[str, str]]:
        """Use LLM to intelligently detect backup and temporary files."""
        if not self.llm_manager:
            return []

        # Get file list
        all_files = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size < 1024 * 1024:  # < 1MB
                        all_files.append(str(file_path.relative_to(directory_path)))
                except (OSError, ValueError):
                    continue

        if not all_files:
            return []

        # Create prompt for backup detection
        file_list = "\n".join(all_files[:100])  # Limit to avoid token limits
        prompt = f"""Analyze this file list and identify backup, temporary, or archive files that should NOT be committed to version control.

File List:
{file_list}

Look for patterns like:
- .bak, .backup, .orig extensions
- Files ending with ~ (Unix backup)
- .tmp, .temp extensions
- Numbered versions (file_v2.py, file2.py when file.py exists)
- OS-specific temporary files (.DS_Store, Thumbs.db, etc.)
- Editor backup files (.swp, .swo, etc.)

Return ONLY a JSON array of objects with this format:
[{{"file": "path/to/file", "reason": "why this should not be committed"}}]
If no backup files found, return: []"""

        try:
            from vibelint.llm.manager import LLMRequest

            llm_request = LLMRequest(
                content=prompt,
                max_tokens=1000,
                temperature=0.1
            )

            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success") and response.get("content"):
                try:
                    backup_files = json.loads(response["content"].strip())
                    if isinstance(backup_files, list):
                        self._llm_call_logs.append({
                            "operation": "backup_file_detection",
                            "response": f"Found {len(backup_files)} backup files",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        return backup_files
                except json.JSONDecodeError:
                    logger.warning("LLM backup file detection returned invalid JSON")

        except Exception as e:
            logger.debug(f"LLM backup file detection failed: {e}")

        return []

    def perform_final_analysis(self, justification_report: str,
                              static_issues: Optional[Dict[str, Any]] = None) -> str:
        """Use orchestrator LLM for comprehensive final analysis."""
        if not self.llm_manager:
            return "LLM orchestrator not available for final analysis"

        # Prepare static analysis section
        static_analysis = ""
        if static_issues:
            static_analysis = self._format_static_issues(static_issues)

        # Create comprehensive prompt
        prompt = f"""You are an expert software architect reviewing a Python codebase analysis. Based on the detailed report below, provide a comprehensive assessment of the codebase architecture.

# Detailed Analysis Report

{justification_report}

{static_analysis}

## Instructions

Analyze this codebase and provide:

### Architectural Health
[Overall structural assessment - is this well-organized or problematic?]

### Critical Issues
[Top 3-5 most important architectural problems that should be addressed first]

### Dead Code Validation
[Assessment of the dependency analysis accuracy and real vs false positives]

### Recommendations
[Prioritized action items for improving the codebase]

Focus on actionable insights that will improve code maintainability and quality.

**Pay special attention to:**
- Files in wrong directories (e.g., vibelint-specific code outside src/vibelint/, scripts in tools/, CLI code outside cli/)
- Package organization and module placement
- Architectural boundaries and separation of concerns
- Redundant or versioned implementations (e.g., FooV2 alongside Foo, engine_v2.py alongside engine.py)
- Unnecessary backward compatibility when not required
- Dead code or unused modules that should be removed

**IMPORTANT**: If the codebase is already well-structured with no significant issues, simply respond with "LGTM - No significant architectural issues found." Don't feel obligated to find problems where none exist."""

        try:
            from vibelint.llm.manager import LLMRequest

            llm_request = LLMRequest(
                content=prompt,
                max_tokens=8192,  # Large output for comprehensive analysis
                temperature=0.3
            )

            logger.info("Calling orchestrator LLM for final analysis...")
            response = self.llm_manager.process_request_sync(llm_request)

            if response and response.get("success") and response.get("content"):
                analysis = response["content"].strip()
                logger.info("Orchestrator final analysis completed successfully")

                self._llm_call_logs.append({
                    "operation": "final_analysis",
                    "response": analysis,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # Combine LLM analysis with static analysis
                combined_analysis = f"{analysis}\n\n---\n\n{static_analysis}" if static_analysis else analysis
                return combined_analysis
            else:
                error_msg = f"Orchestrator LLM analysis failed: {response}"
                logger.warning(error_msg)
                return f"LLM analysis failed: {error_msg}\n\n---\n\n{static_analysis}" if static_analysis else f"LLM analysis failed: {error_msg}"

        except Exception as e:
            error_msg = f"Orchestrator final analysis error: {type(e).__name__}: {e}"
            logger.error(error_msg)
            return f"Analysis error: {error_msg}\n\n---\n\n{static_analysis}" if static_analysis else f"Analysis error: {error_msg}"

    def _format_static_issues(self, static_issues: Dict[str, Any]) -> str:
        """Format static issues for inclusion in analysis report."""
        formatted = "# Static Analysis Issues\n\n"

        # Add backup files if detected
        backup_files = static_issues.get("backup_files", [])
        if backup_files:
            formatted += "## Backup Files Detected\n\n"
            for backup_file in backup_files:
                formatted += f"- **{backup_file['file']}**: {backup_file['reason']}\n"
            formatted += "\n"

        # Add naming issues if any
        naming_issues = static_issues.get("naming_issues", [])
        if naming_issues:
            formatted += "## Naming Issues Detected\n\n"
            for issue in naming_issues:
                formatted += f"- `{issue['file']}`: {issue['issue']} (severity: {issue['severity']})\n"
            formatted += "\n"

        return formatted

    def get_llm_logs(self) -> List[Dict[str, Any]]:
        """Get all LLM call logs for debugging and analysis."""
        return self._llm_call_logs.copy()

    def clear_logs(self) -> None:
        """Clear LLM call logs."""
        self._llm_call_logs.clear()