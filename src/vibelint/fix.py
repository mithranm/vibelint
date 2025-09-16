"""
Automatic fix functionality for vibelint using deterministic fixes and LLM for docstring generation only.

vibelint/src/vibelint/fix.py
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .plugin_system import Finding

__all__ = ["FixEngine", "can_fix_finding", "apply_fixes"]

logger = logging.getLogger(__name__)


class FixEngine:
    """Engine for automatically fixing vibelint issues with deterministic code changes."""

    def __init__(self, config: Config):
        """Initialize fix engine with configuration.

        vibelint/src/vibelint/fix.py
        """
        self.config = config
        self.llm_config = config.settings.get("llm_analysis", {})

    def can_fix_finding(self, finding: Finding) -> bool:
        """Check if a finding can be automatically fixed.

        vibelint/src/vibelint/fix.py
        """
        fixable_rules = {
            "DOCSTRING-MISSING",
            "DOCSTRING-PATH-REFERENCE",
            "EXPORTS-MISSING-ALL",
        }
        return finding.rule_id in fixable_rules

    async def fix_file(self, file_path: Path, findings: list[Finding]) -> bool:
        """Fix all fixable issues in a file deterministically.

        Returns True if any fixes were applied.

        vibelint/src/vibelint/fix.py
        """
        fixable_findings = [f for f in findings if self.can_fix_finding(f)]
        if not fixable_findings:
            return False

        logger.info(f"Fixing {len(fixable_findings)} issues in {file_path}")

        # Read current file content
        try:
            original_content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return False

        # Apply deterministic fixes
        fixed_content = await self._apply_deterministic_fixes(
            file_path, original_content, fixable_findings
        )

        if fixed_content and fixed_content != original_content:
            try:
                # Write fixed content back to file
                file_path.write_text(fixed_content, encoding="utf-8")
                logger.info(f"Applied fixes to {file_path}")
                return True
            except Exception as e:
                logger.error(f"Could not write fixes to {file_path}: {e}")
                return False

        return False

    async def _apply_deterministic_fixes(
        self, file_path: Path, content: str, findings: list[Finding]
    ) -> str:
        """Apply deterministic fixes without LLM file rewriting.

        vibelint/src/vibelint/fix.py
        """
        try:
            # Parse the AST to understand the code structure
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Cannot parse {file_path}: {e}")
            return content

        # Track modifications by line number
        lines = content.splitlines()
        modifications = {}

        # Group findings by type for efficient processing
        findings_by_type = {}
        for finding in findings:
            rule_id = finding.rule_id
            if rule_id not in findings_by_type:
                findings_by_type[rule_id] = []
            findings_by_type[rule_id].append(finding)

        # Apply fixes by type
        if "DOCSTRING-MISSING" in findings_by_type:
            await self._fix_missing_docstrings(
                tree, lines, modifications, findings_by_type["DOCSTRING-MISSING"], file_path
            )

        if "DOCSTRING-PATH-REFERENCE" in findings_by_type:
            self._fix_docstring_path_references(
                lines, modifications, findings_by_type["DOCSTRING-PATH-REFERENCE"], file_path
            )

        if "EXPORTS-MISSING-ALL" in findings_by_type:
            self._fix_missing_exports(
                tree, lines, modifications, findings_by_type["EXPORTS-MISSING-ALL"]
            )

        # Apply all modifications to create fixed content
        return self._apply_modifications(lines, modifications)

    async def _fix_missing_docstrings(
        self,
        tree: ast.AST,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
        file_path: Path,
    ) -> None:
        """Add missing docstrings using LLM for content generation only."""

        # Find functions and classes that need docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Check if this node needs a docstring based on findings
                node_line = node.lineno
                needs_docstring = any(
                    abs(f.line - node_line) <= 2 for f in findings  # Allow some line tolerance
                )

                if needs_docstring and not ast.get_docstring(node):
                    # Generate docstring content using LLM (safe - only returns text)
                    docstring_content = await self._generate_docstring_content(node, file_path)

                    if docstring_content:
                        # Deterministically insert the docstring
                        indent = self._get_indent_for_line(lines, node.lineno)
                        docstring_line = (
                            f'{indent}"""{docstring_content}\n\n{indent}{file_path}\n{indent}"""'
                        )

                        # Insert after the function/class definition line
                        insert_line = node.lineno  # Insert after the def line
                        modifications[insert_line] = docstring_line

    async def _generate_docstring_content(self, node: ast.AST, file_path: Path) -> Optional[str]:
        """Generate only docstring text content using LLM (safe operation)."""
        if not self.llm_config.get("api_base_url"):
            # Fallback to simple docstring without LLM
            if isinstance(node, ast.ClassDef):
                return f"{node.name} class implementation."
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return f"{node.name} function implementation."
            return "Implementation."

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr

            api_key = self.llm_config.get("api_key", "dummy-key")
            base_url = self.llm_config.get("api_base_url")
            model = self.llm_config.get("model", "gpt-3.5-turbo")

            llm = ChatOpenAI(
                base_url=base_url + "/v1" if not base_url.endswith("/v1") else base_url,
                model=model,
                temperature=0.1,
                max_completion_tokens=200,  # Small limit for docstring only
                api_key=SecretStr(api_key),
            )

            # Safe prompt - only asks for docstring text, never code
            if isinstance(node, ast.ClassDef):
                prompt = f"Write a brief docstring for a Python class named '{node.name}'. Return only the docstring text without quotes or formatting."
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args] if hasattr(node, "args") else []
                prompt = f"Write a brief docstring for a Python function named '{node.name}' with parameters {args}. Return only the docstring text without quotes or formatting."
            else:
                return "Implementation."

            response = await llm.ainvoke(prompt)
            if response and response.content:
                # Clean the response to ensure it's just text
                content = response.content.strip()
                # Remove any quotes or markdown that might have been added
                content = content.replace('"""', "").replace("'''", "").replace("`", "")
                return content[:200]  # Limit length

        except Exception as e:
            logger.warning(f"LLM docstring generation failed: {e}, using fallback")
            # Safe fallback
            if isinstance(node, ast.ClassDef):
                return f"{node.name} class implementation."
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return f"{node.name} function implementation."
            return "Implementation."

    def _fix_docstring_path_references(
        self,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
        file_path: Path,
    ) -> None:
        """Add path references to existing docstrings."""
        for finding in findings:
            line_idx = finding.line - 1  # Convert to 0-based index
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                # If this is a docstring line, ensure it has path reference
                if '"""' in line or "'''" in line:
                    # Add path reference if not already present
                    if str(file_path) not in line:
                        # Modify the docstring to include path
                        indent = self._get_indent_for_line(lines, finding.line)
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            # Single line docstring - expand it
                            quote = '"""' if '"""' in line else "'''"
                            content = line.strip().replace(quote, "").strip()
                            new_docstring = (
                                f"{indent}{quote}{content}\n\n{indent}{file_path}\n{indent}{quote}"
                            )
                            modifications[finding.line - 1] = new_docstring

    def _fix_missing_exports(
        self,
        tree: ast.AST,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
    ) -> None:
        """Add missing __all__ exports."""
        # Find all public functions and classes
        public_names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):  # Public items
                    public_names.append(node.name)

        if public_names:
            # Check if __all__ already exists
            has_all = any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                for node in ast.walk(tree)
            )

            if not has_all:
                # Add __all__ at the top of the file after imports
                exports_line = f"__all__ = {public_names!r}"

                # Find a good place to insert (after imports)
                insert_line = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        insert_line = i + 1
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                modifications[insert_line] = exports_line

    def _get_indent_for_line(self, lines: List[str], line_number: int) -> str:
        """Get the indentation for a given line number."""
        if 1 <= line_number <= len(lines):
            line = lines[line_number - 1]
            return line[: len(line) - len(line.lstrip())]
        return "    "  # Default 4-space indent

    def _apply_modifications(self, lines: List[str], modifications: Dict[int, str]) -> str:
        """Apply all modifications to the lines and return the fixed content."""
        # Sort modifications by line number in reverse order to avoid index shifting
        sorted_modifications = sorted(modifications.items(), reverse=True)

        result_lines = lines[:]
        for line_num, new_content in sorted_modifications:
            if line_num < len(result_lines):
                result_lines[line_num] = new_content
            else:
                # Insert at end
                result_lines.append(new_content)

        return "\n".join(result_lines)


# Convenience functions for the CLI
async def apply_fixes(config: Config, file_findings: dict[Path, list[Finding]]) -> int:
    """Apply fixes to all files with fixable findings.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    fixed_count = 0

    for file_path, findings in file_findings.items():
        if await engine.fix_file(file_path, findings):
            fixed_count += 1

    return fixed_count


def can_fix_finding(finding: Finding) -> bool:
    """Check if a finding can be automatically fixed.

    vibelint/src/vibelint/fix.py
    """
    return finding.rule_id in {
        "DOCSTRING-MISSING",
        "DOCSTRING-PATH-REFERENCE",
        "EXPORTS-MISSING-ALL",
    }
