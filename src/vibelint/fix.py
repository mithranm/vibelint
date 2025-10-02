"""
Automatic fix functionality for vibelint using deterministic fixes and LLM for docstring generation only.

vibelint/src/vibelint/fix.py
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from vibelint.config import Config
from vibelint.validators import Finding

__all__ = ["FixEngine", "can_fix_finding", "apply_fixes", "regenerate_all_docstrings"]

logger = logging.getLogger(__name__)


class FixEngine:
    """Engine for automatically fixing vibelint issues with deterministic code changes."""

    def __init__(self, config: Config):
        """Initialize fix engine with configuration.

        vibelint/src/vibelint/fix.py
        """
        self.config = config

        # Initialize LLM manager for dual LLM support
        from vibelint.llm_client import create_llm_manager

        config_dict = config.settings if isinstance(config.settings, dict) else {}
        self.llm_manager = create_llm_manager(config_dict)

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
        """Generate only docstring text content using dual LLM system (safe operation)."""
        if not self.llm_manager:
            logger.debug("No LLM manager configured, skipping docstring generation")
            return None

        try:
            from vibelint.llm import LLMRequest

            # Safe prompt - only asks for docstring text, never code
            if isinstance(node, ast.ClassDef):
                prompt = f"Write a brief docstring for a Python class named '{node.name}'. Return only the docstring text without quotes or formatting."
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args] if hasattr(node, "args") else []
                prompt = f"Write a brief docstring for a Python function named '{node.name}' with parameters {args}. Return only the docstring text without quotes or formatting."
            else:
                logger.debug("Unknown node type for docstring generation")
                return None

            # Use fast LLM for quick docstring generation
            request = LLMRequest(
                content=prompt, task_type="docstring_generation", max_tokens=200, temperature=0.1
            )

            response = await self.llm_manager.process_request(request)

            if response.success and response.content:
                # Clean the response to ensure it's just text
                content = str(response.content).strip()
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
        """Add path references to existing docstrings based on configuration."""
        # Get docstring configuration
        config_dict = self.config.settings if isinstance(self.config.settings, dict) else {}
        docstring_config = config_dict.get("docstring", {})
        require_path_references = docstring_config.get("require_path_references", False)

        # Skip fix if path references are not required
        if not require_path_references:
            return

        # Get path format configuration
        path_format = docstring_config.get("path_reference_format", "relative")
        expected_path = self._get_expected_path_for_fix(file_path, path_format)

        for finding in findings:
            line_idx = finding.line - 1  # Convert to 0-based index
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                # If this is a docstring line, ensure it has path reference
                if '"""' in line or "'''" in line:
                    # Add path reference if not already present
                    if expected_path not in line:
                        # Modify the docstring to include path
                        indent = self._get_indent_for_line(lines, finding.line)
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            # Single line docstring - expand it
                            quote = '"""' if '"""' in line else "'''"
                            content = line.strip().replace(quote, "").strip()
                            new_docstring = f"{indent}{quote}{content}\n\n{indent}{expected_path}\n{indent}{quote}"
                            modifications[finding.line - 1] = new_docstring

    def _get_expected_path_for_fix(self, file_path: Path, path_format: str) -> str:
        """Get expected path reference for fix based on format configuration."""
        if path_format == "absolute":
            return str(file_path)
        elif path_format == "module_path":
            # Convert to Python module path (e.g., vibelint.validators.docstring)
            parts = file_path.parts
            if "src" in parts:
                src_idx = parts.index("src")
                module_parts = parts[src_idx + 1 :]
            else:
                module_parts = parts

            # Remove .py extension and convert to module path
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

            return ".".join(module_parts)
        else:  # relative format (default)
            # Get relative path, removing project root and src/ prefix
            relative_path = str(file_path)
            try:
                # Try to find project root by looking for common markers
                current = file_path.parent
                while current.parent != current:
                    if any(
                        (current / marker).exists()
                        for marker in ["pyproject.toml", "setup.py", ".git"]
                    ):
                        relative_path = str(file_path.relative_to(current))
                        break
                    current = current.parent
            except ValueError:
                pass

            # Remove src/ prefix if present
            if relative_path.startswith("src/"):
                relative_path = relative_path[4:]

            return relative_path

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


async def regenerate_all_docstrings(config: Config, file_paths: List[Path]) -> int:
    """Regenerate ALL docstrings in the specified files using LLM.

    Unlike apply_fixes which only adds missing docstrings, this function
    regenerates existing docstrings as well for consistency and improved quality.

    Returns the number of files successfully processed.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    processed_count = 0

    if not engine.llm_config.get("api_base_url"):
        logger.error("No LLM API configured. Cannot regenerate docstrings.")
        return 0

    for file_path in file_paths:
        try:
            if await _regenerate_docstrings_in_file(engine, file_path):
                processed_count += 1
                logger.info(f"Regenerated docstrings in {file_path}")
            else:
                logger.debug(f"No docstrings to regenerate in {file_path}")
        except Exception as e:
            logger.error(f"Failed to regenerate docstrings in {file_path}: {e}")

    return processed_count


async def preview_docstring_changes(config: Config, file_paths: List[Path]) -> Dict[str, Any]:
    """Preview what docstring changes would be made without modifying files.

    Returns a dictionary containing:
    - files_analyzed: list of files that would be changed
    - total_changes: total number of docstring changes
    - preview_samples: dict of file -> list of preview changes

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    preview_results = {
        "files_analyzed": [],
        "total_changes": 0,
        "preview_samples": {},
        "errors": [],
    }

    if not engine.llm_config.get("api_base_url"):
        preview_results["errors"].append("No LLM API configured. Cannot preview docstring changes.")
        return preview_results

    for file_path in file_paths:
        try:
            file_preview = await _preview_docstrings_in_file(engine, file_path)
            if file_preview["changes"]:
                preview_results["files_analyzed"].append(str(file_path))
                preview_results["total_changes"] += len(file_preview["changes"])
                preview_results["preview_samples"][str(file_path)] = file_preview["changes"]
        except Exception as e:
            preview_results["errors"].append(f"Failed to preview {file_path}: {e}")

    return preview_results


async def _preview_docstrings_in_file(engine: FixEngine, file_path: Path) -> Dict[str, Any]:
    """Preview docstring changes for a single file without modifying it.

    Returns dict with 'changes' list containing preview information.
    """
    file_preview = {"changes": []}

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Parse the file to find all functions and classes
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                change_preview = await _preview_node_docstring(engine, node, lines, file_path)
                if change_preview:
                    file_preview["changes"].append(change_preview)

    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        logger.error(f"Error previewing {file_path}: {e}")

    return file_preview


async def _preview_node_docstring(
    engine: FixEngine,
    node: ast.AST,
    lines: List[str],
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    """Preview what docstring change would be made for a specific AST node.

    Returns preview dict with change information, or None if no change.
    """
    # SAFETY CHECK: Only process functions and classes that we can safely identify
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None

    # Get current docstring if it exists
    current_docstring = ast.get_docstring(node)

    # Generate new docstring content (this calls the LLM)
    new_docstring_content = await engine._generate_docstring_content(node, file_path)
    if not new_docstring_content:
        return None

    # SAFETY VALIDATION: Check that generated content is reasonable
    if not _validate_docstring_content(new_docstring_content):
        return None

    # Determine what type of change this would be
    change_type = "add" if not current_docstring else "modify"
    node_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"

    return {
        "node_name": node.name,
        "node_type": node_type,
        "line_number": node.lineno,
        "change_type": change_type,
        "current_docstring": (
            current_docstring[:100] + "..."
            if current_docstring and len(current_docstring) > 100
            else current_docstring
        ),
        "new_docstring": (
            new_docstring_content[:100] + "..."
            if len(new_docstring_content) > 100
            else new_docstring_content
        ),
    }


async def _regenerate_docstrings_in_file(engine: FixEngine, file_path: Path) -> bool:
    """Regenerate all docstrings in a single file.

    Returns True if any docstrings were regenerated.

    vibelint/src/vibelint/fix.py
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Parse the file to find all functions and classes
        tree = ast.parse(content)
        modifications = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                await _regenerate_node_docstring(engine, node, lines, modifications, file_path)

        if modifications:
            # Apply modifications
            result_content = _apply_line_modifications(lines, modifications)
            file_path.write_text(result_content, encoding="utf-8")
            return True

        return False

    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


async def _regenerate_node_docstring(
    engine: FixEngine,
    node: ast.AST,
    lines: List[str],
    modifications: Dict[int, str],
    file_path: Path,
) -> None:
    """Regenerate docstring for a specific AST node with strict safety validation.

    SAFETY CRITICAL: This function must NEVER modify any Python code, only docstring content.

    vibelint/src/vibelint/fix.py
    """
    # SAFETY CHECK: Only process functions and classes that we can safely identify
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        logger.warning(f"Skipping unsafe node type {type(node)} for safety")
        return

    # Get current docstring if it exists
    current_docstring = ast.get_docstring(node)

    # SAFETY CHECK: Validate LLM-generated content before using it
    new_docstring_content = await engine._generate_docstring_content(node, file_path)
    if not new_docstring_content:
        return

    # SAFETY VALIDATION: Check that generated content is reasonable
    if not _validate_docstring_content(new_docstring_content):
        logger.warning(f"Generated docstring failed safety validation for {node.name}")
        return

    # Determine indentation safely
    node_line = node.lineno - 1
    if node_line < len(lines):
        indent = len(lines[node_line]) - len(lines[node_line].lstrip())
        indent_str = " " * (indent + 4)  # Add 4 spaces for function/class body
    else:
        indent_str = "    "  # Default indentation

    # Format the new docstring with path reference and safety warning
    new_docstring = (
        f'{indent_str}"""{new_docstring_content}\n\n'
        f"{indent_str}[WARNING]  CRITICAL WARNING: This docstring was auto-generated by LLM and MUST be reviewed.\n"
        f"{indent_str}Inaccurate documentation can cause security vulnerabilities, system failures,\n"
        f"{indent_str}and data corruption. Verify all parameters, return types, and behavior descriptions.\n\n"
        f'{indent_str}{file_path}\n{indent_str}"""'
    )

    if current_docstring:
        # ULTRA SAFE: Find docstring using multiple validation methods
        docstring_lines = _find_docstring_lines_safely(node, lines)

        if docstring_lines:
            # SAFETY CHECK: Verify we're only modifying docstring lines
            for line_num in docstring_lines:
                if line_num < len(lines):
                    line_content = lines[line_num]
                    # CRITICAL: Only modify lines that are clearly part of docstring
                    if not _is_safe_docstring_line(line_content):
                        logger.error(
                            f"SAFETY VIOLATION: Attempted to modify non-docstring line {line_num}: {line_content}"
                        )
                        return  # Abort entire operation for safety

            # Safe to proceed - modify only the first docstring line, remove others
            for i, line_num in enumerate(docstring_lines):
                if i == 0:
                    modifications[line_num] = new_docstring
                else:
                    modifications[line_num] = ""  # Remove continuation lines
        else:
            logger.warning(f"Could not safely locate docstring for {node.name}")
    else:
        # Add new docstring after function/class definition
        insert_line = node.lineno  # Line after def/class
        modifications[insert_line] = new_docstring


def _validate_docstring_content(content: str) -> bool:
    """Validate that LLM-generated docstring content is safe and reasonable.

    vibelint/src/vibelint/fix.py
    """
    if not content or not isinstance(content, str):
        return False

    # Check for dangerous content
    dangerous_patterns = [
        "import ",
        "exec(",
        "eval(",
        "__import__",
        "subprocess",
        "os.system",
        "shell=True",
        "DELETE",
        "DROP TABLE",
        "rm -rf",
    ]

    content_lower = content.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in content_lower:
            logger.error(f"SAFETY: Dangerous pattern '{pattern}' found in generated docstring")
            return False

    # Check reasonable length (docstrings shouldn't be huge)
    if len(content) > 2000:
        logger.warning("Generated docstring is suspiciously long")
        return False

    return True


def _find_docstring_lines_safely(node: ast.AST, lines: List[str]) -> List[int]:
    """Safely find the exact line numbers of a docstring using multiple validation methods.

    vibelint/src/vibelint/fix.py
    """
    # Method 1: Use AST to find docstring node
    docstring_node = None
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
            if isinstance(child.value.value, str):
                docstring_node = child
                break

    if not docstring_node:
        return []

    # Get line range from AST
    start_line = docstring_node.lineno - 1  # Convert to 0-based
    end_line = docstring_node.end_lineno - 1 if docstring_node.end_lineno else start_line

    # Method 2: Verify by examining actual line content
    docstring_lines = []
    for line_num in range(start_line, end_line + 1):
        if line_num < len(lines):
            if _is_safe_docstring_line(lines[line_num]):
                docstring_lines.append(line_num)
            else:
                # If any line in the range is not a docstring line, abort for safety
                logger.error(f"SAFETY: Line {line_num} in docstring range is not a docstring line")
                return []

    return docstring_lines


def _is_safe_docstring_line(line: str) -> bool:
    """Check if a line is definitely part of a docstring and safe to modify.

    vibelint/src/vibelint/fix.py
    """
    stripped = line.strip()

    # Must contain quotes or be empty/whitespace (for multi-line docstrings)
    if not stripped:
        return True  # Empty line within docstring

    # Must contain docstring quotes
    if '"""' in stripped or "'''" in stripped:
        return True

    # If it doesn't start with quotes, it might be docstring content
    # But we need to be very careful - check it doesn't look like code
    if any(
        pattern in stripped
        for pattern in ["def ", "class ", "import ", "=", "return ", "if ", "for ", "while "]
    ):
        return False

    # If it's indented and looks like text, probably docstring content
    if line.startswith("    ") and not stripped.startswith(("#", "//")):
        return True

    # Default to false for safety
    return False


def _apply_line_modifications(lines: List[str], modifications: Dict[int, str]) -> str:
    """Apply line modifications to content deterministically.

    vibelint/src/vibelint/fix.py
    """
    result_lines = lines[:]

    # Sort modifications by line number in reverse order to avoid index shifting
    sorted_modifications = sorted(modifications.items(), reverse=True)

    for line_num, new_content in sorted_modifications:
        if line_num < len(result_lines):
            if new_content == "":
                # Remove line
                result_lines.pop(line_num)
            else:
                result_lines[line_num] = new_content
        elif new_content:
            # Insert at end
            result_lines.append(new_content)

    return "\n".join(result_lines)
