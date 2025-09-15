"""
Automatic fix functionality for vibelint using LLM integration.

vibelint/src/vibelint/fix.py
"""

import logging
import re
from pathlib import Path

from .config import Config
from .plugin_system import Finding

__all__ = ["FixEngine", "can_fix_finding", "apply_fixes"]

logger = logging.getLogger(__name__)


class FixEngine:
    """Engine for automatically fixing vibelint issues using LLM."""

    def __init__(self, config: Config):
        """Initialize fix engine with configuration.

        vibelint/src/vibelint/fix.py
        """
        self.config = config
        self.llm_config = getattr(config, "llm_analysis", {})

    def can_fix_finding(self, finding: Finding) -> bool:
        """Check if a finding can be automatically fixed.

        vibelint/src/vibelint/fix.py
        """
        fixable_rules = {
            "DOCSTRING-MISSING",
            "DOCSTRING-PATH-REFERENCE",
            "EXPORTS-MISSING-ALL",
        }
        return finding.rule in fixable_rules

    async def fix_file(self, file_path: Path, findings: list[Finding]) -> bool:
        """Fix all fixable issues in a file.

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

        # Generate fixes using LLM
        fixed_content = await self._generate_fixes_llm(
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

    async def _generate_fixes_llm(
        self, file_path: Path, content: str, findings: list[Finding]
    ) -> str | None:
        """Generate fixes using the configured LLM.

        vibelint/src/vibelint/fix.py
        """
        if not self.llm_config.get("api_base_url"):
            logger.warning("LLM not configured, cannot generate fixes")
            return None

        try:
            # Initialize LLM using langchain (same as the validators)
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr

            api_key = self.llm_config.get("api_key", "dummy-key")
            base_url = self.llm_config.get("api_base_url")
            model = self.llm_config.get("model", "gpt-3.5-turbo")

            llm = ChatOpenAI(
                base_url=base_url + "/v1" if not base_url.endswith("/v1") else base_url,
                model=model,
                temperature=self.llm_config.get("temperature", 0.1),  # Low temperature for fixes
                max_completion_tokens=self.llm_config.get("max_tokens", 4000),
                api_key=SecretStr(api_key),
            )

            # Build fix prompt
            prompt = self._build_fix_prompt(file_path, content, findings)

            # Call LLM
            response = await llm.ainvoke(prompt)

            if response and response.content:
                # Extract fixed code from response
                return self._extract_fixed_code(response.content)

        except Exception as e:
            logger.error(f"LLM fix generation failed: {e}")

        return None

    def _build_fix_prompt(self, file_path: Path, content: str, findings: list[Finding]) -> str:
        """Build a prompt for the LLM to fix the issues.

        vibelint/src/vibelint/fix.py
        """
        issues_description = "\n".join(
            [f"- Line {f.line}: {f.rule} - {f.message}" for f in findings]
        )

        return f"""Fix the following Python code issues automatically:

FILE: {file_path}

ISSUES TO FIX:
{issues_description}

RULES:
1. For DOCSTRING-MISSING: Add proper docstrings with path references
2. For DOCSTRING-PATH-REFERENCE: Add the file path at the end of docstrings like: "filename.py"
3. For EXPORTS-MISSING-ALL: Add __all__ = [...] with public functions/classes

ORIGINAL CODE:
```python
{content}
```

Please provide ONLY the fixed Python code without any explanation or markdown formatting. The response should be valid Python code that can be written directly to the file.
"""

    def _extract_fixed_code(self, llm_response: str) -> str:
        """Extract the fixed code from LLM response.

        vibelint/src/vibelint/fix.py
        """
        # Remove markdown code blocks if present
        response = llm_response.strip()

        # Check for code blocks and extract
        code_block_pattern = r"```(?:python)?\n?(.*?)\n?```"
        match = re.search(code_block_pattern, response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If no code blocks, return the response as-is (assume it's all code)
        return response


def can_fix_finding(finding: Finding) -> bool:
    """Check if a finding can be automatically fixed.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(Config())  # Basic config for rule check
    return engine.can_fix_finding(finding)


async def apply_fixes(config: Config, file_findings: dict[Path, list[Finding]]) -> dict[Path, bool]:
    """Apply fixes to multiple files.

    Returns dict mapping file paths to whether fixes were applied.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    results = {}

    for file_path, findings in file_findings.items():
        try:
            results[file_path] = await engine.fix_file(file_path, findings)
        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")
            results[file_path] = False

    return results
