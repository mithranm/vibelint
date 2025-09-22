"""
Workflow 1: Single File Validation

Implements the single file validation workflow that takes a Python file and runs
validation checks against it, returning structured violation reports with actionable feedback.

This follows the requirements defined in WORKFLOW_1_REQUIREMENTS.md and the workflow
specification in VIBELINT_WORKFLOWS.md.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config
from ..plugin_system import Finding, Severity, plugin_manager
from ..rules import RuleEngine

logger = logging.getLogger(__name__)


class SingleFileValidationResult:
    """Results from single file validation workflow."""

    def __init__(self):
        self.file_path: Optional[Path] = None
        self.success: bool = False
        self.violations: List[Finding] = []
        self.health_score: float = 0.0
        self.error_message: Optional[str] = None
        self.ast_tree: Optional[ast.AST] = None
        self.execution_time_ms: float = 0.0


class SingleFileValidator:
    """Core single file validation functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.rule_engine = RuleEngine(config.settings)

        # Load single file validators
        plugin_manager.load_plugins()
        self.validators = self._load_single_file_validators()

    def _load_single_file_validators(self) -> Dict[str, type]:
        """Load validators from src/vibelint/validators/single_file/."""
        validators = {}
        all_validators = plugin_manager.get_all_validators()

        for rule_id, validator_class in all_validators.items():
            module_name = validator_class.__module__
            # Filter for single file validators
            if "validators.single_file" in module_name:
                validators[rule_id] = validator_class
                logger.debug(f"Loaded single file validator: {rule_id}")

        return validators

    def validate_file(self, file_path: Path) -> SingleFileValidationResult:
        """
        Validate a single Python file.

        Implements the workflow activities from VIBELINT_WORKFLOWS.md:
        1. File Loading - Read and validate file exists
        2. Single File Validator Execution - Run validators
        3. Violation Aggregation - Collect and sort violations
        4. Report Generation - Generate structured output
        """
        import time

        start_time = time.time()

        result = SingleFileValidationResult()
        result.file_path = file_path

        try:
            # Activity 1: File Loading
            if not self._validate_file_exists(file_path):
                result.error_message = f"File not found or not readable: {file_path}"
                return result

            # Parse Python AST
            ast_tree = self._parse_python_file(file_path)
            if ast_tree is None:
                result.error_message = f"Failed to parse Python AST for: {file_path}"
                return result

            result.ast_tree = ast_tree

            # Activity 2: Single File Validator Execution
            violations = self._run_validators(file_path, ast_tree)

            # Activity 3: Violation Aggregation
            violations = self._aggregate_violations(violations)

            # Activity 4: Report Generation
            result.violations = violations
            result.health_score = self._calculate_health_score(violations)
            result.success = True

        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}", exc_info=True)
            result.error_message = str(e)
            result.success = False

        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _validate_file_exists(self, file_path: Path) -> bool:
        """Validate file exists and is readable."""
        try:
            return file_path.exists() and file_path.is_file() and file_path.suffix == ".py"
        except (OSError, PermissionError):
            return False

    def _parse_python_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse Python file into AST representation."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return ast.parse(content, filename=str(file_path))
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def _run_validators(self, file_path: Path, ast_tree: ast.AST) -> List[Finding]:
        """Run each validator against the file."""
        all_violations = []
        enabled_validators = self.rule_engine.get_enabled_validators()
        enabled_rule_ids = {v.rule_id for v in enabled_validators}

        for rule_id, validator_class in self.validators.items():
            if rule_id not in enabled_rule_ids:
                logger.debug(f"Skipping disabled validator: {rule_id}")
                continue

            try:
                # Instantiate validator using rule engine to ensure proper configuration
                validator = self.rule_engine.create_validator_instance(validator_class)
                if not validator:
                    logger.warning(f"Failed to create validator instance for {rule_id}")
                    continue

                # Run validator with proper interface
                if hasattr(validator, "validate_file"):
                    violations = validator.validate_file(file_path, ast_tree)
                elif hasattr(validator, "validate"):
                    # Read file content for validators that need it
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        violations = list(validator.validate(file_path, content))
                    except (OSError, UnicodeDecodeError) as e:
                        logger.warning(
                            f"Could not read file {file_path} for validator {rule_id}: {e}"
                        )
                        continue
                else:
                    logger.warning(f"Validator {rule_id} has no validate method")
                    continue

                if violations:
                    all_violations.extend(violations)
                    logger.debug(f"Validator {rule_id} found {len(violations)} violations")

            except Exception as e:
                logger.error(f"Error running validator {rule_id}: {e}", exc_info=True)
                # Create error finding using the correct interface
                error_finding = Finding(
                    rule_id=rule_id,
                    message=f"Validator error: {e}",
                    file_path=file_path,
                    line=1,
                    column=1,
                    severity=Severity.BLOCK,
                    suggestion=None,
                )
                all_violations.append(error_finding)

        return all_violations

    def _aggregate_violations(self, violations: List[Finding]) -> List[Finding]:
        """
        Aggregate violations from all validators.

        - Sort by line number and severity
        - Apply ignore rules and exceptions
        """
        # Apply ignore rules first
        filtered_violations = self._apply_ignore_rules(violations)

        # Sort by line number, then by severity (BLOCK first, then WARN)
        sorted_violations = sorted(
            filtered_violations, key=lambda v: (v.line or 0, v.severity.value, v.rule_id)
        )

        return sorted_violations

    def _apply_ignore_rules(self, violations: List[Finding]) -> List[Finding]:
        """Apply ignore rules and exceptions to violations."""
        # For now, return all violations
        # TODO: Implement ignore patterns from config
        return violations

    def _calculate_health_score(self, violations: List[Finding]) -> float:
        """
        Calculate file-level health score between 0-100.

        Score calculation:
        - Start with 100
        - Subtract 10 for each ERROR
        - Subtract 5 for each WARN
        - Minimum score is 0
        """
        score = 100.0

        for violation in violations:
            if violation.severity == Severity.BLOCK:
                score -= 10.0
            elif violation.severity == Severity.WARN:
                score -= 5.0

        return max(0.0, score)


class SingleFileValidationWorkflow:
    """
    Workflow 1: Single File Validation

    User runs `vibelint validate file.py`
    """

    def __init__(self, config: Config):
        self.config = config
        self.validator = SingleFileValidator(config)

    def execute(
        self, file_path: Path, output_format: str = "natural"
    ) -> SingleFileValidationResult:
        """
        Execute single file validation workflow.

        Args:
            file_path: Path to Python file to validate
            output_format: Output format (natural, json, etc.)

        Returns:
            SingleFileValidationResult with violations and health score
        """
        logger.info(f"Starting single file validation for: {file_path}")

        result = self.validator.validate_file(file_path)

        if result.success:
            logger.info(f"Validation completed. Health score: {result.health_score:.1f}")
        else:
            logger.error(f"Validation failed: {result.error_message}")

        return result

    def format_output(
        self, result: SingleFileValidationResult, output_format: str = "natural"
    ) -> str:
        """Format validation result for output."""
        if output_format == "json":
            return self._format_json(result)
        elif output_format == "natural":
            return self._format_natural(result)
        else:
            return self._format_natural(result)

    def _format_json(self, result: SingleFileValidationResult) -> str:
        """Format result as JSON."""
        import json

        output = {
            "file_path": str(result.file_path) if result.file_path else None,
            "success": result.success,
            "health_score": result.health_score,
            "execution_time_ms": result.execution_time_ms,
            "violations": [],
        }

        if result.error_message:
            output["error"] = result.error_message

        for violation in result.violations:
            output["violations"].append(
                {
                    "rule_id": violation.rule_id,
                    "severity": violation.severity.name,
                    "message": violation.message,
                    "line_number": violation.line,
                    "column_number": violation.column,
                    "fix_suggestion": violation.suggestion,
                }
            )

        return json.dumps(output, indent=2)

    def _format_natural(self, result: SingleFileValidationResult) -> str:
        """Format result in natural language."""
        if not result.success:
            return f"❌ Validation failed: {result.error_message}"

        output_lines = []

        # Header
        file_name = result.file_path.name if result.file_path else "Unknown"
        output_lines.append(f"📄 File: {file_name}")
        output_lines.append(f"💯 Health Score: {result.health_score:.1f}/100")
        output_lines.append(f"⏱️  Execution Time: {result.execution_time_ms:.1f}ms")

        if not result.violations:
            output_lines.append("\n✅ No violations found!")
            return "\n".join(output_lines)

        # Group violations by severity
        errors = [v for v in result.violations if v.severity == Severity.BLOCK]
        warnings = [v for v in result.violations if v.severity == Severity.WARN]

        if errors:
            output_lines.append(f"\n🚨 Errors ({len(errors)}):")
            for error in errors:
                line_info = f":{error.line}" if error.line else ""
                output_lines.append(f"  • [{error.rule_id}] {error.message} {line_info}")
                if error.suggestion:
                    output_lines.append(f"    💡 Fix: {error.suggestion}")

        if warnings:
            output_lines.append(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                line_info = f":{warning.line}" if warning.line else ""
                output_lines.append(f"  • [{warning.rule_id}] {warning.message} {line_info}")
                if warning.suggestion:
                    output_lines.append(f"    💡 Fix: {warning.suggestion}")

        return "\n".join(output_lines)


def run_single_file_validation(
    file_path: Path, config: Config, output_format: str = "natural"
) -> SingleFileValidationResult:
    """
    Convenience function to run single file validation.

    This is the main entry point for Workflow 1.
    """
    workflow = SingleFileValidationWorkflow(config)
    return workflow.execute(file_path, output_format)
