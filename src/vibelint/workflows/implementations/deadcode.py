"""
LLM-powered runtime dead code detection.

Hybrid approach:
1. Discover all entry points (CLI, tests, __main__.py)
2. Extract CLI command signatures via AST
3. Use LLM to generate realistic test arguments for each command
4. Run coverage with actual execution (not just --help)
5. Use LLM to analyze suspected dead files for false positives
6. Report genuinely unused code

vibelint/src/vibelint/workflows/implementations/deadcode.py
"""

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from vibelint.llm.manager import LLMRole
from vibelint.workflows.core.base import (
    BaseWorkflow,
    WorkflowConfig,
    WorkflowMetrics,
    WorkflowPriority,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

__all__ = ["DeadcodeWorkflow"]


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CLICommand:
    """Extracted CLI command signature."""

    name: str
    path: Path
    function_name: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    help_text: Optional[str] = None


@dataclass
class GeneratedArguments:
    """LLM-generated test arguments for a command."""

    command: str
    args: List[str]
    description: str


@dataclass
class CoverageResult:
    """Coverage analysis result."""

    total_files: int
    covered_files: int
    dead_files: List[Path]
    entry_points_traced: int
    overall_coverage_pct: float
    llm_analysis: Optional[str] = None


# ============================================================================
# Entry Point Discovery
# ============================================================================


def discover_entry_points(project_root: Path) -> List[Path]:
    """Discover all entry points in project."""
    entry_points = []

    # 1. __main__.py modules
    for main_file in project_root.rglob("__main__.py"):
        if not any(p.startswith(".") for p in main_file.parts):
            entry_points.append(main_file)

    # 2. Click CLI files
    src_dir = project_root / "src"
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            if _has_click_decorators(py_file):
                entry_points.append(py_file)

    # 3. Test files
    for test_dir in ["tests", "test"]:
        test_path = project_root / test_dir
        if test_path.exists():
            for test_file in test_path.rglob("test_*.py"):
                entry_points.append(test_file)

    logger.info(f"Discovered {len(entry_points)} entry points")
    return entry_points


def _has_click_decorators(file_path: Path) -> bool:
    """Check if file has Click decorators."""
    try:
        tree = ast.parse(file_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and isinstance(
                        deco.func, ast.Attribute
                    ):
                        if (
                            isinstance(deco.func.value, ast.Name)
                            and deco.func.value.id == "click"
                        ):
                            return True
        return False
    except (SyntaxError, Exception):
        return False


# ============================================================================
# CLI Command Extraction
# ============================================================================


def extract_cli_commands(cli_files: List[Path]) -> List[CLICommand]:
    """Extract CLI command signatures from files."""
    commands = []

    for file_path in cli_files:
        try:
            tree = ast.parse(file_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for @click.command or @click.group decorators
                    has_click_command = False
                    for deco in node.decorator_list:
                        if isinstance(deco, ast.Call):
                            if (
                                isinstance(deco.func, ast.Attribute)
                                and isinstance(deco.func.value, ast.Name)
                                and deco.func.value.id == "click"
                                and deco.func.attr in ("command", "group")
                            ):
                                has_click_command = True

                    if has_click_command:
                        # Extract parameters from decorators
                        params = []
                        help_text = None

                        for deco in node.decorator_list:
                            if isinstance(deco, ast.Call) and isinstance(
                                deco.func, ast.Attribute
                            ):
                                if (
                                    isinstance(deco.func.value, ast.Name)
                                    and deco.func.value.id == "click"
                                ):
                                    # Extract parameter info
                                    if deco.func.attr in (
                                        "option",
                                        "argument",
                                        "command",
                                    ):
                                        param_info = {}
                                        if deco.args:
                                            param_info["name"] = (
                                                deco.args[0].value
                                                if isinstance(
                                                    deco.args[0], ast.Constant
                                                )
                                                else str(deco.args[0])
                                            )

                                        # Extract help text
                                        for keyword in deco.keywords:
                                            if keyword.arg == "help" and isinstance(
                                                keyword.value, ast.Constant
                                            ):
                                                help_text = keyword.value.value

                                        if param_info:
                                            params.append(param_info)

                        commands.append(
                            CLICommand(
                                name=node.name,
                                path=file_path,
                                function_name=node.name,
                                parameters=params,
                                help_text=help_text,
                            )
                        )

        except Exception as e:
            logger.debug(f"Failed to extract commands from {file_path}: {e}")

    logger.info(f"Extracted {len(commands)} CLI commands")
    return commands


# ============================================================================
# Project Context Helpers
# ============================================================================


def _get_file_tree(project_root: Path, max_depth: int = 3) -> str:
    """Generate file tree for project context."""
    src_dir = project_root / "src"
    if not src_dir.exists():
        return "No src/ directory found"

    lines = ["src/"]

    def walk_dir(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return

        try:
            items = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for i, item in enumerate(items):
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue

                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "

                lines.append(f"{prefix}{current_prefix}{item.name}")

                if item.is_dir():
                    walk_dir(item, prefix + next_prefix, depth + 1)

        except PermissionError:
            pass

    walk_dir(src_dir)
    return "\n".join(lines[:200])  # Limit to 200 lines


# ============================================================================
# LLM-Powered Argument Generation
# ============================================================================


def generate_test_arguments(
    commands: List[CLICommand], llm_manager, project_root: Path
) -> List[GeneratedArguments]:
    """Use LLM to generate realistic test arguments for commands."""
    if not llm_manager or not llm_manager.is_llm_available(LLMRole.FAST):
        logger.warning("No LLM available - using minimal test coverage")
        return []

    # Get project file tree for context
    file_tree = _get_file_tree(project_root)

    generated_args = []

    for cmd in commands:
        prompt = f"""Generate 3 realistic test argument sets for this CLI command.

PROJECT CONTEXT:
{file_tree}

COMMAND INFO:
Command: {cmd.name}
Help: {cmd.help_text or 'No help text'}
Parameters: {json.dumps(cmd.parameters, indent=2)}

Generate test arguments that make sense for this project's structure.
For file paths, use actual files from the project tree above.

Return JSON array with format:
[
  {{"args": ["--flag", "value"], "description": "Test basic usage"}},
  {{"args": ["--other"], "description": "Test alternative path"}}
]

Focus on arguments that will exercise different code paths.
"""

        try:
            response = llm_manager.process_request(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            if response.success and response.content:
                arg_sets = json.loads(response.content)
                for arg_set in arg_sets[:3]:  # Limit to 3 per command
                    generated_args.append(
                        GeneratedArguments(
                            command=cmd.name,
                            args=arg_set["args"],
                            description=arg_set["description"],
                        )
                    )

        except Exception as e:
            logger.debug(f"Failed to generate args for {cmd.name}: {e}")

    logger.info(f"Generated {len(generated_args)} argument sets")
    return generated_args


# ============================================================================
# Coverage Tracing with Realistic Arguments
# ============================================================================


def trace_coverage_with_args(
    entry_point: Path,
    project_root: Path,
    test_args: List[List[str]],
) -> Set[str]:
    """Trace coverage for entry point with realistic arguments."""
    all_modules = set()

    # Always try import-only first
    modules = _run_coverage_trace(entry_point, [], 10, project_root)
    all_modules.update(modules)

    # Try each generated argument set
    for args in test_args:
        try:
            modules = _run_coverage_trace(entry_point, args, 30, project_root)
            all_modules.update(modules)
            logger.debug(
                f"{entry_point.name} with {args}: {len(modules)} modules covered"
            )
        except Exception as e:
            logger.debug(f"Args {args} failed: {e}")
            continue

    return all_modules


def _run_coverage_trace(
    script_path: Path, args: List[str], timeout: int, project_root: Path
) -> Set[str]:
    """Run coverage.py trace on a script."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coverage_file = Path(tmpdir) / ".coverage"

        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--data-file={coverage_file}",
            "--source",
            str(project_root / "src"),
            str(script_path),
        ] + args

        try:
            subprocess.run(
                cmd, cwd=project_root, timeout=timeout, capture_output=True, check=False
            )

            return _parse_coverage_modules(coverage_file, project_root)

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout after {timeout}s for {script_path.name}")
            return set()


def _parse_coverage_modules(coverage_file: Path, project_root: Path) -> Set[str]:
    """Extract covered modules from coverage data."""
    if not coverage_file.exists():
        return set()

    try:
        from coverage import Coverage

        cov = Coverage(data_file=str(coverage_file))
        cov.load()

        modules = set()
        for file_path in cov.get_data().measured_files():
            if str(project_root) in file_path:
                try:
                    rel_path = Path(file_path).relative_to(project_root / "src")
                    module = str(rel_path.with_suffix("")).replace("/", ".")
                    modules.add(module)
                except ValueError:
                    continue

        return modules

    except Exception as e:
        logger.warning(f"Failed to parse coverage: {e}")
        return set()


# ============================================================================
# LLM Analysis of Suspected Dead Files
# ============================================================================


def analyze_dead_files(
    suspected_dead: List[Path], project_root: Path, llm_manager
) -> Dict[Path, str]:
    """Use LLM to analyze suspected dead files for false positives."""
    if not llm_manager or not llm_manager.is_llm_available(LLMRole.FAST):
        logger.warning("No LLM available - skipping dead file analysis")
        return {f: "No LLM available for analysis" for f in suspected_dead}

    analysis = {}

    for file_path in suspected_dead:
        try:
            content = file_path.read_text()
            rel_path = file_path.relative_to(project_root)

            prompt = f"""Analyze if this Python file is genuinely unused or has legitimate reasons for zero coverage:

File: {rel_path}

```python
{content[:2000]}  # First 2000 chars
```

Consider:
1. Is this a base class/interface/protocol that's subclassed elsewhere?
2. Is this a utility module imported dynamically?
3. Is this a plugin/extension point?
4. Is this genuinely dead code that should be deleted?

Return JSON: {{"is_dead": true/false, "reason": "explanation"}}
"""

            response = llm_manager.process_request(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            if response.success and response.content:
                result = json.loads(response.content)
                analysis[file_path] = result.get(
                    "reason", "No analysis available"
                )
            else:
                analysis[file_path] = "LLM analysis failed"

        except Exception as e:
            logger.debug(f"Failed to analyze {file_path}: {e}")
            analysis[file_path] = f"Analysis error: {e}"

    return analysis


# ============================================================================
# Main Workflow
# ============================================================================


class DeadcodeWorkflow(BaseWorkflow):
    """LLM-powered dead code detection via runtime coverage."""

    workflow_id = "deadcode"
    name = "Dead Code Detection"
    description = "LLM-powered dead code detection via runtime coverage"
    category = "analysis"
    version = "2.0.0"
    tags = {"deadcode", "coverage", "runtime", "llm", "intelligent"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config=config)
        self.llm_manager = None

    @property
    def default_priority(self) -> WorkflowPriority:
        return WorkflowPriority.MEDIUM

    def get_required_inputs(self) -> Set[str]:
        return {"project_root"}

    def get_produced_outputs(self) -> Set[str]:
        return {"dead_files", "coverage_result", "llm_analysis"}

    def supports_parallel_execution(self) -> bool:
        return False

    def execute(self, project_root: Path, **kwargs) -> WorkflowResult:
        """Execute LLM-powered dead code detection."""
        start_time = time.time()
        metrics = WorkflowMetrics(start_time=start_time)

        try:
            # Initialize LLM manager
            from vibelint.llm.manager import LLMManager

            self.llm_manager = LLMManager()

            # Phase 1: Discover entry points
            entry_points = discover_entry_points(project_root)

            if not entry_points:
                metrics.end_time = time.time()
                metrics.finalize()
                return WorkflowResult(
                    workflow_id=self.workflow_id,
                    status=WorkflowStatus.FAILED,
                    metrics=metrics,
                    error_message="No entry points discovered",
                )

            # Phase 2: Extract CLI commands
            cli_files = [f for f in entry_points if _has_click_decorators(f)]
            commands = extract_cli_commands(cli_files)

            # Phase 3: Generate realistic test arguments
            generated_args = generate_test_arguments(
                commands, self.llm_manager, project_root
            )

            # Group args by command
            args_by_file = {}
            for gen_arg in generated_args:
                for cmd in commands:
                    if cmd.name == gen_arg.command:
                        if cmd.path not in args_by_file:
                            args_by_file[cmd.path] = []
                        args_by_file[cmd.path].append(gen_arg.args)

            # Phase 4: Trace coverage with realistic arguments
            all_covered_modules: Set[str] = set()

            for ep in entry_points:
                logger.info(f"Tracing {ep.name}")
                test_args = args_by_file.get(ep, [[]])  # Empty args if none generated
                covered = trace_coverage_with_args(ep, project_root, test_args)
                all_covered_modules.update(covered)

            # Phase 5: Find all Python files
            all_py_files = _discover_all_python_files(project_root)
            logger.info(f"Total Python files: {len(all_py_files)}")

            # Phase 6: Identify suspected dead files
            covered_paths = {
                _module_to_path(m, project_root) for m in all_covered_modules
            }
            suspected_dead = [f for f in all_py_files if f not in covered_paths]

            # Phase 7: LLM analysis of suspected dead files
            llm_analysis = analyze_dead_files(
                suspected_dead, project_root, self.llm_manager
            )

            # Filter to genuinely dead files
            dead_files = [
                f
                for f in suspected_dead
                if "is_dead\": true" in llm_analysis.get(f, "")
                or "No LLM available" in llm_analysis.get(f, "")
            ]

            # Phase 8: Build result
            coverage_pct = (
                ((len(all_py_files) - len(dead_files)) / len(all_py_files)) * 100
                if all_py_files
                else 0.0
            )

            metrics.end_time = time.time()
            metrics.files_processed = len(entry_points)
            metrics.findings_generated = len(dead_files)
            metrics.coverage_percentage = coverage_pct
            metrics.finalize()

            findings = [
                {
                    "type": "dead_file",
                    "path": str(f.relative_to(project_root)),
                    "message": llm_analysis.get(
                        f, "File has zero coverage and no legitimate reason"
                    ),
                    "severity": "medium",
                }
                for f in dead_files
            ]

            logger.info(
                f"Dead code analysis complete: {len(dead_files)} dead files, "
                f"{coverage_pct:.1f}% coverage"
            )

            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.COMPLETED,
                metrics=metrics,
                findings=findings,
                artifacts={
                    "dead_files": [str(f) for f in dead_files],
                    "coverage_percentage": coverage_pct,
                    "entry_points_traced": len(entry_points),
                    "llm_analysis": {
                        str(k.relative_to(project_root)): v
                        for k, v in llm_analysis.items()
                    },
                },
            )

        except Exception as e:
            logger.exception("Dead code analysis failed")
            metrics.end_time = time.time()
            metrics.finalize()
            return WorkflowResult(
                workflow_id=self.workflow_id,
                status=WorkflowStatus.FAILED,
                metrics=metrics,
                error_message=str(e),
            )


def _discover_all_python_files(project_root: Path) -> List[Path]:
    """Find all Python files in src/."""
    src_dir = project_root / "src"
    if not src_dir.exists():
        return []

    files = []
    for py_file in src_dir.rglob("*.py"):
        if not any(p.startswith(".") or p == "__pycache__" for p in py_file.parts):
            files.append(py_file)

    return files


def _module_to_path(module_name: str, project_root: Path) -> Path:
    """Convert module name to file path."""
    return (project_root / "src" / module_name.replace(".", "/")).with_suffix(".py")
