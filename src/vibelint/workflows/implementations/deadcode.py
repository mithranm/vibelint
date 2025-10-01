"""
Runtime dead code detection via coverage analysis.

Coverage-based approach:
1. Discover all entry points (CLI, tests, __main__.py)
2. Run each with --help/--version/import-only to maximize coverage
3. Any file with 0% coverage = dead code or unreachable edge case

Deterministic analysis with optional LLM for arg generation.

vibelint/src/vibelint/workflows/implementations/deadcode.py
"""

import ast
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

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
class EntryPoint:
    """Discovered entry point."""

    path: Path
    name: str
    type: str  # "main", "console_script", "click_cli", "test"


@dataclass
class CoverageResult:
    """Coverage analysis result."""

    total_files: int
    covered_files: int
    dead_files: List[Path]
    entry_points_traced: int
    overall_coverage_pct: float


# ============================================================================
# Entry Point Discovery
# ============================================================================


def discover_entry_points(project_root: Path) -> List[EntryPoint]:
    """Discover all entry points in project."""
    entry_points = []

    # 1. __main__.py modules
    for main_file in project_root.rglob("__main__.py"):
        if not any(p.startswith(".") for p in main_file.parts):
            entry_points.append(
                EntryPoint(path=main_file, name=main_file.parent.name, type="main")
            )

    # 2. Click CLI commands (simple AST check)
    src_dir = project_root / "src"
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            if _has_click_decorators(py_file):
                entry_points.append(
                    EntryPoint(path=py_file, name=py_file.stem, type="click_cli")
                )

    # 3. Test files
    for test_dir in ["tests", "test"]:
        test_path = project_root / test_dir
        if test_path.exists():
            for test_file in test_path.rglob("test_*.py"):
                entry_points.append(
                    EntryPoint(path=test_file, name=test_file.stem, type="test")
                )

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
# Coverage Tracing
# ============================================================================


def trace_entry_point_coverage(
    entry_point: EntryPoint, project_root: Path
) -> Set[str]:
    """Trace coverage for a single entry point using multiple strategies."""
    all_imports = set()

    strategies = [
        (["--help"], 5),
        (["--version"], 5),
        ([], 10),  # import-only
    ]

    for args, timeout in strategies:
        try:
            imports = _run_coverage_trace(entry_point.path, args, timeout, project_root)
            all_imports.update(imports)
            logger.debug(
                f"{entry_point.name} with {args or 'import-only'}: {len(imports)} modules"
            )
        except Exception as e:
            logger.debug(f"Strategy {args} failed: {e}")
            continue

    return all_imports


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

            # Parse coverage data
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
# Main Workflow
# ============================================================================


class DeadcodeWorkflow(BaseWorkflow):
    """Coverage-based dead code detection."""

    workflow_id = "deadcode"
    name = "Dead Code Detection"
    description = "Detect dead code via runtime coverage analysis"
    category = "analysis"
    version = "1.0.0"
    tags = {"deadcode", "coverage", "runtime", "deterministic"}

    def __init__(self, config: Optional[WorkflowConfig] = None):
        super().__init__(config=config)
        self.llm_manager = None

    @property
    def default_priority(self) -> WorkflowPriority:
        return WorkflowPriority.MEDIUM

    def get_required_inputs(self) -> Set[str]:
        return {"project_root"}

    def get_produced_outputs(self) -> Set[str]:
        return {"dead_files", "coverage_result"}

    def supports_parallel_execution(self) -> bool:
        return False  # Coverage tracing is sequential

    def execute(self, project_root: Path, **kwargs) -> WorkflowResult:
        """Execute dead code detection.

        Returns:
            WorkflowResult with findings containing dead file paths and metrics.
        """
        start_time = time.time()
        metrics = WorkflowMetrics(start_time=start_time)

        try:
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

            # Phase 2: Trace coverage from all entry points
            all_covered_modules: Set[str] = set()

            for ep in entry_points:
                logger.info(f"Tracing {ep.type}: {ep.name}")
                covered = trace_entry_point_coverage(ep, project_root)
                all_covered_modules.update(covered)

            # Phase 3: Find all Python files in project
            all_py_files = _discover_all_python_files(project_root)
            logger.info(f"Total Python files: {len(all_py_files)}")

            # Phase 4: Identify dead files (zero coverage)
            covered_paths = {
                _module_to_path(m, project_root) for m in all_covered_modules
            }

            dead_files = [f for f in all_py_files if f not in covered_paths]

            # Phase 5: Build result
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
                    "message": "File has zero coverage from all entry points",
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
