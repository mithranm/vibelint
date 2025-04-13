# Snapshot

## Filesystem Tree

```
vibelint/
├── src/
│   └── vibelint/
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── docstring.py
│       │   ├── encoding.py
│       │   ├── exports.py
│       │   └── shebang.py
│       ├── __init__.py
│       ├── ascii.py
│       ├── cli.py
│       ├── config.py
│       ├── discovery.py
│       ├── error_codes.py
│       ├── lint.py
│       ├── namespace.py
│       ├── report.py
│       ├── results.py
│       ├── snapshot.py
│       └── utils.py
├── tests/
│   └── test_cli.py
└── pyproject.toml
```

## File Contents

Files are ordered alphabetically by path.

### File: pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "vibelint"
version = "0.1.1"
description = "Suite of tools to enhance the vibe coding process."
authors = [
  { name = "Mithran Mohanraj", email = "mithran.mohanraj@gmail.com" }
]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
]
dependencies = [
    "click>=8.1.0",
    "tomli>=2.0.0; python_version < '3.11'",
    "tomli-w",
    "colorama>=0.4.0",
    "rich>=12.0.0",
    "libcst"
]

[project.scripts]
vibelint = "vibelint.cli:main"

[project.urls]
"Homepage" = "https://github.com/mithranm/vibelint"
"Bug Tracker" = "https://github.com/mithranm/vibelint/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["vibelint*"]

[tool.vibelint]
include_globs = [
    "src/**/*.py",
    "tests/**/*.py",
    "pyproject.toml",
    "VIBECHECKER.txt"
]
exclude_globs = [
    "tests/fixtures/*"
]
peek_globs = ["coverage.xml"]
allowed_shebangs = ["#!/usr/bin/env python3"]
large_dir_threshold = 500
error_on_missing_all_in_init = false

[tool.black]
target-version = ["py310", "py311", "py312"]
line-length=100

[tool.setuptools.package-data]
vibelint = ["VIBECHECKER.txt"]
```

---
### File: src/vibelint/__init__.py

```python
"""
vibelint package initialization module.

vibelint/__init__.py
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vibelint")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
]
```

---
### File: src/vibelint/ascii.py

```python
"""
ASCII Art Scaling and Loading Utilities
This module provides functions to scale ASCII art to fit within the terminal's
dimensions while preserving the original aspect ratio. It also includes a function
to load ASCII art from a file.

src/vibelint/ascii.py
"""

import shutil


def _get_terminal_size():
    """
    Returns the terminal size as a tuple (width, height) of characters.
    Falls back to (80, 24) if the dimensions cannot be determined.

    src/vibelint/ascii.py
    """
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines
    except Exception:
        return 80, 24


def scale_ascii_art_by_height(ascii_art: str, target_height: int) -> str:
    """
    Scales the ASCII art to have a specified target height (in characters)
    while preserving the original aspect ratio. The target width is
    automatically computed based on the scaling factor.

    src/vibelint/ascii.py
    """
    # Split into lines and remove any fully blank lines.
    lines = [line for line in ascii_art.splitlines() if line.strip()]
    if not lines:
        return ""

    orig_height = len(lines)
    orig_width = max(len(line) for line in lines)

    # Pad all lines to the same length (for a rectangular grid)
    normalized_lines = [line.ljust(orig_width) for line in lines]

    # Compute the vertical scale factor and derive the target width.
    scale_factor = target_height / orig_height
    target_width = max(1, int(orig_width * scale_factor))

    # Calculate step sizes for sampling
    row_step = orig_height / target_height
    col_step = orig_width / target_width if target_width > 0 else 1

    result_lines = []
    for r in range(target_height):
        orig_r = min(int(r * row_step), orig_height - 1)
        new_line = []
        for c in range(target_width):
            orig_c = min(int(c * col_step), orig_width - 1)
            new_line.append(normalized_lines[orig_r][orig_c])
        result_lines.append("".join(new_line))

    return "\n".join(result_lines)


def scale_to_terminal_by_height(ascii_art: str) -> str:
    """
    Scales the provided ASCII art to fit based on the terminal's available height.
    The width is computed automatically to maintain the art's original aspect ratio.

    src/vibelint/ascii.py
    """
    _, term_height = _get_terminal_size()
    # Optionally, leave a margin (here, using 90% of available height)
    target_height = max(1, int(term_height * 0.9))
    return scale_ascii_art_by_height(ascii_art, target_height)


def load_ascii_from_file(filepath: str) -> str:
    """
    Reads an ASCII art file from disk and returns its content as a string.
    Using a file is generally wise because it separates your art data from your code,
    making it easier to update without modifying the code.

    src/vibelint/ascii.py
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading {filepath}: {e}")


# Public API
__all__ = ["scale_ascii_art_by_height", "scale_to_terminal_by_height", "load_ascii_from_file"]
```

---
### File: src/vibelint/cli.py

```python
#!/usr/bin/env python3
"""
Command-line interface for vibelint.

Conducts vibe checks on your codebase.

vibelint/cli.py
"""

import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

# Add this import
if sys.version_info >= (3, 7):
    import importlib.resources as pkg_resources
else:
    # Fallback for Python < 3.7
    import importlib_resources as pkg_resources

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Import necessary functions for ASCII art
from .ascii import scale_to_terminal_by_height
from .config import Config, load_config
from .lint import LintRunner
from .namespace import (
    NamespaceCollision,
    build_namespace_tree,
    detect_global_definition_collisions,
    detect_hard_collisions,
    detect_local_export_collisions,
)
from .report import write_report_content
from .results import CheckResult, CommandResult, NamespaceResult, SnapshotResult
from .snapshot import create_snapshot
from .utils import find_project_root, get_relative_path

ValidationIssue = Tuple[str, str]


class VibelintContext:
    """
    Context object to store command results and shared state.

    vibelint/cli.py
    """

    def __init__(self):
        """
        Initializes VibelintContext.

        vibelint/cli.py
        """
        self.command_result: Optional[CommandResult] = None
        self.lint_runner: Optional[LintRunner] = None
        self.project_root: Optional[Path] = None


__all__ = ["snapshot", "check", "cli", "namespace", "main", "VibelintContext"]


console = Console()
logger_cli = logging.getLogger("vibelint")

# --- Helper messages ---
VIBE_CHECK_PASS_MESSAGES = [
    "Immaculate vibes. 💆",
    "Vibes confirmed ✅",
    "Vibe on brother. 🧘",
]

VIBE_CHECK_FAIL_MESSAGES = [
    "Vibe Check Failed. 📉",
    "Vibe Check Failed. 💥",
]


# (Keep _present_check_results, _present_namespace_results,
# _present_snapshot_results, and _display_collisions as they were in the
# last correct version - no changes needed there based on these errors)
def _present_check_results(result: CheckResult, runner: LintRunner, console: Console):
    """
    Presents the results of the 'check' command (the Vibe Check™).

    vibelint/cli.py
    """
    runner._print_summary()
    files_with_issues = sorted(
        [lr for lr in runner.results if lr.has_issues], key=lambda r: r.file_path
    )

    if files_with_issues:
        console.print("\n[bold yellow]🤔 Vibe Check:[/bold yellow]")
        for lr in files_with_issues:
            try:
                # Ensure config.project_root exists before using get_relative_path
                rel_path_str = (
                    str(get_relative_path(lr.file_path, runner.config.project_root))
                    if runner.config.project_root
                    else str(lr.file_path)  # Fallback if root is somehow None
                )
                console.print(f"\n[bold cyan]{rel_path_str}:[/bold cyan]")
            except ValueError:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([yellow]Outside project?[/yellow])"
                )
            except Exception as e:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([red]Error getting relative path: {e}[/red])"
                )

            for code, error_msg in lr.errors:
                console.print(f"  [red]✗[{code}] {error_msg}[/red]")
            for code, warning_msg in lr.warnings:
                console.print(f"  [yellow]▲[{code}] {warning_msg}[/yellow]")

    has_collisions = bool(
        result.hard_collisions or result.global_soft_collisions or result.local_soft_collisions
    )
    if has_collisions:
        console.print()
        _display_collisions(
            result.hard_collisions,
            result.global_soft_collisions,
            result.local_soft_collisions,
            console,
        )
    else:
        logger_cli.debug("No namespace collisions detected.")

    if result.report_path:
        console.print()
        if result.report_generated:
            console.print(
                f"[green]✓ Detailed Vibe Report generated at {result.report_path}[/green]"
            )
        elif result.report_error:
            console.print(
                f"\n[bold red]Error generating vibe report:[/bold red] {result.report_error}"
            )
        else:
            console.print(f"[yellow]Vibe report status unknown for {result.report_path}[/yellow]")

    console.print()
    has_warnings = (
        any(res.warnings for res in runner.results)
        or result.global_soft_collisions
        or result.local_soft_collisions
    )
    files_with_major_failures = sum(1 for r in runner.results if r.errors) + len(
        result.hard_collisions
    )

    if result.exit_code != 0:
        fail_message = random.choice(VIBE_CHECK_FAIL_MESSAGES)
        fail_reason = (
            f"{files_with_major_failures} major failure(s)"
            if files_with_major_failures > 0
            else f"exit code {result.exit_code}"
        )
        console.print(f"[bold red]{fail_message} ({fail_reason}).[/bold red]")
    elif not runner.results:
        console.print("[bold blue]Vibe Check: Skipped. No Python files found to check.[/bold blue]")
    else:  # Passed or passed with warnings
        pass_message = random.choice(VIBE_CHECK_PASS_MESSAGES)
        if has_warnings:
            console.print(
                f"[bold yellow]{pass_message} (But check the minor issues noted above).[/bold yellow]"
            )
        else:
            console.print(f"[bold green]{pass_message}[/bold green]")


def _present_namespace_results(result: NamespaceResult, console: Console):
    """
    Presents the results of the 'namespace' command.

    vibelint/cli.py
    """
    if not result.success and result.error_message:
        console.print(f"[bold red]Error building namespace tree:[/bold red] {result.error_message}")
        # Don't proceed if the core operation failed

    if result.intra_file_collisions:
        console.print(
            "\n[bold yellow]🤔 Intra-file Collisions Found (Duplicate members within one file):[/bold yellow]"
        )
        ctx = click.get_current_context(silent=True)
        project_root = (
            ctx.obj.project_root if ctx and hasattr(ctx.obj, "project_root") else Path(".")
        )
        for c in sorted(result.intra_file_collisions, key=lambda x: (str(x.paths[0]), x.name)):
            try:
                rel_path_str = str(get_relative_path(c.paths[0], project_root))
            except ValueError:
                rel_path_str = str(c.paths[0])

            loc1 = (
                f"{rel_path_str}:{c.linenos[0]}"
                if c.linenos and c.linenos[0] is not None
                else rel_path_str
            )
            line1 = c.linenos[0] if c.linenos and c.linenos[0] is not None else "?"
            line2 = c.linenos[1] if len(c.linenos) > 1 and c.linenos[1] is not None else "?"
            console.print(
                f"- '{c.name}': Duplicate definition/import vibe in {loc1} (lines ~{line1} and ~{line2})"
            )

    # Handle output file status regardless of collisions
    if result.output_path:
        if result.intra_file_collisions:
            console.print()  # Add space
        if result.output_saved:
            console.print(f"\n[green]✓ Namespace tree saved to {result.output_path}[/green]")
        elif result.output_error:
            console.print(
                f"[bold red]Error saving namespace tree:[/bold red] {result.output_error}"
            )
        # No need for 'unknown' status here unless saving was attempted and didn't error but failed

    # Only print tree to console if no output file was specified *and* the tree was built
    elif result.root_node and result.success:
        if result.intra_file_collisions:
            console.print()  # Add space
        console.print("\n[bold blue]👀 Namespace Structure Visualization:[/bold blue]")
        console.print(str(result.root_node))


def _present_snapshot_results(result: SnapshotResult, console: Console):
    """
    Presents the results of the 'snapshot' command. (Keep factual)

    vibelint/cli.py
    """
    if result.success and result.output_path:
        console.print(f"[green]✓ Codebase snapshot created at {result.output_path}[/green]")
    elif not result.success and result.error_message:
        console.print(f"[bold red]Error creating snapshot:[/bold red] {result.error_message}")


def _display_collisions(
    hard_coll: List[NamespaceCollision],
    global_soft_coll: List[NamespaceCollision],
    local_soft_coll: List[NamespaceCollision],
    console: Console,
) -> int:
    """
    Displays collision results in tables and returns an exit code indicating if hard collisions were found.

    vibelint/cli.py
    """
    exit_code = 1 if hard_coll else 0
    total_collisions = len(hard_coll) + len(global_soft_coll) + len(local_soft_coll)

    if total_collisions == 0:
        return 0

    ctx = click.get_current_context(silent=True)
    project_root = ctx.obj.project_root if ctx and hasattr(ctx.obj, "project_root") else Path(".")

    def get_rel_path_display(p: Path) -> str:
        """
        Get a relative path for display purposes, resolving it first.
        This is useful for consistent output in tables.

        src/vibelint/cli.py
        """
        try:
            # Resolve paths before getting relative path for consistency
            return str(get_relative_path(p.resolve(), project_root.resolve()))
        except ValueError:
            return str(p.resolve())  # Fallback to absolute resolved path

    table_title = "Namespace Collision Summary"
    table = Table(title=table_title)
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")

    hard_label = "Hard Collisions (🚨)"
    global_soft_label = "Global Soft Collision (Defs)"
    local_soft_label = "Local Soft Collision (__all__)"

    table.add_row(hard_label, str(len(hard_coll)), style="red" if hard_coll else "")
    table.add_row(
        global_soft_label, str(len(global_soft_coll)), style="yellow" if global_soft_coll else ""
    )
    table.add_row(
        local_soft_label, str(len(local_soft_coll)), style="yellow" if local_soft_coll else ""
    )
    console.print(table)

    if hard_coll:
        hard_header = "[bold red]🚨 Hard Collision Details:[/bold red]"
        console.print(f"\n{hard_header}")
        console.print(
            "These can break imports or indicate unexpected duplicates (Bad Vibes! Fix these!):"
        )
        grouped_hard = defaultdict(list)
        for c in hard_coll:
            grouped_hard[c.name].append(c)

        for name, collisions in sorted(grouped_hard.items()):
            locations = []
            for c in collisions:
                for i, p in enumerate(c.paths):
                    line_info = (
                        f":{c.linenos[i]}"
                        if c.linenos and i < len(c.linenos) and c.linenos[i] is not None
                        else ""
                    )
                    locations.append(f"{get_rel_path_display(p)}{line_info}")
            unique_locations = sorted(list(set(locations)))
            is_intra_file = (
                len(collisions) > 0
                and len(collisions[0].paths) > 1
                and all(
                    p.resolve() == collisions[0].paths[0].resolve() for p in collisions[0].paths[1:]
                )
            )

            if is_intra_file and len(unique_locations) == 1:
                # Intra-file duplicates might resolve to one location string after get_rel_path_display
                console.print(
                    f"- '{name}': Colliding imports (duplicate definition/import) in {unique_locations[0]}"
                )
            else:
                console.print(
                    f"- '{name}': Colliding imports (conflicting definitions/imports) in {', '.join(unique_locations)}"
                )

    if local_soft_coll:
        local_soft_header = "[bold yellow]🤔 Local Soft Collision (__all__) Details:[/bold yellow]"
        console.print(f"\n{local_soft_header}")
        console.print(
            "These names are exported via __all__ in multiple sibling modules (Confusing for `import *`):"
        )
        local_table = Table(show_header=True, header_style="bold yellow")
        local_table.add_column("Name", style="cyan", min_width=15)
        local_table.add_column("Exporting Files")
        grouped_local = defaultdict(list)
        for c in local_soft_coll:
            paths_to_add = c.definition_paths if c.definition_paths else c.paths
            grouped_local[c.name].extend(
                p for p in paths_to_add if p and p not in grouped_local[c.name]
            )

        for name, involved_paths in sorted(grouped_local.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            local_table.add_row(name, "\n".join(paths_str_list))
        console.print(local_table)

    if global_soft_coll:
        global_soft_header = (
            "[bold yellow]🤔 Global Namespace Collision (Definition) Details:[/bold yellow]"
        )
        console.print(f"\n{global_soft_header}")
        console.print(
            "These names are defined in multiple modules (May cause bad vibes for humans & LLMs):"
        )
        global_table = Table(show_header=True, header_style="bold yellow")
        global_table.add_column("Name", style="cyan", min_width=15)
        global_table.add_column("Defining Files")
        grouped_global = defaultdict(list)
        for c in global_soft_coll:
            paths_to_add = c.definition_paths if c.definition_paths else c.paths
            grouped_global[c.name].extend(
                p for p in paths_to_add if p and p not in grouped_global[c.name]
            )

        for name, involved_paths in sorted(grouped_global.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            global_table.add_row(name, "\n".join(paths_str_list))
        console.print(global_table)

    return exit_code


@click.group(invoke_without_command=True)
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug logging output.")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """
    vibelint - Check the vibe ✨, visualize namespaces, and snapshot Python codebases.

    Run commands from the root of your project (where pyproject.toml or .git is located).

    vibelint/cli.py
    """
    ctx.ensure_object(VibelintContext)
    vibelint_ctx: VibelintContext = ctx.obj

    project_root = find_project_root(Path("."))
    vibelint_ctx.project_root = project_root

    log_level = logging.DEBUG if debug else logging.INFO
    app_logger = logging.getLogger("vibelint")

    # Simplified Logging Setup: Configure once based on the initial debug flag.
    # Remove existing handlers to prevent duplicates if run multiple times (e.g., in tests)
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
        handler.close()  # Ensure resources are released

    app_logger.setLevel(log_level)
    app_logger.propagate = False
    rich_handler = RichHandler(
        console=console,
        show_path=debug,  # Use debug flag directly here
        markup=True,
        show_level=debug,  # Use debug flag directly here
        rich_tracebacks=True,
    )
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(formatter)
    app_logger.addHandler(rich_handler)

    # --- Logging Setup Done ---

    logger_cli.debug(f"vibelint started. Debug mode: {'ON' if debug else 'OFF'}")
    if project_root:
        logger_cli.debug(f"Identified project root: {project_root}")
    else:
        logger_cli.debug("Could not identify project root from current directory.")
    logger_cli.debug(f"Log level set to {logging.getLevelName(log_level)}")

    # --- Handle No Subcommand Case ---
    if ctx.invoked_subcommand is None:
        # Load VIBECHECKER.txt from package resources, not project root
        try:
            # Use importlib.resources to find the file within the installed package
            vibechecker_ref = pkg_resources.files("vibelint").joinpath("VIBECHECKER.txt")
            if vibechecker_ref.is_file():
                try:
                    # Read content directly using the reference
                    art = vibechecker_ref.read_text(encoding="utf-8")
                    scaled_art = scale_to_terminal_by_height(art)
                    console.print(scaled_art, style="bright_yellow", highlight=False)
                    console.print("\n✨ How's the vibe? ✨", justify="center")
                except Exception as e:
                    logger_cli.warning(
                        f"Could not load or display VIBECHECKER.txt from package data: {e}",
                        exc_info=debug,
                    )
            else:
                logger_cli.debug(
                    "VIBECHECKER.txt not found in vibelint package data, skipping display."
                )
        except Exception as e:
            logger_cli.warning(
                f"Error accessing package resources for VIBECHECKER.txt: {e}", exc_info=debug
            )

        console.print("\nRun [bold cyan]vibelint --help[/bold cyan] for available commands.")
        ctx.exit(0)

    # --- Subcommand Execution Check ---
    if vibelint_ctx.project_root is None:
        console.print("[bold red]Error:[/bold red] Could not find project root.")
        console.print("  vibelint needs to know where the project starts to check its vibes.")
        console.print("  Make sure you're in a directory with 'pyproject.toml' or '.git'.")
        ctx.exit(1)


@cli.command("check")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt for large directories.")
@click.option(
    "-o",
    "--output-report",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save a comprehensive Vibe Report (Markdown) to the specified file.",
)
@click.pass_context
def check(ctx: click.Context, yes: bool, output_report: Optional[Path]):
    """
    Run a Vibe Check: Lint rules and namespace collision detection.

    Fails if errors (like missing docstrings/`__all__`) or hard collisions are found.
    Warnings indicate potential vibe issues or areas for improvement.

    vibelint/cli.py
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing in check command"

    console.print("\n[bold magenta]Initiating Vibe Check...[/bold magenta]\n")
    logger_cli.debug(f"Running 'check' command (yes={yes}, report={output_report})")

    config: Config = load_config(project_root)
    if config.project_root is None:
        logger_cli.error("Project root lost after config load. Aborting Vibe Check.")
        ctx.exit(1)

    result_data = CheckResult()
    runner: Optional[LintRunner] = None

    try:
        # Use the non-None project_root for target_paths
        target_paths: List[Path] = [project_root]
        runner = LintRunner(config=config, skip_confirmation=yes)
        lint_exit_code = runner.run(target_paths)
        result_data.lint_results = runner.results
        vibelint_ctx.lint_runner = runner

        logger_cli.debug("Linting finished. Checking for namespace vibe collisions...")
        # Pass the non-None target_paths here too
        result_data.hard_collisions = detect_hard_collisions(target_paths, config)
        result_data.global_soft_collisions = detect_global_definition_collisions(
            target_paths, config
        )
        result_data.local_soft_collisions = detect_local_export_collisions(target_paths, config)

        collision_exit_code = 1 if result_data.hard_collisions else 0
        report_failed = False

        if output_report:
            report_path = output_report.resolve()
            result_data.report_path = report_path
            logger_cli.info(f"Generating detailed Vibe Report to {report_path}...")
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                # Pass the non-None target_paths here too
                root_node_for_report, _ = build_namespace_tree(target_paths, config)
                if root_node_for_report is None:
                    raise RuntimeError("Namespace tree building failed for report.")

                with open(report_path, "w", encoding="utf-8") as f:
                    write_report_content(
                        f=f,
                        project_root=config.project_root,
                        target_paths=target_paths,
                        lint_results=result_data.lint_results,
                        hard_coll=result_data.hard_collisions,
                        soft_coll=result_data.global_soft_collisions
                        + result_data.local_soft_collisions,
                        root_node=root_node_for_report,
                        config=config,
                    )
                result_data.report_generated = True
                logger_cli.debug("Vibe Report generation successful.")
            except Exception as e:
                logger_cli.error(f"Error generating vibe report: {e}", exc_info=True)
                result_data.report_error = str(e)
                report_failed = True

        report_failed_code = 1 if report_failed else 0
        final_exit_code = lint_exit_code or collision_exit_code or report_failed_code
        result_data.exit_code = final_exit_code
        result_data.success = final_exit_code == 0
        logger_cli.debug(f"Vibe Check command finished. Final Exit Code: {final_exit_code}")

    except Exception as e:
        logger_cli.error(f"Critical error during Vibe Check execution: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data

    if runner:
        _present_check_results(result_data, runner, console)
    else:
        console.print(
            "[bold red]Vibe Check failed critically before linting could start.[/bold red]"
        )
        if result_data.error_message:
            console.print(f"[red]Error: {result_data.error_message}[/red]")

    ctx.exit(result_data.exit_code)


@cli.command("namespace")
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save the namespace tree visualization to the specified file.",
)
@click.pass_context
def namespace(ctx: click.Context, output: Optional[Path]):
    """
    Visualize the project's Python namespace structure (how things import).

    Useful for untangling vibe conflicts.

    vibelint/cli.py
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing in namespace command"

    logger_cli.debug(f"Running 'namespace' command (output={output})")
    config = load_config(project_root)
    if config.project_root is None:
        # Fix Mapping vs Dict error here
        config = Config(project_root=project_root, config_dict=dict(config.settings))

    result_data = NamespaceResult()

    try:
        # Use the non-None project_root for target_paths
        target_paths: List[Path] = [project_root]
        logger_cli.info("Building namespace tree...")
        # Pass the non-None target_paths here too
        root_node, intra_file_collisions = build_namespace_tree(target_paths, config)
        result_data.root_node = root_node
        result_data.intra_file_collisions = intra_file_collisions

        if root_node is None:
            result_data.success = False
            result_data.error_message = "Namespace tree building resulted in None."
            tree_str = "[Error: Namespace tree could not be built]"
        else:
            result_data.success = True
            tree_str = str(root_node)

        if output:
            output_path = output.resolve()
            result_data.output_path = output_path
            logger_cli.info(f"Saving namespace tree to {output_path}...")
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(tree_str + "\n", encoding="utf-8")
                result_data.output_saved = True
            except Exception as e:
                logger_cli.error(f"Error saving namespace tree: {e}", exc_info=True)
                result_data.output_error = str(e)
        else:
            result_data.output_saved = False

        result_data.exit_code = 0 if result_data.success else 1

    except Exception as e:
        logger_cli.error(f"Error building namespace tree: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_namespace_results(result_data, console)
    ctx.exit(result_data.exit_code)


@cli.command("snapshot")
@click.option(
    "-o",
    "--output",
    default="codebase_snapshot.md",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output Markdown file name (default: codebase_snapshot.md)",
)
@click.pass_context
def snapshot(ctx: click.Context, output: Path):
    """
    Create a Markdown snapshot of project files (for LLMs or humans).

    Respects include/exclude rules from your config. Good for context dumping.

    vibelint/cli.py
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing in snapshot command"

    logger_cli.debug(f"Running 'snapshot' command (output={output})")
    config = load_config(project_root)
    if config.project_root is None:
        # Fix Mapping vs Dict error here
        config = Config(project_root=project_root, config_dict=dict(config.settings))

    result_data = SnapshotResult()
    output_path = output.resolve()
    result_data.output_path = output_path

    try:
        # Use the non-None project_root for target_paths
        target_paths: List[Path] = [project_root]
        logger_cli.info(f"Creating codebase snapshot at {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Pass the non-None target_paths here too
        create_snapshot(output_path=output_path, target_paths=target_paths, config=config)
        result_data.success = True
        result_data.exit_code = 0

    except Exception as e:
        logger_cli.error(f"Error creating snapshot: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_snapshot_results(result_data, console)
    ctx.exit(result_data.exit_code)


def main():
    """
    Main entry point for the vibelint CLI application.

    vibelint/cli.py
    """
    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        logger = logging.getLogger("vibelint")
        # Check if logger was configured before logging error
        if logger.hasHandlers():
            logger.error("Unhandled exception in CLI execution.", exc_info=True)
        else:
            # Fallback if error happened before logging setup
            import traceback

            print("Unhandled exception in CLI execution:", file=sys.stderr)
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---
### File: src/vibelint/config.py

```python
"""
Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/config.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

if sys.version_info >= (3, 11):

    import tomllib
else:

    try:

        import tomli as tomllib
    except ImportError:

        print(
            "Error: vibelint requires Python 3.11+ or the 'tomli' package "
            "to parse pyproject.toml on Python 3.10."
            "\nHint: Try running: pip install tomli"
        )
        sys.exit(1)


from .utils import find_package_root

logger = logging.getLogger(__name__)


class Config:
    """
    Holds the vibelint configuration loaded *exclusively* from pyproject.toml.

    Provides access to the project root and the raw configuration dictionary.
    It does *not* provide default values for missing keys. Callers must
    check for the existence of required settings.

    Attributes:
    project_root: The detected root of the project containing pyproject.toml.
    Can be None if pyproject.toml is not found.
    settings: A read-only view of the dictionary loaded from the
    [tool.vibelint] section of pyproject.toml. Empty if the
    file or section is missing or invalid.

    vibelint/config.py
    """

    def __init__(self, project_root: Optional[Path], config_dict: Dict[str, Any]):
        """
        Initializes Config.

        vibelint/config.py
        """
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Optional[Path]:
        """
        The detected project root directory, or None if not found.

        vibelint/config.py
        """
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Any]:
        """
        Read-only view of the settings loaded from [tool.vibelint].

        vibelint/config.py
        """
        return self._config_dict

    @property
    def ignore_codes(self) -> List[str]:
        """
        Returns the list of error codes to ignore, from config or empty list.

        vibelint/config.py
        """
        ignored = self.get("ignore", [])
        if isinstance(ignored, list) and all(isinstance(item, str) for item in ignored):
            return ignored
        elif ignored:
            logger.warning(
                "Configuration key 'ignore' in [tool.vibelint] is not a list of strings. Ignoring it."
            )
            return []
        else:
            return []

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the loaded settings, returning default if not found.

        vibelint/config.py
        """
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Gets a value, raising KeyError if the key is not found.

        vibelint/config.py
        """
        if key not in self._config_dict:
            raise KeyError(
                f"Required configuration key '{key}' not found in "
                f"[tool.vibelint] section of pyproject.toml."
            )
        return self._config_dict[key]

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key exists in the loaded settings.

        vibelint/config.py
        """
        return key in self._config_dict

    def is_present(self) -> bool:
        """
        Checks if a project root was found and some settings were loaded.

        vibelint/config.py
        """
        return self._project_root is not None and bool(self._config_dict)


def load_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration *only* from the nearest pyproject.toml file.

    Searches upwards from start_path. If pyproject.toml or the [tool.vibelint]
    section isn't found or is invalid, returns a Config object with project_root
    (if found) but an empty settings dictionary.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object. Check `config.project_root` and `config.settings`.

    vibelint/config.py
    """
    project_root = find_package_root(start_path)
    loaded_settings: Dict[str, Any] = {}

    if not project_root:
        logger.warning(
            f"Could not find project root (pyproject.toml) searching from '{start_path}'. "
            "No configuration will be loaded."
        )
        return Config(project_root=None, config_dict=loaded_settings)

    pyproject_path = project_root / "pyproject.toml"
    logger.debug(f"Found project root: {project_root}")
    logger.debug(f"Attempting to load config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:

            full_toml_config = tomllib.load(f)
        logger.debug("Parsed pyproject.toml")

        vibelint_config = full_toml_config.get("tool", {}).get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            if loaded_settings:
                logger.info(f"Loaded [tool.vibelint] settings from {pyproject_path}")
                logger.debug(f"Loaded settings: {loaded_settings}")
            else:
                logger.info(
                    f"Found {pyproject_path}, but the [tool.vibelint] section is empty or missing."
                )
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table (dictionary). "
                "Ignoring this section."
            )

    except FileNotFoundError:

        logger.error(
            f"pyproject.toml not found at {pyproject_path} despite project root detection."
        )
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}. Using empty configuration.")
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}. Using empty configuration.")
    except Exception as e:

        logger.exception(f"Unexpected error loading config from {pyproject_path}: {e}")

    return Config(project_root=project_root, config_dict=loaded_settings)


__all__ = ["Config", "load_config"]
```

---
### File: src/vibelint/discovery.py

```python
"""
Discovers files using pathlib glob/rglob based on include patterns from
pyproject.toml, respecting the pattern's implied scope, then filters
using exclude patterns.

If `include_globs` is missing from the configuration:
- If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
- Otherwise, logs an error and returns an empty list.

Exclusions from `config.exclude_globs` are always applied. Explicitly
provided paths are also excluded.

Warns if files within common VCS directories (.git, .hg, .svn) are found
and not covered by exclude_globs.

src/vibelint/discovery.py
"""

import fnmatch
import logging
import time
from pathlib import Path
from typing import List, Optional, Set

from .config import Config
from .utils import get_relative_path

__all__ = ["discover_files"]
logger = logging.getLogger(__name__)

_VCS_DIRS = {".git", ".hg", ".svn"}  # Keep this if needed for VCS warnings later


def _is_excluded(  # Keep this helper function as is
    file_path_abs: Path,
    project_root: Path,
    exclude_globs: List[str],
    explicit_exclude_paths: Set[Path],
) -> bool:
    """
    Checks if a discovered file path should be excluded.

    Checks explicit paths first, then exclude globs.

    Args:
    file_path_abs: The absolute path of the file found by globbing.
    project_root: The absolute path of the project root.
    exclude_globs: List of glob patterns for exclusion from config.
    explicit_exclude_paths: Set of absolute paths to exclude explicitly.

    Returns:
    True if the file should be excluded, False otherwise.

    vibelint/discovery.py
    """

    if file_path_abs in explicit_exclude_paths:
        logger.debug(f"Excluding explicitly provided path: {file_path_abs}")
        return True

    try:
        # Use resolve() for consistent comparison base
        rel_path = file_path_abs.resolve().relative_to(project_root.resolve())
        rel_path_str = str(rel_path).replace("\\", "/")  # Normalize for fnmatch
    except ValueError:
        logger.warning(f"Path {file_path_abs} is outside project root {project_root}. Excluding.")
        return True
    except Exception as e:
        logger.error(f"Error getting relative path for exclusion check on {file_path_abs}: {e}")
        return True  # Exclude if relative path fails

    for pattern in exclude_globs:
        normalized_pattern = pattern.replace("\\", "/")
        # Use fnmatch which matches based on Unix glob rules
        if fnmatch.fnmatch(rel_path_str, normalized_pattern):
            logger.debug(f"Excluding '{rel_path_str}' due to exclude pattern '{pattern}'")
            return True
        # Also check if the pattern matches any parent directory part
        # Example: exclude "build/" should match "build/lib/module.py"
        if "/" in normalized_pattern.rstrip("/"):  # Check if it's a directory pattern
            dir_pattern = normalized_pattern.rstrip("/") + "/"
            if rel_path_str.startswith(dir_pattern):
                logger.debug(
                    f"Excluding '{rel_path_str}' due to directory exclude pattern '{dir_pattern}'"
                )
                return True

    # logger.debug(f"Path '{rel_path_str}' not excluded by any pattern.") # Too verbose
    return False


def discover_files(
    paths: List[Path],  # Note: This argument seems unused for the main globbing logic
    config: Config,
    default_includes_if_missing: Optional[List[str]] = None,
    explicit_exclude_paths: Optional[Set[Path]] = None,
) -> List[Path]:
    """
    Discovers files using pathlib glob/rglob based on include patterns from
    pyproject.toml, respecting the pattern's implied scope, then filters
    using exclude patterns.

    If `include_globs` is missing from the configuration:
    - If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
    - Otherwise, logs an error and returns an empty list.

    Exclusions from `config.exclude_globs` are always applied. Explicitly
    provided paths are also excluded.

    Warns if files within common VCS directories (.git, .hg, .svn) are found
    and not covered by exclude_globs.

    Args:
    paths: Initial paths (largely ignored, globs operate from project root).
    config: The vibelint configuration object (must have project_root set).
    default_includes_if_missing: Fallback include patterns if 'include_globs'
    is not in config.settings.
    explicit_exclude_paths: A set of absolute file paths to explicitly exclude
    from the results, regardless of other rules.

    Returns:
    A sorted list of unique absolute Path objects for the discovered files.

    Raises:
    ValueError: If config.project_root is None.

    vibelint/discovery.py
    """

    if config.project_root is None:
        raise ValueError("Cannot discover files without a project root defined in Config.")

    project_root = config.project_root.resolve()
    candidate_files: Set[Path] = set()
    _explicit_excludes = {
        p.resolve() for p in (explicit_exclude_paths or set())
    }  # Resolve explicit excludes

    # --- Load include/exclude globs (Same as before) ---
    include_globs_config = config.get("include_globs")
    if include_globs_config is None:
        if default_includes_if_missing is not None:
            logger.warning(
                "Configuration key 'include_globs' missing in [tool.vibelint] section "
                f"of pyproject.toml. Using default patterns: {default_includes_if_missing}"
            )
            include_globs_effective = default_includes_if_missing
        else:
            logger.error(
                "Configuration key 'include_globs' missing in [tool.vibelint] section "
                "of pyproject.toml. No include patterns specified. Add 'include_globs' "
                "to pyproject.toml."
            )
            return []
    elif not isinstance(include_globs_config, list):
        logger.error(
            f"Configuration error: 'include_globs' in pyproject.toml must be a list. "
            f"Found type {type(include_globs_config)}. No files will be included."
        )
        return []
    elif not include_globs_config:
        logger.warning(
            "Configuration: 'include_globs' is present but empty in pyproject.toml. "
            "No files will be included."
        )
        include_globs_effective = []
    else:
        include_globs_effective = include_globs_config

    normalized_includes = [p.replace("\\", "/") for p in include_globs_effective]

    exclude_globs_config = config.get("exclude_globs", [])
    if not isinstance(exclude_globs_config, list):
        logger.error(
            f"Configuration error: 'exclude_globs' in pyproject.toml must be a list. "
            f"Found type {type(exclude_globs_config)}. Ignoring exclusions."
        )
        exclude_globs_effective = []
    else:
        exclude_globs_effective = exclude_globs_config
    normalized_exclude_globs = [p.replace("\\", "/") for p in exclude_globs_effective]

    logger.debug(f"Starting file discovery from project root: {project_root}")
    logger.debug(f"Effective Include globs: {normalized_includes}")
    logger.debug(f"Exclude globs: {normalized_exclude_globs}")
    logger.debug(f"Explicit excludes: {_explicit_excludes}")

    start_time = time.time()
    total_glob_yield_count = 0

    for pattern in normalized_includes:
        pattern_start_time = time.time()
        logger.debug(f"Processing include pattern: '{pattern}'")
        glob_method = project_root.rglob if "**" in pattern else project_root.glob
        pattern_yield_count = 0
        pattern_added_count = 0

        # --- Determine expected base directory for anchored patterns ---
        expected_base_dir: Optional[Path] = None
        pattern_path = Path(pattern)
        # Check if the first part of the pattern is a simple directory name (not a wildcard)
        # and the pattern contains a separator (implying it's not just a root file pattern)
        if (
            not pattern_path.is_absolute()
            and pattern_path.parts
            and not any(c in pattern_path.parts[0] for c in "*?[]")
            and ("/" in pattern or "\\" in pattern)  # Check if it contains a path separator
        ):
            expected_base_dir = project_root / pattern_path.parts[0]
            logger.debug(f"Pattern '{pattern}' implies base directory: {expected_base_dir}")

        try:
            logger.debug(f"Running {glob_method.__name__}('{pattern}')...")
            for p in glob_method(pattern):
                pattern_yield_count += 1
                total_glob_yield_count += 1
                abs_p = p.resolve()  # Resolve once

                logger.debug(
                    f"  Glob yielded (from '{pattern}'): {abs_p} (orig: {p}, is_file: {p.is_file()})"
                )

                # --- <<< FIX: Post-Glob Validation >>> ---
                is_valid_for_pattern = True  # Assume valid unless proven otherwise
                if expected_base_dir:
                    # If pattern implies a base dir (e.g., "src/..."), check path is relative to it
                    try:
                        # Use resolved paths for reliable check
                        abs_p.relative_to(expected_base_dir.resolve())
                    except ValueError:
                        logger.debug(
                            f"    -> Skipping {abs_p}: Yielded by anchored pattern '{pattern}' but not relative to expected base {expected_base_dir}"
                        )
                        is_valid_for_pattern = False
                    except Exception as path_err:
                        logger.warning(
                            f"    -> Error checking relative path for {abs_p} against {expected_base_dir}: {path_err}. Allowing through."
                        )
                elif (
                    "/" not in pattern
                    and "\\" not in pattern
                    and not any(c in pattern for c in "*?[]")
                ):
                    # If pattern is a simple filename (e.g., "pyproject.toml"), check it's directly under root
                    if abs_p.parent != project_root:
                        logger.debug(
                            f"    -> Skipping {abs_p}: Yielded by root pattern '{pattern}' but not in project root directory."
                        )
                        is_valid_for_pattern = False

                if not is_valid_for_pattern:
                    continue  # Skip this path if it didn't belong to the pattern's scope
                # --- <<< END FIX >>> ---

                if p.is_symlink():
                    logger.debug(f"    -> Skipping discovered symlink: {p}")
                    continue

                # We only care about files from here on for candidacy
                if p.is_file():
                    # Add the *resolved* absolute path to the candidates
                    logger.debug(f"    -> Adding candidate: {abs_p} (from pattern '{pattern}')")
                    candidate_files.add(abs_p)
                    pattern_added_count += 1
                # else: # Log directories yielded if needed for debugging
                #      logger.debug(f"    -> Ignoring directory yielded by glob: {p}")

        except PermissionError as e:
            logger.warning(
                f"Permission denied accessing path during glob for pattern '{pattern}': {e}. Skipping."
            )
        except Exception as e:
            logger.error(f"Error during glob execution for pattern '{pattern}': {e}", exc_info=True)

        pattern_time = time.time() - pattern_start_time
        logger.debug(
            f"Pattern '{pattern}' yielded {pattern_yield_count} paths, added {pattern_added_count} candidates in {pattern_time:.4f} seconds."
        )

    discovery_time = time.time() - start_time
    logger.debug(
        f"Initial globbing finished in {discovery_time:.4f} seconds. Total yielded paths: {total_glob_yield_count}. Total candidates: {len(candidate_files)}"
    )

    logger.debug(f"Applying exclude rules to {len(candidate_files)} candidates...")
    final_files_set: Set[Path] = set()
    exclusion_start_time = time.time()

    # Sort candidates for deterministic processing order (optional but good)
    sorted_candidates = sorted(list(candidate_files), key=str)

    for file_path_abs in sorted_candidates:
        if not _is_excluded(
            file_path_abs, project_root, normalized_exclude_globs, _explicit_excludes
        ):
            logger.debug(f"Including file: {file_path_abs}")
            final_files_set.add(file_path_abs)
        # else: # No need to log every exclusion unless debugging excludes specifically
        # try:
        #     rel_path_exc = get_relative_path(file_path_abs, project_root)
        #     logger.debug(f"Excluding file based on rules: {rel_path_exc}")
        # except ValueError:
        #      logger.debug(f"Excluding file based on rules: {file_path_abs}")

    exclusion_time = time.time() - exclusion_start_time
    logger.debug(f"Exclusion phase finished in {exclusion_time:.4f} seconds.")

    # --- VCS Warning Logic (Optional, keep if desired) ---
    # ... (keep the existing VCS warning logic if you want it) ...
    vcs_warnings: Set[Path] = set()
    if final_files_set:  # Check against final included files
        for file_path in final_files_set:
            try:
                is_in_vcs_dir = any(
                    part in _VCS_DIRS for part in file_path.relative_to(project_root).parts
                )
                if is_in_vcs_dir:
                    # Check if it *would* have been excluded if a pattern existed
                    # This check is slightly complex - maybe simplify the warning?
                    # For now, let's just warn if *any* included file is in a dir matching VCS name parts
                    vcs_warnings.add(file_path)
            except ValueError:  # Outside project root
                pass
            except Exception as e_vcs:
                logger.debug(f"Error during VCS check for {file_path}: {e_vcs}")

    if vcs_warnings:
        logger.warning(
            f"Found {len(vcs_warnings)} included files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}). Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )
        # Log first few examples
        try:
            paths_to_log = [
                get_relative_path(p, project_root) for p in sorted(list(vcs_warnings), key=str)[:5]
            ]
            for rel_path_warn in paths_to_log:
                logger.warning(f"  - {rel_path_warn}")
            if len(vcs_warnings) > 5:
                logger.warning(f"  - ... and {len(vcs_warnings) - 5} more.")
        except ValueError:
            logger.warning("  (Could not display example relative paths - outside project root?)")
        except Exception as e_log:
            logger.warning(f"  (Error logging example paths: {e_log})")

    # --- Final Count Logging (Same as before) ---
    final_count = len(final_files_set)
    if final_count == 0 and len(candidate_files) > 0 and include_globs_effective:
        logger.warning("All candidate files were excluded. Check your exclude_globs patterns.")
    elif final_count == 0 and not include_globs_effective:
        pass  # Expected if includes are empty
    elif final_count == 0:
        if include_globs_effective and total_glob_yield_count == 0:
            logger.warning("No files found matching include_globs patterns.")

    logger.debug(f"Discovery complete. Returning {final_count} files.")
    return sorted(list(final_files_set))
```

---
### File: src/vibelint/error_codes.py

```python
"""
Defines error and warning codes used by vibelint, along with descriptions.

Codes follow the pattern VBL<category><id>
Categories:
1xx: Docstrings
2xx: Encoding
3xx: Exports (__all__)
4xx: Shebang
5xx: Namespace (Reserved for future use if needed for collision reporting)
9xx: Internal/Processing Errors

vibelint/error_codes.py
"""

VBL101 = "VBL101"
VBL102 = "VBL102"
# VBL103 removed


VBL201 = "VBL201"


VBL301 = "VBL301"
VBL302 = "VBL302"
VBL303 = "VBL303"
VBL304 = "VBL304"


VBL401 = "VBL401"
VBL402 = "VBL402"
VBL403 = "VBL403"


VBL901 = "VBL901"
VBL902 = "VBL902"
VBL903 = "VBL903"
VBL904 = "VBL904"
VBL905 = "VBL905"


CODE_DESCRIPTIONS = {
    VBL101: "Missing docstring for module, class, or function.",
    VBL102: "Docstring does not end with the expected relative file path reference.",
    VBL201: "Invalid encoding cookie value (must be 'utf-8').",
    VBL301: "`__all__` definition is missing in a module where it is required.",
    VBL302: "`__all__` definition is missing in `__init__.py` (Optional based on config).",
    VBL303: "`__all__` is assigned a value that is not a List or Tuple.",
    VBL304: "SyntaxError parsing file during `__all__` validation.",
    VBL401: "File has a shebang line (`#!...`) but no `if __name__ == '__main__'` block.",
    VBL402: "Shebang line value is not in the list of allowed shebangs (check config).",
    VBL403: "File contains `if __name__ == '__main__'` block but lacks a shebang line.",
    VBL901: "Error reading file content (permissions, encoding, etc.).",
    VBL902: "SyntaxError parsing file during validation.",
    VBL903: "Internal error during validation phase for a file.",
    VBL904: "Error occurred in file processing thread.",
    VBL905: "Critical unhandled error during processing of a file.",
}


__all__ = [
    "VBL101",
    "VBL102",
    "VBL201",
    "VBL301",
    "VBL302",
    "VBL303",
    "VBL304",
    "VBL401",
    "VBL402",
    "VBL403",
    "VBL901",
    "VBL902",
    "VBL903",
    "VBL904",
    "VBL905",
    "CODE_DESCRIPTIONS",
]
```

---
### File: src/vibelint/lint.py

```python
"""
Core linting runner for vibelint.

vibelint/lint.py
"""

import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set, Tuple

import click
from rich.console import Console
from rich.progress import TimeElapsedColumn  # Add for better progress visibility
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from .config import Config
from .discovery import discover_files
from .error_codes import VBL901, VBL902, VBL903, VBL904, VBL905
from .utils import get_relative_path
from .validators.docstring import validate_every_docstring
from .validators.encoding import validate_encoding_cookie
from .validators.exports import validate_exports
from .validators.shebang import file_contains_top_level_main_block, validate_shebang

__all__ = ["LintResult", "LintRunner"]

console = Console()  # Keep console instance if used elsewhere, otherwise remove
logger = logging.getLogger(__name__)

ValidationIssue = Tuple[str, str]


class LintResult:
    """
    Stores the result of a lint operation (Vibe Check) on a single file.

    vibelint/lint.py
    """

    def __init__(self) -> None:
        """
        Initializes a LintResult instance.

        vibelint/lint.py
        """
        self.file_path: Path = Path()
        self.errors: List[ValidationIssue] = []  # Major vibe failures
        self.warnings: List[ValidationIssue] = []  # Minor vibe issues

    @property
    def has_issues(self) -> bool:
        """
        Returns True if there are any errors or warnings.

        vibelint/lint.py
        """
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner for linting operations (Vibe Checks).

    vibelint/lint.py
    """

    def __init__(self, config: Config, skip_confirmation: bool = False) -> None:
        """
        Initializes the LintRunner.

        Args:
            config: The vibelint configuration object.
            skip_confirmation: If True, bypass the confirmation prompt for large directories.

        vibelint/lint.py
        """
        self.config = config
        if not self.config.project_root:
            # Add a check here as project_root is crucial
            raise ValueError("LintRunner requires a config object with project_root set.")

        self.skip_confirmation = skip_confirmation
        self.results: List[LintResult] = []
        self._final_exit_code: int = 0  # Determined by errors

    def run(self, paths: List[Path]) -> int:
        """
        Runs the linting process (Vibe Check) and returns the exit code (0 for pass, 1 for fail).

        Failure is determined by the presence of *errors* (not warnings).

        vibelint/lint.py
        """
        logger.debug("LintRunner.run: Starting file discovery for Vibe Check...")
        if not self.config.project_root:  # Should be caught by __init__, but double-check
            logger.error("Project root not found in config. Cannot run Vibe Check.")
            return 1

        ignore_codes_set = set(self.config.ignore_codes)
        if ignore_codes_set:
            logger.info(f"Ignoring codes: {sorted(list(ignore_codes_set))}")

        discovery_start = time.time()
        all_discovered_files: List[Path] = discover_files(
            paths=paths, config=self.config, explicit_exclude_paths=set()
        )
        discovery_time = time.time() - discovery_start
        logger.debug(f"File discovery took {discovery_time:.4f} seconds.")

        python_files: List[Path] = [
            f for f in all_discovered_files if f.is_file() and f.suffix == ".py"
        ]
        logger.debug(f"LintRunner.run: Discovered {len(python_files)} Python files for Vibe Check.")

        if not python_files:
            logger.info("No Python files found matching includes/excludes. No vibes to check.")
            self._final_exit_code = 0  # No files means no errors
            return self._final_exit_code

        large_dir_threshold = self.config.get("large_dir_threshold", 500)
        if len(python_files) > large_dir_threshold and not self.skip_confirmation:
            logger.debug(
                f"File count {len(python_files)} > threshold {large_dir_threshold}. Requesting confirmation."
            )
            if not self._confirm_large_directory(len(python_files)):
                logger.info("User aborted Vibe Check due to large file count.")
                self._final_exit_code = 1  # User aborted, treat as failure? Or 0? Let's go with 1.
                return self._final_exit_code

        # Consider making MAX_WORKERS configurable? Default to None (cpu_count * 5)
        MAX_WORKERS = self.config.get("max_workers")
        logger.debug(f"LintRunner.run: Processing files with max_workers={MAX_WORKERS}")
        progress_console = Console(
            stderr=True, force_terminal=True if sys.stderr.isatty() else False
        )  # Force terminal for CI if needed
        processing_start = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),  # Show time elapsed
            console=progress_console,
            transient=True,  # Clears progress bar on completion
        ) as progress:

            task_desc = f"Conducting Vibe Check on {len(python_files)} Python files..."
            task_id = progress.add_task(task_desc, total=len(python_files))
            # Use a context manager for the executor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Map filenames to futures for better error reporting
                futures_map = {
                    executor.submit(self._process_file, f, ignore_codes_set): f
                    for f in python_files
                }
                temp_results = []
                # Use as_completed for potentially better responsiveness if needed
                # from concurrent.futures import as_completed
                # for future in as_completed(futures_map):
                #    file_proc = futures_map[future]
                for (
                    future,
                    file_proc,
                ) in (
                    futures_map.items()
                ):  # Iterate directly if order doesn't matter strictly before sorting
                    try:
                        res = future.result()  # Get result or raise exception
                        temp_results.append(res)
                    except Exception as exc:
                        # Log exceptions clearly tied to the file
                        rel_path_log_err = file_proc.name  # Fallback name
                        try:
                            if self.config.project_root:
                                rel_path_log_err = str(
                                    get_relative_path(file_proc, self.config.project_root)
                                )
                        except ValueError:
                            rel_path_log_err = str(
                                file_proc.resolve()
                            )  # Absolute path if outside root

                        logger.error(
                            f"Exception processing {rel_path_log_err}: {exc}",
                            exc_info=True,  # Include traceback in debug log
                        )
                        # Create a result indicating the processing error
                        lr_err = LintResult()
                        lr_err.file_path = file_proc
                        # Use a specific error code for processing errors
                        lr_err.errors.append((VBL904, f"Processing thread error: {exc}"))
                        temp_results.append(lr_err)
                    finally:
                        progress.update(
                            task_id, advance=1
                        )  # Advance progress regardless of success/failure

                # Sort results only once after all futures are complete
                self.results = sorted(temp_results, key=lambda r: r.file_path)

        processing_time = time.time() - processing_start
        logger.debug(f"File processing took {processing_time:.4f} seconds.")

        # Determine exit code based *only* on the presence of errors in the results
        files_with_errors = sum(1 for r in self.results if r.errors)
        self._final_exit_code = 1 if files_with_errors > 0 else 0
        logger.debug(
            f"LintRunner.run: Vibe Check finished. Found {files_with_errors} files with major failures. Final exit code: {self._final_exit_code}"
        )
        return self._final_exit_code

    def _process_file(self, file_path: Path, ignore_codes_set: Set[str]) -> LintResult:
        """
        Processes a single file for linting issues (Vibe Check), filtering by ignored codes.

        Args:
            file_path: The absolute path to the Python file.
            ignore_codes_set: A set of VBL codes to ignore.

        Returns:
            A LintResult object for the file.

        vibelint/lint.py
        """
        lr = LintResult()
        lr.file_path = file_path
        relative_path_str = file_path.name  # Default
        log_prefix = f"[{file_path.name}]"  # Use simple name for shorter logs initially
        original_content: Optional[str] = None
        collected_errors: List[ValidationIssue] = []
        collected_warnings: List[ValidationIssue] = []

        try:
            # Determine relative path for logging and validation use
            if self.config.project_root:
                try:
                    relative_path = get_relative_path(file_path, self.config.project_root)
                    relative_path_str = str(relative_path).replace("\\", "/")  # Use normalized path
                    log_prefix = f"[{relative_path_str}]"  # Update log prefix
                except ValueError:
                    # File might be outside project root (e.g., explicitly passed)
                    relative_path_str = str(file_path.resolve())
                    log_prefix = f"[{relative_path_str}]"
                    logger.warning(f"{log_prefix} File seems outside project root.")
            else:
                # Should not happen if LintRunner checks config on init, but handle defensively
                relative_path_str = str(file_path.resolve())
                log_prefix = f"[{relative_path_str}]"
                logger.error("Project root missing in config during file processing!")

            logger.debug(f"{log_prefix} --- Starting Vibe Validation ---")

            # 1. Read File Content
            try:
                # Read as bytes first to detect encoding issues if not UTF-8? Maybe too complex. Stick to UTF-8.
                original_content = file_path.read_text(encoding="utf-8")
                logger.debug(f"{log_prefix} Read {len(original_content)} bytes.")
            except Exception as read_e:
                logger.error(f"{log_prefix} Error reading file: {read_e}", exc_info=True)
                # Add a specific error code for file reading issues
                lr.errors.append((VBL901, f"Error reading file: {read_e}"))
                # Can't proceed without content
                return lr

            # 2. Run Validators (Order might matter if one depends on syntax)
            try:
                # Docstring Validation (VBL1xx)
                # Pass the already read content and calculated relative path
                doc_res, _ = validate_every_docstring(original_content, relative_path_str)
                if doc_res:
                    collected_errors.extend(doc_res.errors)
                    collected_warnings.extend(doc_res.warnings)

                # Shebang Validation (VBL4xx)
                allowed_sb: List[str] = self.config.get(
                    "allowed_shebangs", ["#!/usr/bin/env python3"]  # Provide a default
                )
                # Pass content to avoid re-reading; pass path for context/AST parsing if needed
                is_script = file_contains_top_level_main_block(file_path, original_content)
                sb_res = validate_shebang(original_content, is_script, allowed_sb)
                collected_errors.extend(sb_res.errors)
                collected_warnings.extend(sb_res.warnings)

                # Encoding Cookie Validation (VBL2xx)
                enc_res = validate_encoding_cookie(original_content)
                collected_errors.extend(enc_res.errors)
                collected_warnings.extend(enc_res.warnings)

                # Export (__all__) Validation (VBL3xx)
                export_res = validate_exports(original_content, relative_path_str, self.config)
                collected_errors.extend(export_res.errors)
                collected_warnings.extend(export_res.warnings)

                logger.debug(
                    f"{log_prefix} Validation Complete. Found E={len(collected_errors)}, W={len(collected_warnings)} (before filtering)"
                )

            # Handle specific parsing errors
            except SyntaxError as se:
                line = f"line {se.lineno}" if se.lineno is not None else "unk line"
                col = f", col {se.offset}" if se.offset is not None else ""
                err_msg = f"SyntaxError parsing file: {se.msg} ({relative_path_str}, {line}{col})"
                logger.error(f"{log_prefix} {err_msg}")
                # Add a specific error code for syntax errors
                collected_errors.append((VBL902, err_msg))

            # Catch internal errors during the validation phase for this file
            except Exception as val_e:
                logger.error(
                    f"{log_prefix} Error during validation phase: {val_e}",
                    exc_info=True,  # Log traceback for internal errors
                )
                # Add a specific error code for internal validation errors
                collected_errors.append((VBL903, f"Internal validation error: {val_e}"))

            # 3. Filter Results based on `ignore_codes_set`
            lr.errors = [
                (code, msg) for code, msg in collected_errors if code not in ignore_codes_set
            ]
            lr.warnings = [
                (code, msg) for code, msg in collected_warnings if code not in ignore_codes_set
            ]

            if len(collected_errors) != len(lr.errors) or len(collected_warnings) != len(
                lr.warnings
            ):
                ignored_count = (len(collected_errors) - len(lr.errors)) + (
                    len(collected_warnings) - len(lr.warnings)
                )
                logger.debug(
                    f"{log_prefix} Filtered {ignored_count} issue(s) based on ignore config. Final E={len(lr.errors)}, W={len(lr.warnings)}"
                )

        except Exception as e:
            # Catch any other unexpected errors during processing for this file
            logger.error(
                f"{log_prefix} Critical unhandled error in _process_file: {e}\n{traceback.format_exc()}"
            )
            # Use a specific error code for critical processing errors
            # Ensure previous errors aren't lost if this happens late
            lr.errors.append((VBL905, f"Critical processing error: {e}"))

        logger.debug(f"{log_prefix} --- Finished Vibe Validation ---")
        return lr

    def _confirm_large_directory(self, file_count: int) -> bool:
        """
        Asks user for confirmation if many files are found.

        Args:
            file_count: The number of Python files found.

        Returns:
            True if the user confirms, False otherwise.

        vibelint/lint.py
        """
        prompt_console = Console(stderr=True)  # Ensure prompts go to stderr
        prompt_console.print(
            f"[yellow]WARNING:[/yellow] Found {file_count} Python files to check. This might take a moment."
        )
        try:
            # Use Click's confirmation prompt
            return click.confirm("Proceed with Vibe Check?", default=False, err=True)
        except click.Abort:
            # User pressed Ctrl+C or similar during prompt
            prompt_console.print("[yellow]Vibe Check aborted by user.[/yellow]", style="yellow")
            return False
        except RuntimeError as e:
            # Handle cases where prompt cannot be shown (e.g., non-interactive session)
            if "Cannot prompt" in str(e) or "tty" in str(e).lower():
                prompt_console.print(
                    "[yellow]Non-interactive session detected. Use --yes to bypass confirmation. Aborting Vibe Check.[/yellow]",
                    style="yellow",
                )
            else:
                # Log unexpected runtime errors during prompt
                logger.error(f"RuntimeError during confirmation prompt: {e}", exc_info=True)
            return False
        except Exception as e:
            # Catch any other unexpected errors during the prompt
            logger.error(f"Error during confirmation prompt: {e}", exc_info=True)
            return False

    def _print_summary(self) -> None:
        """
        Prints a summary table of the linting results using "Vibe Check" terminology.

        vibelint/lint.py
        """
        summary_console = (
            Console()
        )  # Use a separate console instance if needed, or reuse the global one
        # --- Use Vibe Check Terminology ---
        table = Table(title="vibelint Vibe Check Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")

        total = len(self.results)
        # Count files with actual VBL errors (Major Failures)
        files_with_errors = sum(1 for r in self.results if r.errors)
        # Count files with only warnings (Minor Issues)
        files_with_warnings_only = sum(1 for r in self.results if r.warnings and not r.errors)
        files_ok = total - files_with_errors - files_with_warnings_only

        table.add_row("Files Scanned", str(total))
        table.add_row(
            "Files Vibing", str(files_ok), style="green" if files_ok == total and total > 0 else ""
        )
        table.add_row(
            "Files Not Vibing (Errors)",
            str(files_with_errors),
            style="red" if files_with_errors else "",
        )
        table.add_row(
            "Files In Between (Warnings)",
            str(files_with_warnings_only),
            style="yellow" if files_with_warnings_only else "",
        )

        summary_console.print(table)
        # --- End Vibe Check Terminology ---
```

---
### File: src/vibelint/namespace.py

```python
"""
Namespace representation & collision detection for Python code.

vibelint/namespace.py
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .discovery import discover_files
from .utils import find_project_root, get_relative_path

__all__ = [
    "CollisionType",
    "NamespaceCollision",
    "NamespaceNode",
    "detect_hard_collisions",
    "detect_global_definition_collisions",
    "detect_local_export_collisions",
    "build_namespace_tree",
    "get_namespace_collisions_str",
]

logger = logging.getLogger(__name__)


class CollisionType:
    """
    Enum-like class for collision types.

    vibelint/namespace.py
    """

    HARD = "hard"
    LOCAL_SOFT = "local_soft"
    GLOBAL_SOFT = "global_soft"


class NamespaceCollision:
    """
    Represents a collision between two or more same-named entities.

    vibelint/namespace.py
    """

    def __init__(
        self,
        name: str,
        collision_type: str,
        paths: List[Path],
        linenos: Optional[List[Optional[int]]] = None,
    ) -> None:
        """
        Initializes a NamespaceCollision instance.

        Args:
        name: The name of the colliding entity.
        collision_type: The type of collision (HARD, LOCAL_SOFT, GLOBAL_SOFT).
        paths: A list of Path objects for all files involved in the collision.
        linenos: An optional list of line numbers corresponding to each path.

        vibelint/namespace.py
        """

        if not paths:
            raise ValueError("At least one path must be provided for a collision.")

        self.name = name
        self.collision_type = collision_type

        self.paths = sorted(list(set(paths)), key=str)

        self.linenos = (
            linenos if linenos and len(linenos) == len(self.paths) else [None] * len(self.paths)
        )

        self.path1: Path = self.paths[0]
        self.path2: Path = self.paths[1] if len(self.paths) > 1 else self.paths[0]
        self.lineno1: Optional[int] = self.linenos[0] if self.linenos else None
        self.lineno2: Optional[int] = self.linenos[1] if len(self.linenos) > 1 else self.lineno1

        self.definition_paths: List[Path] = (
            self.paths
            if self.collision_type in [CollisionType.GLOBAL_SOFT, CollisionType.LOCAL_SOFT]
            else []
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for debugging.

        vibelint/namespace.py
        """

        return (
            f"NamespaceCollision(name='{self.name}', type='{self.collision_type}', "
            f"paths={self.paths}, linenos={self.linenos})"
        )

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the collision.

        vibelint/namespace.py
        """

        proj_root = find_project_root(Path(".").resolve())
        base_path = proj_root if proj_root else Path(".")

        paths_str_list = []
        for i, p in enumerate(self.paths):
            loc = f":{self.linenos[i]}" if self.linenos and self.linenos[i] is not None else ""
            try:
                paths_str_list.append(f"{get_relative_path(p, base_path)}{loc}")
            except ValueError:
                paths_str_list.append(f"{p}{loc}")
        paths_str = ", ".join(paths_str_list)

        if self.collision_type == CollisionType.HARD:
            if len(self.paths) == 2 and self.paths[0] == self.paths[1]:

                line_info = ""
                if self.lineno1 is not None and self.lineno2 is not None:
                    line_info = f" (lines ~{self.lineno1} and ~{self.lineno2})"
                elif self.lineno1 is not None:
                    line_info = f" (line ~{self.lineno1})"

                return (
                    f"{self.collision_type.upper()} Collision: Duplicate definition/import of '{self.name}' in "
                    f"{paths_str_list[0]}{line_info}"
                )
            else:
                return f"{self.collision_type.upper()} Collision: Name '{self.name}' used by conflicting entities in: {paths_str}"
        elif self.collision_type == CollisionType.LOCAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' exported via __all__ in multiple sibling modules: {paths_str}"
        elif self.collision_type == CollisionType.GLOBAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' defined in multiple modules: {paths_str}"
        else:
            return f"Unknown Collision: '{self.name}' involving paths: {paths_str}"


def detect_hard_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect HARD collisions: member vs. submodule, or duplicate definitions within a file.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected HARD NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, intra_file_collisions = build_namespace_tree(paths, config)

    inter_file_collisions = root_node.get_hard_collisions()

    all_collisions = intra_file_collisions + inter_file_collisions
    for c in all_collisions:
        c.collision_type = CollisionType.HARD
    return all_collisions


def detect_global_definition_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect GLOBAL SOFT collisions: the same name defined/assigned at the top level
    in multiple different modules across the project.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected GLOBAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)

    definition_collisions = root_node.detect_global_definition_collisions()

    return definition_collisions


def detect_local_export_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect LOCAL SOFT collisions: the same name exported via __all__ by multiple
    sibling modules within the same package.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected LOCAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)
    collisions: List[NamespaceCollision] = []
    root_node.find_local_export_collisions(collisions)
    return collisions


def get_namespace_collisions_str(
    paths: List[Path],
    config: Config,
    console=None,
) -> str:
    """
    Return a string representation of all collision types for quick debugging.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.
    console: Optional console object (unused).

    Returns:
    A string summarizing all detected collisions.

    vibelint/namespace.py
    """

    from io import StringIO

    buf = StringIO()

    hard_collisions = detect_hard_collisions(paths, config)
    global_soft_collisions = detect_global_definition_collisions(paths, config)
    local_soft_collisions = detect_local_export_collisions(paths, config)

    proj_root = find_project_root(Path(".").resolve())
    base_path = proj_root if proj_root else Path(".")

    if hard_collisions:
        buf.write("Hard Collisions:\n")
        for c in sorted(hard_collisions, key=lambda x: (x.name, str(x.paths[0]))):
            buf.write(f"- {str(c)}\n")

    if local_soft_collisions:
        buf.write("\nLocal Soft Collisions (__all__):\n")

        grouped = defaultdict(list)
        for c in local_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': exported by {paths_str}\n")

    if global_soft_collisions:
        buf.write("\nGlobal Soft Collisions (Definitions):\n")

        grouped = defaultdict(list)
        for c in global_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': defined in {paths_str}\n")

    return buf.getvalue()


class NamespaceNode:
    """
    A node in the "module" hierarchy (like package/subpackage, or file-level).
    Holds child nodes and top-level members (functions/classes).

    vibelint/namespace.py
    """

    def __init__(self, name: str, path: Optional[Path] = None, is_package: bool = False) -> None:
        """
        Initializes a NamespaceNode.

        Args:
        name: The name of the node (e.g., module name, package name).
        path: The filesystem path associated with this node (optional).
        is_package: True if this node represents a package (directory).

        vibelint/namespace.py
        """

        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: Dict[str, "NamespaceNode"] = {}

        self.members: Dict[str, Tuple[Path, Optional[int]]] = {}

        self.member_collisions: List[NamespaceCollision] = []

        self.exported_names: Optional[List[str]] = None

    def set_exported_names(self, names: List[str]):
        """
        Sets the list of names found in __all__.

        vibelint/namespace.py
        """

        self.exported_names = names

    def add_child(self, name: str, path: Path, is_package: bool = False) -> "NamespaceNode":
        """
        Adds a child node, creating if necessary.

        vibelint/namespace.py
        """

        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)

        elif path:

            if not (self.children[name].is_package and not is_package):
                self.children[name].path = path
            self.children[name].is_package = is_package or self.children[name].is_package
        return self.children[name]

    def get_hard_collisions(self) -> List[NamespaceCollision]:
        """
        Detect HARD collisions recursively: members vs. child modules.

        vibelint/namespace.py
        """

        collisions: List[NamespaceCollision] = []

        member_names_with_info = {}
        if self.is_package and self.path:
            init_path = (self.path / "__init__.py").resolve()
            member_names_with_info = {
                name: (def_path, lineno)
                for name, (def_path, lineno) in self.members.items()
                if def_path.resolve() == init_path
            }

        child_names = set(self.children.keys())
        common_names = set(member_names_with_info.keys()).intersection(child_names)

        for name in common_names:

            member_def_path, member_lineno = member_names_with_info.get(name, (None, None))
            cnode = self.children[name]
            child_path = cnode.path

            if member_def_path and child_path:

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[member_def_path, child_path],
                        linenos=[member_lineno, None],
                    )
                )

        for cnode in self.children.values():
            collisions.extend(cnode.get_hard_collisions())
        return collisions

    def collect_defined_members(self, all_dict: Dict[str, List[Tuple[Path, Optional[int]]]]):
        """
        Recursively collects defined members (path, lineno) for global definition collision check.

        vibelint/namespace.py
        """

        if self.path and self.members:

            for mname, (mpath, mlineno) in self.members.items():
                all_dict.setdefault(mname, []).append((mpath, mlineno))

        for cnode in self.children.values():
            cnode.collect_defined_members(all_dict)

    def detect_global_definition_collisions(self) -> List[NamespaceCollision]:
        """
        Detects GLOBAL SOFT collisions across the whole tree starting from this node.

        vibelint/namespace.py
        """

        all_defined_members: Dict[str, List[Tuple[Path, Optional[int]]]] = defaultdict(list)
        self.collect_defined_members(all_defined_members)

        collisions: List[NamespaceCollision] = []
        for name, path_lineno_list in all_defined_members.items():

            unique_paths_map: Dict[Path, Optional[int]] = {}
            for path, lineno in path_lineno_list:
                resolved_p = path.resolve()

                if resolved_p not in unique_paths_map:
                    unique_paths_map[resolved_p] = lineno

            if len(unique_paths_map) > 1:

                sorted_paths = sorted(unique_paths_map.keys(), key=str)

                sorted_linenos = [unique_paths_map[p] for p in sorted_paths]

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.GLOBAL_SOFT,
                        paths=sorted_paths,
                        linenos=sorted_linenos,
                    )
                )
        return collisions

    def find_local_export_collisions(self, collisions_list: List[NamespaceCollision]):
        """
        Recursively finds LOCAL SOFT collisions (__all__) within packages.

        Args:
        collisions_list: A list to append found collisions to.

        vibelint/namespace.py
        """

        if self.is_package:
            exports_in_package: Dict[str, List[Path]] = defaultdict(list)

            if self.path and self.path.is_dir() and self.exported_names:

                init_path = (self.path / "__init__.py").resolve()

                if init_path.exists() and any(
                    p.resolve() == init_path for p, _ in self.members.values()
                ):
                    for name in self.exported_names:
                        exports_in_package[name].append(init_path)

            for child in self.children.values():

                if (
                    child.path
                    and child.path.is_file()
                    and not child.is_package
                    and child.name != "__init__"
                    and child.exported_names
                ):
                    for name in child.exported_names:
                        exports_in_package[name].append(child.path.resolve())

            for name, paths in exports_in_package.items():
                unique_paths = sorted(list(set(paths)), key=str)
                if len(unique_paths) > 1:
                    collisions_list.append(
                        NamespaceCollision(
                            name=name,
                            collision_type=CollisionType.LOCAL_SOFT,
                            paths=unique_paths,
                            linenos=[None for _ in unique_paths],
                        )
                    )

        for child in self.children.values():
            if child.is_package:
                child.find_local_export_collisions(collisions_list)

    def __str__(self) -> str:
        """
        Provides a string representation of the node and its subtree, including members.
        Uses a revised formatting approach for better clarity relative to project root.

        vibelint/namespace.py
        """

        lines = []

        proj_root = find_project_root(Path(".").resolve())
        base_path_for_display = proj_root if proj_root else Path(".")

        def build_tree_lines(node: "NamespaceNode", prefix: str = "", base: Path = Path(".")):
            """
            Docstring for function 'build_tree_lines'.

            vibelint/namespace.py
            """

            child_items = sorted(node.children.items())

            direct_members = []
            if node.path and node.members:

                expected_def_path = None
                node_path_resolved = node.path.resolve()
                if node.is_package and node_path_resolved.is_dir():
                    expected_def_path = (node_path_resolved / "__init__.py").resolve()
                elif node_path_resolved.is_file():
                    expected_def_path = node_path_resolved

                if expected_def_path:
                    direct_members = sorted(
                        [
                            name
                            for name, (def_path, _) in node.members.items()
                            if def_path.resolve() == expected_def_path
                        ]
                    )

            all_items = child_items + [(name, "member") for name in direct_members]
            total_items = len(all_items)

            for i, (name, item) in enumerate(all_items):
                is_last = i == total_items - 1
                connector = "└── " if is_last else "├── "
                next_level_prefix = prefix + ("    " if is_last else "│   ")

                if item == "member":

                    lines.append(f"{prefix}{connector}{name} (member)")
                else:

                    child: "NamespaceNode" = item
                    child_path_str = ""
                    indicator = ""
                    if child.path:
                        try:
                            rel_p = get_relative_path(child.path, base)

                            if child.is_package:
                                indicator = " (P)"
                            elif child.name == "__init__":
                                indicator = " (I)"
                            else:
                                indicator = " (M)"
                            child_path_str = f"  [{rel_p}{indicator}]"
                        except ValueError:
                            indicator = (
                                " (P)"
                                if child.is_package
                                else (" (I)" if child.name == "__init__" else " (M)")
                            )
                            child_path_str = f"  [{child.path.resolve()}{indicator}]"
                    else:
                        child_path_str = "  [No Path]"

                    lines.append(f"{prefix}{connector}{name}{child_path_str}")

                    if child.children or (
                        child.members
                        and any(
                            m_path.resolve() == (child.path.resolve() if child.path else None)
                            for m, (m_path, _) in child.members.items()
                        )
                    ):
                        build_tree_lines(child, next_level_prefix, base)

        root_path_str = ""
        root_indicator = ""

        if self.path:
            root_path_resolved = self.path.resolve()
            try:

                rel_p = get_relative_path(root_path_resolved, base_path_for_display.parent)

                if rel_p == Path("."):
                    rel_p = Path(self.name)

                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{rel_p}{root_indicator}]"
            except ValueError:
                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{root_path_resolved}{root_indicator}]"
        else:
            root_path_str = "  [No Path]"

        lines.append(f"{self.name}{root_path_str}")
        build_tree_lines(self, prefix="", base=base_path_for_display)
        return "\n".join(lines)


def _extract_module_members(
    file_path: Path,
) -> Tuple[Dict[str, Tuple[Path, Optional[int]]], List[NamespaceCollision], Optional[List[str]]]:
    """
    Parses a Python file and extracts top-level member definitions/assignments,
    intra-file hard collisions, and the contents of __all__ if present.

    Returns:
    - A dictionary mapping defined/assigned names to a tuple of (file path, line number).
    - A list of intra-file hard collisions (NamespaceCollision objects).
    - A list of names in __all__, or None if __all__ is not found or invalid.

    vibelint/namespace.py
    """

    try:
        source = file_path.read_text(encoding="utf-8")

        tree = ast.parse(source, filename=str(file_path))
    except Exception as e:
        logger.warning(f"Could not parse file {file_path} for namespace analysis: {e}")

        return {}, [], None

    defined_members_map: Dict[str, Tuple[Path, Optional[int]]] = {}
    collisions: List[NamespaceCollision] = []
    exported_names: Optional[List[str]] = None

    defined_names_nodes: Dict[str, ast.AST] = {}

    for node in tree.body:
        current_node = node
        name: Optional[str] = None
        is_definition = False
        is_all_assignment = False
        lineno = getattr(current_node, "lineno", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            is_definition = True
        elif isinstance(node, ast.Assign):

            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                is_all_assignment = True

                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exported_names = []
                    for elt in node.value.elts:

                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported_names.append(elt.value)

                if "__all__" not in defined_names_nodes:
                    defined_names_nodes["__all__"] = current_node
                else:
                    first_node = defined_names_nodes["__all__"]
                    collisions.append(
                        NamespaceCollision(
                            name="__all__",
                            collision_type=CollisionType.HARD,
                            paths=[file_path, file_path],
                            linenos=[getattr(first_node, "lineno", None), lineno],
                        )
                    )

            else:

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        is_definition = True

                        if name:
                            if name in defined_names_nodes:

                                first_node = defined_names_nodes[name]
                                collisions.append(
                                    NamespaceCollision(
                                        name=name,
                                        collision_type=CollisionType.HARD,
                                        paths=[file_path, file_path],
                                        linenos=[
                                            getattr(first_node, "lineno", None),
                                            lineno,
                                        ],
                                    )
                                )
                            else:

                                defined_names_nodes[name] = current_node
                                defined_members_map[name] = (
                                    file_path,
                                    lineno,
                                )
                            name = None

        if name and is_definition and not is_all_assignment:
            if name in defined_names_nodes:

                first_node = defined_names_nodes[name]
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[file_path, file_path],
                        linenos=[getattr(first_node, "lineno", None), lineno],
                    )
                )
            else:

                defined_names_nodes[name] = current_node
                defined_members_map[name] = (file_path, lineno)

    return defined_members_map, collisions, exported_names


def build_namespace_tree(
    paths: List[Path], config: Config
) -> Tuple[NamespaceNode, List[NamespaceCollision]]:
    """
    Builds the namespace tree, collects intra-file collisions, and stores members/__all__.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns a tuple: (root_node, all_intra_file_collisions)

    vibelint/namespace.py
    """

    project_root_found = config.project_root or find_project_root(
        paths[0].resolve() if paths else Path(".")
    )
    if not project_root_found:

        project_root_found = Path(".")
        root_node_name = "root"
        logger.warning(
            "Could not determine project root. Using '.' as root for namespace analysis."
        )
    else:
        root_node_name = project_root_found.name

    root = NamespaceNode(root_node_name, path=project_root_found.resolve(), is_package=True)
    root_path_for_rel = project_root_found.resolve()
    all_intra_file_collisions: List[NamespaceCollision] = []

    python_files = [
        f
        for f in discover_files(
            paths,
            config,
        )
        if f.suffix == ".py"
    ]

    if not python_files:
        logger.info("No Python files found for namespace analysis based on configuration.")
        return root, all_intra_file_collisions

    for f in python_files:
        try:

            rel_path = f.relative_to(root_path_for_rel)
            rel_parts = list(rel_path.parts)
        except ValueError:

            rel_parts = [f.name]
            logger.warning(
                f"File {f} is outside the determined project root {root_path_for_rel}. Adding directly under root."
            )

        current = root

        for i, part in enumerate(rel_parts[:-1]):

            dir_path = root_path_for_rel.joinpath(*rel_parts[: i + 1])
            current = current.add_child(part, dir_path, is_package=True)

        file_name = rel_parts[-1]
        mod_name = Path(file_name).stem
        file_abs_path = f

        members, intra_collisions, exported_names = _extract_module_members(file_abs_path)
        all_intra_file_collisions.extend(intra_collisions)

        if mod_name == "__init__":

            package_node = current
            package_node.is_package = True
            package_node.path = file_abs_path.parent

            for m_name, m_info in members.items():
                if m_name not in package_node.members:
                    package_node.members[m_name] = m_info

            if exported_names is not None:
                package_node.set_exported_names(exported_names)

        else:

            module_node = current.add_child(mod_name, file_abs_path, is_package=False)
            module_node.members = members
            if exported_names is not None:
                module_node.set_exported_names(exported_names)
            module_node.member_collisions.extend(intra_collisions)

    return root, all_intra_file_collisions
```

---
### File: src/vibelint/report.py

```python
"""
Report generation functionality for vibelint.

vibelint/report.py
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Set, TextIO

from .config import Config
from .lint import LintResult
from .namespace import NamespaceCollision, NamespaceNode
from .utils import get_relative_path

__all__ = ["write_report_content"]
logger = logging.getLogger(__name__)


def _get_files_in_namespace_order(
    node: NamespaceNode, collected_files: Set[Path], project_root: Path
) -> None:
    """
    Recursively collects file paths from the namespace tree in DFS order,
    including __init__.py files for packages. Populates the collected_files set.

    Args:
        node: The current NamespaceNode.
        collected_files: A set to store the absolute paths of collected files.
        project_root: The project root path for checking containment.

    vibelint/report.py
    """

    if node.is_package and node.path and node.path.is_dir():
        try:

            node.path.relative_to(project_root)
            init_file = node.path / "__init__.py"

            if init_file.is_file() and init_file not in collected_files:

                init_file.relative_to(project_root)
                logger.debug(f"Report: Adding package init file: {init_file}")
                collected_files.add(init_file)
        except ValueError:
            logger.warning(f"Report: Skipping package node outside project root: {node.path}")
        except Exception as e:
            logger.error(f"Report: Error checking package init file for {node.path}: {e}")

    for child_name in sorted(node.children.keys()):
        child_node = node.children[child_name]

        if child_node.path and child_node.path.is_file() and not child_node.is_package:
            try:

                child_node.path.relative_to(project_root)
                if child_node.path not in collected_files:
                    logger.debug(f"Report: Adding module file: {child_node.path}")
                    collected_files.add(child_node.path)
            except ValueError:
                logger.warning(
                    f"Report: Skipping module file outside project root: {child_node.path}"
                )
            except Exception as e:
                logger.error(f"Report: Error checking module file {child_node.path}: {e}")

        _get_files_in_namespace_order(child_node, collected_files, project_root)

    if not node.children and node.path and node.path.is_file():
        try:
            node.path.relative_to(project_root)
            if node.path not in collected_files:
                logger.debug(f"Report: Adding root file node: {node.path}")
                collected_files.add(node.path)
        except ValueError:
            logger.warning(f"Report: Skipping root file node outside project root: {node.path}")
        except Exception as e:
            logger.error(f"Report: Error checking root file node {node.path}: {e}")


def write_report_content(
    f: TextIO,
    project_root: Path,
    target_paths: List[Path],
    lint_results: List[LintResult],
    hard_coll: List[NamespaceCollision],
    soft_coll: List[NamespaceCollision],
    root_node: NamespaceNode,
    config: Config,
) -> None:
    """
    Writes the comprehensive markdown report content to the given file handle.

    Args:
    f: The text file handle to write the report to.
    project_root: The root directory of the project.
    target_paths: List of paths that were analyzed.
    lint_results: List of LintResult objects from the linting phase.
    hard_coll: List of hard NamespaceCollision objects.
    soft_coll: List of definition/export (soft) NamespaceCollision objects.
    root_node: The root NamespaceNode of the project structure.
    config: Configuration object.

    vibelint/report.py
    """

    package_name = project_root.name if project_root else "Unknown"

    f.write("# vibelint Report\n\n")
    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    f.write(f"**Project:** {package_name}\n")

    f.write(f"**Project Root:** `{str(project_root.resolve())}`\n\n")
    f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")

    f.write("## Table of Contents\n\n")
    f.write("1. [Summary](#summary)\n")
    f.write("2. [Linting Results](#linting-results)\n")
    f.write("3. [Namespace Structure](#namespace-structure)\n")
    f.write("4. [Namespace Collisions](#namespace-collisions)\n")
    f.write("5. [File Contents](#file-contents)\n\n")

    f.write("## Summary\n\n")
    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")

    files_analyzed_count = len(lint_results)
    f.write(f"| Files analyzed | {files_analyzed_count} |\n")
    f.write(f"| Files with errors | {sum(1 for r in lint_results if r.errors)} |\n")
    f.write(
        f"| Files with warnings only | {sum(1 for r in lint_results if r.warnings and not r.errors)} |\n"
    )
    f.write(f"| Hard namespace collisions | {len(hard_coll)} |\n")
    total_soft_collisions = len(soft_coll)
    f.write(f"| Definition/Export namespace collisions | {total_soft_collisions} |\n\n")

    f.write("## Linting Results\n\n")

    sorted_lint_results = sorted(lint_results, key=lambda r: r.file_path)
    files_with_issues = [r for r in sorted_lint_results if r.has_issues]

    if not files_with_issues:
        f.write("*No linting issues found.*\n\n")
    else:
        f.write("| File | Errors | Warnings |\n")
        f.write("|------|--------|----------|\n")
        for result in files_with_issues:

            errors_str = (
                "; ".join(f"`[{code}]` {msg}" for code, msg in result.errors)
                if result.errors
                else "None"
            )
            warnings_str = (
                "; ".join(f"`[{code}]` {msg}" for code, msg in result.warnings)
                if result.warnings
                else "None"
            )
            try:

                rel_path = get_relative_path(result.file_path.resolve(), project_root.resolve())
            except ValueError:
                rel_path = result.file_path

            f.write(f"| `{rel_path}` | {errors_str} | {warnings_str} |\n")
        f.write("\n")

    f.write("## Namespace Structure\n\n")
    f.write("```\n")
    try:

        tree_str = root_node.__str__()
        f.write(tree_str)
    except Exception as e:
        logger.error(f"Report: Error generating namespace tree string: {e}")
        f.write(f"[Error generating namespace tree: {e}]\n")
    f.write("\n```\n\n")

    f.write("## Namespace Collisions\n\n")
    f.write("### Hard Collisions\n\n")
    if not hard_coll:
        f.write("*No hard collisions detected.*\n\n")
    else:
        f.write("These collisions can break Python imports or indicate duplicate definitions:\n\n")
        f.write("| Name | Path 1 | Path 2 | Details |\n")
        f.write("|------|--------|--------|---------|\n")
        for collision in sorted(hard_coll, key=lambda c: (c.name, str(c.path1))):
            try:
                p1_rel = (
                    get_relative_path(collision.path1.resolve(), project_root.resolve())
                    if collision.path1
                    else "N/A"
                )
                p2_rel = (
                    get_relative_path(collision.path2.resolve(), project_root.resolve())
                    if collision.path2
                    else "N/A"
                )
            except ValueError:
                p1_rel = collision.path1 or "N/A"
                p2_rel = collision.path2 or "N/A"
            loc1 = f":{collision.lineno1}" if collision.lineno1 else ""
            loc2 = f":{collision.lineno2}" if collision.lineno2 else ""
            details = (
                "Intra-file duplicate" if str(p1_rel) == str(p2_rel) else "Module/Member clash"
            )
            f.write(f"| `{collision.name}` | `{p1_rel}{loc1}` | `{p2_rel}{loc2}` | {details} |\n")
        f.write("\n")

    f.write("### Definition & Export Collisions (Soft)\n\n")
    if not soft_coll:
        f.write("*No definition or export collisions detected.*\n\n")
    else:
        f.write(
            "These names are defined/exported in multiple files, which may confuse humans and LLMs:\n\n"
        )
        f.write("| Name | Type | Files Involved |\n")
        f.write("|------|------|----------------|\n")
        grouped_soft = defaultdict(lambda: {"paths": set(), "types": set()})
        for collision in soft_coll:
            all_paths = collision.definition_paths or [collision.path1, collision.path2]
            grouped_soft[collision.name]["paths"].update(p for p in all_paths if p)
            grouped_soft[collision.name]["types"].add(collision.collision_type)

        for name, data in sorted(grouped_soft.items()):
            paths_str_list = []
            for p in sorted(list(data["paths"]), key=str):
                try:
                    paths_str_list.append(
                        f"`{get_relative_path(p.resolve(), project_root.resolve())}`"
                    )
                except ValueError:
                    paths_str_list.append(f"`{p}`")
            type_str = (
                " & ".join(sorted([t.replace("_soft", "").upper() for t in data["types"]]))
                or "Unknown"
            )
            f.write(f"| `{name}` | {type_str} | {', '.join(paths_str_list)} |\n")
        f.write("\n")

    f.write("## File Contents\n\n")
    f.write("Files are ordered alphabetically by path.\n\n")

    collected_files_set: Set[Path] = set()
    try:
        _get_files_in_namespace_order(root_node, collected_files_set, project_root.resolve())

        python_files_abs = sorted(list(collected_files_set), key=lambda p: str(p))
        logger.info(f"Report: Found {len(python_files_abs)} files for content section.")
    except Exception as e:
        logger.error(f"Report: Error collecting files for content section: {e}", exc_info=True)
        python_files_abs = []

    if not python_files_abs:
        f.write("*No Python files found in the namespace tree to display.*\n\n")
    else:
        for abs_file_path in python_files_abs:

            if abs_file_path and abs_file_path.is_file():
                try:

                    rel_path = get_relative_path(abs_file_path, project_root.resolve())
                    f.write(f"### {rel_path}\n\n")

                    try:
                        lang = "python"
                        content = abs_file_path.read_text(encoding="utf-8", errors="ignore")
                        f.write(f"```{lang}\n")
                        f.write(content)

                        if not content.endswith("\n"):
                            f.write("\n")
                        f.write("```\n\n")
                    except Exception as read_e:
                        logger.warning(
                            f"Report: Error reading file content for {rel_path}: {read_e}"
                        )
                        f.write(f"*Error reading file content: {read_e}*\n\n")

                except ValueError:

                    logger.warning(
                        f"Report: Skipping file outside project root in content section: {abs_file_path}"
                    )
                    f.write(f"### {abs_file_path} (Outside Project Root)\n\n")
                    f.write("*Skipping content as file is outside the detected project root.*\n\n")
                except Exception as e_outer:
                    logger.error(
                        f"Report: Error processing file entry for {abs_file_path}: {e_outer}",
                        exc_info=True,
                    )
                    f.write(f"### Error Processing Entry for {abs_file_path}\n\n")
                    f.write(f"*An unexpected error occurred: {e_outer}*\n\n")
            elif abs_file_path:
                logger.warning(
                    f"Report: Skipping non-file path found during content writing: {abs_file_path}"
                )
                f.write(f"### {abs_file_path} (Not a File)\n\n")
                f.write("*Skipping entry as it is not a file.*\n\n")

            f.write("---\n\n")
```

---
### File: src/vibelint/results.py

```python
"""
Module for vibelint/results.py.

vibelint/results.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .lint import LintResult
from .namespace import NamespaceCollision, NamespaceNode

__all__ = ["CheckResult", "CommandResult", "NamespaceResult", "SnapshotResult"]


@dataclass
class CommandResult:
    """
    Base class for command results.

    vibelint/results.py
    """

    success: bool = True
    error_message: Optional[str] = None
    exit_code: int = 0

    def __post_init__(self):
        """
        Set exit code based on success if not explicitly set.

        vibelint/results.py
        """

        if not self.success and self.exit_code == 0:
            self.exit_code = 1


@dataclass
class CheckResult(CommandResult):
    """
    Result data from the 'check' command.

    vibelint/results.py
    """

    lint_results: List[LintResult] = field(default_factory=list)
    hard_collisions: List[NamespaceCollision] = field(default_factory=list)
    global_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    local_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    report_path: Optional[Path] = None
    report_generated: bool = False
    report_error: Optional[str] = None


@dataclass
class NamespaceResult(CommandResult):
    """
    Result data from the 'namespace' command.

    vibelint/results.py
    """

    root_node: Optional[NamespaceNode] = None
    intra_file_collisions: List[NamespaceCollision] = field(default_factory=list)
    output_path: Optional[Path] = None
    output_saved: bool = False
    output_error: Optional[str] = None


@dataclass
class SnapshotResult(CommandResult):
    """
    Result data from the 'snapshot' command.

    vibelint/results.py
    """

    output_path: Optional[Path] = None
```

---
### File: src/vibelint/snapshot.py

```python
"""
Codebase snapshot generation in markdown format.

vibelint/snapshot.py
"""

import fnmatch
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import Config
from .discovery import discover_files
from .utils import get_relative_path, is_binary

__all__ = ["create_snapshot"]

logger = logging.getLogger(__name__)


def create_snapshot(
    output_path: Path,
    target_paths: List[Path],
    config: Config,
):
    """
    Creates a Markdown snapshot file containing the project structure and file contents,
    respecting the include/exclude rules defined in pyproject.toml.

    Args:
    output_path: The path where the Markdown file will be saved.
    target_paths: List of initial paths (files or directories) to discover from.
    config: The vibelint configuration object.

    vibelint/snapshot.py
    """

    assert config.project_root is not None, "Project root must be set before creating snapshot."
    project_root = config.project_root.resolve()

    absolute_output_path = output_path.resolve()

    logger.debug("create_snapshot: Running discovery based on pyproject.toml config...")

    discovered_files = discover_files(
        paths=target_paths,
        config=config,
        explicit_exclude_paths={absolute_output_path},
    )

    logger.debug(f"create_snapshot: Discovery finished, count: {len(discovered_files)}")

    for excluded_pattern_root in [".pytest_cache", ".ruff_cache", ".git"]:
        present = any(excluded_pattern_root in str(f) for f in discovered_files)

        logger.debug(
            "!!! Check @ start of create_snapshot: '{}' presence in list: {}".format(
                excluded_pattern_root, present
            )
        )

    file_infos: List[Tuple[Path, str]] = []

    peek_globs = config.get("peek_globs", [])
    if not isinstance(peek_globs, list):
        logger.warning("Configuration 'peek_globs' is not a list. Ignoring peek rules.")
        peek_globs = []

    for abs_file_path in discovered_files:
        try:

            rel_path_str = get_relative_path(abs_file_path, project_root)
        except ValueError:

            logger.warning(
                f"Skipping file outside project root during snapshot categorization: {abs_file_path}"
            )
            continue

        if is_binary(abs_file_path):
            cat = "BINARY"
        else:
            cat = "FULL"
            for pk in peek_globs:

                normalized_rel_path = str(rel_path_str).replace("\\", "/")

                normalized_peek_glob = pk.replace("\\", "/")
                if fnmatch.fnmatch(normalized_rel_path, normalized_peek_glob):
                    cat = "PEEK"
                    break
        file_infos.append((abs_file_path, cat))
        logger.debug(f"Categorized {rel_path_str} as {cat}")

    file_infos.sort(key=lambda x: x[0])

    logger.debug(f"Sorted {len(file_infos)} files for snapshot.")

    tree: Dict = {}
    for f_path, f_cat in file_infos:
        try:

            relative_parts = str(get_relative_path(f_path, project_root)).split("/")
        except ValueError:

            logger.warning(
                f"Skipping file outside project root during snapshot tree build: {f_path}"
            )
            continue

        node = tree
        for i, part in enumerate(relative_parts):
            if not part:
                continue
            if i == len(relative_parts) - 1:

                if "__FILES__" not in node:
                    node["__FILES__"] = []

                node["__FILES__"].append((f_path, f_cat))
            else:

                if part not in node:
                    node[part] = {}
                node = node[part]

    logger.info(f"Writing snapshot to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:

        with open(absolute_output_path, "w", encoding="utf-8") as outfile:

            outfile.write("# Snapshot\n\n")

            outfile.write("## Filesystem Tree\n\n```\n")
            tree_root_name = project_root.name if project_root.name else str(project_root)
            outfile.write(f"{tree_root_name}/\n")
            _write_tree(outfile, tree, "")
            outfile.write("```\n\n")

            outfile.write("## File Contents\n\n")
            outfile.write("Files are ordered alphabetically by path.\n\n")
            for f, cat in file_infos:
                try:
                    relpath_header = get_relative_path(f, project_root)
                    outfile.write(f"### File: {relpath_header}\n\n")
                    logger.debug(f"Writing content for {relpath_header} (Category: {cat})")

                    if cat == "BINARY":
                        outfile.write("```\n")
                        outfile.write("[Binary File - Content not displayed]\n")
                        outfile.write("```\n\n---\n")
                    elif cat == "PEEK":
                        outfile.write("```\n")
                        outfile.write("[PEEK - Content truncated]\n")
                        try:
                            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                                lines_read = 0
                                for line in infile:
                                    if lines_read >= 10:
                                        outfile.write("...\n")
                                        break
                                    outfile.write(line)
                                    lines_read += 1
                        except Exception as e:
                            logger.warning(f"Error reading file for peek {relpath_header}: {e}")
                            outfile.write(f"[Error reading file for peek: {e}]\n")
                        outfile.write("```\n\n---\n")
                    else:
                        lang = _get_language(f)
                        outfile.write(f"```{lang}\n")
                        try:
                            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                                content = infile.read()
                                if not content.endswith("\n"):
                                    content += "\n"
                                outfile.write(content)
                        except Exception as e:
                            logger.warning(f"Error reading file content {relpath_header}: {e}")
                            outfile.write(f"[Error reading file: {e}]\n")
                        outfile.write("```\n\n---\n")

                except Exception as e:
                    try:
                        relpath_header = get_relative_path(f, project_root)
                    except Exception:
                        relpath_header = str(f)

                    logger.error(
                        f"Error processing file entry for {relpath_header} in snapshot: {e}",
                        exc_info=True,
                    )
                    outfile.write(f"### File: {relpath_header} (Error)\n\n")
                    outfile.write(f"[Error processing file entry: {e}]\n\n---\n")

            outfile.write("\n")

    except IOError as e:

        logger.error(f"Failed to write snapshot file {absolute_output_path}: {e}", exc_info=True)
        raise
    except Exception as e:

        logger.error(f"An unexpected error occurred during snapshot writing: {e}", exc_info=True)
        raise


def _write_tree(outfile, node: Dict, prefix=""):
    """
    Helper function to recursively write the directory tree.

    vibelint/snapshot.py
    """

    dirs = sorted([k for k in node if k != "__FILES__"])

    files_data: List[Tuple[Path, str]] = sorted(node.get("__FILES__", []), key=lambda x: x[0].name)

    entries = dirs + [f[0].name for f in files_data]

    for i, name in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        outfile.write(f"{prefix}{connector}")

        if name in dirs:

            outfile.write(f"{name}/\n")
            new_prefix = prefix + ("    " if is_last else "│   ")
            _write_tree(outfile, node[name], new_prefix)
        else:

            file_info_tuple = next((info for info in files_data if info[0].name == name), None)
            file_cat = "FULL"
            if file_info_tuple:
                file_cat = file_info_tuple[1]

            peek_indicator = " (PEEK)" if file_cat == "PEEK" else ""
            binary_indicator = " (BINARY)" if file_cat == "BINARY" else ""
            outfile.write(f"{name}{peek_indicator}{binary_indicator}\n")


def _get_language(file_path: Path) -> str:
    """
    Guess language for syntax highlighting based on extension.

    vibelint/snapshot.py
    """

    ext = file_path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".ps1": "powershell",
        ".bat": "batch",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".gitignore": "gitignore",
    }
    return mapping.get(ext, "")
```

---
### File: src/vibelint/utils.py

```python
"""
Utility functions for vibelint.

vibelint/utils.py
"""

import os
from pathlib import Path
from typing import List, Optional

__all__ = [
    "ensure_directory",
    "find_files_by_extension",
    "find_package_root",
    "get_import_path",
    "get_module_name",
    "get_relative_path",
    "is_python_file",
    "read_file_safe",
    "write_file_safe",
    "find_project_root",
    "is_binary",
]


def find_project_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a project containing the given path.

    A project root is identified by containing either:
    1. A pyproject.toml file
    2. A .git directory

    Args:
    start_path: Path to start the search from

    Returns:
    Path to project root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    while True:
        if (current_path / "pyproject.toml").is_file():
            return current_path
        if (current_path / ".git").is_dir():
            return current_path
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def find_package_root(start_path: Path) -> Optional[Path]:
    """
    Find the root directory of a Python package containing the given path.

    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A  file
    3. An  file at the top level with no parent

    Args:
    start_path: Path to start the search from

    Returns:
    Path to package root, or None if not found

    vibelint/utils.py
    """

    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        if (current_path / "__init__.py").is_file():
            project_root_marker = find_project_root(current_path)
            if project_root_marker and current_path.is_relative_to(project_root_marker):
                pass

        if (current_path / "pyproject.toml").is_file() or (current_path / ".git").is_dir():
            src_dir = current_path / "src"
            if src_dir.is_dir():
                if start_path.resolve().is_relative_to(src_dir):
                    for item in src_dir.iterdir():
                        if item.is_dir() and (item / "__init__.py").is_file():
                            return item
                    return src_dir
                else:
                    if (current_path / "__init__.py").is_file():
                        return current_path

            if (current_path / "__init__.py").is_file():
                return current_path
            return current_path

        if current_path.parent == current_path:
            return start_path.parent if start_path.is_file() else start_path

        current_path = current_path.parent


def is_python_file(path: Path) -> bool:
    """
    Check if a path represents a Python file.

    Args:
    path: Path to check

    Returns:
    True if the path is a Python file, False otherwise

    vibelint/utils.py
    """

    return path.is_file() and path.suffix == ".py"


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Safely compute a relative path, falling back to the original path.

    vibelint/utils.py
    """

    try:

        return path.resolve().relative_to(base.resolve())
    except ValueError:

        return path.resolve()


def get_import_path(file_path: Path, package_root: Optional[Path] = None) -> str:
    """
    Get the import path for a Python file.

    Args:
    file_path: Path to the Python file
    package_root: Optional path to the package root

    Returns:
    Import path (e.g., "vibelint.utils")

    vibelint/utils.py
    """

    if package_root is None:
        package_root = find_package_root(file_path)

    if package_root is None:
        return file_path.stem

    try:
        rel_path = file_path.relative_to(package_root)
        import_path = str(rel_path).replace(os.sep, ".").replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]
        return import_path
    except ValueError:
        return file_path.stem


def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.

    Args:
    file_path: Path to a Python file

    Returns:
    Module name

    vibelint/utils.py
    """

    return file_path.stem


def find_files_by_extension(
    root_path: Path,
    extension: str = ".py",
    exclude_globs: List[str] = [],
    include_vcs_hooks: bool = False,
) -> List[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.

    Args:
    root_path: Root path to search in
    extension: File extension to look for (including the dot)
    exclude_globs: Glob patterns to exclude
    include_vcs_hooks: Whether to include version control directories

    Returns:
    List of paths to files with the specified extension

    vibelint/utils.py
    """

    import fnmatch

    if exclude_globs is None:
        exclude_globs = []

    result = []

    for file_path in root_path.glob(f"**/*{extension}"):
        if not include_vcs_hooks:
            if any(
                part.startswith(".") and part in {".git", ".hg", ".svn"} for part in file_path.parts
            ):
                continue

        if any(fnmatch.fnmatch(str(file_path), pattern) for pattern in exclude_globs):
            continue

        result.append(file_path)

    return result


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
    path: Path to directory

    Returns:
    Path to the directory

    vibelint/utils.py
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
    """
    Safely read a file, returning None if any errors occur.

    Args:
    file_path: Path to file
    encoding: File encoding

    Returns:
    File contents or None if error

    vibelint/utils.py
    """

    try:
        return file_path.read_text(encoding=encoding)
    except Exception:
        return None


def write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> bool:
    """
    Safely write content to a file, returning success status.

    Args:
    file_path: Path to file
    content: Content to write
    encoding: File encoding

    Returns:
    True if successful, False otherwise

    vibelint/utils.py
    """

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except Exception:
        return False


def is_binary(file_path: Path, chunk_size: int = 1024) -> bool:
    """
    Check if a file appears to be binary by looking for null bytes
    or a high proportion of non-text bytes in the first chunk.

    Args:
    file_path: The path to the file.
    chunk_size: The number of bytes to read from the beginning.

    Returns:
    True if the file seems binary, False otherwise.

    vibelint/utils.py
    """

    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
        if not chunk:
            return False

        if b"\x00" in chunk:
            return True

        text_characters = bytes(range(32, 127)) + b"\n\r\t\f\b"
        non_text_count = sum(1 for byte in chunk if bytes([byte]) not in text_characters)

        if len(chunk) > 0 and (non_text_count / len(chunk)) > 0.3:
            return True

        return False
    except OSError:

        return True
    except Exception:

        return True
```

---
### File: src/vibelint/validators/__init__.py

```python
"""
vibelint validators sub-package.

Re-exports key classes and functions for easier access.

vibelint/validators/__init__.py
"""

from .docstring import DocstringValidationResult, get_normalized_filepath, validate_every_docstring
from .encoding import EncodingValidationResult, validate_encoding_cookie
from .exports import ExportValidationResult, validate_exports
from .shebang import ShebangValidationResult, file_contains_top_level_main_block, validate_shebang

__all__ = [
    "DocstringValidationResult",
    "validate_every_docstring",
    "get_normalized_filepath",
    "EncodingValidationResult",
    "validate_encoding_cookie",
    "ExportValidationResult",
    "validate_exports",
    "ShebangValidationResult",
    "validate_shebang",
    "file_contains_top_level_main_block",
]
```

---
### File: src/vibelint/validators/docstring.py

```python
"""
Validator for Python docstrings. Checks for presence and path reference.

vibelint/validators/docstring.py
"""

import logging
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import libcst as cst
from libcst import (
    BaseSmallStatement,
    BaseStatement,
    ClassDef,
    Comment,
    CSTNode,
    EmptyLine,
    Expr,
    FunctionDef,
    IndentedBlock,
    Module,
    Pass,
    SimpleStatementLine,
    SimpleString,
)
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
)

from ..error_codes import VBL101, VBL102

logger = logging.getLogger(__name__)


__all__ = [
    "DocstringValidationResult",
    "get_normalized_filepath",
    "validate_every_docstring",
]

IssueKey = int
BodyItem = Union[BaseStatement, BaseSmallStatement, EmptyLine, Comment]
ValidationIssue = Tuple[str, str]


def _get_docstring_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleStatementLine]:
    """
    Attempts to get the CST node representing the docstring from a sequence of body statements.
    Searches for the first non-comment/empty statement and checks if it's a SimpleString expression.

    vibelint/validators/docstring.py
    """

    first_real_stmt = None
    for stmt in body_stmts:
        if not isinstance(stmt, (EmptyLine, Comment)):
            first_real_stmt = stmt
            break

    if (
        first_real_stmt
        and isinstance(first_real_stmt, SimpleStatementLine)
        and len(first_real_stmt.body) == 1
        and isinstance(first_real_stmt.body[0], Expr)
        and isinstance(first_real_stmt.body[0].value, SimpleString)
    ):
        return first_real_stmt
    return None


def _get_simple_string_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleString]:
    """
    Gets the SimpleString node if it's the first statement.

    vibelint/validators/docstring.py
    """

    doc_stmt_line = _get_docstring_node(body_stmts)
    if doc_stmt_line:
        try:
            expr_node = doc_stmt_line.body[0]
            if isinstance(expr_node, Expr) and isinstance(expr_node.value, SimpleString):
                return expr_node.value
        except (IndexError, AttributeError):
            pass
    return None


def _extract_docstring_text(node: Optional[SimpleStatementLine]) -> Optional[str]:
    """
    Extracts the interpreted string value from a docstring node.

    vibelint/validators/docstring.py
    """

    if node:
        try:
            expr_node = node.body[0]
            if isinstance(expr_node, Expr):
                str_node = expr_node.value
                if isinstance(str_node, SimpleString):
                    # Use evaluated_value which interprets escapes etc.
                    evaluated = str_node.evaluated_value
                    return evaluated if isinstance(evaluated, str) else None
        except (IndexError, AttributeError, Exception) as e:
            # Catch potential exceptions during evaluation if needed
            logger.debug(f"Failed to extract/evaluate SimpleString: {e}", exc_info=True)
            return None
    return None


def _get_docstring_node_index(body_stmts: Sequence[CSTNode]) -> Optional[int]:
    """
    Gets the index of the docstring node in a body list.

    vibelint/validators/docstring.py
    """

    for i, stmt in enumerate(body_stmts):
        # Skip initial comments and empty lines
        if isinstance(stmt, (EmptyLine, Comment)):
            continue

        # Check if the first non-empty/comment line is a string expression
        if (
            isinstance(stmt, SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], Expr)
            and isinstance(stmt.body[0].value, SimpleString)
        ):
            return i
        else:
            # If the first real statement isn't a docstring, there is no docstring
            return None

    return None


class DocstringValidationResult:
    """
    Stores the result of docstring validation.

    vibelint/validators/docstring.py
    """

    def __init__(self) -> None:
        """
        Initializes DocstringValidationResult.

        vibelint/validators/docstring.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    def has_issues(self) -> bool:
        """
        Checks if there are any errors or warnings.

        vibelint/validators/docstring.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/docstring.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/docstring.py
        """
        self.warnings.append((code, message))


def get_normalized_filepath(relative_path: str) -> str:
    """
    Normalizes a path for docstring references.
    Removes './', converts '' to '/', and removes leading 'src/'.

    vibelint/validators/docstring.py
    """
    # Normalize separators and remove leading './'
    path = relative_path.replace("\\", "/").lstrip("./")
    # Special handling for paths under 'src/'
    if path.startswith("src/"):
        return path[len("src/") :]
    return path


def get_node_start_line(
    node: CSTNode, metadata: Mapping[ProviderT, Mapping[CSTNode, object]]
) -> int:
    """
    Gets the 1-based start line number of a CST node using metadata.
    Returns 0 if position info is unavailable.

    vibelint/validators/docstring.py
    """
    try:
        pos_info = metadata.get(PositionProvider, {}).get(node)
        return pos_info.start.line if isinstance(pos_info, CodeRange) else 0
    except Exception:
        logger.debug(f"Failed to get start line for node {type(node)}", exc_info=True)
        return 0


class DocstringInfoExtractor(cst.CSTVisitor):
    """
    Visits CST nodes to extract docstring info and validate.

    vibelint/validators/docstring.py
    """

    METADATA_DEPENDENCIES = (
        PositionProvider,
        # WhitespaceInclusivePositionProvider, # No longer needed for VBL103
        ParentNodeProvider,
    )

    def __init__(self, relative_path: str):
        """
        Initializes DocstringInfoExtractor.

        vibelint/validators/docstring.py
        """
        super().__init__()
        self.relative_path = relative_path
        self.path_ref = get_normalized_filepath(relative_path)
        self.result = DocstringValidationResult()

        logger.debug(
            f"[Validator:{self.relative_path}] Initialized. Expecting path ref: '{self.path_ref}'"
        )

    def visit_Module(self, node: Module) -> None:
        """
        Visits Module node.

        vibelint/validators/docstring.py
        """
        doc_node = _get_docstring_node(node.body)
        doc_text = _extract_docstring_text(doc_node)
        self._validate_docstring(node, doc_node, doc_text, "module", "module")

    def leave_Module(self, node: Module) -> None:
        """
        Leaves Module node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def visit_ClassDef(self, node: ClassDef) -> bool:
        """
        Visits ClassDef node.

        vibelint/validators/docstring.py
        """
        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(node, doc_node, doc_text, "class", node.name.value)
        else:
            # Handle cases like `class Foo: pass` (no IndentedBlock)
            self._validate_docstring(node, None, None, "class", node.name.value)
        return True  # Continue traversal

    def leave_ClassDef(self, node: ClassDef) -> None:
        """
        Leaves ClassDef node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def visit_FunctionDef(self, node: FunctionDef) -> bool:
        """
        Visits FunctionDef node.

        vibelint/validators/docstring.py
        """
        parent = self.get_metadata(ParentNodeProvider, node)
        is_method = isinstance(parent, IndentedBlock) and isinstance(
            self.get_metadata(ParentNodeProvider, parent), ClassDef
        )
        node_type = "method" if is_method else "function"

        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(node, doc_node, doc_text, node_type, node.name.value)
        else:
            # Handle cases like `def foo(): pass` (no IndentedBlock)
            self._validate_docstring(node, None, None, node_type, node.name.value)
        return True  # Continue traversal

    def leave_FunctionDef(self, node: FunctionDef) -> None:
        """
        Leaves FunctionDef node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def _validate_docstring(
        self,
        node: Union[Module, ClassDef, FunctionDef],
        node_doc: Optional[SimpleStatementLine],
        text_doc: Optional[str],
        n_type: str,
        n_name: str,
    ) -> None:
        """
        Performs the validation logic for presence (VBL101) and path reference (VBL102).

        vibelint/validators/docstring.py
        """
        start_line = get_node_start_line(node, self.metadata)
        if start_line == 0:
            logger.warning(
                f"Could not get start line for {n_type} '{n_name}', skipping validation."
            )
            return

        doc_present = node_doc is not None

        # Special case: Ignore missing docstring for simple `__init__` methods
        # like `def __init__(self): pass` or `def __init__(self): super().__init__()`
        # This check remains relevant for VBL101.
        is_simple_init_or_pass = False
        if (
            n_name == "__init__"
            and n_type == "method"
            and isinstance(node, FunctionDef)
            and isinstance(node.body, IndentedBlock)
        ):
            non_empty_stmts = [s for s in node.body.body if not isinstance(s, (EmptyLine, Comment))]
            doc_node_in_body = _get_docstring_node(node.body.body)
            actual_code_stmts = [s for s in non_empty_stmts if s is not doc_node_in_body]
            # Check if the only statement is Pass or a simple super call
            if len(actual_code_stmts) == 1:
                stmt = actual_code_stmts[0]
                if isinstance(stmt, SimpleStatementLine):
                    if len(stmt.body) == 1 and isinstance(stmt.body[0], Pass):
                        is_simple_init_or_pass = True
                    # Add check for simple super().__init__() if desired
                    # elif isinstance(stmt.body[0], Expr) and ... check for super call ...:
                    #    is_simple_init_or_pass = True

        # VBL101: Check for presence
        if not doc_present:
            if not is_simple_init_or_pass:  # Don't warn for simple init/pass methods
                msg = f"Missing docstring for {n_type} '{n_name}'."
                self.result.add_error(VBL101, msg)
                logger.debug(
                    f"[Validator:{self.relative_path}] Added issue {VBL101} for line {start_line}: Missing docstring"
                )
            else:
                logger.debug(
                    f"[Validator:{self.relative_path}] Suppressed VBL101 for simple {n_type} '{n_name}' line {start_line}"
                )
            return  # No further checks if docstring is missing

        # --- VBL103 Check Removed ---

        # VBL102: Check for path reference (only if docstring is present)
        path_issue = False
        if text_doc is not None:
            stripped_text = text_doc.rstrip()  # Remove trailing whitespace before checking end
            if not stripped_text.endswith(self.path_ref):
                path_issue = True
        else:
            # If text couldn't be extracted (e.g., complex f-string), assume path is missing/wrong
            path_issue = True

        if path_issue:
            msg = f"Docstring for {n_type} '{n_name}' missing/incorrect path reference (expected '{self.path_ref}')."
            # Note: This is a WARNING, allowing users to ignore it via config if they disagree.
            self.result.add_warning(VBL102, msg)
            logger.debug(
                f"[Validator:{self.relative_path}] Added issue {VBL102} for line {start_line}: Path reference"
            )

        # Log success if no issues were added for this node
        if not path_issue:  # Only need to check path_issue now
            logger.debug(
                f"[Validator:{self.relative_path}] Validation OK for {n_type} '{n_name}' line {start_line}"
            )


def validate_every_docstring(
    content: str, relative_path: str
) -> Tuple[DocstringValidationResult, Optional[Module]]:
    """
    Parse source code and run the DocstringInfoExtractor visitor to validate all docstrings.

    Args:
    content: The source code as a string.
    relative_path: The relative path of the file (used for path refs).

    Returns:
    A tuple containing:
    - DocstringValidationResult object with found issues.
    - The parsed CST Module node (or None if parsing failed).

    Raises:
    SyntaxError: If LibCST encounters a parsing error, it's converted and re-raised.

    vibelint/validators/docstring.py
    """
    result = DocstringValidationResult()
    module = None
    try:
        module = cst.parse_module(content)
        # Ensure required metadata providers are resolved
        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
        required_providers = {
            PositionProvider,
            ParentNodeProvider,
        }  # Removed WhitespaceInclusivePositionProvider
        wrapper.resolve_many(required_providers)

        extractor = DocstringInfoExtractor(relative_path)
        wrapper.visit(extractor)
        logger.debug(
            f"[Validator:{relative_path}] Validation complete. Issues found: E={len(extractor.result.errors)}, W={len(extractor.result.warnings)}"
        )
        return extractor.result, module
    except cst.ParserSyntaxError as e:
        # Convert LibCST error to standard SyntaxError for better reporting upstream
        logger.warning(
            f"CST ParserSyntaxError in {relative_path} L{e.raw_line}:{e.raw_column}: {e.message}"
        )
        err = SyntaxError(e.message)
        err.lineno = e.raw_line
        err.offset = e.raw_column + 1 if e.raw_column is not None else None
        err.filename = relative_path
        try:
            # Attempt to get the line content for the error message
            err.text = content.splitlines()[e.raw_line - 1]
        except IndexError:
            err.text = None
        raise err from e
    except Exception as e:
        # Catch other potential errors during validation
        logger.error(f"Unexpected CST validation error {relative_path}: {e}", exc_info=True)
        # Report as an internal error
        result.add_error("VBL903", f"Internal validation error: {e}")
        return result, None
```

---
### File: src/vibelint/validators/encoding.py

```python
"""
Validator for Python encoding cookies.

vibelint/validators/encoding.py
"""

import re
from typing import List, Tuple

from ..error_codes import VBL201

__all__ = [
    "EncodingValidationResult",
    "validate_encoding_cookie",
]

ValidationIssue = Tuple[str, str]


class EncodingValidationResult:
    """
    Result of a validation for encoding cookies.

    vibelint/validators/encoding.py
    """

    def __init__(self) -> None:
        """
        Initializes EncodingValidationResult.

        vibelint/validators/encoding.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = -1

    def has_issues(self) -> bool:
        """
        Check if there are any issues.

        vibelint/validators/encoding.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/encoding.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/encoding.py
        """
        self.warnings.append((code, message))


def validate_encoding_cookie(content: str) -> EncodingValidationResult:
    """
    Validate the encoding cookie in a Python file.

    vibelint/validators/encoding.py
    """
    result = EncodingValidationResult()
    lines = content.splitlines()
    pattern = r"^# -\*- coding: (.+) -\*-$"

    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1

    if idx < len(lines):
        m = re.match(pattern, lines[idx])
        if m:
            enc = m.group(1).lower()
            result.line_number = idx
            if enc != "utf-8":
                msg = f"Invalid encoding cookie: '{enc}' found on line {idx + 1}, must be 'utf-8'."
                result.add_error(VBL201, msg)

    return result
```

---
### File: src/vibelint/validators/exports.py

```python
"""
Validator for __all__ exports in Python modules.

Checks for presence (optional for __init__.py) and correct format.

vibelint/validators/exports.py
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Config
from ..error_codes import VBL301, VBL302, VBL303, VBL304

__all__ = ["ExportValidationResult", "validate_exports"]

ValidationIssue = Tuple[str, str]


class ExportValidationResult:
    """
    Stores the result of __all__ validation for a single file.

    vibelint/validators/exports.py
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes ExportValidationResult.

        vibelint/validators/exports.py
        """
        self.file_path = file_path
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.has_all: bool = False

        self.all_lineno: Optional[int] = None

    def has_issues(self) -> bool:
        """
        Returns True if there are any errors or warnings.

        vibelint/validators/exports.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/exports.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/exports.py
        """
        self.warnings.append((code, message))


def validate_exports(
    source_code: str, file_path_str: str, config: Config
) -> ExportValidationResult:
    """
    Validates the presence and format of __all__ in the source code.

    Args:
    source_code: The source code of the Python file as a string.
    file_path_str: The path to the file (used for context, e.g., __init__.py).
    config: The loaded vibelint configuration.

    Returns:
    An ExportValidationResult object.

    vibelint/validators/exports.py
    """
    result = ExportValidationResult(file_path_str)
    file_path = Path(file_path_str)
    is_init_py = file_path.name == "__init__.py"

    try:
        tree = ast.parse(source_code, filename=file_path_str)
    except SyntaxError as e:
        result.add_error(VBL304, f"SyntaxError parsing file: {e}")
        return result

    found_all = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                found_all = True
                result.has_all = True
                result.all_lineno = node.lineno

                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    msg = f"__all__ is not assigned a List or Tuple (assigned {type(node.value).__name__}) near line {node.lineno}."
                    result.add_error(VBL303, msg)

                break

    if not found_all:
        error_on_init = config.get("error_on_missing_all_in_init", False)
        if is_init_py and not error_on_init:
            msg = f"Optional: __all__ definition not found in {file_path.name}."
            result.add_warning(VBL302, msg)
        elif not is_init_py:
            msg = f"__all__ definition not found in {file_path.name}."
            result.add_error(VBL301, msg)
        elif is_init_py and error_on_init:
            msg = f"__all__ definition not found in {file_path.name} (required by config)."
            result.add_error(VBL301, msg)

    return result
```

---
### File: src/vibelint/validators/shebang.py

```python
"""
Validator for Python shebang lines.

vibelint/validators/shebang.py
"""

import ast
from pathlib import Path
from typing import List, Tuple

from ..error_codes import VBL401, VBL402, VBL403

__all__ = [
    "ShebangValidationResult",
    "validate_shebang",
    "file_contains_top_level_main_block",
]

ValidationIssue = Tuple[str, str]


class ShebangValidationResult:
    """
    Result of a shebang validation.

    vibelint/validators/shebang.py
    """

    def __init__(self) -> None:
        """
        Initializes ShebangValidationResult.

        vibelint/validators/shebang.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.line_number: int = 0

    def has_issues(self) -> bool:
        """
        Check if any issues were found.

        vibelint/validators/shebang.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/shebang.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/shebang.py
        """
        self.warnings.append((code, message))


def validate_shebang(
    content: str, is_script: bool, allowed_shebangs: List[str]
) -> ShebangValidationResult:
    """
    Validate the shebang line if present; ensure it's correct for scripts with __main__.

    vibelint/validators/shebang.py
    """
    res = ShebangValidationResult()
    lines = content.splitlines()

    if lines and lines[0].startswith("#!"):
        sb = lines[0]
        res.line_number = 0
        if not is_script:
            msg = f"File has shebang '{sb}' but no 'if __name__ == \"__main__\":' block found."
            res.add_error(VBL401, msg)
        elif sb not in allowed_shebangs:
            allowed_str = ", ".join(f"'{s}'" for s in allowed_shebangs)
            msg = f"Invalid shebang '{sb}'. Allowed: {allowed_str}."
            res.add_error(VBL402, msg)
    else:
        if is_script:
            res.line_number = 0
            msg = "Script contains 'if __name__ == \"__main__\":' block but lacks a shebang line ('#!...')."
            res.add_warning(VBL403, msg)

    return res


def file_contains_top_level_main_block(file_path: Path, content: str) -> bool:
    """
    Checks if a Python file contains a top-level 'if __name__ == "__main__":' block using AST.
    Returns True if found, False otherwise (including syntax errors).

    vibelint/validators/shebang.py
    """

    try:
        tree = ast.parse(content, filename=str(file_path))
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and (
                        (
                            isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value == "__main__"
                        )
                        or (
                            isinstance(test.comparators[0], ast.Str)
                            and test.comparators[0].s == "__main__"
                        )
                    )
                ):
                    return True
    except (SyntaxError, Exception):

        return False
    return False
```

---
### File: tests/test_cli.py

```python
# tests/test_cli.py
"""
Baseline tests for the vibelint CLI interface.

tests/test_cli.py
"""
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from click.testing import CliRunner, Result

# Conditional TOML library import
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None

from vibelint import __version__
from vibelint.cli import cli  # Import the main Click group

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# --- Helper Functions ---


def clean_output(output: str) -> str:
    """Removes ANSI escape codes and strips leading/trailing whitespace from each line."""
    cleaned = re.sub(r"\x1b\[.*?m", "", output)  # Remove ANSI codes
    cleaned = re.sub(r"\r\n?", "\n", cleaned)  # Normalize line endings
    # Process line by line to strip, then rejoin. Filter removes empty lines.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]  # Filter empty lines
    return "\n".join(lines)


def assert_output_contains(result: Result, substring: str, msg: str = ""):
    """Asserts substring is in cleaned output."""
    cleaned = clean_output(result.output)
    assert (
        substring in cleaned
    ), f"{msg}\nSubstring '{substring}' not found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_does_not_contain(result: Result, substring: str, msg: str = ""):
    """Asserts substring is NOT in cleaned output."""
    cleaned = clean_output(result.output)
    assert (
        substring not in cleaned
    ), f"{msg}\nSubstring '{substring}' unexpectedly found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_matches(result: Result, pattern: str, msg: str = ""):
    """Asserts regex pattern matches cleaned output (multiline, dotall)."""
    cleaned = clean_output(result.output)
    # Using DOTALL means '.' matches newline characters as well
    # Using MULTILINE ensures ^/$ match line beginnings/ends if needed
    assert re.search(
        pattern, cleaned, re.MULTILINE | re.DOTALL
    ), f"{msg}\nPattern '{pattern}' not found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_does_not_match(result: Result, pattern: str, msg: str = ""):
    """Asserts regex pattern does NOT match cleaned output (multiline, dotall)."""
    cleaned = clean_output(result.output)
    assert not re.search(
        pattern, cleaned, re.MULTILINE | re.DOTALL
    ), f"{msg}\nPattern '{pattern}' unexpectedly found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


# --- Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner()


@pytest.fixture
def setup_test_project(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """
    Copies a fixture project structure (e.g., fixtures/check_success/myproject/*)
    into a temporary directory, changes the CWD to the identified project root
    within that temp structure, ensures pyproject.toml exists there, and yields the path.
    """
    fixture_name = request.param
    source_fixture_path = FIXTURES_DIR / fixture_name
    if not source_fixture_path.is_dir():
        pytest.fail(f"Fixture directory not found: {source_fixture_path}")

    # Target directory in tmp_path named after the fixture
    target_base_dir = tmp_path / fixture_name
    target_base_dir.mkdir(parents=True, exist_ok=True)

    # Copy the *contents* of the source fixture directory
    for item in source_fixture_path.iterdir():
        source_item = source_fixture_path / item.name
        target_item = target_base_dir / item.name
        if source_item.is_dir():
            # Ensure target directory exists before copying into it
            target_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            # Ensure target directory exists before copying file
            target_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_item, target_item)  # copy2 preserves metadata
    print(f"DEBUG: Copied contents of {source_fixture_path} to {target_base_dir}")

    # --- Determine the actual project root within the copied structure ---
    target_project_root = None
    if (target_base_dir / "pyproject.toml").is_file() or (target_base_dir / ".git").is_dir():
        target_project_root = target_base_dir
        print(f"DEBUG: Project marker found in target base directory: {target_project_root}")
    else:
        potential_project_dirs = [
            d for d in target_base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        found_marker_in_subdir = False
        if potential_project_dirs:
            for potential_dir in potential_project_dirs:
                if (potential_dir / "pyproject.toml").is_file() or (
                    potential_dir / ".git"
                ).is_dir():
                    target_project_root = potential_dir
                    print(f"DEBUG: Project marker found in subdirectory: {target_project_root}")
                    found_marker_in_subdir = True
                    break
            if not found_marker_in_subdir and len(potential_project_dirs) == 1:
                target_project_root = potential_project_dirs[0]
                print(
                    f"DEBUG: Assuming single subdirectory is project root (no marker found): {target_project_root}"
                )
        if target_project_root is None:
            target_project_root = target_base_dir
            print(
                f"DEBUG: WARNING - No definitive project root identified via markers or single subdir heuristic. Defaulting to base: {target_project_root}"
            )

    # --- Ensure pyproject.toml exists in the determined root ---
    pyproject_path_in_root = target_project_root / "pyproject.toml"
    if not pyproject_path_in_root.is_file():
        print(f"DEBUG: Creating dummy pyproject.toml in identified root {target_project_root}")
        pyproject_path_in_root.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the dummy toml is valid
        pyproject_path_in_root.write_text(
            "[tool.vibelint]\ninclude_globs = [] # Dummy\n", encoding="utf-8"
        )
    else:
        print(f"DEBUG: Found existing pyproject.toml in identified root: {pyproject_path_in_root}")
        # Ensure the existing file is valid TOML if modifying later
        try:
            with open(pyproject_path_in_root, "rb") as f:
                content = f.read()
                if content.strip():
                    tomllib.loads(content.decode("utf-8", errors="replace"))
        except Exception as e:
            print(
                f"DEBUG: Warning - Existing pyproject.toml at {pyproject_path_in_root} may be invalid: {e}"
            )
            # Overwrite with a basic valid one if parsing fails? Risky. Let modify handle it.
            pass

    # --- Change CWD to the determined project root and Yield ---
    original_cwd = Path.cwd()
    resolved_root = target_project_root.resolve()
    print(f"DEBUG: Original CWD: {original_cwd}")
    print(f"DEBUG: Changing CWD to identified project root: {resolved_root}")
    os.chdir(resolved_root)
    try:
        yield resolved_root
    finally:
        print(f"DEBUG: Restoring CWD to: {original_cwd}")
        os.chdir(original_cwd)


def modify_pyproject(project_path: Path, updates: Dict[str, Any]):
    """Modifies the [tool.vibelint] section of pyproject.toml."""
    if tomllib is None or tomli_w is None:
        pytest.skip("Skipping test: 'tomli'/'tomli-w' not available for modifying pyproject.toml.")

    pyproject_file = project_path / "pyproject.toml"
    if not pyproject_file.is_file():
        pytest.fail(f"pyproject.toml unexpectedly missing in {project_path} despite fixture logic.")

    try:
        with open(pyproject_file, "rb") as f:
            content = f.read()
            if not content.strip():
                data = {}
                print(f"DEBUG: Reading empty or whitespace-only {pyproject_file}")
            else:
                try:
                    data = tomllib.loads(content.decode("utf-8", errors="replace"))
                    print(f"DEBUG: Successfully loaded TOML from {pyproject_file}")
                except tomllib.TOMLDecodeError as e:
                    pytest.fail(
                        f"Failed to parse non-empty {pyproject_file}: {e}\nContent:\n>>>\n{content.decode('utf-8', errors='replace')}\n<<<"
                    )
    except Exception as e:
        pytest.fail(f"Failed to read {pyproject_file}: {e}")

    data.setdefault("tool", {}).setdefault("vibelint", {}).update(updates)

    try:
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(data, f)
        print(f"DEBUG: Successfully modified {pyproject_file} with updates: {updates}")
    except Exception as e:
        pytest.fail(f"Failed to write modified {pyproject_file}: {e}")


# --- Test Cases ---


def test_cli_version(runner: CliRunner):
    """Test the --version flag."""
    result = runner.invoke(cli, ["--version"], prog_name="vibelint", catch_exceptions=False)
    assert result.exit_code == 0
    assert f"vibelint, version {__version__}" in result.output


def test_cli_help(runner: CliRunner):
    """Test the --help flag."""
    result = runner.invoke(cli, ["--help"], prog_name="vibelint", catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "check" in result.output
    assert "namespace" in result.output
    assert "snapshot" in result.output
    assert "Vibe Check" in result.output
    assert "Visualize" in result.output
    assert "snapshot" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_cli_no_command_shows_art_and_help(runner: CliRunner, setup_test_project: Path):
    """Test running vibelint with no command shows ASCII art and help hint."""
    vibe_file = setup_test_project / "VIBECHECKER.txt"
    if not vibe_file.is_file():
        vibe_file.write_text(":-)", encoding="utf-8")

    result = runner.invoke(cli, [], catch_exceptions=False)
    assert result.exit_code == 0
    assert_output_matches(result, r"[^\s]")
    assert_output_contains(result, "Run vibelint --help for available commands.")
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." not in result.output


def test_cli_no_project_root(runner: CliRunner, tmp_path: Path):
    """Test CLI behavior when no project root (pyproject.toml/.git) is found."""
    original_cwd = Path.cwd()
    empty_dir = tmp_path / "empty_test_dir_no_root_v5"  # Use different name
    empty_dir.mkdir()
    os.chdir(empty_dir)
    try:
        result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
        print(f"Output (no project root):\n{result.output}")
        assert (
            result.exit_code == 1
        ), f"Expected exit code 1 when no root found, got {result.exit_code}"
        assert_output_contains(result, "Error: Could not find project root.")
        assert_output_contains(result, "pyproject.toml")
        assert_output_contains(result, ".git")
    finally:
        os.chdir(original_cwd)


# --- 'check' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_success(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check` on a project fixture expected to pass."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    print(f"DEBUG: CWD during test_check_success: {Path.cwd()}")
    print(f"DEBUG: setup_test_project path (yielded root): {setup_test_project}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Vibe Check Summary")
    assert_output_matches(
        result, r"(Immaculate vibes|Vibes confirmed|Vibe on brother)", msg="Missing success message"
    )
    assert_output_does_not_match(result, r"\[VBL\d{3}\]", msg="Unexpected VBL codes found")
    assert_output_does_not_contain(result, "Collision Summary", msg="Unexpected collision summary")


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_failure(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check` on a project fixture expected to fail."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["*.py"]})

    result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    print(f"DEBUG: CWD during test_check_failure: {Path.cwd()}")
    print(f"DEBUG: setup_test_project path (yielded root): {setup_test_project}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Vibe Check Summary")
    assert_output_matches(result, r"Vibe Check Failed", msg="Missing failure message")
    assert_output_matches(result, r"\[VBL101\]", msg="Missing VBL101 code")
    assert_output_matches(result, r"\[VBL102\]", msg="Missing VBL102 code")
    assert_output_matches(result, r"\[VBL301\]", msg="Missing VBL301 code")
    assert_output_contains(result, "VBL101] Missing docstring for function 'func_one'")
    assert_output_matches(
        result, r"VBL102\].*\(expected '[^']*?/?another\.py'\)", msg="Missing VBL102 for another.py"
    )
    assert_output_matches(
        result,
        r"VBL102\].*\(expected '[^']*?/?needs_fix\.py'\)",
        msg="Missing VBL102 for needs_fix.py",
    )
    assert_output_matches(
        result,
        r"VBL301] __all__ definition not found in [^']*/?needs_fix\.py",
        msg="Missing VBL301 message for needs_fix.py",
    )
    assert_output_does_not_contain(result, "Collision Summary", msg="Unexpected collision summary")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_output_report(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check -o report.md` generates a report."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    report_file = setup_test_project / "vibelint_report.md"
    assert not report_file.exists()

    result = runner.invoke(
        cli, ["--debug", "check", "-o", str(report_file)], catch_exceptions=False
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 for check, got {result.exit_code}. Output:\n{result.output}"
    assert report_file.is_file(), f"Report file was not created at {report_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Detailed Vibe Report generated at" in cleaned_output
    ), f"Report text missing. Output:\n{cleaned_output}"
    # Check if the filename exists somewhere in the cleaned output
    assert (
        report_file.name in cleaned_output
    ), f"Report filename missing. Filename: {report_file.name}\nOutput:\n{cleaned_output}"

    report_content = report_file.read_text()
    assert "# vibelint Report" in report_content
    assert "## Linting Results" in report_content
    assert "*No linting issues found.*" in report_content
    assert "## Namespace Structure" in report_content
    assert setup_test_project.name in report_content
    assert "hello (member)" in report_content
    assert "## Namespace Collisions" in report_content
    assert "*No hard collisions detected.*" in report_content
    assert "## File Contents" in report_content
    assert "src/mypkg/module.py" in report_content.replace("\\", "/")


# --- 'namespace' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_basic(runner: CliRunner, setup_test_project: Path):
    """Test basic `vibelint namespace` output."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    result = runner.invoke(cli, ["--debug", "namespace"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Namespace Structure Visualization")
    assert_output_contains(result, setup_test_project.name)
    assert_output_contains(result, "src")
    assert_output_contains(result, "mypkg")
    assert_output_contains(result, "module")
    assert_output_contains(result, "hello (member)")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_output_file(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint namespace -o tree.txt` saves the tree."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    tree_file = setup_test_project / "namespace_tree.txt"
    assert not tree_file.exists()

    result = runner.invoke(
        cli, ["--debug", "namespace", "-o", str(tree_file)], catch_exceptions=False
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert tree_file.is_file(), f"Namespace tree file not created at {tree_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Namespace tree saved to" in cleaned_output
    ), f"Namespace save text missing. Output:\n{cleaned_output}"
    assert (
        tree_file.name in cleaned_output
    ), f"Namespace save filename missing. Filename: {tree_file.name}\nOutput:\n{cleaned_output}"

    tree_content = tree_file.read_text()
    assert setup_test_project.name in tree_content
    assert "hello (member)" in tree_content


# --- 'snapshot' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_basic(runner: CliRunner, setup_test_project: Path):
    """Test basic `vibelint snapshot` default output."""
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(cli, ["--debug", "snapshot"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.is_file(), f"Snapshot file not created at {snapshot_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Codebase snapshot created at" in cleaned_output
    ), f"Snapshot creation text missing. Output:\n{cleaned_output}"
    assert (
        snapshot_file.name in cleaned_output
    ), f"Snapshot creation filename missing. Filename: {snapshot_file.name}\nOutput:\n{cleaned_output}"

    snapshot_content = snapshot_file.read_text()
    normalized_content = snapshot_content.replace("\\", "/")  # Normalize paths in content
    assert "# Snapshot" in normalized_content
    assert "## Filesystem Tree" in normalized_content
    # Check for the project root name (should be the directory name vibelint ran in)
    assert setup_test_project.name + "/" in normalized_content
    assert "pyproject.toml" in normalized_content
    assert "src/" in normalized_content
    assert "__init__.py" in normalized_content
    assert "module.py" in normalized_content
    assert "## File Contents" in normalized_content
    assert "### File: pyproject.toml" in normalized_content
    assert "[tool.vibelint]" in normalized_content
    assert "### File: src/mypkg/__init__.py" in normalized_content
    assert "### File: src/mypkg/module.py" in normalized_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude(runner: CliRunner, setup_test_project: Path):
    """Test snapshot respects exclude_globs from config."""
    modify_pyproject(
        setup_test_project,
        {
            "include_globs": ["src/**/*.py", "pyproject.toml"],
            "exclude_globs": ["src/mypkg/module.py"],
        },
    )

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    result = runner.invoke(cli, ["--debug", "snapshot"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.is_file()

    snapshot_content = snapshot_file.read_text()
    tree_match = re.search(r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL)
    assert tree_match, "Filesystem Tree section not found"
    tree_content = tree_match.group(1)
    print(f"DEBUG Tree Content:\n{tree_content}")

    assert "module.py" not in tree_content
    # Simpler string check for presence in tree
    assert "__init__.py" in tree_content, "__init__.py missing from tree"
    assert "pyproject.toml" in tree_content, "pyproject.toml missing from tree"

    normalized_content = snapshot_content.replace("\\", "/")
    assert "## File Contents" in normalized_content
    assert "### File: src/mypkg/module.py" not in normalized_content
    assert "### File: src/mypkg/__init__.py" in normalized_content
    assert "### File: pyproject.toml" in normalized_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude_output_file(runner: CliRunner, setup_test_project: Path):
    """Test snapshot doesn't include its own output file."""
    modify_pyproject(
        setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml", "*.md"]}
    )

    snapshot_file = setup_test_project / "my_own_snapshot.md"

    # First run creates the file
    result1 = runner.invoke(
        cli, ["--debug", "snapshot", "-o", str(snapshot_file)], catch_exceptions=False
    )
    assert result1.exit_code == 0
    assert snapshot_file.is_file()
    print(f"Snapshot 1 Output:\n{result1.output}")
    snapshot_content_1 = snapshot_file.read_text()
    assert "my_own_snapshot.md" not in snapshot_content_1

    # Second run should explicitly exclude the existing snapshot file
    result2 = runner.invoke(
        cli, ["--debug", "snapshot", "-o", str(snapshot_file)], catch_exceptions=False
    )
    assert result2.exit_code == 0
    print(f"Snapshot 2 Output:\n{result2.output}")

    snapshot_content_2 = snapshot_file.read_text()
    cleaned_content_tree = ""
    tree_match = re.search(r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content_2, re.DOTALL)
    if tree_match:
        cleaned_content_tree = "\n".join(
            [line.strip() for line in tree_match.group(1).splitlines() if line.strip()]
        )
    print(f"DEBUG Tree Content (Run 2):\n{cleaned_content_tree}")

    cleaned_content_files = ""
    files_match = re.search(r"## File Contents\s*(.*)", snapshot_content_2, re.DOTALL)
    if files_match:
        cleaned_content_files = "\n".join(
            line.strip() for line in files_match.group(1).splitlines() if line.strip()
        )
    print(f"DEBUG File Content (Run 2):\n{cleaned_content_files}")

    # Check exclusion from tree
    assert (
        "my_own_snapshot.md" not in cleaned_content_tree
    ), "Snapshot file included in tree section on second run"
    # Check exclusion from content
    assert (
        "### File: my_own_snapshot.md" not in cleaned_content_files
    ), "Snapshot file content included on second run"

    # Verify other expected files are still present using simpler string checks
    assert "pyproject.toml" in cleaned_content_tree, "pyproject.toml missing from tree (run 2)"
    assert "### File: pyproject.toml" in cleaned_content_files
    assert "__init__.py" in cleaned_content_tree, "__init__.py missing from tree (run 2)"
    assert "### File: src/mypkg/__init__.py" in cleaned_content_files.replace("\\", "/")
    assert "module.py" in cleaned_content_tree, "module.py missing from tree (run 2)"
    assert "### File: src/mypkg/module.py" in cleaned_content_files.replace("\\", "/")
```

---

