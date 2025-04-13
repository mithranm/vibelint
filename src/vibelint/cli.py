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
