"""
Command-line interface for vibelint.

Provides commands to check codebase health, visualize namespaces, and create snapshots.

vibelint/cli.py
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler


from .results import CheckResult, NamespaceResult, SnapshotResult, CommandResult


from .config import load_config, Config
from .lint import LintRunner
from .namespace import (
    build_namespace_tree,
    detect_hard_collisions,
    detect_global_definition_collisions,
    detect_local_export_collisions,
    NamespaceCollision,
)
from .snapshot import create_snapshot
from .report import write_report_content
from .utils import get_relative_path, find_project_root


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


def _present_check_results(result: CheckResult, runner: LintRunner, console: Console):
    """
    Presents the results of the 'check' command, including detailed lint issues with codes.

    vibelint/cli.py
    """

    runner._print_summary()

    files_with_issues = sorted(
        [lr for lr in runner.results if lr.has_issues], key=lambda r: r.file_path
    )
    if files_with_issues:
        console.print("\n[bold yellow]Files with Issues:[/bold yellow]")
        for lr in files_with_issues:
            try:
                rel_path = (
                    get_relative_path(lr.file_path, runner.config.project_root)
                    if runner.config.project_root
                    else lr.file_path
                )
                console.print(f"\n[bold cyan]{rel_path}:[/bold cyan]")
            except ValueError:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([yellow]Outside project?[/yellow])"
                )
            except Exception as e:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([red]Error getting relative path: {e}[/red])"
                )

            for code, error_msg in lr.errors:
                console.print(f"  [red]✗ [{code}] {error_msg}[/red]")

            for code, warning_msg in lr.warnings:
                console.print(f"  [yellow]▲ [{code}] {warning_msg}[/yellow]")

    if (
        result.hard_collisions
        or result.global_soft_collisions
        or result.local_soft_collisions
    ):
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
            console.print(f"[green]✓ Report generated at {result.report_path}[/green]")
        elif result.report_error:
            console.print(
                f"\n[bold red]Error generating report:[/bold red] {result.report_error}"
            )
        else:
            console.print(
                f"[yellow]Report status unknown for {result.report_path}[/yellow]"
            )

    console.print()
    if result.exit_code != 0:
        console.print(
            f"[bold red]Check finished with errors (exit code {result.exit_code}).[/bold red]"
        )
    elif runner.results:
        console.print("[bold green]Check finished successfully.[/bold green]")
    else:
        console.print(
            "[bold blue]Check finished. No Python files found or processed.[/bold blue]"
        )


def _present_namespace_results(result: NamespaceResult, console: Console):
    """
    Presents the results of the 'namespace' command.

    vibelint/cli.py
    """

    if not result.success:
        console.print(
            f"[bold red]Error building namespace tree:[/bold red] {result.error_message}"
        )
        return

    if result.intra_file_collisions:
        console.print("\n[bold yellow]Intra-file Collisions Found:[/bold yellow]")
        console.print("These duplicate names were found within the same file:")
        ctx = click.get_current_context(silent=True)
        project_root = (
            ctx.obj.project_root
            if ctx and hasattr(ctx.obj, "project_root")
            else Path(".")
        )

        for c in sorted(
            result.intra_file_collisions, key=lambda x: (str(x.paths[0]), x.name)
        ):
            try:
                rel_path = get_relative_path(c.paths[0], project_root)
            except ValueError:
                rel_path = c.paths[0]

            loc1 = (
                f"{rel_path}:{c.linenos[0]}"
                if c.linenos and c.linenos[0]
                else str(rel_path)
            )
            line1 = c.linenos[0] if c.linenos else "?"
            line2 = c.linenos[1] if len(c.linenos) > 1 else "?"
            console.print(
                f"- '{c.name}': Duplicate definition/import in {loc1} (lines ~{line1} and ~{line2})"
            )

    if result.output_path:
        if result.intra_file_collisions:
            console.print()
        if result.output_saved:
            console.print(
                f"\n[green]✓ Namespace tree saved to {result.output_path}[/green]"
            )
        elif result.output_error:
            console.print(
                f"[bold red]Error saving namespace tree:[/bold red] {result.output_error}"
            )
        else:
            console.print(
                f"[yellow]Namespace tree status unknown for {result.output_path}[/yellow]"
            )
    elif result.root_node:
        if result.intra_file_collisions:
            console.print()
        console.print("\n[bold blue]Namespace Structure:[/bold blue]")
        console.print(str(result.root_node))


def _present_snapshot_results(result: SnapshotResult, console: Console):
    """
    Presents the results of the 'snapshot' command.

    vibelint/cli.py
    """

    if result.success and result.output_path:
        console.print(
            f"[green]✓ Codebase snapshot created at {result.output_path}[/green]"
        )
    elif not result.success:
        console.print(
            f"[bold red]Error creating snapshot:[/bold red] {result.error_message}"
        )


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

    exit_code = 0
    total_collisions = len(hard_coll) + len(global_soft_coll) + len(local_soft_coll)

    if total_collisions == 0:
        return 0

    ctx = click.get_current_context(silent=True)
    project_root = (
        ctx.obj.project_root if ctx and hasattr(ctx.obj, "project_root") else Path(".")
    )

    def get_rel_path_display(p: Path) -> str:
        """
        Function 'get_rel_path_display'.

        vibelint/cli.py
        """

        try:
            return str(get_relative_path(p, project_root))
        except ValueError:
            return str(p)

    table = Table(title="Namespace Collision Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row(
        "Hard Collisions", str(len(hard_coll)), style="red" if hard_coll else ""
    )
    table.add_row(
        "Global Soft Collisions (Definitions)",
        str(len(global_soft_coll)),
        style="yellow" if global_soft_coll else "",
    )
    table.add_row(
        "Local Soft Collisions (__all__)",
        str(len(local_soft_coll)),
        style="yellow" if local_soft_coll else "",
    )
    console.print(table)

    if hard_coll:
        console.print("\n[bold red]Hard Collisions:[/bold red]")
        console.print(
            "These collisions can break Python imports or indicate unexpected duplicates:"
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
                        if c.linenos and i < len(c.linenos) and c.linenos[i]
                        else ""
                    )
                    locations.append(f"{get_rel_path_display(p)}{line_info}")
            unique_locations = sorted(list(set(locations)))
            is_intra_file = len(collisions[0].paths) > 1 and all(
                p == collisions[0].paths[0] for p in collisions[0].paths[1:]
            )
            if len(unique_locations) <= 2 and is_intra_file:
                console.print(
                    f"- '{name}': Duplicate definition/import in {', '.join(unique_locations)}"
                )
            else:
                console.print(
                    f"- '{name}': Conflicting definitions/imports in {', '.join(unique_locations)}"
                )
        exit_code = 1

    if local_soft_coll:
        console.print("\n[bold yellow]Local Soft Collisions (__all__):[/bold yellow]")
        console.print(
            "These names are exported via __all__ in multiple sibling modules:"
        )
        local_table = Table(show_header=True, header_style="bold yellow")
        local_table.add_column("Name", style="cyan", min_width=15)
        local_table.add_column("Exporting Files")
        grouped_local = defaultdict(list)
        for c in local_soft_coll:
            grouped_local[c.name].extend(
                p for p in c.paths if p not in grouped_local[c.name]
            )

        for name, involved_paths in sorted(grouped_local.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            local_table.add_row(name, "\n".join(paths_str_list))
        console.print(local_table)

    if global_soft_coll:
        console.print(
            "\n[bold yellow]Global Soft Collisions (Definitions):[/bold yellow]"
        )
        console.print("These names are defined in multiple modules (may confuse LLMs):")
        global_table = Table(show_header=True, header_style="bold yellow")
        global_table.add_column("Name", style="cyan", min_width=15)
        global_table.add_column("Defining Files")
        grouped_global = defaultdict(list)
        for c in global_soft_coll:
            grouped_global[c.name].extend(
                p for p in c.paths if p not in grouped_global[c.name]
            )

        for name, involved_paths in sorted(grouped_global.items()):
            paths_str_list = sorted([get_rel_path_display(p) for p in involved_paths])
            global_table.add_row(name, "\n".join(paths_str_list))
        console.print(global_table)

    return exit_code


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug logging output.")
@click.pass_context
def cli(ctx: click.Context, debug: bool):
    """
    vibelint - Check, visualize, and create snapshots of Python codebases for LLM-friendliness.

    Run commands from the root of your project (where pyproject.toml or .git is located).

    vibelint/cli.py
    """

    ctx.ensure_object(VibelintContext)
    vibelint_ctx: VibelintContext = ctx.obj

    project_root = find_project_root(Path("."))
    if project_root is None:
        console.print("[bold red]Error:[/bold red] Could not find project root.")
        console.print("  vibelint must be run from within a directory that contains")
        console.print("  a 'pyproject.toml' file or a '.git' directory, or one of")
        console.print("  their subdirectories.")
        sys.exit(1)

    vibelint_ctx.project_root = project_root

    log_level = logging.DEBUG if debug else logging.INFO
    app_logger = logging.getLogger("vibelint")
    app_logger.setLevel(log_level)
    app_logger.propagate = False

    rich_handler = RichHandler(
        console=console,
        show_path=debug,
        markup=True,
        show_level=debug,
        rich_tracebacks=True,
    )
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(formatter)

    if not any(isinstance(h, RichHandler) for h in app_logger.handlers):
        app_logger.addHandler(rich_handler)

    logger_cli.debug(f"vibelint started. Debug mode: {'ON' if debug else 'OFF'}")
    logger_cli.debug(f"Identified project root: {project_root}")
    logger_cli.debug(f"Log level set to {logging.getLevelName(log_level)}")


@cli.command("check")
@click.option(
    "--yes", is_flag=True, help="Skip confirmation prompt for large directories."
)
@click.option(
    "-o",
    "--output-report",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save a comprehensive Markdown report to the specified file.",
)
@click.pass_context
def check(ctx: click.Context, yes: bool, output_report: Optional[Path]):
    """
    Run lint checks and detect namespace collisions within the project.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling check."

    logger_cli.debug(f"Running 'check' command (yes={yes}, report={output_report})")

    config: Config = load_config(project_root)
    if config.project_root is None:
        logger_cli.error("Project root became None after initial check. Aborting.")
        sys.exit(1)

    result_data = CheckResult()
    runner: Optional[LintRunner] = None

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        runner = LintRunner(config=config, skip_confirmation=yes)
        lint_exit_code = runner.run(target_paths)
        result_data.lint_results = runner.results
        vibelint_ctx.lint_runner = runner

        logger_cli.debug("Linting finished. Checking for namespace collisions...")
        result_data.hard_collisions = detect_hard_collisions(target_paths, config)
        result_data.global_soft_collisions = detect_global_definition_collisions(
            target_paths, config
        )
        result_data.local_soft_collisions = detect_local_export_collisions(
            target_paths, config
        )

        collision_exit_code = 1 if result_data.hard_collisions else 0

        report_failed = False
        if output_report:
            report_path = output_report.resolve()
            result_data.report_path = report_path
            logger_cli.info(f"Generating Markdown report to {report_path}...")
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                root_node_for_report, _ = build_namespace_tree(target_paths, config)
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
                logger_cli.debug("Report generation successful.")
            except Exception as e:
                logger_cli.error(f"Error generating report: {e}", exc_info=True)
                result_data.report_error = str(e)
                report_failed = True

        final_exit_code = (
            lint_exit_code or collision_exit_code or (1 if report_failed else 0)
        )
        result_data.exit_code = final_exit_code
        result_data.success = final_exit_code == 0
        logger_cli.debug(f"Check command finished. Exit code: {final_exit_code}")

    except Exception as e:
        logger_cli.error(f"Critical error during 'check' execution: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data

    if runner:
        _present_check_results(result_data, runner, console)
    else:
        console.print(
            "[bold red]Check command failed before linting could start.[/bold red]"
        )
        if result_data.error_message:
            console.print(f"[red]Error: {result_data.error_message}[/red]")

    sys.exit(result_data.exit_code)


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
    Visualize the project's Python namespace structure as a tree.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling namespace."

    logger_cli.debug(f"Running 'namespace' command (output={output})")

    config = load_config(project_root)
    if config.project_root is None:
        logger_cli.warning(
            "Project root missing from loaded config, forcing from context."
        )
        config._project_root = project_root

    result_data = NamespaceResult()

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        logger_cli.info("Building namespace tree...")
        root_node, intra_file_collisions = build_namespace_tree(target_paths, config)
        result_data.root_node = root_node
        result_data.intra_file_collisions = intra_file_collisions

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

        result_data.success = result_data.output_error is None
        result_data.exit_code = 0 if result_data.success else 1

    except Exception as e:
        logger_cli.error(f"Error building namespace tree: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_namespace_results(result_data, console)
    sys.exit(result_data.exit_code)


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
    Create a Markdown snapshot of the project files.

    vibelint/cli.py
    """

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    assert (
        project_root is not None
    ), "Project root must be set in context before calling snapshot."

    logger_cli.debug(f"Running 'snapshot' command (output={output})")

    config = load_config(project_root)
    if config.project_root is None:
        logger_cli.warning(
            "Project root missing from loaded config, forcing from context."
        )
        config._project_root = project_root

    result_data = SnapshotResult()
    output_path = output.resolve()
    result_data.output_path = output_path

    try:
        if not config.project_root:
            raise ValueError(
                "Project root could not be definitively determined in config."
            )
        target_paths = [config.project_root]

        logger_cli.info(f"Creating codebase snapshot at {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_snapshot(
            output_path=output_path, target_paths=target_paths, config=config
        )
        result_data.success = True
        result_data.exit_code = 0

    except Exception as e:
        logger_cli.error(f"Error creating snapshot: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_snapshot_results(result_data, console)
    sys.exit(result_data.exit_code)


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
        logger_cli.error("Unhandled exception in CLI execution.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
