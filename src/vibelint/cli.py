#!/usr/bin/env python3
"""
Command-line interface for vibelint.

Conducts vibe checks on your codebase.

vibelint/src/vibelint/cli.py
"""

import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add this import
if sys.version_info >= (3, 7):
    import importlib.resources as pkg_resources
else:
    # Fallback for Python < 3.7
    import importlib_resources as pkg_resources

import click
from rich.logging import RichHandler
from rich.table import Table

# Import necessary functions for ASCII art
from .ascii import scale_to_terminal_by_height
from .config import Config, load_config
from .console_utils import console
from .formatters import DEFAULT_FORMAT, FORMAT_CHOICES
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
from .validation_engine import run_plugin_validation
from .diagnostics import run_benchmark, run_diagnostics


class VibelintContext:
    """
    Context object to store command results and shared state.

    vibelint/src/vibelint/cli.py
    """

    def __init__(self):
        """
        Initializes VibelintContext.

        vibelint/src/vibelint/cli.py
        """
        self.command_result: CommandResult | None = None
        self.project_root: Path | None = None


__all__ = ["snapshot", "check", "cli", "namespace", "main", "VibelintContext"]

logger_cli = logging.getLogger("vibelint")

# --- Helper messages ---
VIBE_CHECK_PASS_MESSAGES = [
    "Immaculate vibes.",
    "Vibes confirmed",
    "Vibe on brother.",
]

VIBE_CHECK_FAIL_MESSAGES = [
    "Vibe Check Failed.",
    "Vibe Check Failed.",
]


# (Keep _present_check_results, _present_namespace_results,
# _present_snapshot_results, and _display_collisions as they were in the
# last correct version - no changes needed there based on these errors)
def _present_check_results(result: CheckResult, runner):
    """
    Presents the results of the 'check' command (the Vibe Check™).

    vibelint/src/vibelint/cli.py
    """
    runner._print_summary()
    files_with_issues = sorted(
        [lr for lr in runner.results if lr.has_issues], key=lambda r: r.file_path
    )

    if files_with_issues:
        console.print("\n[bold yellow]Vibe Check:[/bold yellow]")
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
            except (TypeError, AttributeError) as e:
                console.print(
                    f"\n[bold cyan]{lr.file_path}:[/bold cyan] ([red]Error getting relative path: {e}[/red])"
                )
                logging.debug("Failed to get relative path for %s: %s", lr.file_path, e)

            for code, error_msg in lr.errors:
                console.print(f"  [red]ERROR[{code}] {error_msg}[/red]")
            for code, warning_msg in lr.warnings:
                console.print(f"  [yellow]WARN[{code}] {warning_msg}[/yellow]")

    has_collisions = bool(
        result.hard_collisions or result.global_soft_collisions or result.local_soft_collisions
    )
    if has_collisions:
        console.print()
        _display_collisions(
            result.hard_collisions,
            result.global_soft_collisions,
            result.local_soft_collisions,
        )
    else:
        logger_cli.debug("No namespace collisions detected.")

    if result.report_path:
        console.print()
        if result.report_generated:
            console.print(
                f"[green]SUCCESS: Detailed Vibe Report generated at {result.report_path}[/green]"
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


def _present_namespace_results(result: NamespaceResult):
    """
    Presents the results of the 'namespace' command.

    vibelint/src/vibelint/cli.py
    """
    if not result.success and result.error_message:
        console.print(f"[bold red]Error building namespace tree:[/bold red] {result.error_message}")
        # Don't proceed if the core operation failed

    if result.intra_file_collisions:
        console.print(
            "\n[bold yellow]Intra-file Collisions Found (Duplicate members within one file):[/bold yellow]"
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
            console.print(f"\n[green]SUCCESS: Namespace tree saved to {result.output_path}[/green]")
        elif result.output_error:
            console.print(
                f"[bold red]Error saving namespace tree:[/bold red] {result.output_error}"
            )
        # No need for 'unknown' status here unless saving was attempted and didn't error but failed

    # Only print tree to console if no output file was specified *and* the tree was built
    elif result.root_node and result.success:
        if result.intra_file_collisions:
            console.print()  # Add space
        console.print("\n[bold blue]Namespace Structure Visualization:[/bold blue]")
        console.print(str(result.root_node))


def _present_snapshot_results(result: SnapshotResult):
    """
    Presents the results of the 'snapshot' command. (Keep factual)

    vibelint/src/vibelint/cli.py
    """
    if result.success and result.output_path:
        console.print(f"[green]SUCCESS: Codebase snapshot created at {result.output_path}[/green]")
    elif not result.success and result.error_message:
        console.print(f"[bold red]Error creating snapshot:[/bold red] {result.error_message}")


def _display_collisions(
    hard_coll: list[NamespaceCollision],
    global_soft_coll: list[NamespaceCollision],
    local_soft_coll: list[NamespaceCollision],
) -> int:
    """
    Displays collision results in tables and returns an exit code indicating if hard collisions were found.

    vibelint/src/vibelint/cli.py
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

        vibelint/src/vibelint/cli.py
        """
        try:
            # Resolve paths before getting relative path for consistency
            return str(get_relative_path(p.resolve(), project_root.resolve()))
        except ValueError as e:
            logging.debug("Cannot get relative path for %s from %s: %s", p, project_root, e)
            return str(p.resolve())  # Fallback to absolute resolved path

    table_title = "Namespace Collision Summary"
    table = Table(title=table_title)
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")

    hard_label = "Hard Collisions (CRITICAL)"
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
        hard_header = "[bold red]CRITICAL: Hard Collision Details:[/bold red]"
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
        local_soft_header = "[bold yellow]Local Soft Collision (__all__) Details:[/bold yellow]"
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
            "[bold yellow]Global Namespace Collision (Definition) Details:[/bold yellow]"
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
def cli(ctx: click.Context, debug: bool) -> None:
    """
    vibelint - Check the vibe, visualize namespaces, and snapshot Python codebases.

    Run commands from the root of your project (where pyproject.toml or .git is located).

    Examples:
        vibelint check src/ --rule ARCHITECTURE-LLM  # Check src with LLM analysis
        vibelint check file.py                       # Check specific file
        vibelint namespace                           # Show namespace visualization
        vibelint snapshot                            # Create project snapshot

    vibelint/src/vibelint/cli.py
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
                    console.print("\nHow's the vibe?", justify="center")
                except (UnicodeDecodeError, OSError) as e:
                    logger_cli.warning(
                        f"Could not load or display VIBECHECKER.txt from package data: {e}",
                        exc_info=debug,
                    )
            else:
                logger_cli.debug(
                    "VIBECHECKER.txt not found in vibelint package data, skipping display."
                )
        except (ImportError, AttributeError, OSError) as e:
            logger_cli.warning(
                f"Error accessing package resources for VIBECHECKER.txt: {e}", exc_info=debug
            )

        console.print("\nRun [bold cyan]vibelint --help[/bold cyan] for available commands.")
        console.print(
            "Common usage: [bold]vibelint check src/[/bold] or [bold]vibelint check file.py --rule ARCHITECTURE-LLM[/bold]"
        )
        ctx.exit(0)

    # --- Subcommand Execution Check ---
    if vibelint_ctx.project_root is None:
        console.print("[bold red]Error:[/bold red] Could not find project root.")
        console.print("  vibelint needs to know where the project starts to check its vibes.")
        console.print("  Make sure you're in a directory with 'pyproject.toml' or '.git'.")
        ctx.exit(1)


@cli.command("check")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt for large directories.")
@click.option(
    "-o",
    "--output-report",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Save a comprehensive Vibe Report (Markdown) to the specified file.",
)
@click.option(
    "--format",
    "output_format",
    default=DEFAULT_FORMAT,
    type=click.Choice(FORMAT_CHOICES),
    help="Report format: natural (default, optimized for humans and AI agents), human (alias for natural), json, or sarif.",
)
@click.option(
    "--categories",
    default="all",
    help="Comma-separated list of rule categories to run: core, static, ai, or 'all' for everything. Default: 'all' (respects pyproject.toml configuration).",
)
@click.option(
    "--exclude-ai",
    is_flag=True,
    help="Exclude AI-powered validators (equivalent to --categories=core,static). Use this if AI validators are too slow or unreliable.",
)
@click.option(
    "--include-ai",
    is_flag=True,
    help="Explicitly include AI-powered validators (redundant since they're included by default now).",
)
@click.option(
    "--rule",
    "specific_rules",
    multiple=True,
    help="Run only specific rule(s). Can be used multiple times. Example: --rule=ARCHITECTURE-LLM --rule=SEMANTIC-SIMILARITY",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Automatically fix low-hanging fruit issues like missing docstrings, path references, and __all__ exports using the configured LLM.",
)
@click.pass_context
def check(
    ctx: click.Context,
    paths: tuple[Path, ...],
    yes: bool,
    output_report: Path | None,
    output_format: str,
    categories: str,
    exclude_ai: bool,
    include_ai: bool,
    specific_rules: tuple[str, ...],
    fix: bool,
) -> None:
    """
    Run a Vibe Check: Lint rules and namespace collision detection.

    PATHS: Optional paths to files or directories to analyze. When provided,
    these override the include_globs configuration and analyze only the specified paths.

    Fails if errors (like missing docstrings/`__all__`) or hard collisions are found.
    Warnings indicate potential vibe issues or areas for improvement.

    By default, runs ALL rules configured in pyproject.toml including AI validators.
    Use --exclude-ai if AI analysis is too slow for your workflow.

    Examples:
        vibelint check                               # Analyze entire project with all configured rules
        vibelint check src/                          # Override: analyze only src directory
        vibelint check file.py                       # Override: analyze only single file
        vibelint check src/ tests/                   # Override: analyze only these paths

        vibelint check --exclude-ai src/             # Skip AI validators for faster analysis
        vibelint check --rule=ARCHITECTURE-LLM src/ # Run only specific rule
        vibelint check --rule=SEMANTIC-SIMILARITY --rule=ARCHITECTURE-LLM src/  # Run multiple specific rules

        vibelint check --fix src/                    # Automatically fix issues like missing docstrings and __all__

    vibelint/src/vibelint/cli.py
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing in check command"

    # Don't show UI messages for machine-readable formats
    if output_format in ["human", "natural"]:
        console.print("\n[bold magenta]Initiating Vibe Check...[/bold magenta]\n")

    logger_cli.debug(f"Running 'check' command (yes={yes}, report={output_report})")

    config: Config = load_config(project_root)
    if config.project_root is None:
        logger_cli.error("Project root lost after config load. Aborting Vibe Check.")
        ctx.exit(1)

    # Use plugin system for validation combined with namespace collision detection
    result_data = CheckResult()

    # Handle category and rule filtering
    if exclude_ai:
        categories = "core,static"
    elif include_ai:
        categories = "all"  # Redundant since "all" is now the default

    # If specific rules are provided, use only those
    if specific_rules:
        enabled_categories = [
            "core",
            "static",
            "ai",
        ]  # Allow all categories when filtering by specific rules
        console.print(f"[bold blue]Running specific rules:[/bold blue] {', '.join(specific_rules)}")
    else:
        # Parse and validate categories
        if categories == "all":
            enabled_categories = ["core", "static", "ai"]
        else:
            enabled_categories = [cat.strip() for cat in categories.split(",")]
            valid_categories = {"core", "static", "ai"}
            invalid_categories = set(enabled_categories) - valid_categories
            if invalid_categories:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid categories: {', '.join(invalid_categories)}"
                )
                console.print(f"Valid categories: {', '.join(valid_categories)}")
                ctx.exit(1)

    # Filter config to only enable rules from selected categories or specific rules
    config_dict = dict(config.settings)

    if specific_rules:
        # Enable only specific rules and disable all others
        if "rules" in config_dict and isinstance(config_dict["rules"], dict):
            for rule_id in list(config_dict["rules"].keys()):
                if rule_id in specific_rules:
                    config_dict["rules"][rule_id] = "WARN"  # Enable specified rules
                else:
                    config_dict["rules"][rule_id] = "OFF"  # Disable all other rules
        else:
            # Create rules dict with only specified rules
            config_dict["rules"] = {rule_id: "WARN" for rule_id in specific_rules}

    elif "rule_categories" in config_dict and enabled_categories != ["core", "static", "ai"]:
        rule_categories = config_dict["rule_categories"]
        if isinstance(rule_categories, dict):
            enabled_rules = set()
            for category in enabled_categories:
                if category in rule_categories and isinstance(rule_categories[category], list):
                    enabled_rules.update(rule_categories[category])

            # Disable rules not in enabled categories
            if "rules" in config_dict and isinstance(config_dict["rules"], dict):
                for rule_id in list(config_dict["rules"].keys()):
                    if rule_id not in enabled_rules:
                        config_dict["rules"][rule_id] = "OFF"

    try:
        # Run plugin-based validation with filtered config
        logger_cli.debug(f"Running plugin-based validation with categories: {enabled_categories}")

        # Convert paths tuple to list for plugin runner
        include_globs_override = list(paths) if paths else None
        if include_globs_override:
            logger_cli.info(
                f"Overriding include_globs: analyzing only specified paths: {[str(p) for p in include_globs_override]}"
            )
        else:
            logger_cli.debug("Using configured include_globs for project-wide analysis")

        plugin_runner = run_plugin_validation(config_dict, project_root, include_globs_override)

        # Apply fixes if --fix flag is provided
        if fix:
            console.print("\n[bold blue]Applying automatic fixes...[/bold blue]")
            import asyncio
            from .fix import apply_fixes

            # Collect fixable findings by file
            from collections import defaultdict

            file_findings = defaultdict(list)

            from .fix import can_fix_finding

            for finding in plugin_runner.findings:
                if can_fix_finding(finding):
                    # Convert relative path back to absolute path
                    absolute_path = project_root / finding.file_path
                    file_findings[absolute_path].append(finding)

            if file_findings:
                # Apply fixes asynchronously
                try:
                    fix_results = asyncio.run(apply_fixes(config, file_findings))

                    # Report fix results
                    fixed_count = sum(1 for was_fixed in fix_results.values() if was_fixed)
                    total_files = len(fix_results)

                    if fixed_count > 0:
                        console.print(
                            f"[green]Applied fixes to {fixed_count}/{total_files} files[/green]"
                        )

                        # Re-run validation to show updated results
                        console.print(
                            "\n[bold blue]Re-running validation after fixes...[/bold blue]"
                        )
                        plugin_runner = run_plugin_validation(
                            config_dict, project_root, include_globs_override
                        )
                    else:
                        console.print("[yellow]No fixes could be applied automatically[/yellow]")

                except Exception as e:
                    console.print(f"[red]Error applying fixes: {e}[/red]")
            else:
                console.print("[yellow]No fixable issues found[/yellow]")

        # For machine-readable formats, output just the validation results and exit
        if output_format not in ["human", "natural"]:
            # Temporarily disable logging for clean JSON/SARIF output
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            try:
                output = plugin_runner.format_output(output_format)
                print(output)
                # Use sys.exit instead of ctx.exit to avoid Rich traceback
                import sys

                sys.exit(plugin_runner.get_exit_code())
            finally:
                logging.getLogger().setLevel(original_level)

        # For human format, display validation results first
        validation_output = plugin_runner.format_output("human")
        print(validation_output, end="")

        # Then check for namespace collisions
        logger_cli.debug("Checking for namespace vibe collisions...")
        target_paths: list[Path] = [project_root]
        result_data.hard_collisions = detect_hard_collisions(target_paths, config)
        result_data.global_soft_collisions = detect_global_definition_collisions(
            target_paths, config
        )
        result_data.local_soft_collisions = detect_local_export_collisions(target_paths, config)

        # Display namespace collision results if any were found
        has_collisions = (
            result_data.hard_collisions
            or result_data.global_soft_collisions
            or result_data.local_soft_collisions
        )
        if has_collisions:
            console.print()
            _display_collisions(
                result_data.hard_collisions,
                result_data.global_soft_collisions,
                result_data.local_soft_collisions,
            )

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
                        findings=result_data.findings,
                        hard_coll=result_data.hard_collisions,
                        soft_coll=result_data.global_soft_collisions
                        + result_data.local_soft_collisions,
                        root_node=root_node_for_report,
                        config=config,
                    )
                result_data.report_generated = True
                logger_cli.debug("Vibe Report generation successful.")
            except (OSError, RuntimeError, ValueError) as e:
                logger_cli.error(f"Error generating vibe report: {e}", exc_info=True)
                result_data.report_error = str(e)
                report_failed = True

        report_failed_code = 1 if report_failed else 0
        plugin_exit_code = plugin_runner.get_exit_code()
        final_exit_code = plugin_exit_code or collision_exit_code or report_failed_code
        result_data.exit_code = final_exit_code
        result_data.success = final_exit_code == 0
        logger_cli.debug(f"Vibe Check command finished. Final Exit Code: {final_exit_code}")

    except (RuntimeError, ValueError, OSError) as e:
        logger_cli.error(f"Critical error during Vibe Check execution: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    # Display report generation status
    if result_data.report_path:
        if result_data.report_generated:
            console.print(
                f"[green]SUCCESS: Detailed Vibe Report generated at {result_data.report_path}[/green]"
            )
        elif result_data.report_error:
            console.print(
                f"[bold red]Error generating vibe report:[/bold red] {result_data.report_error}"
            )

    vibelint_ctx.command_result = result_data
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
def namespace(ctx: click.Context, output: Path | None) -> None:
    """
    Visualize the project's Python namespace structure (how things import).

    Useful for untangling vibe conflicts.

    vibelint/src/vibelint/cli.py
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
        target_paths: list[Path] = [project_root]
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
            except (OSError, UnicodeEncodeError) as e:
                logger_cli.error(f"Error saving namespace tree: {e}", exc_info=True)
                result_data.output_error = str(e)
        else:
            result_data.output_saved = False

        result_data.exit_code = 0 if result_data.success else 1

    except (RuntimeError, ValueError, OSError) as e:
        logger_cli.error(f"Error building namespace tree: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_namespace_results(result_data)
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
def snapshot(ctx: click.Context, output: Path) -> None:
    """
    Create a Markdown snapshot of project files (for LLMs or humans).

    Respects include/exclude rules from your config. Good for context dumping.

    vibelint/src/vibelint/cli.py
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
        target_paths: list[Path] = [project_root]
        logger_cli.info(f"Creating codebase snapshot at {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Pass the non-None target_paths here too
        create_snapshot(output_path=output_path, target_paths=target_paths, config=config)
        result_data.success = True
        result_data.exit_code = 0

    except (OSError, RuntimeError, ValueError) as e:
        logger_cli.error(f"Error creating snapshot: {e}", exc_info=True)
        result_data.success = False
        result_data.error_message = str(e)
        result_data.exit_code = 1

    vibelint_ctx.command_result = result_data
    _present_snapshot_results(result_data)
    ctx.exit(result_data.exit_code)


@cli.command("thinking-tokens")
@click.option("--show-formats", is_flag=True, help="Show all supported thinking token formats")
@click.option(
    "--detect",
    type=click.Path(exists=True, path_type=Path),
    help="Detect thinking tokens in a file",
)
def thinking_tokens(show_formats: bool, detect: Path | None) -> None:
    """
    Get help with configuring thinking token removal for your LLM model.

    Vibelint automatically removes "thinking" tokens from LLM responses. This
    command helps you configure it for your specific model type.

    Examples:
        vibelint thinking-tokens --show-formats  # Show all supported formats
        vibelint thinking-tokens --detect file.py  # Detect tokens in file

    vibelint/src/vibelint/cli.py
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    if show_formats:
        console.print("\n[bold blue]Supported Thinking Token Formats:[/bold blue]")

        # Harmony format
        console.print(
            Panel(
                "[bold]harmony[/bold] (default) - For Claude/Anthropic models\n"
                "Removes: <|channel|>analysis<|message|>..., <thinking>...</thinking>, etc.\n"
                "\n[dim]Configuration:[/dim]\n"
                'thinking_format = "harmony"',
                title="Harmony Format",
                border_style="green",
            )
        )

        # Qwen format
        console.print(
            Panel(
                "[bold]qwen[/bold] - For Qwen model family\n"
                "Removes: <think>...</think>, <思考>...</思考>, 思考：..., etc.\n"
                "\n[dim]Configuration:[/dim]\n"
                'thinking_format = "qwen"',
                title="Qwen Format",
                border_style="blue",
            )
        )

        # Custom format
        console.print(
            Panel(
                "[bold]custom[/bold] - For other models with custom patterns\n"
                "Define your own regex patterns for thinking token removal.\n"
                "\n[dim]Configuration:[/dim]\n"
                'thinking_format = "custom"\n'
                "custom_thinking_patterns = [\n"
                "  r'<think>.*?</think>',\n"
                "  r'<reasoning>.*?</reasoning>',\n"
                "  r'# Thinking:.*?(?=\\n|$)',\n"
                "]",
                title="Custom Format",
                border_style="yellow",
            )
        )

        # Configuration example
        config_example = """[tool.vibelint.llm_analysis]
# Basic LLM settings
api_base_url = "http://localhost:11434"
model = "your-model-name"

# Thinking token configuration
remove_thinking_tokens = true    # Set to false to keep all output
thinking_format = "harmony"      # Options: "harmony", "qwen", "custom"

# For custom patterns:
# thinking_format = "custom"
# custom_thinking_patterns = [
#     "r'<think>.*?</think>'",
#     "r'<reasoning>.*?</reasoning>'"
# ]"""

        syntax = Syntax(config_example, "toml", theme="monokai", line_numbers=True)
        console.print(
            Panel(syntax, title="Example pyproject.toml Configuration", border_style="cyan")
        )

    elif detect:
        console.print(f"\n[bold blue]Analyzing {detect} for thinking tokens...[/bold blue]")

        try:
            content = detect.read_text(encoding="utf-8")

            # Import the detection logic
            from .validators.architecture.llm_analysis import LLMAnalysisValidator

            # Create a temporary validator instance to use detection method
            validator = LLMAnalysisValidator()
            detected = validator._detect_unremoved_thinking_tokens(content)

            if detected:
                console.print("\n[bold yellow]Found potential thinking tokens:[/bold yellow]")
                for token in detected:
                    console.print(f"  • {token}")

                console.print("\n[bold green]Suggested configuration:[/bold green]")

                # Generate suggestions
                suggestions = []
                for pattern_name in detected:
                    if pattern_name == "<think>":
                        suggestions.append("r'<think>.*?</think>'")
                    elif pattern_name == "<reasoning>":
                        suggestions.append("r'<reasoning>.*?</reasoning>'")
                    elif pattern_name == "[THINKING]":
                        suggestions.append("r'\\[THINKING\\].*?\\[/THINKING\\]'")
                    elif pattern_name == "```thinking":
                        suggestions.append("r'```thinking.*?```'")
                    # Add more as needed

                if suggestions:
                    config = f"""[tool.vibelint.llm_analysis]
thinking_format = "custom"
custom_thinking_patterns = {suggestions}"""

                    syntax = Syntax(config, "toml", theme="monokai")
                    console.print(
                        Panel(syntax, title="Add to pyproject.toml", border_style="green")
                    )

            else:
                console.print(
                    "\n[bold green]✓ No thinking tokens detected in this file.[/bold green]"
                )

        except Exception as e:
            console.print(f"[bold red]Error reading file: {e}[/bold red]")

    else:
        # Show general help
        console.print("\n[bold blue]Thinking Token Configuration Help[/bold blue]")
        console.print(
            "\nVibelint automatically removes 'thinking' tokens from LLM responses to provide clean analysis output."
        )
        console.print("\nUse these options to get help:")
        console.print(
            "  [dim]vibelint thinking-tokens --show-formats[/dim]  Show all supported formats"
        )
        console.print(
            "  [dim]vibelint thinking-tokens --detect FILE[/dim]   Detect tokens in a file"
        )
        console.print("\nCommon model configurations:")
        console.print("  • [bold]Claude/Anthropic models:[/bold] Use default (harmony format)")
        console.print('  • [bold]Qwen models:[/bold] Set thinking_format = "qwen"')
        console.print('  • [bold]Other models:[/bold] Set thinking_format = "custom" + patterns')


@cli.command("diagnostics")
@click.option("--benchmark", is_flag=True, help="Run LLM performance benchmark")
@click.option("--test", is_flag=True, help="Run diagnostic tests")
def diagnostics_cmd(benchmark: bool, test: bool) -> None:
    """
    Run vibelint diagnostics and performance tests.

    Examples:
        vibelint diagnostics --benchmark    # Benchmark LLM performance
        vibelint diagnostics --test         # Run diagnostic tests
        vibelint diagnostics --benchmark --test  # Run both

    tools/vibelint/src/vibelint/cli.py
    """
    if not benchmark and not test:
        # Run both by default
        benchmark = test = True

    if benchmark:
        console.print("[bold blue]Running LLM performance benchmark...[/bold blue]")
        try:
            run_benchmark()
            console.print("[green]Benchmark completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            sys.exit(1)

    if test:
        console.print("[bold blue]Running diagnostic tests...[/bold blue]")
        try:
            run_diagnostics()
            console.print("[green]Diagnostics completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Diagnostics failed: {e}[/red]")
            sys.exit(1)


def main() -> None:
    """
    Main entry point for the vibelint CLI application.

    vibelint/src/vibelint/cli.py
    """
    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except (RuntimeError, ValueError, OSError, ImportError) as e:
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
