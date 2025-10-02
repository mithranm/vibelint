"""CLI for vibelint - all commands in one module.

Provides core commands: check, snapshot.

vibelint/src/vibelint/cli.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class VibelintContext:
    """Shared context for CLI commands."""

    project_root: Path | None = None
    config_path: Path | None = None
    verbose: bool = False


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """vibelint: Code quality linter with dynamic plugin discovery."""
    # Auto-detect project root
    current = Path.cwd()
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            project_root = parent
            break

    # Store context for subcommands
    ctx.obj = VibelintContext(
        project_root=project_root,
        config_path=None,
        verbose=verbose,
    )

    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command("check")
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "-f", type=click.Choice(["human", "json"]), default="human", help="Output format"
)
@click.option("--exclude-ai", is_flag=True, help="Skip AI validators (faster)")
@click.option("--rules", help="Comma-separated rules to run")
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
@click.pass_context
def check(
    ctx: click.Context, targets: tuple[Path, ...], format: str, exclude_ai: bool, rules: str | None, fix: bool
) -> None:
    """Run vibelint validation."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    if not project_root:
        console.print("[red]❌ No project root found[/red]")
        ctx.exit(1)

    # Load config
    from vibelint.config import Config, load_config

    config: Config = load_config(project_root)
    if not config.is_present():
        console.print("[red]❌ No vibelint configuration found[/red]")
        ctx.exit(1)

    # Import validation engine
    from vibelint.validation_engine import PluginValidationRunner
    from vibelint.discovery import discover_files_from_paths

    # Determine target files
    if targets:
        files = discover_files_from_paths(list(targets), config)
    else:
        files = discover_files_from_paths([project_root], config)

    if not files:
        console.print("No Python files found")
        ctx.exit(0)

    # Note: exclude_ai and rules filtering should be implemented in RuleEngine
    # For now, Config object is passed as-is
    # TODO: Add rule filtering support in RuleEngine

    # Run validation
    runner = PluginValidationRunner(config, project_root)
    findings = runner.run_validation(files)

    # Apply fixes if requested
    if fix:
        from vibelint.validators.registry import validator_registry

        fixed_count = 0
        for file_path in files:
            file_findings = [f for f in findings if f.file_path == file_path]
            if not file_findings:
                continue

            # Read the file content
            try:
                content = file_path.read_text(encoding="utf-8")
                original_content = content

                # Apply fixes from validators
                for finding in file_findings:
                    validator_class = validator_registry.get_validator(finding.rule_id)
                    if validator_class:
                        # Instantiate the validator
                        validator_instance = validator_class()
                        if hasattr(validator_instance, "can_fix") and validator_instance.can_fix(finding):
                            content = validator_instance.apply_fix(content, finding)

                # Write back if changed
                if content != original_content:
                    file_path.write_text(content, encoding="utf-8")
                    fixed_count += 1
                    console.print(f"[green]✓[/green] Fixed {file_path}")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to fix {file_path}: {e}")

        if fixed_count > 0:
            console.print(f"\n[green]Fixed {fixed_count} file(s)[/green]")
        else:
            console.print("[yellow]No fixable issues found[/yellow]")

        # Exit early after fixing
        ctx.exit(0)

    # Output results
    output = runner.format_output(format)
    print(output)

    # Exit with proper code
    errors = sum(1 for f in findings if f.severity.name == "ERROR")
    warnings = sum(1 for f in findings if f.severity.name == "WARN")

    if format == "human" and (errors or warnings):
        console.print(f"\nFound {errors} error(s), {warnings} warning(s)")

    ctx.exit(1 if errors > 0 else 0)


@cli.command("snapshot")
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="codebase_snapshot.md",
    help="Output markdown file path",
)
@click.pass_context
def snapshot(ctx: click.Context, targets: tuple[Path, ...], output: Path) -> None:
    """Create a markdown snapshot of the codebase structure and contents."""
    from vibelint.config import load_config, Config
    from vibelint.snapshot import create_snapshot

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root or Path.cwd()

    # Load config
    try:
        config = load_config(project_root)
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not load config: {e}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        config = Config(project_root=project_root)

    # Default targets to project root if none provided
    if not targets:
        targets = [project_root]

    target_list = list(targets)

    try:
        console.print(f"[blue]📸 Creating snapshot of {len(target_list)} target(s)...[/blue]")
        create_snapshot(output_path=output, target_paths=target_list, config=config)
        console.print(f"[green]✅ Snapshot saved to {output}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Snapshot failed: {e}[/red]")
        logger.error(f"Snapshot error: {e}", exc_info=True)
        ctx.exit(1)


def _register_workflow_commands() -> None:
    """Dynamically register CLI commands for all workflows in the registry."""
    from vibelint.workflows.registry import workflow_registry

    # Force load all workflows
    workflow_registry._ensure_loaded()

    for workflow_id, workflow_class in workflow_registry.get_all_workflows().items():
        # Get workflow metadata
        temp_instance = workflow_class()
        description = f"{temp_instance.name}: {temp_instance.description}"

        # Create a command for each workflow
        @cli.command(workflow_id, help=description)
        @click.argument("target", required=False, type=click.Path(exists=True, path_type=Path))
        @click.pass_context
        def workflow_cmd(ctx: click.Context, target: Path | None, workflow_id=workflow_id, workflow_class=workflow_class):
            """Run a vibelint workflow."""
            vibelint_ctx: VibelintContext = ctx.obj
            project_root = target or vibelint_ctx.project_root or Path.cwd()

            try:
                # Instantiate workflow
                workflow_instance = workflow_class()
                console.print(f"[blue]Running workflow: {workflow_id}[/blue]")
                console.print(f"[dim]Target: {project_root}[/dim]")

                # Run execute method (sync)
                result = workflow_instance.execute(project_root, {})

                # Display results
                if result.status.value == "completed":
                    console.print(f"[green]✅ Workflow completed successfully[/green]")
                    if result.artifacts.get("report"):
                        console.print("\n" + result.artifacts["report"])
                else:
                    console.print(f"[red]❌ Workflow failed: {result.error_message}[/red]")
                    ctx.exit(1)

            except Exception as e:
                console.print(f"[red]❌ Workflow {workflow_id} failed: {e}[/red]")
                logger.error(f"Workflow error: {e}", exc_info=True)
                ctx.exit(1)


def main() -> None:
    """Entry point for vibelint CLI."""
    import sys

    # Register workflow commands before running CLI
    _register_workflow_commands()

    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.error("CLI error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
