"""
CLI for vibelint - all commands in one module.

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
@click.option("--format", "-f", type=click.Choice(["human", "json"]), default="human", help="Output format")
@click.option("--exclude-ai", is_flag=True, help="Skip AI validators (faster)")
@click.option("--rules", help="Comma-separated rules to run")
@click.pass_context
def check(ctx: click.Context, targets: tuple[Path, ...], format: str, exclude_ai: bool, rules: str | None) -> None:
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

    # Get config dict for filtering
    config_dict = dict(config.settings)

    # Filter AI validators if requested
    if exclude_ai:
        if "rules" in config_dict and "enable" in config_dict["rules"]:
            enabled = config_dict["rules"]["enable"]
            config_dict["rules"]["enable"] = [r for r in enabled if not r.endswith("-LLM")]

    # Filter specific rules if requested
    if rules:
        rule_list = [r.strip() for r in rules.split(",")]
        config_dict["rules"] = {"enable": rule_list}

    # Run validation
    runner = PluginValidationRunner(config_dict, project_root)
    findings = runner.run_validation(files)

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
@click.option("--output", "-o", type=click.Path(path_type=Path), default="codebase_snapshot.md", help="Output markdown file path")
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
        create_snapshot(
            output_path=output,
            target_paths=target_list,
            config=config
        )
        console.print(f"[green]✅ Snapshot saved to {output}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Snapshot failed: {e}[/red]")
        logger.error(f"Snapshot error: {e}", exc_info=True)
        ctx.exit(1)



def main() -> None:
    """Entry point for vibelint CLI."""
    import sys
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
