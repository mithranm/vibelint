"""
Validation commands: check and validate.

These commands handle the core linting and validation functionality.

vibelint/src/vibelint/cli/validation.py
"""

import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from vibelint.config import Config, load_config
from vibelint.cli.cli_group import VibelintContext, cli

console = Console()
logger = logging.getLogger(__name__)


@cli.command("check")
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(["human", "json"]), default="human", help="Output format")
@click.option("--exclude-ai", is_flag=True, help="Skip AI validators (faster)")
@click.option("--rules", help="Comma-separated rules to run")
@click.pass_context
def check(ctx: click.Context, targets: tuple[Path, ...], format: str, exclude_ai: bool, rules: Optional[str]) -> None:
    """Run vibelint validation."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root required"

    # Load config
    config: Config = load_config(project_root)
    if not config.is_present():
        logger.error("No vibelint configuration found")
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
        # Remove AI validator rules
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
    errors = sum(1 for f in findings if f.severity.name in ["ERROR"])
    warnings = sum(1 for f in findings if f.severity.name in ["WARN"])

    if format == "human" and (errors or warnings):
        console.print(f"\nFound {errors} error(s), {warnings} warning(s)")

    ctx.exit(1 if errors > 0 else 0)


# Remove the validate command entirely - check is sufficient
