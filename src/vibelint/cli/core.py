"""
Core CLI group and shared context for vibelint.

This module provides the base CLI group and shared context object.
Individual command modules register their commands with this group.

Responsibility: CLI structure and shared state only.
Command logic belongs in validators/, workflows/, and other subsystems.

vibelint/src/vibelint/cli/core.py
"""

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
@click.option(
    "--project-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Project root directory (auto-detected if not specified)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Configuration file path",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(
    ctx: click.Context,
    project_root: Path | None,
    config: Path | None,
    verbose: bool,
) -> None:
    """
    vibelint: Intelligent code quality and style validator.

    Advanced linting with LLM-powered analysis, namespace collision detection,
    and project-wide validation rules.
    """
    # Auto-detect project root if not specified
    if not project_root:
        current = Path.cwd()
        # Walk up looking for pyproject.toml or .git
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                project_root = parent
                break

    # Store context for subcommands
    ctx.obj = VibelintContext(
        project_root=project_root,
        config_path=config,
        verbose=verbose,
    )

    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


# Import command modules to register them with the CLI group
# Each module registers its commands using @cli.command() decorators
def register_commands():
    """Import all command modules to register their commands."""
    try:
        from . import ai  # justify, thinking-tokens
        from . import analysis  # namespace, snapshot, diagnostics
        from . import maintenance  # setup, rollback, regen-docstrings
        from . import validation  # check, validate

        logger.debug("All command modules registered successfully")
    except ImportError as e:
        logger.error(f"Failed to register command modules: {e}")
        # Don't fail completely - some modules might work


# Register commands on import
register_commands()
