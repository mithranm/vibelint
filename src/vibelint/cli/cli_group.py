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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """vibelint: Code quality linter."""
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


# Import essential command modules
def register_commands():
    """Import essential command modules."""
    try:
        from . import validation  # check
        logger.debug("Command modules registered successfully")
    except ImportError as e:
        logger.error(f"Failed to register command modules: {e}")

# Register commands on import
register_commands()
