#!/usr/bin/env python3
"""
Command-line interface for vibelint.

vibelint/cli.py
"""

import sys
from pathlib import Path
from typing import List

import click
from rich.console import Console

from .lint import LintRunner
from .config import load_config
from .namespace import generate_namespace_representation


console = Console()


@click.group()
@click.version_option()
def cli():
    """vibelint - A linting tool to make Python codebases more LLM-friendly."""
    pass


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Check for violations without fixing them",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation for large directories",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--show-namespace",
    is_flag=True,
    help="Generate and display the namespace representation",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def headers(
    path: str,
    check_only: bool,
    yes: bool,
    include_vcs_hooks: bool,
    show_namespace: bool,
    paths: List[str],
):
    """Lint and fix Python module headers.

    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]

    lint_runner = LintRunner(
        config=config,
        check_only=check_only,
        skip_confirmation=yes,
        include_vcs_hooks=include_vcs_hooks,
    )

    exit_code = lint_runner.run(target_paths)

    if show_namespace:
        namespace_rep = generate_namespace_representation(target_paths, config)
        console.print(namespace_rep)

    sys.exit(exit_code)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
