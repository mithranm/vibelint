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
from rich.table import Table

from .lint import LintRunner
from .config import load_config
from .namespace import (
    build_namespace_tree_representation, 
    get_namespace_collisions_str, 
    detect_namespace_collisions,
    detect_soft_member_collisions
)
from .report import generate_markdown_report


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
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def headers(
    path: str,
    check_only: bool,
    yes: bool,
    include_vcs_hooks: bool,
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
    sys.exit(exit_code)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--show-collisions",
    is_flag=True,
    help="Additionally show namespace collisions",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def namespace(
    path: str,
    include_vcs_hooks: bool,
    show_collisions: bool,
    paths: List[str],
):
    """Visualize the namespace tree of a Python project.
    
    Displays the hierarchical structure of modules and their members.
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]

    # Show namespace tree
    tree = build_namespace_tree_representation(target_paths, config)
    console.print(tree)
    
    # Optionally show collisions
    if show_collisions:
        collision_str = get_namespace_collisions_str(target_paths, config)
        if collision_str:
            console.print(collision_str)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "--ignore-inheritance",
    is_flag=True,
    help="Don't consider inheritance when detecting soft collisions",
)
@click.option(
    "--soft-only",
    is_flag=True,
    help="Only show soft collisions (member names reused in unrelated modules)",
)
@click.option(
    "--hard-only",
    is_flag=True,
    help="Only show hard collisions (namespace conflicts)",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def collisions(
    path: str,
    ignore_inheritance: bool,
    soft_only: bool,
    hard_only: bool,
    paths: List[str],
):
    """Detect namespace collisions in Python code.
    
    Finds both hard collisions (naming conflicts that break Python) and soft collisions
    (member names that appear in unrelated modules and may confuse humans or LLMs).
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)

    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]
    
    console.print("[bold]Checking for namespace collisions...[/bold]")
    
    # Detect collisions
    hard_collisions = [] if soft_only else detect_namespace_collisions(target_paths, config)
    soft_collisions = [] if hard_only else detect_soft_member_collisions(
        target_paths, config, use_inheritance_check=not ignore_inheritance
    )
    
    # Create summary table
    table = Table(title="Collision Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Hard Collisions", str(len(hard_collisions)))
    table.add_row("Soft Collisions", str(len(soft_collisions)))
    table.add_row("Total", str(len(hard_collisions) + len(soft_collisions)))
    
    console.print(table)
    
    # Display detailed results
    if not hard_collisions and not soft_collisions:
        console.print("[green]✓ No namespace collisions detected[/green]")
        sys.exit(0)
    
    # Show detailed reports
    if hard_collisions:
        console.print("\n[bold red]Hard Collisions:[/bold red]")
        console.print("[dim](These can break Python imports)[/dim]")
        for collision in hard_collisions:
            console.print(
                f"- [red]'{collision.name}'[/red] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
            )
            
    if soft_collisions:
        console.print("\n[bold yellow]Soft Collisions:[/bold yellow]")
        console.print("[dim](These don't break Python but may confuse humans and LLMs)[/dim]")
        for collision in soft_collisions:
            console.print(
                f"- [yellow]'{collision.name}'[/yellow] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
            )
    
    # Exit with error code if hard collisions found (soft collisions are just warnings)
    sys.exit(1 if hard_collisions else 0)


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
    help="Path to directory to analyze (default: current directory)",
)
@click.option(
    "-o", "--output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./vibelint_reports",
    help="Output directory for the report (default: ./vibelint_reports)",
)
@click.option(
    "-f", "--filename",
    type=str,
    default=None,
    help="Output filename for the report (default: vibelint_report_TIMESTAMP.md)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only report issues without suggesting fixes",
)
@click.option(
    "--include-vcs-hooks",
    is_flag=True,
    help="Include version control hooks in analysis",
)
@click.option(
    "--ignore-inheritance",
    is_flag=True,
    help="Don't consider inheritance when detecting soft collisions",
)
@click.argument("paths", nargs=-1, type=click.Path(exists=True, readable=True))
def report(
    path: str,
    output: str,
    filename: str,
    check_only: bool,
    include_vcs_hooks: bool,
    ignore_inheritance: bool,
    paths: List[str],
):
    """Generate a comprehensive markdown report of the codebase.
    
    The report includes linting errors, namespace structure, collisions, 
    and file contents organized by namespace hierarchy.
    
    If PATHS are provided, only those files/directories will be analyzed.
    Otherwise, all Python files under PATH will be analyzed.
    """
    root_path = Path(path).resolve()
    output_path = Path(output).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(root_path)
    
    # Use provided paths if available, otherwise use the root path
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]
    
    console.print(f"[bold]Generating comprehensive report for {len(target_paths)} path(s)...[/bold]")
    
    # Generate the report
    report_path = generate_markdown_report(
        target_paths=target_paths,
        output_dir=output_path,
        config=config,
        check_only=check_only,
        include_vcs_hooks=include_vcs_hooks,
        ignore_inheritance=ignore_inheritance,
        output_filename=filename
    )
    
    console.print(f"[green]✓ Report generated successfully at:[/green] {report_path}")
    return 0


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
