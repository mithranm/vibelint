#!/usr/bin/env python3
"""
Command-line interface for vibelint.

src/vibelint/cli.py
"""

import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from .config import load_config
from .lint import LintRunner
from .namespace import (
    detect_namespace_collisions,
    detect_soft_member_collisions
)
from .archive import tape_archive
from .report import generate_markdown_report

console = Console()

@click.group()
@click.version_option()
def cli():
    """
    vibelint - A linting tool to make Python codebases more LLM-friendly.
    
    src/vibelint/cli.py
    """
    pass


@cli.command()
@click.option("--path", default=".", type=click.Path(exists=True),
    help="Path to analyze (default: current directory)")
@click.option("--check-only", is_flag=True, help="Check for violations without fixing them")
@click.option("--yes", is_flag=True, help="Skip confirmation for large directories")
@click.option("--include-vcs-hooks", is_flag=True, help="Include VCS directories (like .git) in analysis")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def headers(path: str, check_only: bool, yes: bool, include_vcs_hooks: bool, paths: List[str]):
    """
    Lint and fix Python docstrings, shebangs, encoding cookies.
    Ensures docstrings exist for modules/classes/functions/methods, 
    referencing file path and import path.

    src/vibelint/cli.py
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)
    if paths:
        target_paths = [Path(p).resolve() for p in paths]
    else:
        target_paths = [root_path]

    runner = LintRunner(
        config=config,
        check_only=check_only,
        skip_confirmation=yes,
        include_vcs_hooks=include_vcs_hooks
    )
    exit_code = runner.run(target_paths)
    sys.exit(exit_code)


@cli.command()
@click.option("--path", default=".", type=click.Path(exists=True),
    help="Path to analyze (default: current directory)")
@click.option("--ignore-inheritance", is_flag=True, help="Ignore class inheritance for soft collisions")
@click.option("--soft-only", is_flag=True, help="Show only soft collisions")
@click.option("--hard-only", is_flag=True, help="Show only hard collisions")
@click.option("--include-vcs-hooks", is_flag=True, help="Include VCS directories in analysis")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def collisions(path: str, ignore_inheritance: bool, soft_only: bool,
               hard_only: bool, include_vcs_hooks: bool, paths: List[str]):
    """
    Detect and report hard & soft namespace collisions in Python code.

    src/vibelint/cli.py
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)
    if paths:
        target_paths = [Path(p).resolve() for p in paths]
    else:
        target_paths = [root_path]

    console.print("[bold]Checking for namespace collisions...[/bold]")
    if not soft_only:
        hard_coll = detect_namespace_collisions(target_paths, config, include_vcs_hooks=include_vcs_hooks)
    else:
        hard_coll = []

    if not hard_only:
        soft_coll = detect_soft_member_collisions(
            target_paths,
            config,
            use_inheritance_check=not ignore_inheritance,
            include_vcs_hooks=include_vcs_hooks
        )
    else:
        soft_coll = []

    table = Table(title="Collision Results Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_row("Hard Collisions", str(len(hard_coll)))
    table.add_row("Soft Collisions", str(len(soft_coll)))
    table.add_row("Total", str(len(hard_coll) + len(soft_coll)))
    console.print(table)

    if not hard_coll and not soft_coll:
        console.print("[green]No namespace collisions detected.[/green]")
        sys.exit(0)

    if hard_coll:
        console.print("\n[bold red]Hard Collisions:[/bold red]")
        for hc in hard_coll:
            console.print(f"- '{hc.name}' in {hc.path1} vs {hc.path2} ({hc.collision_type})")

    if soft_coll:
        console.print("\n[bold yellow]Soft Collisions:[/bold yellow]")
        for sc in soft_coll:
            console.print(f"- '{sc.name}' in {sc.path1} vs {sc.path2} ({sc.collision_type})")

    sys.exit(1)


@cli.command(name="archive")
@click.option("--path", default=".", type=click.Path(exists=True),
    help="Path to analyze (default: current directory)")
@click.option("-o", "--output", default="code_tape_archive.md", type=click.Path(writable=True),
    help="Output markdown file for the tape archive.")
@click.option("--include-vcs-hooks", is_flag=True, help="Include .git, .hg, .svn, etc.")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def tape_archive_cmd(path: str, output: str, include_vcs_hooks: bool, paths: List[str]):
    """
    Generate a markdown tape archive of the code, with a filesystem tree up top,
    then each file's contents (unless it's 'peeked' in config).

    src/vibelint/cli.py
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)
    if paths:
        target_paths = [Path(p).resolve() for p in paths]
    else:
        target_paths = [root_path]

    tape_archive(target_paths, config, Path(output), include_vcs_hooks=include_vcs_hooks)


# Optional: a "report" command if you want a comprehensive markdown report
@cli.command()
@click.option("--path", default=".", type=click.Path(exists=True),
    help="Path to analyze (default: current directory)")
@click.option("-o", "--output", default="./vibelint_reports", type=click.Path(writable=True),
    help="Output directory for the report.")
@click.option("--filename", default=None, type=str, help="Output filename (defaults to vibelint_report_TIMESTAMP.md)")
@click.option("--ignore-inheritance", is_flag=True, help="Ignore class inheritance for soft collisions")
@click.option("--include-vcs-hooks", is_flag=True, help="Include .git, .hg, .svn, etc.")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def report(path: str, output: str, filename: Optional[str], ignore_inheritance: bool,
           include_vcs_hooks: bool, paths: List[str]):
    """
    Generate a comprehensive markdown report (docstring lint, collisions, file contents).
    """
    root_path = Path(path).resolve()
    config = load_config(root_path)
    target_paths = [Path(p).resolve() for p in paths] if paths else [root_path]

    out_dir = Path(output).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    from .report import generate_markdown_report
    report_path = generate_markdown_report(
        target_paths=target_paths,
        output_dir=out_dir,
        config=config,
        check_only=True,
        include_vcs_hooks=include_vcs_hooks,
        ignore_inheritance=ignore_inheritance,
        output_filename=filename
    )
    console.print(f"[green]✓ Report generated at {report_path}[/green]")
    sys.exit(0)


def main():
    """
    Entry point for vibelint CLI.

    src/vibelint/cli.py
    """
    cli()


if __name__ == "__main__":
    main()
