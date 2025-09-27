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

from ..config import Config, load_config
from .core import VibelintContext, cli

console = Console()
logger = logging.getLogger(__name__)


@cli.command("check")
@click.argument(
    "targets",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt for large directories.")
@click.option(
    "--output-format",
    type=click.Choice(["human", "json", "natural", "sarif", "llm"]),
    default="human",
    help="Output format for results.",
)
@click.option(
    "--categories",
    help="Comma-separated list of rule categories to run: core, static, ai, or 'all' for everything. Default: 'all'.",
)
@click.option(
    "--exclude-ai",
    is_flag=True,
    help="Exclude AI-powered validators (faster, no API calls).",
)
@click.option(
    "--rules",
    help="Comma-separated list of specific rules to run (overrides categories).",
)
@click.option(
    "--report",
    is_flag=True,
    help="Generate detailed analysis reports (saved to .vibelint-reports/).",
)
@click.pass_context
def check(
    ctx: click.Context,
    targets: tuple[Path, ...],
    yes: bool,
    output_format: str,
    categories: Optional[str],
    exclude_ai: bool,
    rules: Optional[str],
    report: bool,
) -> None:
    """
    Run vibelint validation on the specified targets.

    By default, runs ALL rules configured in pyproject.toml including AI validators.
    Use --exclude-ai for faster runs without API calls.

    TARGETS can be files or directories. If none specified, checks entire project.

    Examples:
      vibelint check                    # Check entire project
      vibelint check src/               # Check src directory
      vibelint check file.py            # Check single file
      vibelint check --exclude-ai       # Skip AI validators
      vibelint check --categories=core  # Only core rules
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing in check command"

    # Don't show UI messages for machine-readable formats
    if output_format in ["human", "natural"]:
        console.print("\n[bold magenta]Initiating Vibe Check...[/bold magenta]\n")

    logger.debug(f"Running 'check' command (yes={yes}, report={report})")

    # Load configuration using the file-based approach
    config: Config = load_config(project_root)
    if config.project_root is None:
        logger.error("Project root lost after config load. Aborting Vibe Check.")
        ctx.exit(1)

    # TODO: Implement the actual validation logic
    # This would move from the monolithic cli.py
    console.print("[yellow]⚠️  Check command implementation moved to modular structure[/yellow]")
    console.print(f"   Project root: {project_root}")
    console.print(f"   Config loaded: {config.is_present()}")
    console.print(f"   Targets: {targets if targets else 'entire project'}")
    console.print(f"   Format: {output_format}")
    console.print(f"   Categories: {categories or 'all'}")
    console.print(f"   Exclude AI: {exclude_ai}")


@cli.command("validate")
@click.argument(
    "path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-format",
    type=click.Choice(["human", "json", "natural"]),
    default="json",
    help="Output format for results (default: json for programmatic use).",
)
@click.option(
    "--recursive",
    is_flag=True,
    help="Recursively validate all Python files in directory.",
)
@click.pass_context
def validate_cmd(ctx: click.Context, path: Path, output_format: str, recursive: bool) -> None:
    """
    Validate a specific file or directory (optimized for single-file use).

    This command is designed for:
    - IDE integrations and editor plugins
    - CI/CD pipeline file-by-file validation
    - Pre-commit hooks and git hooks
    - Quick validation of individual files

    Unlike 'check', this command:
    - Focuses on single file or directory validation
    - Uses file-based config discovery (starts from file location)
    - Optimized for speed and minimal output
    - JSON output by default for programmatic use

    Examples:
      vibelint validate src/module.py           # Validate single file
      vibelint validate src/ --recursive        # Validate directory recursively
      vibelint validate file.py --output-format human  # Human-readable output
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    # For validate command, we use file-based config discovery
    # This matches how other linters work (ESLint, Black, etc.)
    from ..config import load_config

    # Start config search from the file/directory being validated
    start_path = path if path.is_dir() else path.parent
    config: Config = load_config(start_path)

    logger.debug(f"Validating: {path}")
    logger.debug(f"Config source: {config.project_root}")
    logger.debug(f"Recursive: {recursive}")

    # Import and use the single file validation workflow
    try:
        from ..workflows.implementations.single_file_validation import \
            SingleFileValidationWorkflow

        # Create workflow instance
        workflow = SingleFileValidationWorkflow(config)

        # Determine files to validate
        files_to_validate = []

        if path.is_file():
            if path.suffix == ".py":
                files_to_validate = [path]
            else:
                console.print(f"[bold red]Error:[/bold red] {path} is not a Python file")
                ctx.exit(1)
        else:
            # Directory validation
            if recursive:
                files_to_validate = list(path.rglob("*.py"))
            else:
                files_to_validate = list(path.glob("*.py"))

            if not files_to_validate:
                console.print("[bold red]Error:[/bold red] No Python files found to validate")
                ctx.exit(1)

        # Validate all files
        results = []
        total_violations = 0
        failed_files = 0

        for file_path in files_to_validate:
            try:
                import time

                start_time = time.time()
                result = workflow.validate_file(file_path)
                execution_time = (time.time() - start_time) * 1000

                file_result = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "health_score": result.health_score,
                    "violations": [v.to_dict() for v in result.violations],
                    "execution_time_ms": execution_time,
                    "success": True,
                    "error": None,
                }
                results.append(file_result)
                total_violations += len(result.violations)

                if result.violations:
                    failed_files += 1

            except Exception as e:
                failed_files += 1
                file_result = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "health_score": 0.0,
                    "violations": [],
                    "execution_time_ms": 0.0,
                    "success": False,
                    "error": str(e),
                }
                results.append(file_result)
                console.print(f"[bold red]Error validating {file_path}:[/bold red] {e}")

        # Display results
        if output_format == "json":
            import json

            output = {
                "summary": {
                    "total_files": len(results),
                    "failed_files": failed_files,
                    "total_violations": total_violations,
                },
                "files": results,
            }
            print(json.dumps(output, indent=2))
        else:
            # Natural format display
            total_files = len(results)

            if total_files == 1:
                # Single file display
                result = results[0]
                console.print(f"\nFile: {result['file_name']}")
                console.print(f"Health Score: {result['health_score']}/100")
                console.print(f"Execution Time: {result['execution_time_ms']:.1f}ms")

                if result["success"] and not result["violations"]:
                    console.print("\n[green]No violations found![/green]")
                elif result["violations"]:
                    console.print(f"\n[red]Found {len(result['violations'])} violation(s):[/red]")
                    for violation in result["violations"]:
                        console.print(f"  - {violation}")

                if not result["success"]:
                    console.print(f"[bold red]Error:[/bold red] {result['error']}")
            else:
                # Multi-file summary
                console.print(f"\nRunning validation on {total_files} Python files...")
                console.print("\nValidation Summary")
                console.print(f"Files processed: {total_files}")
                console.print(f"Failed files: {failed_files}")
                console.print(f"Total violations: {total_violations}")

                if failed_files == 0 and total_violations == 0:
                    console.print("\n[green]All files passed validation![/green]")
                else:
                    console.print(
                        f"\n[red]{failed_files} files failed, {total_violations} violations found[/red]"
                    )

        # Determine exit code
        exit_code = 1 if failed_files > 0 or total_violations > 0 else 0
        ctx.exit(exit_code)

    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Missing module for validation: {e}")
        ctx.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Unexpected error during validation: {e}")
        if logger.level == logging.DEBUG:
            import traceback

            console.print(traceback.format_exc())
        ctx.exit(1)
