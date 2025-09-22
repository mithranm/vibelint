"""
AI-powered commands: justify, thinking-tokens.

These commands use LLM capabilities for code analysis and explanation.

vibelint/src/vibelint/cli/ai.py
"""

import logging
from pathlib import Path

import click
from rich.console import Console

from .core import VibelintContext, cli

console = Console()
logger = logging.getLogger(__name__)


@cli.command("justify")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--rule-id",
    help="Specific rule violation to justify",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactive justification session",
)
@click.pass_context
def justify(
    ctx: click.Context,
    file_path: Path,
    rule_id: str | None,
    interactive: bool,
) -> None:
    """
    Ask LLM to justify or explain code patterns.

    Uses configured LLM to provide explanations for:
    - Rule violations and why they might be acceptable
    - Complex code patterns and architectural decisions
    - Trade-offs in specific implementation choices

    Examples:
      vibelint justify src/complex_module.py
      vibelint justify --rule-id EMOJI-IN-STRING src/cli.py
      vibelint justify --interactive src/algorithm.py
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing"

    console.print("[bold purple]🤖 Generating Code Justification...[/bold purple]\n")

    # TODO: Move implementation from monolithic cli.py
    # This should delegate to the AI subsystem, not implement AI logic
    console.print("[yellow]⚠️  Justify command moved to modular structure[/yellow]")
    console.print(f"   File: {file_path}")
    console.print(f"   Rule: {rule_id or 'All violations'}")
    console.print(f"   Interactive: {interactive}")


@cli.command("debug-llm")
@click.option("--show-formats", is_flag=True, help="Show supported thinking token formats")
@click.option(
    "--detect",
    type=click.Path(exists=True, path_type=Path),
    help="Detect thinking tokens in a file",
)
@click.pass_context
def debug_llm(
    ctx: click.Context,
    show_formats: bool,
    detect: Path | None,
) -> None:
    """
    Debug and configure LLM thinking token removal.

    When LLMs analyze code, they often include internal reasoning (thinking tokens)
    that should be filtered out. This command helps configure that filtering.

    Examples:
      vibelint debug-llm --show-formats      # Show supported LLM formats
      vibelint debug-llm --detect file.py    # Detect tokens in file output
    """
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    console.print("[bold cyan]🔧 LLM Debug & Configuration[/bold cyan]\n")

    # TODO: Move implementation from monolithic cli.py
    # This should delegate to the LLM subsystem, not implement LLM logic
    console.print("[yellow]⚠️  Debug-llm command moved to modular structure[/yellow]")
    console.print(f"   Show formats: {show_formats}")
    console.print(f"   Detect file: {detect or 'None'}")
    
    if show_formats:
        console.print("\n[dim]Available formats: harmony (Claude), qwen, custom[/dim]")
    elif detect:
        console.print(f"\n[dim]Would analyze {detect} for thinking tokens[/dim]")
