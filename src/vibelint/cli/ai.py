"""
AI-powered commands: justify, thinking-tokens.

These commands use LLM capabilities for code analysis and explanation.

vibelint/src/vibelint/cli/ai.py
"""

import logging
from pathlib import Path

import click
from rich.console import Console

from .cli_group import VibelintContext, cli

console = Console()
logger = logging.getLogger(__name__)


@cli.command("justify")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--rule-id", help="Specific rule to justify")
@click.pass_context
def justify(ctx: click.Context, file_path: Path, rule_id: str | None) -> None:
    """Justify code decisions using static analysis and minimal LLM calls."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing"

    console.print("[bold purple]🤖 Analyzing Code Justification...[/bold purple]\n")

    try:
        from ..workflows.implementations.justification import JustificationEngine
        from ..config import load_config

        # Load config
        config = load_config(project_root)

        # Create justification engine
        engine = JustificationEngine(config)

        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Perform justification analysis
        result = engine.justify_file(file_path, content)

        # Save session logs
        detailed_log, summary_log = engine.save_session_logs(str(file_path), result)

        # Display results
        console.print(f"[green]✅ Analyzed {file_path}[/green]")
        console.print(f"Quality Score: {result.quality_score:.1%}")
        console.print(f"[dim]Logs saved to: {summary_log}[/dim]")

        if rule_id:
            # Filter to specific rule if requested
            relevant_justifications = [
                j for j in result.justifications
                if rule_id.lower() in j.justification.lower()
            ]
            console.print(f"\nJustifications for rule '{rule_id}':")
            for just in relevant_justifications:
                console.print(f"  • {just.element_name}: {just.justification}")
        else:
            # Show all justifications
            console.print(f"\nFound {len(result.justifications)} code elements:")
            for just in result.justifications:
                confidence_color = "green" if just.confidence > 0.7 else "yellow" if just.confidence > 0.4 else "red"
                console.print(f"  [{confidence_color}]•[/] {just.element_name} ({just.element_type}): {just.justification}")

        if result.redundancies_found:
            console.print(f"\n[yellow]⚠️ Potential redundancies:[/yellow]")
            for redundancy in result.redundancies_found:
                console.print(f"  • {redundancy}")

        if result.recommendations:
            console.print(f"\n[blue]💡 Recommendations:[/blue]")
            for rec in result.recommendations:
                console.print(f"  • {rec}")

    except Exception as e:
        console.print(f"[red]❌ Justification failed: {e}[/red]")
        logger.error(f"Justification error: {e}")


@cli.command("compare")
@click.argument("method1", type=str)
@click.argument("method2", type=str)
@click.pass_context
def compare_methods(ctx: click.Context, method1: str, method2: str) -> None:
    """Compare two methods for similarity using fast LLM."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root
    assert project_root is not None, "Project root missing"

    console.print("[bold cyan]🔍 Comparing Methods...[/bold cyan]\n")

    try:
        from ..workflows.implementations.justification import JustificationEngine
        from ..config import load_config

        # Parse method specifications (file:method format)
        try:
            path1, name1 = method1.rsplit(":", 1)
            path2, name2 = method2.rsplit(":", 1)
        except ValueError:
            console.print("[red]❌ Use format: file_path:method_name[/red]")
            return

        # Load config and create engine
        config = load_config(project_root)
        engine = JustificationEngine(config)

        # Compare methods
        result = engine.justify_method_comparison(path1, name1, path2, name2)

        # Display results
        if result["similar"]:
            console.print(f"[yellow]⚠️ Methods appear similar (confidence: {result['confidence']:.1%})[/yellow]")
        else:
            console.print(f"[green]✅ Methods are distinct (confidence: {result['confidence']:.1%})[/green]")

        console.print(f"Reasoning: {result.get('reason', result.get('reasoning', 'No explanation'))}")

    except Exception as e:
        console.print(f"[red]❌ Comparison failed: {e}[/red]")
        logger.error(f"Method comparison error: {e}")


@cli.command("status")
@click.pass_context
def llm_status(ctx: click.Context) -> None:
    """Show LLM configuration status."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    console.print("[bold cyan]🔧 LLM Configuration Status[/bold cyan]\n")

    try:
        from ..config import load_config
        from ..llm import create_llm_manager

        config = load_config(project_root)
        llm_manager = create_llm_manager(config)

        if llm_manager:
            status = llm_manager.get_status()
            console.print(f"Fast LLM: {'✅ Available' if status['fast_configured'] else '❌ Not configured'}")
            console.print(f"Orchestrator LLM: {'✅ Available' if status['orchestrator_configured'] else '❌ Not configured'}")
        else:
            console.print("[red]❌ No LLM configuration found[/red]")

    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
