"""
Dynamic CLI with plugin discovery for vibelint.

Automatically discovers and exposes all validators and workflows
through a unified command interface.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from .registry import PluginRegistry, get_plugin, list_plugins

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
@click.option("--exclude-ai", is_flag=True, help="Skip AI validators")
@click.option("--rules", help="Comma-separated rules to run")
@click.pass_context
def check(ctx: click.Context, targets: tuple[Path, ...], format: str, exclude_ai: bool, rules: Optional[str]) -> None:
    """Run configured validation rules."""
    # Import the existing validation logic
    from .validation import check as validation_check
    # Call the existing function
    validation_check(ctx, targets, format, exclude_ai, rules)


@cli.command("run")
@click.argument("plugin_name", type=str)
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(["human", "json"]), default="human", help="Output format")
@click.pass_context
def run_plugin(ctx: click.Context, plugin_name: str, targets: tuple[Path, ...], format: str) -> None:
    """Run a specific validator or workflow plugin."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    # Get the plugin
    plugin = get_plugin(plugin_name)
    if not plugin:
        available = list_plugins()
        console.print(f"[red]❌ Plugin '{plugin_name}' not found[/red]")
        console.print(f"Available plugins: {', '.join(available['validators'] + available['workflows'])}")
        ctx.exit(1)

    # Load config
    try:
        from ..config import load_config
        config = load_config(project_root) if project_root else {}
        config_dict = config.settings if hasattr(config, 'settings') else {}
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not load config: {e}[/yellow]")
        config_dict = {}

    # Determine targets
    if not targets:
        targets = [project_root] if project_root else [Path.cwd()]

    # Convert to list for plugin
    target_list = list(targets)

    try:
        console.print(f"[blue]🔧 Running {plugin.plugin_type} '{plugin_name}'...[/blue]")
        result = plugin.run(target_list, config_dict)

        if result.success:
            console.print(f"[green]✅ {plugin_name} completed successfully[/green]")
            if result.output_files:
                console.print(f"Output files: {', '.join(result.output_files)}")

            # Show summary
            if result.summary:
                for key, value in result.summary.items():
                    console.print(f"  {key}: {value}")

        else:
            console.print(f"[red]❌ {plugin_name} failed: {result.error_message}[/red]")
            ctx.exit(1)

    except Exception as e:
        console.print(f"[red]❌ Plugin execution failed: {e}[/red]")
        logger.error(f"Plugin {plugin_name} execution error: {e}")
        ctx.exit(1)


@cli.command("list")
@click.option("--type", "plugin_type", type=click.Choice(["validator", "workflow"]), help="Filter by plugin type")
@click.pass_context
def list_available_plugins(ctx: click.Context, plugin_type: Optional[str]) -> None:
    """List all available validators and workflows."""
    plugins_info = list_plugins()

    # Create table
    table = Table(title="Available Vibelint Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="white")

    # Get detailed plugin info
    all_plugins = PluginRegistry.get_plugins(plugin_type)

    for plugin in all_plugins:
        description = getattr(plugin, 'description', 'No description available')
        table.add_row(plugin.name, plugin.plugin_type, description)

    console.print(table)
    console.print(f"\nTotal: {len(all_plugins)} plugin(s)")


@cli.command("describe")
@click.argument("plugin_name", type=str)
@click.pass_context
def describe_plugin(ctx: click.Context, plugin_name: str) -> None:
    """Show detailed information about a specific plugin."""
    plugin = get_plugin(plugin_name)
    if not plugin:
        console.print(f"[red]❌ Plugin '{plugin_name}' not found[/red]")
        ctx.exit(1)

    console.print(f"[bold cyan]Plugin: {plugin.name}[/bold cyan]")
    console.print(f"Type: {plugin.plugin_type}")
    console.print(f"Version: {getattr(plugin, 'version', 'Unknown')}")
    console.print(f"Description: {getattr(plugin, 'description', 'No description available')}")

    if hasattr(plugin, 'tags') and plugin.tags:
        console.print(f"Tags: {', '.join(plugin.tags)}")

    # Show CLI options if available
    cli_options = plugin.get_cli_options()
    if cli_options:
        console.print("\n[bold]Available Options:[/bold]")
        for option in cli_options:
            console.print(f"  --{option.get('name', 'unknown')}: {option.get('help', 'No description')}")


# Auto-register dynamic commands for workflows with special syntax
@cli.group("workflow")
@click.pass_context
def workflow_group(ctx: click.Context) -> None:
    """Run specific workflows with workflow-specific options."""
    pass


# Dynamically add workflow commands
def _register_workflow_commands():
    """Register workflow-specific commands dynamically."""
    try:
        workflows = PluginRegistry.get_workflows()
        for workflow in workflows:
            # Create a dynamic command for each workflow
            @workflow_group.command(workflow.name)
            @click.argument("targets", nargs=-1, type=click.Path(path_type=Path))
            @click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory for reports")
            @click.pass_context
            def workflow_command(ctx: click.Context, targets: tuple[Path, ...], output_dir: Optional[Path], _workflow=workflow) -> None:
                f"""Run the {_workflow.name} workflow."""
                vibelint_ctx: VibelintContext = ctx.obj
                project_root = vibelint_ctx.project_root

                # Default to project root if no targets
                if not targets:
                    targets = [project_root] if project_root else [Path.cwd()]

                # Load config
                try:
                    from ..config import load_config
                    config = load_config(project_root) if project_root else {}
                    config_dict = config.settings if hasattr(config, 'settings') else {}
                except Exception:
                    config_dict = {}

                # Add output directory to config if specified
                if output_dir:
                    config_dict['output_dir'] = str(output_dir)

                try:
                    console.print(f"[blue]🔍 Running {_workflow.name} workflow...[/blue]")
                    result = _workflow.run(list(targets), config_dict)

                    if result.success:
                        console.print(f"[green]✅ {_workflow.name} completed[/green]")
                        if result.output_files:
                            console.print(f"Reports generated: {', '.join(result.output_files)}")
                    else:
                        console.print(f"[red]❌ {_workflow.name} failed: {result.error_message}[/red]")
                        ctx.exit(1)

                except Exception as e:
                    console.print(f"[red]❌ Workflow failed: {e}[/red]")
                    ctx.exit(1)

    except Exception as e:
        logger.error(f"Failed to register workflow commands: {e}")


# Register workflow commands on import
_register_workflow_commands()
