"""
Modular CLI architecture for vibelint.

This package breaks down the monolithic CLI into logical, maintainable modules:

- core.py: Main CLI group and shared utilities
- validation.py: check, validate commands
- analysis.py: namespace, snapshot, diagnostics commands
- maintenance.py: regen-docstrings, rollback, setup commands
- ai.py: thinking-tokens, justify commands
- presentation.py: Result formatting and display utilities

Each module is focused on a specific domain, making the codebase more maintainable
and easier to extend.

vibelint/src/vibelint/cli/__init__.py
"""

import sys
import logging
from .core import cli, VibelintContext

__all__ = ["cli", "main"]


def main() -> None:
    """
    Main entry point for the vibelint CLI application.
    
    This function provides the entry point specified in pyproject.toml.
    """
    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except (RuntimeError, ValueError, OSError, ImportError) as e:
        from rich.console import Console
        console = Console()
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        
        logger = logging.getLogger(__name__)
        # Check if logger was configured before logging error
        if logger.hasHandlers():
            logger.error("Unhandled exception in CLI execution.", exc_info=True)
        else:
            # Fallback if error happened before logging setup
            import traceback
            traceback.print_exc()
        sys.exit(1)
