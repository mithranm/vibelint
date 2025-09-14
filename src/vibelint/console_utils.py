"""
Shared console utilities for vibelint.

Provides a centralized Rich Console instance to avoid duplication.

vibelint/console_utils.py
"""

from rich.console import Console

__all__ = ["console"]

# Global console instance used throughout vibelint
console = Console()
