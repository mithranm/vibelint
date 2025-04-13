"""
Module for vibelint/results.py.

vibelint/results.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .lint import LintResult
from .namespace import NamespaceCollision, NamespaceNode

__all__ = ["CheckResult", "CommandResult", "NamespaceResult", "SnapshotResult"]


@dataclass
class CommandResult:
    """
    Base class for command results.

    vibelint/results.py
    """

    success: bool = True
    error_message: Optional[str] = None
    exit_code: int = 0

    def __post_init__(self):
        """
        Set exit code based on success if not explicitly set.

        vibelint/results.py
        """

        if not self.success and self.exit_code == 0:
            self.exit_code = 1


@dataclass
class CheckResult(CommandResult):
    """
    Result data from the 'check' command.

    vibelint/results.py
    """

    lint_results: List[LintResult] = field(default_factory=list)
    hard_collisions: List[NamespaceCollision] = field(default_factory=list)
    global_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    local_soft_collisions: List[NamespaceCollision] = field(default_factory=list)
    report_path: Optional[Path] = None
    report_generated: bool = False
    report_error: Optional[str] = None


@dataclass
class NamespaceResult(CommandResult):
    """
    Result data from the 'namespace' command.

    vibelint/results.py
    """

    root_node: Optional[NamespaceNode] = None
    intra_file_collisions: List[NamespaceCollision] = field(default_factory=list)
    output_path: Optional[Path] = None
    output_saved: bool = False
    output_error: Optional[str] = None


@dataclass
class SnapshotResult(CommandResult):
    """
    Result data from the 'snapshot' command.

    vibelint/results.py
    """

    output_path: Optional[Path] = None
