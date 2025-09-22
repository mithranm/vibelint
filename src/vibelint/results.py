"""
Module for vibelint/results.py.

vibelint/src/vibelint/results.py
"""

from dataclasses import dataclass, field
from pathlib import Path

from .plugin_system import Finding
from .validators.project_wide.namespace_collisions import (NamespaceCollision,
                                                           NamespaceNode)

__all__ = ["CheckResult", "CommandResult", "NamespaceResult", "SnapshotResult"]


@dataclass
class CommandResult:
    """
    Base class for command results.

    vibelint/src/vibelint/results.py
    """

    success: bool = True
    error_message: str | None = None
    exit_code: int = 0

    def __post_init__(self):
        """
        Set exit code based on success if not explicitly set.

        vibelint/src/vibelint/results.py
        """

        if not self.success and self.exit_code == 0:
            self.exit_code = 1


@dataclass
class CheckResult(CommandResult):
    """
    Result data from the 'check' command.

    vibelint/src/vibelint/results.py
    """

    findings: list[Finding] = field(default_factory=list)
    hard_collisions: list[NamespaceCollision] = field(default_factory=list)
    global_soft_collisions: list[NamespaceCollision] = field(default_factory=list)
    local_soft_collisions: list[NamespaceCollision] = field(default_factory=list)
    report_path: Path | None = None
    report_generated: bool = False
    report_error: str | None = None


@dataclass
class NamespaceResult(CommandResult):
    """
    Result data from the 'namespace' command.

    vibelint/src/vibelint/results.py
    """

    root_node: NamespaceNode | None = None
    intra_file_collisions: list[NamespaceCollision] = field(default_factory=list)
    output_path: Path | None = None
    output_saved: bool = False
    output_error: str | None = None


@dataclass
class SnapshotResult(CommandResult):
    """
    Result data from the 'snapshot' command.

    vibelint/src/vibelint/results.py
    """

    output_path: Path | None = None
