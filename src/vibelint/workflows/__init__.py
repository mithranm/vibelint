"""
Extensible workflow system for vibelint.

Provides modular, composable workflows for complex code analysis tasks
with built-in evaluation and metrics collection capabilities.

vibelint/src/vibelint/workflows/__init__.py
"""

from .base import BaseWorkflow, WorkflowResult, WorkflowConfig, WorkflowMetrics
from .justification import FileJustificationWorkflow
from ..workflow.registry import workflow_registry, register_workflow

# Register built-in workflows
register_workflow(FileJustificationWorkflow)

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowConfig",
    "WorkflowMetrics",
    "FileJustificationWorkflow",
    "workflow_registry",
    "register_workflow",
]
