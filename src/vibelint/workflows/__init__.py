"""
Extensible workflow system for vibelint.

Provides modular, composable workflows for complex code analysis tasks
with built-in evaluation and metrics collection capabilities.

vibelint/src/vibelint/workflows/__init__.py
"""

from .base import BaseWorkflow, WorkflowResult, WorkflowConfig, WorkflowMetrics
from .manager import WorkflowManager
from .evaluation import WorkflowEvaluator, EvaluationResult
from .registry import workflow_registry, register_workflow

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowConfig",
    "WorkflowMetrics",
    "WorkflowManager",
    "WorkflowEvaluator",
    "EvaluationResult",
    "workflow_registry",
    "register_workflow",
]
