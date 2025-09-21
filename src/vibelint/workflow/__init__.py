"""Workflow management subsystem for vibelint.

This package contains all workflow-related functionality:
- Workflow manager for execution coordination
- Registry for workflow discovery and management
- Orchestrator for complex workflow execution
- Evaluation tools for workflow analysis

All workflow components have been organized into this subpackage
to improve code organization and maintainability.
"""

from .manager import WorkflowManager
from .registry import WorkflowRegistry
from .orchestrator import AnalysisOrchestrator
from .evaluation import WorkflowEvaluator

__all__ = [
    "WorkflowManager",
    "WorkflowRegistry",
    "AnalysisOrchestrator",
    "WorkflowEvaluator",
]