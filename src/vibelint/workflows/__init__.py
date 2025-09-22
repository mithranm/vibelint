"""
Workflow management subsystem for vibelint.

Modular workflow system with centralized registry and clear separation:
- core/: Base classes, registry, orchestration
- implementations/: Actual workflow implementations

Responsibility: Workflow module organization and re-exports only.
Individual workflow logic belongs in specific implementation modules.

vibelint/src/vibelint/workflow/__init__.py
"""

# Import implementation categories for direct access
from . import implementations
# Import core workflow system
from .core.base import (BaseWorkflow, WorkflowConfig, WorkflowMetrics,
                        WorkflowPriority, WorkflowResult, WorkflowStatus)
from .evaluation import WorkflowEvaluator
# Import orchestration
from .manager import WorkflowManager
from .orchestrator import AnalysisOrchestrator
# Import registry system
from .registry import WorkflowRegistry, register_workflow, workflow_registry

# Import specific implementations for backward compatibility
try:
    from .implementations.justification import FileJustificationWorkflow
    from .implementations.single_file_validation import \
        SingleFileValidationWorkflow
except ImportError:
    # These might not exist yet or have import issues
    pass

__all__ = [
    # Core workflow system
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowConfig",
    "WorkflowMetrics",
    "WorkflowStatus",
    "WorkflowPriority",
    # Registry system
    "WorkflowRegistry",
    "workflow_registry",
    "register_workflow",
    # Orchestration
    "WorkflowManager",
    "AnalysisOrchestrator",
    "WorkflowEvaluator",
    # Implementation modules
    "implementations",
]
