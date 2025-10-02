"""Workflow management subsystem for vibelint.

Modular workflow system with centralized registry and clear separation:
- core/: Base classes, registry, orchestration
- implementations/: Actual workflow implementations

Responsibility: Workflow module organization and re-exports only.
Individual workflow logic belongs in specific implementation modules.

vibelint/src/vibelint/workflow/__init__.py
"""

# Avoid importing implementations directly to prevent circular imports
# Access implementations through lazy loading
# Import core workflow system
from .core.base import (
    BaseWorkflow,
    WorkflowConfig,
    WorkflowMetrics,
    WorkflowPriority,
    WorkflowResult,
    WorkflowStatus,
)

# Import registry system
from .registry import WorkflowRegistry, register_workflow, workflow_registry


# Lazy imports for specific implementations to avoid circular dependencies
def get_justification_engine():
    """Get JustificationEngine class."""
    from .implementations.justification import JustificationEngine

    return JustificationEngine


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
    # Lazy import functions
    "get_justification_engine",
]
