"""
Workflow registry for managing available workflows.

Simple registration system for BaseWorkflow subclasses.
Workflows must properly inherit from BaseWorkflow and implement required methods.

vibelint/src/vibelint/workflows/registry.py
"""

import logging
from typing import Dict, List, Optional, Type

from .core.base import BaseWorkflow

logger = logging.getLogger(__name__)

__all__ = ["WorkflowRegistry", "workflow_registry", "register_workflow"]


class WorkflowRegistry:
    """Registry for managing available workflows."""

    def __init__(self):
        self._workflows: Dict[str, Type[BaseWorkflow]] = {}
        self._loaded = False

    def register(self, workflow_class: Type[BaseWorkflow]) -> None:
        """Register a workflow class that inherits from BaseWorkflow."""
        if not issubclass(workflow_class, BaseWorkflow):
            raise TypeError(f"{workflow_class.__name__} must inherit from BaseWorkflow")

        # Create temporary instance to get workflow_id
        temp_instance = workflow_class()
        workflow_id = temp_instance.workflow_id

        if not workflow_id:
            raise ValueError(f"Workflow {workflow_class.__name__} must define workflow_id")

        if workflow_id in self._workflows:
            logger.warning(f"Overwriting existing workflow: {workflow_id}")

        self._workflows[workflow_id] = workflow_class
        logger.debug(f"Registered workflow: {workflow_id}")

    def get_workflow(self, workflow_id: str) -> Optional[Type[BaseWorkflow]]:
        """Get workflow class by ID."""
        self._ensure_loaded()
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> Dict[str, Type[BaseWorkflow]]:
        """Get all registered workflows."""
        self._ensure_loaded()
        return self._workflows.copy()

    def list_workflow_ids(self) -> List[str]:
        """List all workflow IDs."""
        self._ensure_loaded()
        return list(self._workflows.keys())

    def unregister(self, workflow_id: str) -> bool:
        """Unregister a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            logger.debug(f"Unregistered workflow: {workflow_id}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered workflows."""
        self._workflows.clear()
        self._loaded = False
        logger.debug("Cleared all workflows from registry")

    def _ensure_loaded(self) -> None:
        """Ensure workflows are loaded."""
        if self._loaded:
            return

        # Auto-discover workflows in implementations/ directory
        import importlib
        from pathlib import Path

        try:
            implementations_path = Path(__file__).parent / "implementations"
            if implementations_path.exists():
                for py_file in implementations_path.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue

                    module_name = f"vibelint.workflows.implementations.{py_file.stem}"
                    try:
                        module = importlib.import_module(module_name)

                        # Find all BaseWorkflow subclasses in the module
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseWorkflow)
                                and attr is not BaseWorkflow
                            ):
                                try:
                                    self.register(attr)
                                except (TypeError, ValueError) as e:
                                    logger.debug(
                                        f"Skipping {attr_name} from {module_name}: {e}"
                                    )

                    except ImportError as e:
                        logger.debug(f"Could not import {module_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to auto-discover workflows: {e}")

        self._loaded = True


# Global registry instance
workflow_registry = WorkflowRegistry()


def register_workflow(workflow_class: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
    """Decorator for registering workflows."""
    workflow_registry.register(workflow_class)
    return workflow_class
