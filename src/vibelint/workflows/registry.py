"""
Workflow registry for managing available workflows.

Provides centralized registration and discovery of workflows with
metadata and dependency information.

vibelint/src/vibelint/workflows/registry.py
"""

import logging
from typing import Dict, Type, Optional, List

from .base import BaseWorkflow

logger = logging.getLogger(__name__)

__all__ = ["WorkflowRegistry", "workflow_registry", "register_workflow"]


class WorkflowRegistry:
    """Registry for managing available workflows."""

    def __init__(self):
        self._workflows: Dict[str, Type[BaseWorkflow]] = {}
        self._metadata: Dict[str, Dict] = {}

    def register(self, workflow_class: Type[BaseWorkflow]) -> None:
        """Register a workflow class."""
        # Create temporary instance to get metadata
        temp_instance = workflow_class()
        workflow_id = temp_instance.workflow_id

        if not workflow_id:
            raise ValueError(f"Workflow {workflow_class.__name__} must define workflow_id")

        if workflow_id in self._workflows:
            logger.warning(f"Overwriting existing workflow: {workflow_id}")

        self._workflows[workflow_id] = workflow_class

        # Store metadata
        self._metadata[workflow_id] = {
            "name": temp_instance.name,
            "description": temp_instance.description,
            "category": temp_instance.category,
            "version": temp_instance.version,
            "tags": list(temp_instance.tags),
            "required_inputs": list(temp_instance.get_required_inputs()),
            "produced_outputs": list(temp_instance.get_produced_outputs()),
            "dependencies": temp_instance.get_dependencies(),
            "supports_parallel": temp_instance.supports_parallel_execution(),
            "class_name": workflow_class.__name__
        }

        logger.debug(f"Registered workflow: {workflow_id}")

    def get_workflow(self, workflow_id: str) -> Optional[Type[BaseWorkflow]]:
        """Get workflow class by ID."""
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> Dict[str, Type[BaseWorkflow]]:
        """Get all registered workflows."""
        return self._workflows.copy()

    def get_workflows_by_category(self, category: str) -> Dict[str, Type[BaseWorkflow]]:
        """Get workflows by category."""
        filtered = {}
        for workflow_id, metadata in self._metadata.items():
            if metadata["category"] == category:
                filtered[workflow_id] = self._workflows[workflow_id]
        return filtered

    def get_workflows_by_tag(self, tag: str) -> Dict[str, Type[BaseWorkflow]]:
        """Get workflows by tag."""
        filtered = {}
        for workflow_id, metadata in self._metadata.items():
            if tag in metadata["tags"]:
                filtered[workflow_id] = self._workflows[workflow_id]
        return filtered

    def get_workflow_metadata(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow metadata."""
        return self._metadata.get(workflow_id)

    def list_workflow_ids(self) -> List[str]:
        """List all workflow IDs."""
        return list(self._workflows.keys())

    def validate_dependencies(self, workflow_ids: List[str]) -> Dict[str, List[str]]:
        """Validate workflow dependencies."""
        missing_deps = {}

        for workflow_id in workflow_ids:
            if workflow_id not in self._workflows:
                missing_deps[workflow_id] = [f"Workflow '{workflow_id}' not found"]
                continue

            metadata = self._metadata[workflow_id]
            for dep_id in metadata["dependencies"]:
                if dep_id not in self._workflows:
                    if workflow_id not in missing_deps:
                        missing_deps[workflow_id] = []
                    missing_deps[workflow_id].append(f"Missing dependency: '{dep_id}'")

        return missing_deps

    def unregister(self, workflow_id: str) -> bool:
        """Unregister a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            del self._metadata[workflow_id]
            logger.debug(f"Unregistered workflow: {workflow_id}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered workflows."""
        self._workflows.clear()
        self._metadata.clear()
        logger.debug("Cleared all workflows from registry")


# Global registry instance
workflow_registry = WorkflowRegistry()


def register_workflow(workflow_class: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
    """Decorator for registering workflows."""
    workflow_registry.register(workflow_class)
    return workflow_class


# Auto-register built-in workflows
def _register_builtin_workflows():
    """Register built-in workflows."""
    try:
        from .redundancy_detection import RedundancyDetectionWorkflow
        workflow_registry.register(RedundancyDetectionWorkflow)
    except ImportError as e:
        logger.warning(f"Failed to register built-in workflow: {e}")


# Register built-ins on module import
_register_builtin_workflows()