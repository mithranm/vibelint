"""
Workflow registry for managing available workflows.

Provides centralized registration and discovery of workflows with
metadata and dependency information.

Responsibility: Workflow discovery and registration only.
Workflow logic belongs in individual workflow implementation modules.

vibelint/src/vibelint/workflow/registry.py
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
        self._metadata: Dict[str, Dict] = {}
        self._builtin_loaded = False
        self._plugins_loaded = False

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
            "class_name": workflow_class.__name__,
        }

        logger.debug(f"Registered workflow: {workflow_id}")

    def get_workflow(self, workflow_id: str) -> Optional[Type[BaseWorkflow]]:
        """Get workflow class by ID."""
        self.ensure_loaded()
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> Dict[str, Type[BaseWorkflow]]:
        """Get all registered workflows."""
        self.ensure_loaded()
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
        self.ensure_loaded()
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
        self._builtin_loaded = False
        self._plugins_loaded = False
        logger.debug("Cleared all workflows from registry")

    def _load_builtin_workflows(self) -> None:
        """Load built-in workflows directly."""
        if self._builtin_loaded:
            return

        # Import and register built-in workflows
        try:
            from .implementations.justification import JustificationWorkflow
            self.register(JustificationWorkflow)
            logger.debug("Registered built-in workflow: justification")
        except ImportError as e:
            logger.warning(f"Failed to load built-in justification workflow: {e}")

        try:
            from .implementations.deadcode import DeadcodeWorkflow
            self.register(DeadcodeWorkflow)
            logger.debug("Registered built-in workflow: deadcode")
        except ImportError as e:
            logger.warning(f"Failed to load built-in deadcode workflow: {e}")

        self._builtin_loaded = True

    def _load_plugin_workflows(self) -> None:
        """Load plugin workflows from entry points."""
        if self._plugins_loaded:
            return

        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points("vibelint.workflows"):
                try:
                    workflow_class = entry_point.load()
                    self.register(workflow_class)
                    logger.debug(f"Registered plugin workflow: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin workflow {entry_point.name}: {e}")
        except ImportError:
            logger.debug("pkg_resources not available, skipping plugin workflows")

        self._plugins_loaded = True

    def ensure_loaded(self) -> None:
        """Ensure both built-in and plugin workflows are loaded."""
        self._load_builtin_workflows()
        self._load_plugin_workflows()


# Global registry instance
workflow_registry = WorkflowRegistry()


def register_workflow(workflow_class: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
    """Decorator for registering workflows."""
    workflow_registry.register(workflow_class)
    return workflow_class


# Note: Built-in workflows are now registered via workflow_registry._load_builtin_workflows()
# Plugin workflows are loaded via workflow_registry._load_plugin_workflows()
# This ensures clean separation between built-in and external workflows
