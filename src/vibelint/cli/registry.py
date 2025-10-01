"""
Dynamic plugin registry for validators and workflows.

Provides automatic discovery and registration of vibelint plugins,
enabling the CLI to dynamically expose all available functionality.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PluginResult:
    """Standard result format for all vibelint plugins."""
    success: bool
    plugin_name: str
    targets_processed: List[str]
    findings: List[Dict[str, Any]]
    summary: Dict[str, Any]
    output_files: List[str] = None
    error_message: Optional[str] = None


class VibelintPlugin(ABC):
    """Base interface for all vibelint plugins (validators and workflows)."""

    # Plugin metadata (must be defined by subclasses)
    name: str = None
    description: str = None
    plugin_type: str = None  # "validator" or "workflow"
    version: str = "1.0"
    tags: Set[str] = None

    def __init_subclass__(cls, **kwargs):
        """Automatically register plugins when they're defined."""
        super().__init_subclass__(**kwargs)
        if cls.name:  # Only register if name is defined
            PluginRegistry.register(cls())

    @abstractmethod
    def run(self, targets: List[Path], config: Dict[str, Any]) -> PluginResult:
        """Execute the plugin on the given targets."""
        pass

    def supports_file(self, file_path: Path) -> bool:
        """Check if this plugin can process the given file."""
        return file_path.suffix == '.py'

    def supports_directory(self, dir_path: Path) -> bool:
        """Check if this plugin can process the given directory."""
        return True

    def get_cli_options(self) -> List[Dict[str, Any]]:
        """Return CLI options specific to this plugin."""
        return []

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin-specific configuration."""
        return True


class PluginRegistry:
    """Central registry for all vibelint plugins."""

    _plugins: Dict[str, VibelintPlugin] = {}
    _initialized = False

    @classmethod
    def register(cls, plugin: VibelintPlugin):
        """Register a plugin in the registry."""
        if not plugin.name:
            logger.warning(f"Plugin {plugin.__class__.__name__} has no name, skipping registration")
            return

        if plugin.name in cls._plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, overwriting")

        cls._plugins[plugin.name] = plugin
        logger.debug(f"Registered plugin: {plugin.name} ({plugin.plugin_type})")

    @classmethod
    def get_plugin(cls, name: str) -> Optional[VibelintPlugin]:
        """Get a specific plugin by name."""
        cls._ensure_initialized()
        return cls._plugins.get(name)

    @classmethod
    def get_plugins(cls, plugin_type: Optional[str] = None) -> List[VibelintPlugin]:
        """Get all plugins, optionally filtered by type."""
        cls._ensure_initialized()
        plugins = list(cls._plugins.values())

        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]

        return sorted(plugins, key=lambda p: p.name)

    @classmethod
    def get_validators(cls) -> List[VibelintPlugin]:
        """Get all validator plugins."""
        return cls.get_plugins("validator")

    @classmethod
    def get_workflows(cls) -> List[VibelintPlugin]:
        """Get all workflow plugins."""
        return cls.get_plugins("workflow")

    @classmethod
    def list_plugins(cls) -> Dict[str, List[str]]:
        """Get a summary of all registered plugins."""
        cls._ensure_initialized()

        result = {
            "validators": [p.name for p in cls.get_validators()],
            "workflows": [p.name for p in cls.get_workflows()],
            "total": len(cls._plugins)
        }
        return result

    @classmethod
    def _ensure_initialized(cls):
        """Ensure plugins are discovered and registered."""
        if cls._initialized:
            return

        cls._discover_plugins()
        cls._initialized = True

    @classmethod
    def _discover_plugins(cls):
        """Discover and import all plugin modules to trigger registration."""
        logger.debug("Discovering vibelint plugins...")

        # Import known plugin modules to trigger auto-registration
        try:
            # Import validators (would need specific validator plugins here)
            # For now, validators use the existing plugin system
            logger.debug("Validators discovery skipped (using existing system)")
        except ImportError as e:
            logger.warning(f"Could not import validators: {e}")

        # Workflows are registered via workflow registry, not CLI plugins
        logger.debug("Workflows registered via workflow registry")

        logger.info(f"Plugin discovery complete: {len(cls._plugins)} plugins registered")


# Convenience functions for external use
def get_plugin(name: str) -> Optional[VibelintPlugin]:
    """Get a plugin by name."""
    return PluginRegistry.get_plugin(name)

def list_plugins() -> Dict[str, List[str]]:
    """List all available plugins."""
    return PluginRegistry.list_plugins()

def get_validators() -> List[VibelintPlugin]:
    """Get all validator plugins."""
    return PluginRegistry.get_validators()

def get_workflows() -> List[VibelintPlugin]:
    """Get all workflow plugins."""
    return PluginRegistry.get_workflows()