"""
Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/src/vibelint/config.py
"""

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from .utils import find_package_root

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):

    import tomllib
else:

    try:

        import tomli as tomllib
    except ImportError as e:

        raise ImportError(
            "vibelint requires Python 3.11+ or the 'tomli' package "
            "to parse pyproject.toml on Python 3.10. "
            "Hint: Try running: pip install tomli"
        ) from e


class Config:
    """
    Holds the vibelint configuration loaded *exclusively* from pyproject.toml.

    Provides access to the project root and the raw configuration dictionary.
    It does *not* provide default values for missing keys. Callers must
    check for the existence of required settings.

    Attributes:
    project_root: The detected root of the project containing pyproject.toml.
    Can be None if pyproject.toml is not found.
    settings: A read-only view of the dictionary loaded from the
    [tool.vibelint] section of pyproject.toml. Empty if the
    file or section is missing or invalid.

    vibelint/src/vibelint/config.py
    """

    def __init__(self, project_root: Path | None, config_dict: dict[str, Any]):
        """
        Initializes Config.

        vibelint/src/vibelint/config.py
        """
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Path | None:
        """
        The detected project root directory, or None if not found.

        vibelint/src/vibelint/config.py
        """
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Union[str, bool, int, list, dict]]:
        """
        Read-only view of the settings loaded from [tool.vibelint].

        vibelint/src/vibelint/config.py
        """
        return self._config_dict

    @property
    def ignore_codes(self) -> list[str]:
        """
        Returns the list of error codes to ignore, from config or empty list.

        vibelint/src/vibelint/config.py
        """
        ignored = self.get("ignore", [])
        if isinstance(ignored, list) and all(isinstance(item, str) for item in ignored):
            return ignored

        # Handle invalid configuration
        if ignored:
            logger.warning(
                "Configuration key 'ignore' in [tool.vibelint] is not a list of strings. Ignoring it."
            )

        return []

    def get(
        self, key: str, default: Union[str, bool, int, list, dict, None] = None
    ) -> Union[str, bool, int, list, dict, None]:
        """
        Gets a value from the loaded settings, returning default if not found.

        vibelint/src/vibelint/config.py
        """
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Union[str, bool, int, list, dict]:
        """
        Gets a value, raising KeyError if the key is not found.

        vibelint/src/vibelint/config.py
        """
        if key not in self._config_dict:
            raise KeyError(
                f"Required configuration key '{key}' not found in "
                f"[tool.vibelint] section of pyproject.toml."
            )
        return self._config_dict[key]

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key exists in the loaded settings.

        vibelint/src/vibelint/config.py
        """
        return key in self._config_dict

    def is_present(self) -> bool:
        """
        Checks if a project root was found and some settings were loaded.

        vibelint/src/vibelint/config.py
        """
        return self._project_root is not None and bool(self._config_dict)


def load_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration with auto-discovery fallback.

    First tries manual config from pyproject.toml, then falls back to
    zero-config auto-discovery for seamless single->multi-project scaling.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object with either manual or auto-discovered settings.

    vibelint/src/vibelint/config.py
    """
    project_root = find_package_root(start_path)
    loaded_settings: dict[str, Any] = {}

    # Try auto-discovery first for zero-config scaling
    try:
        from .auto_discovery import discover_and_configure

        auto_config = discover_and_configure(start_path)

        # If we found a multi-project setup, use auto-discovery by default
        if auto_config.get("discovered_topology") == "multi_project":
            logger.info(f"Auto-discovered multi-project setup from {start_path}")
            # Convert auto-discovered config to vibelint config format
            loaded_settings = _convert_auto_config_to_vibelint(auto_config)
            project_root = project_root or start_path

            # Still allow manual config to override auto-discovery
            manual_override = _load_manual_config(project_root)
            if manual_override:
                logger.debug("Manual config found, merging with auto-discovery")
                loaded_settings.update(manual_override)

            return Config(project_root=project_root, config_dict=loaded_settings)

    except ImportError:
        logger.debug("Auto-discovery not available, using manual config only")
    except Exception as e:
        logger.debug(f"Auto-discovery failed: {e}, falling back to manual config")

    if not project_root:
        logger.warning(
            f"Could not find project root (pyproject.toml) searching from '{start_path}'. "
            "No configuration will be loaded."
        )
        return Config(project_root=None, config_dict=loaded_settings)

    pyproject_path = project_root / "pyproject.toml"
    logger.debug(f"Found project root: {project_root}")
    logger.debug(f"Attempting to load config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:

            full_toml_config = tomllib.load(f)
        logger.debug("Parsed pyproject.toml")

        # Validate required configuration structure explicitly
        tool_section = full_toml_config.get("tool")
        if not isinstance(tool_section, dict):
            logger.warning("pyproject.toml [tool] section is missing or invalid")
            vibelint_config = {}
        else:
            vibelint_config = tool_section.get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            if loaded_settings:
                logger.debug(f"Loaded [tool.vibelint] settings from {pyproject_path}")
                logger.debug(f"Loaded settings: {loaded_settings}")
            else:
                logger.info(
                    f"Found {pyproject_path}, but the [tool.vibelint] section is empty or missing."
                )
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table (dictionary). "
                "Ignoring this section."
            )

    except FileNotFoundError:

        logger.error(
            f"pyproject.toml not found at {pyproject_path} despite project root detection."
        )
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}. Using empty configuration.")
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}. Using empty configuration.")
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error processing configuration from {pyproject_path}: {e}")
        logger.debug("Unexpected error loading config", exc_info=True)

    return Config(project_root=project_root, config_dict=loaded_settings)


def _convert_auto_config_to_vibelint(auto_config: dict[str, Any]) -> dict[str, Any]:
    """Convert auto-discovered config to vibelint config format."""
    vibelint_config = {}

    # Auto-route validation based on discovered services
    services = auto_config.get("services", {})
    routing = auto_config.get("auto_routing", {})

    # Set include globs based on discovered projects
    include_globs = []
    for service_info in services.values():
        service_path = Path(service_info["path"])
        include_globs.extend([f"{service_path.name}/src/**/*.py", f"{service_path.name}/**/*.py"])

    vibelint_config["include_globs"] = include_globs

    # Configure distributed services if available
    if auto_config.get("discovered_topology") == "multi_project":
        vibelint_config["distributed"] = {
            "enabled": True,
            "auto_discovered": True,
            "services": services,
            "routing": routing,
        }

        # Use shared resources if discovered
        shared_resources = auto_config.get("shared_resources", {})
        if shared_resources.get("vector_stores"):
            vibelint_config["vector_store"] = {
                "backend": "qdrant",
                "qdrant_collection": shared_resources["vector_stores"][0],
            }

    return vibelint_config


def _load_manual_config(project_root: Path | None) -> dict[str, Any]:
    """Load manual configuration from pyproject.toml."""
    if not project_root:
        return {}

    pyproject_path = project_root / "pyproject.toml"
    logger.debug(f"Attempting to load manual config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:
            full_toml_config = tomllib.load(f)
        logger.debug("Parsed pyproject.toml")

        # Validate required configuration structure explicitly
        tool_section = full_toml_config.get("tool")
        if not isinstance(tool_section, dict):
            logger.warning("pyproject.toml [tool] section is missing or invalid")
            return {}

        vibelint_config = tool_section.get("vibelint", {})

        if isinstance(vibelint_config, dict):
            if vibelint_config:
                logger.debug(f"Loaded manual [tool.vibelint] settings from {pyproject_path}")
                return vibelint_config
            else:
                logger.debug(f"Found {pyproject_path}, but [tool.vibelint] section is empty")
                return {}
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table. Ignoring."
            )
            return {}

    except FileNotFoundError:
        logger.debug(f"No pyproject.toml found at {pyproject_path}")
        return {}
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}")
        return {}
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading manual config: {e}")
        return {}


__all__ = ["Config", "load_config"]
