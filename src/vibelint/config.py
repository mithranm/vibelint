"""
Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/config.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

if sys.version_info >= (3, 11):

    import tomllib
else:

    try:

        import tomli as tomllib
    except ImportError:

        print(
            "Error: vibelint requires Python 3.11+ or the 'tomli' package "
            "to parse pyproject.toml on Python 3.10."
            "\nHint: Try running: pip install tomli"
        )
        sys.exit(1)


from .utils import find_package_root

logger = logging.getLogger(__name__)


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

    vibelint/config.py
    """

    def __init__(self, project_root: Optional[Path], config_dict: Dict[str, Any]):
        """
        Initializes Config.

        vibelint/config.py
        """
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Optional[Path]:
        """
        The detected project root directory, or None if not found.

        vibelint/config.py
        """
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Any]:
        """
        Read-only view of the settings loaded from [tool.vibelint].

        vibelint/config.py
        """
        return self._config_dict

    @property
    def ignore_codes(self) -> List[str]:
        """
        Returns the list of error codes to ignore, from config or empty list.

        vibelint/config.py
        """
        ignored = self.get("ignore", [])
        if isinstance(ignored, list) and all(isinstance(item, str) for item in ignored):
            return ignored
        elif ignored:
            logger.warning(
                "Configuration key 'ignore' in [tool.vibelint] is not a list of strings. Ignoring it."
            )
            return []
        else:
            return []

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the loaded settings, returning default if not found.

        vibelint/config.py
        """
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Gets a value, raising KeyError if the key is not found.

        vibelint/config.py
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

        vibelint/config.py
        """
        return key in self._config_dict

    def is_present(self) -> bool:
        """
        Checks if a project root was found and some settings were loaded.

        vibelint/config.py
        """
        return self._project_root is not None and bool(self._config_dict)


def load_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration *only* from the nearest pyproject.toml file.

    Searches upwards from start_path. If pyproject.toml or the [tool.vibelint]
    section isn't found or is invalid, returns a Config object with project_root
    (if found) but an empty settings dictionary.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object. Check `config.project_root` and `config.settings`.

    vibelint/config.py
    """
    project_root = find_package_root(start_path)
    loaded_settings: Dict[str, Any] = {}

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

        vibelint_config = full_toml_config.get("tool", {}).get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            if loaded_settings:
                logger.info(f"Loaded [tool.vibelint] settings from {pyproject_path}")
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
    except Exception as e:

        logger.exception(f"Unexpected error loading config from {pyproject_path}: {e}")

    return Config(project_root=project_root, config_dict=loaded_settings)


__all__ = ["Config", "load_config"]
