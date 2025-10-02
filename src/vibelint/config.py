"""Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/src/vibelint/config.py
"""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from vibelint.filesystem import find_package_root, walk_up_for_config

logger = logging.getLogger(__name__)


def _find_config_file(project_root: Path) -> Path | None:
    """Find the config file (pyproject.toml or dev.pyproject.toml) with vibelint settings."""
    # Check standard pyproject.toml first
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "tool" in data and "vibelint" in data.get("tool", {}):
                    return pyproject_path
        except Exception:
            pass

    # Check dev.pyproject.toml (kaia pattern)
    dev_pyproject_path = project_root / "dev.pyproject.toml"
    if dev_pyproject_path.exists():
        try:
            with open(dev_pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "tool" in data and "vibelint" in data.get("tool", {}):
                    return dev_pyproject_path
        except Exception:
            pass

    return None


def _load_parent_config(project_root: Path, current_config_path: Path) -> dict | None:
    """Load parent configuration for inheritance."""
    # Walk up from the project root to find parent configurations
    parent_path = project_root.parent

    while parent_path != parent_path.parent:  # Stop at filesystem root
        # Check for dev.pyproject.toml (kaia pattern)
        dev_config = parent_path / "dev.pyproject.toml"
        if dev_config.exists() and dev_config != current_config_path:
            try:
                with open(dev_config, "rb") as f:
                    data = tomllib.load(f)
                    vibelint_config = data.get("tool", {}).get("vibelint", {})
                    if vibelint_config:
                        logger.debug(f"Found parent config in {dev_config}")
                        return vibelint_config
            except Exception:
                pass

        # Check for regular pyproject.toml
        parent_config = parent_path / "pyproject.toml"
        if parent_config.exists() and parent_config != current_config_path:
            try:
                with open(parent_config, "rb") as f:
                    data = tomllib.load(f)
                    vibelint_config = data.get("tool", {}).get("vibelint", {})
                    if vibelint_config:
                        logger.debug(f"Found parent config in {parent_config}")
                        return vibelint_config
            except Exception:
                pass

        parent_path = parent_path.parent

    return None


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
    """Holds the vibelint configuration loaded *exclusively* from pyproject.toml.

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
        """Initializes Config.

        vibelint/src/vibelint/config.py
        """
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Path | None:
        """The detected project root directory, or None if not found.

        vibelint/src/vibelint/config.py
        """
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Union[str, bool, int, list, dict]]:
        """Read-only view of the settings loaded from [tool.vibelint].

        vibelint/src/vibelint/config.py
        """
        return self._config_dict

    @property
    def ignore_codes(self) -> list[str]:
        """Returns the list of error codes to ignore, from config or empty list.

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
        """Gets a value from the loaded settings, returning default if not found.

        vibelint/src/vibelint/config.py
        """
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Union[str, bool, int, list, dict]:
        """Gets a value, raising KeyError if the key is not found.

        vibelint/src/vibelint/config.py
        """
        if key not in self._config_dict:
            raise KeyError(
                f"Required configuration key '{key}' not found in "
                f"[tool.vibelint] section of pyproject.toml."
            )
        return self._config_dict[key]

    def __contains__(self, key: str) -> bool:
        """Checks if a key exists in the loaded settings.

        vibelint/src/vibelint/config.py
        """
        return key in self._config_dict

    def is_present(self) -> bool:
        """Checks if a project root was found and some settings were loaded.

        vibelint/src/vibelint/config.py
        """
        return self._project_root is not None and bool(self._config_dict)


def load_hierarchical_config(start_path: Path) -> Config:
    """Loads vibelint configuration with hierarchical merging.

    1. Loads local config (file patterns, local settings)
    2. Walks up to find parent config (LLM settings, shared config)
    3. Merges them: local config takes precedence for file patterns,
       parent config provides LLM settings

    Args:
    start_path: The directory to start searching from.

    Returns:
    A Config object with merged local and parent settings.

    """
    # Find local config first
    local_root = find_package_root(start_path)
    local_settings = {}

    if local_root:
        pyproject_path = local_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    local_settings = data.get("tool", {}).get("vibelint", {})
                    if local_settings:
                        logger.info(f"Loaded local vibelint config from {pyproject_path}")
            except Exception as e:
                logger.warning(f"Failed to load local config from {pyproject_path}: {e}")

    # Walk up to find parent config with LLM settings
    parent_settings = {}
    current_path = start_path.parent if start_path.is_file() else start_path

    while current_path.parent != current_path:
        parent_pyproject = current_path / "pyproject.toml"
        if parent_pyproject.exists() and parent_pyproject != (
            local_root / "pyproject.toml" if local_root else None
        ):
            try:
                with open(parent_pyproject, "rb") as f:
                    data = tomllib.load(f)
                    parent_config = data.get("tool", {}).get("vibelint", {})
                    if parent_config:
                        parent_settings = parent_config
                        logger.info(f"Found parent vibelint config at {parent_pyproject}")
                        break
            except Exception as e:
                logger.debug(f"Failed to read {parent_pyproject}: {e}")
        current_path = current_path.parent

    # Merge configs: local file patterns override parent, but inherit LLM settings
    merged_settings = parent_settings.copy()

    # Local config takes precedence for file discovery patterns
    if local_settings:
        for key in ["include_globs", "exclude_globs", "ignore"]:
            if key in local_settings:
                merged_settings[key] = local_settings[key]

        # Also copy other local-specific settings
        for key in local_settings:
            if key not in ["include_globs", "exclude_globs", "ignore"]:
                merged_settings[key] = local_settings[key]

    return Config(local_root or start_path, merged_settings)


def load_config(start_path: Path) -> Config:
    """Loads vibelint configuration with auto-discovery fallback.

    First tries manual config from pyproject.toml, then falls back to
    zero-config auto-discovery for seamless single->multi-project scaling.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object with either manual or auto-discovered settings.

    vibelint/src/vibelint/config.py

    """
    project_root = walk_up_for_config(start_path)
    loaded_settings: dict[str, Any] = {}

    # Try auto-discovery first for zero-config scaling
    try:
        from vibelint.auto_discovery import discover_and_configure

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

    # Try both pyproject.toml and dev.pyproject.toml
    pyproject_path = _find_config_file(project_root)
    logger.debug(f"Found project root: {project_root}")

    if not pyproject_path:
        logger.debug(f"No vibelint configuration found in {project_root}")
        return Config(project_root, {})

    logger.debug(f"Attempting to load config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:
            full_toml_config = tomllib.load(f)
        logger.debug(f"Parsed {pyproject_path.name}")

        # Validate required configuration structure explicitly
        tool_section = full_toml_config.get("tool")
        if not isinstance(tool_section, dict):
            logger.warning("pyproject.toml [tool] section is missing or invalid")
            vibelint_config = {}
        else:
            vibelint_config = tool_section.get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            # Check for parent config inheritance
            parent_config = _load_parent_config(project_root, pyproject_path)
            if parent_config:
                # Merge parent config with local config (local takes precedence)
                merged_settings = parent_config.copy()
                merged_settings.update(loaded_settings)
                loaded_settings = merged_settings
                logger.debug("Merged parent configuration")

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


# Configuration constants for LLM
DEFAULT_FAST_TEMPERATURE = 0.1
DEFAULT_FAST_MAX_TOKENS = 2048
DEFAULT_ORCHESTRATOR_TEMPERATURE = 0.2
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 8192
DEFAULT_CONTEXT_THRESHOLD = 3000


@dataclass
class LLMConfig:
    """Typed LLM configuration with explicit validation."""

    # Required fields first
    fast_api_url: str
    fast_model: str
    orchestrator_api_url: str
    orchestrator_model: str

    # Optional fields with defaults
    fast_backend: str = "vllm"
    fast_temperature: float = DEFAULT_FAST_TEMPERATURE
    fast_max_tokens: int = DEFAULT_FAST_MAX_TOKENS
    fast_max_context_tokens: Optional[int] = None
    fast_api_key: Optional[str] = None

    orchestrator_backend: str = "llamacpp"
    orchestrator_temperature: float = DEFAULT_ORCHESTRATOR_TEMPERATURE
    orchestrator_max_tokens: int = DEFAULT_ORCHESTRATOR_MAX_TOKENS
    orchestrator_max_context_tokens: Optional[int] = None
    orchestrator_api_key: Optional[str] = None

    context_threshold: int = DEFAULT_CONTEXT_THRESHOLD
    enable_context_probing: bool = True
    enable_fallback: bool = False

    def __post_init__(self):
        """Validate required configuration."""
        if not self.fast_api_url:
            raise ValueError("fast_api_url is required - configure in [tool.vibelint.llm]")
        if not self.fast_model:
            raise ValueError("fast_model is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_api_url:
            raise ValueError("orchestrator_api_url is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_model:
            raise ValueError("orchestrator_model is required - configure in [tool.vibelint.llm]")


@dataclass
class EmbeddingConfig:
    """Typed embedding configuration with explicit validation."""

    # Required fields first
    code_api_url: str
    natural_api_url: str

    # Optional fields with defaults
    code_model: str = "text-embedding-ada-002"
    natural_model: str = "text-embedding-ada-002"
    use_specialized_embeddings: bool = True

    def __post_init__(self):
        """Validate required configuration."""
        if not self.code_api_url:
            raise ValueError("code_api_url is required - configure in [tool.vibelint.embeddings]")
        if not self.natural_api_url:
            raise ValueError(
                "natural_api_url is required - configure in [tool.vibelint.embeddings]"
            )


def _get_env_float(key: str) -> Optional[float]:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_env_int(key: str) -> Optional[int]:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None


def get_llm_config(config: Optional[Config] = None) -> LLMConfig:
    """Get typed LLM configuration for vibelint.

    Args:
        config: Optional Config object. If None, loads from current directory.

    Returns:
        Validated LLMConfig object with environment variable overrides

    """
    if config is None:
        config = load_config(Path.cwd())

    llm_dict = config.get("llm", {})
    if not isinstance(llm_dict, dict):
        llm_dict = {}

    # Build kwargs with environment overrides
    kwargs = {
        # Fast LLM
        "fast_api_url": (
            os.getenv("VIBELINT_FAST_LLM_API_URL")
            or os.getenv("FAST_LLM_API_URL")
            or llm_dict.get("fast_api_url")
        ),
        "fast_model": (
            os.getenv("VIBELINT_FAST_LLM_MODEL")
            or os.getenv("FAST_LLM_MODEL")
            or llm_dict.get("fast_model")
        ),
        "fast_backend": (
            os.getenv("VIBELINT_FAST_LLM_BACKEND")
            or os.getenv("FAST_LLM_BACKEND")
            or llm_dict.get("fast_backend", "vllm")
        ),
        "fast_api_key": (
            os.getenv("VIBELINT_FAST_LLM_API_KEY")
            or os.getenv("FAST_LLM_API_KEY")
            or llm_dict.get("fast_api_key")
        ),
        "fast_temperature": (
            _get_env_float("VIBELINT_FAST_LLM_TEMPERATURE")
            or _get_env_float("FAST_LLM_TEMPERATURE")
            or llm_dict.get("fast_temperature", DEFAULT_FAST_TEMPERATURE)
        ),
        "fast_max_tokens": (
            _get_env_int("VIBELINT_FAST_LLM_MAX_TOKENS")
            or _get_env_int("FAST_LLM_MAX_TOKENS")
            or llm_dict.get("fast_max_tokens", DEFAULT_FAST_MAX_TOKENS)
        ),
        "fast_max_context_tokens": llm_dict.get("fast_max_context_tokens"),
        # Orchestrator LLM
        "orchestrator_api_url": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_URL")
            or os.getenv("ORCHESTRATOR_LLM_API_URL")
            or llm_dict.get("orchestrator_api_url")
        ),
        "orchestrator_model": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_MODEL")
            or os.getenv("ORCHESTRATOR_LLM_MODEL")
            or llm_dict.get("orchestrator_model")
        ),
        "orchestrator_backend": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_BACKEND")
            or os.getenv("ORCHESTRATOR_LLM_BACKEND")
            or llm_dict.get("orchestrator_backend", "llamacpp")
        ),
        "orchestrator_api_key": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_KEY")
            or os.getenv("ORCHESTRATOR_LLM_API_KEY")
            or llm_dict.get("orchestrator_api_key")
        ),
        "orchestrator_temperature": (
            _get_env_float("VIBELINT_ORCHESTRATOR_LLM_TEMPERATURE")
            or _get_env_float("ORCHESTRATOR_LLM_TEMPERATURE")
            or llm_dict.get("orchestrator_temperature", DEFAULT_ORCHESTRATOR_TEMPERATURE)
        ),
        "orchestrator_max_tokens": (
            _get_env_int("VIBELINT_ORCHESTRATOR_LLM_MAX_TOKENS")
            or _get_env_int("ORCHESTRATOR_LLM_MAX_TOKENS")
            or llm_dict.get("orchestrator_max_tokens", DEFAULT_ORCHESTRATOR_MAX_TOKENS)
        ),
        "orchestrator_max_context_tokens": llm_dict.get("orchestrator_max_context_tokens"),
        # Routing configuration
        "context_threshold": (
            _get_env_int("VIBELINT_LLM_CONTEXT_THRESHOLD")
            or _get_env_int("LLM_CONTEXT_THRESHOLD")
            or llm_dict.get("context_threshold", DEFAULT_CONTEXT_THRESHOLD)
        ),
        "enable_context_probing": llm_dict.get("enable_context_probing", True),
        "enable_fallback": llm_dict.get("enable_fallback", False),
    }

    return LLMConfig(**kwargs)


def get_embedding_config(config: Optional[Config] = None) -> EmbeddingConfig:
    """Get typed embedding configuration for vibelint.

    Args:
        config: Optional Config object. If None, loads from current directory.

    Returns:
        Validated EmbeddingConfig object

    """
    if config is None:
        config = load_config(Path.cwd())

    embedding_dict = config.get("embeddings", {})
    if not isinstance(embedding_dict, dict):
        embedding_dict = {}

    kwargs = {
        "code_api_url": embedding_dict.get("code_api_url"),
        "natural_api_url": embedding_dict.get("natural_api_url"),
        "code_model": embedding_dict.get("code_model", "text-embedding-ada-002"),
        "natural_model": embedding_dict.get("natural_model", "text-embedding-ada-002"),
        "use_specialized_embeddings": embedding_dict.get("use_specialized_embeddings", True),
    }

    return EmbeddingConfig(**kwargs)


__all__ = [
    "Config",
    "load_config",
    "LLMConfig",
    "EmbeddingConfig",
    "get_llm_config",
    "get_embedding_config",
]
