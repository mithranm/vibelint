"""vibelint validators sub-package.

Modular validator system with centralized registry and discovery.

Responsibility: Validator module organization and re-exports only.
Individual validation logic belongs in specific validator modules.

vibelint/src/vibelint/validators/__init__.py
"""

# Import core types FIRST (before subdirectories to avoid circular imports)
from .types import (
    BaseFormatter,
    BaseValidator,
    Finding,
    Formatter,
    Severity,
    Validator,
    get_all_formatters,
    get_formatter,
    plugin_manager,
)

# Import registry system (also before subdirectories)
from .registry import get_all_validators, get_validator, register_validator, validator_registry

# Import validator categories for direct access (LAST to avoid circular imports)
from . import project_wide, single_file

# Note: Individual validators should be imported from their specific modules
# to prevent duplicate import paths and keep the module hierarchy clear.

__all__ = [
    # Core types
    "Severity",
    "Finding",
    "BaseValidator",
    "BaseFormatter",
    "Validator",
    "Formatter",
    "get_formatter",
    "get_all_formatters",
    "plugin_manager",
    # Registry system
    "validator_registry",
    "register_validator",
    "get_validator",
    "get_all_validators",
    # Category modules
    "single_file",
    "project_wide",
]
