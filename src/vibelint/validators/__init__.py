"""
vibelint validators sub-package.

Modular validator system with centralized registry and discovery.

Responsibility: Validator module organization and re-exports only.
Individual validation logic belongs in specific validator modules.

vibelint/src/vibelint/validators/__init__.py
"""

# Import validator categories for direct access
from . import architecture, project_wide, single_file

# Import registry system
from .registry import (get_all_validators, get_validator, register_validator,
                       validator_registry)

# Note: Individual validators should be imported from their specific modules:
# from vibelint.validators.single_file.absolute_imports import AbsoluteImportValidator
# from vibelint.validators.architecture.basic_patterns import ArchitectureValidator
# etc.
#
# This prevents duplicate import paths and keeps the module hierarchy clear.

__all__ = [
    # Registry system
    "validator_registry",
    "register_validator",
    "get_validator",
    "get_all_validators",
    # Category modules
    "single_file",
    "project_wide",
    "architecture",
]
