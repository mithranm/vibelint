"""Validator registry and discovery system.

Provides centralized registration and discovery of validators with
automatic loading from entry points and modular organization.

Responsibility: Validator discovery and registration only.
Validation logic belongs in individual validator modules.

vibelint/src/vibelint/validators/registry.py
"""

import importlib.metadata
import logging
from typing import Dict, List, Optional, Type

from .types import BaseValidator

logger = logging.getLogger(__name__)


class ValidatorRegistry:
    """Registry for managing available validators."""

    def __init__(self):
        self._validators: Dict[str, Type[BaseValidator]] = {}
        self._loaded = False

    def register_validator(self, validator_class: Type[BaseValidator]) -> None:
        """Register a validator class."""
        if not issubclass(validator_class, BaseValidator):
            raise ValueError(f"Validator {validator_class} must inherit from BaseValidator")

        rule_id = validator_class.rule_id
        if rule_id in self._validators:
            logger.warning(f"Overriding existing validator: {rule_id}")

        self._validators[rule_id] = validator_class
        logger.debug(f"Registered validator: {rule_id}")

    def get_validator(self, rule_id: str) -> Optional[Type[BaseValidator]]:
        """Get a validator by rule ID."""
        if not self._loaded:
            self._load_all_validators()
        return self._validators.get(rule_id)

    def get_all_validators(self) -> Dict[str, Type[BaseValidator]]:
        """Get all registered validators."""
        if not self._loaded:
            self._load_all_validators()
        return self._validators.copy()

    def get_validators_by_category(self, category: str) -> Dict[str, Type[BaseValidator]]:
        """Get validators by category (single_file, project_wide, architecture)."""
        if not self._loaded:
            self._load_all_validators()

        filtered = {}
        for rule_id, validator_class in self._validators.items():
            # Determine category from module path
            module_path = validator_class.__module__
            if f".{category}." in module_path:
                filtered[rule_id] = validator_class
        return filtered

    def list_rule_ids(self) -> List[str]:
        """List all available rule IDs."""
        if not self._loaded:
            self._load_all_validators()
        return list(self._validators.keys())

    def _load_all_validators(self) -> None:
        """Load all validators from entry points and built-ins."""
        if self._loaded:
            return

        # Load from entry points
        self._load_entry_point_validators()

        # Load built-in validators
        self._load_builtin_validators()

        self._loaded = True
        logger.info(f"Loaded {len(self._validators)} validators")

    def _load_entry_point_validators(self) -> None:
        """Load validators from entry points."""
        try:
            for entry_point in importlib.metadata.entry_points(group="vibelint.validators"):
                try:
                    validator_class = entry_point.load()
                    self.register_validator(validator_class)
                except Exception as e:
                    logger.warning(f"Failed to load validator {entry_point.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load entry point validators: {e}")

    def _load_builtin_validators(self) -> None:
        """Load built-in validators from modules."""
        builtin_modules = [
            # Single-file validators
            "vibelint.validators.single_file.dict_get_fallback",
            "vibelint.validators.single_file.emoji",
            "vibelint.validators.single_file.exports",
            "vibelint.validators.single_file.logger_names",
            "vibelint.validators.single_file.typing_quality",
            "vibelint.validators.single_file.self_validation",
            "vibelint.validators.single_file.strict_config",
            # Project-wide validators
            "vibelint.validators.project_wide.api_consistency",
            "vibelint.validators.project_wide.namespace_collisions",
        ]

        for module_name in builtin_modules:
            try:
                module = importlib.import_module(module_name)

                # Look for get_validators function
                if hasattr(module, "get_validators"):
                    validators = module.get_validators()
                    for validator_class in validators:
                        self.register_validator(validator_class)

                # Look for individual validator classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseValidator)
                        and attr != BaseValidator
                        and hasattr(attr, "rule_id")
                    ):
                        self.register_validator(attr)

            except ImportError as e:
                logger.debug(f"Could not import builtin validator module {module_name}: {e}")
            except Exception as e:
                logger.warning(f"Error loading validators from {module_name}: {e}")


# Global registry instance
validator_registry = ValidatorRegistry()


# Convenience functions
def register_validator(validator_class: Type[BaseValidator]) -> None:
    """Register a validator class with the global registry."""
    validator_registry.register_validator(validator_class)


def get_validator(rule_id: str) -> Optional[Type[BaseValidator]]:
    """Get a validator by rule ID from the global registry."""
    return validator_registry.get_validator(rule_id)


def get_all_validators() -> Dict[str, Type[BaseValidator]]:
    """Get all validators from the global registry."""
    return validator_registry.get_all_validators()
