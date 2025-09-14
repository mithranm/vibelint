"""
Rule management system for vibelint.

Handles rule configuration, severity overrides, and policy management.
"""

from typing import Dict, List, Optional, Set
from .plugin_system import Severity, BaseValidator, plugin_manager

__all__ = ["RuleEngine", "create_default_rule_config"]


class RuleEngine:
    """Manages rule configuration and policy decisions."""

    def __init__(self, config: Dict):
        """
        Initialize rule engine with configuration.

        Args:
            config: Configuration dictionary from pyproject.toml
        """
        self.config = config
        self._rule_overrides: Dict[str, Severity] = {}
        self._enabled_plugins: Set[str] = set()
        self._shared_models = {}  # Cache for expensive models like EmbeddingGemma
        self._load_rule_config()

    def _load_rule_config(self):
        """Load rule configuration from config."""
        # Load rule severity overrides
        rules_config = self.config.get("rules", {})
        for rule_id, setting in rules_config.items():
            if isinstance(setting, str):
                try:
                    self._rule_overrides[rule_id] = Severity(setting.upper())
                except ValueError:
                    # Invalid severity, skip
                    pass
            elif isinstance(setting, bool):
                # Boolean: True=default severity, False=OFF
                if not setting:
                    self._rule_overrides[rule_id] = Severity.OFF

        # Load enabled plugins
        plugins_config = self.config.get("plugins", {})
        enabled = plugins_config.get("enabled", ["vibelint.core"])
        if isinstance(enabled, list):
            self._enabled_plugins.update(enabled)
        elif isinstance(enabled, str):
            self._enabled_plugins.add(enabled)

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled (not set to OFF)."""
        severity = self._rule_overrides.get(rule_id)
        return severity != Severity.OFF if severity else True

    def get_rule_severity(self, rule_id: str, default: Severity = Severity.WARN) -> Severity:
        """Get effective severity for a rule."""
        # Primary: semantic rule IDs
        severity = self._rule_overrides.get(rule_id)
        if severity is not None:
            return severity

        return default

    def create_validator_instance(
        self, validator_class: type[BaseValidator]
    ) -> Optional[BaseValidator]:
        """
        Create validator instance with configured severity.

        Args:
            validator_class: Validator class to instantiate

        Returns:
            Validator instance or None if rule is disabled
        """
        if not self.is_rule_enabled(validator_class.rule_id):
            return None

        severity = self.get_rule_severity(validator_class.rule_id, validator_class.default_severity)

        # Handle special cases that need shared resources
        if validator_class.rule_id == "SEMANTIC-SIMILARITY":
            shared_model = self._get_or_create_embedding_model()
            return validator_class(severity=severity, config=self.config, shared_model=shared_model)

        return validator_class(severity=severity, config=self.config)

    def get_enabled_validators(self) -> List[BaseValidator]:
        """Get all enabled validator instances."""
        validators = []
        all_validators = plugin_manager.get_all_validators()

        for rule_id, validator_class in all_validators.items():
            instance = self.create_validator_instance(validator_class)
            if instance:
                validators.append(instance)

        return validators

    def filter_enabled_validators(
        self, validator_classes: List[type[BaseValidator]]
    ) -> List[BaseValidator]:
        """Filter and instantiate only enabled validators from a list."""
        validators = []
        for validator_class in validator_classes:
            instance = self.create_validator_instance(validator_class)
            if instance:
                validators.append(instance)
        return validators

    def _get_or_create_embedding_model(self):
        """Get or create the shared EmbeddingGemma model for semantic similarity analysis."""
        model_key = "embedding_gemma"

        if model_key not in self._shared_models:
            try:
                from sentence_transformers import SentenceTransformer
                import logging
                import os

                logger = logging.getLogger(__name__)

                # Check configuration
                embedding_config = self.config.get("embedding_analysis", {})
                model_name = embedding_config.get("model", "google/embeddinggemma-300m")

                # Check if embedding analysis is enabled
                if not embedding_config.get("enabled", False):
                    logger.debug("Semantic similarity analysis disabled in configuration")
                    return None

                # Handle HF token from config, .env file, or environment
                hf_token = embedding_config.get("hf_token")
                if not hf_token:
                    # Try to load from .env file
                    env_file = getattr(self.config, 'project_root', None)
                    if env_file:
                        env_file = env_file / ".env"
                        if env_file and env_file.exists():
                            for line in env_file.read_text().splitlines():
                                if line.startswith("HF_TOKEN="):
                                    hf_token = line.split("=", 1)[1].strip().strip('"\'')
                                    break
                    # Fallback to environment variable
                    if not hf_token:
                        hf_token = os.getenv("HF_TOKEN")

                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token

                logger.info(f"Loading shared embedding model: {model_name}")
                model = SentenceTransformer(model_name)
                self._shared_models[model_key] = model
                logger.info("Shared embedding model loaded successfully")

            except ImportError:
                logger.debug("Semantic similarity analysis disabled: sentence-transformers not available")
                self._shared_models[model_key] = None
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._shared_models[model_key] = None

        return self._shared_models[model_key]

    def get_rule_summary(self) -> Dict[str, any]:
        """Get summary of rule configuration."""
        all_validators = plugin_manager.get_all_validators()
        enabled_count = sum(1 for rule_id in all_validators.keys() if self.is_rule_enabled(rule_id))

        return {
            "total_rules": len(all_validators),
            "enabled_rules": enabled_count,
            "disabled_rules": len(all_validators) - enabled_count,
            "overrides": len(self._rule_overrides),
            "plugins": list(self._enabled_plugins),
        }


def create_default_rule_config() -> Dict[str, any]:
    """Create default rule configuration for new projects."""
    return {
        "rules": {
            # Semantic rule IDs (primary system)
            "DOCSTRING-MISSING": "INFO",  # Missing docstring is just info
            "EXPORTS-MISSING-ALL": "WARN",  # Missing __all__ is warning
            "PRINT-STATEMENT": "WARN",  # Print statements are warnings
            "EMOJI-IN-STRING": "WARN",  # Emojis can cause encoding issues
            "TODO-FOUND": "INFO",  # TODOs are informational
            "PARAMETERS-KEYWORD-ONLY": "INFO",  # Parameter suggestions are info
        },
        "plugins": {"enabled": ["vibelint.core"]},
    }
