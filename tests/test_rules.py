"""Tests for rules engine."""

from pathlib import Path
from typing import Iterator

from vibelint.config import Config
from vibelint.rules import RuleEngine
from vibelint.validators import BaseValidator, Finding, Severity


class DummyValidator(BaseValidator):
    """Dummy validator for testing."""

    rule_id = "DUMMY-RULE"
    default_severity = Severity.WARN

    def validate(
        self, _file_path: Path, _content: str, _config: Config | None = None
    ) -> Iterator[Finding]:
        """Dummy validation."""
        return iter([])


def test_rule_engine_initialization(sample_config: Config):
    """Test RuleEngine initialization."""
    engine = RuleEngine(sample_config)

    assert engine.config == sample_config
    assert isinstance(engine._rule_overrides, dict)


def test_rule_engine_default_enabled(sample_config: Config):
    """Test that rules are enabled by default."""
    engine = RuleEngine(sample_config)

    assert engine.is_rule_enabled("SOME-RULE")
    assert engine.is_rule_enabled("ANOTHER-RULE")


def test_rule_engine_disable_rule(temp_dir: Path):
    """Test disabling a rule via config."""
    config = Config(
        project_root=temp_dir,
        config_dict={
            "rules": {"DISABLED-RULE": "OFF"},
        },
    )

    engine = RuleEngine(config)

    assert not engine.is_rule_enabled("DISABLED-RULE")
    assert engine.is_rule_enabled("ENABLED-RULE")


def test_rule_engine_severity_override(temp_dir: Path):
    """Test overriding rule severity."""
    config = Config(
        project_root=temp_dir,
        config_dict={
            "rules": {
                "WARN-RULE": "WARN",
                "BLOCK-RULE": "BLOCK",
                "INFO-RULE": "INFO",
            },
        },
    )

    engine = RuleEngine(config)

    assert engine.get_rule_severity("WARN-RULE") == Severity.WARN
    assert engine.get_rule_severity("BLOCK-RULE") == Severity.BLOCK
    assert engine.get_rule_severity("INFO-RULE") == Severity.INFO


def test_rule_engine_default_severity(sample_config: Config):
    """Test default severity fallback."""
    engine = RuleEngine(sample_config)

    # Rule not in config should use provided default
    assert engine.get_rule_severity("UNKNOWN-RULE", Severity.INFO) == Severity.INFO


def test_rule_engine_ignore_list(temp_dir: Path):
    """Test ignore list disables rules."""
    config = Config(
        project_root=temp_dir,
        config_dict={
            "ignore": ["IGNORED-RULE-1", "IGNORED-RULE-2"],
        },
    )

    engine = RuleEngine(config)

    assert not engine.is_rule_enabled("IGNORED-RULE-1")
    assert not engine.is_rule_enabled("IGNORED-RULE-2")
    assert engine.is_rule_enabled("OTHER-RULE")


def test_create_validator_instance(sample_config: Config):
    """Test creating validator instance with config."""
    engine = RuleEngine(sample_config)

    validator = engine.create_validator_instance(DummyValidator)

    assert validator is not None
    assert validator.rule_id == "DUMMY-RULE"
    assert validator.severity == Severity.WARN
    assert validator.config == sample_config


def test_create_disabled_validator_returns_none(temp_dir: Path):
    """Test that disabled validators return None."""
    config = Config(
        project_root=temp_dir,
        config_dict={
            "rules": {"DUMMY-RULE": "OFF"},
        },
    )

    engine = RuleEngine(config)
    validator = engine.create_validator_instance(DummyValidator)

    assert validator is None


def test_get_enabled_validators(sample_config: Config):
    """Test getting all enabled validators."""
    engine = RuleEngine(sample_config)

    validators = engine.get_enabled_validators()

    assert isinstance(validators, list)
    assert len(validators) > 0
    assert all(isinstance(v, BaseValidator) for v in validators)


def test_get_enabled_validators_respects_config(temp_dir: Path):
    """Test that get_enabled_validators respects config."""
    # Get all validators first
    from vibelint.validators import get_all_validators

    all_validators = get_all_validators()
    first_rule_id = list(all_validators.keys())[0]

    # Disable first validator
    config = Config(
        project_root=temp_dir,
        config_dict={
            "rules": {first_rule_id: "OFF"},
        },
    )

    engine = RuleEngine(config)
    validators = engine.get_enabled_validators()

    # Check that disabled validator is not in list
    validator_ids = [v.rule_id for v in validators]
    assert first_rule_id not in validator_ids
