"""Tests for LLM client classes and dataclasses."""

import pytest
from vibelint.llm_client import (
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMBackendConfig,
    FeatureAvailability,
    LLMStatus,
)


class TestLLMRequest:
    """Test LLMRequest dataclass."""

    def test_basic_creation(self):
        """Test basic LLMRequest creation with required fields."""
        request = LLMRequest(content="Test prompt")
        assert request.content == "Test prompt"
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.system_prompt is None
        assert request.structured_output is None

    def test_with_optional_fields(self):
        """Test LLMRequest with all optional fields."""
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        request = LLMRequest(
            content="Test prompt",
            max_tokens=100,
            temperature=0.7,
            system_prompt="You are a helpful assistant",
            structured_output={"json_schema": {"schema": schema}},
        )
        assert request.content == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.system_prompt == "You are a helpful assistant"
        assert request.structured_output == {"json_schema": {"schema": schema}}

    def test_keyword_arguments(self):
        """Test LLMRequest accepts keyword arguments."""
        request = LLMRequest(content="Test", max_tokens=50, temperature=0.1)
        assert request.content == "Test"
        assert request.max_tokens == 50
        assert request.temperature == 0.1


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_basic_creation(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            content="Test response",
            success=True,
            llm_used="fast",
            duration_seconds=0.5,
            input_tokens=10,
            reasoning_content="",
            error=None,
        )
        assert response.content == "Test response"
        assert response.success is True
        assert response.llm_used == "fast"
        assert response.duration_seconds == 0.5
        assert response.input_tokens == 10
        assert response.reasoning_content == ""
        assert response.error is None

    def test_with_reasoning(self):
        """Test LLMResponse with reasoning content."""
        response = LLMResponse(
            content="Answer",
            success=True,
            llm_used="orchestrator",
            duration_seconds=1.2,
            input_tokens=20,
            reasoning_content="Detailed reasoning here",
            error=None,
        )
        assert response.reasoning_content == "Detailed reasoning here"

    def test_with_error(self):
        """Test LLMResponse with error."""
        response = LLMResponse(
            content="",
            success=False,
            llm_used="fast",
            duration_seconds=0.1,
            input_tokens=5,
            reasoning_content="",
            error="Connection timeout",
        )
        assert response.success is False
        assert response.error == "Connection timeout"


class TestLLMRole:
    """Test LLMRole enum."""

    def test_enum_values(self):
        """Test LLMRole enum has expected values."""
        assert LLMRole.FAST.value == "fast"
        assert LLMRole.ORCHESTRATOR.value == "orchestrator"

    def test_enum_members(self):
        """Test LLMRole enum members."""
        roles = list(LLMRole)
        assert len(roles) == 2
        assert LLMRole.FAST in roles
        assert LLMRole.ORCHESTRATOR in roles


class TestLLMBackendConfig:
    """Test LLMBackendConfig dataclass."""

    def test_basic_creation(self):
        """Test LLMBackendConfig creation."""
        config = LLMBackendConfig(
            backend="openai",
            api_url="http://localhost:8000",
            model="test-model",
            api_key="test-key",
            temperature=0.2,
            max_tokens=2048,
            max_context_tokens=4096,
        )
        assert config.backend == "openai"
        assert config.api_url == "http://localhost:8000"
        assert config.model == "test-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.2
        assert config.max_tokens == 2048
        assert config.max_context_tokens == 4096


class TestFeatureAvailability:
    """Test FeatureAvailability dataclass."""

    def test_all_features_available(self):
        """Test FeatureAvailability when all features are available."""
        features = FeatureAvailability(
            architecture_analysis=True,
            docstring_generation=True,
            code_smell_detection=True,
            coverage_assessment=True,
            llm_validation=True,
            semantic_similarity=True,
            embedding_clustering=True,
            duplicate_detection=True,
        )
        assert features.architecture_analysis is True
        assert features.docstring_generation is True
        assert features.llm_validation is True

    def test_no_features_available(self):
        """Test FeatureAvailability when no features are available."""
        features = FeatureAvailability(
            architecture_analysis=False,
            docstring_generation=False,
            code_smell_detection=False,
            coverage_assessment=False,
            llm_validation=False,
            semantic_similarity=False,
            embedding_clustering=False,
            duplicate_detection=False,
        )
        assert features.architecture_analysis is False
        assert features.docstring_generation is False
        assert features.llm_validation is False


class TestLLMStatus:
    """Test LLMStatus dataclass."""

    def test_basic_creation(self):
        """Test LLMStatus creation."""
        status = LLMStatus(
            fast_configured=True,
            orchestrator_configured=True,
            context_threshold=3000,
            fallback_enabled=False,
            available_features=FeatureAvailability(
                architecture_analysis=True,
                docstring_generation=True,
                code_smell_detection=False,
                coverage_assessment=False,
                llm_validation=True,
                semantic_similarity=False,
                embedding_clustering=False,
                duplicate_detection=False,
            ),
        )
        assert status.fast_configured is True
        assert status.orchestrator_configured is True
        assert status.context_threshold == 3000
        assert status.fallback_enabled is False
        assert status.available_features.architecture_analysis is True
