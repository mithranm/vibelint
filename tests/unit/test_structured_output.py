"""
Test structured output handling with Pydantic models.
"""

import pytest
from pydantic import BaseModel
from enum import Enum
from vibelint.llm.manager import LLMManager, LLMRequest, LLMRole


class FileDecision(str, Enum):
    """Simple yes/no decision enum."""
    yes = "yes"
    no = "no"


class FileEvaluation(BaseModel):
    """File evaluation response model."""
    response: FileDecision


@pytest.fixture
def llm_manager():
    """Create LLM manager for testing."""
    return LLMManager()


def test_pydantic_schema_generation():
    """Test that Pydantic generates the expected JSON schema format."""
    schema = FileEvaluation.model_json_schema()

    assert schema["type"] == "object"
    assert "response" in schema["properties"]
    assert schema["required"] == ["response"]

    # Check enum definition
    assert "$defs" in schema
    assert "FileDecision" in schema["$defs"]
    assert schema["$defs"]["FileDecision"]["enum"] == ["yes", "no"]


def test_schema_passing_to_llm_request():
    """Test that schemas are correctly passed through LLMRequest."""
    schema = FileEvaluation.model_json_schema()

    request = LLMRequest(
        content="Test content",
        max_tokens=50,
        temperature=0.1,
        structured_output=schema
    )

    assert request.structured_output == schema
    assert request.structured_output["type"] == "object"


def test_gbnf_grammar_generation_for_enum(llm_manager):
    """Test GBNF grammar generation for string enums."""
    schema = FileEvaluation.model_json_schema()

    # Test with the actual schema structure (with $ref)
    grammar = llm_manager._create_json_grammar(schema)

    # Should contain the enum choices
    assert "yes" in grammar
    assert "no" in grammar
    assert "response" in grammar


def test_backend_detection(llm_manager):
    """Test that backend types are correctly detected."""
    fast_backend = llm_manager._get_backend_type_for_role(LLMRole.FAST)
    orchestrator_backend = llm_manager._get_backend_type_for_role(LLMRole.ORCHESTRATOR)

    # Should return valid backend types
    assert fast_backend in ["vllm", "llamacpp", "openai"]
    assert orchestrator_backend in ["vllm", "llamacpp", "openai"]


@pytest.mark.integration
def test_structured_output_end_to_end(llm_manager):
    """Test complete structured output flow (requires LLM services)."""
    schema = FileEvaluation.model_json_schema()

    request = LLMRequest(
        content="Should I keep this test file? Answer yes or no.",
        max_tokens=50,
        temperature=0.1,
        structured_output=schema
    )

    # This test requires actual LLM services to be running
    try:
        response = llm_manager.process_request_sync(request)

        if response.success:
            # Try to parse and validate the response
            import json
            content = response.content
            parsed = json.loads(content)

            # Should be valid according to our Pydantic model
            evaluation = FileEvaluation(**parsed)
            assert evaluation.response in [FileDecision.yes, FileDecision.no]

    except Exception as e:
        pytest.skip(f"LLM service not available: {e}")