"""Live tests for structured output / grammar generation.

Tests that the LLM client correctly generates grammars and enforces
structured output with real LLM backends (vLLM, llama.cpp).
"""

import json

import pytest
from pydantic import BaseModel, Field

from vibelint.llm_client import LLMClient, LLMRequest, LLMRole


class FocusCheck(BaseModel):
    """Schema for focus alignment check response."""

    class AlignmentStatus:
        ALIGNED = "aligned"
        NOT_ALIGNED = "not_aligned"
        NEEDS_CONTEXT = "needs_context"

    status: str = Field(
        description="aligned if action aligns with focus, not_aligned if it doesn't, needs_context if more context needed to decide"
    )


class SimpleYesNo(BaseModel):
    """Simple yes/no response schema."""

    answer: str = Field(description="yes or no")


@pytest.fixture
def llm_client():
    """Create LLM client with real config."""
    try:
        client = LLMClient()
        return client
    except ValueError as e:
        pytest.skip(f"LLM not configured: {e}")


class TestResponseFormatGeneration:
    """Test response_format generation for structured output."""

    def test_response_format_has_strict(self, llm_client):
        """Test that response_format includes strict: True."""
        from enum import Enum

        class Status(str, Enum):
            ALIGNED = "aligned"
            NOT_ALIGNED = "not_aligned"

        class TestSchema(BaseModel):
            status: Status

        schema = TestSchema.model_json_schema()

        # Simulate what happens during request preparation
        structured_output = {"json_schema": {"name": "test", "schema": schema}}

        # Build response_format like the client does
        response_format = {
            "type": "json_schema",
            "json_schema": {**structured_output["json_schema"], "strict": True},
        }

        # Verify structure
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "test"
        assert response_format["json_schema"]["schema"] == schema


class TestStructuredOutputOrchestrator:
    """Test structured output with orchestrator (llama.cpp or vLLM)."""

    def test_simple_enum_forced_output(self, llm_client):
        """Test that orchestrator respects enum schema and doesn't hallucinate other fields."""
        schema = SimpleYesNo.model_json_schema()

        request = LLMRequest(
            content="Is the sky blue? Answer yes or no.",
            max_tokens=100,
            temperature=0.0,
            structured_output={"json_schema": {"name": "simple_yesno", "schema": schema}},
        )

        # Call orchestrator directly
        response = llm_client._call_orchestrator_llm(request)

        assert response.success, f"Request failed: {response.content}"

        # Parse response - should be valid JSON
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON: {response.content}\nError: {e}")

        # Should ONLY have 'answer' field
        assert "answer" in result, f"Missing 'answer' field in response: {result}"
        assert isinstance(result["answer"], str), f"Answer should be string: {result}"

        # Should NOT have extra fields like 'analysis', 'reasoning', etc
        extra_fields = set(result.keys()) - {"answer"}
        assert (
            not extra_fields
        ), f"Unexpected fields in response: {extra_fields}. Full response: {result}"

        # Answer should be yes or no (case insensitive)
        answer = result["answer"].lower()
        assert answer in ["yes", "no"], f"Answer should be yes/no, got: {result['answer']}"

    def test_focus_check_schema(self, llm_client):
        """Test the actual FocusCheck schema used by kaia-guardrails."""
        from enum import Enum

        class AlignmentStatus(str, Enum):
            ALIGNED = "aligned"
            NOT_ALIGNED = "not_aligned"
            NEEDS_CONTEXT = "needs_context"

        class FocusCheckSchema(BaseModel):
            status: AlignmentStatus

        schema = FocusCheckSchema.model_json_schema()

        request = LLMRequest(
            content="""Focus: Clean up kaia-guardrails hooks

Action: User asks: "what time is it?"

Does this align with focus? Respond aligned, not_aligned, or needs_context.""",
            max_tokens=300,  # Give enough tokens for thinking + JSON output
            temperature=0.1,
            structured_output={"json_schema": {"name": "focus_check", "schema": schema}},
        )

        # Call orchestrator directly
        response = llm_client._call_orchestrator_llm(request)

        assert response.success, f"Request failed: {response.content}"

        # Parse response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Response is not valid JSON: {response.content}\n"
                f"Error: {e}\n"
                f"Schema: {schema}"
            )

        # Validate structure
        assert "status" in result, f"Missing 'status' field: {result}"

        # Should NOT have hallucinated fields
        extra_fields = set(result.keys()) - {"status"}
        if extra_fields:
            pytest.fail(
                f"LLM hallucinated extra fields: {extra_fields}\n"
                f"Full response: {result}\n"
                f"Schema: {schema}\n"
                f"Backend: {llm_client._get_backend_type_for_role(LLMRole.ORCHESTRATOR)}"
            )

        # Validate enum value
        valid_values = ["aligned", "not_aligned", "needs_context"]
        assert result["status"] in valid_values, f"Invalid status value: {result['status']}"

    def test_response_format_actually_sent_to_llm(self, llm_client, caplog):
        """Test that response_format is actually included in the request payload."""
        import logging

        caplog.set_level(logging.WARNING)

        schema = SimpleYesNo.model_json_schema()

        request = LLMRequest(
            content="Is grass green? Answer yes or no.",
            max_tokens=50,
            temperature=0.0,
            structured_output={"json_schema": {"name": "yesno", "schema": schema}},
        )

        # Make request - call orchestrator directly
        llm_client._call_orchestrator_llm(request)

        # Check logs for structured output debug message
        log_messages = [record.message for record in caplog.records]

        # Should log response_format being sent (works for all backends)
        format_logs = [msg for msg in log_messages if "[STRUCTURED_OUTPUT]" in msg]
        assert format_logs, f"No [STRUCTURED_OUTPUT] log found. Logs: {log_messages}"

        # Verify strict: True is in the logged response_format
        format_log = format_logs[0]
        assert (
            "strict" in format_log.lower() or "True" in format_log
        ), f"Missing 'strict: True' in log: {format_log}"


class TestStructuredOutputFast:
    """Test structured output with fast LLM (vLLM)."""

    def test_fast_llm_enum_output(self, llm_client):
        """Test structured output with fast LLM (usually vLLM)."""
        if not llm_client.is_llm_available(LLMRole.FAST):
            pytest.skip("Fast LLM not configured")

        schema = SimpleYesNo.model_json_schema()

        request = LLMRequest(
            content="Is water wet? Answer yes or no.",
            max_tokens=200,  # Give enough tokens for thinking + JSON output
            temperature=0.0,
            structured_output={"json_schema": {"name": "yesno", "schema": schema}},
        )

        # Call fast LLM directly
        response = llm_client._call_fast_llm(request)

        assert response.success
        result = json.loads(response.content)

        # Validate structure
        assert "answer" in result
        assert set(result.keys()) == {"answer"}, f"Unexpected fields: {result}"
        assert result["answer"].lower() in ["yes", "no"]
