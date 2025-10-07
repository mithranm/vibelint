# Live Integration Tests

Tests in this directory make **real API calls** to LLM endpoints. Unlike unit tests which use mocks, these tests verify that vibelint works correctly with actual LLM backends.

## What's Tested

- **Structured Output**: Grammar generation and enforcement for llama.cpp
- **Response Format**: JSON schema validation for vLLM/OpenAI
- **LLM Client**: End-to-end request/response flow
- **Backend Detection**: Correct handling of different LLM backends

## Running Tests

### Run all live tests:
```bash
pytest tests/live/ -v
```

### Run specific test file:
```bash
pytest tests/live/test_structured_output.py -v
```

### Run specific test:
```bash
pytest tests/live/test_structured_output.py::TestStructuredOutputOrchestrator::test_focus_check_schema -v
```

### Skip live tests when running full suite:
```bash
pytest tests/ --ignore=tests/live/
```

## Requirements

Live tests require:
1. Valid LLM configuration in `pyproject.toml` under `[tool.vibelint.llm]`
2. Network access to LLM endpoints
3. Sufficient API quotas/rate limits

Tests will automatically skip if LLM is not configured.

## Why Separate From Unit Tests?

- **Speed**: Live tests take seconds; unit tests take milliseconds
- **Reliability**: Live tests can fail due to network issues or API changes
- **Cost**: Some LLM APIs charge per request
- **CI/CD**: Unit tests run on every commit; live tests run periodically
- **Rate Limits**: Batch running all tests could hit rate limits

## Adding New Live Tests

When adding tests:
1. Use `@pytest.fixture` for LLM client with skip on missing config
2. Test real behavior, not implementation details
3. Keep prompts minimal to reduce costs
4. Use `temperature=0.0` for deterministic results when possible
5. Add clear assertions with helpful error messages

Example:
```python
def test_my_feature(llm_client):
    """Test description."""
    request = LLMRequest(
        content="Simple prompt",
        max_tokens=50,
        temperature=0.0
    )
    response = llm_client.process_request_sync(request)
    assert response.success
    # ... more assertions
```
