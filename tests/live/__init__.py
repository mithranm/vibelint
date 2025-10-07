"""Live integration tests that hit real LLM APIs.

These tests are separated from unit tests because they:
- Make real network requests to LLM endpoints
- Take longer to run (seconds vs milliseconds)
- Require valid API configuration in pyproject.toml
- May incur costs or rate limits

Run with: pytest tests/live/ -v
Skip with: pytest tests/ --ignore=tests/live/
"""
