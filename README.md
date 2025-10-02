# vibelint

Python code quality linter with plugin-based validators and LLM-powered analysis workflows.

## Features

- **Plugin-based validators** - Extensible architecture for custom code quality rules
- **AI-powered analysis** - LLM integration for semantic validation and architectural analysis
- **Justification workflow** - Analyzes project structure and provides architectural recommendations
- **Multiple output formats** - JSON, Markdown, SARIF, natural language
- **Configuration management** - TOML-based config with fail-fast validation

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run validators on current directory
vibelint check .

# Run with automatic fixing
vibelint check --fix .

# Run justification workflow for architectural analysis
vibelint justification .

# Show available validators
vibelint check --help
```

## Configuration

Add to `pyproject.toml`:

```toml
[tool.vibelint]
# Validator severity overrides
rules = { EMOJI-USAGE = "BLOCK", DICT-GET-FALLBACK = "WARN" }

# LLM configuration (optional)
[tool.vibelint.llm]
fast_api_url = "http://localhost:1234/v1"
orchestrator_api_url = "http://localhost:1235/v1"

# Embedding configuration (optional)
[tool.vibelint.embedding_analysis]
enabled = true
model = "google/embeddinggemma-300m"
```

## Development

See [AGENTS.instructions.md](./AGENTS.instructions.md) for development guidelines.

### Testing

```bash
# Run tests
tox -e py311

# Run formatters and linters
isort src/ tests/
black src/ tests/
ruff check --fix src/ tests/
pyright src/
```

## Documentation

- [Development Instructions](./AGENTS.instructions.md)
- [Claude Code Config](./.claude/settings.json)

## License

See LICENSE file.
