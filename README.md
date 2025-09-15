# vibelint

[![CI](https://github.com/mithranm/vibelint/actions/workflows/ci.yml/badge.svg)](https://github.com/mithranm/vibelint/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/vibelint.svg)](https://badge.fury.io/py/vibelint)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive Python code quality tool with AI-powered analysis for better maintainability and LLM interaction.**

`vibelint` is a modern code quality tool that combines traditional linting with AI-powered analysis to identify code smells, architectural issues, and patterns that hinder both human understanding and Large Language Model (LLM) effectiveness. It helps you build codebases with good "vibes" - clean, maintainable, and AI-friendly code.

## Table of Contents

- [Why Use vibelint?](#why-use-vibelint)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [AI-Powered Analysis](#ai-powered-analysis)
- [Thinking Token Configuration](#thinking-token-management)
- [Output Formats](#output-formats)
- [Plugin System](#plugin-system)
- [Configuration](#configuration)
- [Error Categories](#error-categories)
- [Contributing](#contributing)
- [License](#license)

## Why Use vibelint?

Modern Python development involves both human developers and AI assistants. Code that's hard for humans to understand is also problematic for LLMs. vibelint addresses:

### 🧠 **Human & AI Understanding**
- **Missing Documentation**: Functions and modules without proper docstrings
- **Unclear Context**: Missing file path references that help LLMs locate code
- **Poor Type Annotations**: Functions without type hints reduce clarity

### 🏗️ **Architectural Issues**
- **Dead Code**: Unused functions and imports cluttering the codebase
- **Silent Failures**: Exception handling that masks errors
- **Namespace Collisions**: Conflicting names that create ambiguity

### 🤖 **AI Integration**
- **Emoji Usage**: Characters that break text encoding in AI tools
- **Print Statements**: Debug code left in production
- **Semantic Similarity**: Duplicate functionality across modules

## Key Features

### 🔍 **Traditional Linting**
- **Documentation Quality**: Ensures proper docstrings with path references
- **Code Hygiene**: Detects dead code, print statements, and encoding issues
- **Type Safety**: Identifies missing type annotations
- **Export Management**: Validates `__all__` declarations

### 🧠 **AI-Powered Analysis**
- **Semantic Similarity**: Uses embeddings to find functionally similar code
- **LLM Architecture Review**: Detects over-engineering and unnecessary abstractions
- **Fallback Pattern Analysis**: Identifies problematic exception handling

### 📊 **Project Intelligence**
- **Namespace Visualization**: Interactive project structure analysis
- **Code Snapshots**: Generate LLM-ready codebase summaries
- **Collision Detection**: Find naming conflicts and ambiguities
- **Comprehensive Reporting**: Detailed Markdown reports with actionable insights

## Installation

```bash
pip install vibelint
```

vibelint requires Python 3.10 or higher.

**Optional AI Features:**
```bash
# For semantic similarity analysis
pip install sentence-transformers

# For advanced embedding models
pip install torch transformers
```

## Quick Start

```bash
# Check your entire codebase
vibelint check

# Generate a detailed report
vibelint check -o report.md

# Create an LLM-ready snapshot
vibelint snapshot

# Visualize project structure
vibelint namespace
```

## Core Commands

### 🔍 `vibelint check`
Analyze your codebase for quality issues:

```bash
# Basic check
vibelint check

# Check with AI analysis
vibelint check --categories core,static,ai

# Output in different formats
vibelint check --format json
vibelint check --format sarif  # GitHub integration
```

#### 🎯 **Path Override for Faster Analysis**

When working with large codebases, you can analyze specific directories or files by providing paths as arguments. This **overrides** the `include_globs` configuration and analyzes only the specified paths:

```bash
# Analyze only the src directory (fast)
vibelint check src/

# Analyze a specific file
vibelint check src/mymodule.py

# Analyze multiple paths
vibelint check src/ tests/ docs/

# Skip AI analysis for even faster results
vibelint check src/mymodule.py --exclude-ai

# Combine with specific rules for targeted analysis
vibelint check src/ --rule EMOJI-IN-STRING --rule TYPING-POOR-PRACTICE
```

**Why use path override?**
- ⚡ **Faster analysis**: Analyze 10 files instead of 100+
- 🎯 **Focused feedback**: Get issues for specific areas you're working on
- 🔄 **Iterative workflow**: Fix issues incrementally while developing
- 📊 **AI analysis**: Make LLM analysis practical for large projects

**Example workflow:**
```bash
# Fast check while developing audio features
vibelint check src/myproject/audio/ --exclude-ai

# Deep analysis on specific module with AI
vibelint check src/myproject/core/engine.py

# Quick emoji/typing check across project
vibelint check --rule EMOJI-IN-STRING --rule TYPING-POOR-PRACTICE
```

#### ⏱️ **Managing Timeouts & Long-Running Analysis**

vibelint provides **time estimates** and **completion indicators** to help you work within timeout constraints common to AI coding tools and CI systems (typically 5-10 minutes):

**Time Estimation:**
```bash
# vibelint automatically estimates analysis time
vibelint check src/ --rule ARCHITECTURE-LLM

# Output:
# Starting LLM architectural analysis on 25 files
# ESTIMATED TIME: 4-7 minutes (depends on LLM response speed)
# TIMEOUT RISK: Analysis may take 4-7 minutes
# If using AI coding tools or CI, consider analyzing smaller chunks
```

**Timeout Management Strategies:**
```bash
# Strategy 1: Analyze in chunks
vibelint check src/module1/ --rule ARCHITECTURE-LLM  # ~2 minutes
vibelint check src/module2/ --rule ARCHITECTURE-LLM  # ~2 minutes

# Strategy 2: Skip AI analysis for speed
vibelint check src/ --exclude-ai  # Fast: <30 seconds

# Strategy 3: Focus on specific rules
vibelint check --rule EMOJI-IN-STRING  # Very fast: <10 seconds
```

**Completion Indicators:**
When analysis finishes, vibelint shows:
```
LLM architectural analysis COMPLETED on 25 files
Status: Analysis finished successfully (not interrupted)
```

If you see timeout/interruption, the analysis was cut short and you should use smaller chunks.

### 📸 `vibelint snapshot`
Create comprehensive codebase documentation:

```bash
# Generate snapshot for LLM context
vibelint snapshot

# Custom output file
vibelint snapshot -o context.md

# Exclude test files
vibelint snapshot --exclude "tests/**"
```

### 🌲 `vibelint namespace`
Visualize project structure:

```bash
# Display namespace tree
vibelint namespace

# Save to file
vibelint namespace -o structure.txt
```

## AI-Powered Analysis

vibelint includes cutting-edge AI analysis capabilities:

### 🔗 **Semantic Similarity Detection**
Uses sentence transformers to find functionally duplicate code:

```toml
[tool.vibelint.embedding_analysis]
enabled = true
model = "google/embeddinggemma-300m"
similarity_threshold = 0.85
```

### 🤖 **LLM Architecture Review**
Connects to local LLM endpoints for architectural analysis:

```toml
[tool.vibelint.llm_analysis]
api_base_url = "http://localhost:11434"
model = "codellama:13b"
temperature = 0.3
```

### 🧠 **Thinking Token Management**
vibelint automatically removes "thinking" tokens from LLM responses to provide clean analysis output:

```toml
[tool.vibelint.llm_analysis]
# Configure thinking token removal
remove_thinking_tokens = true    # Set to false to keep all output
thinking_format = "harmony"      # Options: "harmony", "qwen", "custom"

# For models with custom thinking patterns:
# thinking_format = "custom"
# custom_thinking_patterns = [
#     "r'<think>.*?</think>'",
#     "r'<reasoning>.*?</reasoning>'"
# ]
```

**Common Model Configurations:**
- **Claude/Anthropic models**: Use default `thinking_format = "harmony"`
- **Qwen models**: Set `thinking_format = "qwen"`
- **OpenAI o1 models**: Use `thinking_format = "custom"` with appropriate patterns
- **Other models**: Use `thinking_format = "custom"` and define your patterns

**Get configuration help:**
```bash
vibelint thinking-tokens --show-formats  # Show all supported formats
vibelint thinking-tokens --detect file   # Detect tokens in a file
```

### ⚡ **Performance Analysis**
Detects common performance anti-patterns and suggests optimizations.

## Output Formats

### 📝 **Natural Language** (Default)
Human-readable output with colors and suggestions:

```
WARN:
  DEAD-CODE-FOUND: Function 'unused_helper' is defined but never referenced (src/utils.py:42)
    → Consider removing unused definition or adding to __all__

INFO:
  DOCSTRING-PATH-REFERENCE: Missing path reference in docstring (src/main.py:10)
    → Add 'src/main.py' at the end of the docstring for LLM context
```

### 🔧 **JSON**
Machine-readable for CI/CD integration:

```bash
vibelint check --format json > results.json
```

### 🔒 **SARIF**
GitHub Security scanning format:

```bash
vibelint check --format sarif > results.sarif
```

### 🤖 **LLM**
Optimized format for AI analysis:

```bash
vibelint check --format llm > issues.txt
```

## Plugin System

vibelint supports custom validators and formatters:

### Creating Custom Validators

```python
from vibelint.plugin_system import BaseValidator, Severity, Finding
from pathlib import Path
from typing import Iterator

class NoHardcodedSecretsValidator(BaseValidator):
    rule_id = "SECURITY-001"
    name = "No Hardcoded Secrets"
    description = "Detects potential hardcoded secrets"
    default_severity = Severity.BLOCK

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        for line_num, line in enumerate(content.splitlines(), 1):
            if "password" in line.lower() and "=" in line:
                yield self.create_finding(
                    message="Potential hardcoded password detected",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Use environment variables or secret management"
                )
```

### Register in pyproject.toml

```toml
[project.entry-points."vibelint.validators"]
SECURITY-001 = "mypackage.validators:NoHardcodedSecretsValidator"
```

## Configuration

Configure vibelint in your `pyproject.toml`:

```toml
[tool.vibelint]
# File patterns to analyze
include_globs = [
    "src/**/*.py",
    "tests/**/*.py",
    "*.py"
]

# Patterns to exclude
exclude_globs = [
    ".venv/**",
    "**/migrations/**",
    "**/__pycache__/**"
]

# Rule configuration
[tool.vibelint.rules]
"DEAD-CODE-FOUND" = "WARN"
"EMOJI-IN-STRING" = "BLOCK"
"DOCSTRING-MISSING" = "INFO"

# AI Analysis Configuration
[tool.vibelint.embedding_analysis]
enabled = true
model = "google/embeddinggemma-300m"
similarity_threshold = 0.85

[tool.vibelint.llm_analysis]
api_base_url = "http://localhost:11434"
model = "codellama:13b"
max_tokens = 2048
temperature = 0.3

# Rule categories for targeted analysis
[tool.vibelint.rule_categories]
core = [
    "DOCSTRING-MISSING",
    "EXPORTS-MISSING-ALL",
    "PRINT-STATEMENT",
    "EMOJI-IN-STRING"
]
static = [
    "DEAD-CODE-FOUND",
    "ARCHITECTURE-INCONSISTENT",
    "TYPING-POOR-PRACTICE",
    "FALLBACK-SILENT-FAILURE"
]
ai = [
    "ARCHITECTURE-LLM",
    "SEMANTIC-SIMILARITY"
]
```

## Error Categories

vibelint organizes issues into logical categories:

### 🔧 **Core Issues** (Always Run)
- `DOCSTRING-MISSING`: Missing function/module documentation
- `DOCSTRING-PATH-REFERENCE`: Missing file paths in docstrings
- `EXPORTS-MISSING-ALL`: Missing `__all__` declarations
- `PRINT-STATEMENT`: Debug print statements
- `EMOJI-IN-STRING`: Encoding-problematic characters

### 🏗️ **Static Analysis**
- `DEAD-CODE-FOUND`: Unused functions and imports
- `ARCHITECTURE-INCONSISTENT`: Architectural violations
- `TYPING-POOR-PRACTICE`: Missing type annotations
- `FALLBACK-SILENT-FAILURE`: Problematic exception handling

### 🤖 **AI-Powered**
- `ARCHITECTURE-LLM`: LLM-detected architectural issues
- `SEMANTIC-SIMILARITY`: Functionally duplicate code

## Advanced Usage

### CI/CD Integration

```yaml
# .github/workflows/quality.yml
- name: Run vibelint
  run: |
    vibelint check --format sarif > vibelint.sarif
    vibelint check --format json > vibelint.json

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: vibelint.sarif
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: vibelint
        name: vibelint
        entry: vibelint check
        language: system
        types: [python]
```

### Quality Gates

```bash
# Fail build if too many issues
python -c "
import json
data = json.load(open('vibelint.json'))
errors = data['summary'].get('BLOCK', 0)
warnings = data['summary'].get('WARN', 0)
if errors > 0 or warnings > 10:
    exit(1)
"
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/mithranm/vibelint.git
cd vibelint
pip install -e ".[dev]"
pytest
```

## License

vibelint is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for better Python codebases and AI collaboration**