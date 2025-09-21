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
- **Print Statements**: Debug code left in production (with suppression support)
- **Semantic Similarity**: Duplicate functionality across modules

## Key Features

### 🔍 **Traditional Linting**
- **Documentation Quality**: Ensures proper docstrings with path references
- **Code Hygiene**: Detects dead code, print statements, and encoding issues
- **Type Safety**: Identifies missing type annotations
- **Export Management**: Validates `__all__` declarations

### 🧠 **AI-Powered Analysis**
- **Multi-Phase Architecture Review**: 4-phase analysis pipeline with batched processing
- **Semantic Similarity**: Uses EmbeddingGemma to find functionally similar code
- **Dynamic Context Discovery**: Automatically probes LLM for real context window limits
- **Token Usage Optimization**: Real-time diagnostics for optimal LLM utilization
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

### 🔑 **API Key Setup**

Before using AI features, set up your API keys:

```bash
# Set up configuration (creates .env file)
vibelint setup

# Edit the created .env file with your API keys:
# FAST_LLM_API_KEY=sk-llm-proxy-your-fast-llm-key-here
# ORCHESTRATOR_LLM_API_KEY=sk-chip-proxy-your-orchestrator-llm-key-here
# HF_TOKEN=your_huggingface_token_here
```

**Configuration Options:**
- **Project-specific**: `vibelint setup` (creates `.env` in current directory)
- **Global**: `vibelint setup --global` (creates `~/.vibelint.env` for all projects)
- **Environment variables**: Set `FAST_LLM_API_KEY`, `ORCHESTRATOR_LLM_API_KEY`, `HF_TOKEN` directly

**Optional AI Features:**
```bash
# For semantic similarity analysis
pip install sentence-transformers

# For advanced embedding models
pip install torch transformers
```

## Quick Start

```bash
# 1. Install and set up API keys
pip install vibelint
vibelint setup
# Edit the .env file with your API keys

# 2. Check your entire codebase
vibelint check

# 3. Generate a detailed report
vibelint check -o report.md

# 4. Create an LLM-ready snapshot
vibelint snapshot

# 5. Visualize project structure
vibelint namespace

# 6. Enable context discovery when switching LLM providers
# Edit pyproject.toml: enable_context_probing = true
# Then run: vibelint check --rule ARCHITECTURE-LLM
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

vibelint features a sophisticated dual-LLM architecture analysis system with intelligent routing and context discovery:

### 🏗️ **Architecture Analysis Workflow**

vibelint uses a **dual LLM architecture** with intelligent routing between fast and orchestrator LLMs for optimal performance:

```
┌─────────────────────────────────────────────────────────────┐
│                    vibelint Architecture Analysis            │
├─────────────────────────────────────────────────────────────┤
│  1. File Discovery & Batching                               │
│     • Discovers Python files via include_globs             │
│     • Batches files for efficient processing               │
│     • Estimates analysis time and warns about timeouts     │
│                                                             │
│  2. Dual LLM System Initialization                         │
│     ┌──────────────────┐    ┌─────────────────────────────┐ │
│     │   FAST LLM       │    │   ORCHESTRATOR LLM          │ │
│     │   (vLLM)         │    │   (llama.cpp)               │ │
│     │                  │    │                             │ │
│     │ • Small context  │    │ • Large context (32k+)     │ │
│     │ • Quick response │    │ • Complex reasoning         │ │
│     │ • Docstrings     │    │ • Architecture analysis    │ │
│     │ • Simple tasks   │    │ • Multi-file synthesis     │ │
│     └──────────────────┘    └─────────────────────────────┘ │
│              │                           │                  │
│              └──────── Smart Router ─────┘                  │
│                        (Context-aware)                      │
│                                                             │
│  3. Context Discovery & Calibration                        │
│     • Linear ramp-up testing (100→500→1k→2k→4k tokens)    │
│     • Discovers actual hardware limits                     │
│     • Handles timeout gracefully (30s fast, 60s orchestrator)│
│     • Generates assessment report with recommendations     │
│                                                             │
│  4. Multi-Phase Analysis Pipeline                          │
│     Phase 1: File Summarization (Fast LLM)                │
│     Phase 2: Semantic Similarity (Embeddings)             │
│     Phase 3: Pairwise Analysis (Orchestrator LLM)         │
│     Phase 4: Global Synthesis (Orchestrator LLM)          │
│                                                             │
│  5. Intelligent Routing Logic                              │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Task Type        → LLM Selection                    │ │
│     │                                                     │ │
│     │ • Small content  → Fast LLM                        │ │
│     │ • >3000 tokens   → Orchestrator LLM               │ │
│     │ • Architecture   → Orchestrator LLM               │ │
│     │ • Planning       → Orchestrator LLM               │ │
│     │ • Multi-file     → Orchestrator LLM               │ │
│     │ • Docstrings     → Fast LLM                        │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  6. Diagnostic & Assessment System                         │
│     • Real-time context monitoring                         │
│     • Performance benchmarking                             │
│     • Configuration assessment vs actual performance       │
│     • Provides recommendations without modifying files    │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 **Dual LLM Configuration**
Clean, intuitive configuration supporting two complementary LLMs:

```toml
[tool.vibelint.llm]
# Fast LLM: High-speed inference for quick tasks (vLLM)
fast_api_url = "http://100.94.250.88:8001"
fast_model = "openai/gpt-oss-20b"
fast_temperature = 0.1
fast_max_tokens = 2048

# Orchestrator LLM: Large context for complex reasoning (llama.cpp)
orchestrator_api_url = "http://100.116.54.128:11434"
orchestrator_model = "llama3.2:latest"
orchestrator_temperature = 0.2
orchestrator_max_tokens = 8192
orchestrator_max_context_tokens = 3600  # Discovered via diagnostics

# Automatic routing (when to use which LLM)
context_threshold = 3000              # Use orchestrator for >3k tokens
enable_context_probing = true         # Auto-discover actual limits
enable_fallback = true                # Fallback between LLMs on failure
```

### 🔍 **Context Discovery & Calibration**

vibelint automatically discovers real hardware limits through systematic testing:

```bash
# Run diagnostics to discover optimal configuration
vibelint diagnostics

# Output:
🔍 Testing ORCHESTRATOR LLM context limits...
  Testing 100 tokens... ✅ 1.5s
  Testing 500 tokens... ✅ 2.3s
  Testing 1000 tokens... ✅ 2.1s
  Testing 2000 tokens... ✅ 2.9s
  Testing 4000 tokens... ✅ 4.4s
  📊 Result: 4000 tokens max, 2632ms avg, 100% success

=== LLM Routing Benchmark ===
Routing Accuracy: 100.0%
  ✓ docstring: fast (expected: fast)
  ✓ analysis: orchestrator (expected: orchestrator)
  ✓ architecture: orchestrator (expected: orchestrator)

✅ Assessment completed successfully!
📄 Check LLM_ASSESSMENT_RESULTS.md for detailed results
```

**Key Benefits:**
- **Hardware-Aware**: Discovers actual RTX 5090/128GB limits, not theoretical maximums
- **Engine-Agnostic**: Works with vLLM, llama.cpp, Ollama, OpenAI-compatible APIs
- **Assessment-Based**: Warns about misconfigurations without modifying files
- **Performance-Optimized**: Linear ramp-up minimizes token waste and timeouts

### 📊 **GPT-OSS Benchmark Results**

Based on comprehensive testing with real hardware configurations:

**GPT-OSS-20B (Fast LLM via vLLM):**
- **Hardware**: RTX 5060 Ti 16GB + 32GB DDR5-6000 + AM5 processor
- **Context Limit**: 1,000 tokens confirmed
- **Performance**:
  - 100 tokens: 0.7 seconds
  - 500 tokens: 0.5 seconds
  - 1,000 tokens: 0.5 seconds
- **Success Rate**: 75.0% (limited by vLLM context configuration)
- **Average Latency**: 569ms
- **Use Cases**: Ultra-fast docstring generation, quick code analysis, simple refactoring

**GPT-OSS-120B (Orchestrator LLM via llama.cpp):**
- **Hardware**: RTX 5090 + 128GB DDR5-3600 + AM5 processor
- **Context Limit**: 32,000 tokens confirmed (theoretical: 131k)
- **Context Processing Performance** (with minimal 20-token output):
  - 1,000 tokens: 3.2 seconds
  - 4,000 tokens: 5.6 seconds
  - 8,000 tokens: 7.0 seconds
  - 16,000 tokens: 11.9 seconds
  - 24,000 tokens: 12.3 seconds
  - 32,000 tokens: 13.8 seconds
- **Token Generation Performance**:
  - **Speed**: ~13.6 tokens/second
  - **Context Processing**: ~27.5 tokens/second
  - **Example**: 120 tokens generated in ~8.8 seconds
- **Success Rate**: 87.5% at maximum context
- **Use Cases**: Complex architecture analysis, large context summarization, multi-file analysis

**Recommended Configuration:**
```toml
[tool.vibelint.llm]
# Fast LLM for quick tasks
fast_api_url = "http://your-vllm-server:8001"
fast_model = "openai/gpt-oss-20b"
fast_max_tokens = 2048

# Orchestrator LLM for complex analysis
orchestrator_api_url = "http://your-llamacpp-server:11434"
orchestrator_model = "openai_gpt-oss-120b-MXFP4.gguf"
orchestrator_max_context_tokens = 28800  # 32k with 10% safety margin
context_threshold = 3000  # Route to orchestrator for >3k tokens
```

### 🔗 **Semantic Similarity Detection**
Uses sentence transformers to find functionally duplicate code:

```toml
[tool.vibelint.embedding_analysis]
enabled = true
model = "google/embeddinggemma-300m"
similarity_threshold = 0.85
```

### 🎯 **Multi-Phase Analysis Pipeline**
Sophisticated architectural analysis with intelligent LLM routing:

1. **Phase 1: File Summarization** (Fast LLM)
   - Generate concise summaries of each Python file
   - Extract key functions, classes, and patterns
   - Optimized for speed with small context LLM

2. **Phase 2: Semantic Clustering** (Local Embeddings)
   - Compute embeddings using EmbeddingGemma-300M
   - Identify semantically similar code groups
   - No LLM API calls required

3. **Phase 3: Pairwise Analysis** (Orchestrator LLM)
   - Deep analysis of similar file pairs
   - Detect architectural inconsistencies
   - Uses large context LLM for complex reasoning

4. **Phase 4: Global Synthesis** (Orchestrator LLM)
   - Synthesize findings across the entire codebase
   - Generate architectural recommendations
   - Produce comprehensive quality assessment

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

### 📊 **Token Usage Diagnostics & Context Discovery**
Automatically optimizes LLM utilization and discovers real context limits:

```toml
[tool.vibelint.llm_analysis]
# Token usage diagnostics (for optimal LLM utilization)
enable_token_diagnostics = true        # Enable detailed token usage analysis
max_context_tokens = 4000             # Model's maximum context window (discovered)
max_prompt_tokens = 3500              # Reserve tokens for generation

# Dynamic context discovery (discovers actual LLM limits)
enable_context_probing = false        # Set to true when switching providers to re-discover limits
```

**What it provides:**
- **Real-time Context Monitoring**: Shows actual token usage vs. available context
- **Dynamic Context Discovery**: Probes LLM to find real context window size (not just configured)
- **Efficiency Warnings**: Alerts when context usage is too low (<30%) or too high (>80%)
- **Cost Optimization**: Tracks total input/output tokens and provides utilization recommendations
- **Batch Size Optimization**: Suggests optimal batch sizes based on discovered limits

**Example output:**
```
🔍 Probing LLM for actual context window limits...
⚠️ Actual context limit lower than configured: 4,000 tokens vs 32,768
💡 RECOMMENDATION: Reduce max_context_tokens to avoid errors

Context utilization: 11.7% (469/4,000 tokens, discovered)
=== LLM Usage Diagnostics ===
Context window: 4,000 tokens (discovered)
Average context efficiency: 7.8%
⚠️ Context discovery found 88% smaller window than configured
```

**Why this matters:**
- **Prevents Failures**: Discovers real limits before hitting context overflow errors
- **Optimizes Costs**: Maximizes token utilization within actual constraints
- **Saves Time**: Avoids failed analysis runs due to incorrect assumptions
- **Provider Agnostic**: Works with any OpenAI-compatible API endpoint

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

# Dual LLM Configuration - Clean and Intuitive
[tool.vibelint.llm]
# Fast LLM: High-speed inference for quick tasks (vLLM)
fast_api_url = "http://100.94.250.88:8001"
fast_model = "openai/gpt-oss-20b"
fast_temperature = 0.1
fast_max_tokens = 2048

# Orchestrator LLM: Large context for complex reasoning (llama.cpp)
orchestrator_api_url = "http://100.116.54.128:11434"
orchestrator_model = "llama3.2:latest"
orchestrator_temperature = 0.2
orchestrator_max_tokens = 8192
orchestrator_max_context_tokens = 3600  # Discovered via diagnostics

# Automatic routing (when to use which LLM)
context_threshold = 3000              # Use orchestrator for >3k tokens
enable_context_probing = true         # Auto-discover actual limits
enable_fallback = true                # Fallback between LLMs on failure

# Thinking token removal configuration
remove_thinking_tokens = true        # Set to false to keep all model output
thinking_format = "harmony"          # Options: "harmony", "qwen", "custom"

# AI Analysis Configuration
[tool.vibelint.embedding_analysis]
enabled = true
model = "google/embeddinggemma-300m"
similarity_threshold = 0.85

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