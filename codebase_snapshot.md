# Snapshot

## Filesystem Tree

```
vibelint/
├── src/
│   └── vibelint/
│       ├── validators/
│       │   ├── project_wide/
│       │   │   ├── __init__.py
│       │   │   ├── api_consistency.py
│       │   │   ├── code_smells.py
│       │   │   ├── dead_code.py
│       │   │   ├── module_cohesion.py
│       │   │   ├── namespace_collisions.py
│       │   │   └── namespace_report.py
│       │   ├── single_file/
│       │   │   ├── __init__.py
│       │   │   ├── dict_get_fallback.py
│       │   │   ├── docstring.py
│       │   │   ├── emoji.py
│       │   │   ├── exports.py
│       │   │   ├── logger_names.py
│       │   │   ├── print_statements.py
│       │   │   ├── relative_imports.py
│       │   │   ├── self_validation.py
│       │   │   ├── strict_config.py
│       │   │   └── typing_quality.py
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   └── types.py
│       ├── workflows/
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   └── base.py
│       │   ├── implementations/
│       │   │   ├── __init__.py
│       │   │   └── justification.py
│       │   ├── __init__.py
│       │   ├── cleanup.py
│       │   ├── evaluation.py
│       │   └── registry.py
│       ├── VIBECHECKER.txt
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── ast_utils.py
│       ├── cli.py
│       ├── config.py
│       ├── discovery.py
│       ├── embedding_client.py
│       ├── filesystem.py
│       ├── fix.py
│       ├── llm_client.py
│       ├── reporting.py
│       ├── rules.py
│       ├── snapshot.py
│       ├── ui.py
│       └── validation_engine.py
├── AGENTS.instructions.md
├── CLAUDE.md
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
└── tox.ini
```

## File Contents

Files are ordered alphabetically by path.

### File: AGENTS.instructions.md

```markdown
# Vibelint Development Instructions

## File Organization & Project Structure Rules

**CRITICAL**: These rules are enforced by the justification workflow.

### Forbidden Directories
- **NO `scripts/` directory**: Utility scripts belong at project root or deleted if not essential
- **NO `utils/` directory in src/**: Utilities belong in properly named modules (e.g., `io/`, `fs/`)

### One-Off Scripts Policy
**CRITICAL**: No one-off scripts that outlive their single use.

- **Temporary scripts**: Must be deleted immediately after use
- **Recurring utility**: Must be integrated into the codebase as proper modules/commands
- **Root-level `.py` files**: Only allowed if they serve ongoing project needs (e.g., setup.py)
- **Exception**: Migration scripts may live temporarily but must be deleted post-migration

**Rationale**: Scripts that linger create maintenance debt and confusion about their purpose/usage.

### Root Directory Rules
**Root should ONLY contain**:
- Project metadata: `setup.py`, `pyproject.toml`, `MANIFEST.in`, `tox.ini`, `LICENSE`
- Single documentation entry: `README.md`
- Configuration: `.gitignore`, `.env.example`
- Package entry points: `__init__.py`, `__main__.py`, `conftest.py`
- AI/Agent instructions: `CLAUDE.md`, `AGENTS.instructions.md`, `*.instructions.md`
- Standard tool outputs: `coverage.xml` (pytest-cov), `.coverage` (coverage.py)
- Essential utility scripts (minimize these - prefer proper modules)

### Source Organization
- All Python source → `src/vibelint/`
- Tests → `tests/`
- Documentation → `docs/`
- Generated artifacts → `.vibes/`, `.vibelint-reports/` (gitignored)
- **Runtime resources** (ASCII art, templates, etc.) → `src/vibelint/` alongside code that uses them

**Rationale**: Clean root = clear intent. Every file's location should be immediately justifiable. Runtime resources must stay in the package for `importlib.resources` access.

## Development Methodology

**PRIMARY REFERENCE**: Follow `DEVELOPMENT_METHODOLOGY.md` for comprehensive requirements-driven development process.

**KEY PRINCIPLES**:
- Human-in-loop orchestration (not autopilot)
- Requirements-driven development with acceptance criteria
- Multi-representation analysis (filesystem, vector, graph, runtime)
- Four-model coordination (VanguardOne/VanguardTwo + Chip/Claudia)

## Quick Development Workflow

After making changes:
1. **IMMEDIATELY run vibelint on files you just modified**:
   ```bash
   PYTHONPATH=src vibelint check path/to/modified/files.py
   ```

2. **For comprehensive analysis**:
   ```bash
   PYTHONPATH=src vibelint check src/ --format json > .vibelint-reports/$(date +%Y-%m-%d-%H%M%S)-analysis.json
   ```

3. **Run standard linting**: black, isort, ruff, pyright

4. **Run tests and assess quality** (see testing procedures below)

### Justification Workflow Execution

**IMPORTANT**: Justification analysis is resource-intensive (8-10 minutes for 120 files).

- **With cache**: ~8-10 minutes (most files cached, 2-3 LLM calls)
- **No cache**: ~30+ minutes (all files need summarization)

**Best practices**:
```bash
# Background execution (recommended)
nohup python -c "..." > /tmp/justification.log 2>&1 &

# Monitor progress
tail -f .vibes/logs/justification_*.log

# High timeout for synchronous runs
timeout 600  # 10 minutes minimum with cache
```

**CRITICAL - Human Review Required**:
- Justification workflow output is **LLM-generated analysis** - not ground truth
- **Always critically review** recommendations before implementing
- LLMs can misunderstand project context, runtime requirements, or architectural decisions
- Verify each suggestion aligns with actual project needs
- Example: VIBECHECKER.txt was flagged for moving to docs/, but it's a runtime resource needed by CLI

## LLM Analysis Behavior

**IMPORTANT**: Vibelint is modular with fail-fast behavior:
- **Unconfigured LLMs**: AI features gracefully degrade (feature unavailable)
- **Configured but unavailable LLMs**: Analysis immediately aborts with clear error
- No fallback between models - prevents unexpected behavior and masked issues
- Single LLM users can still access available AI features

Check what's available:
```bash
PYTHONPATH=src vibelint diagnostics  # Shows configured models and available features
```

## Testing Procedures

### Linting and Formatting Pipeline
1. black
2. isort
3. ruff check --fix
4. pyright

### Standard Testing Pipeline
1. **Run existing tests**: `tox -e py311` (or appropriate Python version)
2. **Check coverage**: Tests should maintain high pass rate and reasonable coverage
```

---
### File: CLAUDE.md

```markdown
# vibelint

This project uses kaia-guardrails hooks for development workflow automation.

## Shell Configuration

Claude Code has limitations with shell configuration loading. To work around this:

1. **For commands needing shell environment**:
   ```bash
   source .claude/shell-helper.sh && your_command
   ```

2. **For one-off commands with shell config**:
   ```bash
   zsh -c 'source ~/.zshrc && your_command'
   ```

3. **Always use absolute paths** when possible to avoid PATH issues

4. **Quote paths with spaces** to prevent command parsing errors

5. **Set environment variables** in `.claude/settings.json` rather than shell config

See `.claude/shell-config.md` for detailed troubleshooting.

```

---
### File: LICENSE

```
MIT License

Copyright (c) 2025 vibelint Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---
### File: MANIFEST.in

```
# Include these specific top-level files in the sdist
include LICENSE
include README.md
include pyproject.toml
include tox.ini

# Yeah
include src/vibelint/VIBECHECKER.txt

# Recursively include all files (*) found within the 'examples' directory
recursive-include examples *
# Recursively include all Python files (*.py) found within the 'tests' directory
recursive-include tests *.py

# Recursively exclude any directory named '__pycache__' anywhere in the project
recursive-exclude * __pycache__
# Recursively exclude compiled Python files anywhere
recursive-exclude * *.py[cod]
# Recursively exclude compiled C extensions anywhere
recursive-exclude * *.so
# Recursively exclude VIM swap files anywhere
recursive-exclude * .*.swp
# Recursively exclude macOS metadata files anywhere
recursive-exclude * .DS_Store
```

---
### File: pyproject.toml

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "vibelint"
version = "0.1.2"
description = "Suite of tools to enhance the vibe coding process."
authors = [
  { name = "Mithran Mohanraj", email = "mithran.mohanraj@gmail.com" }
]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
]
dependencies = [
    "click>=8.1.0",
    "tomli>=2.0.0; python_version < '3.11'",
    "tomli-w",
    "colorama>=0.4.0",
    "rich>=12.0.0",
    "libcst",
    "requests>=2.25.0",
    "python-dotenv>=1.0.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.3.0",
    "langchain-core>=0.3.0"
]

[project.optional-dependencies]
embedding = [
    "sentence-transformers>=2.2.0",
    "torch>=1.9.0",
    "numpy>=1.21.0"
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "black>=23.0.0"
]

[project.scripts]
vibelint = "vibelint.cli:main"

[project.entry-points."vibelint.validators"]
# Built-in validators are auto-discovered from the filesystem.
# Only third-party validators need to be registered here as entry points.
#
# Third-party packages can add validators here:
# Example:
# "MY-CUSTOM-RULE" = "my_package.validators:MyCustomValidator"


[project.entry-points."vibelint.formatters"]
human = "vibelint.reporting:HumanFormatter"
natural = "vibelint.reporting:NaturalLanguageFormatter"
json = "vibelint.reporting:JsonFormatter"
sarif = "vibelint.reporting:SarifFormatter"
llm = "vibelint.reporting:LLMFormatter"

[project.entry-points."vibelint.workflows"]
# Entry points for EXTERNAL packages to extend vibelint with custom workflows
# Built-in workflows are registered directly in code, not here

# Third-party workflows can be added here by other packages:
# Examples of different workflow types:
# Simple function workflow:
# "my-simple-check" = "my_package.workflows:simple_check_function"
#
# Class-based workflow:
# "my-complex-workflow" = "my_package.workflows:MyComplexWorkflow"
#
# Multi-file workflow module:
# "my-enterprise-workflow" = "my_package.workflows.enterprise_suite:EnterpriseWorkflowEngine"

[project.urls]
"Homepage" = "https://github.com/mithranm/vibelint"
"Bug Tracker" = "https://github.com/mithranm/vibelint/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["vibelint*"]

[tool.black]
target-version = ["py310", "py311", "py312"]
line-length=100

[tool.ruff]
line-length = 100
target-version = "py310"
exclude = [
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv"
]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
]
ignore = [
    "E501",   # Line too long (handled by black)
    "B008",   # Allow function calls in argument defaults
    "B904",   # Allow raise without from for now
    "UP036",  # Allow outdated version blocks
    "C414",   # Allow unnecessary list() calls for clarity
    "SIM102", # Allow nested if statements
    "SIM210", # Allow if-else instead of bool()
    "SIM113", # Allow manual index tracking
    "SIM103", # Allow explicit return True/False
    "N802",   # Allow uppercase function names (LibCST visitor methods)
    "N806",   # Allow uppercase variables
    "B006",   # Allow mutable defaults for now
    "B007",   # Allow unused loop variables
    "UP007",  # Allow Union syntax for now
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["F401", "F811"]  # Allow unused imports in tests

[tool.setuptools.package-data]
vibelint = ["VIBECHECKER.txt"]

# Local vibelint config for analyzing vibelint itself
[tool.vibelint]
include_globs = [
    "**/*"
]
exclude_globs = [
    "**/__pycache__/**",
    "**/.*",           # dotfiles/directories (nested)
    ".*",              # dotfiles at root level
    "build/**",
    "dist/**",
    "*cache*/**",      # any cache directories
    "*.vibelint*",     # vibelint artifacts
    ".vibelint-*/**",  # old vibelint directories (deprecated)
    ".vibes/**",       # vibelint analysis output
    "*.pyc", "*.pyo", "*.pyd",  # compiled python
    "*.log",            # log files
    "*.xml",            # xml files
    "*.json",           # json files
    "docs/**",
]
# LLM config will be inherited from parent project

# Workflow-specific configurations
[tool.vibelint.workflows.justification]
# Configuration for justification workflow
max_file_size_lines = 2000  # Files larger than this get flagged for splitting
enable_llm_analysis = true
enable_static_analysis = true
enable_dependency_analysis = true
reports_directory = ".vibes/reports/justification_workflow"

[tool.vibelint.workflows.cleanup]
# Configuration for cleanup workflow
aggressive_mode = false
backup_before_cleanup = true

# Third-party workflows can define their own config sections:
# [tool.vibelint.workflows.my-custom-workflow]
# custom_setting = "value"
# another_setting = 123

```

---
### File: src/vibelint/VIBECHECKER.txt

```
                                                   :=+*++=:                                         
                                                :+*=:.   -+=                                        
                                      :=+*+==:. -=.==+*+=+=*                                        
                                   :==::. ...:: :+*+:.::**+#                                        
                                 ..-:-:=+*+==:.   :-=**%*:%=                                        
                              :-:. :==:  ...    .-***%*-+#=:      ::=+=: ....... :=+=:.             
      :==+***:              :==:-...   :=%@%#*#%@@@%+.+%%=     :+=: -:=:.     .  :-.-:-**:          
    :::: .:.-=*=.         :=#=:=: :--::*@%#%%@@%%--**@%*:   :==-:-::=+=: ....... :=+=:=--=*+:       
   :-::==+**+-:+*+:      -#%=++:  -+*%%@#=:.::*%@@@@@*:   :+=:-:+=:                   -*+-:=*=.     
   =:=:     :+*=-*#-.  :*@@:**:   -#*%@#=-:=%@@%%*:     .-+::==:                        :=*+.=*=    
   =:==::.    :+%=+::+#@@@=**-.  :*@#-:..:*@%+:        .==.++:                            .=*=-#:   
   :=:**=:      =**::-=##:+#*====+@%:==::+@%-         :+=:+-                                .**:*=. 
   .-*=-**:        .:+%@%-==*#%#++*-*+-::@@-         :*=-+:                                  :=*.*: 
    .=%*:#%*=.        -#@=::-=-==--::== %@+         :#==+:       :=#*=++-    :*%@%%*:          =*:* 
      -%+=*%%+.     .:=+%=-----==-:.:*==@*:        :**=*:       :==.=:--==: :=+--=-+=-.         =++ 
       +%==:%@%*=: :*%#%%=:::-=+==:::=:@@=         =%:%-       :*+::=%*==+=-*=:=-==-*#-         -*: 
      :=:*##:#%@@@*+%%##+:: . :.::::-:%@*:         #%##        :#:=*%##*.=+**==@%*#--#*         .=* 
   .-==:-:**=::-*@@#*=:=.:::::::-:.==+@*:         :%*%%        :%-==*=+=:=++%.==+=+=-#*          -% 
   -=-:===%#-. .::**+:.=.:: .:.::.:=:@@:          :@+@#        .+%=:==+==*-:*#+::=+***-          -% 
  :*-==: =@#::..:.:=:..=.::....:::::@@+           :@=@*         :*%###*##=. :*%%*#@%+:           -% 
  =*-*.  :@@%=::::.-::.=.::...:-:::*@#:           .@*%*           .-===:.     -*#*=:             :% 
 .====    =%@@%*==:-:.:-:-=-:.::.-*@%:             #***.                                         :# 
  ===*.    .-*%%#**=:.::=-:.: ::=%@*-              =#-#:                                         =# 
  =#-#=-:     -+#%#=:.::---:.=#@@%=.               :%:%+:  ::::-----::::::::::---:::::::-=-:.  .-#+ 
 :==++:*%*:   ..:=**-::-==-#@@@@*:                 .*%:%-. :. ..   ...   .    . ... ..  . .:: .-#*= 
 =--*=+==%%=.  .:=%*:..==@@@@*=.                    :#+**:.::::-----::::::::::---:::::::--::.:=**-% 
 +.+:.=*==@@@%**#%#-:..-@@@=.                       .=%:**-.   ...                     .:::::=**.#+ 
 *:+. +%+=-%@@@@#*=.:.:#@*.                          .=%==*- .::::                     :::::=#=:**: 
 +-+..=#==-:-:==--::.:%@*:                            .=*+:*-:::::...                 :-::=*%=+%*:  
 === .-*%%*:=::::: =+%@=.                               :=*:=+=-:::::       ...      .-===+-:#%=.   
 -:- .:-*%*-=:-:.:*@@%=                                   -+-:-=+==-:      .:::.    .:++-.+%@#-     
 =.=.   -*#=..:=*%@%=:                                     :=**=:=*%#=-=:. :::-==+**%#--=*#*:       
 -:=+:  :=#**%%@@%*-                                         :=*##+==:::=*%@%%%#+-:.::+**=:         
 :==**: :=#.%%%*-                                               :*%@*=-=:-.::-==++*##%+:            
  -#:*==*%-**                                                       ..::=*%@@%%%+-:.                
   =*=:.-=*#:                                                                                       
   .-===*#*-     
```

---
### File: src/vibelint/__init__.py

```python
"""
Vibelint: Code Quality and Style Validator

A focused code analysis tool for Python projects.
"""

from vibelint.config import (
    Config,
    EmbeddingConfig,
    LLMConfig,
    get_embedding_config,
    get_llm_config,
    load_config,
)
from vibelint.llm_client import (
    LLMManager,
    LLMRequest,
    LLMResponse,
    LLMRole,
    create_llm_manager,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "Config",
    "load_config",
    "LLMConfig",
    "EmbeddingConfig",
    "get_llm_config",
    "get_embedding_config",
    # LLM Client
    "LLMManager",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "create_llm_manager",
]
```

---
### File: src/vibelint/__main__.py

```python
"""
Main entry point for vibelint when run as a module.

Allows execution via: python -m vibelint

vibelint/src/vibelint/__main__.py
"""

from vibelint.cli import main

if __name__ == "__main__":
    main()
```

---
### File: src/vibelint/api.py

```python
"""
Public API for vibelint that returns results instead of calling sys.exit().

This module provides a clean library interface for programmatic usage of vibelint,
allowing integration with other tools without subprocess overhead.

The CLI commands wrap these API functions and handle exit codes/formatting.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vibelint.filesystem import walk_up_for_config
from vibelint.validation_engine import PluginValidationRunner


@dataclass
class FindingDict:
    """Serializable representation of a Finding for API responses."""
    rule: str
    level: str
    path: str
    line: int
    column: int
    msg: str
    context: str = ""
    suggestion: str = ""


@dataclass
class FindingSummary:
    """Summary of findings by severity level."""
    INFO: int = 0
    WARN: int = 0
    BLOCK: int = 0


@dataclass
class CheckResults:
    """Results from a check operation."""
    findings: List[FindingDict]
    summary: FindingSummary
    total_files_checked: int


@dataclass
class VibelintResult:
    """Container for vibelint operation results."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class VibelintAPI:
    """Main API interface for vibelint operations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None, working_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the vibelint API.

        Args:
            config_path: Path to vibelint config file (optional)
            working_dir: Working directory for operations (optional, defaults to current dir)
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

        # Find project root and load config
        self.project_root = walk_up_for_config(self.working_dir)
        self.config_dict = self._load_config()

        # Set up logging to capture errors without outputting to console
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load vibelint configuration or return defaults."""
        if not self.project_root:
            return {}

        try:
            import tomllib
            config_file = self.project_root / "pyproject.toml"
            if config_file.exists():
                with open(config_file, "rb") as f:
                    data = tomllib.load(f)
                    return data.get("tool", {}).get("vibelint", {})
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")

        return {}

    def check(self, targets: Optional[List[str]] = None, exclude_ai: bool = False,
              rules: Optional[List[str]] = None) -> VibelintResult:
        """
        Run vibelint validation checks.

        Args:
            targets: List of files/directories to check (defaults to current directory)
            exclude_ai: Skip AI-powered validators for faster execution
            rules: Specific rules to run (comma-separated list)

        Returns:
            VibelintResult with validation results
        """
        try:
            targets = targets or ["."]

            # Convert string targets to Path objects and discover files
            from vibelint.discovery import discover_files_from_paths
            target_paths = [Path(t) for t in targets]
            file_paths = discover_files_from_paths(target_paths)

            # Create runner with config
            runner = PluginValidationRunner(self.config_dict, self.project_root or self.working_dir)

            # Run validation
            findings = runner.run_validation(file_paths)

            # Convert findings to our format and create summary
            all_findings = []
            summary = FindingSummary()

            for finding in findings:
                finding_dict = FindingDict(
                    rule=finding.rule_id,
                    level=finding.severity.name,
                    path=str(finding.file_path),
                    line=finding.line_number,
                    column=finding.column_number,
                    msg=finding.message,
                    context=finding.context or "",
                    suggestion=finding.suggestion or ""
                )
                all_findings.append(finding_dict)

                # Update summary
                level = finding.severity.name
                if level == "INFO":
                    summary.INFO += 1
                elif level == "WARN":
                    summary.WARN += 1
                elif level == "BLOCK":
                    summary.BLOCK += 1

            check_results = CheckResults(
                findings=all_findings,
                summary=summary,
                total_files_checked=len(file_paths)
            )
            return VibelintResult(True, asdict(check_results))

        except Exception as e:
            self.logger.error(f"Check operation failed: {e}")
            return VibelintResult(False, errors=[str(e)])

    def validate_file(self, file_path: Union[str, Path]) -> VibelintResult:
        """
        Validate a single file.

        Args:
            file_path: Path to file to validate

        Returns:
            VibelintResult with validation results for the file
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return VibelintResult(False, errors=[f"File does not exist: {path}"])

            if not path.is_file():
                return VibelintResult(False, errors=[f"Path is not a file: {path}"])

            # Create runner with config
            runner = PluginValidationRunner(self.config_dict, self.project_root or self.working_dir)

            # Run validation on single file
            findings = runner.run_validation([path])

            # Convert findings to our format and create summary
            all_findings = []
            summary = {"INFO": 0, "WARN": 0, "BLOCK": 0}

            for finding in findings:
                finding_dict = {
                    "rule": finding.rule_id,
                    "level": finding.severity.name,
                    "path": str(finding.file_path),
                    "line": finding.line_number,
                    "column": finding.column_number,
                    "msg": finding.message,
                    "context": finding.context or "",
                    "suggestion": finding.suggestion or ""
                }
                all_findings.append(finding_dict)

                # Update summary
                level = finding.severity.name
                summary[level] = summary.get(level, 0) + 1

            return VibelintResult(True, {
                "file": str(path),
                "findings": all_findings,
                "summary": summary
            })

        except Exception as e:
            return VibelintResult(False, errors=[str(e)])

    def run_justification(self, target_dir: Optional[str] = None) -> VibelintResult:
        """
        Run justification workflow for architectural analysis.

        Args:
            target_dir: Directory to analyze (defaults to current directory)

        Returns:
            VibelintResult with justification analysis
        """
        try:
            target_path = Path(target_dir) if target_dir else self.working_dir

            if not target_path.exists():
                return VibelintResult(False, errors=[f"Target directory does not exist: {target_path}"])

            # For now, return a simplified justification result
            # The full justification engine requires more complex setup
            return VibelintResult(True, {
                "analysis": {"message": "Justification analysis not yet implemented in API"},
                "target_directory": str(target_path)
            })

        except Exception as e:
            self.logger.error(f"Justification workflow failed: {e}")
            return VibelintResult(False, errors=[str(e)])


# Convenience functions for common operations
def check_files(targets: Optional[List[str]] = None, config_path: Optional[str] = None,
                exclude_ai: bool = False, rules: Optional[List[str]] = None) -> VibelintResult:
    """
    Convenience function to check files/directories.

    Args:
        targets: Files/directories to check
        config_path: Path to config file
        exclude_ai: Skip AI validators
        rules: Specific rules to run

    Returns:
        VibelintResult with validation results
    """
    api = VibelintAPI(config_path)
    return api.check(targets, exclude_ai, rules)


def validate_single_file(file_path: Union[str, Path], config_path: Optional[str] = None) -> VibelintResult:
    """
    Convenience function to validate a single file.

    Args:
        file_path: Path to file to validate
        config_path: Path to config file

    Returns:
        VibelintResult with validation results
    """
    api = VibelintAPI(config_path)
    return api.validate_file(file_path)


def run_project_justification(target_dir: Optional[str] = None, config_path: Optional[str] = None) -> VibelintResult:
    """
    Convenience function to run justification analysis.

    Args:
        target_dir: Directory to analyze
        config_path: Path to config file

    Returns:
        VibelintResult with justification analysis
    """
    api = VibelintAPI(config_path)
    return api.run_justification(target_dir)
```

---
### File: src/vibelint/ast_utils.py

```python
"""
AST parsing utilities for vibelint validators.

Provides common AST parsing functionality with consistent error handling,
reducing code duplication across validators.

vibelint/src/vibelint/ast_utils.py
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "safe_parse",
    "parse_or_none",
]


def safe_parse(content: str, filename: str | Path = "<unknown>") -> ast.AST:
    """
    Parse Python source code into an AST.

    Args:
        content: Python source code as string
        filename: Optional filename for error messages

    Returns:
        Parsed AST

    Raises:
        SyntaxError: If the code has syntax errors
    """
    return ast.parse(content, filename=str(filename))


def parse_or_none(content: str, filename: str | Path = "<unknown>") -> Optional[ast.AST]:
    """
    Parse Python source code into an AST, returning None on syntax errors.

    This is the recommended function for validators to use - it handles
    syntax errors gracefully and logs them appropriately.

    Args:
        content: Python source code as string
        filename: Optional filename for error messages

    Returns:
        Parsed AST or None if parsing failed

    Example:
        >>> tree = parse_or_none(file_content, file_path)
        >>> if tree is None:
        >>>     return  # Skip validation for files with syntax errors
    """
    try:
        return ast.parse(content, filename=str(filename))
    except SyntaxError as e:
        logger.debug(f"Syntax error in {filename}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing {filename}: {e}")
        return None


def get_docstring(node: ast.AST) -> Optional[str]:
    """
    Extract docstring from an AST node (module, function, or class).

    Args:
        node: AST node (Module, FunctionDef, ClassDef, or AsyncFunctionDef)

    Returns:
        Docstring text or None if no docstring found
    """
    if not isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return None

    body = node.body if hasattr(node, "body") else []
    if not body:
        return None

    first_stmt = body[0]
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        value = first_stmt.value.value
        if isinstance(value, str):
            return value

    return None


def get_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """
    Get argument names from a function definition.

    Args:
        node: Function definition node

    Returns:
        List of argument names
    """
    args = []

    # Regular arguments
    for arg in node.args.args:
        args.append(arg.arg)

    # *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")

    # **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return args


def is_private_name(name: str) -> bool:
    """
    Check if a name is private (starts with underscore).

    Args:
        name: Name to check

    Returns:
        True if name is private (starts with _ but not __)
    """
    return name.startswith("_") and not name.startswith("__")


def is_dunder_name(name: str) -> bool:
    """
    Check if a name is a dunder/magic method (starts and ends with __).

    Args:
        name: Name to check

    Returns:
        True if name is a dunder method
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4
```

---
### File: src/vibelint/cli.py

```python
"""
CLI for vibelint - all commands in one module.

Provides core commands: check, snapshot.

vibelint/src/vibelint/cli.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class VibelintContext:
    """Shared context for CLI commands."""

    project_root: Path | None = None
    config_path: Path | None = None
    verbose: bool = False


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """vibelint: Code quality linter with dynamic plugin discovery."""
    # Auto-detect project root
    current = Path.cwd()
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            project_root = parent
            break

    # Store context for subcommands
    ctx.obj = VibelintContext(
        project_root=project_root,
        config_path=None,
        verbose=verbose,
    )

    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command("check")
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--format", "-f", type=click.Choice(["human", "json"]), default="human", help="Output format")
@click.option("--exclude-ai", is_flag=True, help="Skip AI validators (faster)")
@click.option("--rules", help="Comma-separated rules to run")
@click.pass_context
def check(ctx: click.Context, targets: tuple[Path, ...], format: str, exclude_ai: bool, rules: str | None) -> None:
    """Run vibelint validation."""
    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root

    if not project_root:
        console.print("[red]❌ No project root found[/red]")
        ctx.exit(1)

    # Load config
    from vibelint.config import Config, load_config

    config: Config = load_config(project_root)
    if not config.is_present():
        console.print("[red]❌ No vibelint configuration found[/red]")
        ctx.exit(1)

    # Import validation engine
    from vibelint.validation_engine import PluginValidationRunner
    from vibelint.discovery import discover_files_from_paths

    # Determine target files
    if targets:
        files = discover_files_from_paths(list(targets), config)
    else:
        files = discover_files_from_paths([project_root], config)

    if not files:
        console.print("No Python files found")
        ctx.exit(0)

    # Get config dict for filtering
    config_dict = dict(config.settings)

    # Filter AI validators if requested
    if exclude_ai:
        if "rules" in config_dict and "enable" in config_dict["rules"]:
            enabled = config_dict["rules"]["enable"]
            config_dict["rules"]["enable"] = [r for r in enabled if not r.endswith("-LLM")]

    # Filter specific rules if requested
    if rules:
        rule_list = [r.strip() for r in rules.split(",")]
        config_dict["rules"] = {"enable": rule_list}

    # Run validation
    runner = PluginValidationRunner(config_dict, project_root)
    findings = runner.run_validation(files)

    # Output results
    output = runner.format_output(format)
    print(output)

    # Exit with proper code
    errors = sum(1 for f in findings if f.severity.name == "ERROR")
    warnings = sum(1 for f in findings if f.severity.name == "WARN")

    if format == "human" and (errors or warnings):
        console.print(f"\nFound {errors} error(s), {warnings} warning(s)")

    ctx.exit(1 if errors > 0 else 0)


@cli.command("snapshot")
@click.argument("targets", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default="codebase_snapshot.md", help="Output markdown file path")
@click.pass_context
def snapshot(ctx: click.Context, targets: tuple[Path, ...], output: Path) -> None:
    """Create a markdown snapshot of the codebase structure and contents."""
    from vibelint.config import load_config, Config
    from vibelint.snapshot import create_snapshot

    vibelint_ctx: VibelintContext = ctx.obj
    project_root = vibelint_ctx.project_root or Path.cwd()

    # Load config
    try:
        config = load_config(project_root)
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not load config: {e}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        config = Config(project_root=project_root)

    # Default targets to project root if none provided
    if not targets:
        targets = [project_root]

    target_list = list(targets)

    try:
        console.print(f"[blue]📸 Creating snapshot of {len(target_list)} target(s)...[/blue]")
        create_snapshot(
            output_path=output,
            target_paths=target_list,
            config=config
        )
        console.print(f"[green]✅ Snapshot saved to {output}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Snapshot failed: {e}[/red]")
        logger.error(f"Snapshot error: {e}", exc_info=True)
        ctx.exit(1)



def main() -> None:
    """Entry point for vibelint CLI."""
    import sys
    try:
        cli(obj=VibelintContext(), prog_name="vibelint")
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.error("CLI error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---
### File: src/vibelint/config.py

```python
"""
Configuration loading for vibelint.

Reads settings *only* from pyproject.toml under the [tool.vibelint] section.
No default values are assumed by this module. Callers must handle missing
configuration keys.

vibelint/src/vibelint/config.py
"""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from vibelint.filesystem import walk_up_for_config

logger = logging.getLogger(__name__)


def _find_config_file(project_root: Path) -> Path | None:
    """Find the config file (pyproject.toml or dev.pyproject.toml) with vibelint settings."""
    # Check standard pyproject.toml first
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "tool" in data and "vibelint" in data.get("tool", {}):
                    return pyproject_path
        except Exception:
            pass

    # Check dev.pyproject.toml (kaia pattern)
    dev_pyproject_path = project_root / "dev.pyproject.toml"
    if dev_pyproject_path.exists():
        try:
            with open(dev_pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "tool" in data and "vibelint" in data.get("tool", {}):
                    return dev_pyproject_path
        except Exception:
            pass

    return None


def _load_parent_config(project_root: Path, current_config_path: Path) -> dict | None:
    """Load parent configuration for inheritance."""
    # Walk up from the project root to find parent configurations
    parent_path = project_root.parent

    while parent_path != parent_path.parent:  # Stop at filesystem root
        # Check for dev.pyproject.toml (kaia pattern)
        dev_config = parent_path / "dev.pyproject.toml"
        if dev_config.exists() and dev_config != current_config_path:
            try:
                with open(dev_config, "rb") as f:
                    data = tomllib.load(f)
                    vibelint_config = data.get("tool", {}).get("vibelint", {})
                    if vibelint_config:
                        logger.debug(f"Found parent config in {dev_config}")
                        return vibelint_config
            except Exception:
                pass

        # Check for regular pyproject.toml
        parent_config = parent_path / "pyproject.toml"
        if parent_config.exists() and parent_config != current_config_path:
            try:
                with open(parent_config, "rb") as f:
                    data = tomllib.load(f)
                    vibelint_config = data.get("tool", {}).get("vibelint", {})
                    if vibelint_config:
                        logger.debug(f"Found parent config in {parent_config}")
                        return vibelint_config
            except Exception:
                pass

        parent_path = parent_path.parent

    return None


if sys.version_info >= (3, 11):

    import tomllib
else:

    try:

        import tomli as tomllib
    except ImportError as e:

        raise ImportError(
            "vibelint requires Python 3.11+ or the 'tomli' package "
            "to parse pyproject.toml on Python 3.10. "
            "Hint: Try running: pip install tomli"
        ) from e


class Config:
    """
    Holds the vibelint configuration loaded *exclusively* from pyproject.toml.

    Provides access to the project root and the raw configuration dictionary.
    It does *not* provide default values for missing keys. Callers must
    check for the existence of required settings.

    Attributes:
    project_root: The detected root of the project containing pyproject.toml.
    Can be None if pyproject.toml is not found.
    settings: A read-only view of the dictionary loaded from the
    [tool.vibelint] section of pyproject.toml. Empty if the
    file or section is missing or invalid.

    vibelint/src/vibelint/config.py
    """

    def __init__(self, project_root: Path | None, config_dict: dict[str, Any]):
        """
        Initializes Config.

        vibelint/src/vibelint/config.py
        """
        self._project_root = project_root
        self._config_dict = config_dict.copy()

    @property
    def project_root(self) -> Path | None:
        """
        The detected project root directory, or None if not found.

        vibelint/src/vibelint/config.py
        """
        return self._project_root

    @property
    def settings(self) -> Mapping[str, Union[str, bool, int, list, dict]]:
        """
        Read-only view of the settings loaded from [tool.vibelint].

        vibelint/src/vibelint/config.py
        """
        return self._config_dict

    @property
    def ignore_codes(self) -> list[str]:
        """
        Returns the list of error codes to ignore, from config or empty list.

        vibelint/src/vibelint/config.py
        """
        ignored = self.get("ignore", [])
        if isinstance(ignored, list) and all(isinstance(item, str) for item in ignored):
            return ignored

        # Handle invalid configuration
        if ignored:
            logger.warning(
                "Configuration key 'ignore' in [tool.vibelint] is not a list of strings. Ignoring it."
            )

        return []

    def get(
        self, key: str, default: Union[str, bool, int, list, dict, None] = None
    ) -> Union[str, bool, int, list, dict, None]:
        """
        Gets a value from the loaded settings, returning default if not found.

        vibelint/src/vibelint/config.py
        """
        return self._config_dict.get(key, default)

    def __getitem__(self, key: str) -> Union[str, bool, int, list, dict]:
        """
        Gets a value, raising KeyError if the key is not found.

        vibelint/src/vibelint/config.py
        """
        if key not in self._config_dict:
            raise KeyError(
                f"Required configuration key '{key}' not found in "
                f"[tool.vibelint] section of pyproject.toml."
            )
        return self._config_dict[key]

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key exists in the loaded settings.

        vibelint/src/vibelint/config.py
        """
        return key in self._config_dict

    def is_present(self) -> bool:
        """
        Checks if a project root was found and some settings were loaded.

        vibelint/src/vibelint/config.py
        """
        return self._project_root is not None and bool(self._config_dict)


def load_hierarchical_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration with hierarchical merging.

    1. Loads local config (file patterns, local settings)
    2. Walks up to find parent config (LLM settings, shared config)
    3. Merges them: local config takes precedence for file patterns,
       parent config provides LLM settings

    Args:
    start_path: The directory to start searching from.

    Returns:
    A Config object with merged local and parent settings.
    """
    # Find local config first
    local_root = find_package_root(start_path)
    local_settings = {}

    if local_root:
        pyproject_path = local_root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    local_settings = data.get("tool", {}).get("vibelint", {})
                    if local_settings:
                        logger.info(f"Loaded local vibelint config from {pyproject_path}")
            except Exception as e:
                logger.warning(f"Failed to load local config from {pyproject_path}: {e}")

    # Walk up to find parent config with LLM settings
    parent_settings = {}
    current_path = start_path.parent if start_path.is_file() else start_path

    while current_path.parent != current_path:
        parent_pyproject = current_path / "pyproject.toml"
        if parent_pyproject.exists() and parent_pyproject != (local_root / "pyproject.toml" if local_root else None):
            try:
                with open(parent_pyproject, "rb") as f:
                    data = tomllib.load(f)
                    parent_config = data.get("tool", {}).get("vibelint", {})
                    if parent_config:
                        parent_settings = parent_config
                        logger.info(f"Found parent vibelint config at {parent_pyproject}")
                        break
            except Exception as e:
                logger.debug(f"Failed to read {parent_pyproject}: {e}")
        current_path = current_path.parent

    # Merge configs: local file patterns override parent, but inherit LLM settings
    merged_settings = parent_settings.copy()

    # Local config takes precedence for file discovery patterns
    if local_settings:
        for key in ["include_globs", "exclude_globs", "ignore"]:
            if key in local_settings:
                merged_settings[key] = local_settings[key]

        # Also copy other local-specific settings
        for key in local_settings:
            if key not in ["include_globs", "exclude_globs", "ignore"]:
                merged_settings[key] = local_settings[key]

    return Config(local_root or start_path, merged_settings)


def load_config(start_path: Path) -> Config:
    """
    Loads vibelint configuration with auto-discovery fallback.

    First tries manual config from pyproject.toml, then falls back to
    zero-config auto-discovery for seamless single->multi-project scaling.

    Args:
    start_path: The directory to start searching upwards for pyproject.toml.

    Returns:
    A Config object with either manual or auto-discovered settings.

    vibelint/src/vibelint/config.py
    """
    project_root = walk_up_for_config(start_path)
    loaded_settings: dict[str, Any] = {}

    # Try auto-discovery first for zero-config scaling
    try:
        from vibelint.auto_discovery import discover_and_configure

        auto_config = discover_and_configure(start_path)

        # If we found a multi-project setup, use auto-discovery by default
        if auto_config.get("discovered_topology") == "multi_project":
            logger.info(f"Auto-discovered multi-project setup from {start_path}")
            # Convert auto-discovered config to vibelint config format
            loaded_settings = _convert_auto_config_to_vibelint(auto_config)
            project_root = project_root or start_path

            # Still allow manual config to override auto-discovery
            manual_override = _load_manual_config(project_root)
            if manual_override:
                logger.debug("Manual config found, merging with auto-discovery")
                loaded_settings.update(manual_override)

            return Config(project_root=project_root, config_dict=loaded_settings)

    except ImportError:
        logger.debug("Auto-discovery not available, using manual config only")
    except Exception as e:
        logger.debug(f"Auto-discovery failed: {e}, falling back to manual config")

    if not project_root:
        logger.warning(
            f"Could not find project root (pyproject.toml) searching from '{start_path}'. "
            "No configuration will be loaded."
        )
        return Config(project_root=None, config_dict=loaded_settings)

    # Try both pyproject.toml and dev.pyproject.toml
    pyproject_path = _find_config_file(project_root)
    logger.debug(f"Found project root: {project_root}")

    if not pyproject_path:
        logger.debug(f"No vibelint configuration found in {project_root}")
        return Config(project_root, {})

    logger.debug(f"Attempting to load config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:
            full_toml_config = tomllib.load(f)
        logger.debug(f"Parsed {pyproject_path.name}")

        # Validate required configuration structure explicitly
        tool_section = full_toml_config.get("tool")
        if not isinstance(tool_section, dict):
            logger.warning("pyproject.toml [tool] section is missing or invalid")
            vibelint_config = {}
        else:
            vibelint_config = tool_section.get("vibelint", {})

        if isinstance(vibelint_config, dict):
            loaded_settings = vibelint_config
            # Check for parent config inheritance
            parent_config = _load_parent_config(project_root, pyproject_path)
            if parent_config:
                # Merge parent config with local config (local takes precedence)
                merged_settings = parent_config.copy()
                merged_settings.update(loaded_settings)
                loaded_settings = merged_settings
                logger.debug("Merged parent configuration")

            if loaded_settings:
                logger.debug(f"Loaded [tool.vibelint] settings from {pyproject_path}")
                logger.debug(f"Loaded settings: {loaded_settings}")
            else:
                logger.info(
                    f"Found {pyproject_path}, but the [tool.vibelint] section is empty or missing."
                )
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table (dictionary). "
                "Ignoring this section."
            )

    except FileNotFoundError:

        logger.error(
            f"pyproject.toml not found at {pyproject_path} despite project root detection."
        )
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}. Using empty configuration.")
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}. Using empty configuration.")
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error processing configuration from {pyproject_path}: {e}")
        logger.debug("Unexpected error loading config", exc_info=True)

    return Config(project_root=project_root, config_dict=loaded_settings)


def _convert_auto_config_to_vibelint(auto_config: dict[str, Any]) -> dict[str, Any]:
    """Convert auto-discovered config to vibelint config format."""
    vibelint_config = {}

    # Auto-route validation based on discovered services
    services = auto_config.get("services", {})
    routing = auto_config.get("auto_routing", {})

    # Set include globs based on discovered projects
    include_globs = []
    for service_info in services.values():
        service_path = Path(service_info["path"])
        include_globs.extend([f"{service_path.name}/src/**/*.py", f"{service_path.name}/**/*.py"])

    vibelint_config["include_globs"] = include_globs

    # Configure distributed services if available
    if auto_config.get("discovered_topology") == "multi_project":
        vibelint_config["distributed"] = {
            "enabled": True,
            "auto_discovered": True,
            "services": services,
            "routing": routing,
        }

        # Use shared resources if discovered
        shared_resources = auto_config.get("shared_resources", {})
        if shared_resources.get("vector_stores"):
            vibelint_config["vector_store"] = {
                "backend": "qdrant",
                "qdrant_collection": shared_resources["vector_stores"][0],
            }

    return vibelint_config


def _load_manual_config(project_root: Path | None) -> dict[str, Any]:
    """Load manual configuration from pyproject.toml."""
    if not project_root:
        return {}

    pyproject_path = project_root / "pyproject.toml"
    logger.debug(f"Attempting to load manual config from: {pyproject_path}")

    try:
        with open(pyproject_path, "rb") as f:
            full_toml_config = tomllib.load(f)
        logger.debug("Parsed pyproject.toml")

        # Validate required configuration structure explicitly
        tool_section = full_toml_config.get("tool")
        if not isinstance(tool_section, dict):
            logger.warning("pyproject.toml [tool] section is missing or invalid")
            return {}

        vibelint_config = tool_section.get("vibelint", {})

        if isinstance(vibelint_config, dict):
            if vibelint_config:
                logger.debug(f"Loaded manual [tool.vibelint] settings from {pyproject_path}")
                return vibelint_config
            else:
                logger.debug(f"Found {pyproject_path}, but [tool.vibelint] section is empty")
                return {}
        else:
            logger.warning(
                f"[tool.vibelint] section in {pyproject_path} is not a valid table. Ignoring."
            )
            return {}

    except FileNotFoundError:
        logger.debug(f"No pyproject.toml found at {pyproject_path}")
        return {}
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing {pyproject_path}: {e}")
        return {}
    except OSError as e:
        logger.error(f"Error reading {pyproject_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading manual config: {e}")
        return {}


# Configuration constants for LLM
DEFAULT_FAST_TEMPERATURE = 0.1
DEFAULT_FAST_MAX_TOKENS = 2048
DEFAULT_ORCHESTRATOR_TEMPERATURE = 0.2
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 8192
DEFAULT_CONTEXT_THRESHOLD = 3000


@dataclass
class LLMConfig:
    """Typed LLM configuration with explicit validation."""

    # Required fields first
    fast_api_url: str
    fast_model: str
    orchestrator_api_url: str
    orchestrator_model: str

    # Optional fields with defaults
    fast_backend: str = "vllm"
    fast_temperature: float = DEFAULT_FAST_TEMPERATURE
    fast_max_tokens: int = DEFAULT_FAST_MAX_TOKENS
    fast_max_context_tokens: Optional[int] = None
    fast_api_key: Optional[str] = None

    orchestrator_backend: str = "llamacpp"
    orchestrator_temperature: float = DEFAULT_ORCHESTRATOR_TEMPERATURE
    orchestrator_max_tokens: int = DEFAULT_ORCHESTRATOR_MAX_TOKENS
    orchestrator_max_context_tokens: Optional[int] = None
    orchestrator_api_key: Optional[str] = None

    context_threshold: int = DEFAULT_CONTEXT_THRESHOLD
    enable_context_probing: bool = True
    enable_fallback: bool = False

    def __post_init__(self):
        """Validate required configuration."""
        if not self.fast_api_url:
            raise ValueError("fast_api_url is required - configure in [tool.vibelint.llm]")
        if not self.fast_model:
            raise ValueError("fast_model is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_api_url:
            raise ValueError("orchestrator_api_url is required - configure in [tool.vibelint.llm]")
        if not self.orchestrator_model:
            raise ValueError("orchestrator_model is required - configure in [tool.vibelint.llm]")


@dataclass
class EmbeddingConfig:
    """Typed embedding configuration with explicit validation."""

    # Required fields first
    code_api_url: str
    natural_api_url: str

    # Optional fields with defaults
    code_model: str = "text-embedding-ada-002"
    natural_model: str = "text-embedding-ada-002"
    use_specialized_embeddings: bool = True

    def __post_init__(self):
        """Validate required configuration."""
        if not self.code_api_url:
            raise ValueError("code_api_url is required - configure in [tool.vibelint.embeddings]")
        if not self.natural_api_url:
            raise ValueError(
                "natural_api_url is required - configure in [tool.vibelint.embeddings]"
            )


def _get_env_float(key: str) -> Optional[float]:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_env_int(key: str) -> Optional[int]:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None


def get_llm_config(config: Optional[Config] = None) -> LLMConfig:
    """
    Get typed LLM configuration for vibelint.

    Args:
        config: Optional Config object. If None, loads from current directory.

    Returns:
        Validated LLMConfig object with environment variable overrides
    """
    if config is None:
        config = load_config(Path.cwd())

    llm_dict = config.get("llm", {})
    if not isinstance(llm_dict, dict):
        llm_dict = {}

    # Build kwargs with environment overrides
    kwargs = {
        # Fast LLM
        "fast_api_url": (
            os.getenv("VIBELINT_FAST_LLM_API_URL")
            or os.getenv("FAST_LLM_API_URL")
            or llm_dict.get("fast_api_url")
        ),
        "fast_model": (
            os.getenv("VIBELINT_FAST_LLM_MODEL")
            or os.getenv("FAST_LLM_MODEL")
            or llm_dict.get("fast_model")
        ),
        "fast_backend": (
            os.getenv("VIBELINT_FAST_LLM_BACKEND")
            or os.getenv("FAST_LLM_BACKEND")
            or llm_dict.get("fast_backend", "vllm")
        ),
        "fast_api_key": (
            os.getenv("VIBELINT_FAST_LLM_API_KEY")
            or os.getenv("FAST_LLM_API_KEY")
            or llm_dict.get("fast_api_key")
        ),
        "fast_temperature": (
            _get_env_float("VIBELINT_FAST_LLM_TEMPERATURE")
            or _get_env_float("FAST_LLM_TEMPERATURE")
            or llm_dict.get("fast_temperature", DEFAULT_FAST_TEMPERATURE)
        ),
        "fast_max_tokens": (
            _get_env_int("VIBELINT_FAST_LLM_MAX_TOKENS")
            or _get_env_int("FAST_LLM_MAX_TOKENS")
            or llm_dict.get("fast_max_tokens", DEFAULT_FAST_MAX_TOKENS)
        ),
        "fast_max_context_tokens": llm_dict.get("fast_max_context_tokens"),
        # Orchestrator LLM
        "orchestrator_api_url": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_URL")
            or os.getenv("ORCHESTRATOR_LLM_API_URL")
            or llm_dict.get("orchestrator_api_url")
        ),
        "orchestrator_model": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_MODEL")
            or os.getenv("ORCHESTRATOR_LLM_MODEL")
            or llm_dict.get("orchestrator_model")
        ),
        "orchestrator_backend": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_BACKEND")
            or os.getenv("ORCHESTRATOR_LLM_BACKEND")
            or llm_dict.get("orchestrator_backend", "llamacpp")
        ),
        "orchestrator_api_key": (
            os.getenv("VIBELINT_ORCHESTRATOR_LLM_API_KEY")
            or os.getenv("ORCHESTRATOR_LLM_API_KEY")
            or llm_dict.get("orchestrator_api_key")
        ),
        "orchestrator_temperature": (
            _get_env_float("VIBELINT_ORCHESTRATOR_LLM_TEMPERATURE")
            or _get_env_float("ORCHESTRATOR_LLM_TEMPERATURE")
            or llm_dict.get("orchestrator_temperature", DEFAULT_ORCHESTRATOR_TEMPERATURE)
        ),
        "orchestrator_max_tokens": (
            _get_env_int("VIBELINT_ORCHESTRATOR_LLM_MAX_TOKENS")
            or _get_env_int("ORCHESTRATOR_LLM_MAX_TOKENS")
            or llm_dict.get("orchestrator_max_tokens", DEFAULT_ORCHESTRATOR_MAX_TOKENS)
        ),
        "orchestrator_max_context_tokens": llm_dict.get("orchestrator_max_context_tokens"),
        # Routing configuration
        "context_threshold": (
            _get_env_int("VIBELINT_LLM_CONTEXT_THRESHOLD")
            or _get_env_int("LLM_CONTEXT_THRESHOLD")
            or llm_dict.get("context_threshold", DEFAULT_CONTEXT_THRESHOLD)
        ),
        "enable_context_probing": llm_dict.get("enable_context_probing", True),
        "enable_fallback": llm_dict.get("enable_fallback", False),
    }

    return LLMConfig(**kwargs)


def get_embedding_config(config: Optional[Config] = None) -> EmbeddingConfig:
    """
    Get typed embedding configuration for vibelint.

    Args:
        config: Optional Config object. If None, loads from current directory.

    Returns:
        Validated EmbeddingConfig object
    """
    if config is None:
        config = load_config(Path.cwd())

    embedding_dict = config.get("embeddings", {})
    if not isinstance(embedding_dict, dict):
        embedding_dict = {}

    kwargs = {
        "code_api_url": embedding_dict.get("code_api_url"),
        "natural_api_url": embedding_dict.get("natural_api_url"),
        "code_model": embedding_dict.get("code_model", "text-embedding-ada-002"),
        "natural_model": embedding_dict.get("natural_model", "text-embedding-ada-002"),
        "use_specialized_embeddings": embedding_dict.get("use_specialized_embeddings", True),
    }

    return EmbeddingConfig(**kwargs)


__all__ = ["Config", "load_config", "LLMConfig", "EmbeddingConfig", "get_llm_config", "get_embedding_config"]
```

---
### File: src/vibelint/discovery.py

```python
"""
Discovers files using pathlib glob/rglob based on include patterns from
pyproject.toml, respecting the pattern's implied scope, then filters
using exclude patterns.

If `include_globs` is missing from the configuration:
- If `default_includes_if_missing` is provided, uses those patterns and logs a warning.
- Otherwise, logs an error and returns an empty list.

Exclusions from `config.exclude_globs` are always applied. Explicitly
provided paths are also excluded.

Warns if files within common VCS directories (.git, .hg, .svn) are found
and not covered by exclude_globs.

vibelint/src/vibelint/discovery.py
"""

import fnmatch
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from vibelint.config import Config
from vibelint.filesystem import get_relative_path

__all__ = ["discover_files", "discover_files_from_paths"]
logger = logging.getLogger(__name__)

_VCS_DIRS = {".git", ".hg", ".svn"}

# Default exclude patterns to avoid __pycache__, .git, etc. when using custom paths
_DEFAULT_EXCLUDE_PATTERNS = [
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    ".git/**",
    ".hg/**",
    ".svn/**",
    ".pytest_cache/**",
    ".coverage",
    ".mypy_cache/**",
    ".tox/**",
    "venv/**",
    ".venv/**",
    "env/**",
    ".env/**",
    "node_modules/**",
    ".DS_Store",
    "*.egg-info/**",
]


def _is_excluded(
    path_abs: Path,
    project_root: Path,
    exclude_globs: list[str],
    explicit_exclude_paths: set[Path],
    is_checking_directory_for_prune: bool = False,
) -> bool:
    """
    Checks if a discovered path (file or directory) should be excluded.

    For files: checks explicit paths first, then exclude globs.
    For directories (for pruning): checks if the directory itself matches an exclude glob.

    Args:
    path_abs: The absolute path of the file or directory to check.
    project_root: The absolute path of the project root.
    exclude_globs: List of glob patterns for exclusion from config.
    explicit_exclude_paths: Set of absolute paths to exclude explicitly (applies to files).
    is_checking_directory_for_prune: True if checking a directory for os.walk pruning.

    Returns:
    True if the path should be excluded/pruned, False otherwise.

    vibelint/src/vibelint/discovery.py
    """

    if not is_checking_directory_for_prune and path_abs in explicit_exclude_paths:
        logger.debug(f"Excluding explicitly provided path: {path_abs}")
        return True

    try:
        # Use resolve() for consistent comparison base
        rel_path = path_abs.resolve().relative_to(project_root.resolve())
        # Normalize for fnmatch and consistent comparisons
        rel_path_str = str(rel_path).replace("\\", "/")
    except ValueError:
        # Path is outside project root, consider it excluded for safety
        logger.warning(f"Path {path_abs} is outside project root {project_root}. Excluding.")
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Error getting relative path for exclusion check on {path_abs}: {e}")
        return True  # Exclude if relative path fails

    for pattern in exclude_globs:
        normalized_pattern = pattern.replace("\\", "/")

        if is_checking_directory_for_prune:
            # Logic for pruning directories:
            # 1. Exact match: pattern "foo", rel_path_str "foo" (dir name)
            if fnmatch.fnmatch(rel_path_str, normalized_pattern):
                logger.debug(
                    f"Pruning dir '{rel_path_str}' due to direct match with exclude pattern '{pattern}'"
                )
                return True
            # 2. Dir pattern like "foo/" or "foo/**":
            #    pattern "build/", rel_path_str "build" -> match
            #    pattern "build/**", rel_path_str "build" -> match
            if normalized_pattern.endswith("/"):
                if rel_path_str == normalized_pattern[:-1]:  # pattern "build/", rel_path "build"
                    logger.debug(
                        f"Pruning dir '{rel_path_str}' due to match with dir pattern '{pattern}'"
                    )
                    return True
            elif normalized_pattern.endswith("/**"):
                if (
                    rel_path_str == normalized_pattern[:-3]
                ):  # e.g. pattern 'dir/**', rel_path_str 'dir'
                    logger.debug(
                        f"Pruning dir '{rel_path_str}' due to match with dir/** pattern '{pattern}'"
                    )
                    return True
        else:
            # Logic for excluding files:
            # Rule 1: File path matches the glob pattern directly
            # This handles patterns like "*.pyc", "temp/*", "specific_file.txt"
            if fnmatch.fnmatch(rel_path_str, normalized_pattern):
                logger.debug(f"Excluding file '{rel_path_str}' due to exclude pattern '{pattern}'")
                return True

            # Rule 2: File is within a directory excluded by a pattern ending with '/' or '/**'
            # e.g., exclude_glob is "build/", file is "build/lib/module.py"
            # e.g., exclude_glob is "output/**", file is "output/data/log.txt"
            if normalized_pattern.endswith("/"):  # Pattern "build/"
                if rel_path_str.startswith(normalized_pattern):
                    logger.debug(
                        f"Excluding file '{rel_path_str}' because it's in excluded dir prefix '{normalized_pattern}'"
                    )
                    return True
            elif normalized_pattern.endswith("/**"):  # Pattern "build/**"
                # For "build/**", we want to match files starting with "build/"
                base_dir_pattern = normalized_pattern[:-2]  # Results in "build/"
                if rel_path_str.startswith(base_dir_pattern):
                    logger.debug(
                        f"Excluding file '{rel_path_str}' because it's in excluded dir prefix '{normalized_pattern}'"
                    )
                    return True
            # Note: A simple exclude pattern like "build" (without / or **) for files
            # will only match a file *named* "build" via the fnmatch rule above.
            # To exclude all contents of a directory "build", the pattern should be
            # "build/" or "build/**". The pruning logic for directories handles these
            # patterns effectively for `os.walk`.

    return False


def _recursive_glob_with_pruning(
    search_root_abs: Path,
    glob_suffix_pattern: str,  # e.g., "*.py" or "data/*.json"
    project_root: Path,
    config_exclude_globs: list[str],
    explicit_exclude_paths: set[Path],
) -> Iterator[Path]:
    """
    Recursively walks a directory, prunes excluded subdirectories, and yields files
    matching the glob_suffix_pattern that are not otherwise excluded.

    Args:
        search_root_abs: Absolute path to the directory to start the search from.
        glob_suffix_pattern: The glob pattern to match files against (relative to directories in the walk).
        project_root: Absolute path of the project root.
        config_exclude_globs: List of exclude glob patterns from config.
        explicit_exclude_paths: Set of absolute file paths to explicitly exclude.

    Yields:
        Absolute Path objects for matching files.
    """
    logger.debug(
        f"Recursive walk starting at '{search_root_abs}' for pattern '.../{glob_suffix_pattern}'"
    )
    for root_str, dir_names, file_names in os.walk(str(search_root_abs), topdown=True):
        current_dir_abs = Path(root_str)

        # Prune directories
        original_dir_count = len(dir_names)
        dir_names[:] = [
            d_name
            for d_name in dir_names
            if not _is_excluded(
                current_dir_abs / d_name,
                project_root,
                config_exclude_globs,
                explicit_exclude_paths,  # Not used for dir pruning but passed for func signature
                is_checking_directory_for_prune=True,
            )
        ]
        if len(dir_names) < original_dir_count:
            logger.debug(
                f"Pruned {original_dir_count - len(dir_names)} subdirectories under {current_dir_abs}"
            )

        # Match files in the current (potentially non-pruned) directory
        for f_name in file_names:
            file_abs = current_dir_abs / f_name

            # Path of file relative to where the glob_suffix_pattern matching should start (search_root_abs)
            try:
                rel_to_search_root = file_abs.relative_to(search_root_abs)
            except ValueError:
                # Should not happen if os.walk starts at search_root_abs and yields descendants
                logger.warning(
                    f"File {file_abs} unexpectedly not relative to search root {search_root_abs}. Skipping."
                )
                continue

            normalized_rel_to_search_root_str = str(rel_to_search_root).replace("\\", "/")

            if fnmatch.fnmatch(normalized_rel_to_search_root_str, glob_suffix_pattern):
                # File matches the include pattern's suffix.
                # Now, perform a final check against global exclude rules for this specific file.
                if not _is_excluded(
                    file_abs,
                    project_root,
                    config_exclude_globs,
                    explicit_exclude_paths,
                    is_checking_directory_for_prune=False,
                ):
                    yield file_abs.resolve()  # Yield resolved path


def discover_files(
    paths: list[Path],
    config: Config,
    default_includes_if_missing: list[str] | None = None,
    explicit_exclude_paths: set[Path] | None = None,
) -> list[Path]:
    """
    Discovers files based on include/exclude patterns from configuration.
    Uses a custom walker for recursive globs (**) to enable directory pruning.

    Args:
    paths: Initial paths (largely ignored, globs operate from project root).
    config: The vibelint configuration object (must have project_root set).
    default_includes_if_missing: Fallback include patterns if 'include_globs' is not in config.
    explicit_exclude_paths: A set of absolute file paths to explicitly exclude.

    Returns:
    A sorted list of unique absolute Path objects for the discovered files.

    Raises:
    ValueError: If config.project_root is None.
    """

    if config.project_root is None:
        raise ValueError("Cannot discover files without a project root defined in Config.")

    project_root = config.project_root.resolve()
    candidate_files: set[Path] = set()
    _explicit_excludes = {p.resolve() for p in (explicit_exclude_paths or set())}

    # Validate and process include_globs configuration
    include_globs_config = config.get("include_globs")
    include_globs_effective = []

    if include_globs_config is None:
        if default_includes_if_missing is not None:
            logger.warning(
                "Configuration key 'include_globs' missing in [tool.vibelint] section "
                f"of pyproject.toml. Using default patterns: {default_includes_if_missing}"
            )
            include_globs_effective = default_includes_if_missing
        else:
            logger.error(
                "Configuration key 'include_globs' missing. No include patterns specified."
            )
    elif not isinstance(include_globs_config, list):
        logger.error(
            f"Config error: 'include_globs' must be a list. Found {type(include_globs_config)}."
        )
    elif not include_globs_config:
        logger.warning("Config: 'include_globs' is empty. No files will be included.")
    else:
        include_globs_effective = include_globs_config

    # Early return if no valid include patterns
    if not include_globs_effective:
        return []

    normalized_includes = [p.replace("\\", "/") for p in include_globs_effective]

    exclude_globs_config = config.get("exclude_globs", [])
    if not isinstance(exclude_globs_config, list):
        logger.error(
            f"Config error: 'exclude_globs' must be a list. Found {type(exclude_globs_config)}. Ignoring."
        )
        exclude_globs_effective = []
    else:
        exclude_globs_effective = exclude_globs_config
    normalized_exclude_globs = [p.replace("\\", "/") for p in exclude_globs_effective]

    logger.debug(f"Starting file discovery from project root: {project_root}")
    logger.debug(f"Effective Include globs: {normalized_includes}")
    logger.debug(f"Exclude globs: {normalized_exclude_globs}")
    logger.debug(f"Explicit excludes: {_explicit_excludes}")

    start_time = time.time()

    for pattern in normalized_includes:
        pattern_start_time = time.time()
        logger.debug(f"Processing include pattern: '{pattern}'")

        if "**" in pattern:
            parts = pattern.split("**", 1)
            base_dir_glob_part = parts[0].rstrip("/")  # "src" or ""
            # glob_suffix is the part after '**/', e.g., "*.py" or "some_dir/*.txt"
            glob_suffix = parts[1].lstrip("/")

            current_search_root_abs = project_root
            if base_dir_glob_part:
                # Handle potential multiple directory components in base_dir_glob_part
                # e.g. pattern "src/app/**/... -> base_dir_glob_part = "src/app"
                current_search_root_abs = (project_root / base_dir_glob_part).resolve()

            if not current_search_root_abs.is_dir():
                logger.debug(
                    f"Skipping include pattern '{pattern}': base '{current_search_root_abs}' not a directory."
                )
                continue

            logger.debug(
                f"Using recursive walker for pattern '{pattern}' starting at '{current_search_root_abs}', suffix '{glob_suffix}'"
            )
            for p_found in _recursive_glob_with_pruning(
                current_search_root_abs,
                glob_suffix,
                project_root,
                normalized_exclude_globs,
                _explicit_excludes,
            ):
                # _recursive_glob_with_pruning already yields resolved, filtered paths
                if p_found.is_file():  # Final check, though walker should only yield files
                    candidate_files.add(p_found)  # p_found is already resolved
        else:
            # Non-recursive glob (no "**")
            logger.debug(f"Using Path.glob for non-recursive pattern: '{pattern}'")
            try:
                for p in project_root.glob(pattern):
                    abs_p = p.resolve()
                    if p.is_symlink():
                        logger.debug(f"    -> Skipping discovered symlink: {p}")
                        continue
                    if p.is_file():
                        if not _is_excluded(
                            abs_p,
                            project_root,
                            normalized_exclude_globs,
                            _explicit_excludes,
                            False,
                        ):
                            candidate_files.add(abs_p)
            except PermissionError as e:
                logger.warning(
                    f"Permission denied for non-recursive glob '{pattern}': {e}. Skipping."
                )
            except (OSError, ValueError) as e:
                logger.error(f"Error during non-recursive glob '{pattern}': {e}", exc_info=True)

        pattern_time = time.time() - pattern_start_time
        logger.debug(f"Pattern '{pattern}' processing took {pattern_time:.4f} seconds.")

    discovery_time = time.time() - start_time
    logger.debug(
        f"Globbing and initial filtering finished in {discovery_time:.4f} seconds. Total candidates: {len(candidate_files)}"
    )

    final_files_set = candidate_files

    # VCS Warning Logic
    vcs_warnings: set[Path] = set()
    if final_files_set:
        for file_path in final_files_set:
            try:
                if any(part in _VCS_DIRS for part in file_path.relative_to(project_root).parts):
                    is_actually_excluded_by_vcs_pattern = False
                    for vcs_dir_name in _VCS_DIRS:
                        if _is_excluded(
                            file_path,
                            project_root,
                            [f"{vcs_dir_name}/", f"{vcs_dir_name}/**"],
                            set(),
                            False,
                        ):
                            is_actually_excluded_by_vcs_pattern = True
                            break
                    if not is_actually_excluded_by_vcs_pattern:
                        vcs_warnings.add(file_path)
            except ValueError:
                pass
            except (OSError, TypeError) as e_vcs:
                logger.debug(f"Error during VCS check for {file_path}: {e_vcs}")

    if vcs_warnings:
        logger.warning(
            f"Found {len(vcs_warnings)} included files within potential VCS directories "
            f"({', '.join(_VCS_DIRS)}). Consider adding patterns like '.git/**' to 'exclude_globs' "
            "in your [tool.vibelint] section if this was unintended."
        )
        try:
            paths_to_log = [
                get_relative_path(p, project_root) for p in sorted(list(vcs_warnings), key=str)[:5]
            ]
            for rel_path_warn in paths_to_log:
                logger.warning(f"  - {rel_path_warn}")
            if len(vcs_warnings) > 5:
                logger.warning(f"  - ... and {len(vcs_warnings) - 5} more.")
        except Exception as e_log:
            logger.warning(f"  (Error logging VCS warning example paths: {e_log})")

    final_count = len(final_files_set)
    if final_count == 0 and include_globs_effective:
        logger.warning("No files found matching include_globs patterns or all were excluded.")

    logger.debug(f"Discovery complete. Returning {final_count} files.")
    return sorted(list(final_files_set))


def discover_files_from_paths(
    custom_paths: list[Path],
    config: Config,
    explicit_exclude_paths: set[Path] | None = None,
) -> list[Path]:
    """
    Discover files from explicitly provided paths (include_globs override).

    This function handles user-provided paths as an override to the configured
    include_globs, while still respecting exclude_globs and sensible defaults
    to avoid processing __pycache__, .git, etc.

    Args:
        custom_paths: List of file or directory paths (include_globs override)
        config: The vibelint configuration object
        explicit_exclude_paths: Additional paths to explicitly exclude

    Returns:
        A sorted list of unique absolute Path objects for Python files
    """
    if config.project_root is None:
        raise ValueError("Cannot discover files without a project root defined in Config.")

    project_root = config.project_root.resolve()
    candidate_files: set[Path] = set()
    _explicit_excludes = {p.resolve() for p in (explicit_exclude_paths or set())}

    # Combine config exclude patterns with defaults
    config_exclude_globs = config.get("exclude_globs", [])
    if not isinstance(config_exclude_globs, list):
        config_exclude_globs = []

    # Always apply default exclude patterns to avoid __pycache__, .git, etc.
    all_exclude_patterns = _DEFAULT_EXCLUDE_PATTERNS + config_exclude_globs

    logger.info(f"Include globs override: processing {len(custom_paths)} custom path(s)")
    logger.debug(f"Using exclude patterns: {all_exclude_patterns}")

    for path in custom_paths:
        abs_path = path.resolve()

        if abs_path.is_file():
            # Single file - check if it's a Python file and not excluded
            if abs_path.suffix == ".py":
                if not _is_excluded(
                    abs_path,
                    project_root,
                    all_exclude_patterns,
                    _explicit_excludes,
                    is_checking_directory_for_prune=False,
                ):
                    candidate_files.add(abs_path)
                else:
                    logger.debug(f"Excluding file {abs_path} due to exclude patterns")

        elif abs_path.is_dir():
            # Directory - recursively find Python files while respecting exclusions
            logger.debug(f"Scanning directory: {abs_path}")

            # Use the existing recursive walker with Python file pattern
            for py_file in _recursive_glob_with_pruning(
                abs_path,
                "*.py",  # Only Python files
                project_root,
                all_exclude_patterns,
                _explicit_excludes,
            ):
                candidate_files.add(py_file)
        else:
            logger.warning(f"Path does not exist or is not a file/directory: {abs_path}")

    sorted_files = sorted(candidate_files)
    logger.info(f"Include globs override result: discovered {len(sorted_files)} Python files")
    return sorted_files
```

---
### File: src/vibelint/embedding_client.py

```python
"""
Embedding Client for Specialized Code and Natural Language Embeddings.

This module provides a unified interface for accessing both local and remote
embedding models, with specialized endpoints for code analysis and natural
language processing.

Usage:
    from vibelint.embedding_client import EmbeddingClient

    client = EmbeddingClient()

    # Code embeddings (optimized for code similarity, patterns, architecture)
    code_embeddings = client.get_code_embeddings(["def function():", "class MyClass:"])

    # Natural language embeddings (optimized for documentation, comments)
    natural_embeddings = client.get_natural_embeddings(["docstring text", "comment text"])

vibelint/src/vibelint/embedding_client.py
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Unified embedding client supporting both specialized remote endpoints
    and local fallback models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding client with configuration.

        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config or {}
        self._load_configuration()
        self._initialize_clients()

    def _load_configuration(self):
        """Load configuration from various sources."""
        from .llm.llm_config import get_embedding_config

        # Use typed configuration with explicit validation
        embedding_config = get_embedding_config()

        self.code_api_url = embedding_config.code_api_url
        self.natural_api_url = embedding_config.natural_api_url
        self.code_model = embedding_config.code_model
        self.natural_model = embedding_config.natural_model
        self.use_specialized = embedding_config.use_specialized_embeddings
        self.similarity_threshold = embedding_config.get("similarity_threshold", 0.85)

        # Load API keys from environment
        self.code_api_key = os.getenv("CODE_EMBEDDING_API_KEY")
        self.natural_api_key = os.getenv("NATURAL_EMBEDDING_API_KEY")

        # Local model fallback
        self.local_model_name = embedding_config.get("local_model", "google/embeddinggemma-300m")

    def _initialize_clients(self):
        """Initialize embedding clients."""
        self._local_model = None

        # Check if we can use specialized endpoints
        # Allow endpoints without API keys for internal services
        self._can_use_code_api = bool(self.code_api_url and self.use_specialized)
        self._can_use_natural_api = bool(self.natural_api_url and self.use_specialized)

        # Initialize local model as fallback
        if not (self._can_use_code_api and self._can_use_natural_api):
            self._initialize_local_model()

        logger.info("Embedding client initialized:")
        logger.info(f"  Code API: {'✓' if self._can_use_code_api else '✗'}")
        logger.info(f"  Natural API: {'✓' if self._can_use_natural_api else '✗'}")
        logger.info(f"  Local fallback: {'✓' if self._local_model else '✗'}")

    def _initialize_local_model(self):
        """Initialize local embedding model as fallback."""
        try:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self.local_model_name)
                logger.info(f"Local embedding model loaded: {self.local_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, remote endpoints required")
        except Exception as e:
            logger.warning(f"Failed to load local model {self.local_model_name}: {e}")

    def _call_remote_api(
        self, api_url: str, api_key: str, model: str, texts: List[str]
    ) -> List[List[float]]:
        """
        Call remote embedding API.

        Args:
            api_url: API endpoint URL
            api_key: API authentication key
            model: Model name to use
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {"input": texts, "model": model}

        try:
            response = requests.post(
                f"{api_url}/v1/embeddings", headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            logger.debug(f"Generated {len(embeddings)} embeddings via {api_url}")
            return embeddings

        except Exception as e:
            logger.warning(f"Remote embedding API failed ({api_url}): {e}")
            raise

    def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using local model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._local_model:
            raise RuntimeError("Local embedding model not available")

        embeddings = self._local_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def get_code_embeddings(self, code_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings optimized for code analysis.

        Args:
            code_texts: List of code snippets to embed

        Returns:
            List of embedding vectors optimized for code similarity
        """
        if self._can_use_code_api:
            try:
                return self._call_remote_api(
                    self.code_api_url, self.code_api_key, self.code_model, code_texts
                )
            except Exception as e:
                logger.warning(f"Code API failed, falling back to local: {e}")

        # Fallback to local model
        return self._get_local_embeddings(code_texts)

    def get_natural_embeddings(self, natural_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings optimized for natural language analysis.

        Args:
            natural_texts: List of natural language texts to embed

        Returns:
            List of embedding vectors optimized for natural language understanding
        """
        if self._can_use_natural_api:
            try:
                return self._call_remote_api(
                    self.natural_api_url, self.natural_api_key, self.natural_model, natural_texts
                )
            except Exception as e:
                logger.warning(f"Natural API failed, falling back to local: {e}")

        # Fallback to local model
        return self._get_local_embeddings(natural_texts)

    def get_embeddings(self, texts: List[str], content_type: str = "mixed") -> List[List[float]]:
        """
        Get embeddings with automatic routing based on content type.

        Args:
            texts: List of texts to embed
            content_type: "code", "natural", or "mixed"

        Returns:
            List of embedding vectors
        """
        if content_type == "code":
            return self.get_code_embeddings(texts)
        elif content_type == "natural":
            return self.get_natural_embeddings(texts)
        else:
            # For mixed content, use natural language embeddings as default
            return self.get_natural_embeddings(texts)

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def find_similar_pairs(
        self, texts: List[str], content_type: str = "mixed", threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of texts that exceed similarity threshold.

        Args:
            texts: List of texts to analyze
            content_type: Type of content for optimal embedding selection
            threshold: Similarity threshold (uses config default if None)

        Returns:
            List of similar pairs with metadata
        """
        if threshold is None:
            threshold = self.similarity_threshold

        embeddings = self.get_embeddings(texts, content_type)
        similar_pairs = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self.compute_similarity(embeddings[i], embeddings[j])

                if similarity >= threshold:
                    similar_pairs.append(
                        {
                            "index1": i,
                            "index2": j,
                            "text1": texts[i],
                            "text2": texts[j],
                            "similarity": similarity,
                            "content_type": content_type,
                        }
                    )

        # Sort by similarity descending
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_pairs
```

---
### File: src/vibelint/filesystem.py

```python
"""
Filesystem and path utility functions for vibelint.

vibelint/src/vibelint/fs.py
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_directory",
    "find_files_by_extension",
    "find_package_root",
    "find_project_root",
    "get_import_path",
    "get_module_name",
    "get_relative_path",
    "is_binary",
    "is_python_file",
    "read_file_safe",
    "walk_up_for_config",
    "walk_up_for_project_root",
    "write_file_safe",
]


def walk_up_for_project_root(start_path: Path) -> Path | None:
    """
    Walk up directory tree to find project root markers.

    Project root markers (in order of precedence):
    1. .git directory (definitive project boundary)
    2. pyproject.toml file (Python project config)
    3. dev.pyproject.toml file (development/parent project config)

    Args:
        start_path: Path to start walking up from

    Returns:
        Path to project root, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    while True:
        # Check for git repo (strongest indicator of project root)
        if (current_path / ".git").is_dir():
            return current_path
        # Check for standard Python project config
        if (current_path / "pyproject.toml").is_file():
            return current_path
        # Check for development/parent project config (e.g., kaia's dev.pyproject.toml)
        if (current_path / "dev.pyproject.toml").is_file():
            return current_path
        # Stop at filesystem root
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def walk_up_for_config(start_path: Path) -> Path | None:
    """
    Walk up directory tree to find vibelint configuration.

    Searches for configuration files in this order:
    1. pyproject.toml with [tool.vibelint] section
    2. dev.pyproject.toml with [tool.vibelint] section
    3. .git directory (fallback to git repo root)

    Args:
        start_path: Path to start walking up from

    Returns:
        Path containing viable configuration, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        # Check for standard pyproject.toml with vibelint config
        pyproject_path = current_path / "pyproject.toml"
        if pyproject_path.is_file():
            if _has_vibelint_config(pyproject_path):
                return current_path

        # Check for dev.pyproject.toml with vibelint config (kaia pattern)
        dev_pyproject_path = current_path / "dev.pyproject.toml"
        if dev_pyproject_path.is_file():
            if _has_vibelint_config(dev_pyproject_path):
                return current_path

        # Fallback to git repo root
        if (current_path / ".git").is_dir():
            return current_path

        # Stop at filesystem root
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent


def _has_vibelint_config(toml_path: Path) -> bool:
    """Check if a TOML file contains vibelint configuration."""
    try:
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            return "tool" in data and "vibelint" in data.get("tool", {})
    except Exception:
        return False


# Backward compatibility alias
find_project_root = walk_up_for_project_root


def find_package_root(start_path: Path) -> Path | None:
    """
    Find the root directory of a Python package containing the given path.

    A package root is identified by containing either:
    1. A pyproject.toml file
    2. A setup.py file
    3. An __init__.py file at the top level with no parent

    Args:
        start_path: Path to start the search from

    Returns:
        Path to package root, or None if not found

    vibelint/src/vibelint/fs.py
    """
    current_path = start_path.resolve()
    if current_path.is_file():
        current_path = current_path.parent

    while True:
        if (current_path / "__init__.py").is_file():
            project_root_marker = walk_up_for_project_root(current_path)
            if project_root_marker and current_path.is_relative_to(project_root_marker):
                pass

        if (current_path / "pyproject.toml").is_file() or (current_path / ".git").is_dir():
            src_dir = current_path / "src"
            if src_dir.is_dir():
                if start_path.resolve().is_relative_to(src_dir):
                    for item in src_dir.iterdir():
                        if item.is_dir() and (item / "__init__.py").is_file():
                            return item
                    return src_dir
                else:
                    if (current_path / "__init__.py").is_file():
                        return current_path

            if (current_path / "__init__.py").is_file():
                return current_path
            return current_path

        if current_path.parent == current_path:
            return start_path.parent if start_path.is_file() else start_path

        current_path = current_path.parent


def is_python_file(path: Path) -> bool:
    """
    Check if a path represents a Python file.

    Args:
        path: Path to check

    Returns:
        True if the path is a Python file, False otherwise

    vibelint/src/vibelint/fs.py
    """
    return path.is_file() and path.suffix == ".py"


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Safely compute a relative path, falling back to the original path.

    vibelint/src/vibelint/fs.py
    """
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError as e:
        logger.debug(f"Path {path} is not relative to {base}: {e}")
        return path.resolve()


def get_import_path(file_path: Path, package_root: Path | None = None) -> str:
    """
    Get the import path for a Python file.

    Args:
        file_path: Path to the Python file
        package_root: Optional path to the package root

    Returns:
        Import path (e.g., "vibelint.utils")

    vibelint/src/vibelint/fs.py
    """
    if package_root is None:
        package_root = find_package_root(file_path)

    if package_root is None:
        return file_path.stem

    try:
        rel_path = file_path.relative_to(package_root)
        import_path = str(rel_path).replace(os.sep, ".").replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]
        return import_path
    except ValueError as e:
        logger.debug(f"Could not determine import path for {file_path}: {e}")
        return file_path.stem


def get_module_name(file_path: Path) -> str:
    """
    Extract module name from a Python file path.

    Args:
        file_path: Path to a Python file

    Returns:
        Module name

    vibelint/src/vibelint/fs.py
    """
    return file_path.stem


def find_files_by_extension(
    root_path: Path,
    extension: str = ".py",
    exclude_globs: list[str] = [],
    include_vcs_hooks: bool = False,
) -> list[Path]:
    """
    Find all files with a specific extension in a directory and its subdirectories.

    Args:
        root_path: Root path to search in
        extension: File extension to look for (including the dot)
        exclude_globs: Glob patterns to exclude
        include_vcs_hooks: Whether to include version control directories

    Returns:
        List of paths to files with the specified extension

    vibelint/src/vibelint/fs.py
    """
    if exclude_globs is None:
        exclude_globs = []

    result = []

    for file_path in root_path.glob(f"**/*{extension}"):
        if not include_vcs_hooks:
            if any(
                part.startswith(".") and part in {".git", ".hg", ".svn"} for part in file_path.parts
            ):
                continue

        if any(fnmatch.fnmatch(str(file_path), pattern) for pattern in exclude_globs):
            continue

        result.append(file_path)

    return result


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path to the directory

    vibelint/src/vibelint/fs.py
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file_safe(file_path: Path, encoding: str = "utf-8") -> str | None:
    """
    Safely read a file, returning None if any errors occur.

    Args:
        file_path: Path to file
        encoding: File encoding

    Returns:
        File contents or None if error

    vibelint/src/vibelint/fs.py
    """
    try:
        return file_path.read_text(encoding=encoding)
    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not read file {file_path}: {e}")
        return None


def write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> bool:
    """
    Safely write content to a file, returning success status.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding

    Returns:
        True if successful, False otherwise

    vibelint/src/vibelint/fs.py
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
        return True
    except (OSError, UnicodeEncodeError) as e:
        logger.debug(f"Could not write file {file_path}: {e}")
        return False


def is_binary(file_path: Path, chunk_size: int = 1024) -> bool:
    """
    Check if a file appears to be binary by looking for null bytes
    or a high proportion of non-text bytes in the first chunk.

    Args:
        file_path: The path to the file.
        chunk_size: The number of bytes to read from the beginning.

    Returns:
        True if the file seems binary, False otherwise.

    vibelint/src/vibelint/fs.py
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
        if not chunk:
            return False

        if b"\x00" in chunk:
            return True

        text_characters = bytes(range(32, 127)) + b"\n\r\t\f\b"
        non_text_count = sum(1 for byte in chunk if bytes([byte]) not in text_characters)

        if len(chunk) > 0 and (non_text_count / len(chunk)) > 0.3:
            return True

        return False
    except OSError:
        return True
    except (TypeError, AttributeError) as e:
        logger.debug(f"Error checking if {file_path} is binary: {e}")
        return True
```

---
### File: src/vibelint/fix.py

```python
"""
Automatic fix functionality for vibelint using deterministic fixes and LLM for docstring generation only.

vibelint/src/vibelint/fix.py
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from vibelint.config import Config
from vibelint.validators import Finding

__all__ = ["FixEngine", "can_fix_finding", "apply_fixes", "regenerate_all_docstrings"]

logger = logging.getLogger(__name__)


class FixEngine:
    """Engine for automatically fixing vibelint issues with deterministic code changes."""

    def __init__(self, config: Config):
        """Initialize fix engine with configuration.

        vibelint/src/vibelint/fix.py
        """
        self.config = config

        # Initialize LLM manager for dual LLM support
        from vibelint.llm_client import create_llm_manager

        config_dict = config.settings if isinstance(config.settings, dict) else {}
        self.llm_manager = create_llm_manager(config_dict)

    def can_fix_finding(self, finding: Finding) -> bool:
        """Check if a finding can be automatically fixed.

        vibelint/src/vibelint/fix.py
        """
        fixable_rules = {
            "DOCSTRING-MISSING",
            "DOCSTRING-PATH-REFERENCE",
            "EXPORTS-MISSING-ALL",
        }
        return finding.rule_id in fixable_rules

    async def fix_file(self, file_path: Path, findings: list[Finding]) -> bool:
        """Fix all fixable issues in a file deterministically.

        Returns True if any fixes were applied.

        vibelint/src/vibelint/fix.py
        """
        fixable_findings = [f for f in findings if self.can_fix_finding(f)]
        if not fixable_findings:
            return False

        logger.info(f"Fixing {len(fixable_findings)} issues in {file_path}")

        # Read current file content
        try:
            original_content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return False

        # Apply deterministic fixes
        fixed_content = await self._apply_deterministic_fixes(
            file_path, original_content, fixable_findings
        )

        if fixed_content and fixed_content != original_content:
            try:
                # Write fixed content back to file
                file_path.write_text(fixed_content, encoding="utf-8")
                logger.info(f"Applied fixes to {file_path}")
                return True
            except Exception as e:
                logger.error(f"Could not write fixes to {file_path}: {e}")
                return False

        return False

    async def _apply_deterministic_fixes(
        self, file_path: Path, content: str, findings: list[Finding]
    ) -> str:
        """Apply deterministic fixes without LLM file rewriting.

        vibelint/src/vibelint/fix.py
        """
        try:
            # Parse the AST to understand the code structure
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Cannot parse {file_path}: {e}")
            return content

        # Track modifications by line number
        lines = content.splitlines()
        modifications = {}

        # Group findings by type for efficient processing
        findings_by_type = {}
        for finding in findings:
            rule_id = finding.rule_id
            if rule_id not in findings_by_type:
                findings_by_type[rule_id] = []
            findings_by_type[rule_id].append(finding)

        # Apply fixes by type
        if "DOCSTRING-MISSING" in findings_by_type:
            await self._fix_missing_docstrings(
                tree, lines, modifications, findings_by_type["DOCSTRING-MISSING"], file_path
            )

        if "DOCSTRING-PATH-REFERENCE" in findings_by_type:
            self._fix_docstring_path_references(
                lines, modifications, findings_by_type["DOCSTRING-PATH-REFERENCE"], file_path
            )

        if "EXPORTS-MISSING-ALL" in findings_by_type:
            self._fix_missing_exports(
                tree, lines, modifications, findings_by_type["EXPORTS-MISSING-ALL"]
            )

        # Apply all modifications to create fixed content
        return self._apply_modifications(lines, modifications)

    async def _fix_missing_docstrings(
        self,
        tree: ast.AST,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
        file_path: Path,
    ) -> None:
        """Add missing docstrings using LLM for content generation only."""

        # Find functions and classes that need docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Check if this node needs a docstring based on findings
                node_line = node.lineno
                needs_docstring = any(
                    abs(f.line - node_line) <= 2 for f in findings  # Allow some line tolerance
                )

                if needs_docstring and not ast.get_docstring(node):
                    # Generate docstring content using LLM (safe - only returns text)
                    docstring_content = await self._generate_docstring_content(node, file_path)

                    if docstring_content:
                        # Deterministically insert the docstring
                        indent = self._get_indent_for_line(lines, node.lineno)
                        docstring_line = (
                            f'{indent}"""{docstring_content}\n\n{indent}{file_path}\n{indent}"""'
                        )

                        # Insert after the function/class definition line
                        insert_line = node.lineno  # Insert after the def line
                        modifications[insert_line] = docstring_line

    async def _generate_docstring_content(self, node: ast.AST, file_path: Path) -> Optional[str]:
        """Generate only docstring text content using dual LLM system (safe operation)."""
        if not self.llm_manager:
            logger.debug("No LLM manager configured, skipping docstring generation")
            return None

        try:
            from vibelint.llm import LLMRequest

            # Safe prompt - only asks for docstring text, never code
            if isinstance(node, ast.ClassDef):
                prompt = f"Write a brief docstring for a Python class named '{node.name}'. Return only the docstring text without quotes or formatting."
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args] if hasattr(node, "args") else []
                prompt = f"Write a brief docstring for a Python function named '{node.name}' with parameters {args}. Return only the docstring text without quotes or formatting."
            else:
                logger.debug("Unknown node type for docstring generation")
                return None

            # Use fast LLM for quick docstring generation
            request = LLMRequest(
                content=prompt, task_type="docstring_generation", max_tokens=200, temperature=0.1
            )

            response = await self.llm_manager.process_request(request)

            if response.success and response.content:
                # Clean the response to ensure it's just text
                content = str(response.content).strip()
                # Remove any quotes or markdown that might have been added
                content = content.replace('"""', "").replace("'''", "").replace("`", "")
                return content[:200]  # Limit length

        except Exception as e:
            logger.warning(f"LLM docstring generation failed: {e}, using fallback")

        # Safe fallback
        if isinstance(node, ast.ClassDef):
            return f"{node.name} class implementation."
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"{node.name} function implementation."
        return "Implementation."

    def _fix_docstring_path_references(
        self,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
        file_path: Path,
    ) -> None:
        """Add path references to existing docstrings based on configuration."""
        # Get docstring configuration
        config_dict = self.config.settings if isinstance(self.config.settings, dict) else {}
        docstring_config = config_dict.get("docstring", {})
        require_path_references = docstring_config.get("require_path_references", False)

        # Skip fix if path references are not required
        if not require_path_references:
            return

        # Get path format configuration
        path_format = docstring_config.get("path_reference_format", "relative")
        expected_path = self._get_expected_path_for_fix(file_path, path_format)

        for finding in findings:
            line_idx = finding.line - 1  # Convert to 0-based index
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                # If this is a docstring line, ensure it has path reference
                if '"""' in line or "'''" in line:
                    # Add path reference if not already present
                    if expected_path not in line:
                        # Modify the docstring to include path
                        indent = self._get_indent_for_line(lines, finding.line)
                        if line.strip().endswith('"""') or line.strip().endswith("'''"):
                            # Single line docstring - expand it
                            quote = '"""' if '"""' in line else "'''"
                            content = line.strip().replace(quote, "").strip()
                            new_docstring = f"{indent}{quote}{content}\n\n{indent}{expected_path}\n{indent}{quote}"
                            modifications[finding.line - 1] = new_docstring

    def _get_expected_path_for_fix(self, file_path: Path, path_format: str) -> str:
        """Get expected path reference for fix based on format configuration."""
        if path_format == "absolute":
            return str(file_path)
        elif path_format == "module_path":
            # Convert to Python module path (e.g., vibelint.validators.docstring)
            parts = file_path.parts
            if "src" in parts:
                src_idx = parts.index("src")
                module_parts = parts[src_idx + 1 :]
            else:
                module_parts = parts

            # Remove .py extension and convert to module path
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

            return ".".join(module_parts)
        else:  # relative format (default)
            # Get relative path, removing project root and src/ prefix
            relative_path = str(file_path)
            try:
                # Try to find project root by looking for common markers
                current = file_path.parent
                while current.parent != current:
                    if any(
                        (current / marker).exists()
                        for marker in ["pyproject.toml", "setup.py", ".git"]
                    ):
                        relative_path = str(file_path.relative_to(current))
                        break
                    current = current.parent
            except ValueError:
                pass

            # Remove src/ prefix if present
            if relative_path.startswith("src/"):
                relative_path = relative_path[4:]

            return relative_path

    def _fix_missing_exports(
        self,
        tree: ast.AST,
        lines: List[str],
        modifications: Dict[int, str],
        findings: List[Finding],
    ) -> None:
        """Add missing __all__ exports."""
        # Find all public functions and classes
        public_names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):  # Public items
                    public_names.append(node.name)

        if public_names:
            # Check if __all__ already exists
            has_all = any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                for node in ast.walk(tree)
            )

            if not has_all:
                # Add __all__ at the top of the file after imports
                exports_line = f"__all__ = {public_names!r}"

                # Find a good place to insert (after imports)
                insert_line = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        insert_line = i + 1
                    elif line.strip() and not line.strip().startswith("#"):
                        break

                modifications[insert_line] = exports_line

    def _get_indent_for_line(self, lines: List[str], line_number: int) -> str:
        """Get the indentation for a given line number."""
        if 1 <= line_number <= len(lines):
            line = lines[line_number - 1]
            return line[: len(line) - len(line.lstrip())]
        return "    "  # Default 4-space indent

    def _apply_modifications(self, lines: List[str], modifications: Dict[int, str]) -> str:
        """Apply all modifications to the lines and return the fixed content."""
        # Sort modifications by line number in reverse order to avoid index shifting
        sorted_modifications = sorted(modifications.items(), reverse=True)

        result_lines = lines[:]
        for line_num, new_content in sorted_modifications:
            if line_num < len(result_lines):
                result_lines[line_num] = new_content
            else:
                # Insert at end
                result_lines.append(new_content)

        return "\n".join(result_lines)


# Convenience functions for the CLI
async def apply_fixes(config: Config, file_findings: dict[Path, list[Finding]]) -> int:
    """Apply fixes to all files with fixable findings.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    fixed_count = 0

    for file_path, findings in file_findings.items():
        if await engine.fix_file(file_path, findings):
            fixed_count += 1

    return fixed_count


def can_fix_finding(finding: Finding) -> bool:
    """Check if a finding can be automatically fixed.

    vibelint/src/vibelint/fix.py
    """
    return finding.rule_id in {
        "DOCSTRING-MISSING",
        "DOCSTRING-PATH-REFERENCE",
        "EXPORTS-MISSING-ALL",
    }


async def regenerate_all_docstrings(config: Config, file_paths: List[Path]) -> int:
    """Regenerate ALL docstrings in the specified files using LLM.

    Unlike apply_fixes which only adds missing docstrings, this function
    regenerates existing docstrings as well for consistency and improved quality.

    Returns the number of files successfully processed.

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    processed_count = 0

    if not engine.llm_config.get("api_base_url"):
        logger.error("No LLM API configured. Cannot regenerate docstrings.")
        return 0

    for file_path in file_paths:
        try:
            if await _regenerate_docstrings_in_file(engine, file_path):
                processed_count += 1
                logger.info(f"Regenerated docstrings in {file_path}")
            else:
                logger.debug(f"No docstrings to regenerate in {file_path}")
        except Exception as e:
            logger.error(f"Failed to regenerate docstrings in {file_path}: {e}")

    return processed_count


async def preview_docstring_changes(config: Config, file_paths: List[Path]) -> Dict[str, Any]:
    """Preview what docstring changes would be made without modifying files.

    Returns a dictionary containing:
    - files_analyzed: list of files that would be changed
    - total_changes: total number of docstring changes
    - preview_samples: dict of file -> list of preview changes

    vibelint/src/vibelint/fix.py
    """
    engine = FixEngine(config)
    preview_results = {
        "files_analyzed": [],
        "total_changes": 0,
        "preview_samples": {},
        "errors": [],
    }

    if not engine.llm_config.get("api_base_url"):
        preview_results["errors"].append("No LLM API configured. Cannot preview docstring changes.")
        return preview_results

    for file_path in file_paths:
        try:
            file_preview = await _preview_docstrings_in_file(engine, file_path)
            if file_preview["changes"]:
                preview_results["files_analyzed"].append(str(file_path))
                preview_results["total_changes"] += len(file_preview["changes"])
                preview_results["preview_samples"][str(file_path)] = file_preview["changes"]
        except Exception as e:
            preview_results["errors"].append(f"Failed to preview {file_path}: {e}")

    return preview_results


async def _preview_docstrings_in_file(engine: FixEngine, file_path: Path) -> Dict[str, Any]:
    """Preview docstring changes for a single file without modifying it.

    Returns dict with 'changes' list containing preview information.
    """
    file_preview = {"changes": []}

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Parse the file to find all functions and classes
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                change_preview = await _preview_node_docstring(engine, node, lines, file_path)
                if change_preview:
                    file_preview["changes"].append(change_preview)

    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        logger.error(f"Error previewing {file_path}: {e}")

    return file_preview


async def _preview_node_docstring(
    engine: FixEngine,
    node: ast.AST,
    lines: List[str],
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    """Preview what docstring change would be made for a specific AST node.

    Returns preview dict with change information, or None if no change.
    """
    # SAFETY CHECK: Only process functions and classes that we can safely identify
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None

    # Get current docstring if it exists
    current_docstring = ast.get_docstring(node)

    # Generate new docstring content (this calls the LLM)
    new_docstring_content = await engine._generate_docstring_content(node, file_path)
    if not new_docstring_content:
        return None

    # SAFETY VALIDATION: Check that generated content is reasonable
    if not _validate_docstring_content(new_docstring_content):
        return None

    # Determine what type of change this would be
    change_type = "add" if not current_docstring else "modify"
    node_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"

    return {
        "node_name": node.name,
        "node_type": node_type,
        "line_number": node.lineno,
        "change_type": change_type,
        "current_docstring": (
            current_docstring[:100] + "..."
            if current_docstring and len(current_docstring) > 100
            else current_docstring
        ),
        "new_docstring": (
            new_docstring_content[:100] + "..."
            if len(new_docstring_content) > 100
            else new_docstring_content
        ),
    }


async def _regenerate_docstrings_in_file(engine: FixEngine, file_path: Path) -> bool:
    """Regenerate all docstrings in a single file.

    Returns True if any docstrings were regenerated.

    vibelint/src/vibelint/fix.py
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Parse the file to find all functions and classes
        tree = ast.parse(content)
        modifications = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                await _regenerate_node_docstring(engine, node, lines, modifications, file_path)

        if modifications:
            # Apply modifications
            result_content = _apply_line_modifications(lines, modifications)
            file_path.write_text(result_content, encoding="utf-8")
            return True

        return False

    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


async def _regenerate_node_docstring(
    engine: FixEngine,
    node: ast.AST,
    lines: List[str],
    modifications: Dict[int, str],
    file_path: Path,
) -> None:
    """Regenerate docstring for a specific AST node with strict safety validation.

    SAFETY CRITICAL: This function must NEVER modify any Python code, only docstring content.

    vibelint/src/vibelint/fix.py
    """
    # SAFETY CHECK: Only process functions and classes that we can safely identify
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        logger.warning(f"Skipping unsafe node type {type(node)} for safety")
        return

    # Get current docstring if it exists
    current_docstring = ast.get_docstring(node)

    # SAFETY CHECK: Validate LLM-generated content before using it
    new_docstring_content = await engine._generate_docstring_content(node, file_path)
    if not new_docstring_content:
        return

    # SAFETY VALIDATION: Check that generated content is reasonable
    if not _validate_docstring_content(new_docstring_content):
        logger.warning(f"Generated docstring failed safety validation for {node.name}")
        return

    # Determine indentation safely
    node_line = node.lineno - 1
    if node_line < len(lines):
        indent = len(lines[node_line]) - len(lines[node_line].lstrip())
        indent_str = " " * (indent + 4)  # Add 4 spaces for function/class body
    else:
        indent_str = "    "  # Default indentation

    # Format the new docstring with path reference and safety warning
    new_docstring = (
        f'{indent_str}"""{new_docstring_content}\n\n'
        f"{indent_str}[WARNING]  CRITICAL WARNING: This docstring was auto-generated by LLM and MUST be reviewed.\n"
        f"{indent_str}Inaccurate documentation can cause security vulnerabilities, system failures,\n"
        f"{indent_str}and data corruption. Verify all parameters, return types, and behavior descriptions.\n\n"
        f'{indent_str}{file_path}\n{indent_str}"""'
    )

    if current_docstring:
        # ULTRA SAFE: Find docstring using multiple validation methods
        docstring_lines = _find_docstring_lines_safely(node, lines)

        if docstring_lines:
            # SAFETY CHECK: Verify we're only modifying docstring lines
            for line_num in docstring_lines:
                if line_num < len(lines):
                    line_content = lines[line_num]
                    # CRITICAL: Only modify lines that are clearly part of docstring
                    if not _is_safe_docstring_line(line_content):
                        logger.error(
                            f"SAFETY VIOLATION: Attempted to modify non-docstring line {line_num}: {line_content}"
                        )
                        return  # Abort entire operation for safety

            # Safe to proceed - modify only the first docstring line, remove others
            for i, line_num in enumerate(docstring_lines):
                if i == 0:
                    modifications[line_num] = new_docstring
                else:
                    modifications[line_num] = ""  # Remove continuation lines
        else:
            logger.warning(f"Could not safely locate docstring for {node.name}")
    else:
        # Add new docstring after function/class definition
        insert_line = node.lineno  # Line after def/class
        modifications[insert_line] = new_docstring


def _validate_docstring_content(content: str) -> bool:
    """Validate that LLM-generated docstring content is safe and reasonable.

    vibelint/src/vibelint/fix.py
    """
    if not content or not isinstance(content, str):
        return False

    # Check for dangerous content
    dangerous_patterns = [
        "import ",
        "exec(",
        "eval(",
        "__import__",
        "subprocess",
        "os.system",
        "shell=True",
        "DELETE",
        "DROP TABLE",
        "rm -rf",
    ]

    content_lower = content.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in content_lower:
            logger.error(f"SAFETY: Dangerous pattern '{pattern}' found in generated docstring")
            return False

    # Check reasonable length (docstrings shouldn't be huge)
    if len(content) > 2000:
        logger.warning("Generated docstring is suspiciously long")
        return False

    return True


def _find_docstring_lines_safely(node: ast.AST, lines: List[str]) -> List[int]:
    """Safely find the exact line numbers of a docstring using multiple validation methods.

    vibelint/src/vibelint/fix.py
    """
    # Method 1: Use AST to find docstring node
    docstring_node = None
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
            if isinstance(child.value.value, str):
                docstring_node = child
                break

    if not docstring_node:
        return []

    # Get line range from AST
    start_line = docstring_node.lineno - 1  # Convert to 0-based
    end_line = docstring_node.end_lineno - 1 if docstring_node.end_lineno else start_line

    # Method 2: Verify by examining actual line content
    docstring_lines = []
    for line_num in range(start_line, end_line + 1):
        if line_num < len(lines):
            if _is_safe_docstring_line(lines[line_num]):
                docstring_lines.append(line_num)
            else:
                # If any line in the range is not a docstring line, abort for safety
                logger.error(f"SAFETY: Line {line_num} in docstring range is not a docstring line")
                return []

    return docstring_lines


def _is_safe_docstring_line(line: str) -> bool:
    """Check if a line is definitely part of a docstring and safe to modify.

    vibelint/src/vibelint/fix.py
    """
    stripped = line.strip()

    # Must contain quotes or be empty/whitespace (for multi-line docstrings)
    if not stripped:
        return True  # Empty line within docstring

    # Must contain docstring quotes
    if '"""' in stripped or "'''" in stripped:
        return True

    # If it doesn't start with quotes, it might be docstring content
    # But we need to be very careful - check it doesn't look like code
    if any(
        pattern in stripped
        for pattern in ["def ", "class ", "import ", "=", "return ", "if ", "for ", "while "]
    ):
        return False

    # If it's indented and looks like text, probably docstring content
    if line.startswith("    ") and not stripped.startswith(("#", "//")):
        return True

    # Default to false for safety
    return False


def _apply_line_modifications(lines: List[str], modifications: Dict[int, str]) -> str:
    """Apply line modifications to content deterministically.

    vibelint/src/vibelint/fix.py
    """
    result_lines = lines[:]

    # Sort modifications by line number in reverse order to avoid index shifting
    sorted_modifications = sorted(modifications.items(), reverse=True)

    for line_num, new_content in sorted_modifications:
        if line_num < len(result_lines):
            if new_content == "":
                # Remove line
                result_lines.pop(line_num)
            else:
                result_lines[line_num] = new_content
        elif new_content:
            # Insert at end
            result_lines.append(new_content)

    return "\n".join(result_lines)
```

---
### File: src/vibelint/llm_client.py

```python
"""
Consolidated LLM system for vibelint.

Manages dual LLMs, tracing, and dynamic validator generation:
- Fast: High-speed inference for quick tasks
- Orchestrator: Large context for complex reasoning
- Dynamic: On-demand validator generation from prompts

vibelint/src/vibelint/llm.py
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


def _load_env_files():
    """Load environment variables from multiple possible locations."""
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path.home() / ".vibelint.env",  # User home directory
        Path(__file__).parent.parent.parent / ".env",  # Project root (fallback)
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


_load_env_files()

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_CONTEXT_THRESHOLD = 3000
DEFAULT_FAST_TEMPERATURE = 0.1
DEFAULT_FAST_MAX_TOKENS = 2048
DEFAULT_ORCHESTRATOR_TEMPERATURE = 0.2
DEFAULT_ORCHESTRATOR_MAX_TOKENS = 8192
ORCHESTRATOR_TIMEOUT_SECONDS = 600
FAST_TIMEOUT_SECONDS = 30
TOKEN_ESTIMATION_DIVISOR = 4

__all__ = [
    "LLMRole",
    "LLMManager",
    "LLMRequest",
    "LLMResponse",
    "LLMBackendConfig",
    "LogEntry",
    "APIPayload",
    "FeatureAvailability",
    "LLMStatus",
    "create_llm_manager",
]


class LLMRole(Enum):
    """LLM roles for different types of tasks."""

    FAST = "fast"  # High-speed inference, small context
    ORCHESTRATOR = "orchestrator"  # Large context, complex reasoning


@dataclass
class LLMRequest:
    """Simple request specification for LLM processing."""

    content: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None  # JSON schema for structured responses
    # If structured_output is None, expects unstructured natural language response


@dataclass
class LLMResponse:
    """Typed response from LLM processing."""

    content: str
    success: bool
    llm_used: str
    duration_seconds: float
    input_tokens: int
    reasoning_content: str = ""
    error: Optional[str] = None


@dataclass
class LLMBackendConfig:
    """Configuration for a single LLM backend."""

    backend: str
    api_url: str
    model: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    max_context_tokens: int


@dataclass
class LogEntry:
    """Log entry for LLM request/response pairs."""

    type: str
    timestamp: str
    llm_used: str
    request_content_length: int
    request_content_preview: str
    request_max_tokens: Optional[int]
    request_temperature: Optional[float]
    response_success: bool
    response_content_length: int
    response_content_preview: str
    response_duration_seconds: float
    response_error: Optional[str]


@dataclass
class APIPayload:
    """API payload for LLM requests."""

    model: str
    messages: list[Dict[str, str]]
    temperature: float
    max_tokens: int
    stream: bool
    response_format: Optional[Dict[str, Any]] = None
    grammar: Optional[str] = None


@dataclass
class FeatureAvailability:
    """Feature availability based on LLM configuration."""

    architecture_analysis: bool
    docstring_generation: bool
    code_smell_detection: bool
    coverage_assessment: bool
    llm_validation: bool
    semantic_similarity: bool
    embedding_clustering: bool
    duplicate_detection: bool


@dataclass
class LLMStatus:
    """Status of LLM manager."""

    fast_configured: bool
    orchestrator_configured: bool
    context_threshold: int
    fallback_enabled: bool
    available_features: FeatureAvailability


class LLMManager:
    """Simple manager for dual LLM setup."""

    def __init__(self, config: Optional["LLMConfig"] = None):
        """Initialize with vibelint configuration.

        Args:
            config: Optional typed LLMConfig - if None, loads from config files
        """
        from vibelint.config import get_llm_config, LLMConfig

        # Load typed configuration
        self.llm_config = config if config is not None else get_llm_config()

        # Build configs for fast and orchestrator using typed config
        self.fast_config = self._build_fast_config(self.llm_config)
        self.orchestrator_config = self._build_orchestrator_config(self.llm_config)

        # Routing configuration from typed config
        self.context_threshold = self.llm_config.context_threshold
        self.enable_fallback = self.llm_config.enable_fallback

        # Session for HTTP requests
        self.session = requests.Session()
        # Note: timeout is set per-request rather than on session

        # Optional logging callback for external logging (e.g., JSONL workflow logs)
        self.log_callback = None

    def set_log_callback(self, callback):
        """Set a callback function for logging LLM requests/responses.

        Callback signature: callback(log_entry: LogEntry) -> None
        """
        self.log_callback = callback

    def _log_request_response(self, request: LLMRequest, response: LLMResponse, llm_used: str):
        """Log request/response pair if callback is registered."""
        if self.log_callback:
            try:
                log_entry = LogEntry(
                    type="llm_call",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    llm_used=llm_used,
                    request_content_length=len(request.content),
                    request_content_preview=request.content[:500] + "..." if len(request.content) > 500 else request.content,
                    request_max_tokens=request.max_tokens,
                    request_temperature=request.temperature,
                    response_success=response.success,
                    response_content_length=len(response.content),
                    response_content_preview=response.content[:500] + "..." if len(response.content) > 500 else response.content,
                    response_duration_seconds=response.duration_seconds,
                    response_error=response.error
                )
                self.log_callback(log_entry)
            except Exception as e:
                logger.debug(f"Log callback failed: {e}")

    def _build_fast_config(self, llm_config) -> LLMBackendConfig:
        """Build configuration for fast LLM from typed config."""
        return LLMBackendConfig(
            backend=llm_config.fast_backend,
            api_url=llm_config.fast_api_url,
            model=llm_config.fast_model,
            api_key=llm_config.fast_api_key,
            temperature=llm_config.fast_temperature,
            max_tokens=llm_config.fast_max_tokens,
            max_context_tokens=llm_config.fast_max_context_tokens,
        )

    def _build_orchestrator_config(self, llm_config) -> LLMBackendConfig:
        """Build configuration for orchestrator LLM from typed config."""
        return LLMBackendConfig(
            backend=llm_config.orchestrator_backend,
            api_url=llm_config.orchestrator_api_url,
            model=llm_config.orchestrator_model,
            api_key=llm_config.orchestrator_api_key,
            temperature=llm_config.orchestrator_temperature,
            max_tokens=llm_config.orchestrator_max_tokens,
            max_context_tokens=llm_config.orchestrator_max_context_tokens,
        )

    def select_llm(self, request: LLMRequest) -> LLMRole:
        """Select appropriate LLM based on hard constraints.

        Routes based on actual LLM hard limits:
        - Context window size (input tokens)
        - Max output tokens
        - Use cheapest/fastest LLM that can handle the request
        """
        content_size = len(request.content)
        max_tokens = request.max_tokens or 50

        # Get LLM hard limits from typed config
        fast_max_tokens = self.llm_config.fast_max_tokens
        fast_max_context = self.llm_config.fast_max_context_tokens or 1000  # Default if not specified

        orchestrator_max_tokens = self.llm_config.orchestrator_max_tokens
        # Orchestrator typically has much larger context window

        fast_available = bool(self.fast_config.api_url)
        orchestrator_available = bool(self.orchestrator_config.api_url)

        # Estimate input tokens (conservative: 3 chars per token)
        estimated_input_tokens = content_size // 3

        # Hard constraint: If request exceeds fast LLM's output token limit
        if max_tokens > fast_max_tokens:
            if orchestrator_available:
                logger.debug(f"Routing to orchestrator: output tokens ({max_tokens}) > fast limit ({fast_max_tokens})")
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(f"Request needs {max_tokens} tokens but orchestrator unavailable, truncating to fast LLM limit")
                return LLMRole.FAST

        # Hard constraint: If input exceeds fast LLM's context window
        if estimated_input_tokens > fast_max_context:
            if orchestrator_available:
                logger.debug(f"Routing to orchestrator: input tokens (~{estimated_input_tokens}) > fast context ({fast_max_context})")
                return LLMRole.ORCHESTRATOR
            else:
                logger.warning(f"Large input (~{estimated_input_tokens} tokens) but orchestrator unavailable, truncating to fast LLM")
                return LLMRole.FAST

        # No hard constraints violated - use fast LLM (cheaper/faster)
        if fast_available:
            logger.debug(f"Routing to fast: within limits (input~{estimated_input_tokens}, output={max_tokens})")
            return LLMRole.FAST
        elif orchestrator_available:
            logger.debug(f"Routing to orchestrator: fast LLM unavailable")
            return LLMRole.ORCHESTRATOR
        else:
            raise ValueError("No LLMs configured - need either fast_api_url or orchestrator_api_url in config")

    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process request using intelligent routing with fallback."""
        # Check if at least one LLM is configured
        fast_available = bool(self.llm_config.fast_api_url)
        orchestrator_available = bool(self.llm_config.orchestrator_api_url)

        if not fast_available and not orchestrator_available:
            raise ValueError("No LLM configured. Add fast_api_url or orchestrator_api_url to pyproject.toml")

        # Handle oversized content with truncation warning
        if len(request.content) > 50000:  # ~50k chars for huge log files
            logger.warning(f"Content too large ({len(request.content)} chars), truncating to 50k chars")
            request.content = request.content[:50000] + "\n[...content truncated...]"

        # Use intelligent routing to select appropriate LLM
        selected_llm = self.select_llm(request)

        # Try primary LLM based on routing decision
        primary_failed = False
        if selected_llm == LLMRole.FAST and fast_available:
            try:
                logger.debug("Attempting fast LLM (selected by routing)")
                result = await self._call_fast_llm(request)
                if result.content and result.content.strip():
                    self._log_request_response(request, result, "fast")
                    return result
                else:
                    logger.warning("Fast LLM returned empty content")
                    primary_failed = True
            except Exception as e:
                logger.warning(f"Fast LLM failed: {e}")
                primary_failed = True
        elif selected_llm == LLMRole.ORCHESTRATOR and orchestrator_available:
            try:
                logger.debug("Attempting orchestrator LLM (selected by routing)")
                # Ensure minimum tokens for orchestrator
                orchestrator_request = LLMRequest(
                    content=request.content,
                    max_tokens=max(request.max_tokens or 0, 1000),
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                    structured_output=request.structured_output
                )
                result = await self._call_orchestrator_llm(orchestrator_request)
                if result.content and result.content.strip():
                    self._log_request_response(request, result, "orchestrator")
                    return result
                else:
                    logger.warning("Orchestrator LLM returned empty content")
                    primary_failed = True
            except Exception as e:
                logger.warning(f"Orchestrator LLM failed: {e}")
                primary_failed = True

        # Fallback to other LLM if primary failed
        if primary_failed:
            if selected_llm == LLMRole.FAST and orchestrator_available:
                try:
                    logger.info("Falling back to orchestrator LLM")
                    orchestrator_request = LLMRequest(
                        content=request.content,
                        max_tokens=max(request.max_tokens or 0, 1000),
                        temperature=request.temperature,
                        system_prompt=request.system_prompt,
                        structured_output=request.structured_output
                    )
                    result = await self._call_orchestrator_llm(orchestrator_request)
                    if result.content and result.content.strip():
                        self._log_request_response(request, result, "orchestrator_fallback")
                        return result
                except Exception as e:
                    logger.warning(f"Orchestrator fallback failed: {e}")
            elif selected_llm == LLMRole.ORCHESTRATOR and fast_available:
                try:
                    logger.info("Falling back to fast LLM")
                    result = await self._call_fast_llm(request)
                    if result.content and result.content.strip():
                        self._log_request_response(request, result, "fast_fallback")
                        return result
                except Exception as e:
                    logger.warning(f"Fast LLM fallback failed: {e}")

        # All attempts failed - return graceful failure
        return LLMResponse(
            content="[LLM analysis unavailable: All configured LLMs failed or returned empty content]",
            llm_used="none",
            duration_seconds=0,
            input_tokens=0,
            success=False,
            error="All LLM attempts failed or returned empty content"
        )

    async def _call_fast_llm(self, request: LLMRequest) -> LLMResponse:
        """Call the fast LLM."""
        return await self._make_api_call(self.fast_config, request, LLMRole.FAST)

    async def _call_orchestrator_llm(self, request: LLMRequest) -> LLMResponse:
        """Call the orchestrator (large context) LLM."""
        return await self._make_api_call(self.orchestrator_config, request, LLMRole.ORCHESTRATOR)

    async def _make_api_call(
        self, llm_config: LLMBackendConfig, request: LLMRequest, role: LLMRole
    ) -> LLMResponse:
        """Make API call to specified LLM."""
        start_time = time.time()

        if not llm_config.api_url:
            raise ValueError(f"No API URL configured for {role.value} LLM")

        url = f"{llm_config.api_url.rstrip('/')}/v1/chat/completions"

        headers = {}
        if llm_config.api_key:
            headers["Authorization"] = f"Bearer {llm_config.api_key}"

        # Build messages with optional system prompt
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.content})

        payload = APIPayload(
            model=llm_config.model,
            messages=messages,
            temperature=request.temperature or llm_config.temperature,
            max_tokens=request.max_tokens or llm_config.max_tokens,
            stream=False,
        )

        # Add structured output if requested
        if request.structured_output:
            backend_type = self._get_backend_type_for_role(role)

            if backend_type == "vllm":
                # vLLM now uses OpenAI-compatible response_format (updated API)
                if "json_schema" in request.structured_output:
                    payload.response_format = {
                        "type": "json_schema",
                        "json_schema": request.structured_output["json_schema"]
                    }
                else:
                    payload.response_format = {"type": "json_object"}
            elif backend_type == "llamacpp":
                # llama.cpp uses grammar constraints (simplified JSON grammar)
                json_grammar = self._create_json_grammar(request.structured_output)
                payload.grammar = json_grammar
            else:
                # OpenAI and other backends use response_format
                if "json_schema" in request.structured_output:
                    payload.response_format = {
                        "type": "json_schema",
                        "json_schema": request.structured_output["json_schema"]
                    }
                else:
                    payload.response_format = {"type": "json_object"}

        # Convert payload to dict for requests library
        payload_dict = {
            "model": payload.model,
            "messages": payload.messages,
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "stream": payload.stream,
        }
        if payload.response_format:
            payload_dict["response_format"] = payload.response_format
        if payload.grammar:
            payload_dict["grammar"] = payload.grammar

        # Debug: Log the request details
        logger.info(f"LLM Request: {role.value} to {url}")
        logger.info(f"Model: {payload.model}, Max tokens: {payload.max_tokens}")
        logger.debug(f"Request payload: {payload_dict}")

        # Set timeout based on LLM role - orchestrator needs more time for large prompts
        timeout_seconds = (
            ORCHESTRATOR_TIMEOUT_SECONDS if role == LLMRole.ORCHESTRATOR else FAST_TIMEOUT_SECONDS
        )

        logger.debug(f"Making HTTP request to {url} with timeout {timeout_seconds}s")
        response = self.session.post(url, json=payload_dict, headers=headers, timeout=timeout_seconds)
        logger.debug(f"HTTP response status: {response.status_code}")
        logger.debug(f"HTTP response headers: {dict(response.headers)}")

        response.raise_for_status()

        logger.debug(f"Raw response text length: {len(response.text)} chars")
        logger.debug(f"Raw response preview: {response.text[:500]}...")

        data = response.json()
        duration = time.time() - start_time
        logger.debug(f"Response parsed successfully, duration: {duration:.2f}s")

        if "choices" not in data or not data["choices"]:
            logger.error(f"Invalid LLM response format. Response keys: {list(data.keys())}")
            logger.error(f"Full response: {data}")
            raise ValueError("Invalid LLM response format")

        message = data["choices"][0]["message"]
        logger.debug(f"Message keys: {list(message.keys())}")

        # Extract content and reasoning content separately (vLLM/llama.cpp format)
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        logger.debug(f"Extracted content length: {len(content)} chars")
        logger.debug(f"Content preview: {content[:200]}...")

        if not content:
            logger.error(f"LLM response content is empty. Message: {message}")
            raise ValueError("LLM response content is empty")

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            llm_used=role.value,
            duration_seconds=duration,
            input_tokens=len(request.content) // TOKEN_ESTIMATION_DIVISOR,
            success=True,
        )

    def _get_backend_type_for_role(self, role: LLMRole) -> str:
        """Get backend type for the specified LLM role."""
        if role == LLMRole.FAST:
            return getattr(self.llm_config, 'fast_backend', 'vllm')
        elif role == LLMRole.ORCHESTRATOR:
            return getattr(self.llm_config, 'orchestrator_backend', 'llamacpp')
        else:
            return 'openai'  # Default fallback

    def _create_json_grammar(self, json_schema: Dict[str, Any]) -> str:
        """Create a GBNF JSON grammar for llama.cpp from JSON schema."""
        if json_schema.get("type") == "object":
            properties = json_schema.get("properties", {})
            required = json_schema.get("required", [])
            defs = json_schema.get("$defs", {})

            # Handle single-property objects (common case)
            if len(properties) == 1:
                prop_name = list(properties.keys())[0]
                prop_schema = list(properties.values())[0]

                # Handle $ref to enum definitions (Pydantic style)
                if "$ref" in prop_schema:
                    ref_path = prop_schema["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            ref_def = defs[ref_name]
                            if ref_def.get("type") == "string" and "enum" in ref_def:
                                enum_values = ref_def["enum"]
                                enum_choices = " | ".join(f'"{value}"' for value in enum_values)
                                return f'root ::= "{{" ws "\"{prop_name}\"" ws ":" ws ({enum_choices}) ws "}}" ws ::= [ \\t\\n\\r]*'

                # Handle direct string enums (like "yes"/"no")
                elif prop_schema.get("type") == "string" and "enum" in prop_schema:
                    enum_values = prop_schema["enum"]
                    enum_choices = " | ".join(f'"{value}"' for value in enum_values)
                    return f'root ::= "{{" ws "\"{prop_name}\"" ws ":" ws ({enum_choices}) ws "}}" ws ::= [ \\t\\n\\r]*'

                # Handle boolean fields
                elif prop_schema.get("type") == "boolean":
                    return f'root ::= "{{" ws "\"{prop_name}\"" ws ":" ws ("true" | "false") ws "}}" ws ::= [ \\t\\n\\r]*'

        # Fallback to basic JSON object grammar
        return '''root ::= "{" ws "}" | "{" ws object-item (ws "," ws object-item)* ws "}"
ws ::= [ \\t\\n\\r]*
object-item ::= "\\"" [a-zA-Z_][a-zA-Z0-9_]* "\\"" ws ":" ws value
value ::= "true" | "false" | "\\"" [^"]* "\\""'''

    def is_llm_available(self, role: LLMRole) -> bool:
        """Check if a specific LLM is configured and available."""
        if role == LLMRole.FAST:
            return bool(self.fast_config.api_url and self.fast_config.model)
        else:
            return bool(self.orchestrator_config.api_url and self.orchestrator_config.model)

    def get_available_features(self) -> FeatureAvailability:
        """Get which AI features are available based on LLM configuration."""
        fast_available = self.is_llm_available(LLMRole.FAST)
        orchestrator_available = self.is_llm_available(LLMRole.ORCHESTRATOR)
        any_llm_available = fast_available or orchestrator_available

        return FeatureAvailability(
            # LLM-powered features
            architecture_analysis=orchestrator_available,  # Requires orchestrator LLM
            docstring_generation=fast_available,  # Can use fast LLM
            code_smell_detection=fast_available,  # Can use fast LLM
            coverage_assessment=orchestrator_available,  # Requires orchestrator LLM
            llm_validation=any_llm_available,  # Any LLM works
            # Embedding-only features (no LLM required)
            semantic_similarity=True,  # Always available (uses local embeddings)
            embedding_clustering=True,  # Always available (uses local embeddings)
            duplicate_detection=True,  # Always available (uses local embeddings)
        )

    def get_status(self) -> LLMStatus:
        """Get status of both LLMs."""
        return LLMStatus(
            fast_configured=self.is_llm_available(LLMRole.FAST),
            orchestrator_configured=self.is_llm_available(LLMRole.ORCHESTRATOR),
            context_threshold=self.context_threshold,
            fallback_enabled=False,  # Always disabled for predictable behavior
            available_features=self.get_available_features(),
        )

    def process_request_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for process_request to support legacy code."""
        import asyncio
        try:
            # Always create a new event loop for sync calls
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.process_request(request))
            finally:
                new_loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            logger.error(f"Sync LLM request failed: {e}")
            return LLMResponse(
                content=f"LLM sync call failed: {e}",
                llm_used="error",
                duration_seconds=0,
                input_tokens=0,
                success=False,
                error=str(e)
            )


def create_llm_manager(config: Optional["LLMConfig"] = None) -> Optional[LLMManager]:
    """Create LLM manager from vibelint configuration.

    Always returns an LLMManager instance, even if no LLMs are configured.
    This allows embedding-only analysis to work without LLM endpoints.

    Args:
        config: Optional typed LLMConfig - if None, loads from config files
    """
    return LLMManager(config)
```

---
### File: src/vibelint/reporting.py

```python
"""
Comprehensive reporting system for vibelint analysis results.

Provides structured report generation with granular verbosity levels,
artifact management, hyperlinked reports, and multiple output formatters
for different consumers (humans, CI/CD, GitHub, LLMs).

vibelint/src/vibelint/reporting.py
"""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from vibelint.validators import BaseFormatter, Finding, Severity

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "VerbosityLevel",
    "NaturalLanguageFormatter",
    "JsonFormatter",
    "SarifFormatter",
    "LLMFormatter",
    "HumanFormatter",
    "BUILTIN_FORMATTERS",
    "FORMAT_CHOICES",
    "DEFAULT_FORMAT",
    "ExecutiveSummary",
    "PriorityAction",
    "TreeViolation",
    "Synthesis",
    "AnalysisResults",
    "FileAnalysisEntry",
    "ContentAnalysis",
    "TreeAnalysis",
]


class VerbosityLevel(Enum):
    """Report verbosity levels for different use cases."""

    EXECUTIVE = "executive"  # High-level summary for planning
    TACTICAL = "tactical"  # Actionable items for development
    DETAILED = "detailed"  # Comprehensive analysis with context
    FORENSIC = "forensic"  # Complete diagnostic information


@dataclass
class ExecutiveSummary:
    """High-level summary of analysis results."""

    overall_health: str
    critical_issues: int
    improvement_opportunities: int
    estimated_effort: str


@dataclass
class PriorityAction:
    """Priority action item from analysis."""

    title: str
    priority: str
    description: str
    effort_hours: str
    risk_if_ignored: str


@dataclass
class TreeViolation:
    """Organizational/tree structure violation."""

    violation_type: str
    message: str


@dataclass
class Synthesis:
    """Synthesis of all analysis results."""

    executive_summary: ExecutiveSummary
    priority_actions: List[PriorityAction]
    quick_wins: List[str] = field(default_factory=list)


@dataclass
class TreeAnalysis:
    """Tree/organizational analysis results."""

    quick_violations: List[TreeViolation] = field(default_factory=list)


@dataclass
class FileAnalysisEntry:
    """Single file analysis entry with findings."""

    file_path: str
    findings: List[Finding] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Content/structural analysis results."""

    file_analyses: List[FileAnalysisEntry] = field(default_factory=list)


@dataclass
class AnalysisResults:
    """Complete analysis results container."""

    synthesis: Synthesis
    tree_analysis: Optional[TreeAnalysis] = None
    content_analysis: Optional[ContentAnalysis] = None
    deep_analysis: Optional[Dict[str, Any]] = None  # Keep as dict for flexibility


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Output settings
    output_directory: Path
    report_name: str = "vibelint_analysis"
    verbosity_level: VerbosityLevel = VerbosityLevel.TACTICAL

    # Format settings
    formats: List[str] = None  # ["markdown", "json", "html"]
    include_artifacts: bool = True
    create_index: bool = True

    # Content settings
    max_findings_per_category: int = 20
    include_raw_llm_responses: bool = False
    include_performance_metrics: bool = True

    # Navigation settings
    generate_hyperlinks: bool = True
    create_quick_nav: bool = True

    def __post_init__(self):
        if self.formats is None:
            self.formats = ["markdown", "html"]


class ReportGenerator:
    """Generates structured analysis reports with granular verbosity control."""

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_comprehensive_report(
        self, analysis_results: AnalysisResults, timestamp: Optional[str] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive report with all artifacts."""

        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create report directory structure
        report_dir = self.config.output_directory / f"{self.config.report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Generate main report
        main_report_path = self._generate_main_report(analysis_results, report_dir, timestamp)
        generated_files["main_report"] = main_report_path

        # Generate artifacts if enabled
        if self.config.include_artifacts:
            artifact_paths = self._generate_artifacts(analysis_results, report_dir, timestamp)
            generated_files.update(artifact_paths)

        # Generate index/navigation if enabled
        if self.config.create_index:
            index_path = self._generate_index(report_dir, generated_files, timestamp)
            generated_files["index"] = index_path

        # Generate quick action plan
        quick_plan_path = self._generate_quick_action_plan(analysis_results, report_dir, timestamp)
        generated_files["quick_plan"] = quick_plan_path

        return generated_files

    def _generate_main_report(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Path:
        """Generate main analysis report."""

        # Filter content based on verbosity level
        filtered_results = self._filter_by_verbosity(analysis_results)

        # Generate markdown content
        content = self._format_main_report_markdown(filtered_results, timestamp)

        # Save to file
        report_path = report_dir / "main_report.md"
        report_path.write_text(content, encoding="utf-8")

        return report_path

    def _format_main_report_markdown(self, analysis_results: AnalysisResults, timestamp: str) -> str:
        """Format main report as markdown."""

        executive = analysis_results.synthesis.executive_summary
        priority_actions = analysis_results.synthesis.priority_actions

        content = f"""# Vibelint Analysis Report

Generated: {timestamp}
Verbosity Level: {self.config.verbosity_level.value}

## Executive Summary

- **Overall Health**: {executive.overall_health}
- **Critical Issues**: {executive.critical_issues}
- **Improvement Opportunities**: {executive.improvement_opportunities}
- **Estimated Effort**: {executive.estimated_effort}

## Priority Actions

"""

        for i, action in enumerate(priority_actions[:5], 1):
            content += f"""### {i}. {action.title} ({action.priority})

{action.description}

**Effort**: {action.effort_hours} hours
**Risk if ignored**: {action.risk_if_ignored}

"""

        # Add findings summary based on verbosity
        if self.config.verbosity_level != VerbosityLevel.EXECUTIVE:
            content += self._add_findings_section(analysis_results)

        return content

    def _add_findings_section(self, analysis_results: AnalysisResults) -> str:
        """Add findings section based on verbosity level."""
        content = "\n## Findings Summary\n\n"

        # Tree violations
        if analysis_results.tree_analysis:
            tree_violations = analysis_results.tree_analysis.quick_violations
            if tree_violations:
                content += f"### Organizational Issues ({len(tree_violations)})\n\n"
                for violation in tree_violations[: self.config.max_findings_per_category]:
                    content += f"- **{violation.violation_type}**: {violation.message}\n"
                content += "\n"

        # Content findings
        if analysis_results.content_analysis:
            content_findings = []
            for file_analysis in analysis_results.content_analysis.file_analyses:
                content_findings.extend(file_analysis.findings)

            if content_findings:
                content += f"### Structural Issues ({len(content_findings)})\n\n"
                for finding in content_findings[: self.config.max_findings_per_category]:
                    content += f"- **{finding.rule_id}**: {finding.message}\n"
                content += "\n"

        return content

    def _generate_artifacts(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Dict[str, Path]:
        """Generate detailed artifacts for different analysis aspects."""

        artifacts_dir = report_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        artifact_paths = {}

        # Generate JSON artifacts for each analysis level
        if analysis_results.tree_analysis:
            tree_path = artifacts_dir / "organizational_analysis.json"
            # Convert dataclass to dict for JSON serialization using asdict()
            tree_data = {
                "quick_violations": [asdict(v) for v in analysis_results.tree_analysis.quick_violations]
            }
            tree_path.write_text(
                json.dumps(tree_data, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["organizational"] = tree_path

        if analysis_results.content_analysis:
            content_path = artifacts_dir / "structural_analysis.json"
            # Convert dataclass to dict for JSON serialization using asdict()
            content_data = {
                "file_analyses": [
                    {
                        "file_path": fa.file_path,
                        "findings": [asdict(f) for f in fa.findings],
                    }
                    for fa in analysis_results.content_analysis.file_analyses
                ]
            }
            content_path.write_text(
                json.dumps(content_data, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["structural"] = content_path

        if analysis_results.deep_analysis:
            arch_path = artifacts_dir / "architectural_analysis.json"
            arch_path.write_text(
                json.dumps(analysis_results.deep_analysis, indent=2, default=str),
                encoding="utf-8",
            )
            artifact_paths["architectural"] = arch_path

        return artifact_paths

    def _generate_quick_action_plan(
        self, analysis_results: AnalysisResults, report_dir: Path, timestamp: str
    ) -> Path:
        """Generate quick action plan for immediate development focus."""

        synthesis = analysis_results.synthesis

        quick_plan = f"""# Quick Action Plan
Generated: {timestamp}

## Immediate Actions (< 1 hour each)

"""

        # Add quick wins
        quick_wins = synthesis.quick_wins
        for i, win in enumerate(quick_wins[:5], 1):
            quick_plan += f"{i}. {win}\n"

        quick_plan += "\n## Priority Issues (requires planning)\n\n"

        # Add priority actions
        priority_actions = synthesis.priority_actions
        for action in priority_actions[:3]:
            quick_plan += f"### {action.title} ({action.priority})\n"
            quick_plan += f"**Effort**: {action.effort_hours} hours\n"
            quick_plan += f"**Description**: {action.description}\n\n"

        quick_plan_path = report_dir / "QUICK_ACTION_PLAN.md"
        quick_plan_path.write_text(quick_plan, encoding="utf-8")

        return quick_plan_path

    def _generate_index(
        self, report_dir: Path, generated_files: Dict[str, Path], timestamp: str
    ) -> Path:
        """Generate navigation index for the report."""

        index_content = f"""# Vibelint Analysis Report Index
Generated: {timestamp}

## Main Reports

- [[REPORT] Main Analysis Report](main_report.md)
- [[ROCKET] Quick Action Plan](QUICK_ACTION_PLAN.md)

## Detailed Artifacts

"""

        # Add artifact links
        artifact_types = {
            "organizational": "[BUILD] Organizational Analysis",
            "structural": "[TOOL] Structural Analysis",
            "architectural": "[ARCH] Architectural Analysis",
        }

        for artifact_key, description in artifact_types.items():
            if artifact_key in generated_files:
                artifact_path = generated_files[artifact_key]
                relative_path = f"artifacts/{artifact_path.name}"
                index_content += f"- [{description}]({relative_path})\n"

        index_content += f"\n---\n\nReport generated by vibelint at {timestamp}\n"

        index_path = report_dir / "index.md"
        index_path.write_text(index_content, encoding="utf-8")

        return index_path

    def _filter_by_verbosity(self, analysis_results: AnalysisResults) -> AnalysisResults:
        """Filter analysis results based on configured verbosity level."""

        if self.config.verbosity_level == VerbosityLevel.EXECUTIVE:
            # Only high-level summary and critical issues
            critical_actions = self._extract_critical_issues(analysis_results)
            return AnalysisResults(
                synthesis=Synthesis(
                    executive_summary=analysis_results.synthesis.executive_summary,
                    priority_actions=critical_actions,
                    quick_wins=[],
                )
            )

        elif self.config.verbosity_level == VerbosityLevel.TACTICAL:
            # Actionable items and priority information
            # Limit findings per category
            limited_content = None
            if analysis_results.content_analysis:
                limited_content = self._limit_content_findings(analysis_results.content_analysis)

            return AnalysisResults(
                synthesis=analysis_results.synthesis,
                tree_analysis=analysis_results.tree_analysis,
                content_analysis=limited_content,
                deep_analysis=analysis_results.deep_analysis,
            )

        else:  # DETAILED or FORENSIC
            # Most or all information
            return analysis_results

    def _extract_critical_issues(self, analysis_results: AnalysisResults) -> List[PriorityAction]:
        """Extract only critical/blocking issues for executive summary."""
        critical_issues = []

        # Check synthesis for critical items
        for action in analysis_results.synthesis.priority_actions:
            if action.priority in ["P0", "P1"]:
                critical_issues.append(action)

        return critical_issues

    def _limit_content_findings(self, content_analysis: ContentAnalysis) -> ContentAnalysis:
        """Limit content findings for tactical verbosity."""
        limited_file_analyses = []

        for file_analysis in content_analysis.file_analyses:
            # Keep only high-severity findings for tactical view
            high_severity = [
                f
                for f in file_analysis.findings
                if f.severity in [Severity.BLOCK, Severity.WARN]
            ]
            limited_file_analyses.append(
                FileAnalysisEntry(
                    file_path=file_analysis.file_path, findings=high_severity[:5]  # Limit to 5
                )
            )

        return ContentAnalysis(file_analyses=limited_file_analyses)


# ===== FORMATTERS =====
# Consolidated from formatters.py


class NaturalLanguageFormatter(BaseFormatter):
    """Natural language output formatter for humans and AI agents."""

    name = "natural"
    description = "Natural language output format optimized for human and AI agent consumption"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results for human reading."""
        if not findings:
            return "All checks passed!"

        # Get max display limit from config
        max_displayed = 50  # Default to 50 issues for readability (0 means no limit)
        # Config can be a dict (external API data) - .get() is acceptable here
        if config and isinstance(config, dict):
            max_displayed = config.get("max_displayed_issues", 50)

        # Group findings by severity
        by_severity = {Severity.BLOCK: [], Severity.WARN: [], Severity.INFO: []}

        for finding in findings:
            if finding.severity in by_severity:
                by_severity[finding.severity].append(finding)

        lines = []
        displayed_count = 0
        total_count = len(findings)

        # Add findings by severity (highest first)
        for severity in [Severity.BLOCK, Severity.WARN, Severity.INFO]:
            if by_severity[severity]:
                lines.append(f"\n{severity.value}:")

                severity_findings = by_severity[severity]
                for _, finding in enumerate(severity_findings):
                    if max_displayed > 0 and displayed_count >= max_displayed:
                        remaining_total = total_count - displayed_count
                        lines.append("")
                        lines.append(
                            f"  WARNING: Showing first {max_displayed} issues. {remaining_total} more found."
                        )
                        lines.append(
                            "  TIP: Set max_displayed_issues = 0 in pyproject.toml to show all issues."
                        )
                        break

                    location = (
                        f"{finding.file_path}:{finding.line}"
                        if finding.line > 0
                        else str(finding.file_path)
                    )
                    lines.append(f"  {finding.rule_id}: {finding.message} ({location})")
                    if finding.suggestion:
                        lines.append(f"    → {finding.suggestion}")

                    displayed_count += 1

                if max_displayed > 0 and displayed_count >= max_displayed:
                    break

        # Add summary with full counts
        total_errors = sum(1 for f in findings if f.severity == Severity.BLOCK)
        total_warnings = sum(1 for f in findings if f.severity == Severity.WARN)
        total_info = sum(1 for f in findings if f.severity == Severity.INFO)

        summary_line = (
            f"\nSummary: {total_errors} errors, {total_warnings} warnings, {total_info} info"
        )
        if max_displayed > 0 and total_count > max_displayed:
            summary_line += f" (showing first {min(max_displayed, total_count)} of {total_count})"

        lines.append(summary_line)

        return "\n".join(lines)


class JsonFormatter(BaseFormatter):
    """JSON output formatter for machine processing."""

    name = "json"
    description = "JSON output format for CI/tooling integration"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results as JSON."""
        result = {"summary": summary, "findings": [finding.to_dict() for finding in findings]}
        return json.dumps(result, indent=2, default=str)


class SarifFormatter(BaseFormatter):
    """SARIF output formatter for GitHub integration."""

    name = "sarif"
    description = "SARIF format for GitHub code scanning"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results as SARIF JSON."""
        rules = {}
        results = []

        for finding in findings:
            # Collect unique rules
            if finding.rule_id not in rules:
                rules[finding.rule_id] = {
                    "id": finding.rule_id,
                    "name": finding.rule_id,
                    "shortDescription": {"text": finding.message},
                    "defaultConfiguration": {
                        "level": self._severity_to_sarif_level(finding.severity)
                    },
                }

            # Add result
            result = {
                "ruleId": finding.rule_id,
                "level": self._severity_to_sarif_level(finding.severity),
                "message": {"text": finding.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": str(finding.file_path)},
                            "region": {
                                "startLine": max(1, finding.line),
                                "startColumn": max(1, finding.column),
                            },
                        }
                    }
                ],
            }

            if finding.suggestion:
                result["fixes"] = [{"description": {"text": finding.suggestion}}]

            results.append(result)

        sarif_output = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "vibelint",
                            "version": "0.1.2",
                            "informationUri": "https://github.com/mithranm/vibelint",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }

        return json.dumps(sarif_output, separators=(",", ":"))

    def _severity_to_sarif_level(self, severity: Severity) -> str:
        """Convert vibelint severity to SARIF level."""
        # Mapping dict - .get() is legitimate for safe enum->string conversion
        mapping = {
            Severity.BLOCK: "error",
            Severity.WARN: "warning",
            Severity.INFO: "note",
            Severity.OFF: "none",
        }
        return mapping.get(severity, "warning")


class LLMFormatter(BaseFormatter):
    """LLM-optimized formatter for AI analysis."""

    name = "llm"
    description = "LLM-optimized format for AI analysis"

    def format_results(
        self, findings: List[Finding], summary: Dict[str, int], config: Optional[Any] = None
    ) -> str:
        """Format results for LLM analysis."""
        if not findings:
            return "No issues found."

        output = []
        for finding in findings:
            output.append(
                f"{finding.rule_id}: {finding.message} " f"({finding.file_path}:{finding.line})"
            )

        return "\n".join(output)


class HumanFormatter(NaturalLanguageFormatter):
    """Human-readable formatter (alias for NaturalLanguageFormatter)."""

    name = "human"
    description = "Human-readable format with colors and styling"


# Built-in report formatters
BUILTIN_FORMATTERS = {
    "natural": NaturalLanguageFormatter,
    "human": HumanFormatter,  # Separate class for plugin system compatibility
    "json": JsonFormatter,
    "sarif": SarifFormatter,
    "llm": LLMFormatter,
}

# Format choices for CLI - single source of truth
FORMAT_CHOICES = list(BUILTIN_FORMATTERS.keys())
DEFAULT_FORMAT = "natural"
```

---
### File: src/vibelint/rules.py

```python
"""
Rule management system for vibelint.

Handles rule configuration, severity overrides, and policy management.

vibelint/src/vibelint/rules.py
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from vibelint.validators import BaseValidator, Severity, plugin_manager

logger = logging.getLogger(__name__)


class DefaultSeverity(Enum):
    """Predefined severity levels for consistency."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


__all__ = ["RuleEngine", "create_default_rule_config", "DefaultSeverity"]


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
                except ValueError as e:
                    logger.debug(f"Invalid severity setting for rule {rule_id}: {setting} - {e}")
                    pass
            elif isinstance(setting, bool):
                # Boolean: True=default severity, False=OFF
                if not setting:
                    self._rule_overrides[rule_id] = Severity.OFF

        # Load disabled validators from ignore list
        ignore_codes = self.config.get("ignore", [])
        if isinstance(ignore_codes, list):
            for rule_id in ignore_codes:
                if isinstance(rule_id, str):
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
            # Pass the model through config
            config_with_model = dict(self.config)
            config_with_model["_shared_model"] = shared_model
            return validator_class(severity=severity, config=config_with_model)

        return validator_class(severity=severity, config=self.config)

    def get_enabled_validators(self) -> List[BaseValidator]:
        """Get all enabled validator instances."""
        validators = []
        all_validators = plugin_manager.get_all_validators()

        for _, validator_class in all_validators.items():
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
            import logging
            import os

            logger = logging.getLogger(__name__)

            try:
                from sentence_transformers import SentenceTransformer

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
                    project_root = getattr(self.config, "project_root", None)
                    if project_root and hasattr(self.config, "project_root"):
                        env_file = project_root / ".env"
                        if env_file and env_file.exists():
                            for line in env_file.read_text().splitlines():
                                if line.startswith("HF_TOKEN="):
                                    hf_token = line.split("=", 1)[1].strip().strip("\"'")
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
                logger.debug(
                    "Semantic similarity analysis disabled: sentence-transformers not available"
                )
                self._shared_models[model_key] = None
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._shared_models[model_key] = None

        return self._shared_models[model_key]

    def get_rule_summary(self) -> Dict[str, Any]:
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


def create_default_rule_config() -> Dict[str, Any]:
    """Create default rule configuration for new projects."""
    return {
        "rules": {
            # Semantic rule IDs (primary system)
            "DOCSTRING-MISSING": DefaultSeverity.INFO.value,  # Missing docstring is just info
            "EXPORTS-MISSING-ALL": DefaultSeverity.WARN.value,  # Missing __all__ is warning
            "PRINT-STATEMENT": DefaultSeverity.WARN.value,  # Print statements are warnings
            "EMOJI-IN-STRING": DefaultSeverity.WARN.value,  # Emojis can cause encoding issues
            "TODO-FOUND": DefaultSeverity.INFO.value,  # TODOs are informational
            "PARAMETERS-KEYWORD-ONLY": DefaultSeverity.INFO.value,  # Parameter suggestions are info
        },
        "plugins": {"enabled": ["vibelint.core"]},
    }
```

---
### File: src/vibelint/snapshot.py

```python
# vibelint/src/vibelint/snapshot.py
"""
Codebase snapshot generation in markdown format.

vibelint/src/vibelint/snapshot.py
"""

import fnmatch
import logging
from pathlib import Path

from vibelint.config import Config
from vibelint.discovery import discover_files
from vibelint.filesystem import get_relative_path, is_binary

__all__ = ["create_snapshot"]

logger = logging.getLogger(__name__)

# Constants for file tree structure
FILES_KEY = "__FILES__"


def create_snapshot(
    output_path: Path,
    target_paths: list[Path],
    config: Config,
) -> None:
    """
    Creates a Markdown snapshot file containing the project structure and file contents,
    respecting the include/exclude rules defined in pyproject.toml.

    Args:
    output_path: The path where the Markdown file will be saved.
    target_paths: List of initial paths (files or directories) to discover from.
    config: The vibelint configuration object.

    vibelint/src/vibelint/snapshot.py
    """

    assert config.project_root is not None, "Project root must be set before creating snapshot."
    project_root = config.project_root.resolve()

    absolute_output_path = output_path.resolve()

    logger.debug("create_snapshot: Running discovery based on pyproject.toml config...")

    discovered_files = discover_files(
        paths=target_paths,
        config=config,
        explicit_exclude_paths={absolute_output_path},
    )

    logger.debug(f"create_snapshot: Discovery finished, count: {len(discovered_files)}")

    # Debugging check (can be removed later)
    for excluded_pattern_root in [".pytest_cache", ".ruff_cache", ".git"]:
        present = any(excluded_pattern_root in str(f) for f in discovered_files)
        logger.debug(
            f"!!! Check @ start of create_snapshot: '{excluded_pattern_root}' presence in list: {present}"
        )

    file_infos: list[tuple[Path, str]] = []

    peek_globs = config.get("peek_globs", [])
    if not isinstance(peek_globs, list):
        logger.warning("Configuration 'peek_globs' is not a list. Ignoring peek rules.")
        peek_globs = []

    for abs_file_path in discovered_files:
        try:
            rel_path_obj = get_relative_path(abs_file_path, project_root)
            rel_path_str = str(rel_path_obj)  # Still useful for fnmatch below
        except ValueError:
            logger.warning(
                f"Skipping file outside project root during snapshot categorization: {abs_file_path}"
            )
            continue

        if is_binary(abs_file_path):
            cat = "BINARY"
        else:
            cat = "FULL"
            for pk in peek_globs:
                normalized_rel_path = rel_path_str.replace("\\", "/")
                normalized_peek_glob = pk.replace("\\", "/")
                if fnmatch.fnmatch(normalized_rel_path, normalized_peek_glob):
                    cat = "PEEK"
                    break
        file_infos.append((abs_file_path, cat))
        logger.debug(f"Categorized {rel_path_str} as {cat}")

    file_infos.sort(key=lambda x: x[0])

    logger.debug(f"Sorted {len(file_infos)} files for snapshot.")

    # Build the tree structure using a dictionary
    tree: dict = {}
    for f_path, f_cat in file_infos:
        try:
            # --- FIX START ---
            # Get the relative path object
            relative_path_obj = get_relative_path(f_path, project_root)
            # Use the .parts attribute which is OS-independent
            relative_parts = relative_path_obj.parts
            # --- FIX END ---
        except ValueError:
            # Handle files outside the project root if they somehow got here
            logger.warning(
                f"Skipping file outside project root during snapshot tree build: {f_path}"
            )
            continue

        node = tree
        # Iterate through the path components tuple
        for i, part in enumerate(relative_parts):
            # Skip empty parts if any somehow occur (unlikely with .parts)
            if not part:
                continue

            is_last_part = i == len(relative_parts) - 1

            if is_last_part:
                # This is the filename part
                if FILES_KEY not in node:
                    node[FILES_KEY] = []
                # Add the tuple (absolute path, category)
                node[FILES_KEY].append((f_path, f_cat))
            else:
                # This is a directory part
                if part not in node:
                    node[part] = {}  # Create a new dictionary for the subdirectory
                # Move deeper into the tree structure
                node = node[part]

    logger.info(f"Writing snapshot to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(absolute_output_path, "w", encoding="utf-8") as outfile:

            outfile.write("# Snapshot\n\n")

            # Write Filesystem Tree section
            outfile.write("## Filesystem Tree\n\n```\n")
            # Use project root name for the tree root display
            tree_root_name = project_root.name if project_root.name else str(project_root)
            outfile.write(f"{tree_root_name}/\n")
            _write_tree(outfile, tree, "")  # Pass the populated tree dictionary
            outfile.write("```\n\n")

            # Write File Contents section
            outfile.write("## File Contents\n\n")
            outfile.write("Files are ordered alphabetically by path.\n\n")
            for f, cat in file_infos:  # Iterate through the sorted list again
                try:
                    relpath_header = get_relative_path(f, project_root)
                    outfile.write(f"### File: {relpath_header}\n\n")
                    logger.debug(f"Writing content for {relpath_header} (Category: {cat})")

                    if cat == "BINARY":
                        outfile.write("```\n")
                        outfile.write("[Binary File - Content not displayed]\n")
                        outfile.write("```\n\n---\n")
                    elif cat == "PEEK":
                        outfile.write("```\n")
                        outfile.write("[PEEK - Content truncated]\n")
                        try:
                            with open(f, encoding="utf-8", errors="ignore") as infile:
                                lines_read = 0
                                for line in infile:
                                    if lines_read >= 10:  # Peek limit (e.g., 10 lines)
                                        outfile.write("...\n")
                                        break
                                    outfile.write(line)
                                    lines_read += 1
                        except (OSError, UnicodeDecodeError) as e:
                            logger.warning(f"Error reading file for peek {relpath_header}: {e}")
                            outfile.write(f"[Error reading file for peek: {e}]\n")
                        outfile.write("```\n\n---\n")
                    else:  # cat == "FULL"
                        lang = _get_language(f)
                        outfile.write(f"```{lang}\n")
                        try:
                            with open(f, encoding="utf-8", errors="ignore") as infile:
                                content = infile.read()
                                # Ensure final newline for cleaner markdown rendering
                                if not content.endswith("\n"):
                                    content += "\n"
                                outfile.write(content)
                        except (OSError, UnicodeDecodeError) as e:
                            logger.warning(f"Error reading file content {relpath_header}: {e}")
                            outfile.write(f"[Error reading file: {e}]\n")
                        outfile.write("```\n\n---\n")

                except (OSError, ValueError, TypeError) as e:
                    # General error handling for processing a single file entry
                    try:
                        relpath_header_err = get_relative_path(f, project_root)
                    except ValueError:
                        relpath_header_err = str(f)  # Fallback to absolute path if rel path fails

                    logger.error(
                        f"Error processing file entry for {relpath_header_err} in snapshot: {e}",
                        exc_info=True,
                    )
                    outfile.write(f"### File: {relpath_header_err} (Error)\n\n")
                    outfile.write(f"[Error processing file entry: {e}]\n\n---\n")

            # Add a final newline for good measure
            outfile.write("\n")

    except OSError as e:
        # Error writing the main output file
        logger.error(f"Failed to write snapshot file {absolute_output_path}: {e}", exc_info=True)
        raise  # Re-raise IOErrors
    except (ValueError, TypeError, RuntimeError) as e:
        # Catch-all for other unexpected errors during writing
        logger.error(f"An unexpected error occurred during snapshot writing: {e}", exc_info=True)
        raise  # Re-raise other critical exceptions


def _write_tree(outfile, node: dict, prefix=""):
    """
    Helper function to recursively write the directory tree structure
    from the prepared dictionary.

    Args:
        outfile: The file object to write to.
        node: The current dictionary node representing a directory.
        prefix: The string prefix for drawing tree lines.

    vibelint/src/vibelint/snapshot.py
    """
    # Separate directories (keys other than FILES_KEY) from files (items in FILES_KEY)
    dirs = sorted([k for k in node if k != FILES_KEY])
    files_data: list[tuple[Path, str]] = sorted(node.get(FILES_KEY, []), key=lambda x: x[0].name)

    # Combine directory names and file names for iteration order
    entries = dirs + [f_info[0].name for f_info in files_data]

    for i, name in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        outfile.write(f"{prefix}{connector}")

        if name in dirs:
            # It's a directory - write its name and recurse
            outfile.write(f"{name}/\n")
            new_prefix = prefix + ("    " if is_last else "│   ")
            _write_tree(outfile, node[name], new_prefix)  # Recurse into the sub-dictionary
        else:
            # It's a file - find its category and write name with indicators
            file_info_tuple = next((info for info in files_data if info[0].name == name), None)
            file_cat = "FULL"  # Default category
            if file_info_tuple:
                file_cat = file_info_tuple[1]  # Get category ('FULL', 'PEEK', 'BINARY')

            # Add indicators for non-full content files
            peek_indicator = " (PEEK)" if file_cat == "PEEK" else ""
            binary_indicator = " (BINARY)" if file_cat == "BINARY" else ""
            outfile.write(f"{name}{peek_indicator}{binary_indicator}\n")


def _get_language(file_path: Path) -> str:
    """
    Guess language for syntax highlighting based on extension.
    Returns an empty string if no specific language is known.

    Args:
        file_path: The path to the file.

    Returns:
        A string representing the language identifier for markdown code blocks,
        or an empty string.

    vibelint/src/vibelint/snapshot.py
    """
    ext = file_path.suffix.lower()
    # Mapping from file extension to markdown language identifier
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".less": "less",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".ps1": "powershell",
        ".bat": "batch",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".gitignore": "gitignore",
        ".env": "bash",  # Treat .env like bash for highlighting often
        ".tf": "terraform",
        ".hcl": "terraform",
        ".lua": "lua",
        ".perl": "perl",
        ".pl": "perl",
        ".r": "r",
        ".ex": "elixir",
        ".exs": "elixir",
        ".dart": "dart",
        ".groovy": "groovy",
        ".gradle": "groovy",  # Gradle files often use groovy
        ".vb": "vbnet",
        ".fs": "fsharp",
        ".fsi": "fsharp",
        ".fsx": "fsharp",
        ".fsscript": "fsharp",
    }
    return mapping.get(ext, "")  # Return the mapped language or empty string
```

---
### File: src/vibelint/ui.py

```python
"""
Console and UI utility functions for vibelint.

vibelint/src/vibelint/ui.py
"""

import logging
import shutil

from rich.console import Console

logger = logging.getLogger(__name__)

__all__ = [
    "console",
    "scale_ascii_art_by_height",
    "scale_to_terminal_by_height",
]

# Global console instance used throughout vibelint
console = Console()


# === ASCII Art Utilities ===


def _get_terminal_size():
    """
    Returns the terminal size as a tuple (width, height) of characters.
    Falls back to (80, 24) if the dimensions cannot be determined.

    vibelint/src/vibelint/ui.py
    """
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines
    except OSError as e:
        # Terminal size unavailable in non-interactive environments
        logger.debug("Failed to get terminal size: %s", e)
        return 80, 24


def scale_ascii_art_by_height(ascii_art: str, target_height: int) -> str:
    """
    Scales the ASCII art to have a specified target height (in characters)
    while preserving the original aspect ratio. The target width is
    automatically computed based on the scaling factor.

    Args:
        ascii_art: The ASCII art string to scale
        target_height: Target height in character lines

    Returns:
        Scaled ASCII art string

    vibelint/src/vibelint/ui.py
    """
    # Split into lines and remove any fully blank lines.
    lines = [line for line in ascii_art.splitlines() if line.strip()]
    if not lines:
        return ""

    orig_height = len(lines)
    orig_width = max(len(line) for line in lines)

    # Pad all lines to the same length (for a rectangular grid)
    normalized_lines = [line.ljust(orig_width) for line in lines]

    # Compute the vertical scale factor and derive the target width.
    scale_factor = target_height / orig_height
    target_width = max(1, int(orig_width * scale_factor))

    # Calculate step sizes for sampling
    row_step = orig_height / target_height
    col_step = orig_width / target_width if target_width > 0 else 1

    result_lines = []
    for r in range(target_height):
        orig_r = min(int(r * row_step), orig_height - 1)
        new_line = []
        for c in range(target_width):
            orig_c = min(int(c * col_step), orig_width - 1)
            new_line.append(normalized_lines[orig_r][orig_c])
        result_lines.append("".join(new_line))

    return "\n".join(result_lines)


def scale_to_terminal_by_height(ascii_art: str) -> str:
    """
    Scales the provided ASCII art to fit based on the terminal's available height.
    The width is computed automatically to maintain the art's original aspect ratio.

    Args:
        ascii_art: The ASCII art string to scale

    Returns:
        Scaled ASCII art string

    vibelint/src/vibelint/ui.py
    """
    _, term_height = _get_terminal_size()
    # Optionally, leave a margin (here, using 90% of available height)
    target_height = max(1, int(term_height * 0.9))
    return scale_ascii_art_by_height(ascii_art, target_height)
```

---
### File: src/vibelint/validation_engine.py

```python
"""
Plugin-aware validation runner for vibelint.

This module provides the PluginValidationRunner that uses the new plugin system
to run validators and format output according to user configuration.

vibelint/src/vibelint/plugin_runner.py
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from vibelint.discovery import discover_files, discover_files_from_paths
from vibelint.validators import Finding, Severity, plugin_manager
from vibelint.reporting import BUILTIN_FORMATTERS
from vibelint.rules import RuleEngine

# Note: No longer importing BUILTIN_VALIDATORS - using plugin discovery instead

__all__ = ["PluginValidationRunner", "run_plugin_validation"]


class PluginValidationRunner:
    """Runs validation using the plugin system."""

    def __init__(self, config_dict: Dict[str, Any], project_root: Path):
        """Initialize the plugin validation runner."""
        self.project_root = project_root
        self.config_dict = config_dict
        self.config = config_dict  # Add config property for formatters
        self.rule_engine = RuleEngine(config_dict)
        self.findings: List[Finding] = []

        # Register built-in validators with plugin manager
        self._register_builtin_validators()

    def _register_builtin_validators(self):
        """Register built-in validators with the plugin manager via entry point discovery."""
        # Built-in validators are now discovered automatically via entry points
        plugin_manager.load_plugins()

    def run_validation(self, file_paths: List[Path]) -> List[Finding]:
        """Run validation on the specified files."""
        self.findings = []

        # Get enabled validators
        validators = self.rule_engine.get_enabled_validators()

        # Create extended config with analysis context
        analysis_config = dict(self.config)
        analysis_config["_analysis_files"] = file_paths  # Pass the actual files being analyzed

        for file_path in file_paths:
            if not file_path.exists() or not file_path.is_file():
                continue

            if file_path.suffix != ".py":
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logging.getLogger(__name__).debug(f"Could not read file {file_path}: {e}")
                continue

            # Run all validators on this file
            for validator in validators:
                try:
                    for finding in validator.validate(file_path, content, analysis_config):
                        # Make path relative to project root
                        relative_path = file_path.relative_to(self.project_root)
                        finding.file_path = relative_path
                        self.findings.append(finding)
                except (ImportError, AttributeError, ValueError, SyntaxError) as e:
                    logging.getLogger(__name__).debug(
                        f"Validator {validator.__class__.__name__} failed on {file_path}: {e}"
                    )
                    continue

        return self.findings

    def get_summary(self) -> Dict[str, int]:
        """Get summary counts by severity level."""
        summary = defaultdict(int)
        for finding in self.findings:
            summary[finding.severity.value] += 1
        return dict(summary)

    def format_output(self, output_format: str = "human") -> str:
        """Format the validation results."""
        # Get formatter
        if output_format in BUILTIN_FORMATTERS:
            formatter_class = BUILTIN_FORMATTERS[output_format]
            formatter = formatter_class()
        else:
            # Try plugin formatters
            formatter_class = plugin_manager.get_formatter(output_format)
            if formatter_class:
                formatter = formatter_class()
            else:
                # Fallback to human format
                formatter = BUILTIN_FORMATTERS["human"]()

        summary = self.get_summary()
        return formatter.format_results(self.findings, summary, self.config)

    def has_blocking_issues(self) -> bool:
        """Check if any findings are blocking (BLOCK severity)."""
        return any(finding.severity == Severity.BLOCK for finding in self.findings)

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on findings."""
        if self.has_blocking_issues():
            return 1
        return 0


def run_plugin_validation(
    config_dict: Dict[str, Any],
    project_root: Path,
    include_globs_override: List[Path] | None = None,
) -> PluginValidationRunner:
    """
    Run validation using the plugin system.

    Args:
        config_dict: Configuration dictionary from pyproject.toml
        project_root: Project root path
        include_globs_override: Optional list of paths to override include_globs.
                               If provided, only these paths are analyzed instead of
                               using the configured include_globs patterns.

    Returns:
        PluginValidationRunner with results
    """
    from vibelint.config import Config

    runner = PluginValidationRunner(config_dict, project_root)

    # Create a fake config object for discovery
    fake_config = Config(project_root, config_dict)

    # Choose discovery method based on whether include_globs are overridden
    if include_globs_override:
        # Use custom path discovery (include_globs override)
        files = discover_files_from_paths(
            custom_paths=include_globs_override, config=fake_config, explicit_exclude_paths=set()
        )
    else:
        # Use original discovery method with configured include_globs
        files = discover_files(
            paths=[project_root], config=fake_config, explicit_exclude_paths=set()
        )

    # Run validation
    runner.run_validation(files)

    return runner
```

---
### File: src/vibelint/validators/__init__.py

```python
"""
vibelint validators sub-package.

Modular validator system with centralized registry and discovery.

Responsibility: Validator module organization and re-exports only.
Individual validation logic belongs in specific validator modules.

vibelint/src/vibelint/validators/__init__.py
"""

# Import core types FIRST (before subdirectories to avoid circular imports)
from .types import (
    BaseFormatter,
    BaseValidator,
    Finding,
    Formatter,
    Severity,
    Validator,
    get_all_formatters,
    get_formatter,
    plugin_manager,
)

# Import registry system (also before subdirectories)
from .registry import (get_all_validators, get_validator, register_validator,
                       validator_registry)

# Import validator categories for direct access (LAST to avoid circular imports)
from . import project_wide, single_file

# Note: Individual validators should be imported from their specific modules
# to prevent duplicate import paths and keep the module hierarchy clear.

__all__ = [
    # Core types
    "Severity",
    "Finding",
    "BaseValidator",
    "BaseFormatter",
    "Validator",
    "Formatter",
    "get_formatter",
    "get_all_formatters",
    "plugin_manager",
    # Registry system
    "validator_registry",
    "register_validator",
    "get_validator",
    "get_all_validators",
    # Category modules
    "single_file",
    "project_wide",
]
```

---
### File: src/vibelint/validators/project_wide/__init__.py

```python
"""
Project-wide validators for vibelint.

These validators analyze entire projects and require knowledge
of multiple files to identify issues like:
- Dead code across modules
- API consistency violations
- Architecture pattern violations
- Semantic similarity between modules

vibelint/src/vibelint/validators/project_wide/__init__.py
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List

from ...validators.types import BaseValidator, Finding


class ProjectWideValidator(BaseValidator):
    """Base class for validators that analyze entire projects."""

    def validate_project(self, project_files: Dict[Path, str], config=None) -> Iterator[Finding]:
        """
        Validate entire project with knowledge of all files.

        Args:
            project_files: Dictionary mapping file paths to their content
            config: Configuration object

        Yields:
            Finding objects for any issues found
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Project-wide validators must implement validate_project")

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Single-file validate method for project-wide validators.

        Project-wide validators should not be called on individual files.
        This method raises an error to prevent misuse.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a project-wide validator. "
            "Use validate_project() instead of validate()."
        )

    def requires_project_context(self) -> bool:
        """Project-wide validators require full project context."""
        return True


def get_project_wide_validators() -> List[str]:
    """Get list of project-wide validator names."""
    return [
        "DEAD-CODE-FOUND",
        "ARCHITECTURE-INCONSISTENT",
        "ARCHITECTURE-LLM",
        "SEMANTIC-SIMILARITY",
        "FALLBACK-SILENT-FAILURE",
        "API-CONSISTENCY",
        "CODE-SMELLS",
        "MODULE-COHESION",
    ]
```

---
### File: src/vibelint/validators/project_wide/api_consistency.py

```python
"""
API consistency validator for vibelint.

Detects inconsistent API usage patterns, missing required parameters,
and architectural violations that lead to runtime failures.

vibelint/src/vibelint/validators/api_consistency.py
"""

import ast
import logging
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["APIConsistencyValidator"]


def _get_function_name(node: ast.Call) -> str:
    """Extract function name from call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


class APIConsistencyValidator(BaseValidator):
    """Validator for API consistency and usage patterns."""

    rule_id = "API-CONSISTENCY"
    name = "API Consistency Checker"
    description = "Detects inconsistent API usage, missing parameters, and architectural violations"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        # Known API signatures and their requirements
        self.known_apis = {
            "load_config": {
                "required_args": ["start_path"],
                "module": "config",
                "common_mistakes": [
                    "Called without required start_path parameter",
                    "Often needs Path('.') or similar as argument",
                ],
            },
            "create_llm_manager": {
                "required_args": ["config"],
                "module": "llm_manager",
                "common_mistakes": ["Requires config dict with llm section"],
            },
        }

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for API consistency issues."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Check function calls for API misuse
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield from self._check_function_call(node, file_path)

            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                yield from self._check_import_usage(node, file_path, tree)

    def _check_function_call(self, node: ast.Call, file_path: Path) -> Iterator[Finding]:
        """Check individual function calls for API consistency."""
        func_name = _get_function_name(node)

        if func_name in self.known_apis:
            api_info = self.known_apis[func_name]

            # Check required arguments
            provided_args = len(node.args)
            required_args = len(api_info["required_args"])

            if provided_args < required_args:
                missing_args = api_info["required_args"][provided_args:]
                yield self.create_finding(
                    message=f"API misuse: {func_name}() missing required arguments: {', '.join(missing_args)}",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion=f"Add required arguments: {func_name}({', '.join(api_info['required_args'])})",
                )

    def _check_import_usage(
        self, node: ast.AST, file_path: Path, tree: ast.AST
    ) -> Iterator[Finding]:
        """Check for inconsistent import and usage patterns."""
        if isinstance(node, ast.ImportFrom):
            if node.module == "config" and any(
                alias.name == "load_config" for alias in (node.names or [])
            ):
                # Check if load_config is used correctly in this file
                for call_node in ast.walk(tree):
                    if (
                        isinstance(call_node, ast.Call)
                        and isinstance(call_node.func, ast.Name)
                        and call_node.func.id == "load_config"
                    ):

                        if not call_node.args:
                            yield self.create_finding(
                                message="Configuration anti-pattern: load_config() called without start_path",
                                file_path=file_path,
                                line=call_node.lineno,
                                suggestion="Use load_config(Path('.')) or pass explicit path for config discovery",
                            )


class ConfigurationPatternValidator(BaseValidator):
    """Validator for configuration pattern consistency."""

    rule_id = "CONFIG-PATTERN"
    name = "Configuration Pattern Checker"
    description = "Ensures consistent configuration loading and usage patterns"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for configuration pattern issues."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Track how config is loaded and used
        config_loading_patterns = []
        config_usage_patterns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = _get_function_name(node)

                if func_name == "load_config":
                    config_loading_patterns.append(node.lineno)

                elif "config" in str(node).lower():
                    config_usage_patterns.append(node.lineno)

        # Check for multiple config loading approaches in same file
        if len(config_loading_patterns) > 1:
            yield self.create_finding(
                message="Configuration inconsistency: Multiple config loading patterns detected",
                file_path=file_path,
                line=config_loading_patterns[0],
                suggestion="Consolidate to single source of truth for configuration",
            )

        # Check for config dict creation vs proper loading
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "llm" for key in node.keys if key
            ):

                yield self.create_finding(
                    message="Configuration anti-pattern: Manual config dict creation detected",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Use load_config() for single source of truth instead of manual dict creation",
                )
```

---
### File: src/vibelint/validators/project_wide/code_smells.py

```python
"""
Code smell detection validator implementing Martin Fowler's taxonomy.

Detects common code smells like long methods, large classes, magic numbers,
and other patterns that indicate design issues.

vibelint/src/vibelint/validators/code_smells.py
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ...validators.types import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)

__all__ = ["CodeSmellValidator"]


class CodeSmellValidator(BaseValidator):
    """Detects common code smells based on Martin Fowler's taxonomy."""

    rule_id = "CODE-SMELLS"
    name = "Code Smell Detector"
    description = "Detects long methods, large classes, magic numbers, and other code smells"
    default_severity = Severity.INFO

    def validate(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Single-pass AST analysis for code smell detection."""

        # Check file length first (doesn't require AST parsing)
        yield from self._check_file_length(file_path, content, config)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        # Single AST walk detecting all smell categories
        for node in ast.walk(tree):
            yield from self._check_bloaters(node, file_path)
            yield from self._check_lexical_abusers(node, file_path)
            yield from self._check_couplers(node, file_path)
            yield from self._check_obfuscators(node, file_path)

    def _check_bloaters(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Bloater code smells: Large Class, Long Method, Long Parameter List."""
        # Long Method (>20 lines suspicious, >50 bad)
        if isinstance(node, ast.FunctionDef):
            method_length = self._count_logical_lines(node)
            if method_length > 50:
                yield self.create_finding(
                    message=f"Method '{node.name}' is too long ({method_length} lines)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break method into smaller, focused functions",
                )
            elif method_length > 20:
                yield self.create_finding(
                    message=f"Method '{node.name}' is getting long ({method_length} lines)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider breaking into smaller functions",
                )

        # Large Class (>500 lines or >20 methods)
        elif isinstance(node, ast.ClassDef):
            class_length = self._count_logical_lines(node)
            method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
            if class_length > 500 or method_count > 20:
                yield self.create_finding(
                    message=f"Class '{node.name}' is too large ({class_length} lines, {method_count} methods)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Split class responsibilities using Single Responsibility Principle",
                )

        # Long Parameter List (>3 suspicious, >5 bad)
        if isinstance(node, ast.FunctionDef):
            param_count = len(node.args.args)
            if param_count > 5:
                yield self.create_finding(
                    message=f"Function '{node.name}' has too many parameters ({param_count})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider parameter object or builder pattern",
                )

    def _check_lexical_abusers(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Lexical Abuser code smells: Magic Numbers, Uncommunicative Names."""
        # Magic Numbers
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in [0, 1, -1, 2] and not self._is_in_test_context(node):
                yield self.create_finding(
                    message=f"Magic number '{node.value}' should be named constant",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion=f"Replace with named constant: MEANINGFUL_NAME = {node.value}",
                )

        # Uncommunicative Names
        name_patterns = [
            (ast.FunctionDef, "function", "name"),
            (ast.ClassDef, "class", "name"),
            (ast.arg, "parameter", "arg"),
        ]
        for node_type, context, attr in name_patterns:
            if isinstance(node, node_type):
                if hasattr(node, attr):
                    name = getattr(node, attr)
                    if name and self._is_uncommunicative_name(name):
                        yield self.create_finding(
                            message=f"Uncommunicative {context} name '{name}'",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion=f"Use descriptive name that explains {context} purpose",
                        )

    def _check_couplers(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Coupler code smells: Message Chain, Feature Envy."""
        # Message Chain (a.b.c.d.method())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain_length = self._count_attribute_chain(node.func)
            if chain_length > 3:
                yield self.create_finding(
                    message=f"Long message chain detected ({chain_length} levels)",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Consider introducing intermediate methods to reduce coupling",
                )

    def _check_obfuscators(self, node: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Detect Obfuscator code smells: Complicated Boolean Expression, Clever Code."""
        # Complicated Boolean Expression
        if isinstance(node, ast.BoolOp):
            complexity = self._calculate_boolean_complexity(node)
            if complexity > 4:
                yield self.create_finding(
                    message=f"Complex boolean expression (complexity: {complexity})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break into intermediate boolean variables with descriptive names",
                )

        # Clever Code (nested comprehensions)
        if isinstance(node, ast.ListComp):
            nesting_level = self._count_comprehension_nesting(node)
            if nesting_level > 2:
                yield self.create_finding(
                    message=f"Overly complex list comprehension (nesting level: {nesting_level})",
                    file_path=file_path,
                    line=node.lineno,
                    suggestion="Break into multiple steps or use traditional loops for clarity",
                )

    # Helper methods
    def _count_logical_lines(self, node: ast.AST) -> int:
        """Count logical lines of code (excluding comments/blank lines)."""
        lines = set()
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                lines.add(child.lineno)
        return len(lines)

    def _is_uncommunicative_name(self, name: str) -> bool:
        """Check if name is uncommunicative."""
        if len(name) == 1 and name not in ["i", "j", "k", "x", "y", "z"]:
            return True
        if len(name) > 2 and not any(c in "aeiou" for c in name.lower()):
            return True
        return False

    def _count_attribute_chain(self, node: ast.Attribute) -> int:
        """Count depth of attribute chain."""
        count = 1
        current = node.value
        while isinstance(current, ast.Attribute):
            count += 1
            current = current.value
        return count

    def _calculate_boolean_complexity(self, node: ast.BoolOp) -> int:
        """Calculate complexity of boolean expression."""
        complexity = 1
        for value in node.values:
            if isinstance(value, ast.BoolOp):
                complexity += self._calculate_boolean_complexity(value)
            else:
                complexity += 1
        return complexity

    def _count_comprehension_nesting(self, node: ast.ListComp) -> int:
        """Count nesting level of comprehensions."""
        max_nesting = 1
        for generator in node.generators:
            for comp in ast.walk(generator.iter):
                if isinstance(comp, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    max_nesting = max(max_nesting, 1 + self._count_comprehension_nesting(comp))
        return max_nesting

    def _is_in_test_context(self, node: ast.AST) -> bool:
        """Check if node is in test context where magic numbers are more acceptable."""
        return False  # Simplified for now

    def _check_file_length(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Check if file exceeds recommended line count limits (FILE-TOO-LONG smell)."""
        # Default thresholds can be overridden in config
        warning_threshold = 500
        error_threshold = 1000
        exclude_patterns = ["test_*.py", "*_test.py", "conftest.py", "__init__.py"]

        if config:
            warning_threshold = config.get("line_count_warning", warning_threshold)
            error_threshold = config.get("line_count_error", error_threshold)
            exclude_patterns = config.get("line_count_exclude", exclude_patterns)

        # Skip excluded files
        filename = file_path.name
        for pattern in exclude_patterns:
            if (
                filename == pattern
                or (pattern.startswith("*") and filename.endswith(pattern[1:]))
                or (pattern.endswith("*") and filename.startswith(pattern[:-1]))
            ):
                return

        # Count non-empty lines (excluding pure whitespace)
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        line_count = len(non_empty_lines)

        # Check thresholds
        if line_count >= error_threshold:
            yield self.create_finding(
                message=f"File is too long ({line_count} lines, threshold: {error_threshold})",
                file_path=file_path,
                line=1,
                suggestion=self._get_file_split_suggestion(file_path, line_count, content),
            )
        elif line_count >= warning_threshold:
            yield self.create_finding(
                message=f"File is getting long ({line_count} lines, threshold: {warning_threshold})",
                file_path=file_path,
                line=1,
                suggestion=self._get_file_split_suggestion(file_path, line_count, content),
            )

    def _get_file_split_suggestion(self, file_path: Path, line_count: int, content: str) -> str:
        """Generate context-appropriate file splitting suggestions."""
        try:
            tree = ast.parse(content)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if len(classes) > 1:
                return f"Split {len(classes)} classes into separate files"
            elif len(functions) > 10:
                return f"Group {len(functions)} functions into classes or separate modules"
            elif "cli" in str(file_path).lower():
                return "Break CLI into command modules (validation, analysis, etc.)"
            elif "config" in str(file_path).lower():
                return "Separate config loading, validation, and defaults"
            else:
                return "Extract classes or functions into separate modules"
        except (SyntaxError, UnicodeDecodeError):
            return f"Break this {line_count}-line file into smaller, focused modules"
```

---
### File: src/vibelint/validators/project_wide/dead_code.py

```python
"""
Dead code detection validator with project-wide call graph analysis.

Identifies unused imports, unreferenced functions, duplicate implementations,
and other forms of dead code that can be safely removed.

Performance optimized: builds a single project-wide call graph and uses
graph traversal to identify unreachable code.

vibelint/src/vibelint/validators/dead_code.py
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from ...validators.types import BaseValidator, Finding, Severity
from ...filesystem import find_files_by_extension, find_project_root

__all__ = ["DeadCodeValidator"]


class CallGraph:
    """Project-wide call graph for efficient dead code detection."""

    def __init__(self):
        self.definitions: Dict[str, Tuple[Path, int]] = {}  # name -> (file, line)
        self.calls: Dict[str, Set[str]] = defaultdict(set)  # caller -> set of callees
        self.exports: Dict[str, Set[str]] = {}  # file -> set of exported names
        self.imports: Dict[str, Set[str]] = defaultdict(set)  # file -> set of imports
        self.file_modules: Dict[Path, str] = {}  # file -> module name

    def add_definition(self, name: str, file_path: Path, line: int):
        """Record a function/class definition."""
        self.definitions[name] = (file_path, line)

    def add_call(self, caller: str, callee: str):
        """Record a function call."""
        self.calls[caller].add(callee)

    def add_export(self, file_path: Path, name: str):
        """Record an exported name from __all__."""
        file_key = str(file_path)
        if file_key not in self.exports:
            self.exports[file_key] = set()
        self.exports[file_key].add(name)

    def add_import(self, file_path: Path, name: str):
        """Record an imported name."""
        self.imports[str(file_path)].add(name)

    def is_exported(self, name: str, file_path: Path) -> bool:
        """Check if name is exported from file."""
        return name in self.exports.get(str(file_path), set())

    def get_reachable_from_exports(self) -> Set[str]:
        """Get all names reachable from exported functions."""
        reachable = set()
        queue = []

        # Start with all exported names
        for exports in self.exports.values():
            queue.extend(exports)
            reachable.update(exports)

        # BFS traversal of call graph
        while queue:
            current = queue.pop(0)
            for callee in self.calls.get(current, []):
                if callee not in reachable:
                    reachable.add(callee)
                    queue.append(callee)

        return reachable


class DeadCodeValidator(BaseValidator):
    """Detects various forms of dead code using project-wide call graph analysis."""

    rule_id = "DEAD-CODE-FOUND"
    name = "Dead Code Detector"
    description = "Identifies unused imports, unreferenced functions, and other dead code"
    default_severity = Severity.WARN

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        self._call_graph: CallGraph | None = None
        self._project_root: Path | None = None
        self._analyzed_files: Set[Path] = set()

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze file for dead code patterns."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Build project context on first file
        if self._project_root is None:
            self._project_root = find_project_root(file_path) or file_path.parent

        # Build call graph if this is the first analysis
        if self._call_graph is None:
            self._call_graph = self._build_call_graph(self._project_root)

        # Mark this file as analyzed
        self._analyzed_files.add(file_path)

        # Analyze the file using the call graph
        yield from self._check_unused_imports(file_path, tree, content)
        yield from self._check_unreferenced_definitions(file_path, tree)
        yield from self._check_duplicate_patterns(file_path, content)
        yield from self._check_legacy_patterns(file_path, content)

    def _build_call_graph(self, project_root: Path) -> CallGraph:
        """Build project-wide call graph for all Python files."""
        graph = CallGraph()
        exclude_patterns = ["*/__pycache__/*", "*/.pytest_cache/*", "*/build/*", "*/dist/*"]
        project_files = find_files_by_extension(
            project_root, extension=".py", exclude_globs=exclude_patterns
        )

        for py_file in project_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                # Extract module name
                module_name = self._get_module_name(py_file, project_root)
                graph.file_modules[py_file] = module_name

                # Extract definitions, calls, and exports
                self._extract_from_ast(py_file, tree, graph)

            except (UnicodeDecodeError, SyntaxError):
                continue

        return graph

    def _extract_from_ast(self, file_path: Path, tree: ast.AST, graph: CallGraph):
        """Extract definitions, calls, and exports from an AST."""
        # Extract __all__ exports
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    graph.add_export(file_path, elt.value)

        # Extract function/class definitions and their internal calls
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                name = node.name
                graph.add_definition(name, file_path, node.lineno)

                # Extract calls within this function/class
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.Call):
                        if isinstance(inner_node.func, ast.Name):
                            graph.add_call(name, inner_node.func.id)
                        elif isinstance(inner_node.func, ast.Attribute):
                            # Record method calls
                            graph.add_call(name, inner_node.func.attr)

    def _check_unused_imports(
        self, file_path: Path, tree: ast.AST, content: str
    ) -> Iterator[Finding]:
        """Check for imported names that are never used."""
        imported_names: Dict[str, int] = {}  # name -> line number
        used_names: Set[str] = set()

        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names[name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue  # Skip wildcard imports
                    name = alias.asname if alias.asname else alias.name
                    imported_names[name] = node.lineno

        # Collect all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # For attribute access like `os.path`, record `os` as used
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused imports
        for name, line_num in imported_names.items():
            if name not in used_names:
                # Check if exported
                if self._call_graph.is_exported(name, file_path):
                    continue

                # Check for dynamic string references
                if self._scan_string_references(content, name):
                    continue

                yield self.create_finding(
                    message=f"Imported '{name}' is never used",
                    file_path=file_path,
                    line=line_num,
                    suggestion=f"Remove unused import: {name}",
                )

    def _check_unreferenced_definitions(self, file_path: Path, tree: ast.AST) -> Iterator[Finding]:
        """Check for functions/classes that are defined but never referenced."""
        # Skip this check for __init__.py files and test files
        if file_path.name == "__init__.py" or "test" in file_path.name.lower():
            return

        defined_names: Dict[str, int] = {}  # name -> line number
        referenced_names: Set[str] = set()

        # Collect all function and class definitions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # Skip private/dunder methods and main blocks
                if not node.name.startswith("_") and node.name != "main":
                    defined_names[node.name] = node.lineno

        # Collect all references within this file
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced_names.add(node.attr)

        # Get reachable names from call graph
        reachable_names = self._call_graph.get_reachable_from_exports()

        # Find unreferenced definitions
        for name, line_num in defined_names.items():
            # Skip if used locally
            if name in referenced_names:
                continue

            # Skip if exported
            if self._call_graph.is_exported(name, file_path):
                continue

            # Skip if reachable from any exported function
            if name in reachable_names:
                continue

            # Check for dynamic string references
            content = file_path.read_text(encoding="utf-8")
            if self._scan_string_references(content, name):
                continue

            yield self.create_finding(
                message=f"Function/class '{name}' is defined but never referenced",
                file_path=file_path,
                line=line_num,
                suggestion="Consider removing unused definition or adding to __all__",
            )

    def _scan_string_references(self, content: str, name: str) -> bool:
        """Check if name is referenced in strings (getattr, importlib, etc.)."""
        patterns = [
            rf"getattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"hasattr\([^,]+,\s*['\"]({re.escape(name)})['\"]",
            rf"importlib\.import_module\(['\"].*{re.escape(name)}.*['\"]",
            rf"__import__\(['\"].*{re.escape(name)}.*['\"]",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    def _get_module_name(self, file_path: Path, project_root: Path) -> str:
        """Convert file path to Python module name."""
        try:
            rel_path = file_path.relative_to(project_root)
            # Handle src layout
            if rel_path.parts[0] == "src" and len(rel_path.parts) > 1:
                rel_path = Path(*rel_path.parts[1:])

            if rel_path.name == "__init__.py":
                module_parts = rel_path.parent.parts
            else:
                module_parts = rel_path.with_suffix("").parts
            return ".".join(module_parts)
        except ValueError:
            return ""

    def _check_duplicate_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for duplicate code patterns that suggest redundancy."""
        lines = content.splitlines()

        # Check for duplicate validation result classes
        validation_classes = []
        for line_num, line in enumerate(lines, 1):
            if "ValidationResult" in line and "class " in line:
                validation_classes.append((line_num, line.strip()))

        if len(validation_classes) > 1:
            for line_num, _ in validation_classes:
                yield self.create_finding(
                    message="Validation result class found - may be duplicating plugin system",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider using plugin system's Finding class instead",
                )

        # Check for duplicate validation functions
        validation_functions = []
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith("def validate_") and not line.strip().startswith(
                "def validate("
            ):
                validation_functions.append((line_num, line.strip()))

        if len(validation_functions) > 0:
            for line_num, _ in validation_functions:
                yield self.create_finding(
                    message="Legacy validation function found - may duplicate BaseValidator",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Consider migrating to BaseValidator plugin system",
                )

    def _check_legacy_patterns(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Check for legacy code patterns that might be dead."""
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for manual console instantiation (except in ui.py which creates the shared instance)
            if "= Console()" in stripped and file_path.name not in ["utils.py", "ui.py"]:
                yield self.create_finding(
                    message="Manual Console instantiation - use shared utils instead",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with: from vibelint.ui import console",
                )
```

---
### File: src/vibelint/validators/project_wide/module_cohesion.py

```python
"""
Module cohesion validator for detecting scattered related modules.

Identifies when related modules should be grouped together in subpackages
based on naming patterns, import relationships, and functional cohesion.

vibelint/src/vibelint/validators/module_cohesion.py
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set

from ...validators.types import BaseValidator, Finding, Severity
from ...filesystem import find_project_root

logger = logging.getLogger(__name__)

__all__ = ["ModuleCohesionValidator"]


class ModuleCohesionValidator(BaseValidator):
    """Detects module organization issues and unjustified file existence."""

    rule_id = "MODULE-COHESION"
    name = "Module Cohesion & File Justification Analyzer"
    description = (
        "Identifies scattered related modules and files without clear purpose justification"
    )
    default_severity = Severity.INFO

    def __init__(self, severity=None, config=None):
        super().__init__(severity, config)
        # Common patterns that suggest related modules
        self.cohesion_patterns = [
            # Prefixed modules (llm_, api_, db_, etc.)
            r"^([a-z]+)_[a-z_]+\.py$",
            # Service/handler patterns
            r"^([a-z]+)_(service|handler|manager|client|adapter)\.py$",
            # Model/schema patterns
            r"^([a-z]+)_(model|schema|entity|dto)\.py$",
            # Test patterns
            r"^test_([a-z]+)_.*\.py$",
            # Utils patterns
            r"^([a-z]+)_(utils|helpers|tools)\.py$",
        ]

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Analyze project structure for module cohesion issues."""
        # Only analyze from project root to avoid duplicates
        project_root = find_project_root(file_path) or file_path.parent
        if file_path.parent != project_root / "src" / "vibelint":
            return

        # Get all Python files in the project
        src_dir = project_root / "src" / "vibelint"
        python_files = list(src_dir.glob("*.py"))

        if len(python_files) < 3:  # Need multiple files to detect patterns
            return

        # Group files by naming patterns
        pattern_groups = self._group_files_by_patterns(python_files)

        # Check for scattered modules that should be grouped
        for pattern, files in pattern_groups.items():
            if len(files) >= 2:  # 2+ files with same prefix suggest a module group
                yield from self._suggest_module_grouping(pattern, files, src_dir)

        # Check for functional cohesion based on imports
        yield from self._analyze_import_cohesion(python_files, src_dir)

        # Check for unjustified files
        yield from self._check_file_justification(project_root)

    def _group_files_by_patterns(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by common naming patterns."""
        import re

        groups = defaultdict(list)

        for file_path in files:
            filename = file_path.name

            # Skip special files
            if filename in ["__init__.py", "cli.py", "main.py"]:
                continue

            # Check each pattern
            for pattern in self.cohesion_patterns:
                match = re.match(pattern, filename)
                if match:
                    prefix = match.group(1)
                    groups[prefix].append(file_path)
                    break

        return groups

    def _suggest_module_grouping(
        self, prefix: str, files: List[Path], src_dir: Path
    ) -> Iterator[Finding]:
        """Suggest grouping related files into a submodule."""
        if len(files) < 2:
            return

        # Check if they're already in a submodule
        if any(len(f.relative_to(src_dir).parts) > 1 for f in files):
            return  # Already organized

        file_names = [f.name for f in files]

        yield Finding(
            rule_id=self.rule_id,
            message=f"Related modules with '{prefix}_' prefix should be grouped: {', '.join(file_names)}",
            file_path=files[0],  # Report on first file
            line=1,
            severity=self.default_severity,
            suggestion=f"Create 'src/vibelint/{prefix}/' subpackage and move related modules:\n"
            f"  mkdir src/vibelint/{prefix}/\n"
            f"  mv {' '.join(file_names)} src/vibelint/{prefix}/\n"
            f"  # Rename files to remove prefix: {prefix}_manager.py -> manager.py",
        )

    def _analyze_import_cohesion(self, files: List[Path], src_dir: Path) -> Iterator[Finding]:
        """Analyze import patterns to suggest functional grouping."""
        # Build import graph
        import_graph = defaultdict(set)

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)

                imports = self._extract_local_imports(tree, src_dir)
                module_name = file_path.stem
                import_graph[module_name].update(imports)

            except (UnicodeDecodeError, SyntaxError):
                continue

        # Find tightly coupled modules (import each other frequently)
        coupled_groups = self._find_coupled_modules(import_graph)

        for group in coupled_groups:
            if len(group) >= 3:  # Suggest grouping for 3+ tightly coupled modules
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Tightly coupled modules should be grouped: {', '.join(group)}",
                    file_path=src_dir / f"{list(group)[0]}.py",
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Consider grouping these functionally related modules into a subpackage",
                )

    def _extract_local_imports(self, tree: ast.AST, src_dir: Path) -> Set[str]:
        """Extract imports that reference local modules."""
        local_imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("vibelint"):
                    # Extract module name
                    parts = node.module.split(".")
                    if len(parts) >= 2:
                        local_imports.add(parts[-1])

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("vibelint"):
                        parts = alias.name.split(".")
                        if len(parts) >= 2:
                            local_imports.add(parts[-1])

        return local_imports

    def _find_coupled_modules(self, import_graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find groups of modules that frequently import each other."""
        # Simple clustering based on mutual imports
        coupled_groups = []
        processed = set()

        for module, imports in import_graph.items():
            if module in processed:
                continue

            # Find modules that import this one and vice versa
            mutual_imports = set()
            for imported in imports:
                if imported in import_graph and module in import_graph[imported]:
                    mutual_imports.add(imported)

            if mutual_imports:
                group = {module} | mutual_imports
                coupled_groups.append(group)
                processed.update(group)

        return coupled_groups

    def _check_file_justification(self, project_root: Path) -> Iterator[Finding]:
        """Check that every file has clear justification for existence."""
        # Files that are automatically justified
        auto_justified = {
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "README.md",
            "README.rst",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            ".gitignore",
            ".gitattributes",
            "Dockerfile",
            "docker-compose.yml",
            "Makefile",
            "tox.ini",
            ".pre-commit-config.yaml",
            "__init__.py",
            "conftest.py",
            "pytest.ini",
        }

        # Check all files in project
        for file_path in project_root.rglob("*"):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue

            # Skip auto-justified files
            if file_path.name in auto_justified:
                continue

            # Skip files in build/cache directories
            if any(
                part
                in [
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "build",
                    "dist",
                    ".tox",
                    ".mypy_cache",
                    "node_modules",
                ]
                for part in file_path.parts
            ):
                continue

            # Check file justification based on type
            # Note: Python file docstring checks are handled by DOCSTRING-MISSING validator
            if file_path.suffix in [".md", ".rst", ".txt"]:
                yield from self._check_documentation_justification(file_path)
            elif file_path.suffix in [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]:
                yield from self._check_config_file_justification(file_path)
            elif file_path.suffix in [".sh", ".bat", ".ps1"]:
                yield from self._check_script_justification(file_path)
            else:
                # Unknown file type - requires explicit justification
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Unknown file type without clear justification: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comment/documentation explaining file purpose or remove if unnecessary",
                )


    def _check_documentation_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that documentation files have meaningful content."""
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            # Check for minimal content
            if len(content) < 50:  # Very short docs
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Documentation file is too short to be useful: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add meaningful content or remove if unnecessary",
                )

            # Check for placeholder/template content
            placeholder_indicators = ["TODO", "FIXME", "PLACEHOLDER", "Lorem ipsum", "Example text"]
            if any(indicator in content for indicator in placeholder_indicators):
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Documentation file contains placeholder content: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Replace placeholder content with actual documentation or remove file",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Documentation file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )

    def _check_config_file_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that config files have clear purpose."""
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            # Check for empty config files
            if len(content) < 10:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Config file is nearly empty: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add configuration content or remove if unnecessary",
                )

            # Check for comments explaining purpose
            has_explanatory_comments = any(
                line.strip().startswith("#") and len(line.strip()) > 10
                for line in content.splitlines()[:5]  # Check first 5 lines
            )

            if not has_explanatory_comments and file_path.suffix in [".json", ".yaml", ".yml"]:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Config file lacks explanatory comments: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comments explaining configuration purpose and usage",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Config file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )

    def _check_script_justification(self, file_path: Path) -> Iterator[Finding]:
        """Check that script files have clear purpose."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for shebang and comments
            lines = content.splitlines()

            # Look for explanatory comments in first 10 lines
            has_explanation = any(
                line.strip().startswith("#") and len(line.strip()) > 20 for line in lines[:10]
            )

            if not has_explanation:
                yield Finding(
                    rule_id=self.rule_id,
                    message=f"Script file lacks explanatory comments: {file_path.name}",
                    file_path=file_path,
                    line=1,
                    severity=Severity.INFO,
                    suggestion="Add comments explaining script purpose, usage, and requirements",
                )

        except UnicodeDecodeError:
            yield Finding(
                rule_id=self.rule_id,
                message=f"Script file has encoding issues: {file_path.name}",
                file_path=file_path,
                line=1,
                severity=Severity.WARN,
                suggestion="Fix encoding issues or remove if unnecessary",
            )
```

---
### File: src/vibelint/validators/project_wide/namespace_collisions.py

```python
"""
Namespace representation & collision detection for Python code.

vibelint/src/vibelint/namespace.py
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path

from ...config import Config
from ...discovery import discover_files
from ...filesystem import find_project_root, get_relative_path

__all__ = [
    "CollisionType",
    "NamespaceCollision",
    "NamespaceNode",
    "detect_hard_collisions",
    "detect_global_definition_collisions",
    "detect_local_export_collisions",
    "build_namespace_tree",
    "get_namespace_collisions_str",
]

logger = logging.getLogger(__name__)


class CollisionType:
    """
    Enum-like class for collision types.

    vibelint/src/vibelint/namespace.py
    """

    HARD = "hard"
    LOCAL_SOFT = "local_soft"
    GLOBAL_SOFT = "global_soft"


class NamespaceCollision:
    """
    Represents a collision between two or more same-named entities.

    vibelint/src/vibelint/namespace.py
    """

    def __init__(
        self,
        name: str,
        collision_type: str,
        paths: list[Path],
        linenos: list[int | None] | None = None,
    ) -> None:
        """
        Initializes a NamespaceCollision instance.

        Args:
        name: The name of the colliding entity.
        collision_type: The type of collision (HARD, LOCAL_SOFT, GLOBAL_SOFT).
        paths: A list of Path objects for all files involved in the collision.
        linenos: An optional list of line numbers corresponding to each path.

        vibelint/src/vibelint/namespace.py
        """

        if not paths:
            raise ValueError("At least one path must be provided for a collision.")

        self.name = name
        self.collision_type = collision_type

        self.paths = sorted(list(set(paths)), key=str)

        self.linenos = (
            linenos if linenos and len(linenos) == len(self.paths) else [None] * len(self.paths)
        )

        self.path1: Path = self.paths[0]
        self.path2: Path = self.paths[1] if len(self.paths) > 1 else self.paths[0]
        self.lineno1: int | None = self.linenos[0] if self.linenos else None
        self.lineno2: int | None = self.linenos[1] if len(self.linenos) > 1 else self.lineno1

        self.definition_paths: list[Path] = (
            self.paths
            if self.collision_type in [CollisionType.GLOBAL_SOFT, CollisionType.LOCAL_SOFT]
            else []
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for debugging.

        vibelint/src/vibelint/namespace.py
        """

        return (
            f"NamespaceCollision(name='{self.name}', type='{self.collision_type}', "
            f"paths={self.paths}, linenos={self.linenos})"
        )

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the collision.

        vibelint/src/vibelint/namespace.py
        """

        proj_root = find_project_root(Path(".").resolve())
        base_path = proj_root if proj_root else Path(".")

        paths_str_list = []
        for i, p in enumerate(self.paths):
            loc = f":{self.linenos[i]}" if self.linenos and self.linenos[i] is not None else ""
            try:
                paths_str_list.append(f"{get_relative_path(p, base_path)}{loc}")
            except ValueError:
                paths_str_list.append(f"{p}{loc}")
        paths_str = ", ".join(paths_str_list)

        if self.collision_type == CollisionType.HARD:
            if len(self.paths) == 2 and self.paths[0] == self.paths[1]:

                line_info = ""
                if self.lineno1 is not None and self.lineno2 is not None:
                    line_info = f" (lines ~{self.lineno1} and ~{self.lineno2})"
                elif self.lineno1 is not None:
                    line_info = f" (line ~{self.lineno1})"

                return (
                    f"{self.collision_type.upper()} Collision: Duplicate definition/import of '{self.name}' in "
                    f"{paths_str_list[0]}{line_info}"
                )
            else:
                return f"{self.collision_type.upper()} Collision: Name '{self.name}' used by conflicting entities in: {paths_str}"
        elif self.collision_type == CollisionType.LOCAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' exported via __all__ in multiple sibling modules: {paths_str}"
        elif self.collision_type == CollisionType.GLOBAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' defined in multiple modules: {paths_str}"
        else:
            return f"Unknown Collision: '{self.name}' involving paths: {paths_str}"


def detect_hard_collisions(
    paths: list[Path],
    config: Config,
) -> list[NamespaceCollision]:
    """
    Detect HARD collisions: member vs. submodule, or duplicate definitions within a file.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected HARD NamespaceCollision objects.

    vibelint/src/vibelint/namespace.py
    """

    root_node, intra_file_collisions = build_namespace_tree(paths, config)

    inter_file_collisions = root_node.get_hard_collisions()

    all_collisions = intra_file_collisions + inter_file_collisions
    for c in all_collisions:
        c.collision_type = CollisionType.HARD
    return all_collisions


def detect_global_definition_collisions(
    paths: list[Path],
    config: Config,
) -> list[NamespaceCollision]:
    """
    Detect GLOBAL SOFT collisions: the same name defined/assigned at the top level
    in multiple different modules across the project.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected GLOBAL_SOFT NamespaceCollision objects.

    vibelint/src/vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)

    definition_collisions = root_node.detect_global_definition_collisions()

    return definition_collisions


def detect_local_export_collisions(
    paths: list[Path],
    config: Config,
) -> list[NamespaceCollision]:
    """
    Detect LOCAL SOFT collisions: the same name exported via __all__ by multiple
    sibling modules within the same package.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected LOCAL_SOFT NamespaceCollision objects.

    vibelint/src/vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)
    collisions: list[NamespaceCollision] = []
    root_node.find_local_export_collisions(collisions)
    return collisions


def get_namespace_collisions_str(
    paths: list[Path],
    config: Config,
    console=None,
) -> str:
    """
    Return a string representation of all collision types for quick debugging.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.
    console: Optional console object (unused).

    Returns:
    A string summarizing all detected collisions.

    vibelint/src/vibelint/namespace.py
    """

    from io import StringIO

    buf = StringIO()

    hard_collisions = detect_hard_collisions(paths, config)
    global_soft_collisions = detect_global_definition_collisions(paths, config)
    local_soft_collisions = detect_local_export_collisions(paths, config)

    proj_root = find_project_root(Path(".").resolve())
    base_path = proj_root if proj_root else Path(".")

    if hard_collisions:
        buf.write("Hard Collisions:\n")
        for c in sorted(hard_collisions, key=lambda x: (x.name, str(x.paths[0]))):
            buf.write(f"- {str(c)}\n")

    if local_soft_collisions:
        buf.write("\nLocal Soft Collisions (__all__):\n")

        grouped = defaultdict(list)
        for c in local_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': exported by {paths_str}\n")

    if global_soft_collisions:
        buf.write("\nGlobal Soft Collisions (Definitions):\n")

        grouped = defaultdict(list)
        for c in global_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': defined in {paths_str}\n")

    return buf.getvalue()


class NamespaceNode:
    """
    A node in the "module" hierarchy (like package/subpackage, or file-level).
    Holds child nodes and top-level members (functions/classes).

    vibelint/src/vibelint/namespace.py
    """

    def __init__(self, name: str, path: Path | None = None, is_package: bool = False) -> None:
        """
        Initializes a NamespaceNode.

        Args:
        name: The name of the node (e.g., module name, package name).
        path: The filesystem path associated with this node (optional).
        is_package: True if this node represents a package (directory).

        vibelint/src/vibelint/namespace.py
        """

        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: dict[str, NamespaceNode] = {}

        self.members: dict[str, tuple[Path, int | None]] = {}

        self.member_collisions: list[NamespaceCollision] = []

        self.exported_names: list[str] | None = None

    def set_exported_names(self, names: list[str]) -> None:
        """
        Sets the list of names found in __all__.

        vibelint/src/vibelint/namespace.py
        """

        self.exported_names = names

    def add_child(self, name: str, path: Path, is_package: bool = False) -> "NamespaceNode":
        """
        Adds a child node, creating if necessary.

        vibelint/src/vibelint/namespace.py
        """

        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)

        elif path:

            if not (self.children[name].is_package and not is_package):
                self.children[name].path = path
            self.children[name].is_package = is_package or self.children[name].is_package
        return self.children[name]

    def get_hard_collisions(self) -> list[NamespaceCollision]:
        """
        Detect HARD collisions recursively: members vs. child modules.

        vibelint/src/vibelint/namespace.py
        """

        collisions: list[NamespaceCollision] = []

        member_names_with_info = {}
        if self.is_package and self.path:
            init_path = (self.path / "__init__.py").resolve()
            member_names_with_info = {
                name: (def_path, lineno)
                for name, (def_path, lineno) in self.members.items()
                if def_path.resolve() == init_path
            }

        child_names = set(self.children.keys())
        common_names = set(member_names_with_info.keys()).intersection(child_names)

        for name in common_names:

            member_def_path, member_lineno = member_names_with_info.get(name, (None, None))
            cnode = self.children[name]
            child_path = cnode.path

            if member_def_path and child_path:

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[member_def_path, child_path],
                        linenos=[member_lineno, None],
                    )
                )

        for cnode in self.children.values():
            collisions.extend(cnode.get_hard_collisions())
        return collisions

    def collect_defined_members(self, all_dict: dict[str, list[tuple[Path, int | None]]]) -> None:
        """
        Recursively collects defined members (path, lineno) for global definition collision check.

        vibelint/src/vibelint/namespace.py
        """

        if self.path and self.members:

            for mname, (mpath, mlineno) in self.members.items():
                all_dict.setdefault(mname, []).append((mpath, mlineno))

        for cnode in self.children.values():
            cnode.collect_defined_members(all_dict)

    def detect_global_definition_collisions(self) -> list[NamespaceCollision]:
        """
        Detects GLOBAL SOFT collisions across the whole tree starting from this node.

        vibelint/src/vibelint/namespace.py
        """

        all_defined_members: dict[str, list[tuple[Path, int | None]]] = defaultdict(list)
        self.collect_defined_members(all_defined_members)

        collisions: list[NamespaceCollision] = []
        for name, path_lineno_list in all_defined_members.items():

            unique_paths_map: dict[Path, int | None] = {}
            for path, lineno in path_lineno_list:
                resolved_p = path.resolve()

                if resolved_p not in unique_paths_map:
                    unique_paths_map[resolved_p] = lineno

            if len(unique_paths_map) > 1:

                sorted_paths = sorted(unique_paths_map.keys(), key=str)

                sorted_linenos = [unique_paths_map[p] for p in sorted_paths]

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.GLOBAL_SOFT,
                        paths=sorted_paths,
                        linenos=sorted_linenos,
                    )
                )
        return collisions

    def find_local_export_collisions(self, collisions_list: list[NamespaceCollision]) -> None:
        """
        Recursively finds LOCAL SOFT collisions (__all__) within packages.

        Args:
        collisions_list: A list to append found collisions to.

        vibelint/src/vibelint/namespace.py
        """

        if self.is_package:
            exports_in_package: dict[str, list[Path]] = defaultdict(list)

            if self.path and self.path.is_dir() and self.exported_names:

                init_path = (self.path / "__init__.py").resolve()

                if init_path.exists() and any(
                    p.resolve() == init_path for p, _ in self.members.values()
                ):
                    for name in self.exported_names:
                        exports_in_package[name].append(init_path)

            for child in self.children.values():

                if (
                    child.path
                    and child.path.is_file()
                    and not child.is_package
                    and child.name != "__init__"
                    and child.exported_names
                ):
                    for name in child.exported_names:
                        exports_in_package[name].append(child.path.resolve())

            for name, paths in exports_in_package.items():
                unique_paths = sorted(list(set(paths)), key=str)
                if len(unique_paths) > 1:
                    collisions_list.append(
                        NamespaceCollision(
                            name=name,
                            collision_type=CollisionType.LOCAL_SOFT,
                            paths=unique_paths,
                            linenos=[None for _ in unique_paths],
                        )
                    )

        for child in self.children.values():
            if child.is_package:
                child.find_local_export_collisions(collisions_list)

    def __str__(self) -> str:
        """
        Provides a string representation of the node and its subtree, including members.
        Uses a revised formatting approach for better clarity relative to project root.

        vibelint/src/vibelint/namespace.py
        """

        lines = []

        proj_root = find_project_root(Path(".").resolve())
        base_path_for_display = proj_root if proj_root else Path(".")

        def build_tree_lines(
            node: "NamespaceNode", prefix: str = "", base: Path = Path(".")
        ) -> list[str]:
            """
            Docstring for function 'build_tree_lines'.

            vibelint/src/vibelint/namespace.py
            """

            child_items = sorted(node.children.items())

            direct_members = []
            if node.path and node.members:

                expected_def_path = None
                node_path_resolved = node.path.resolve()
                if node.is_package and node_path_resolved.is_dir():
                    expected_def_path = (node_path_resolved / "__init__.py").resolve()
                elif node_path_resolved.is_file():
                    expected_def_path = node_path_resolved

                if expected_def_path:
                    direct_members = sorted(
                        [
                            name
                            for name, (def_path, _) in node.members.items()
                            if def_path.resolve() == expected_def_path
                        ]
                    )

            all_items = child_items + [(name, "member") for name in direct_members]
            total_items = len(all_items)

            for i, (name, item) in enumerate(all_items):
                is_last = i == total_items - 1
                connector = "└── " if is_last else "├── "
                next_level_prefix = prefix + ("    " if is_last else "│   ")

                if item == "member":

                    lines.append(f"{prefix}{connector}{name} (member)")
                else:

                    child: NamespaceNode = item
                    child_path_str = ""
                    indicator = ""
                    if child.path:
                        try:
                            rel_p = get_relative_path(child.path, base)

                            if child.is_package:
                                indicator = " (P)"
                            elif child.name == "__init__":
                                indicator = " (I)"
                            else:
                                indicator = " (M)"
                            child_path_str = f"  [{rel_p}{indicator}]"
                        except ValueError:
                            indicator = (
                                " (P)"
                                if child.is_package
                                else (" (I)" if child.name == "__init__" else " (M)")
                            )
                            child_path_str = f"  [{child.path.resolve()}{indicator}]"
                    else:
                        child_path_str = "  [No Path]"

                    lines.append(f"{prefix}{connector}{name}{child_path_str}")

                    if child.children or (
                        child.members
                        and any(
                            m_path.resolve() == (child.path.resolve() if child.path else None)
                            for _, (m_path, _) in child.members.items()
                        )
                    ):
                        lines.extend(build_tree_lines(child, next_level_prefix, base))

            return lines

        root_path_str = ""
        root_indicator = ""

        if self.path:
            root_path_resolved = self.path.resolve()
            try:

                rel_p = get_relative_path(root_path_resolved, base_path_for_display.parent)

                if rel_p == Path("."):
                    rel_p = Path(self.name)

                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{rel_p}{root_indicator}]"
            except ValueError:
                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{root_path_resolved}{root_indicator}]"
        else:
            root_path_str = "  [No Path]"

        lines.append(f"{self.name}{root_path_str}")
        lines.extend(build_tree_lines(self, prefix="", base=base_path_for_display))
        return "\n".join(lines)


def _extract_module_members(
    file_path: Path,
) -> tuple[dict[str, tuple[Path, int | None]], list[NamespaceCollision], list[str] | None]:
    """
    Parses a Python file and extracts top-level member definitions/assignments,
    intra-file hard collisions, and the contents of __all__ if present.

    Returns:
    - A dictionary mapping defined/assigned names to a tuple of (file path, line number).
    - A list of intra-file hard collisions (NamespaceCollision objects).
    - A list of names in __all__, or None if __all__ is not found or invalid.

    vibelint/src/vibelint/namespace.py
    """

    try:
        source = file_path.read_text(encoding="utf-8")

        tree = ast.parse(source, filename=str(file_path))
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Could not read file {file_path} for namespace analysis: {e}")
        return {}, [], None
    except (SyntaxError, ValueError) as e:
        logger.warning(f"Could not parse file {file_path} for namespace analysis: {e}")
        return {}, [], None

    defined_members_map: dict[str, tuple[Path, int | None]] = {}
    collisions: list[NamespaceCollision] = []
    exported_names: list[str] | None = None

    defined_names_nodes: dict[str, ast.AST] = {}

    for node in tree.body:
        current_node = node
        name: str | None = None
        is_definition = False
        is_all_assignment = False
        lineno = getattr(current_node, "lineno", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            is_definition = True
        elif isinstance(node, ast.Assign):

            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                is_all_assignment = True

                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exported_names = []
                    for elt in node.value.elts:

                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported_names.append(elt.value)

                if "__all__" not in defined_names_nodes:
                    defined_names_nodes["__all__"] = current_node
                else:
                    first_node = defined_names_nodes["__all__"]
                    collisions.append(
                        NamespaceCollision(
                            name="__all__",
                            collision_type=CollisionType.HARD,
                            paths=[file_path, file_path],
                            linenos=[getattr(first_node, "lineno", None), lineno],
                        )
                    )

            else:

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        is_definition = True

                        if name:
                            if name in defined_names_nodes:

                                first_node = defined_names_nodes[name]
                                collisions.append(
                                    NamespaceCollision(
                                        name=name,
                                        collision_type=CollisionType.HARD,
                                        paths=[file_path, file_path],
                                        linenos=[
                                            getattr(first_node, "lineno", None),
                                            lineno,
                                        ],
                                    )
                                )
                            else:

                                defined_names_nodes[name] = current_node
                                defined_members_map[name] = (
                                    file_path,
                                    lineno,
                                )
                            name = None

        if name and is_definition and not is_all_assignment:
            if name in defined_names_nodes:

                first_node = defined_names_nodes[name]
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[file_path, file_path],
                        linenos=[getattr(first_node, "lineno", None), lineno],
                    )
                )
            else:

                defined_names_nodes[name] = current_node
                defined_members_map[name] = (file_path, lineno)

    return defined_members_map, collisions, exported_names


def build_namespace_tree(
    paths: list[Path], config: Config
) -> tuple[NamespaceNode, list[NamespaceCollision]]:
    """
    Builds the namespace tree, collects intra-file collisions, and stores members/__all__.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns a tuple: (root_node, all_intra_file_collisions)

    vibelint/src/vibelint/namespace.py
    """

    project_root_found = config.project_root or find_project_root(
        paths[0].resolve() if paths else Path(".")
    )
    if not project_root_found:

        project_root_found = Path(".")
        root_node_name = "root"
        logger.warning(
            "Could not determine project root. Using '.' as root for namespace analysis."
        )
    else:
        root_node_name = project_root_found.name

    root = NamespaceNode(root_node_name, path=project_root_found.resolve(), is_package=True)
    root_path_for_rel = project_root_found.resolve()
    all_intra_file_collisions: list[NamespaceCollision] = []

    python_files = [
        f
        for f in discover_files(
            paths,
            config,
        )
        if f.suffix == ".py"
    ]

    if not python_files:
        logger.info("No Python files found for namespace analysis based on configuration.")
        return root, all_intra_file_collisions

    for f in python_files:
        try:

            rel_path = f.relative_to(root_path_for_rel)
            rel_parts = list(rel_path.parts)
        except ValueError:

            rel_parts = [f.name]
            logger.warning(
                f"File {f} is outside the determined project root {root_path_for_rel}. Adding directly under root."
            )

        current = root

        for i, part in enumerate(rel_parts[:-1]):

            dir_path = root_path_for_rel.joinpath(*rel_parts[: i + 1])
            current = current.add_child(part, dir_path, is_package=True)

        file_name = rel_parts[-1]
        mod_name = Path(file_name).stem
        file_abs_path = f

        members, intra_collisions, exported_names = _extract_module_members(file_abs_path)
        all_intra_file_collisions.extend(intra_collisions)

        if mod_name == "__init__":

            package_node = current
            package_node.is_package = True
            package_node.path = file_abs_path.parent

            for m_name, m_info in members.items():
                if m_name not in package_node.members:
                    package_node.members[m_name] = m_info

            if exported_names is not None:
                package_node.set_exported_names(exported_names)

        else:

            module_node = current.add_child(mod_name, file_abs_path, is_package=False)
            module_node.members = members
            if exported_names is not None:
                module_node.set_exported_names(exported_names)
            module_node.member_collisions.extend(intra_collisions)

    return root, all_intra_file_collisions
```

---
### File: src/vibelint/validators/project_wide/namespace_report.py

```python
"""
Report generation functionality for vibelint.

vibelint/src/vibelint/validators/namespace_report.py
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TextIO

from ...filesystem import get_relative_path
from ..config import Config
from .namespace_collisions import NamespaceCollision, NamespaceNode

__all__ = ["write_report_content"]
logger = logging.getLogger(__name__)


def _get_files_in_namespace_order(
    node: NamespaceNode, collected_files: set[Path], project_root: Path
) -> None:
    """
    Recursively collects file paths from the namespace tree in DFS order,
    including __init__.py files for packages. Populates the collected_files set.

    Args:
        node: The current NamespaceNode.
        collected_files: A set to store the absolute paths of collected files.
        project_root: The project root path for checking containment.

    vibelint/src/vibelint/validators/namespace_report.py
    """

    if node.is_package and node.path and node.path.is_dir():
        try:

            node.path.relative_to(project_root)
            init_file = node.path / "__init__.py"

            if init_file.is_file() and init_file not in collected_files:

                init_file.relative_to(project_root)
                logger.debug(f"Report: Adding package init file: {init_file}")
                collected_files.add(init_file)
        except ValueError:
            logger.warning(f"Report: Skipping package node outside project root: {node.path}")
        except (OSError, TypeError) as e:
            logger.error(f"Report: Error checking package init file for {node.path}: {e}")

    for child_name in sorted(node.children.keys()):
        child_node = node.children[child_name]

        if child_node.path and child_node.path.is_file() and not child_node.is_package:
            try:

                child_node.path.relative_to(project_root)
                if child_node.path not in collected_files:
                    logger.debug(f"Report: Adding module file: {child_node.path}")
                    collected_files.add(child_node.path)
            except ValueError:
                logger.warning(
                    f"Report: Skipping module file outside project root: {child_node.path}"
                )
            except (OSError, TypeError) as e:
                logger.error(f"Report: Error checking module file {child_node.path}: {e}")

        _get_files_in_namespace_order(child_node, collected_files, project_root)

    if not node.children and node.path and node.path.is_file():
        try:
            node.path.relative_to(project_root)
            if node.path not in collected_files:
                logger.debug(f"Report: Adding root file node: {node.path}")
                collected_files.add(node.path)
        except ValueError:
            logger.warning(f"Report: Skipping root file node outside project root: {node.path}")
        except (OSError, TypeError) as e:
            logger.error(f"Report: Error checking root file node {node.path}: {e}")


def write_report_content(
    f: TextIO,
    project_root: Path,
    target_paths: list[Path],
    findings: list,  # List of Finding objects from plugin system
    hard_coll: list[NamespaceCollision],
    soft_coll: list[NamespaceCollision],
    root_node: NamespaceNode,
    config: Config,
) -> None:
    """
    Writes the comprehensive markdown report content to the given file handle.

    Args:
    f: The text file handle to write the report to.
    project_root: The root directory of the project.
    target_paths: List of paths that were analyzed.
    findings: List of Finding objects from the plugin validation phase.
    hard_coll: List of hard NamespaceCollision objects.
    soft_coll: List of definition/export (soft) NamespaceCollision objects.
    root_node: The root NamespaceNode of the project structure.
    config: Configuration object.

    vibelint/src/vibelint/validators/namespace_report.py
    """

    package_name = project_root.name if project_root else "Unknown"

    f.write("# vibelint Report\n\n")
    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    f.write(f"**Project:** {package_name}\n")

    f.write(f"**Project Root:** `{str(project_root.resolve())}`\n\n")
    f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")

    f.write("## Table of Contents\n\n")
    f.write("1. [Summary](#summary)\n")
    f.write("2. [Linting Results](#linting-results)\n")
    f.write("3. [Namespace Structure](#namespace-structure)\n")
    f.write("4. [Namespace Collisions](#namespace-collisions)\n")
    f.write("5. [File Contents](#file-contents)\n\n")

    f.write("## Summary\n\n")
    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")

    # Get unique files from findings
    analyzed_files = set(f.file_path for f in findings)
    files_analyzed_count = len(analyzed_files)
    f.write(f"| Files analyzed | {files_analyzed_count} |\n")

    # Count findings by severity
    from .plugin_system import Severity

    error_findings = [f for f in findings if f.severity == Severity.BLOCK]
    warn_findings = [f for f in findings if f.severity == Severity.WARN]

    f.write(f"| Findings with errors | {len(error_findings)} |\n")
    f.write(f"| Findings with warnings | {len(warn_findings)} |\n")
    f.write(f"| Hard namespace collisions | {len(hard_coll)} |\n")
    total_soft_collisions = len(soft_coll)
    f.write(f"| Definition/Export namespace collisions | {total_soft_collisions} |\n\n")

    f.write("## Linting Results\n\n")

    if not findings:
        f.write("*No validation issues found.*\n\n")
    else:
        # Group findings by file for better reporting
        files_with_findings = defaultdict(list)
        for finding in findings:
            files_with_findings[finding.file_path].append(finding)

        f.write("| File | Rule | Severity | Message |\n")
        f.write("|------|------|----------|---------|\n")

        for file_path in sorted(files_with_findings.keys(), key=str):
            file_findings = files_with_findings[file_path]
            try:
                rel_path = get_relative_path(file_path.resolve(), project_root.resolve())
            except ValueError:
                rel_path = file_path

            for finding in sorted(file_findings, key=lambda f: f.line):
                location = f":{finding.line}" if finding.line > 0 else ""
                f.write(
                    f"| `{rel_path}{location}` | `{finding.rule_id}` | {finding.severity.value} | {finding.message} |\n"
                )
        f.write("\n")

    f.write("## Namespace Structure\n\n")
    f.write("```\n")
    try:

        tree_str = root_node.__str__()
        f.write(tree_str)
    except (ValueError, TypeError) as e:
        logger.error(f"Report: Error generating namespace tree string: {e}")
        f.write(f"[Error generating namespace tree: {e}]\n")
    f.write("\n```\n\n")

    f.write("## Namespace Collisions\n\n")
    f.write("### Hard Collisions\n\n")
    if not hard_coll:
        f.write("*No hard collisions detected.*\n\n")
    else:
        f.write("These collisions can break Python imports or indicate duplicate definitions:\n\n")
        f.write("| Name | Path 1 | Path 2 | Details |\n")
        f.write("|------|--------|--------|---------|\n")
        for collision in sorted(hard_coll, key=lambda c: (c.name, str(c.path1))):
            try:
                p1_rel = (
                    get_relative_path(collision.path1.resolve(), project_root.resolve())
                    if collision.path1
                    else "N/A"
                )
                p2_rel = (
                    get_relative_path(collision.path2.resolve(), project_root.resolve())
                    if collision.path2
                    else "N/A"
                )
            except ValueError:
                p1_rel = collision.path1 or "N/A"
                p2_rel = collision.path2 or "N/A"
            loc1 = f":{collision.lineno1}" if collision.lineno1 else ""
            loc2 = f":{collision.lineno2}" if collision.lineno2 else ""
            details = (
                "Intra-file duplicate" if str(p1_rel) == str(p2_rel) else "Module/Member clash"
            )
            f.write(f"| `{collision.name}` | `{p1_rel}{loc1}` | `{p2_rel}{loc2}` | {details} |\n")
        f.write("\n")

    f.write("### Definition & Export Collisions (Soft)\n\n")
    if not soft_coll:
        f.write("*No definition or export collisions detected.*\n\n")
    else:
        f.write(
            "These names are defined/exported in multiple files, which may confuse humans and LLMs:\n\n"
        )
        f.write("| Name | Type | Files Involved |\n")
        f.write("|------|------|----------------|\n")
        grouped_soft = defaultdict(lambda: {"paths": set(), "types": set()})
        for collision in soft_coll:
            all_paths = collision.definition_paths or [collision.path1, collision.path2]
            grouped_soft[collision.name]["paths"].update(p for p in all_paths if p)
            grouped_soft[collision.name]["types"].add(collision.collision_type)

        for name, data in sorted(grouped_soft.items()):
            paths_str_list = []
            for p in sorted(list(data["paths"]), key=str):
                try:
                    paths_str_list.append(
                        f"`{get_relative_path(p.resolve(), project_root.resolve())}`"
                    )
                except ValueError:
                    paths_str_list.append(f"`{p}`")
            type_str = (
                " & ".join(sorted([t.replace("_soft", "").upper() for t in data["types"]]))
                or "Unknown"
            )
            f.write(f"| `{name}` | {type_str} | {', '.join(paths_str_list)} |\n")
        f.write("\n")

    f.write("## File Contents\n\n")
    f.write("Files are ordered alphabetically by path.\n\n")

    collected_files_set: set[Path] = set()
    try:
        _get_files_in_namespace_order(root_node, collected_files_set, project_root.resolve())

        python_files_abs = sorted(list(collected_files_set), key=lambda p: str(p))
        logger.info(f"Report: Found {len(python_files_abs)} files for content section.")
    except (ValueError, TypeError, OSError) as e:
        logger.error(f"Report: Error collecting files for content section: {e}", exc_info=True)
        python_files_abs = []

    if not python_files_abs:
        f.write("*No Python files found in the namespace tree to display.*\n\n")
    else:
        for abs_file_path in python_files_abs:

            if abs_file_path and abs_file_path.is_file():
                try:

                    rel_path = get_relative_path(abs_file_path, project_root.resolve())
                    f.write(f"### {rel_path}\n\n")

                    try:
                        lang = "python"
                        content = abs_file_path.read_text(encoding="utf-8", errors="ignore")
                        f.write(f"```{lang}\n")
                        f.write(content)

                        if not content.endswith("\n"):
                            f.write("\n")
                        f.write("```\n\n")
                    except (OSError, UnicodeDecodeError) as read_e:
                        logger.warning(
                            f"Report: Error reading file content for {rel_path}: {read_e}"
                        )
                        f.write(f"*Error reading file content: {read_e}*\n\n")

                except ValueError:

                    logger.warning(
                        f"Report: Skipping file outside project root in content section: {abs_file_path}"
                    )
                    f.write(f"### {abs_file_path} (Outside Project Root)\n\n")
                    f.write("*Skipping content as file is outside the detected project root.*\n\n")
                except (OSError, ValueError, TypeError) as e_outer:
                    logger.error(
                        f"Report: Error processing file entry for {abs_file_path}: {e_outer}",
                        exc_info=True,
                    )
                    f.write(f"### Error Processing Entry for {abs_file_path}\n\n")
                    f.write(f"*An unexpected error occurred: {e_outer}*\n\n")
            elif abs_file_path:
                logger.warning(
                    f"Report: Skipping non-file path found during content writing: {abs_file_path}"
                )
                f.write(f"### {abs_file_path} (Not a File)\n\n")
                f.write("*Skipping entry as it is not a file.*\n\n")

            f.write("---\n\n")
```

---
### File: src/vibelint/validators/registry.py

```python
"""
Validator registry and discovery system.

Provides centralized registration and discovery of validators with
automatic loading from entry points and modular organization.

Responsibility: Validator discovery and registration only.
Validation logic belongs in individual validator modules.

vibelint/src/vibelint/validators/registry.py
"""

import importlib.metadata
import logging
from typing import Dict, List, Optional, Type

from .types import BaseValidator

logger = logging.getLogger(__name__)


class ValidatorRegistry:
    """Registry for managing available validators."""

    def __init__(self):
        self._validators: Dict[str, Type[BaseValidator]] = {}
        self._loaded = False

    def register_validator(self, validator_class: Type[BaseValidator]) -> None:
        """Register a validator class."""
        if not issubclass(validator_class, BaseValidator):
            raise ValueError(f"Validator {validator_class} must inherit from BaseValidator")

        rule_id = validator_class.rule_id
        if rule_id in self._validators:
            logger.warning(f"Overriding existing validator: {rule_id}")

        self._validators[rule_id] = validator_class
        logger.debug(f"Registered validator: {rule_id}")

    def get_validator(self, rule_id: str) -> Optional[Type[BaseValidator]]:
        """Get a validator by rule ID."""
        if not self._loaded:
            self._load_all_validators()
        return self._validators.get(rule_id)

    def get_all_validators(self) -> Dict[str, Type[BaseValidator]]:
        """Get all registered validators."""
        if not self._loaded:
            self._load_all_validators()
        return self._validators.copy()

    def get_validators_by_category(self, category: str) -> Dict[str, Type[BaseValidator]]:
        """Get validators by category (single_file, project_wide, architecture)."""
        if not self._loaded:
            self._load_all_validators()

        filtered = {}
        for rule_id, validator_class in self._validators.items():
            # Determine category from module path
            module_path = validator_class.__module__
            if f".{category}." in module_path:
                filtered[rule_id] = validator_class
        return filtered

    def list_rule_ids(self) -> List[str]:
        """List all available rule IDs."""
        if not self._loaded:
            self._load_all_validators()
        return list(self._validators.keys())

    def _load_all_validators(self) -> None:
        """Load all validators from entry points and built-ins."""
        if self._loaded:
            return

        # Load from entry points
        self._load_entry_point_validators()

        # Load built-in validators
        self._load_builtin_validators()

        self._loaded = True
        logger.info(f"Loaded {len(self._validators)} validators")

    def _load_entry_point_validators(self) -> None:
        """Load validators from entry points."""
        try:
            for entry_point in importlib.metadata.entry_points(group="vibelint.validators"):
                try:
                    validator_class = entry_point.load()
                    self.register_validator(validator_class)
                except Exception as e:
                    logger.warning(f"Failed to load validator {entry_point.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load entry point validators: {e}")

    def _load_builtin_validators(self) -> None:
        """Load built-in validators from modules."""
        builtin_modules = [
            # Single-file validators
            "vibelint.validators.single_file.docstring",
            "vibelint.validators.single_file.emoji",
            "vibelint.validators.single_file.exports",
            "vibelint.validators.single_file.logger_names",
            "vibelint.validators.single_file.print_statements",
            "vibelint.validators.single_file.typing_quality",
            "vibelint.validators.single_file.self_validation",
            # Project-wide validators
            "vibelint.validators.project_wide.dead_code",
            "vibelint.validators.project_wide.namespace_collisions",
            "vibelint.validators.project_wide.code_smells",
            "vibelint.validators.project_wide.module_cohesion",
            # Architecture validators
            "vibelint.validators.architecture.basic_patterns",
        ]

        for module_name in builtin_modules:
            try:
                module = importlib.import_module(module_name)

                # Look for get_validators function
                if hasattr(module, "get_validators"):
                    validators = module.get_validators()
                    for validator_class in validators:
                        self.register_validator(validator_class)

                # Look for individual validator classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseValidator)
                        and attr != BaseValidator
                        and hasattr(attr, "rule_id")
                    ):
                        self.register_validator(attr)

            except ImportError as e:
                logger.debug(f"Could not import builtin validator module {module_name}: {e}")
            except Exception as e:
                logger.warning(f"Error loading validators from {module_name}: {e}")


# Global registry instance
validator_registry = ValidatorRegistry()


# Convenience functions
def register_validator(validator_class: Type[BaseValidator]) -> None:
    """Register a validator class with the global registry."""
    validator_registry.register_validator(validator_class)


def get_validator(rule_id: str) -> Optional[Type[BaseValidator]]:
    """Get a validator by rule ID from the global registry."""
    return validator_registry.get_validator(rule_id)


def get_all_validators() -> Dict[str, Type[BaseValidator]]:
    """Get all validators from the global registry."""
    return validator_registry.get_all_validators()
```

---
### File: src/vibelint/validators/single_file/__init__.py

```python
"""
Single-file validators for vibelint.

These validators analyze individual Python files in isolation.
They should not require knowledge of other files in the project.

vibelint/src/vibelint/validators/single_file/__init__.py
"""

from pathlib import Path
from typing import Iterator, List

from ...validators.types import BaseValidator, Finding


class SingleFileValidator(BaseValidator):
    """Base class for validators that analyze individual files."""

    def validate_file(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate a single file in isolation.

        Args:
            file_path: Path to the file being validated
            content: File content as string
            config: Configuration object

        Yields:
            Finding objects for any issues found
        """
        # Default implementation delegates to validate method
        yield from self.validate(file_path, content, config)

    def requires_project_context(self) -> bool:
        """Single-file validators do not require project context."""
        return False


def get_single_file_validators() -> List[str]:
    """Get list of single-file validator names."""
    return [
        "DOCSTRING-MISSING",
        "DOCSTRING-PATH-REFERENCE",
        "PRINT-STATEMENT",
        "EMOJI-IN-STRING",
        "TYPING-POOR-PRACTICE",
        "EXPORTS-MISSING-ALL",
        "EXPORTS-MISSING-ALL-INIT",
    ]
```

---
### File: src/vibelint/validators/single_file/dict_get_fallback.py

```python
"""
Validator for detecting .get() with fallback values on typed dictionaries.

In strictly-typed code, .get() with defaults hides missing keys. This is appropriate
for ETL/JSON parsing, but not for internal typed structures which should fail fast.
"""

import ast
import logging
from pathlib import Path
from typing import Iterator

from vibelint.validators import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class DictGetFallbackValidator(BaseValidator):
    """Detects .get() with fallback on typed dictionaries that should use direct access."""

    rule_id = "DICT-GET-FALLBACK"
    description = "Detect .get() with fallback that should use direct key access"

    def __init__(self, config=None, severity=None):
        super().__init__(config)
        self.config = config or {}
        self.severity = severity or Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate Python file for .get() antipatterns.

        Args:
            file_path: Path to the Python file
            content: File content as string
            config: Optional configuration

        Yields:
            Finding objects for .get() antipatterns found
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_dict_get_with_fallback(node):
                    yield from self._check_get_pattern(node, file_path)

    def _is_dict_get_with_fallback(self, node: ast.Call) -> bool:
        """Check if this is a .get() call with a fallback value."""
        # Must be a method call
        if not isinstance(node.func, ast.Attribute):
            return False

        # Method name must be 'get'
        if node.func.attr != "get":
            return False

        # Must have 2 arguments (key, default) or 1 with keywords
        if len(node.args) == 2:
            return True
        if len(node.args) == 1 and any(kw.arg == "default" for kw in node.keywords):
            return True

        return False

    def _check_get_pattern(self, node: ast.Call, file_path: Path) -> Iterator[Finding]:
        """Check if this .get() call is an antipattern."""
        # Get the key being accessed
        key = self._extract_key(node)
        fallback = self._extract_fallback(node)

        message = f"Using .get() with fallback hides missing keys - use direct access for typed structures"

        if key:
            message = f"dict.get('{key}', {fallback}) hides missing keys - use direct access dict['{key}'] for typed structures"

        suggestion = self._suggest_direct_access(key, fallback)

        yield Finding(
            rule_id=self.rule_id,
            message=message,
            file_path=file_path,
            line=node.lineno,
            column=node.col_offset,
            severity=self.severity,
            suggestion=suggestion
        )

    def _extract_key(self, node: ast.Call) -> str:
        """Extract the key from .get() call."""
        if not node.args:
            return None

        key_node = node.args[0]
        if isinstance(key_node, ast.Constant):
            return str(key_node.value)
        return None

    def _extract_fallback(self, node: ast.Call) -> str:
        """Extract the fallback value as a string."""
        # Check args
        if len(node.args) >= 2:
            fallback = node.args[1]
            return ast.unparse(fallback) if hasattr(ast, "unparse") else "..."

        # Check keywords
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.unparse(kw.value) if hasattr(ast, "unparse") else "..."

        return "None"

    def _suggest_direct_access(self, key: str, fallback: str) -> str:
        """Suggest direct access replacement."""
        if not key:
            return "Consider: Use direct key access if dictionary is typed/validated"

        suggestion = f"""Consider replacing with direct access:

# If the key is required (fail fast):
value = data["{key}"]

# If the key is truly optional, use explicit check:
value = data.get("{key}")  # Returns None if missing
if "{key}" not in data:
    value = {fallback}

# Best: Use typed structures (dataclass/NamedTuple) instead of dict"""

        return suggestion

    def can_fix(self) -> bool:
        """Returns True if this validator can automatically fix issues."""
        return False  # Requires context to determine if key is required
```

---
### File: src/vibelint/validators/single_file/docstring.py

```python
"""
Docstring validator using BaseValidator plugin system.

Checks for missing docstrings and proper path references in modules,
classes, and functions.

vibelint/src/vibelint/validators/docstring.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...ast_utils import parse_or_none
from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["MissingDocstringValidator", "DocstringPathValidator"]


class MissingDocstringValidator(BaseValidator):
    """Validator for missing docstrings."""

    rule_id = "DOCSTRING-MISSING"
    name = "Missing Docstring Checker"
    description = "Checks for missing docstrings in modules, classes, and functions"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing docstrings based on configuration."""
        tree = parse_or_none(content, file_path)
        if tree is None:
            return

        # Get docstring configuration
        docstring_config = (config and config.get("docstring", {})) or {}
        require_module = docstring_config.get("require_module_docstrings", True)
        require_class = docstring_config.get("require_class_docstrings", True)
        require_function = docstring_config.get("require_function_docstrings", False)
        include_private = docstring_config.get("include_private_functions", False)

        # Check module docstring
        if require_module and not ast.get_docstring(tree):
            yield self.create_finding(
                message="Module is missing docstring",
                file_path=file_path,
                line=1,
                suggestion="Add a module-level docstring explaining the module's purpose",
            )

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check class docstrings
                if require_class and (include_private or not node.name.startswith("_")):
                    if not ast.get_docstring(node):
                        yield self.create_finding(
                            message=f"Class '{node.name}' is missing docstring",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion=f"Add docstring to {node.name} explaining its purpose",
                        )
            elif isinstance(node, ast.FunctionDef):
                # Check function docstrings
                if require_function and (include_private or not node.name.startswith("_")):
                    if not ast.get_docstring(node):
                        yield self.create_finding(
                            message=f"Function '{node.name}' is missing docstring",
                            file_path=file_path,
                            line=node.lineno,
                            suggestion=f"Add docstring to {node.name}() explaining its purpose",
                        )


class DocstringPathValidator(BaseValidator):
    """Validator for missing path references in docstrings."""

    rule_id = "DOCSTRING-PATH-REFERENCE"
    name = "Missing Path Reference in Docstring"
    description = "Checks that docstrings end with the expected relative file path reference"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing path references in docstrings based on configuration."""
        # Get docstring configuration
        docstring_config = (config and config.get("docstring", {})) or {}
        require_path_references = docstring_config.get("require_path_references", False)

        # Skip validation if path references are not required
        if not require_path_references:
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Get expected path reference based on format configuration
        path_format = docstring_config.get("path_reference_format", "relative")
        expected_path = self._get_expected_path(file_path, path_format)

        # Check module docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring and not module_docstring.strip().endswith(expected_path):
            yield self.create_finding(
                message=f"Module docstring missing/incorrect path reference (expected '{expected_path}')",
                file_path=file_path,
                line=1,
                suggestion=f"Add '{expected_path}' at the end of the module docstring for LLM context",
            )

        # Check function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring and not docstring.strip().endswith(expected_path):
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    yield self.create_finding(
                        message=f"{node_type} '{node.name}' docstring missing/incorrect path reference (expected '{expected_path}')",
                        file_path=file_path,
                        line=node.lineno,
                        suggestion=f"Add '{expected_path}' at the end of the docstring for LLM context",
                    )

    def _get_expected_path(self, file_path: Path, path_format: str) -> str:
        """Get expected path reference based on format configuration."""
        if path_format == "absolute":
            return str(file_path)
        elif path_format == "module_path":
            # Convert to Python module path (e.g., vibelint.validators.docstring)
            parts = file_path.parts
            if "src" in parts:
                src_idx = parts.index("src")
                module_parts = parts[src_idx + 1 :]
            else:
                module_parts = parts

            # Remove .py extension and convert to module path
            if module_parts and module_parts[-1].endswith(".py"):
                module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

            return ".".join(module_parts)
        else:  # relative format (default)
            # Get relative path, removing project root and src/ prefix
            relative_path = str(file_path)
            try:
                # Try to find project root by looking for common markers
                current = file_path.parent
                while current.parent != current:
                    if any(
                        (current / marker).exists()
                        for marker in ["pyproject.toml", "setup.py", ".git"]
                    ):
                        relative_path = str(file_path.relative_to(current))
                        break
                    current = current.parent
            except ValueError:
                pass

            # Remove src/ prefix if present
            if relative_path.startswith("src/"):
                relative_path = relative_path[4:]

            return relative_path
```

---
### File: src/vibelint/validators/single_file/emoji.py

```python
"""
Emoji usage validator using BaseValidator plugin system.

Detects emoji usage that can cause encoding issues, reduce readability,
and create compatibility problems across different terminals and systems.

vibelint/src/vibelint/validators/emoji.py
"""

import re
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["EmojiUsageValidator"]


class EmojiUsageValidator(BaseValidator):
    """Detects emoji usage that can cause encoding issues."""

    rule_id = "EMOJI-IN-STRING"
    name = "Emoji Usage Detector"
    description = "Detects emojis that can cause MCP and Windows shell issues"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for emoji usage in code."""
        # Comprehensive emoji regex pattern - matches ALL possible emojis
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F"  # Emoticons
            r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
            r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
            r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
            r"\U0001F700-\U0001F77F"  # Alchemical Symbols
            r"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            r"\U0001FA00-\U0001FA6F"  # Chess Symbols
            r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            r"\U00002600-\U000026FF"  # Miscellaneous Symbols
            r"\U00002700-\U000027BF"  # Dingbats
            r"\U0000FE00-\U0000FE0F"  # Variation Selectors
            r"\U0001F018-\U0001F270"  # Various Asian characters
            r"\U0000238C-\U00002454"  # Misc technical (fixed with leading zeros)
            r"\U000020D0-\U000020FF"  # Combining Diacritical Marks for Symbols (fixed)
            r"]+"
        )

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            emoji_matches = emoji_pattern.findall(line)
            if emoji_matches:
                emojis_found = "".join(emoji_matches)

                # Check if emoji is in code vs strings/comments
                if self._is_emoji_in_code_context(line):
                    # More severe for emojis in actual code
                    yield self.create_finding(
                        message=f"Emoji in code: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emoji in code with ASCII alternatives immediately",
                    )
                else:
                    # Less severe for emojis in strings/comments
                    yield self.create_finding(
                        message=f"Emoji usage detected: {emojis_found}",
                        file_path=file_path,
                        line=line_num,
                        suggestion="Replace emojis with text descriptions to avoid encoding issues in MCP and Windows shells",
                    )

    def can_fix(self, finding: "Finding") -> bool:
        """Check if this finding can be automatically fixed."""
        return finding.rule_id == self.rule_id

    def apply_fix(self, content: str, finding: "Finding") -> str:
        """Automatically remove emojis from content."""
        lines = content.splitlines(True)  # Keep line endings
        if finding.line <= len(lines):
            line = lines[finding.line - 1]

            # Comprehensive emoji pattern for removal
            emoji_pattern = re.compile(
                r"[\U0001F600-\U0001F64F"  # Emoticons
                r"\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
                r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
                r"\U0001F700-\U0001F77F"  # Alchemical Symbols
                r"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                r"\U0001FA00-\U0001FA6F"  # Chess Symbols
                r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                r"\U00002600-\U000026FF"  # Miscellaneous Symbols
                r"\U00002700-\U000027BF"  # Dingbats
                r"\U0000FE00-\U0000FE0F"  # Variation Selectors
                r"\U0001F018-\U0001F270"  # Various Asian characters
                r"\U0000238C-\U00002454"  # Misc technical (fixed with leading zeros)
                r"\U000020D0-\U000020FF"  # Combining Diacritical Marks for Symbols (fixed)
                r"]+"
            )

            # Remove emojis from the line
            fixed_line = emoji_pattern.sub("", line)

            # Clean up any double spaces that might result, but preserve indentation and newlines
            # Extract leading whitespace (indentation)
            leading_whitespace = ""
            content_start = 0
            for char in fixed_line:
                if char in [' ', '\t']:
                    leading_whitespace += char
                    content_start += 1
                else:
                    break

            # Split into content and line ending
            if fixed_line.endswith('\n'):
                content = fixed_line[content_start:-1]  # Remove leading whitespace and newline
                line_ending = '\n'
            elif fixed_line.endswith('\r\n'):
                content = fixed_line[content_start:-2]  # Remove leading whitespace and CRLF
                line_ending = '\r\n'
            else:
                content = fixed_line[content_start:]  # Remove leading whitespace
                line_ending = ''

            # Clean up only multiple consecutive spaces in content
            content = re.sub(r" {2,}", " ", content)  # Multiple spaces to single space
            content = content.rstrip()  # Remove trailing spaces only

            # Reconstruct the line with original indentation
            fixed_line = leading_whitespace + content + line_ending

            lines[finding.line - 1] = fixed_line

        return "".join(lines)

    def _is_emoji_in_code_context(self, line: str) -> bool:
        """
        Check if emoji appears to be in code rather than in strings or comments.

        This is a heuristic - emojis in strings/comments are less problematic
        than emojis used as identifiers or in code structure.
        """
        # Remove string literals and comments to see if emoji remains
        line_without_strings = re.sub(r'["\'].*?["\']', "", line)
        line_without_comments = re.sub(r"#.*$", "", line_without_strings)

        # If emoji still exists after removing strings/comments, it's likely in code
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]"
        )
        return bool(emoji_pattern.search(line_without_comments))
```

---
### File: src/vibelint/validators/single_file/exports.py

```python
"""
__all__ exports validator using BaseValidator plugin system.

Checks for presence and correct format of __all__ definitions in Python modules.

vibelint/src/vibelint/validators/exports.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["MissingAllValidator"]


class MissingAllValidator(BaseValidator):
    """Validator for missing __all__ definitions."""

    rule_id = "EXPORTS-MISSING-ALL"
    name = "Missing __all__ Checker"
    description = "Checks for missing __all__ definitions in modules"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing __all__ definition."""
        # Skip if it's __init__.py or private module
        if file_path.name.startswith("_"):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            yield self.create_finding(
                message="SyntaxError parsing file during __all__ validation",
                file_path=file_path,
                line=1,
                suggestion="Fix Python syntax errors in the file",
            )
            return

        # Look for __all__ definition
        has_all = False
        has_exports = False

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        # Check if __all__ is properly formatted
                        if not self._is_valid_all_format(node.value):
                            yield self.create_finding(
                                message="__all__ is not assigned a list or tuple value",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion='Ensure __all__ = ["item1", "item2"] or __all__ = ("item1", "item2")',
                            )
                        break
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith(
                "_"
            ):
                has_exports = True

        if has_exports and not has_all:
            yield self.create_finding(
                message="Module has public functions/classes but no __all__ definition",
                file_path=file_path,
                line=1,
                suggestion="Add __all__ = [...] to explicitly define public API",
            )

    def _is_valid_all_format(self, node: ast.AST) -> bool:
        """Check if __all__ assignment value is a valid list or tuple."""
        if isinstance(node, (ast.List, ast.Tuple)):
            # Check that all elements are strings
            return all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.elts
            )
        return False


class InitAllValidator(BaseValidator):
    """Validator for missing __all__ in __init__.py files."""

    rule_id = "EXPORTS-MISSING-ALL-INIT"
    name = "Missing __all__ in __init__.py"
    description = "__init__.py file is missing __all__ definition (optional based on config)"
    default_severity = Severity.INFO

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Check for missing __all__ in __init__.py files."""
        # Only check __init__.py files
        if file_path.name != "__init__.py":
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        # Look for __all__ definition
        has_all = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        break

        if not has_all:
            yield self.create_finding(
                message="__init__.py file is missing __all__ definition",
                file_path=file_path,
                line=1,
                suggestion="Add __all__ = [...] to control package imports",
            )
```

---
### File: src/vibelint/validators/single_file/logger_names.py

```python
"""
Logger name validator using BaseValidator plugin system.

Detects hardcoded logger names that should use __name__ instead
for proper module hierarchy and maintainability.

vibelint/src/vibelint/validators/logger_names.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["LoggerNameValidator"]


class LoggerNameValidator(BaseValidator):
    """Validator for detecting hardcoded logger names."""

    rule_id = "LOGGER-NAME"
    name = "Logger Name Checker"
    description = "Detects get_logger() calls with hardcoded strings instead of __name__"
    default_severity = Severity.BLOCK

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate a single file for hardcoded logger names."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        class LoggerNameVisitor(ast.NodeVisitor):
            """Visitor class that traverses an AST to collect all logger names used in the code."""

            def __init__(self, validator):
                """Initialize logger name visitor."""
                self.validator = validator
                self.logger_names = []
                self.findings = []

            def visit_Call(self, node: ast.Call):
                # Check for get_logger() calls
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "get_logger"
                    and len(node.args) == 1
                ):

                    arg = node.args[0]

                    # If argument is a string literal (not __name__)
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        self.findings.append(
                            Finding(
                                rule_id=self.validator.rule_id,
                                severity=self.validator.severity,
                                message=f"Use __name__ instead of hardcoded string '{arg.value}' for logger name",
                                file_path=file_path,
                                line=node.lineno,
                                suggestion="get_logger(__name__)",
                            )
                        )

                # Also check for attribute access like logging.getLogger()
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ["getLogger", "get_logger"]
                    and len(node.args) == 1
                ):

                    arg = node.args[0]

                    # If argument is a string literal (not __name__)
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        # Skip if this is in logging infrastructure getting specific loggers
                        if "logging.py" in str(file_path) and "kaia." in arg.value:
                            pass  # This is logging infrastructure, skip
                        else:
                            self.findings.append(
                                Finding(
                                    rule_id=self.validator.rule_id,
                                    severity=Severity.WARN,  # Less severe for standard logging
                                    message=f"Consider using __name__ instead of hardcoded string '{arg.value}' for logger name",
                                    file_path=file_path,
                                    line=node.lineno,
                                    suggestion="Use __name__ for consistent logger hierarchy",
                                )
                            )

                self.generic_visit(node)

        visitor = LoggerNameVisitor(self)
        visitor.visit(tree)
        yield from visitor.findings
```

---
### File: src/vibelint/validators/single_file/print_statements.py

```python
"""
Print statement validator using BaseValidator plugin system.

Detects print statements that should be replaced with proper logging
for better maintainability, configurability, and production readiness.

vibelint/src/vibelint/validators/print_statements.py
"""

import ast
import fnmatch
import re
from pathlib import Path
from typing import Iterator

from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["PrintStatementValidator"]


class PrintStatementValidator(BaseValidator):
    """Validator for detecting print statements."""

    rule_id = "PRINT-STATEMENT"
    name = "Print Statement Checker"
    description = "Detects print() calls that should be replaced with logging"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate print statement usage in a Python file."""
        # Check if file should be excluded based on configuration
        if self._should_exclude_file(file_path):
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        visitor = _PrintVisitor()
        visitor.visit(tree)

        # Split content into lines for suppression comment checking
        lines = content.split("\n")

        for line_num, context, print_content in visitor.print_calls:
            # Check for suppression comments on the same line
            if self._has_suppression_comment(lines, line_num):
                continue

            # Check if this looks like legitimate CLI output
            if self._is_legitimate_cli_print(print_content, context, content, line_num):
                continue

            message = (
                f"Print statement found{context}. Replace with logging for better maintainability."
            )
            suggestion = "Use logger.info(), logger.debug(), or logger.error() instead"

            yield self.create_finding(
                message=message, file_path=file_path, line=line_num, suggestion=suggestion
            )

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from print statement validation."""
        # Get exclude patterns from configuration
        print_config = self.config.get("print_validation", {})
        exclude_globs = print_config.get(
            "exclude_globs",
            [
                # Default patterns if no configuration is provided
                "test_*.py",
                "*_test.py",
                "conftest.py",
                "tests/**/*.py",
                "cli.py",
                "main.py",
                "__main__.py",
                "*_cli.py",
                "*_cmd.py",
            ],
        )

        # Check if file matches any exclude pattern
        for pattern in exclude_globs:
            # Check against file name
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

            # Check against relative path pattern
            relative_path = str(file_path).replace("\\", "/")  # Normalize path separators
            if fnmatch.fnmatch(relative_path, pattern):
                return True

            # Check against path from parent directories
            for parent in file_path.parents:
                parent_relative = str(file_path.relative_to(parent)).replace("\\", "/")
                if fnmatch.fnmatch(parent_relative, pattern):
                    return True

        return False

    def _has_suppression_comment(self, lines: list[str], line_num: int) -> bool:
        """Check if the line has a suppression comment for print statements.

        Supports:
        - # vibelint: stdout  - Explicit stdout communication marker
        - # vibelint: ignore  - General vibelint suppression
        - # noqa: print       - Specific print suppression
        - # noqa              - General linting suppression
        """
        # Line numbers in AST are 1-indexed
        if line_num <= 0 or line_num > len(lines):
            return False

        line = lines[line_num - 1]

        # Check for suppression patterns in comments
        suppression_patterns = [
            r"#\s*vibelint:\s*stdout",  # Explicit stdout marker
            r"#\s*vibelint:\s*ignore",  # General vibelint ignore
            r"#\s*noqa:\s*print",  # Specific print suppression
            r"#\s*noqa(?:\s|$)",  # General noqa
            r"#\s*type:\s*ignore",  # Type ignore (sometimes used for prints)
            r"#\s*pragma:\s*no\s*cover",  # Coverage pragma
        ]

        for pattern in suppression_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _is_legitimate_cli_print(
        self, print_content: str, context: str, file_content: str, line_num: int
    ) -> bool:
        """Check if a print statement appears to be legitimate CLI output."""
        # Patterns that suggest legitimate CLI usage
        cli_indicators = [
            # UI symbols and formatting
            r"[[EMOJI][TIP][EMOJI][ALERT]⭐[SUCCESS][ROCKET]]",  # Emoji indicators for user interface
            r"^[-=]{3,}",  # Headers/separators (----, ====)
            r"^\s*\*{2,}",  # Emphasis markers (***, etc.)
            r"^\s*#{2,}",  # Section headers (##, ###)
            # CLI instruction patterns
            r"(run|execute|visit|go to|open)",
            r"(http://|https://)",  # URLs
            r"(localhost|127\.0\.0\.1)",  # Local server addresses
            r"port\s+\d+",  # Port numbers
            # Status/progress indicators
            r"(starting|completed|finished|ready)",
            r"(success|error|warning|info).*:",
            r"^\s*\[.*\]",  # [INFO], [ERROR], etc.
            # Calibration/setup specific
            r"(calibration|configuration|setup)",
            r"(device|microphone|audio)",
            r"(instruction|step \d+)",
        ]

        # Function names that suggest CLI interface
        cli_function_names = [
            "show_",
            "display_",
            "print_",
            "output_",
            "start_",
            "run_",
            "main",
            "cli",
            "calibrat",
            "setup",
            "config",
            "instruction",
            "help",
            "usage",
        ]

        # Check print content against CLI patterns
        if print_content:
            for pattern in cli_indicators:
                if re.search(pattern, print_content, re.IGNORECASE | re.MULTILINE):
                    return True

        # Check if function name suggests CLI usage
        if context:
            func_name = context.replace(" in function ", "").lower()
            for cli_pattern in cli_function_names:
                if cli_pattern in func_name:
                    return True

        # Check file context - look for CLI-related imports or patterns
        file_lines = file_content.split("\n")

        # Look around the print statement for context clues
        start_line = max(0, line_num - 5)
        end_line = min(len(file_lines), line_num + 3)
        surrounding_context = "\n".join(file_lines[start_line:end_line])

        # Check for CLI-related context around the print
        context_patterns = [
            r"def\s+(show|display|print|output|start|run|main|cli)",
            r"(server|port|url|http)",
            r"(calibration|setup|config)",
            r"(instruction|help|usage)",
            r"input\s*\(",  # User input nearby
            r"argparse",  # Command line arguments
        ]

        for pattern in context_patterns:
            if re.search(pattern, surrounding_context, re.IGNORECASE):
                return True

        return False


class _PrintVisitor(ast.NodeVisitor):
    """AST visitor to detect print statements."""

    def __init__(self):
        self.print_calls = []
        self.current_function = None

    def visit_Call(self, node):
        """Visit Call nodes to detect print() function calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            context = f" in function {self.current_function}" if self.current_function else ""

            # Extract print content for analysis
            print_content = ""
            if node.args:
                try:
                    # Try to extract string literals from print arguments
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            print_content += arg.value + " "
                        elif isinstance(arg, ast.JoinedStr):  # f-strings
                            for value in arg.values:
                                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                    print_content += value.value
                except (AttributeError, TypeError):
                    # If we can't parse the content, just use empty string
                    pass

            self.print_calls.append((node.lineno, context, print_content.strip()))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit FunctionDef nodes to track current function context for print detection."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
```

---
### File: src/vibelint/validators/single_file/relative_imports.py

```python
"""
Validator for detecting and suggesting fixes for relative imports.

Relative imports can cause issues in larger codebases and make modules less portable.
This validator detects relative imports and suggests absolute import alternatives.
"""

import ast
import logging
from pathlib import Path
from typing import Iterator, List, Optional

from vibelint.validators import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class RelativeImportValidator(BaseValidator):
    """Validates and suggests fixes for relative imports."""

    rule_id = "RELATIVE-IMPORTS"
    description = "Detect relative imports and suggest absolute alternatives"

    def __init__(self, config=None, severity=None):
        super().__init__(config)
        self.config = config or {}
        self.severity = severity

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """
        Validate Python file for relative imports.

        Args:
            file_path: Path to the Python file
            content: File content as string
            config: Optional configuration

        Yields:
            Finding objects for relative imports found
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            # If file doesn't parse, skip validation
            logger.debug(f"Syntax error in {file_path}: {e}")
            return

        # Extract package structure information
        package_info = self._analyze_package_structure(file_path)

        # Find all import nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                findings = self._check_import_node(node, file_path, package_info)
                yield from findings

    def _analyze_package_structure(self, file_path: Path) -> dict:
        """Analyze the package structure to understand how to convert relative imports."""
        package_info = {
            "file_path": file_path,
            "package_parts": [],
            "is_package": False,
            "project_root": None
        }

        # Find project root (look for pyproject.toml, setup.py, or .git)
        current = file_path.parent
        while current != current.parent:
            if any((current / name).exists() for name in ["pyproject.toml", "setup.py", ".git"]):
                package_info["project_root"] = current
                break
            current = current.parent

        if not package_info["project_root"]:
            package_info["project_root"] = file_path.parent

        # Determine package path relative to project root
        try:
            relative_path = file_path.relative_to(package_info["project_root"])
            package_info["package_parts"] = list(relative_path.parent.parts)

            # Remove common non-package directories
            if package_info["package_parts"] and package_info["package_parts"][0] in ["src", "lib"]:
                package_info["package_parts"] = package_info["package_parts"][1:]

            package_info["is_package"] = file_path.name == "__init__.py"
        except ValueError:
            # File is not under project root
            package_info["package_parts"] = []

        return package_info

    def _check_import_node(self, node: ast.AST, file_path: Path, package_info: dict) -> List[Finding]:
        """Check an import node for relative import issues."""
        findings = []

        if isinstance(node, ast.ImportFrom):
            # Check for relative imports (those starting with . or ..)
            if node.level > 0:  # Relative import detected
                absolute_suggestion = self._suggest_absolute_import(node, package_info)

                if absolute_suggestion:
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        message=f"Relative import detected: {'.' * node.level}{node.module or ''}",
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity=Severity.WARN,
                        suggestion=f"Replace with absolute import: {absolute_suggestion}"
                    ))

        return findings

    def _suggest_absolute_import(self, node: ast.ImportFrom, package_info: dict) -> Optional[str]:
        """Suggest an absolute import to replace the relative import."""
        if not package_info["package_parts"]:
            return None

        # Calculate the absolute module path
        current_package = package_info["package_parts"].copy()

        # Handle different levels of relative imports
        if node.level == 1:  # from .module import something
            # Same package level
            target_package = current_package
        elif node.level > 1:  # from ..module import something
            # Go up the package hierarchy
            levels_up = node.level - 1
            if levels_up >= len(current_package):
                return None  # Can't go up that many levels
            target_package = current_package[:-levels_up] if levels_up > 0 else current_package
        else:
            return None

        # Build the absolute import
        if node.module:
            absolute_module = ".".join(target_package + [node.module])
        else:
            absolute_module = ".".join(target_package)

        # Format the import statement
        if node.names:
            if len(node.names) == 1 and node.names[0].name == "*":
                return f"from {absolute_module} import *"
            else:
                imports = []
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"{alias.name} as {alias.asname}")
                    else:
                        imports.append(alias.name)
                return f"from {absolute_module} import {', '.join(imports)}"
        else:
            return f"import {absolute_module}"

    def can_fix(self) -> bool:
        """Returns True if this validator can automatically fix issues."""
        return True

    def fix_finding(self, file_path: Path, content: str, finding: Finding) -> str:
        """
        Automatically fix a relative import finding.

        Args:
            file_path: Path to the file
            content: Current file content
            finding: The finding to fix

        Returns:
            Updated file content with the fix applied
        """
        if "Replace with absolute import:" not in finding.suggestion:
            return content

        # Extract the suggested absolute import
        suggestion_parts = finding.suggestion.split("Replace with absolute import: ", 1)
        if len(suggestion_parts) != 2:
            return content

        absolute_import = suggestion_parts[1]

        # Find and replace the relative import on the specific line
        lines = content.split('\n')
        if finding.line <= len(lines):
            line_idx = finding.line - 1  # Convert to 0-based index
            original_line = lines[line_idx]

            # Try to identify and replace the relative import
            # This is a simplified replacement - in practice, you might want more sophisticated parsing
            if original_line.strip().startswith('from .'):
                # Find the indentation and replace the import
                indent = len(original_line) - len(original_line.lstrip())
                lines[line_idx] = ' ' * indent + absolute_import

                return '\n'.join(lines)

        return content
```

---
### File: src/vibelint/validators/single_file/self_validation.py

```python
"""
Self-validation hooks for vibelint.

This module implements validation hooks that ensure vibelint follows
its own coding standards and architectural principles.

Key principles enforced:
- Single-file validators must not access other files
- Project-wide validators must implement validate_project()
- No emoji in code or comments (project rule)
- Proper validator categorization
- Adherence to killeraiagent project standards

vibelint/src/vibelint/self_validation.py
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ...validators.types import BaseValidator, Finding, Severity

logger = logging.getLogger(__name__)


class SelfValidationHook:
    """
    Hook that validates vibelint's own code against its standards.

    This runs automatically when vibelint analyzes its own codebase
    to ensure we follow our own rules.
    """

    def __init__(self):
        self.project_root = None
        self.violations_found = []

    def should_apply_self_validation(self, file_path: Path) -> bool:
        """Check if self-validation should apply to this file."""
        try:
            # Check if we're analyzing vibelint's own code
            path_str = str(file_path.absolute())
            return (
                "vibelint" in path_str
                and "/src/vibelint/" in path_str
                and file_path.suffix == ".py"
            )
        except Exception:
            return False

    def validate_single_file_validator(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate that single-file validators don't violate isolation."""
        if not self._is_validator_file(file_path):
            return

        # Check if this is a single-file validator
        if self._is_single_file_validator(content):
            # Check for project context violations
            violations = self._check_project_context_violations(content)
            for violation in violations:
                yield Finding(
                    file_path=file_path,
                    line=violation["line"],
                    message=f"Single-file validator violates isolation: {violation['issue']}",
                    rule_id="VIBELINT-SINGLE-FILE-ISOLATION",
                    severity=Severity.BLOCK,
                    suggestion="Single-file validators should not access other files or require project context",
                )

    def validate_project_standards(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate adherence to killeraiagent project standards."""
        if not self.should_apply_self_validation(file_path):
            return

        # Check for emoji violations (project rule: no emoji)
        emoji_violations = self._check_emoji_violations(content)
        for line_num, line_content in emoji_violations:
            yield Finding(
                file_path=file_path,
                line=line_num,
                message="Code contains emoji characters (violates project standards)",
                rule_id="VIBELINT-NO-EMOJI",
                severity=Severity.BLOCK,
                suggestion="Remove emoji characters from code and comments",
            )

        # Check for proper absolute path usage
        path_violations = self._check_path_violations(content)
        for line_num, issue in path_violations:
            yield Finding(
                file_path=file_path,
                line=line_num,
                message=f"Path usage issue: {issue}",
                rule_id="VIBELINT-ABSOLUTE-PATHS",
                severity=Severity.WARN,
                suggestion="Use absolute paths for file operations",
            )

    def validate_validator_categorization(self, file_path: Path, content: str) -> Iterator[Finding]:
        """Validate that validators are properly categorized."""
        if not self._is_validator_file(file_path):
            return

        # Check if validator is in correct directory
        is_single_file = self._is_single_file_validator(content)
        is_project_wide = self._is_project_wide_validator(content)

        path_str = str(file_path)
        in_single_file_dir = "/single_file/" in path_str
        in_project_wide_dir = "/project_wide/" in path_str
        in_architecture_dir = "/architecture/" in path_str

        if is_single_file and not in_single_file_dir and not in_architecture_dir:
            yield Finding(
                file_path=file_path,
                line=1,
                message="Single-file validator should be in validators/single_file/ directory",
                rule_id="VIBELINT-VALIDATOR-ORGANIZATION",
                severity=Severity.WARN,
                suggestion="Move to validators/single_file/ or implement project-wide validation",
            )

        if is_project_wide and not in_project_wide_dir and not in_architecture_dir:
            yield Finding(
                file_path=file_path,
                line=1,
                message="Project-wide validator should be in validators/project_wide/ directory",
                rule_id="VIBELINT-VALIDATOR-ORGANIZATION",
                severity=Severity.WARN,
                suggestion="Move to validators/project_wide/ or implement single-file validation",
            )

    def _is_validator_file(self, file_path: Path) -> bool:
        """Check if file is a validator."""
        return (
            "/validators/" in str(file_path)
            and file_path.name != "__init__.py"
            and file_path.suffix == ".py"
        )

    def _is_single_file_validator(self, content: str) -> bool:
        """Check if validator is designed for single-file analysis."""
        patterns = [
            r"class\s+\w+Validator.*BaseValidator",
            r"def validate\(self, file_path.*content.*\)",
            r"requires_project_context.*False",
        ]

        project_patterns = [
            r"validate_project",
            r"project_files",
            r"requires_project_context.*True",
        ]

        has_single_file_indicators = any(re.search(pattern, content) for pattern in patterns)
        has_project_indicators = any(re.search(pattern, content) for pattern in project_patterns)

        return has_single_file_indicators and not has_project_indicators

    def _is_project_wide_validator(self, content: str) -> bool:
        """Check if validator is designed for project-wide analysis."""
        patterns = [r"validate_project", r"project_files.*Dict", r"requires_project_context.*True"]
        return any(re.search(pattern, content) for pattern in patterns)

    def requires_project_context(self) -> bool:
        """Self-validation does not require project context."""
        return False

    def _check_project_context_violations(self, content: str) -> List[Dict[str, Any]]:
        """Check for violations of single-file validator isolation."""
        violations = []
        lines = content.split("\n")

        violation_patterns = [
            (r"import.*discovery", "Should not import discovery module"),
            (r"import.*project_map", "Should not import project mapping"),
            (r"glob\.glob", "Should not use glob to find other files"),
            (r"os\.walk", "Should not walk directory tree"),
            (r"Path.*glob", "Should not glob for other files"),
            (r"open\(.*\.py", "Should not open other Python files"),
        ]

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("#"):  # Skip comments
                continue

            for pattern, issue in violation_patterns:
                if re.search(pattern, line):
                    violations.append({"line": line_num, "issue": issue, "content": line})

        return violations

    def _check_emoji_violations(self, content: str) -> List[tuple]:
        """Check for emoji characters in code."""
        violations = []
        lines = content.split("\n")

        # Unicode ranges for emoji
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"  # enclosed characters
            "]+",
            flags=re.UNICODE,
        )

        for line_num, line in enumerate(lines, 1):
            if emoji_pattern.search(line):
                violations.append((line_num, line.strip()))

        return violations

    def _check_path_violations(self, content: str) -> List[tuple]:
        """Check for improper path usage."""
        violations = []
        lines = content.split("\n")

        # Patterns that suggest relative path usage in file operations
        problematic_patterns = [
            (r'open\(["\'][^/]', "Relative path in open()"),
            (r'Path\(["\'][^/]', "Relative path in Path()"),
            (r'glob\(["\'][^/]', "Relative path in glob()"),
        ]

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("#"):  # Skip comments
                continue

            for pattern, issue in problematic_patterns:
                if re.search(pattern, line):
                    violations.append((line_num, issue))

        return violations


class VibelintSelfValidator(BaseValidator):
    """
    Validator that applies vibelint's self-validation hooks.

    This ensures vibelint follows its own standards when analyzing
    its own codebase.
    """

    def __init__(self, severity: Severity = Severity.WARN, config: Optional[Dict[str, Any]] = None):
        super().__init__(severity)
        self.hook = SelfValidationHook()
        self.config = config or {}

    @property
    def rule_id(self) -> str:
        return "VIBELINT-SELF-VALIDATION"

    @property
    def name(self) -> str:
        return "Vibelint Self-Validation"

    @property
    def description(self) -> str:
        return "Ensures vibelint follows its own coding standards and architectural principles"

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Apply all self-validation checks."""
        if not self.hook.should_apply_self_validation(file_path):
            return

        # Apply all self-validation checks
        yield from self.hook.validate_single_file_validator(file_path, content)
        yield from self.hook.validate_project_standards(file_path, content)
        yield from self.hook.validate_validator_categorization(file_path, content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m vibelint.validators.self_validation <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        content = file_path.read_text(encoding="utf-8")
        validator = VibelintSelfValidator()
        findings = list(validator.validate(file_path, content))

        if findings:
            print(f"Self-validation violations found in {file_path}:")
            for finding in findings:
                print(f"  Line {finding.line}: {finding.message}")
                if finding.suggestion:
                    print(f"    Suggestion: {finding.suggestion}")
            sys.exit(1)
        else:
            print(f"Self-validation passed for {file_path}")
            sys.exit(0)

    except Exception as e:
        print(f"Error during self-validation: {e}")
        sys.exit(1)
```

---
### File: src/vibelint/validators/single_file/strict_config.py

```python
"""
Strict Configuration Validator

Enforces strict configuration management by detecting and flagging fallback patterns.
All configuration should go through the CM (Configuration Management) system without fallbacks.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, NamedTuple


# Standalone versions for CLI usage
class ValidationResult(NamedTuple):
    rule_name: str
    severity: str
    message: str
    line_number: int
    column: int
    suggestion: str
    fix_suggestion: str = ""
    category: str = "general"


class CodeContext(NamedTuple):
    file_path: Path
    content: str


class ValidationRule:
    def __init__(self, name: str, description: str, category: str, severity: str):
        self.name = name
        self.description = description
        self.category = category
        self.severity = severity


class StrictConfigRule(ValidationRule):
    """Detects configuration fallbacks and enforces strict config management."""

    def __init__(self):
        super().__init__(
            name="strict-config",
            description="Enforce strict configuration management - no fallbacks",
            category="configuration",
            severity="error",
        )

    def validate(self, context: CodeContext) -> List[ValidationResult]:
        """Check for configuration fallback patterns."""
        results = []

        # Check Python files for .get() patterns with fallbacks
        if context.file_path.suffix == ".py":
            results.extend(self._check_python_config_fallbacks(context))

        # Check for hardcoded workers.dev URLs
        results.extend(self._check_hardcoded_endpoints(context))

        # Check TOML/YAML config files for hardcoded fallbacks
        if context.file_path.suffix in [".toml", ".yaml", ".yml"]:
            results.extend(self._check_config_file_fallbacks(context))

        return results

    def _check_python_config_fallbacks(self, context: CodeContext) -> List[ValidationResult]:
        """Check Python code for config.get() patterns with fallbacks."""
        results = []

        try:
            tree = ast.parse(context.content)
        except SyntaxError:
            return results

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for .get() calls with default values
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) >= 2
                ):

                    # Get the object being called (e.g., 'config', 'embeddings_config')
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                    elif isinstance(node.func.value, ast.Attribute):
                        var_name = ast.unparse(node.func.value)
                    else:
                        continue

                    # Check if this looks like a config object
                    if self._is_config_variable(var_name):
                        # Get the key and default value
                        key_node = node.args[0]
                        default_node = node.args[1]

                        key = self._extract_string_value(key_node)
                        default_value = self._extract_node_value(default_node)

                        # Flag as error
                        results.append(
                            ValidationResult(
                                rule_name=self.name,
                                severity="error",
                                message=f"Configuration fallback detected: {var_name}.get('{key}', {default_value})",
                                line_number=node.lineno,
                                column=node.col_offset,
                                suggestion=f"Use strict config: {var_name}['{key}'] and ensure value exists in config",
                                fix_suggestion=f"{var_name}['{key}']  # STRICT: No fallbacks",
                                category="configuration",
                            )
                        )

        return results

    def _check_hardcoded_endpoints(self, context: CodeContext) -> List[ValidationResult]:
        """Check for hardcoded endpoints that bypass CM."""
        results = []

        # Patterns that indicate hardcoded endpoints
        dangerous_patterns = [
            (r"workers\.dev", "Cloudflare Workers endpoint"),
            (r"https?://[^/]*\.workers\.dev", "Cloudflare Workers URL"),
            (r"https?://\d+\.\d+\.\d+\.\d+:\d+", "Hardcoded IP endpoint"),
            (r"localhost:\d+", "Hardcoded localhost endpoint"),
            (r"127\.0\.0\.1:\d+", "Hardcoded localhost endpoint"),
        ]

        lines = context.content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern, description in dangerous_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Skip if it's in a comment explaining the pattern
                    if "#" in line and line.index("#") < match.start():
                        continue

                    results.append(
                        ValidationResult(
                            rule_name=self.name,
                            severity="error",
                            message=f"Hardcoded endpoint detected: {description}",
                            line_number=line_num,
                            column=match.start(),
                            suggestion="Move endpoint configuration to dev.pyproject.toml or pyproject.toml",
                            fix_suggestion="# FIXME: Move to configuration management",
                            category="configuration",
                        )
                    )

        return results

    def _check_config_file_fallbacks(self, context: CodeContext) -> List[ValidationResult]:
        """Check TOML/YAML files for fallback patterns."""
        results = []

        # Check for production URLs in config files
        production_patterns = [
            r"workers\.dev",
            r"\.vercel\.app",
            r"\.netlify\.app",
            r"\.herokuapp\.com",
        ]

        lines = context.content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern in production_patterns:
                if re.search(pattern, line):
                    results.append(
                        ValidationResult(
                            rule_name=self.name,
                            severity="warning",
                            message="Production URL in config file may need dev override",
                            line_number=line_num,
                            column=0,
                            suggestion="Ensure dev.pyproject.toml overrides production URLs",
                            category="configuration",
                        )
                    )

        return results

    def _is_config_variable(self, var_name: str) -> bool:
        """Check if variable name suggests it's a config object."""
        config_indicators = [
            "config",
            "settings",
            "cfg",
            "conf",
            "embedding_config",
            "embeddings_config",
            "llm_config",
            "kaia_config",
            "tool_config",
            "vibelint_config",
            "guardrails_config",
        ]

        var_lower = var_name.lower()
        return any(indicator in var_lower for indicator in config_indicators)

    def _extract_string_value(self, node: ast.AST) -> str:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        else:
            return ast.unparse(node) if hasattr(ast, "unparse") else "<complex>"

    def _extract_node_value(self, node: ast.AST) -> str:
        """Extract a readable representation of the node value."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Str):
            return repr(node.s)
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Dict)):
            return ast.unparse(node) if hasattr(ast, "unparse") else "<collection>"
        else:
            return ast.unparse(node) if hasattr(ast, "unparse") else "<complex>"


class ConfigFallbackDetector:
    """Standalone utility for detecting configuration fallbacks."""

    def __init__(self):
        self.rule = StrictConfigRule()

    def scan_directory(self, directory: Path) -> Dict[str, List[ValidationResult]]:
        """Scan a directory for configuration fallbacks."""
        results = {}

        for file_path in directory.rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                context = CodeContext(file_path=file_path, content=content)
                file_results = self.rule.validate(context)

                if file_results:
                    results[str(file_path)] = file_results

            except Exception as e:
                print(f"Error scanning {file_path}: {e}")

        return results

    def generate_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate a human-readable report."""
        if not results:
            return "✅ No configuration fallbacks detected!"

        report = ["🚨 CONFIGURATION FALLBACKS DETECTED", "=" * 50, ""]

        total_issues = sum(len(issues) for issues in results.values())
        report.append(f"Total files with issues: {len(results)}")
        report.append(f"Total fallback patterns: {total_issues}")
        report.append("")

        for file_path, issues in results.items():
            report.append(f"📁 {file_path}")
            report.append("-" * 50)

            for issue in issues:
                report.append(f"  ❌ Line {issue.line_number}: {issue.message}")
                report.append(f"     💡 {issue.suggestion}")
                if issue.fix_suggestion:
                    report.append(f"     🔧 Fix: {issue.fix_suggestion}")
                report.append("")

        report.append("=" * 50)
        report.append("🎯 RECOMMENDATION: Move all configuration to CM system")
        report.append("   1. Add required config to dev.pyproject.toml")
        report.append("   2. Replace .get() calls with strict [] access")
        report.append("   3. Let configuration errors fail loudly")

        return "\n".join(report)


# CLI interface for standalone usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python strict_config.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    detector = ConfigFallbackDetector()
    results = detector.scan_directory(directory)
    report = detector.generate_report(results)
    print(report)

    # Exit with error code if issues found
    if results:
        sys.exit(1)
```

---
### File: src/vibelint/validators/single_file/typing_quality.py

```python
"""
Type quality validator using BaseValidator plugin system.

Detects poor typing practices that reduce code clarity and type safety:
- Raw tuples instead of dataclasses/NamedTuples
- Untyped dictionaries instead of TypedDict
- Excessive use of Any
- Missing type annotations on public functions
- String literals that should be Enums

vibelint/src/vibelint/validators/typing_quality.py
"""

import ast
from pathlib import Path
from typing import Iterator

from ...ast_utils import parse_or_none
from ...validators.types import BaseValidator, Finding, Severity

__all__ = ["TypingQualityValidator"]


class TypingQualityValidator(BaseValidator):
    """Validator for detecting poor typing practices."""

    rule_id = "TYPING-POOR-PRACTICE"
    name = "Type Quality Checker"
    description = "Detects poor typing practices that reduce code clarity and type safety"
    default_severity = Severity.WARN

    def validate(self, file_path: Path, content: str, config=None) -> Iterator[Finding]:
        """Validate typing practices in a Python file."""
        tree = parse_or_none(content, file_path)
        if tree is None:
            return

        # Check for tuple type annotations that should be dataclasses
        visitor = _TypingVisitor()
        visitor.visit(tree)

        # Check for dictionary anti-patterns (should be dataclasses)
        yield from self._check_dict_antipatterns(tree, file_path)

        # Report tuple type aliases
        for line_num, name, tuple_info in visitor.tuple_type_aliases:
            if self._looks_like_data_structure(name, tuple_info):
                yield self.create_finding(
                    message=f"Type alias '{name}' uses raw Tuple{tuple_info} - consider using a dataclass or NamedTuple for better clarity",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace with: @dataclass class {name}: ...",
                )

        # Report untyped dictionaries
        for line_num, context in visitor.untyped_dicts:
            yield self.create_finding(
                message=f"Using untyped Dict{context} - consider using TypedDict for better type safety",
                file_path=file_path,
                line=line_num,
                suggestion="Define a TypedDict with explicit field types",
            )

        # Report excessive Any usage
        for line_num, context in visitor.any_usage:
            if not self._is_acceptable_any_usage(context):
                yield self.create_finding(
                    message=f"Using Any type{context} - specify a more precise type",
                    file_path=file_path,
                    line=line_num,
                    suggestion="Replace Any with a specific type or Union of types",
                )

        # Report missing type annotations on public functions
        for line_num, func_name in visitor.untyped_public_functions:
            yield self.create_finding(
                message=f"Public function '{func_name}' is missing type annotations",
                file_path=file_path,
                line=line_num,
                suggestion=f"Add type hints: def {func_name}(...) -> ReturnType:",
            )

        # Report string literals that look like enums
        enum_candidates = self._find_enum_candidates(visitor.string_constants)
        for pattern, locations in enum_candidates.items():
            if len(locations) >= 3:  # Same string pattern used 3+ times
                first_line = locations[0]
                yield self.create_finding(
                    message=f"String literal '{pattern}' used {len(locations)} times - consider using an Enum",
                    file_path=file_path,
                    line=first_line,
                    suggestion="Create an Enum for these related string constants",
                )

    def _looks_like_data_structure(self, name: str, tuple_info: str) -> bool:
        """Check if a tuple type alias looks like it should be a data structure."""
        # Skip if it's a simple pair like (bool, str) for return values
        if tuple_info.count(",") == 1 and "bool" in tuple_info.lower():
            return False

        # If the name suggests it's data (Issue, Result, Info, etc.)
        data_suffixes = ["Issue", "Result", "Info", "Data", "Record", "Entry", "Item"]
        return any(name.endswith(suffix) for suffix in data_suffixes)

    def _is_acceptable_any_usage(self, context: str) -> bool:
        """Check if Any usage is acceptable in this context."""
        # Any is acceptable for **kwargs, *args, or when interfacing with external libs
        acceptable_patterns = ["**kwargs", "*args", "json", "yaml", "config"]
        return any(pattern in context.lower() for pattern in acceptable_patterns)

    def _find_enum_candidates(self, string_constants: list) -> dict:
        """Find string literals that are used repeatedly and could be enums."""
        from collections import defaultdict

        # Group by string pattern (uppercase, prefix, etc.)
        patterns = defaultdict(list)

        for line_num, value in string_constants:
            # Skip short strings and file paths
            if len(value) < 3 or "/" in value or "\\" in value:
                continue

            # Look for patterns like "ERROR", "WARNING", "INFO"
            if value.isupper() and "_" in value:
                patterns[value].append(line_num)
            # Or prefixed patterns like "RULE101", "ERR102"
            elif any(value.startswith(prefix) for prefix in ["RULE", "ERR", "WARN"]):
                prefix = value[:3]
                patterns[f"{prefix}*"].append(line_num)

        return patterns

    def _check_dict_antipatterns(self, tree: ast.AST, file_path: Path) -> Iterator[Finding]:
        """Check for dictionaries that should be dataclasses (merged from dict_antipattern.py)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                # Analyze dictionary literal
                if not node.keys or len(node.keys) < 2:
                    continue

                # Extract string keys
                string_keys = []
                for key in node.keys:
                    if key is None:  # **kwargs expansion
                        break
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        string_keys.append(key.value)
                    else:
                        break  # Non-string keys, not a candidate

                if len(string_keys) >= 3:
                    yield self.create_finding(
                        message=f"Dictionary with {len(string_keys)} fixed keys should use dataclass/NamedTuple",
                        file_path=file_path,
                        line=node.lineno,
                        suggestion=self._suggest_dataclass_for_dict(string_keys),
                    )
                elif len(string_keys) == 2 and self._dict_keys_look_structured(string_keys):
                    yield self.create_finding(
                        message=f"Dictionary with structured keys '{', '.join(string_keys)}' might benefit from dataclass",
                        file_path=file_path,
                        line=node.lineno,
                        severity=Severity.INFO,
                        suggestion=self._suggest_dataclass_for_dict(string_keys),
                    )

            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
                # Analyze dict() constructor call
                keys = [kw.arg for kw in node.keywords if kw.arg]
                if len(keys) >= 3:
                    yield self.create_finding(
                        message=f"dict() call with {len(keys)} fixed keys should use dataclass/NamedTuple",
                        file_path=file_path,
                        line=node.lineno,
                        suggestion=self._suggest_dataclass_for_dict(keys),
                    )

    def _dict_keys_look_structured(self, keys: list) -> bool:
        """Check if dict keys look like structured data rather than dynamic mapping."""
        structured_indicators = 0
        for key in keys:
            if len(key) > 3:  # Not single letters
                structured_indicators += 1
            if '_' in key:  # Snake case
                structured_indicators += 1
            if key in ['id', 'name', 'type', 'value', 'data', 'config', 'status',
                      'created', 'updated', 'url', 'path', 'file', 'directory']:
                structured_indicators += 1
        return structured_indicators >= len(keys)

    def _suggest_dataclass_for_dict(self, keys: list) -> str:
        """Generate a dataclass suggestion for dictionary keys."""
        fields = []
        for key in keys:
            if key in ['id', 'count', 'size', 'length']:
                fields.append(f"{key}: int")
            elif key in ['name', 'path', 'url', 'type', 'status']:
                fields.append(f"{key}: str")
            elif key in ['active', 'enabled', 'valid', 'success']:
                fields.append(f"{key}: bool")
            else:
                fields.append(f"{key}: Any  # TODO: specify type")

        return f"Consider dataclass:\n@dataclass\nclass Data:\n    " + "\n    ".join(fields)


class _TypingVisitor(ast.NodeVisitor):
    """AST visitor to detect typing issues."""

    def __init__(self):
        self.tuple_type_aliases = []  # (line, name, tuple_info)
        self.untyped_dicts = []  # (line, context)
        self.any_usage = []  # (line, context)
        self.untyped_public_functions = []  # (line, func_name)
        self.string_constants = []  # (line, value)

    def visit_AnnAssign(self, node):
        """Visit annotated assignments to find type aliases."""
        if isinstance(node.target, ast.Name):
            name = node.target.id

            # Check for Tuple type annotations
            if self._is_tuple_annotation(node.annotation):
                tuple_info = (
                    ast.unparse(node.annotation)
                    if hasattr(ast, "unparse")
                    else str(node.annotation)
                )
                self.tuple_type_aliases.append((node.lineno, name, tuple_info))

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit assignments to find type aliases using old syntax."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith(("Issue", "Result", "Info")):
                # Check if it's a type alias assignment like: ValidationIssue = Tuple[str, str]
                if self._is_tuple_annotation(node.value):
                    tuple_info = (
                        ast.unparse(node.value) if hasattr(ast, "unparse") else str(node.value)
                    )
                    self.tuple_type_aliases.append((node.lineno, target.id, tuple_info))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definitions to check for type annotations."""
        # Check if public function (doesn't start with _)
        if not node.name.startswith("_"):
            # Check if it has return type annotation
            if node.returns is None:
                self.untyped_public_functions.append((node.lineno, node.name))
            else:
                # Check for Any in return type
                if self._contains_any(node.returns):
                    context = f" in return type of {node.name}"
                    self.any_usage.append((node.lineno, context))

            # Check parameters
            for arg in node.args.args:
                if arg.annotation is None and arg.arg != "self":
                    self.untyped_public_functions.append((node.lineno, f"{node.name}({arg.arg})"))
                elif arg.annotation and self._contains_any(arg.annotation):
                    context = f" in parameter '{arg.arg}' of {node.name}"
                    self.any_usage.append((node.lineno, context))

        self.generic_visit(node)

    def visit_Constant(self, node):
        """Visit string constants to find potential enums."""
        if isinstance(node.value, str):
            self.string_constants.append((node.lineno, node.value))
        self.generic_visit(node)

    def _is_tuple_annotation(self, node) -> bool:
        """Check if a node is a Tuple type annotation."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "Tuple":
                return True
            # Check for typing.Tuple
            if isinstance(node.value, ast.Attribute):
                if node.value.attr == "Tuple":
                    return True
        return False

    def _contains_any(self, node) -> bool:
        """Check if a type annotation contains Any."""
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
        # Recursively check in subscripts (like Optional[Any], List[Any])
        if isinstance(node, ast.Subscript):
            return self._contains_any(node.value) or any(
                self._contains_any(arg)
                for arg in (node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice])
            )
        return False
```

---
### File: src/vibelint/validators/types.py

```python
"""
Core validation types for vibelint.

Defines fundamental types used throughout the validation system:
- Severity: Severity levels for findings (INFO, WARN, BLOCK)
- Finding: Dataclass representing a validation finding
- BaseValidator: Abstract base class for validators
- BaseFormatter: Abstract base class for formatters
- Validator/Formatter: Protocol types for duck typing

Also provides simple registration and discovery functions.

vibelint/src/vibelint/validators/types.py
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Type

logger = logging.getLogger(__name__)

__all__ = [
    "Severity",
    "Finding",
    "Validator",
    "Formatter",
    "BaseValidator",
    "BaseFormatter",
    "get_all_validators",
    "get_all_formatters",
    "get_validator",
    "get_formatter",
]


class Severity(Enum):
    """Severity levels for validation findings."""

    OFF = "OFF"
    INFO = "INFO"
    WARN = "WARN"
    BLOCK = "BLOCK"

    def __lt__(self, other):
        """Enable sorting by severity."""
        order = {"OFF": 0, "INFO": 1, "WARN": 2, "BLOCK": 3}
        return order[self.value] < order[other.value]


@dataclass
class Finding:
    """A validation finding from a validator."""

    rule_id: str
    message: str
    file_path: Path
    line: int = 0
    column: int = 0
    severity: Severity = Severity.WARN
    context: str = ""
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for JSON output."""
        return {
            "rule": self.rule_id,
            "level": self.severity.value,
            "path": str(self.file_path),
            "line": self.line,
            "column": self.column,
            "msg": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
        }


class Validator(Protocol):
    """Protocol for validator classes - simpler than abstract base class."""

    rule_id: str
    default_severity: Severity

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict[str, Any]] = None
    ) -> None: ...

    def validate(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Validate a file and yield findings."""
        ...

    def create_finding(
        self,
        message: str,
        file_path: Path,
        line: int = 0,
        column: int = 0,
        context: str = "",
        suggestion: Optional[str] = None,
    ) -> Finding:
        """Create a Finding object with this validator's rule_id and severity."""
        ...


class Formatter(Protocol):
    """Protocol for formatter classes - simpler than abstract base class."""

    name: str

    def format_results(
        self,
        findings: List[Finding],
        summary: Dict[str, int],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format validation results for output."""
        ...


# Simple registry - no complex plugin discovery needed
_VALIDATORS: Dict[str, Type[Validator]] = {}
_FORMATTERS: Dict[str, Type[Formatter]] = {}


def register_validator(validator_class: Type[Validator]) -> None:
    """Register a validator class."""
    _VALIDATORS[validator_class.rule_id] = validator_class


def register_formatter(formatter_class: Type[Formatter]) -> None:
    """Register a formatter class."""
    _FORMATTERS[formatter_class.name] = formatter_class


def get_validator(rule_id: str) -> Optional[Type[Validator]]:
    """Get validator class by rule ID."""
    return _VALIDATORS.get(rule_id)


def get_all_validators() -> Dict[str, Type[Validator]]:
    """Get all registered validator classes."""
    # Lazy load validators from entry points on first access
    if not _VALIDATORS:
        _load_builtin_validators()
    return _VALIDATORS.copy()


def get_formatter(name: str) -> Optional[Type[Formatter]]:
    """Get formatter class by name."""
    return _FORMATTERS.get(name)


def get_all_formatters() -> Dict[str, Type[Formatter]]:
    """Get all registered formatter classes."""
    # Lazy load formatters from entry points on first access
    if not _FORMATTERS:
        _load_builtin_formatters()
    return _FORMATTERS.copy()


def _load_builtin_validators() -> None:
    """
    Load built-in validators via filesystem auto-discovery.

    Scans vibelint.validators.* packages and auto-discovers BaseValidator subclasses.
    Third-party validators can still use entry points.
    """
    import importlib
    import importlib.util
    import pkgutil

    # Auto-discover built-in validators from filesystem
    try:
        import vibelint.validators
        validators_path = Path(vibelint.validators.__file__).parent

        # Scan all subdirectories: single_file, project_wide, architecture
        for subdir in ["single_file", "project_wide", "architecture"]:
            subdir_path = validators_path / subdir
            if not subdir_path.is_dir():
                continue

            # Import all Python modules in this subdirectory
            for module_file in subdir_path.glob("*.py"):
                if module_file.name.startswith("_"):
                    continue

                module_name = f"vibelint.validators.{subdir}.{module_file.stem}"
                try:
                    module = importlib.import_module(module_name)

                    # Find all BaseValidator subclasses in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseValidator)
                            and attr is not BaseValidator
                            and hasattr(attr, "rule_id")
                            and attr.rule_id  # Must have non-empty rule_id
                        ):
                            _VALIDATORS[attr.rule_id] = attr
                            logger.debug(f"Auto-discovered validator: {attr.rule_id} from {module_name}")

                except (ImportError, AttributeError) as e:
                    logger.debug(f"Failed to load validator module {module_name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to auto-discover built-in validators: {e}")

    # Also load third-party validators from entry points
    try:
        import importlib.metadata
        for entry_point in importlib.metadata.entry_points(group="vibelint.validators"):
            try:
                validator_class = entry_point.load()
                if hasattr(validator_class, "rule_id") and validator_class.rule_id:
                    _VALIDATORS[validator_class.rule_id] = validator_class
                    logger.debug(f"Loaded third-party validator from entry point: {validator_class.rule_id}")
            except (ImportError, AttributeError, TypeError) as e:
                logger.debug(f"Failed to load validator from entry point {entry_point.name}: {e}")
    except Exception as e:
        logger.debug(f"Entry point discovery failed: {e}")


def _load_builtin_formatters() -> None:
    """Load built-in formatters from entry points."""
    import importlib.metadata

    for entry_point in importlib.metadata.entry_points(group="vibelint.formatters"):
        try:
            formatter_class = entry_point.load()
            if hasattr(formatter_class, "name"):
                _FORMATTERS[formatter_class.name] = formatter_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to load formatter from entry point {entry_point.name}: {e}")
            pass


# Concrete base classes
class BaseValidator:
    """Base class for validators."""

    rule_id: str = ""
    default_severity: Severity = Severity.WARN

    def __init__(
        self, severity: Optional[Severity] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.severity = severity or self.default_severity
        self.config = config or {}

    def validate(
        self, file_path: Path, content: str, config: Optional[Dict[str, Any]] = None
    ) -> Iterator[Finding]:
        """Validate a file and yield findings."""
        raise NotImplementedError

    def create_finding(
        self,
        message: str,
        file_path: Path,
        line: int = 0,
        column: int = 0,
        context: str = "",
        suggestion: Optional[str] = None,
    ) -> Finding:
        """Create a Finding object with this validator's rule_id and severity."""
        return Finding(
            rule_id=self.rule_id,
            message=message,
            file_path=file_path,
            line=line,
            column=column,
            severity=self.severity,
            context=context,
            suggestion=suggestion,
        )


class BaseFormatter(ABC):
    """Base class for formatters."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def format_results(
        self,
        findings: List[Finding],
        summary: Dict[str, int],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format validation results for output."""
        pass


# Legacy global manager for backward compatibility
class _LegacyPluginManager:
    """Legacy compatibility wrapper."""

    def load_plugins(self) -> None:
        """Load plugins - delegated to new system."""
        get_all_validators()
        get_all_formatters()

    def get_validator(self, rule_id: str) -> Optional[Any]:
        """Get validator by rule ID."""
        return get_validator(rule_id)

    def get_all_validators(self) -> Dict[str, type]:
        """Get all validators."""
        return get_all_validators()

    def get_formatter(self, name: str) -> Optional[Any]:
        """Get formatter by name."""
        return get_formatter(name)

    def get_all_formatters(self) -> Dict[str, type]:
        """Get all formatters."""
        return get_all_formatters()


plugin_manager = _LegacyPluginManager()
```

---
### File: src/vibelint/workflows/__init__.py

```python
"""
Workflow management subsystem for vibelint.

Modular workflow system with centralized registry and clear separation:
- core/: Base classes, registry, orchestration
- implementations/: Actual workflow implementations

Responsibility: Workflow module organization and re-exports only.
Individual workflow logic belongs in specific implementation modules.

vibelint/src/vibelint/workflow/__init__.py
"""

# Avoid importing implementations directly to prevent circular imports
# Access implementations through lazy loading
# Import core workflow system
from .core.base import (BaseWorkflow, WorkflowConfig, WorkflowMetrics,
                        WorkflowPriority, WorkflowResult, WorkflowStatus)
from .evaluation import WorkflowEvaluator
# Import registry system
from .registry import WorkflowRegistry, register_workflow, workflow_registry

# Lazy imports for specific implementations to avoid circular dependencies
def get_justification_engine():
    """Get JustificationEngine class."""
    from .implementations.justification import JustificationEngine
    return JustificationEngine

__all__ = [
    # Core workflow system
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowConfig",
    "WorkflowMetrics",
    "WorkflowStatus",
    "WorkflowPriority",
    # Registry system
    "WorkflowRegistry",
    "workflow_registry",
    "register_workflow",
    # Evaluation
    "WorkflowEvaluator",
    # Lazy import functions
    "get_justification_engine",
]
```

---
### File: src/vibelint/workflows/cleanup.py

```python
"""
Vibelint Project Cleanup Workflow

Implements systematic project cleanup based on Workflow 7 principles.
Human-in-the-loop orchestration for cleaning up messy repositories.
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DuplicateFile:
    """Represents a duplicate file found in the project."""

    original: str
    duplicate: str
    size: int
    hash: str


@dataclass
class TempFile:
    """Represents a temporary file or directory."""

    path: str
    type: str  # "temp_file" or "temp_directory"
    size: int
    pattern: str


@dataclass
class UnusedFile:
    """Represents a potentially unused Python module."""

    path: str
    type: str
    size: int


@dataclass
class LargeFile:
    """Represents an unusually large file."""

    path: str
    size: int
    size_mb: float


@dataclass
class ConfigFile:
    """Represents a configuration file."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class DebugScript:
    """Represents a debug/test script."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class BackupFile:
    """Represents a backup file."""

    path: str
    type: str
    pattern: str
    size: int


@dataclass
class UntrackedFile:
    """Represents an untracked file that might be important."""

    path: str
    type: str
    size: int


@dataclass
class CleanupRecommendation:
    """Represents a cleanup recommendation."""

    type: str
    priority: str
    description: str
    impact: str
    files: List[Any]


@dataclass
class ProjectAnalysis:
    """Complete project analysis results."""

    duplicate_files: List[DuplicateFile]
    temp_files: List[TempFile]
    unused_files: List[UnusedFile]
    large_files: List[LargeFile]
    empty_directories: List[str]
    config_fragments: List[ConfigFile]
    debug_scripts: List[DebugScript]
    backup_files: List[BackupFile]
    untracked_important: List[UntrackedFile]
    mess_score: float
    recommendations: List[CleanupRecommendation]


@dataclass
class CleanupAction:
    """Represents a cleanup action that was executed or skipped."""

    description: str
    type: str = ""
    path: str = ""


@dataclass
class CleanupError:
    """Represents an error encountered during cleanup."""

    error: str
    path: str = ""
    action_type: str = ""


@dataclass
class CleanupResults:
    """Results from executing cleanup actions."""

    executed: List[CleanupAction] = field(default_factory=list)
    skipped: List[CleanupAction] = field(default_factory=list)
    errors: List[CleanupError] = field(default_factory=list)
    space_saved: int = 0


@dataclass
class WorkflowStatus:
    """Status of the cleanup workflow."""

    analysis: ProjectAnalysis
    workflow: "ProjectCleanupWorkflow"
    next_step: str


class ProjectCleanupWorkflow:
    """Systematic project cleanup with human decision points."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cleanup_log = []
        self.temp_backup_dir = None

    def analyze_project_mess(self) -> ProjectAnalysis:
        """
        Analyze the project to identify cleanup opportunities.
        Human Decision Point: What types of mess to look for.
        """
        duplicate_files = self._find_duplicate_files()
        temp_files = self._find_temp_files()
        unused_files = self._find_unused_files()
        large_files = self._find_large_files()
        empty_directories = self._find_empty_directories()
        config_fragments = self._find_config_fragments()
        debug_scripts = self._find_debug_scripts()
        backup_files = self._find_backup_files()
        untracked_important = self._find_untracked_important_files()

        # Calculate mess score
        mess_score = self._calculate_mess_score(
            duplicate_files,
            temp_files,
            unused_files,
            empty_directories,
            debug_scripts,
            backup_files,
            large_files,
        )
        recommendations = self._generate_cleanup_recommendations(
            duplicate_files,
            temp_files,
            unused_files,
            empty_directories,
            debug_scripts,
            backup_files,
        )

        return ProjectAnalysis(
            duplicate_files=duplicate_files,
            temp_files=temp_files,
            unused_files=unused_files,
            large_files=large_files,
            empty_directories=empty_directories,
            config_fragments=config_fragments,
            debug_scripts=debug_scripts,
            backup_files=backup_files,
            untracked_important=untracked_important,
            mess_score=mess_score,
            recommendations=recommendations,
        )

    def _find_duplicate_files(self) -> List[DuplicateFile]:
        """Find duplicate files by content hash."""
        file_hashes = {}
        duplicates = []

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    if file_hash in file_hashes:
                        duplicates.append(
                            DuplicateFile(
                                original=str(file_hashes[file_hash]),
                                duplicate=str(file_path),
                                size=file_path.stat().st_size,
                                hash=file_hash,
                            )
                        )
                    else:
                        file_hashes[file_hash] = file_path

                except (IOError, OSError):
                    continue

        return duplicates

    def _find_temp_files(self) -> List[TempFile]:
        """Find temporary and build artifacts."""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.swp",
            "*.swo",
            "*~",
            "#*#",
            ".#*",
            "*.orig",
            "*.rej",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".pytest_cache/",
            ".coverage",
            "*.egg-info/",
            "node_modules/",
            ".npm/",
            "yarn-error.log",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            ".vscode/",
            ".idea/",
            "*.log",
        ]

        temp_files = []
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.exists():
                    temp_files.append(
                        TempFile(
                            path=str(file_path),
                            type="temp_file" if file_path.is_file() else "temp_directory",
                            size=self._get_size(file_path),
                            pattern=pattern,
                        )
                    )

        return temp_files

    def _find_unused_files(self) -> List[UnusedFile]:
        """Find files that appear unused (not imported/referenced)."""
        # Simple heuristic - look for Python files that aren't imported
        python_files = list(self.project_root.rglob("*.py"))
        all_content = ""

        # Read all Python files to check for imports
        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    all_content += f.read() + "\n"
            except (IOError, UnicodeDecodeError):
                continue

        unused_files = []
        for py_file in python_files:
            # Skip if it's a main script or test file
            if py_file.name in ["__main__.py", "__init__.py"] or "test" in py_file.name.lower():
                continue

            module_name = py_file.stem
            # Check if module is imported anywhere
            if (
                f"import {module_name}" not in all_content
                and f"from {module_name}" not in all_content
            ):
                try:
                    size = py_file.stat().st_size
                except OSError:
                    size = 0
                unused_files.append(
                    UnusedFile(
                        path=str(py_file), type="potentially_unused_python_module", size=size
                    )
                )

        return unused_files

    def _find_large_files(self) -> List[LargeFile]:
        """Find unusually large files that might need attention."""
        large_files = []
        size_threshold = 1024 * 1024  # 1MB

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > size_threshold:
                large_files.append(
                    LargeFile(
                        path=str(file_path),
                        size=file_path.stat().st_size,
                        size_mb=file_path.stat().st_size / (1024 * 1024),
                    )
                )

        return sorted(large_files, key=lambda x: x.size, reverse=True)

    def _find_empty_directories(self) -> List[str]:
        """Find empty directories that can be removed."""
        empty_dirs = []

        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                empty_dirs.append(str(dir_path))

        return empty_dirs

    def _find_config_fragments(self) -> List[ConfigFile]:
        """Find scattered configuration files that might be consolidated."""
        config_patterns = [
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.ini",
            "*.cfg",
            ".env*",
            "Dockerfile*",
            "requirements*.txt",
            "setup.py",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
            ".gitignore",
        ]

        config_files = []
        for pattern in config_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    config_files.append(
                        ConfigFile(
                            path=str(file_path),
                            type="config_file",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return config_files

    def _find_debug_scripts(self) -> List[DebugScript]:
        """Find debug/test scripts that might be temporary."""
        debug_patterns = [
            "debug_*.py",
            "test_*.py",
            "*_debug.py",
            "*_test.py",
            "scratch*.py",
            "temp*.py",
            "fix_*.py",
            "quick_*.py",
        ]

        debug_files = []
        for pattern in debug_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    debug_files.append(
                        DebugScript(
                            path=str(file_path),
                            type="debug_script",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return debug_files

    def _find_backup_files(self) -> List[BackupFile]:
        """Find backup files that can be cleaned up."""
        backup_patterns = [
            "*.backup",
            "*.bkp",
            "*_backup.*",
            "*.old",
            "*_old.*",
            "*.save",
            "*_save.*",
            "*.copy",
        ]

        backup_files = []
        for pattern in backup_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    backup_files.append(
                        BackupFile(
                            path=str(file_path),
                            type="backup_file",
                            pattern=pattern,
                            size=file_path.stat().st_size,
                        )
                    )

        return backup_files

    def _find_untracked_important_files(self) -> List[UntrackedFile]:
        """Find untracked files that might be important."""
        # Do not swallow subprocess errors – allow caller to handle if needed.
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        untracked_files = []
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = self.project_root / line
                    if file_path.is_file():
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        untracked_files.append(
                            UntrackedFile(path=str(file_path), type="untracked_file", size=size)
                        )

        else:
            # If git failed, raise to avoid silently returning a fallback result.
            raise RuntimeError(
                f"git ls-files failed with return code {result.returncode}: {result.stderr}"
            )

        return untracked_files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis."""
        ignore_patterns = [
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            "node_modules/",
            ".venv/",
            "venv/",
            ".env/",
            "env/",
        ]

        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)

    def _get_size(self, path: Path) -> int:
        """Get size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0

    def _calculate_mess_score(
        self,
        duplicate_files: List[DuplicateFile],
        temp_files: List[TempFile],
        unused_files: List[UnusedFile],
        empty_directories: List[str],
        debug_scripts: List[DebugScript],
        backup_files: List[BackupFile],
        large_files: List[LargeFile],
    ) -> float:
        """Calculate overall mess score (0-100)."""
        score = 0

        # Weight different types of mess
        score += len(duplicate_files) * 5
        score += len(temp_files) * 2
        score += len(unused_files) * 3
        score += len(empty_directories) * 1
        score += len(debug_scripts) * 2
        score += len(backup_files) * 3

        # Large files add to mess
        total_large_size = sum(f.size for f in large_files)
        score += total_large_size / (1024 * 1024 * 10)  # 10MB = 1 point

        return min(score, 100)  # Cap at 100

    def _generate_cleanup_recommendations(
        self,
        duplicate_files: List[DuplicateFile],
        temp_files: List[TempFile],
        unused_files: List[UnusedFile],
        empty_directories: List[str],
        debug_scripts: List[DebugScript],
        backup_files: List[BackupFile],
    ) -> List[CleanupRecommendation]:
        """Generate prioritized cleanup recommendations."""
        recommendations = []

        if duplicate_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_duplicates",
                    priority="high",
                    description=f"Remove {len(duplicate_files)} duplicate files",
                    impact="disk_space",
                    files=duplicate_files,
                )
            )

        if temp_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_temp_files",
                    priority="high",
                    description=f"Remove {len(temp_files)} temporary files",
                    impact="cleanliness",
                    files=temp_files,
                )
            )

        if backup_files:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_backup_files",
                    priority="medium",
                    description=f"Remove {len(backup_files)} backup files",
                    impact="cleanliness",
                    files=backup_files,
                )
            )

        if debug_scripts:
            recommendations.append(
                CleanupRecommendation(
                    type="review_debug_scripts",
                    priority="medium",
                    description=f"Review {len(debug_scripts)} debug scripts",
                    impact="organization",
                    files=debug_scripts,
                )
            )

        if empty_directories:
            recommendations.append(
                CleanupRecommendation(
                    type="remove_empty_dirs",
                    priority="low",
                    description=f"Remove {len(empty_directories)} empty directories",
                    impact="cleanliness",
                    files=empty_directories,
                )
            )

        return recommendations

    def commit_cleanup_results(self, results: CleanupResults, cleanup_name: str):
        """Commit cleanup results with detailed message."""
        if not results.executed:
            print("No cleanup actions were executed - nothing to commit")
            return

        # Stage all changes
        subprocess.run(["git", "add", "-A"], cwd=self.project_root, check=True)

        # Create comprehensive commit message
        commit_msg = f"Cleanup: {cleanup_name}\n\n"
        commit_msg += f"Space saved: {results.space_saved / (1024*1024):.1f} MB\n"
        commit_msg += f"Actions executed: {len(results.executed)}\n"
        commit_msg += f"Actions skipped: {len(results.skipped)}\n\n"

        if results.executed:
            commit_msg += "Executed cleanup actions:\n"
            for action in results.executed:
                commit_msg += f"- {action.description}\n"

        if results.errors:
            commit_msg += f"\nErrors encountered: {len(results.errors)}\n"
            for error in results.errors:
                commit_msg += f"- {error.error}\n"

        commit_msg += "\nGenerated with vibelint cleanup workflow"

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=self.project_root, check=True)

        print(f"Committed cleanup results: {cleanup_name}")


def run_cleanup_workflow(project_root: str, cleanup_name: str = "general") -> WorkflowStatus:
    """
    Main entry point for cleanup workflow.
    Human Decision Points throughout the process.
    """
    workflow = ProjectCleanupWorkflow(Path(project_root))

    print(f"Starting cleanup analysis for: {project_root}")

    # Step 1: Analyze project mess
    print("Analyzing project structure...")
    analysis = workflow.analyze_project_mess()

    print(f"Mess score: {analysis.mess_score:.1f}/100")
    print(f"Found {len(analysis.recommendations)} cleanup recommendations")

    # Step 2: Present recommendations to human
    print("\nCleanup Recommendations:")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"{i}. [{rec.priority.upper()}] {rec.description}")
        print(f"   Impact: {rec.impact}")

    # HUMAN DECISION POINT: Which recommendations to execute
    print("\nHUMAN DECISION REQUIRED:")
    print("Which cleanup actions would you like to execute?")
    print("Available types:", [rec.type for rec in analysis.recommendations])

    # For now, return analysis for human review
    # In interactive mode, human would approve specific actions
    return WorkflowStatus(analysis=analysis, workflow=workflow, next_step="human_approval_required")
```

---
### File: src/vibelint/workflows/core/__init__.py

```python
"""
Core workflow system package.

Contains base classes, registry, and orchestration infrastructure.

Responsibility: Core workflow infrastructure only.
Workflow implementations belong in the implementations/ package.

vibelint/src/vibelint/workflow/core/__init__.py
"""

from .base import (BaseWorkflow, WorkflowConfig, WorkflowMetrics,
                   WorkflowPriority, WorkflowResult, WorkflowStatus)

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowConfig",
    "WorkflowMetrics",
    "WorkflowStatus",
    "WorkflowPriority",
]
```

---
### File: src/vibelint/workflows/core/base.py

```python
"""
Base workflow system for extensible analysis tasks.

Provides framework for creating modular, composable workflows with
built-in evaluation, metrics collection, and plugin integration.

vibelint/src/vibelint/workflows/base.py
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = [
    "WorkflowStatus",
    "WorkflowPriority",
    "WorkflowConfig",
    "WorkflowMetrics",
    "WorkflowResult",
    "BaseWorkflow",
]


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowPriority(Enum):
    """Workflow execution priority."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    # Basic settings
    enabled: bool = True
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    timeout_seconds: Optional[int] = None

    # Dependencies and requirements
    required_tools: Set[str] = field(default_factory=set)
    required_data: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)  # Other workflow IDs

    # Execution settings
    parallel_execution: bool = False
    max_retries: int = 0
    cache_results: bool = True

    # Input/output settings
    input_filters: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "json"

    # Custom parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowMetrics:
    """Metrics collected during workflow execution."""

    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    execution_time_seconds: Optional[float] = None

    # Resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Analysis metrics
    files_processed: int = 0
    findings_generated: int = 0
    errors_encountered: int = 0

    # Quality metrics
    confidence_score: float = 0.0
    accuracy_score: Optional[float] = None
    coverage_percentage: Optional[float] = None

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.execution_time_seconds = self.end_time - self.start_time


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    # Execution info
    workflow_id: str
    status: WorkflowStatus
    metrics: WorkflowMetrics

    # Results
    findings: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


class BaseWorkflow(ABC):
    """Abstract base class for vibelint workflows."""

    # Workflow identification
    workflow_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Workflow categorization
    category: str = "analysis"  # analysis, validation, reporting, maintenance
    tags: Set[str] = set()

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize workflow with configuration."""
        self.config = config or WorkflowConfig()
        self.metrics = WorkflowMetrics(start_time=time.time())
        self._status = WorkflowStatus.PENDING

        # Validate workflow setup
        self._validate_configuration()

    @abstractmethod
    async def execute(self, project_root: Path, context: Dict[str, Any]) -> WorkflowResult:
        """Execute the workflow with given context.

        Args:
            project_root: Root directory of the project
            context: Execution context with shared data

        Returns:
            WorkflowResult with findings and artifacts
        """
        pass

    @abstractmethod
    def get_required_inputs(self) -> Set[str]:
        """Get set of required input data keys."""
        pass

    @abstractmethod
    def get_produced_outputs(self) -> Set[str]:
        """Get set of output data keys this workflow produces."""
        pass

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if workflow can execute with given context."""
        # Check if enabled
        if not self.config.enabled:
            return False

        # Check required inputs
        required_inputs = self.get_required_inputs()
        available_inputs = set(context.keys())

        if not required_inputs.issubset(available_inputs):
            missing = required_inputs - available_inputs
            logger.debug(f"Workflow {self.workflow_id} missing inputs: {missing}")
            return False

        # Check required tools
        if self.config.required_tools:
            # This would check for tool availability
            pass

        return True

    def estimate_execution_time(self, context: Dict[str, Any]) -> float:
        """Estimate execution time in seconds based on context."""
        # Default implementation - workflows can override
        base_time = 10.0  # 10 seconds base

        # Scale by number of files if available
        if "file_count" in context:
            file_count = context["file_count"]
            base_time += file_count * 0.1  # 0.1 seconds per file

        return base_time

    def get_dependencies(self) -> List[str]:
        """Get list of workflow IDs this workflow depends on."""
        return self.config.dependencies

    def get_priority(self) -> WorkflowPriority:
        """Get workflow execution priority."""
        return self.config.priority

    def supports_parallel_execution(self) -> bool:
        """Check if workflow supports parallel execution."""
        return self.config.parallel_execution

    def _validate_configuration(self):
        """Validate workflow configuration."""
        if not self.workflow_id:
            raise ValueError(f"Workflow {self.__class__.__name__} must define workflow_id")

        if not self.name:
            raise ValueError(f"Workflow {self.workflow_id} must define name")

    def _update_status(self, status: WorkflowStatus):
        """Update workflow execution status."""
        self._status = status

        if status == WorkflowStatus.COMPLETED:
            self.metrics.end_time = time.time()
            self.metrics.finalize()

    def _create_result(
        self,
        status: WorkflowStatus,
        findings: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> WorkflowResult:
        """Create workflow result."""
        self._update_status(status)

        return WorkflowResult(
            workflow_id=self.workflow_id,
            status=status,
            metrics=self.metrics,
            findings=findings or [],
            artifacts=artifacts or {},
            error_message=error_message,
        )

    async def _execute_with_error_handling(
        self, project_root: Path, context: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute workflow with comprehensive error handling."""
        try:
            self._update_status(WorkflowStatus.RUNNING)

            # Check timeout
            if self.config.timeout_seconds:
                # Implementation would use asyncio.wait_for
                pass

            # Execute the workflow
            result = await self.execute(project_root, context)

            # Validate result
            if result.status == WorkflowStatus.PENDING:
                result.status = WorkflowStatus.COMPLETED

            return result

        except Exception as e:
            logger.error(f"Workflow {self.workflow_id} failed: {e}", exc_info=True)
            self.metrics.errors_encountered += 1

            return self._create_result(WorkflowStatus.FAILED, error_message=str(e))

    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """Get criteria for evaluating workflow effectiveness."""
        return {
            "performance": {
                "max_execution_time": 60.0,  # seconds
                "max_memory_usage": 500.0,  # MB
            },
            "quality": {
                "min_confidence_score": 0.7,
                "min_coverage_percentage": 80.0,
            },
            "reliability": {
                "max_error_rate": 0.05,  # 5%
                "max_timeout_rate": 0.01,  # 1%
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.workflow_id}, status={self._status.value})"
```

---
### File: src/vibelint/workflows/evaluation.py

```python
"""
Workflow evaluation framework for measuring effectiveness and performance.

Provides metrics collection, benchmarking, and continuous improvement
capabilities for workflow analysis quality.

vibelint/src/vibelint/workflow_evaluation.py
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.base import BaseWorkflow, WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)

__all__ = ["WorkflowEvaluator", "EvaluationResult", "PerformanceMetrics", "QualityMetrics"]


@dataclass
class PerformanceMetrics:
    """Performance evaluation metrics."""

    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    throughput_files_per_second: Optional[float] = None

    def meets_performance_criteria(self, criteria: Dict[str, float]) -> bool:
        """Check if performance meets specified criteria."""
        if "max_execution_time" in criteria:
            if self.execution_time_seconds > criteria["max_execution_time"]:
                return False

        if "max_memory_usage" in criteria and self.memory_usage_mb:
            if self.memory_usage_mb > criteria["max_memory_usage"]:
                return False

        return True


@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""

    confidence_score: float
    accuracy_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    coverage_percentage: Optional[float] = None
    false_positive_rate: Optional[float] = None

    def meets_quality_criteria(self, criteria: Dict[str, float]) -> bool:
        """Check if quality meets specified criteria."""
        if "min_confidence_score" in criteria:
            if self.confidence_score < criteria["min_confidence_score"]:
                return False

        if "min_coverage_percentage" in criteria and self.coverage_percentage:
            if self.coverage_percentage < criteria["min_coverage_percentage"]:
                return False

        return True


@dataclass
class EvaluationResult:
    """Result of workflow evaluation."""

    workflow_id: str
    timestamp: str
    overall_score: float  # 0.0 to 1.0

    # Detailed metrics
    performance: PerformanceMetrics
    quality: QualityMetrics

    # Compliance checks
    meets_criteria: bool
    criteria_violations: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)

    # Comparison data
    baseline_comparison: Optional[Dict[str, float]] = None
    trend_analysis: Optional[Dict[str, Any]] = None


class WorkflowEvaluator:
    """Evaluates workflow effectiveness and performance."""

    def __init__(self):
        self.evaluation_history: Dict[str, List[EvaluationResult]] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}

    def evaluate_workflow_execution(
        self, workflow: BaseWorkflow, result: WorkflowResult
    ) -> EvaluationResult:
        """Evaluate a completed workflow execution."""

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Extract performance metrics
        performance = PerformanceMetrics(
            execution_time_seconds=result.metrics.execution_time_seconds or 0.0,
            memory_usage_mb=result.metrics.memory_usage_mb,
            cpu_usage_percent=result.metrics.cpu_usage_percent,
            throughput_files_per_second=self._calculate_throughput(result.metrics),
        )

        # Extract quality metrics
        quality = QualityMetrics(
            confidence_score=result.metrics.confidence_score,
            accuracy_score=result.metrics.accuracy_score,
            coverage_percentage=result.metrics.coverage_percentage,
        )

        # Get evaluation criteria
        criteria = workflow.get_evaluation_criteria()

        # Check compliance
        meets_criteria = True
        violations = []

        if not performance.meets_performance_criteria(criteria.get("performance", {})):
            meets_criteria = False
            violations.append("Performance criteria not met")

        if not quality.meets_quality_criteria(criteria.get("quality", {})):
            meets_criteria = False
            violations.append("Quality criteria not met")

        # Calculate overall score
        overall_score = self._calculate_overall_score(performance, quality, result)

        # Generate recommendations
        recommendations = self._generate_recommendations(workflow, performance, quality, result)

        # Create evaluation result
        evaluation = EvaluationResult(
            workflow_id=workflow.workflow_id,
            timestamp=timestamp,
            overall_score=overall_score,
            performance=performance,
            quality=quality,
            meets_criteria=meets_criteria,
            criteria_violations=violations,
            recommendations=recommendations,
        )

        # Add baseline comparison if available
        if workflow.workflow_id in self.baselines:
            evaluation.baseline_comparison = self._compare_to_baseline(
                workflow.workflow_id, performance, quality
            )

        # Store evaluation
        if workflow.workflow_id not in self.evaluation_history:
            self.evaluation_history[workflow.workflow_id] = []
        self.evaluation_history[workflow.workflow_id].append(evaluation)

        # Update baseline if this is a good execution
        if overall_score > 0.8 and meets_criteria:
            self._update_baseline(workflow.workflow_id, performance, quality)

        return evaluation

    def _calculate_throughput(self, metrics) -> Optional[float]:
        """Calculate files processed per second."""
        if metrics.execution_time_seconds and metrics.files_processed:
            if metrics.execution_time_seconds > 0:
                return metrics.files_processed / metrics.execution_time_seconds
        return None

    def _calculate_overall_score(
        self, performance: PerformanceMetrics, quality: QualityMetrics, result: WorkflowResult
    ) -> float:
        """Calculate overall workflow score."""

        # Base score from execution status
        if result.status == WorkflowStatus.COMPLETED:
            base_score = 0.6
        elif result.status == WorkflowStatus.FAILED:
            return 0.0
        else:
            base_score = 0.3

        # Quality contribution (40% weight)
        quality_score = quality.confidence_score * 0.4

        # Performance contribution (20% weight)
        # Penalize slow execution
        performance_score = 0.2
        if performance.execution_time_seconds > 0:
            # Assume target is 30 seconds, linear penalty after that
            if performance.execution_time_seconds <= 30:
                performance_score = 0.2
            else:
                performance_score = max(0.0, 0.2 * (60 - performance.execution_time_seconds) / 30)

        # Findings contribution (20% weight)
        findings_score = 0.0
        if result.metrics.findings_generated > 0:
            # More findings generally better, up to a point
            findings_score = min(0.2, result.metrics.findings_generated * 0.02)

        return min(1.0, base_score + quality_score + performance_score + findings_score)

    def _generate_recommendations(
        self,
        workflow: BaseWorkflow,
        performance: PerformanceMetrics,
        quality: QualityMetrics,
        result: WorkflowResult,
    ) -> List[str]:
        """Generate improvement recommendations."""

        recommendations = []

        # Performance recommendations
        if performance.execution_time_seconds > 60:
            recommendations.append("Consider optimizing workflow execution time")

        if performance.memory_usage_mb and performance.memory_usage_mb > 1000:
            recommendations.append("Consider reducing memory usage")

        # Quality recommendations
        if quality.confidence_score < 0.7:
            recommendations.append(
                "Consider improving analysis confidence through better heuristics"
            )

        if result.metrics.errors_encountered > 0:
            recommendations.append("Address error handling to improve reliability")

        # Findings recommendations
        if result.metrics.findings_generated == 0:
            recommendations.append("Verify workflow is detecting issues appropriately")

        if quality.coverage_percentage and quality.coverage_percentage < 80:
            recommendations.append("Improve analysis coverage of target files")

        return recommendations

    def _compare_to_baseline(
        self, workflow_id: str, performance: PerformanceMetrics, quality: QualityMetrics
    ) -> Dict[str, float]:
        """Compare current metrics to baseline."""

        baseline = self.baselines[workflow_id]
        comparison = {}

        if "execution_time" in baseline:
            comparison["execution_time_ratio"] = (
                performance.execution_time_seconds / baseline["execution_time"]
            )

        if "confidence_score" in baseline:
            comparison["confidence_improvement"] = (
                quality.confidence_score - baseline["confidence_score"]
            )

        return comparison

    def _update_baseline(
        self, workflow_id: str, performance: PerformanceMetrics, quality: QualityMetrics
    ):
        """Update baseline metrics for workflow."""

        if workflow_id not in self.baselines:
            self.baselines[workflow_id] = {}

        baseline = self.baselines[workflow_id]

        # Update with exponential smoothing
        alpha = 0.3  # Smoothing factor

        if "execution_time" in baseline:
            baseline["execution_time"] = (
                alpha * performance.execution_time_seconds
                + (1 - alpha) * baseline["execution_time"]
            )
        else:
            baseline["execution_time"] = performance.execution_time_seconds

        if "confidence_score" in baseline:
            baseline["confidence_score"] = (
                alpha * quality.confidence_score + (1 - alpha) * baseline["confidence_score"]
            )
        else:
            baseline["confidence_score"] = quality.confidence_score

        logger.debug(f"Updated baseline for workflow {workflow_id}")

    def get_workflow_trends(self, workflow_id: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """Get trend analysis for workflow over specified period."""

        if workflow_id not in self.evaluation_history:
            return None

        evaluations = self.evaluation_history[workflow_id]
        if len(evaluations) < 2:
            return None

        # Simple trend analysis
        recent_evaluations = evaluations[-min(days, len(evaluations)) :]

        execution_times = [e.performance.execution_time_seconds for e in recent_evaluations]
        confidence_scores = [e.quality.confidence_score for e in recent_evaluations]
        overall_scores = [e.overall_score for e in recent_evaluations]

        trends = {
            "evaluation_count": len(recent_evaluations),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_confidence_score": sum(confidence_scores) / len(confidence_scores),
            "avg_overall_score": sum(overall_scores) / len(overall_scores),
            "performance_trend": self._calculate_trend(execution_times),
            "quality_trend": self._calculate_trend(confidence_scores),
            "overall_trend": self._calculate_trend(overall_scores),
        }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "degrading"
        else:
            return "stable"

    def generate_evaluation_report(
        self, workflow_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        if workflow_ids is None:
            workflow_ids = list(self.evaluation_history.keys())

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflows_evaluated": len(workflow_ids),
            "workflow_summaries": {},
        }

        for workflow_id in workflow_ids:
            if workflow_id in self.evaluation_history:
                evaluations = self.evaluation_history[workflow_id]
                latest = evaluations[-1] if evaluations else None

                if latest:
                    trends = self.get_workflow_trends(workflow_id)

                    report["workflow_summaries"][workflow_id] = {
                        "latest_score": latest.overall_score,
                        "meets_criteria": latest.meets_criteria,
                        "execution_count": len(evaluations),
                        "recommendations": latest.recommendations,
                        "trends": trends,
                    }

        return report

    def export_evaluation_data(self, output_path: Path):
        """Export evaluation data for analysis."""
        import json

        export_data = {
            "evaluation_history": {},
            "baselines": self.baselines,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Convert evaluation history to serializable format
        for workflow_id, evaluations in self.evaluation_history.items():
            export_data["evaluation_history"][workflow_id] = []
            for eval_result in evaluations:
                eval_dict = {
                    "workflow_id": eval_result.workflow_id,
                    "timestamp": eval_result.timestamp,
                    "overall_score": eval_result.overall_score,
                    "meets_criteria": eval_result.meets_criteria,
                    "performance": {
                        "execution_time_seconds": eval_result.performance.execution_time_seconds,
                        "memory_usage_mb": eval_result.performance.memory_usage_mb,
                        "throughput_files_per_second": eval_result.performance.throughput_files_per_second,
                    },
                    "quality": {
                        "confidence_score": eval_result.quality.confidence_score,
                        "coverage_percentage": eval_result.quality.coverage_percentage,
                    },
                    "recommendations": eval_result.recommendations,
                }
                export_data["evaluation_history"][workflow_id].append(eval_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Evaluation data exported to {output_path}")
```

---
### File: src/vibelint/workflows/implementations/__init__.py

```python
"""
Workflow implementations package.

Contains all concrete workflow implementations organized by functionality.

Responsibility: Workflow implementation organization only.
Individual workflow logic belongs in specific implementation modules.

vibelint/src/vibelint/workflow/implementations/__init__.py
"""

# Import available implementations - avoid circular imports by importing lazily
__all__ = [
    # Implementation modules
    "justification",
    "single_file_validation",
]

# Lazy imports to avoid circular dependencies
def get_justification_engine():
    """Get JustificationEngine class."""
    from .justification import JustificationEngine
    return JustificationEngine

def get_single_file_validation_workflow():
    """Get SingleFileValidationWorkflow class."""
    from .single_file_validation import SingleFileValidationWorkflow
    return SingleFileValidationWorkflow
```

---
### File: src/vibelint/workflows/implementations/justification.py

```python
"""
Clean, focused justification engine.

Core workflow:
1. Discover files and build filesystem tree
2. Summarize each file's purpose with fast LLM (cached by hash)
3. Generate XML context with structure + summaries
4. Orchestrator LLM analyzes for misplaced/useless/redundant files
"""

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JustificationEngine:
    """Clean justification engine focused on the essential workflow."""

    # Workflow metadata for registry
    workflow_id: str = "justification"
    name: str = "Code Justification Analysis"
    description: str = "Analyze code architecture for misplaced/useless/redundant files using LLM"
    version: str = "3.0"
    category: str = "analysis"
    tags: set = {"code-quality", "llm-analysis", "architecture"}

    def __init__(self, config: Optional["WorkflowConfig"] = None):
        from vibelint.workflows.core.base import WorkflowConfig
        self.config = config or WorkflowConfig()
        self.llm_manager = None
        self.cache_file = Path(".vibes/cache/file_summaries.json")
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.xml_output = Path(f".vibes/reports/project_analysis_{self.timestamp}.xml")
        self.jsonl_log_file = None
        self.master_log_file = None
        self.xml_file = None

        # Initialize LLM manager
        self._init_llm()

        # Set up logging (both JSONL for LLM calls and master log for process)
        self._setup_logging()

        # Load summary cache
        self.summary_cache = self._load_cache()

    def _init_llm(self):
        """Initialize LLM manager and get token limits from config."""
        try:
            from vibelint.llm_client import LLMManager, LLMRequest

            self.llm_manager = LLMManager()
            self.LLMRequest = LLMRequest

            # Get token limits from LLM config for proper chunking
            self.fast_max_tokens = self.llm_manager.llm_config.fast_max_tokens
            self.fast_max_context_tokens = self.llm_manager.llm_config.fast_max_context_tokens or (
                self.fast_max_tokens * 4
            )
            self.orchestrator_max_tokens = self.llm_manager.llm_config.orchestrator_max_tokens
            self.orchestrator_max_context_tokens = (
                self.llm_manager.llm_config.orchestrator_max_context_tokens or 131072
            )

            logger.info(
                f"LLM manager initialized (fast: {self.fast_max_tokens} output tokens, {self.fast_max_context_tokens} context tokens, orchestrator: {self.orchestrator_max_tokens} tokens)"
            )
        except Exception as e:
            logger.warning(f"LLM manager not available: {e}")
            # Fallback defaults if LLM not available
            self.fast_max_tokens = 2048
            self.fast_max_context_tokens = 8192
            self.orchestrator_max_tokens = 8192
            self.orchestrator_max_context_tokens = 131072

    def _setup_logging(self):
        """Set up both JSONL (LLM calls) and master log (process) files."""
        logs_dir = Path(".vibes/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Use the same timestamp as the report files for consistency
        # Master log file for human-readable process logging
        self.master_log_file = logs_dir / f"justification_{self.timestamp}.log"

        # JSONL log file for LLM prompt-response pairs
        if self.llm_manager:
            self.jsonl_log_file = logs_dir / f"justification_{self.timestamp}.jsonl"
            # Create the file immediately so it always exists
            self.jsonl_log_file.touch()

            # Register callback with LLM manager to log all requests/responses
            def log_callback(log_entry):
                """Write log entry to JSONL file."""
                if self.jsonl_log_file is None:
                    return
                try:
                    # Convert LogEntry dataclass to dict for JSON serialization
                    from dataclasses import asdict
                    log_dict = asdict(log_entry) if hasattr(log_entry, '__dataclass_fields__') else log_entry
                    with open(self.jsonl_log_file, "a") as f:
                        f.write(json.dumps(log_dict) + "\n")
                except Exception as e:
                    logger.debug(f"Failed to write JSONL log: {e}")

            self.llm_manager.set_log_callback(log_callback)
            self._log(f"JSONL logging enabled: {self.jsonl_log_file}")

        self._log(f"Master log file: {self.master_log_file}")

    def _log(self, message: str):
        """Write to master log file with timestamp."""
        if self.master_log_file:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.master_log_file, "a") as f:
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                logger.debug(f"Failed to write master log: {e}")
        # Also log to Python logger
        logger.info(message)

    def _load_cache(self) -> Dict[str, str]:
        """Load file summary cache."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save file summary cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.summary_cache, indent=2))

    def _get_file_hash(self, file_path: Path) -> str:
        """Get file content hash for caching."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]

    def _discover_files(self, root: Path) -> List[Path]:
        """Discover all relevant files using vibelint's discovery system."""
        from vibelint.config import load_config
        from vibelint.discovery import discover_files

        # Load config for the project
        config = load_config(root)

        # Use vibelint's discover_files with the project root
        files = discover_files(paths=[root], config=config, explicit_exclude_paths=set())

        # Filter out very large files
        return sorted([f for f in files if f.stat().st_size < 2 * 1024 * 1024])

    def _build_tree_xml(self, root: Path, files: List[Path]) -> ET.Element:
        """Build XML tree structure."""
        project_elem = ET.Element("project", name=root.name, path=str(root))

        # Group files by directory
        dir_structure = {}
        for file_path in files:
            relative = file_path.relative_to(root)
            parts = relative.parts

            current = dir_structure
            for part in parts[:-1]:  # All but the filename
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add the file
            filename = parts[-1]
            current[filename] = file_path

        def add_to_xml(parent_elem, structure, base_path=""):
            for name, content in sorted(structure.items()):
                if isinstance(content, dict):
                    # It's a directory
                    dir_elem = ET.SubElement(
                        parent_elem, "directory", name=name, path=f"{base_path}/{name}".strip("/")
                    )
                    add_to_xml(dir_elem, content, f"{base_path}/{name}".strip("/"))
                else:
                    # It's a file
                    file_path = content
                    file_elem = ET.SubElement(
                        parent_elem,
                        "file",
                        name=name,
                        path=str(file_path.relative_to(root)),
                        size=str(file_path.stat().st_size),
                    )

                    # Add file summary if available
                    file_hash = self._get_file_hash(file_path)
                    summary = self.summary_cache.get(file_hash, "")
                    if summary:
                        summary_elem = ET.SubElement(file_elem, "summary")
                        summary_elem.text = summary

        add_to_xml(project_elem, dir_structure)
        return project_elem

    def _chunk_content(self, content: str, max_chars: int) -> List[str]:
        """Chunk content to fit in LLM context window."""
        if len(content) <= max_chars:
            return [content]

        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _summarize_file(self, file_path: Path) -> str:
        """Summarize file purpose using fast LLM with proper chunking."""
        from vibelint.filesystem import is_binary

        if not self.llm_manager:
            return "[LLM not available]"

        # Check if file is binary
        if is_binary(file_path):
            return "[Binary file - cannot summarize content]"

        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                return "[Empty file]"

            # Calculate max content size for fast LLM context window
            # Use 3 chars/token estimation, reserve space for prompt overhead (~100 tokens)
            prompt_overhead_tokens = 100
            max_content_tokens = self.fast_max_context_tokens - prompt_overhead_tokens
            max_content_chars = max_content_tokens * 3

            chunks = self._chunk_content(content, max_content_chars)

            # Analyze each chunk and concatenate summaries (no synthesis step for speed)
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if len(chunks) == 1:
                    chunk_prompt = f"""File: {file_path.name}

```
{chunk}
```

Output a 1-2 sentence summary of what this file does. No preamble, just the summary:"""
                else:
                    chunk_prompt = f"""File: {file_path.name} (part {i+1}/{len(chunks)})

```
{chunk}
```

One sentence describing what this code does:"""

                request = self.LLMRequest(
                    content=chunk_prompt, max_tokens=self.fast_max_tokens, temperature=0.1
                )

                # LLM manager automatically cascades to orchestrator if fast fails
                response = self.llm_manager.process_request_sync(request)

                if response and response.success and response.content:
                    chunk_summaries.append(response.content.strip())
                else:
                    chunk_summaries.append(f"[Chunk {i+1} analysis failed]")

            # Concatenate all chunk summaries
            return " ".join(chunk_summaries)

        except Exception as e:
            logger.warning(f"Failed to summarize {file_path}: {e}")
            return f"[Error: {e}]"

    def _chunk_xml_semantically(self, xml_content: str) -> List[str]:
        """Chunk XML content by directory structure to fit in orchestrator context."""
        # Parse XML to extract file entries
        root = ET.fromstring(xml_content)

        # Group files by their top-level directory
        dir_groups = {}
        for file_elem in root.findall(".//file"):
            file_path = file_elem.get("path", "")
            parts = file_path.split("/")
            top_dir = parts[0] if len(parts) > 1 else "root"

            if top_dir not in dir_groups:
                dir_groups[top_dir] = []

            # Serialize this file element
            file_xml = ET.tostring(file_elem, encoding="unicode")
            dir_groups[top_dir].append(file_xml)

        # Calculate max chunk size (90% of orchestrator context for prompt overhead)
        max_chunk_tokens = int(self.orchestrator_max_context_tokens * 0.9)
        max_chunk_chars = max_chunk_tokens * 3

        # Build chunks from directory groups
        chunks = []
        current_chunk = []
        current_size = 0

        project_header = f'<project name="{root.get("name")}" path="{root.get("path")}">\n'
        project_footer = "</project>"
        header_size = len(project_header) + len(project_footer)

        for dir_name, files in sorted(dir_groups.items()):
            dir_content = "\n".join(files)
            dir_size = len(dir_content)

            # If adding this directory would exceed limit, start new chunk
            if current_size + dir_size + header_size > max_chunk_chars and current_chunk:
                chunk_xml = project_header + "\n".join(current_chunk) + "\n" + project_footer
                chunks.append(chunk_xml)
                current_chunk = [dir_content]
                current_size = dir_size
            else:
                current_chunk.append(dir_content)
                current_size += dir_size

        # Add final chunk
        if current_chunk:
            chunk_xml = project_header + "\n".join(current_chunk) + "\n" + project_footer
            chunks.append(chunk_xml)

        return chunks

    def _analyze_xml_chunk(
        self, chunk_xml: str, chunk_num: int, total_chunks: int, root: Path
    ) -> str:
        """Analyze a single XML chunk with orchestrator LLM."""
        if not self.llm_manager:
            return "LLM not available for chunk analysis"

        chunk_context = (
            f" (analyzing part {chunk_num} of {total_chunks})" if total_chunks > 1 else ""
        )

        prompt = f"""Analyze this project structure{chunk_context} to identify issues:

**PROJECT ROOT PRINCIPLE**: A well-organized Python project keeps its root directory minimal and purposeful.

**Root directory should ONLY contain**:
- Project metadata: setup.py, pyproject.toml, MANIFEST.in, tox.ini, LICENSE
- Documentation: README.md (single entry point)
- Configuration: .gitignore, .env.example
- Package entry points: __init__.py, __main__.py, conftest.py (for testing)

**Everything else belongs in subdirectories**:
- Source code → src/ or package_name/
- Scripts and utilities → scripts/ or tools/
- Documentation → docs/
- Tests → tests/
- Examples → examples/

1. **Misplaced files**: Files in wrong directories based on their purpose
   - **Scan file paths**: Files at root level have NO `/` in their path attribute
   - **Evaluate purpose from summary**: Does this file's functionality belong at project root?
   - Common misplacements:
     * Scripts that transform/convert/modify files → should be in scripts/ or tools/
     * Utility code that's imported by the project → should be in src/
     * Additional documentation → should be in docs/
     * Test helpers or fixtures → should be in tests/

2. **Useless files**: Dead code, unused files, or files with no clear purpose
   - One-time migration scripts that have completed their purpose
   - Backup files (*.bak, *_old.py, *_backup.py)
   - Duplicate files with similar names
   - Files that are never imported or executed

3. **Redundant files**: Files with duplicate or overlapping functionality
   - Multiple files with similar summaries or purposes
   - Duplicate utility functions across modules
   - Multiple documentation files covering the same topic

4. **Overly large files**: Files >25000 bytes indicate poor modularity (check size attribute)
   - Files that could be split into logical components
   - Orchestrators or workflows that should be decomposed

5. **Consolidation opportunities**: Files that could be merged or deduplicated
   - Similar validators or workflows
   - Related documentation that should be combined

**ANALYSIS APPROACH**:
1. First, scan all files with path containing no `/` → these are at root level
2. For each root-level file, evaluate: "Should this be at root based on the principles above?"
3. If NO, flag it as misplaced with its actual purpose and recommended location
4. Then analyze the rest of the structure for other categories

Project: {root.name}

{chunk_xml}

Provide specific findings for each category with file paths and consolidation recommendations:"""

        try:
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_manager.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                return f"Chunk {chunk_num} analysis failed"

        except Exception as e:
            logger.error(f"Chunk {chunk_num} analysis failed: {e}")
            return f"Chunk {chunk_num} error: {e}"

    def _build_tree_structure(self, files: List[Path], project_root: Path) -> str:
        """Build a tree-style representation of the file structure (paths only)."""
        tree_lines = []

        for file_path in sorted(files):
            rel_path = file_path.resolve().relative_to(project_root)
            tree_lines.append(str(rel_path))

        return "\n".join(tree_lines)

    def _load_project_rules(self, project_root: Path) -> str:
        """Load project-specific justification rules from AGENTS.instructions.md."""
        agents_file = project_root / "AGENTS.instructions.md"

        if not agents_file.exists():
            return ""

        try:
            content = agents_file.read_text()

            # Extract the "File Organization & Project Structure Rules" section
            if "## File Organization & Project Structure Rules" in content:
                # Find the section
                start = content.find("## File Organization & Project Structure Rules")
                # Find the next ## heading or end of file
                next_section = content.find("\n## ", start + 1)

                if next_section == -1:
                    rules_section = content[start:]
                else:
                    rules_section = content[start:next_section]

                return rules_section.strip()

            return ""
        except Exception as e:
            logger.warning(f"Failed to load project rules from AGENTS.instructions.md: {e}")
            return ""

    def _analyze_structure(self, files: List[Path], project_root: Path) -> str:
        """Phase 1: Analyze project structure based on file paths alone."""
        if not self.llm_manager:
            return "LLM not available for structural analysis"

        tree = self._build_tree_structure(files, project_root)
        project_rules = self._load_project_rules(project_root)

        if project_rules:
            self._log("Loaded project-specific rules from AGENTS.instructions.md")

        # Build base prompt
        base_prompt = """Analyze this project's file structure for organizational issues.

**PROJECT ROOT PRINCIPLE**: A well-organized Python project keeps its root directory minimal.

**Root directory should ONLY contain**:
- Project metadata: setup.py, pyproject.toml, MANIFEST.in, tox.ini, LICENSE
- Single documentation entry point: README.md
- Configuration: .gitignore, .env.example
- Package entry points: __init__.py, __main__.py, conftest.py
- AI/Agent instructions: CLAUDE.md, AGENTS.instructions.md, *.instructions.md

**Everything else belongs in subdirectories**:
- Source code → src/ or package_name/
- Documentation → docs/
- Tests → tests/
- Examples → examples/"""

        # Add project-specific rules if found
        if project_rules:
            prompt = f"""{base_prompt}

**PROJECT-SPECIFIC RULES** (from AGENTS.instructions.md):
{project_rules}

**File Structure** ({len(files)} files):
```
{tree}
```

**Task**: Identify files that are misplaced based on BOTH the general principles above AND the project-specific rules. For each misplaced file:
1. State the file path
2. Explain why it's misplaced (what principle or project rule it violates)
3. Suggest where it should be moved OR if it should be deleted

Focus on:
- Root-level files that don't belong there
- Files in wrong subdirectories
- Forbidden directories (per project rules)
- One-off scripts that should be deleted or integrated

Be specific and direct. List misplaced files clearly."""
        else:
            prompt = f"""{base_prompt}

**File Structure** ({len(files)} files):
```
{tree}
```

**Task**: Identify files that are misplaced based ONLY on their location. For each misplaced file:
1. State the file path
2. Explain why it's misplaced (what principle it violates)
3. Suggest where it should be moved

Focus on:
- Root-level files that don't belong there
- Files in wrong subdirectories
- Missing organizational structure

Be specific and direct. List misplaced files clearly."""

        try:
            self._log("Running structural analysis (file paths only)...")
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_manager.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                return "Structural analysis failed"

        except Exception as e:
            logger.error(f"Structural analysis failed: {e}")
            return f"Structural analysis error: {e}"

    def _synthesize_chunk_analyses(
        self, structural_analysis: str, chunk_analyses: List[str], root: Path
    ) -> str:
        """Synthesize structural + semantic analyses into final report."""
        if not self.llm_manager:
            return (
                f"## Structural Analysis\n\n{structural_analysis}\n\n"
                + "\n\n---\n\n".join(chunk_analyses)
            )

        combined_semantic = "\n\n---\n\n".join(
            [f"**Semantic Analysis Part {i+1}:**\n{analysis}" for i, analysis in enumerate(chunk_analyses)]
        )

        prompt = f"""Synthesize this multi-phase project analysis into a comprehensive report:

**PHASE 1: Structural Analysis (file organization)**
{structural_analysis}

**PHASE 2: Semantic Analysis (file content and purpose)**
{combined_semantic}

Create a unified report that:
1. Combines structural and semantic findings (avoid duplication)
2. Cross-references findings from both phases
3. Provides prioritized, actionable recommendations
4. Groups related issues together

Project: {root.name}"""

        try:
            request = self.LLMRequest(
                content=prompt, max_tokens=self.orchestrator_max_tokens, temperature=0.2
            )

            response = self.llm_manager.process_request_sync(request)
            if response and response.success and response.content:
                return response.content.strip()
            else:
                # Fallback to simple concatenation
                return (
                    f"## Structural Analysis\n\n{structural_analysis}\n\n"
                    + f"## Semantic Analysis\n\n{combined_semantic}"
                )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return (
                f"## Structural Analysis\n\n{structural_analysis}\n\n"
                + f"## Semantic Analysis\n\n{combined_semantic}"
            )

    def run_justification(self, project_root: Path) -> Dict:
        """Run the complete justification workflow."""
        try:
            start_time = time.time()
            # Ensure project_root is absolute
            project_root = project_root.resolve()
            self._log(f"Starting justification analysis for: {project_root}")

            # Step 1: Discover files
            files = self._discover_files(project_root)
            self._log(f"Discovered {len(files)} files")

            # Step 2: Initialize XML file for streaming
            self.xml_output.parent.mkdir(parents=True, exist_ok=True)
            with open(self.xml_output, "w") as xml_file:
                xml_file.write(f'<project name="{project_root.name}" path="{project_root}">\n')

                # Step 3: Summarize files and write to XML incrementally
                cache_hits = 0
                cache_misses = 0

                for idx, file_path in enumerate(files, 1):
                    file_hash = self._get_file_hash(file_path)
                    # Ensure file_path is absolute before making relative
                    rel_path = file_path.resolve().relative_to(project_root)

                    # Get or generate summary
                    if file_hash in self.summary_cache:
                        summary = self.summary_cache[file_hash]
                        cache_hits += 1
                        print(f"[{idx}/{len(files)}] Cached: {rel_path}")
                    else:
                        print(f"[{idx}/{len(files)}] Summarizing: {rel_path}")
                        self._log(f"Summarizing: {rel_path}")
                        summary = self._summarize_file(file_path)
                        self.summary_cache[file_hash] = summary
                        cache_misses += 1
                        # Save cache after each new summary
                        self._save_cache()

                    # Write file entry to XML immediately with CDATA wrapping
                    xml_file.write(
                        f'  <file path="{rel_path}" size="{file_path.stat().st_size}">\n'
                    )
                    xml_file.write(f"    <summary><![CDATA[{summary}]]></summary>\n")
                    xml_file.write("  </file>\n")
                    xml_file.flush()  # Force write to disk

                # Close project tag
                xml_file.write("</project>\n")

            self._log(f"File summaries: {cache_hits} cached, {cache_misses} new")
            self._log(f"XML written to: {self.xml_output}")

            # Step 4: PHASE 1 - Structural analysis (file paths only)
            self._log("=" * 60)
            self._log("PHASE 1: Structural Analysis (file organization)")
            self._log("=" * 60)
            structural_analysis = self._analyze_structure(files, project_root)

            # Step 5: PHASE 2 - Semantic analysis (file content/purpose)
            self._log("=" * 60)
            self._log("PHASE 2: Semantic Analysis (file content)")
            self._log("=" * 60)

            xml_content = self.xml_output.read_text()
            xml_size_chars = len(xml_content)
            xml_size_tokens = xml_size_chars // 3

            self._log(f"XML size: {xml_size_chars:,} chars (~{xml_size_tokens:,} tokens)")

            # Chunk XML semantically by directory structure
            xml_chunks = self._chunk_xml_semantically(xml_content)
            self._log(f"XML chunked into {len(xml_chunks)} parts for semantic analysis")

            # Analyze each chunk
            chunk_analyses = []
            for i, chunk in enumerate(xml_chunks):
                self._log(f"Analyzing semantic chunk {i+1}/{len(xml_chunks)}...")
                chunk_analysis = self._analyze_xml_chunk(
                    chunk, i + 1, len(xml_chunks), project_root
                )
                chunk_analyses.append(chunk_analysis)

            # Step 6: Synthesize both phases into final report
            self._log("=" * 60)
            self._log("SYNTHESIS: Combining structural + semantic findings")
            self._log("=" * 60)
            analysis = self._synthesize_chunk_analyses(
                structural_analysis, chunk_analyses, project_root
            )

            # Step 5: Generate final report
            duration = time.time() - start_time
            report = f"""# Project Justification Analysis

**Project:** {project_root.name}
**Files Analyzed:** {len(files)}
**Cache Performance:** {cache_hits} hits, {cache_misses} misses
**Analysis Time:** {duration:.1f}s

## Orchestrator Analysis

{analysis}

## Project Structure

See: {self.xml_output}
"""

            # Save report with timestamp
            report_file = Path(f".vibes/reports/justification_analysis_{self.timestamp}.md")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(report)

            logger.info(f"Analysis complete in {duration:.1f}s")

            return {
                "success": True,
                "exit_code": 0,
                "report": report,
                "files_analyzed": len(files),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "duration": duration,
            }

        except Exception as e:
            logger.error(f"Justification analysis failed: {e}")
            return {
                "success": False,
                "exit_code": 1,
                "error": str(e),
                "report": f"Analysis failed: {e}",
            }


# Alias for registry compatibility
JustificationWorkflow = JustificationEngine
```

---
### File: src/vibelint/workflows/registry.py

```python
"""
Workflow registry for managing available workflows.

Provides centralized registration and discovery of workflows with
metadata and dependency information.

Responsibility: Workflow discovery and registration only.
Workflow logic belongs in individual workflow implementation modules.

vibelint/src/vibelint/workflow/registry.py
"""

import logging
from typing import Dict, List, Optional, Type

from .core.base import BaseWorkflow

logger = logging.getLogger(__name__)

__all__ = ["WorkflowRegistry", "workflow_registry", "register_workflow"]


class WorkflowRegistry:
    """Registry for managing available workflows."""

    def __init__(self):
        self._workflows: Dict[str, Type[BaseWorkflow]] = {}
        self._metadata: Dict[str, Dict] = {}
        self._builtin_loaded = False
        self._plugins_loaded = False

    def register(self, workflow_class: Type[BaseWorkflow]) -> None:
        """Register a workflow class."""
        # Create temporary instance to get metadata
        temp_instance = workflow_class()
        workflow_id = temp_instance.workflow_id

        if not workflow_id:
            raise ValueError(f"Workflow {workflow_class.__name__} must define workflow_id")

        if workflow_id in self._workflows:
            logger.warning(f"Overwriting existing workflow: {workflow_id}")

        self._workflows[workflow_id] = workflow_class

        # Store metadata
        self._metadata[workflow_id] = {
            "name": temp_instance.name,
            "description": temp_instance.description,
            "category": temp_instance.category,
            "version": temp_instance.version,
            "tags": list(temp_instance.tags),
            "required_inputs": list(temp_instance.get_required_inputs()),
            "produced_outputs": list(temp_instance.get_produced_outputs()),
            "dependencies": temp_instance.get_dependencies(),
            "supports_parallel": temp_instance.supports_parallel_execution(),
            "class_name": workflow_class.__name__,
        }

        logger.debug(f"Registered workflow: {workflow_id}")

    def get_workflow(self, workflow_id: str) -> Optional[Type[BaseWorkflow]]:
        """Get workflow class by ID."""
        self.ensure_loaded()
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> Dict[str, Type[BaseWorkflow]]:
        """Get all registered workflows."""
        self.ensure_loaded()
        return self._workflows.copy()

    def get_workflows_by_category(self, category: str) -> Dict[str, Type[BaseWorkflow]]:
        """Get workflows by category."""
        filtered = {}
        for workflow_id, metadata in self._metadata.items():
            if metadata["category"] == category:
                filtered[workflow_id] = self._workflows[workflow_id]
        return filtered

    def get_workflows_by_tag(self, tag: str) -> Dict[str, Type[BaseWorkflow]]:
        """Get workflows by tag."""
        filtered = {}
        for workflow_id, metadata in self._metadata.items():
            if tag in metadata["tags"]:
                filtered[workflow_id] = self._workflows[workflow_id]
        return filtered

    def get_workflow_metadata(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow metadata."""
        return self._metadata.get(workflow_id)

    def list_workflow_ids(self) -> List[str]:
        """List all workflow IDs."""
        self.ensure_loaded()
        return list(self._workflows.keys())

    def validate_dependencies(self, workflow_ids: List[str]) -> Dict[str, List[str]]:
        """Validate workflow dependencies."""
        missing_deps = {}

        for workflow_id in workflow_ids:
            if workflow_id not in self._workflows:
                missing_deps[workflow_id] = [f"Workflow '{workflow_id}' not found"]
                continue

            metadata = self._metadata[workflow_id]
            for dep_id in metadata["dependencies"]:
                if dep_id not in self._workflows:
                    if workflow_id not in missing_deps:
                        missing_deps[workflow_id] = []
                    missing_deps[workflow_id].append(f"Missing dependency: '{dep_id}'")

        return missing_deps

    def unregister(self, workflow_id: str) -> bool:
        """Unregister a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            del self._metadata[workflow_id]
            logger.debug(f"Unregistered workflow: {workflow_id}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered workflows."""
        self._workflows.clear()
        self._metadata.clear()
        self._builtin_loaded = False
        self._plugins_loaded = False
        logger.debug("Cleared all workflows from registry")

    def _load_builtin_workflows(self) -> None:
        """Load built-in workflows directly."""
        if self._builtin_loaded:
            return

        # Import and register built-in workflows
        try:
            from .implementations.justification import JustificationWorkflow
            self.register(JustificationWorkflow)
            logger.debug("Registered built-in workflow: justification")
        except ImportError as e:
            logger.warning(f"Failed to load built-in justification workflow: {e}")

        try:
            from .implementations.deadcode import DeadcodeWorkflow
            self.register(DeadcodeWorkflow)
            logger.debug("Registered built-in workflow: deadcode")
        except ImportError as e:
            logger.warning(f"Failed to load built-in deadcode workflow: {e}")

        self._builtin_loaded = True

    def _load_plugin_workflows(self) -> None:
        """Load plugin workflows from entry points."""
        if self._plugins_loaded:
            return

        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points("vibelint.workflows"):
                try:
                    workflow_class = entry_point.load()
                    self.register(workflow_class)
                    logger.debug(f"Registered plugin workflow: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin workflow {entry_point.name}: {e}")
        except ImportError:
            logger.debug("pkg_resources not available, skipping plugin workflows")

        self._plugins_loaded = True

    def ensure_loaded(self) -> None:
        """Ensure both built-in and plugin workflows are loaded."""
        self._load_builtin_workflows()
        self._load_plugin_workflows()


# Global registry instance
workflow_registry = WorkflowRegistry()


def register_workflow(workflow_class: Type[BaseWorkflow]) -> Type[BaseWorkflow]:
    """Decorator for registering workflows."""
    workflow_registry.register(workflow_class)
    return workflow_class


# Note: Built-in workflows are now registered via workflow_registry._load_builtin_workflows()
# Plugin workflows are loaded via workflow_registry._load_plugin_workflows()
# This ensures clean separation between built-in and external workflows
```

---
### File: tox.ini

```ini
[tox]
envlist = py310, py311, py312, ruff, black
isolated_build = True

[gh-actions]
python =
    3.10: py310, ruff, black
    3.11: py311
    3.12: py312

[testenv]
deps =
    pytest>=7.0.0
    pytest-cov>=4.0.0
    pytest-asyncio>=0.21.0
commands =
    pytest {posargs:tests} --cov=vibelint --cov-report=xml

[testenv:py310]
basepython = python3.10

[testenv:py311]
basepython = python3.11

[testenv:py312]
basepython = python3.12

[testenv:ruff]
basepython = python3.10
deps = ruff>=0.1.0
commands = ruff check src tests

[testenv:black]
basepython = python3.10
deps = black>=23.0.0
commands = black --check src tests

# ruff configuration is now in pyproject.toml
```

---

