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

**Standard tools (use these instead of custom validators):**

**IMPORTANT - Run formatters in this exact order to avoid conflicts:**

1. **isort** - Import sorting (run first)
2. **black** - Code formatting (run second - overrides isort's multi-line imports)
3. **ruff check --fix** - Fast Python linter
   - `ruff check --select D` - Docstring checks (replaces docstring.py validator)
   - `ruff check --select T201` - Print statement detection (replaces print_statements.py validator)
   - `ruff check --select TID` - Import style enforcement (replaces relative_imports.py validator)
   - `ruff check --select C901` - Complexity checks (replaces code_smells.py validator)
4. **pyright** - Static type checking
5. **vulture** - Dead code detection (replaces dead_code.py validator)

**Why this order matters:**
- isort may wrap long import lines across multiple lines
- black prefers imports on a single line and will reformat them
- Running black after isort ensures consistent formatting

### Removed Validators (Use Standard Tools Instead)

We removed these validators because standard tools do the job better:

| Removed Validator | Replacement | Command |
|------------------|-------------|---------|
| `docstring.py` | ruff D rules (pydocstyle) | `ruff check --select D` |
| `print_statements.py` | ruff T201 | `ruff check --select T201` |
| `relative_imports.py` | ruff TID rules | `ruff check --select TID` |
| `dead_code.py` | vulture | `vulture src/ --min-confidence 80` |
| `code_smells.py` | ruff C901 complexity | `ruff check --select C901` |
| `module_cohesion.py` | Not actionable | N/A (deleted) |
| `namespace_report.py` | Dead code | N/A (deleted) |

### Remaining Custom Validators

Vibelint keeps only validators that provide unique value not covered by standard tools:

- **dict_get_fallback.py** - Enforces `dict.get(key, default)` with explicit defaults
- **emoji.py** - Detects emoji usage in code (project-specific preference)
- **exports.py** - Enforces `__all__` declarations in modules
- **logger_names.py** - Validates `logging.getLogger(__name__)` conventions
- **typing_quality.py** - Detects design smells (tuple→dataclass refactoring opportunities)
- **api_consistency.py** - Cross-module API consistency analysis
- **namespace_collisions.py** - Namespace conflict detection
- **strict_config.py** - Vibelint configuration validation
- **self_validation.py** - Meta-validation for vibelint's own code

### Dead Code Detection
Run `vulture` to find unused code:
```bash
vulture src/vibelint/ --min-confidence 80
```

**Tips**:
- `--min-confidence 80`: Reduces false positives (default 60 is noisy)
- Review findings carefully - vulture can't detect dynamic usage (getattr, importlib, etc.)
- Mark intentional unused code with `# noqa: vulture` comments
- Common false positives: `__all__` exports, plugin entry points, abstract methods

### Standard Testing Pipeline
1. **Run existing tests**: `tox -e py311` (or appropriate Python version)
2. **Check coverage**: Tests should maintain high pass rate and reasonable coverage