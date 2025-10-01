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

### Standard Testing Pipeline
1. **Run existing tests**: `tox -e py311` (or appropriate Python version)
2. **Check coverage**: Tests should maintain high pass rate and reasonable coverage
3. **Run agentic test assessment**: `vibelint coverage-vibe src/ tests/ --assess-tests`

### Agentic Test Quality Assessment
When implementing new features or fixing bugs:

1. **Self-Validating Assessment**:
   ```bash
   vibelint coverage-vibe src/vibelint/ tests/ --assess-tests --max-suggestions 10
   ```

2. **Review AI Assessment Results**:
   - Input validation scores (target: >0.8)
   - Output validation scores (target: >0.7)
   - Behavioral issues identified
   - Missing requirements discovered

3. **Address High-Priority Issues**:
   - Fix tests with confidence scores <0.6
   - Add tests for uncovered requirements
   - Implement AI-suggested improvements

4. **Validate Changes**:
   - Re-run assessment after improvements
   - Ensure scores improve and issues decrease
   - Verify requirements coverage increases

### Test Development Guidelines
- **Behavior-driven testing**: Validate behavior, not just coverage
- **Human decision point testing**: Test all human checkpoint interactions
- **Multi-model integration testing**: Test four-model coordination
- **Performance benchmarks**: Validate speed requirements
- **Requirements traceability**: Each test maps to specific requirements

### Vibelint-Specific Testing Patterns
- **Workflow testing**: Test complete workflows from `VIBELINT_WORKFLOWS.md`
- **Multi-representation testing**: Test filesystem/vector/graph/runtime building
- **Safety integration testing**: Test kaia-guardrails integration points
- **Human interaction testing**: Simulate and validate human decision points

**Reference**: See `DEVELOPMENT_METHODOLOGY.md` for comprehensive testing strategy and quality gates.