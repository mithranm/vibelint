# Vibelint Development Instructions

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