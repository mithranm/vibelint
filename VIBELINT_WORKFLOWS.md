# Vibelint Core Workflows

*Human-in-the-loop orchestration for rapid POC development*

**Philosophy:** Human-in-the-loop orchestration where users make strategic decisions while agents handle execution. Not autopilot - intelligent assistance with human decision points at critical junctions.

## Workflow 1: Single File Validation

**Trigger:** User runs `vibelint validate file.py`

**Activities:**
1. **File Loading**
   - Read Python file from filesystem
   - Validate file exists and is readable
   - Parse Python AST

2. **Single File Validator Execution**
   - Load validators from `src/vibelint/validators/single_file/`
   - Run each validator against the file:
     * `emoji.py` - Check for emoji usage violations
     * `docstring.py` - Validate docstring presence/format
     * `naming.py` - Check naming conventions
     * `complexity.py` - Calculate cyclomatic complexity
   - Each validator returns structured violations

3. **Violation Aggregation**
   - Collect all violations from validators
   - Sort by line number and severity
   - Apply any ignore rules or exceptions

4. **Report Generation**
   - Output violations in requested format
   - Include fix suggestions where deterministic
   - Provide file-level health score

**Output:** File validation report with specific, actionable violations

---

## Workflow 2: Multi-Representation Analysis (The Real Meat)

**Trigger:** Human orchestrator runs `vibelint analyze` when they need deep insight for next development step

**Human Decision Points:**
- **Scope Selection**: Human chooses which parts of codebase to analyze deeply
- **Model Configuration**: Human selects which of the four models to use based on budget/time
- **Analysis Depth**: Human decides how deep to go (quick overview vs comprehensive analysis)

**Activities:**
1. **Project Discovery & Setup**
   - Find Python project root (pyproject.toml, setup.py, .git)
   - Discover all .py files (respect .gitignore)
   - **HUMAN CHECKPOINT**: Show discovered files, let human exclude/include specific areas
   - Initialize four-model system (VanguardOne, VanguardTwo, Chip, Claudia)

2. **Multi-Representation Building**
   - **Filesystem Representation**: Build complete file tree with metadata
   - **Vector Representation**:
     * Embed code chunks using VanguardOne/VanguardTwo
     * Store in Qdrant vector database with timestamps for EBR
     * **HUMAN CHECKPOINT**: Show embedding clusters, let human label/categorize patterns
   - **Graph Representation**:
     * Build NetworkX dependency graph from imports/calls
     * **HUMAN CHECKPOINT**: Visualize graph, let human identify critical paths to focus on
     * Enhance with embedding similarities based on human feedback
   - **Runtime Representation**: Mock execution tracing with call patterns

3. **Four-Model Orchestration**
   - **VanguardOne/VanguardTwo**: Generate embeddings for semantic analysis
   - **Chip/Claudia**: Process embeddings + graph for architectural insights
   - **HUMAN CHECKPOINT**: Review model outputs, resolve conflicts manually
   - Coordinate between models for consensus on complex issues
   - Use memory system for Evidence-Based Reasoning on conflicts

4. **Context Engineering for Human Consumption**
   - **HUMAN CHECKPOINT**: Present analysis summary, let human choose focus areas
   - Synthesize filesystem + vector + graph + runtime insights
   - Create rich context packages for next development steps
   - **HUMAN CHECKPOINT**: Review and refine context before using with external agents
   - Generate issue metrics with severity and interconnection analysis
   - Prepare LLM-ready context for external consumption

**Output:** Human-validated comprehensive analysis with curated context ready for next POC development iteration

---

## Workflow 3: Deterministic Fix Application

**Trigger:** User runs `vibelint fix` or `vibelint fix --auto`

**Activities:**
1. **Fixable Violation Filtering**
   - Load violations from Workflow 1 (single file) or summary from Workflow 2
   - Filter for deterministic fixes only:
     * Emoji removal/replacement
     * Missing docstring insertion
     * Simple formatting issues
     * Obvious naming convention fixes

2. **Deterministic Fix Application**
   - Apply text-based fixes (no LLM/nondeterministic outputs)
   - Remove emojis: `🚀` → `` (deletion)
   - Add docstrings: Insert template docstrings for missing ones
   - Fix naming: `CamelCase` → `snake_case` for variables
   - Each fix is guaranteed safe and reversible

3. **Verification**
   - Re-parse AST to ensure syntax still valid
   - Re-run single file validators to confirm fixes applied
   - Report success/failure for each attempted fix

4. **Summary Report**
   - List all fixes applied
   - Show before/after examples
   - Identify any violations that couldn't be auto-fixed

**Output:** Modified files with deterministic fixes applied and verification results

---

## Workflow 4: Watch Mode (Real Linters Do This)

**Trigger:** User runs `vibelint watch`

**Activities:**
1. **File System Monitoring**
   - Watch .py files in project directory
   - Debounce rapid changes (save spam detection)
   - Trigger on file save events

2. **Incremental Validation**
   - Run Workflow 1 (single file validation) on changed files
   - Cache results for unchanged files
   - Update violation status in real-time

3. **Live Feedback**
   - Display violations in terminal as they occur
   - Show health score changes
   - Optionally auto-fix deterministic issues

4. **Integration with IDEs**
   - Output in Language Server Protocol format
   - Provide real-time diagnostics
   - Support quick-fix suggestions

**Output:** Continuous validation feedback during development

---

## Workflow 5: Smoke Testing

**Trigger:** User runs `vibelint smoke-test`

**Activities:**
1. **Basic Functionality Verification**
   - Import all Python modules in project
   - Verify no syntax errors
   - Check basic function signatures are callable

2. **Dependency Resolution**
   - Verify all imports resolve
   - Check for circular import issues
   - Validate external dependencies available

3. **Configuration Validation**
   - Test vibelint configuration is valid
   - Verify validators can load and run
   - Check external service connections (Qdrant, models)

4. **Lightweight Performance Check**
   - Time basic operations (file parsing, validation)
   - Check memory usage stays within bounds
   - Verify no obvious performance regressions

**Output:** Pass/fail report with timing metrics and basic health indicators

---

## Workflow 6: Validator Extension

**Trigger:** User runs `vibelint add-validator <service-name>` or configures external validator

**Activities:**
1. **External Service Discovery**
   - Read from greater configuration management system
   - Validate service endpoints and credentials
   - Test connectivity to external validator service

2. **Validator Registration**
   - Register new validator in vibelint plugin system
   - Configure input/output format for service
   - Set up caching and rate limiting

3. **Integration Testing**
   - Run new validator on sample files
   - Verify output format matches expectations
   - Test error handling and fallback behavior

4. **Configuration Persistence**
   - Save validator configuration
   - Document new validator capabilities
   - Update workflow to include new validator

**Output:** Successfully integrated external validator ready for use in analysis workflows

---

## Workflow 7: Experimental Branch Management

**Trigger:** Human orchestrator runs `vibelint experiment <experiment-name>` to systematically test different approaches

**Human Decision Points:**
- **Experiment Design**: Human defines what approaches to test and success criteria
- **Approach Selection**: Human chooses which improvement strategies to try
- **Results Evaluation**: Human interprets metrics and decides which approach is best
- **Branch Strategy**: Human decides whether to merge, keep, or discard experimental branches

**Activities:**
1. **Experiment Setup**
   - **HUMAN CHECKPOINT**: Review current state, define experiment goals and success metrics
   - Create new branch from current HEAD: `vibelint-experiment-<name>-<timestamp>`
   - Commit current state as baseline
   - Tag experiment start point for easy reset

2. **Approach Testing Pipeline**
   - **HUMAN CHECKPOINT**: Present list of possible approaches, let human select which to test
   - Apply different improvement approaches on separate commits
   - Run Workflow 2 (multi-representation analysis) after each approach
   - Run Workflow 5 (smoke testing) to verify functionality
   - **HUMAN CHECKPOINT**: Review intermediate results, decide whether to continue or pivot
   - Capture metrics for each approach (performance, violation count, etc.)

3. **Comparative Analysis**
   - Compare metrics across different approaches
   - Use four-model system to evaluate approach quality
   - **HUMAN CHECKPOINT**: Present ranked results with pros/cons, let human interpret
   - Generate ranking of approaches based on multiple criteria
   - Store experiment results in memory system for EBR

4. **Branch Management Decisions**
   - **HUMAN CHECKPOINT**: Review all results, make final decision on next steps:
     * **Best approach found**: Human approves merge back to main branch
     * **Promising approach**: Human decides to keep branch for future exploration
     * **Failed approach**: Human confirms reset to baseline, delete experimental commits
     * **Multiple good approaches**: Human chooses A/B testing strategy with parallel branches

**Output:** Human-validated experimental results with strategic branch management decisions and documented approach comparisons

---

## Core Primitives (Lowest Level)

All workflows build on these fundamental operations:

### File Operations
- `read_file(path)` - Read file with encoding detection
- `write_file(path, content)` - Write file with backup
- `parse_ast(content, language)` - Parse to AST
- `discover_files(root, patterns)` - Find files to analyze

### Analysis Operations
- `extract_metrics(ast)` - Calculate complexity, etc.
- `run_validators(ast, config)` - Apply validation rules
- `detect_patterns(ast, patterns)` - Find code patterns
- `generate_violations(results)` - Create violation reports

### Configuration Operations
- `load_config(path)` - Load vibelint configuration
- `merge_configs(configs)` - Combine configuration sources
- `validate_config(config)` - Check configuration validity
- `save_config(config, path)` - Persist configuration

### Safety Operations (via kaia-guardrails)
- `capture_behavior(function)` - Monitor function execution
- `verify_change(old, new)` - Check change safety
- `rollback_change(change_id)` - Revert unsafe changes
- `update_safety_rules(rules)` - Evolve safety constraints

These workflows define exactly what vibelint should do at the lowest level, making it clear how to implement each feature systematically.