# Workflow 1: Single File Validation - Requirements

## Target Workflow
- Primary: Workflow 1 (Single File Validation)
- Secondary: Foundation for all other workflows

## Problem Statement
Implement fast, reliable validation of individual Python files using the existing validator structure. This is the foundation that all other vibelint workflows depend on.

## Human Decision Points
- [ ] HD-001: Human chooses which validators to run on specific files
- [ ] HD-002: Human interprets violation severity and decides on fixes
- [ ] HD-003: Human reviews and approves validator configuration changes

## Four-Model Integration
- [ ] REQ-VM-001: VanguardOne embedding - NOT REQUIRED for Workflow 1 (single file only)
- [ ] REQ-VM-002: VanguardTwo embedding - NOT REQUIRED for Workflow 1 (single file only)
- [ ] REQ-VM-003: Chip model processing - NOT REQUIRED for Workflow 1 (deterministic only)
- [ ] REQ-VM-004: Claudia model processing - NOT REQUIRED for Workflow 1 (deterministic only)

## Multi-Representation Requirements
- [ ] REQ-MR-001: Filesystem representation - Parse single Python file AST
- [ ] REQ-MR-002: Vector representation - NOT REQUIRED for Workflow 1
- [ ] REQ-MR-003: Graph representation - NOT REQUIRED for Workflow 1
- [ ] REQ-MR-004: Runtime representation - NOT REQUIRED for Workflow 1

## Python-Only Constraints
- [ ] REQ-PY-001: AST parsing must handle all Python 3.9+ syntax correctly
- [ ] REQ-PY-002: Import analysis limited to single file scope (no cross-file dependencies)
- [ ] REQ-PY-003: Validator structure must follow existing pattern in `src/vibelint/validators/single_file/`

## Functional Requirements
- [ ] REQ-F-001: Load and execute all validators from `src/vibelint/validators/single_file/`
- [ ] REQ-F-002: Parse Python file to AST without executing code
- [ ] REQ-F-003: Run each validator against file AST and return structured violations
- [ ] REQ-F-004: Aggregate violations with line numbers, severity, and fix suggestions
- [ ] REQ-F-005: Support ignore rules and exceptions per file
- [ ] REQ-F-006: Generate file-level health score based on violations

## Performance Requirements
- [ ] REQ-PERF-001: Single file validation completes in <1 second for files up to 1000 lines
- [ ] REQ-PERF-002: AST parsing completes in <100ms for typical Python files
- [ ] REQ-PERF-003: All validators execute in <500ms total per file
- [ ] REQ-PERF-004: Memory usage stays under 50MB per file validation

## Safety Integration
- [ ] REQ-SAFE-001: Integration point for kaia-guardrails when file modifications suggested
- [ ] REQ-SAFE-002: No file system modifications in validation phase (read-only)
- [ ] REQ-SAFE-003: Graceful error handling for malformed Python files

## Existing Validator Integration
- [ ] REQ-VAL-001: Use existing emoji validator at `src/vibelint/validators/single_file/emoji.py`
- [ ] REQ-VAL-002: Support validator discovery and loading from validators directory
- [ ] REQ-VAL-003: Each validator follows standard interface (takes AST, returns violations)
- [ ] REQ-VAL-004: Violation format includes: line, column, rule, message, severity, fix_suggestion

## Configuration Requirements
- [ ] REQ-CFG-001: Support per-file ignore patterns (.vibelint-ignore comments)
- [ ] REQ-CFG-002: Support project-level validator configuration
- [ ] REQ-CFG-003: Allow severity level customization per validator
- [ ] REQ-CFG-004: Human-readable configuration format (TOML preferred)

## Output Requirements
- [ ] REQ-OUT-001: JSON format for machine consumption
- [ ] REQ-OUT-002: Human-readable terminal format with colors
- [ ] REQ-OUT-003: Structured violation objects with all metadata
- [ ] REQ-OUT-004: File health score (0-100) based on violation count and severity

## Error Handling Requirements
- [ ] REQ-ERR-001: Handle syntax errors in Python files gracefully
- [ ] REQ-ERR-002: Handle missing file errors with clear messages
- [ ] REQ-ERR-003: Handle validator loading errors without crashing
- [ ] REQ-ERR-004: Log errors to debug file while continuing validation

## CLI Interface Requirements
- [ ] REQ-CLI-001: `vibelint validate file.py` command
- [ ] REQ-CLI-002: `--format json|human` output format option
- [ ] REQ-CLI-003: `--validators emoji,docstring` validator selection option
- [ ] REQ-CLI-004: `--severity error|warning|info` filter option
- [ ] REQ-CLI-005: Exit code 0 for no violations, 1 for violations found

## Acceptance Criteria

### AC-001: Core Validation Pipeline Works
- [ ] Can parse any valid Python file to AST
- [ ] Can load and execute emoji validator successfully
- [ ] Returns structured violation data with line numbers
- [ ] Handles syntax errors without crashing

### AC-002: Performance Targets Met
- [ ] Validates 100-line Python file in <200ms
- [ ] Validates 1000-line Python file in <1s
- [ ] Memory usage stays reasonable for large files
- [ ] No memory leaks during repeated validations

### AC-003: Output Formats Correct
- [ ] JSON output is valid and parseable
- [ ] Human output is readable and colored appropriately
- [ ] All required fields present in violation objects
- [ ] Health score calculation is reasonable

### AC-004: Error Handling Robust
- [ ] Gracefully handles files with syntax errors
- [ ] Clear error messages for missing files
- [ ] Continues processing when one validator fails
- [ ] Debug logs help troubleshoot issues

### AC-005: CLI Interface Usable
- [ ] Basic `vibelint validate file.py` command works
- [ ] Format selection works correctly
- [ ] Validator selection filters correctly
- [ ] Exit codes match expectations

### AC-006: Foundation for Other Workflows
- [ ] Validation logic can be reused by multi-file workflows
- [ ] Validator interface supports future extensions
- [ ] Output format supports aggregation across files
- [ ] Performance scales to batch processing

## Implementation Priority
1. **Phase 1**: Core AST parsing and validator loading
2. **Phase 2**: Emoji validator integration and violation formatting
3. **Phase 3**: CLI interface and output formatting
4. **Phase 4**: Error handling and edge cases
5. **Phase 5**: Performance optimization and testing

## Dependencies
- Existing emoji validator implementation
- Python AST module
- Project configuration system
- CLI framework (Click or similar)

## Risks and Mitigations
- **Risk**: AST parsing fails on edge case Python syntax
  - **Mitigation**: Comprehensive test suite with diverse Python files
- **Risk**: Validator interface changes break existing validators
  - **Mitigation**: Careful interface design with backward compatibility
- **Risk**: Performance degrades with large files
  - **Mitigation**: Early performance testing and optimization

## Success Metrics
- All acceptance criteria pass
- Performance requirements met
- Can serve as foundation for Workflow 2 (Multi-Representation Analysis)
- Developer experience is smooth and fast
- Zero regressions in existing emoji validator behavior