# Vibelint Development Methodology

*Requirements-driven development for the multi-representation code analysis engine*

## Core Philosophy

Vibelint is a **human-in-loop orchestration system** for code analysis and improvement. Every feature must enable human decision-making while automating the heavy lifting.

**Not**: Full autopilot code generation (thousands have tried and failed)
**Yes**: Intelligent assistance with strategic human checkpoints

## Requirements Framework for Vibelint

### Workflow-Based Requirements

Every vibelint feature maps to one of the 7 core workflows defined in `VIBELINT_WORKFLOWS.md`:

1. **Single File Validation** - Fast feedback on individual Python files
2. **Multi-Representation Analysis** - Deep codebase understanding (the real meat)
3. **Deterministic Fix Application** - Safe, predictable fixes only
4. **Watch Mode** - Real-time feedback during development
5. **Smoke Testing** - Basic functionality verification
6. **Validator Extension** - Integration with external services
7. **Experimental Branch Management** - Systematic approach testing

### Requirements Template for Vibelint Features

```markdown
# [Feature Name] - Vibelint Requirements

## Target Workflow
- Primary: Workflow X (e.g., Multi-Representation Analysis)
- Secondary: Workflow Y (if applicable)

## Human Decision Points
- [ ] HD-001: Where human must choose analysis scope
- [ ] HD-002: Where human must interpret results
- [ ] HD-003: Where human must approve actions

## Four-Model Integration
- [ ] REQ-VM-001: VanguardOne embedding requirements
- [ ] REQ-VM-002: VanguardTwo embedding requirements
- [ ] REQ-VM-003: Chip model processing requirements
- [ ] REQ-VM-004: Claudia model processing requirements

## Multi-Representation Requirements
- [ ] REQ-MR-001: Filesystem representation needs
- [ ] REQ-MR-002: Vector representation (Qdrant) needs
- [ ] REQ-MR-003: Graph representation (NetworkX) needs
- [ ] REQ-MR-004: Runtime representation needs

## Python-Only Constraints
- [ ] REQ-PY-001: AST parsing requirements
- [ ] REQ-PY-002: Import/dependency analysis needs
- [ ] REQ-PY-003: Validator structure compliance

## Performance Requirements
- [ ] REQ-PERF-001: Single file validation <1s
- [ ] REQ-PERF-002: Project analysis scales to 10k+ files
- [ ] REQ-PERF-003: Watch mode <100ms latency

## Safety Integration
- [ ] REQ-SAFE-001: Kaia-guardrails integration points
- [ ] REQ-SAFE-002: No functionality deletion safeguards
- [ ] REQ-SAFE-003: Rollback and recovery capabilities

## Acceptance Criteria
- [ ] AC-001: Human checkpoints tested and working
- [ ] AC-002: Four-model coordination verified
- [ ] AC-003: Multi-representation data generated correctly
- [ ] AC-004: Performance benchmarks met
- [ ] AC-005: Safety integration functional
```

## Development Process for Vibelint

### Phase 1: Workflow Mapping

Before writing any code, answer:
1. **Which workflow** does this feature belong to?
2. **What human decision points** are needed?
3. **Which of the four models** will be involved?
4. **What representations** (filesystem/vector/graph/runtime) are needed?

### Phase 2: Requirements Definition

Use the vibelint requirements template above. Focus on:
- **Human-in-loop design** - Where do humans make decisions?
- **Model orchestration** - How do the four models coordinate?
- **Performance constraints** - Python-only, real-time feedback needs
- **Safety integration** - How does kaia-guardrails protect users?

### Phase 3: Implementation Strategy

#### Core Implementation Priority
1. **Workflow 1 (Single File Validation)** - Foundation for everything else
2. **Workflow 2 (Multi-Representation Analysis)** - The main value proposition
3. **Workflow 3 (Deterministic Fixes)** - Safe, reliable automation
4. **Other workflows** - Supporting capabilities

#### Four-Model Integration Pattern
```python
# Standard pattern for multi-model coordination
async def coordinate_four_models(analysis_context):
    # VanguardOne/Two: Generate embeddings
    embeddings_v1 = await vanguard_one.embed(code_chunks)
    embeddings_v2 = await vanguard_two.embed(code_chunks)

    # Chip/Claudia: Process embeddings + graph
    chip_insights = await chip.analyze(embeddings_v1, dependency_graph)
    claudia_insights = await claudia.analyze(embeddings_v2, execution_graph)

    # Human decision point: Resolve conflicts
    final_insights = await human_resolve_conflicts(chip_insights, claudia_insights)

    return final_insights
```

#### Multi-Representation Building Pattern
```python
# Standard pattern for building representations
class MultiRepresentationBuilder:
    async def build_representations(self, project_path):
        # Filesystem representation
        fs_repr = self.build_filesystem_representation(project_path)

        # Vector representation (with human checkpoint)
        vector_repr = await self.build_vector_representation(fs_repr)
        await self.human_checkpoint_vector_clusters(vector_repr)

        # Graph representation (with human checkpoint)
        graph_repr = self.build_graph_representation(fs_repr)
        await self.human_checkpoint_critical_paths(graph_repr)

        # Runtime representation
        runtime_repr = await self.build_runtime_representation(fs_repr)

        return MultiRepresentation(fs_repr, vector_repr, graph_repr, runtime_repr)
```

### Phase 4: Human Decision Point Testing

Every human decision point must be testable:

```python
# Example: Testing human checkpoint for vector clusters
async def test_vector_cluster_checkpoint():
    # Setup: Create test data with known patterns
    test_embeddings = create_test_embeddings()

    # Action: Run clustering
    clusters = await build_vector_clusters(test_embeddings)

    # Human checkpoint simulation
    human_feedback = simulate_human_cluster_review(clusters)

    # Verify: Human feedback properly integrated
    assert clusters.incorporate_human_feedback(human_feedback)
    assert clusters.human_validated == True
```

### Phase 5: Acceptance Criteria Verification

#### Automated Verification
```bash
# Core functionality tests
pytest tests/workflows/ -v

# Performance benchmarks
pytest tests/performance/ --benchmark-only

# Multi-model coordination tests
pytest tests/integration/four_models/ -v

# Human decision point tests
pytest tests/human_interaction/ -v
```

#### Human Verification Checklist
- [ ] **Human checkpoints work**: Can human actually make decisions at each point?
- [ ] **Information quality**: Is the information presented to humans useful?
- [ ] **Decision impact**: Do human decisions actually affect the outcome?
- [ ] **Override capability**: Can humans override system recommendations?

## Vibelint-Specific Quality Gates

### Gate 1: Workflow Compliance
- [ ] Feature maps to defined workflow in `VIBELINT_WORKFLOWS.md`
- [ ] Human decision points clearly identified
- [ ] Model coordination strategy defined
- [ ] Multi-representation needs specified

### Gate 2: Four-Model Integration
- [ ] VanguardOne/VanguardTwo embedding integration tested
- [ ] Chip/Claudia processing integration tested
- [ ] Model conflict resolution mechanism works
- [ ] Performance within acceptable bounds

### Gate 3: Human-in-Loop Validation
- [ ] Human checkpoints accessible and functional
- [ ] Information presented is actionable
- [ ] Human decisions properly propagated
- [ ] Override mechanisms work correctly

### Gate 4: Python Ecosystem Integration
- [ ] AST parsing works correctly
- [ ] Import/dependency analysis accurate
- [ ] Validator system extensible
- [ ] Standard Python tooling compatibility

## Safety and Guardrails Integration

### SATLUTION-Inspired Safety

Based on the SATLUTION paper insights:
- **Stage 1**: Fast validation (compilation, basic checks)
- **Stage 2**: Behavior verification (no functionality deletion)
- **Stage 3**: Comprehensive testing (performance, integration)

### Kaia-Guardrails Integration Points
```python
# Standard pattern for vibelint safety
from kaia_guardrails.safety_rails import SafetyRails

async def safe_code_modification(changes):
    safety_rails = SafetyRails(project_path)

    # Verify changes won't break functionality
    async with safety_rails.protection_mode():
        verification_result = await safety_rails.verify_change(changes)

        if verification_result.approved:
            await apply_changes(changes)
        else:
            await human_review_violations(verification_result.violations)
```

## Performance Requirements for Vibelint

### Single File Operations
- **Validation**: <1 second for any Python file
- **AST parsing**: <100ms for typical files
- **Validator execution**: <500ms total

### Project-Wide Operations
- **Small projects** (<100 files): <10 seconds
- **Medium projects** (<1000 files): <60 seconds
- **Large projects** (<10k files): <300 seconds

### Watch Mode Operations
- **File change detection**: <100ms latency
- **Incremental validation**: <1 second for changed files
- **Live feedback**: Real-time display updates

### Multi-Model Operations
- **Embedding generation**: Depends on model size, but cached
- **Graph analysis**: <5 seconds for typical projects
- **Context engineering**: <10 seconds for comprehensive analysis

## Testing Strategy for Vibelint

### Unit Testing
- **Individual validators**: Test each validator in isolation
- **AST processing**: Test parsing and analysis logic
- **Model interfaces**: Mock external models for fast testing

### Integration Testing
- **Four-model coordination**: Test with real models when available
- **Multi-representation building**: Test full pipeline
- **Workflow execution**: Test complete workflow paths

### Human Interaction Testing
- **Checkpoint simulation**: Automated testing of human decision points
- **UI/UX testing**: Manual validation of human interfaces
- **Override testing**: Verify human overrides work correctly

### Performance Testing
- **Benchmark suite**: Standard projects of various sizes
- **Memory profiling**: Ensure reasonable memory usage
- **Concurrency testing**: Multi-file processing stress tests

## Continuous Improvement for Vibelint

### Weekly Development Reviews
- **Workflow effectiveness**: Are the 7 workflows the right ones?
- **Human decision quality**: Are checkpoints in the right places?
- **Model coordination**: Is four-model approach working?
- **Performance trends**: Are we maintaining speed requirements?

### Monthly Methodology Updates
- **New workflow patterns**: Based on usage experience
- **Human interaction improvements**: Better UX for decision points
- **Model integration enhancements**: New capabilities from models
- **Safety integration updates**: New guardrails capabilities

## Emergency Procedures for Vibelint

### When Human Checkpoints Fail
1. **Identify failure mode**: UI issue, data issue, or design issue
2. **Implement fallback**: Manual override or safe default
3. **Fix root cause**: Update checkpoint implementation
4. **Test extensively**: Ensure fix doesn't break other checkpoints

### When Model Coordination Fails
1. **Isolate failing model**: Which of the four models has issues
2. **Graceful degradation**: Use remaining models only
3. **User notification**: Clear communication about reduced capability
4. **Recovery plan**: How to restore full four-model operation

### When Performance Degrades
1. **Profile immediately**: Identify performance bottleneck
2. **Temporary mitigation**: Reduce scope or disable expensive features
3. **Root cause fix**: Address underlying performance issue
4. **Benchmark validation**: Ensure fix restores performance

This vibelint-specific methodology ensures that we build a robust, human-in-loop code analysis system that actually works in practice, rather than trying to fully automate everything like so many others have attempted.