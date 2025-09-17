# Vibelint Consolidation Plan

## Current State: 42 files ❌
## Target State: 5 files ✅

### Problem
- 15+ individual validator files for simple pattern matching
- Scattered LLM functionality across multiple files
- Duplicate utility functions
- Over-engineered plugin system for basic linting

### Solution: LLM-Driven Dynamic Analysis

#### Core Architecture (5 files)

```
src/vibelint/
├── core.py          # Main analysis engine + LLM orchestration
├── cli.py           # Command interface (existing)
├── config.py        # Configuration handling (existing)
├── llm.py           # Consolidated LLM management
└── output.py        # Formatting and reporting
```

#### Dynamic Validator Generation

Instead of static validators, use LLMs to:

1. **Generate validators on-demand** based on:
   - Martin Fowler's refactoring catalog
   - User-specific requirements
   - Codebase context

2. **Synthesize rules** from literature:
   ```python
   # Instead of 15 validator files
   def generate_validator(rule_type: str, context: str) -> str:
       prompt = f"""
       Generate a Python validator for {rule_type} based on Martin Fowler's catalog.
       Context: {context}
       Return only the validation logic as executable Python.
       """
       return llm.generate_code(prompt)
   ```

3. **Context-aware analysis**:
   ```python
   def analyze_file(file_path: str, content: str) -> List[Finding]:
       # LLM analyzes the specific file context
       prompt = f"""
       Analyze this code for quality issues:
       File: {file_path}
       {content}

       Focus on: architecture, naming, complexity, maintainability
       Return structured findings with line numbers and suggestions.
       """
       return llm.analyze(prompt)
   ```

#### Benefits

1. **Dramatic file reduction**: 42 → 5 files (88% reduction)
2. **Dynamic capabilities**: Generate new validators without code changes
3. **Context awareness**: LLM sees full file context, not just AST patterns
4. **Literature-based**: Draws from complete refactoring knowledge
5. **Maintainable**: No validator code to maintain

#### Migration Strategy

**Phase 1: Consolidate LLM files**
- Merge `llm/manager.py`, `llm/orchestrator.py`, `llm/trace.py` → `llm.py`

**Phase 2: Create dynamic validator engine**
- Replace all `validators/*.py` with LLM-driven `core.py`

**Phase 3: Consolidate utilities**
- Merge `formatters.py`, `results.py`, `console_utils.py` → `output.py`

**Phase 4: Remove obsolete files**
- Delete 35+ files that become redundant

#### Efficiency Gains

- **Development**: Add new rules via prompts, not code
- **Maintenance**: One analysis engine vs 15+ validators
- **Quality**: LLM sees patterns humans miss
- **Flexibility**: Adapt to any language/framework
- **Intelligence**: Uses full refactoring literature

### Implementation Priority

1. ✅ Consolidate LLM modules
2. 🔄 Create dynamic validator engine
3. 📋 Merge utility functions
4. 📋 Remove obsolete files
5. 📋 Update documentation

This transforms vibelint from a traditional static analyzer to an intelligent, LLM-powered code quality system.