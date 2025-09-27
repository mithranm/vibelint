# Justification Workflow: Core of vibelint

## Overview

The justification workflow is the **foundation of vibelint's code quality analysis**. Every other feature (linting, validation, reporting) builds on top of justification. It ensures every code element justifies its existence through static analysis and targeted LLM usage.

## Static Analysis Foundation

**Primary Tool**: AST parsing and graph analysis
**No LLM Required**: Imports, dependencies, dead code detection

### Import Analysis
```python
# Static analysis can detect:
from typing import Optional  # unused import
import networkx as nx       # circular dependency
```

### Method Call Graph
```python
# Static analysis can build:
def caller() -> None:
    helper_function()  # maps caller -> helper_function

def unused_method():  # unreachable code detected
    pass
```

## Limited LLM Usage

**Fast LLM**: 750 token limit, structured JSON only
**Purpose**: Compare two methods for semantic similarity

### Fast LLM Pattern
```python
# Input: Two methods for comparison
# Output: JSON only, reasoning in thinking tokens
{
  "are_duplicate": "yes",
  "confidence": "high"
}
```

### Orchestrator Usage
- Cross-file semantic analysis
- Larger context understanding
- Architectural pattern detection

## CLI Structure for Development

### Core Commands
```bash
vibelint justify --output justification.md     # Run full workflow
vibelint justify-imports --file src/main.py   # Import analysis only
vibelint justify-methods --compare method1 method2  # Method comparison
vibelint justify-graph --output deps.json     # Dependency graph
```

### Development Commands
```bash
vibelint justify-dev --stage imports          # Test import analysis
vibelint justify-dev --stage methods          # Test method comparison
vibelint justify-dev --stage graph            # Test graph building
vibelint justify-dev --stage llm-compare      # Test fast LLM usage
```

### Integration Commands
```bash
vibelint snapshot --justify                   # Include justifications
vibelint check --with-justification          # Validation + justification
```

## Implementation Strategy

### Phase 1: Static Analysis Engine
```python
# vibelint/src/vibelint/justification/
├── __init__.py
├── import_analyzer.py      # AST-based import analysis
├── method_analyzer.py      # Method extraction and comparison
├── dependency_graph.py     # Project dependency mapping
└── dead_code_detector.py   # Unreachable code detection
```

### Phase 2: LLM Integration
```python
# vibelint/src/vibelint/justification/
├── fast_llm_comparator.py  # Two-method comparison
├── orchestrator_bridge.py  # Cross-file semantic analysis
└── justification_engine.py # Main orchestrator
```

### Phase 3: CLI Integration
```python
# vibelint/src/vibelint/cli/
├── justification.py        # Justification commands
└── analysis.py             # Updated with justify integration
```

This structure supports incremental development of the justification workflow while maintaining vibelint's existing functionality.