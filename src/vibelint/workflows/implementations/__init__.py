"""
Workflow implementations package.

Contains all concrete workflow implementations organized by functionality.

Responsibility: Workflow implementation organization only.
Individual workflow logic belongs in specific implementation modules.

vibelint/src/vibelint/workflow/implementations/__init__.py
"""

# Import available implementations
from . import (coverage_analysis, justification, justification_analysis,
               redundancy_detection, single_file_validation)

# Re-export specific workflows for convenience
try:
    from .justification import FileJustificationWorkflow
    from .single_file_validation import SingleFileValidationWorkflow
except ImportError:
    # These might have import issues
    pass

__all__ = [
    # Implementation modules
    "justification",
    "coverage_analysis",
    "single_file_validation",
    "redundancy_detection",
    "justification_analysis",
]
