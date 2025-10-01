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
