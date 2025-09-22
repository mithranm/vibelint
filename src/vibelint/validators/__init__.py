"""
vibelint validators sub-package.

Modular validator system with centralized registry and discovery.

Responsibility: Validator module organization and re-exports only.
Individual validation logic belongs in specific validator modules.

vibelint/src/vibelint/validators/__init__.py
"""

# Import validator categories for direct access
from . import architecture, project_wide, single_file
# Import registry system
from .registry import (get_all_validators, get_validator, register_validator,
                       validator_registry)
# Re-export specific validators for backward compatibility
from .single_file.docstring import (DocstringPathValidator,
                                    MissingDocstringValidator)
from .single_file.emoji import EmojiUsageValidator
from .single_file.exports import InitAllValidator, MissingAllValidator
from .single_file.line_count import LineCountValidator
from .single_file.logger_names import LoggerNameValidator
from .single_file.print_statements import PrintStatementValidator
from .single_file.self_validation import VibelintSelfValidator

try:
    from .architecture.basic_patterns import ArchitectureValidator
    from .project_wide.dead_code import DeadCodeValidator
except ImportError:
    # These might not exist yet or have import issues
    pass

__all__ = [
    # Registry system
    "validator_registry",
    "register_validator",
    "get_validator",
    "get_all_validators",
    # Category modules
    "single_file",
    "project_wide",
    "architecture",
    # Individual validators (backward compatibility)
    "MissingDocstringValidator",
    "DocstringPathValidator",
    "EmojiUsageValidator",
    "MissingAllValidator",
    "InitAllValidator",
    "LoggerNameValidator",
    "PrintStatementValidator",
    "VibelintSelfValidator",
    "LineCountValidator",
]
