"""
Validator for Python docstrings. Checks for presence and path reference.

vibelint/validators/docstring.py
"""

import logging
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import libcst as cst
from libcst import (
    BaseSmallStatement,
    BaseStatement,
    ClassDef,
    Comment,
    CSTNode,
    EmptyLine,
    Expr,
    FunctionDef,
    IndentedBlock,
    Module,
    Pass,
    SimpleStatementLine,
    SimpleString,
)
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
)

from ..error_codes import VBL101, VBL102

logger = logging.getLogger(__name__)


__all__ = [
    "DocstringValidationResult",
    "get_normalized_filepath",
    "validate_every_docstring",
]

IssueKey = int
BodyItem = Union[BaseStatement, BaseSmallStatement, EmptyLine, Comment]
ValidationIssue = Tuple[str, str]


def _get_docstring_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleStatementLine]:
    """
    Attempts to get the CST node representing the docstring from a sequence of body statements.
    Searches for the first non-comment/empty statement and checks if it's a SimpleString expression.

    vibelint/validators/docstring.py
    """

    first_real_stmt = None
    for stmt in body_stmts:
        if not isinstance(stmt, (EmptyLine, Comment)):
            first_real_stmt = stmt
            break

    if (
        first_real_stmt
        and isinstance(first_real_stmt, SimpleStatementLine)
        and len(first_real_stmt.body) == 1
        and isinstance(first_real_stmt.body[0], Expr)
        and isinstance(first_real_stmt.body[0].value, SimpleString)
    ):
        return first_real_stmt
    return None


def _get_simple_string_node(body_stmts: Sequence[CSTNode]) -> Optional[SimpleString]:
    """
    Gets the SimpleString node if it's the first statement.

    vibelint/validators/docstring.py
    """

    doc_stmt_line = _get_docstring_node(body_stmts)
    if doc_stmt_line:
        try:
            expr_node = doc_stmt_line.body[0]
            if isinstance(expr_node, Expr) and isinstance(expr_node.value, SimpleString):
                return expr_node.value
        except (IndexError, AttributeError):
            pass
    return None


def _extract_docstring_text(node: Optional[SimpleStatementLine]) -> Optional[str]:
    """
    Extracts the interpreted string value from a docstring node.

    vibelint/validators/docstring.py
    """

    if node:
        try:
            expr_node = node.body[0]
            if isinstance(expr_node, Expr):
                str_node = expr_node.value
                if isinstance(str_node, SimpleString):
                    # Use evaluated_value which interprets escapes etc.
                    evaluated = str_node.evaluated_value
                    return evaluated if isinstance(evaluated, str) else None
        except (IndexError, AttributeError, Exception) as e:
            # Catch potential exceptions during evaluation if needed
            logger.debug(f"Failed to extract/evaluate SimpleString: {e}", exc_info=True)
            return None
    return None


def _get_docstring_node_index(body_stmts: Sequence[CSTNode]) -> Optional[int]:
    """
    Gets the index of the docstring node in a body list.

    vibelint/validators/docstring.py
    """

    for i, stmt in enumerate(body_stmts):
        # Skip initial comments and empty lines
        if isinstance(stmt, (EmptyLine, Comment)):
            continue

        # Check if the first non-empty/comment line is a string expression
        if (
            isinstance(stmt, SimpleStatementLine)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], Expr)
            and isinstance(stmt.body[0].value, SimpleString)
        ):
            return i
        else:
            # If the first real statement isn't a docstring, there is no docstring
            return None

    return None


class DocstringValidationResult:
    """
    Stores the result of docstring validation.

    vibelint/validators/docstring.py
    """

    def __init__(self) -> None:
        """
        Initializes DocstringValidationResult.

        vibelint/validators/docstring.py
        """
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    def has_issues(self) -> bool:
        """
        Checks if there are any errors or warnings.

        vibelint/validators/docstring.py
        """
        return bool(self.errors or self.warnings)

    def add_error(self, code: str, message: str):
        """
        Adds an error with its code.

        vibelint/validators/docstring.py
        """
        self.errors.append((code, message))

    def add_warning(self, code: str, message: str):
        """
        Adds a warning with its code.

        vibelint/validators/docstring.py
        """
        self.warnings.append((code, message))


def get_normalized_filepath(relative_path: str) -> str:
    """
    Normalizes a path for docstring references.
    Removes './', converts '' to '/', and removes leading 'src/'.

    vibelint/validators/docstring.py
    """
    # Normalize separators and remove leading './'
    path = relative_path.replace("\\", "/").lstrip("./")
    # Special handling for paths under 'src/'
    if path.startswith("src/"):
        return path[len("src/") :]
    return path


def get_node_start_line(
    node: CSTNode, metadata: Mapping[ProviderT, Mapping[CSTNode, object]]
) -> int:
    """
    Gets the 1-based start line number of a CST node using metadata.
    Returns 0 if position info is unavailable.

    vibelint/validators/docstring.py
    """
    try:
        pos_info = metadata.get(PositionProvider, {}).get(node)
        return pos_info.start.line if isinstance(pos_info, CodeRange) else 0
    except Exception:
        logger.debug(f"Failed to get start line for node {type(node)}", exc_info=True)
        return 0


class DocstringInfoExtractor(cst.CSTVisitor):
    """
    Visits CST nodes to extract docstring info and validate.

    vibelint/validators/docstring.py
    """

    METADATA_DEPENDENCIES = (
        PositionProvider,
        # WhitespaceInclusivePositionProvider, # No longer needed for VBL103
        ParentNodeProvider,
    )

    def __init__(self, relative_path: str):
        """
        Initializes DocstringInfoExtractor.

        vibelint/validators/docstring.py
        """
        super().__init__()
        self.relative_path = relative_path
        self.path_ref = get_normalized_filepath(relative_path)
        self.result = DocstringValidationResult()

        logger.debug(
            f"[Validator:{self.relative_path}] Initialized. Expecting path ref: '{self.path_ref}'"
        )

    def visit_Module(self, node: Module) -> None:
        """
        Visits Module node.

        vibelint/validators/docstring.py
        """
        doc_node = _get_docstring_node(node.body)
        doc_text = _extract_docstring_text(doc_node)
        self._validate_docstring(node, doc_node, doc_text, "module", "module")

    def leave_Module(self, node: Module) -> None:
        """
        Leaves Module node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def visit_ClassDef(self, node: ClassDef) -> bool:
        """
        Visits ClassDef node.

        vibelint/validators/docstring.py
        """
        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(node, doc_node, doc_text, "class", node.name.value)
        else:
            # Handle cases like `class Foo: pass` (no IndentedBlock)
            self._validate_docstring(node, None, None, "class", node.name.value)
        return True  # Continue traversal

    def leave_ClassDef(self, node: ClassDef) -> None:
        """
        Leaves ClassDef node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def visit_FunctionDef(self, node: FunctionDef) -> bool:
        """
        Visits FunctionDef node.

        vibelint/validators/docstring.py
        """
        parent = self.get_metadata(ParentNodeProvider, node)
        is_method = isinstance(parent, IndentedBlock) and isinstance(
            self.get_metadata(ParentNodeProvider, parent), ClassDef
        )
        node_type = "method" if is_method else "function"

        if isinstance(node.body, IndentedBlock):
            doc_node = _get_docstring_node(node.body.body)
            doc_text = _extract_docstring_text(doc_node)
            self._validate_docstring(node, doc_node, doc_text, node_type, node.name.value)
        else:
            # Handle cases like `def foo(): pass` (no IndentedBlock)
            self._validate_docstring(node, None, None, node_type, node.name.value)
        return True  # Continue traversal

    def leave_FunctionDef(self, node: FunctionDef) -> None:
        """
        Leaves FunctionDef node.

        vibelint/validators/docstring.py
        """
        pass  # No action needed on leave

    def _validate_docstring(
        self,
        node: Union[Module, ClassDef, FunctionDef],
        node_doc: Optional[SimpleStatementLine],
        text_doc: Optional[str],
        n_type: str,
        n_name: str,
    ) -> None:
        """
        Performs the validation logic for presence (VBL101) and path reference (VBL102).

        vibelint/validators/docstring.py
        """
        start_line = get_node_start_line(node, self.metadata)
        if start_line == 0:
            logger.warning(
                f"Could not get start line for {n_type} '{n_name}', skipping validation."
            )
            return

        doc_present = node_doc is not None

        # Special case: Ignore missing docstring for simple `__init__` methods
        # like `def __init__(self): pass` or `def __init__(self): super().__init__()`
        # This check remains relevant for VBL101.
        is_simple_init_or_pass = False
        if (
            n_name == "__init__"
            and n_type == "method"
            and isinstance(node, FunctionDef)
            and isinstance(node.body, IndentedBlock)
        ):
            non_empty_stmts = [s for s in node.body.body if not isinstance(s, (EmptyLine, Comment))]
            doc_node_in_body = _get_docstring_node(node.body.body)
            actual_code_stmts = [s for s in non_empty_stmts if s is not doc_node_in_body]
            # Check if the only statement is Pass or a simple super call
            if len(actual_code_stmts) == 1:
                stmt = actual_code_stmts[0]
                if isinstance(stmt, SimpleStatementLine):
                    if len(stmt.body) == 1 and isinstance(stmt.body[0], Pass):
                        is_simple_init_or_pass = True
                    # Add check for simple super().__init__() if desired
                    # elif isinstance(stmt.body[0], Expr) and ... check for super call ...:
                    #    is_simple_init_or_pass = True

        # VBL101: Check for presence
        if not doc_present:
            if not is_simple_init_or_pass:  # Don't warn for simple init/pass methods
                msg = f"Missing docstring for {n_type} '{n_name}'."
                self.result.add_error(VBL101, msg)
                logger.debug(
                    f"[Validator:{self.relative_path}] Added issue {VBL101} for line {start_line}: Missing docstring"
                )
            else:
                logger.debug(
                    f"[Validator:{self.relative_path}] Suppressed VBL101 for simple {n_type} '{n_name}' line {start_line}"
                )
            return  # No further checks if docstring is missing

        # --- VBL103 Check Removed ---

        # VBL102: Check for path reference (only if docstring is present)
        path_issue = False
        if text_doc is not None:
            stripped_text = text_doc.rstrip()  # Remove trailing whitespace before checking end
            if not stripped_text.endswith(self.path_ref):
                path_issue = True
        else:
            # If text couldn't be extracted (e.g., complex f-string), assume path is missing/wrong
            path_issue = True

        if path_issue:
            msg = f"Docstring for {n_type} '{n_name}' missing/incorrect path reference (expected '{self.path_ref}')."
            # Note: This is a WARNING, allowing users to ignore it via config if they disagree.
            self.result.add_warning(VBL102, msg)
            logger.debug(
                f"[Validator:{self.relative_path}] Added issue {VBL102} for line {start_line}: Path reference"
            )

        # Log success if no issues were added for this node
        if not path_issue:  # Only need to check path_issue now
            logger.debug(
                f"[Validator:{self.relative_path}] Validation OK for {n_type} '{n_name}' line {start_line}"
            )


def validate_every_docstring(
    content: str, relative_path: str
) -> Tuple[DocstringValidationResult, Optional[Module]]:
    """
    Parse source code and run the DocstringInfoExtractor visitor to validate all docstrings.

    Args:
    content: The source code as a string.
    relative_path: The relative path of the file (used for path refs).

    Returns:
    A tuple containing:
    - DocstringValidationResult object with found issues.
    - The parsed CST Module node (or None if parsing failed).

    Raises:
    SyntaxError: If LibCST encounters a parsing error, it's converted and re-raised.

    vibelint/validators/docstring.py
    """
    result = DocstringValidationResult()
    module = None
    try:
        module = cst.parse_module(content)
        # Ensure required metadata providers are resolved
        wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
        required_providers = {
            PositionProvider,
            ParentNodeProvider,
        }  # Removed WhitespaceInclusivePositionProvider
        wrapper.resolve_many(required_providers)

        extractor = DocstringInfoExtractor(relative_path)
        wrapper.visit(extractor)
        logger.debug(
            f"[Validator:{relative_path}] Validation complete. Issues found: E={len(extractor.result.errors)}, W={len(extractor.result.warnings)}"
        )
        return extractor.result, module
    except cst.ParserSyntaxError as e:
        # Convert LibCST error to standard SyntaxError for better reporting upstream
        logger.warning(
            f"CST ParserSyntaxError in {relative_path} L{e.raw_line}:{e.raw_column}: {e.message}"
        )
        err = SyntaxError(e.message)
        err.lineno = e.raw_line
        err.offset = e.raw_column + 1 if e.raw_column is not None else None
        err.filename = relative_path
        try:
            # Attempt to get the line content for the error message
            err.text = content.splitlines()[e.raw_line - 1]
        except IndexError:
            err.text = None
        raise err from e
    except Exception as e:
        # Catch other potential errors during validation
        logger.error(f"Unexpected CST validation error {relative_path}: {e}", exc_info=True)
        # Report as an internal error
        result.add_error("VBL903", f"Internal validation error: {e}")
        return result, None
