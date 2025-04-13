"""
Core linting runner for vibelint.

vibelint/lint.py
"""

import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set, Tuple

import click
from rich.console import Console
from rich.progress import TimeElapsedColumn  # Add for better progress visibility
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from .config import Config
from .discovery import discover_files
from .error_codes import VBL901, VBL902, VBL903, VBL904, VBL905
from .utils import get_relative_path
from .validators.docstring import validate_every_docstring
from .validators.encoding import validate_encoding_cookie
from .validators.exports import validate_exports
from .validators.shebang import file_contains_top_level_main_block, validate_shebang

__all__ = ["LintResult", "LintRunner"]

console = Console()  # Keep console instance if used elsewhere, otherwise remove
logger = logging.getLogger(__name__)

ValidationIssue = Tuple[str, str]


class LintResult:
    """
    Stores the result of a lint operation (Vibe Check) on a single file.

    vibelint/lint.py
    """

    def __init__(self) -> None:
        """
        Initializes a LintResult instance.

        vibelint/lint.py
        """
        self.file_path: Path = Path()
        self.errors: List[ValidationIssue] = []  # Major vibe failures
        self.warnings: List[ValidationIssue] = []  # Minor vibe issues

    @property
    def has_issues(self) -> bool:
        """
        Returns True if there are any errors or warnings.

        vibelint/lint.py
        """
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner for linting operations (Vibe Checks).

    vibelint/lint.py
    """

    def __init__(self, config: Config, skip_confirmation: bool = False) -> None:
        """
        Initializes the LintRunner.

        Args:
            config: The vibelint configuration object.
            skip_confirmation: If True, bypass the confirmation prompt for large directories.

        vibelint/lint.py
        """
        self.config = config
        if not self.config.project_root:
            # Add a check here as project_root is crucial
            raise ValueError("LintRunner requires a config object with project_root set.")

        self.skip_confirmation = skip_confirmation
        self.results: List[LintResult] = []
        self._final_exit_code: int = 0  # Determined by errors

    def run(self, paths: List[Path]) -> int:
        """
        Runs the linting process (Vibe Check) and returns the exit code (0 for pass, 1 for fail).

        Failure is determined by the presence of *errors* (not warnings).

        vibelint/lint.py
        """
        logger.debug("LintRunner.run: Starting file discovery for Vibe Check...")
        if not self.config.project_root:  # Should be caught by __init__, but double-check
            logger.error("Project root not found in config. Cannot run Vibe Check.")
            return 1

        ignore_codes_set = set(self.config.ignore_codes)
        if ignore_codes_set:
            logger.info(f"Ignoring codes: {sorted(list(ignore_codes_set))}")

        discovery_start = time.time()
        all_discovered_files: List[Path] = discover_files(
            paths=paths, config=self.config, explicit_exclude_paths=set()
        )
        discovery_time = time.time() - discovery_start
        logger.debug(f"File discovery took {discovery_time:.4f} seconds.")

        python_files: List[Path] = [
            f for f in all_discovered_files if f.is_file() and f.suffix == ".py"
        ]
        logger.debug(f"LintRunner.run: Discovered {len(python_files)} Python files for Vibe Check.")

        if not python_files:
            logger.info("No Python files found matching includes/excludes. No vibes to check.")
            self._final_exit_code = 0  # No files means no errors
            return self._final_exit_code

        large_dir_threshold = self.config.get("large_dir_threshold", 500)
        if len(python_files) > large_dir_threshold and not self.skip_confirmation:
            logger.debug(
                f"File count {len(python_files)} > threshold {large_dir_threshold}. Requesting confirmation."
            )
            if not self._confirm_large_directory(len(python_files)):
                logger.info("User aborted Vibe Check due to large file count.")
                self._final_exit_code = 1  # User aborted, treat as failure? Or 0? Let's go with 1.
                return self._final_exit_code

        # Consider making MAX_WORKERS configurable? Default to None (cpu_count * 5)
        MAX_WORKERS = self.config.get("max_workers")
        logger.debug(f"LintRunner.run: Processing files with max_workers={MAX_WORKERS}")
        progress_console = Console(
            stderr=True, force_terminal=True if sys.stderr.isatty() else False
        )  # Force terminal for CI if needed
        processing_start = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),  # Show time elapsed
            console=progress_console,
            transient=True,  # Clears progress bar on completion
        ) as progress:

            task_desc = f"Conducting Vibe Check on {len(python_files)} Python files..."
            task_id = progress.add_task(task_desc, total=len(python_files))
            # Use a context manager for the executor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Map filenames to futures for better error reporting
                futures_map = {
                    executor.submit(self._process_file, f, ignore_codes_set): f
                    for f in python_files
                }
                temp_results = []
                # Use as_completed for potentially better responsiveness if needed
                # from concurrent.futures import as_completed
                # for future in as_completed(futures_map):
                #    file_proc = futures_map[future]
                for (
                    future,
                    file_proc,
                ) in (
                    futures_map.items()
                ):  # Iterate directly if order doesn't matter strictly before sorting
                    try:
                        res = future.result()  # Get result or raise exception
                        temp_results.append(res)
                    except Exception as exc:
                        # Log exceptions clearly tied to the file
                        rel_path_log_err = file_proc.name  # Fallback name
                        try:
                            if self.config.project_root:
                                rel_path_log_err = str(
                                    get_relative_path(file_proc, self.config.project_root)
                                )
                        except ValueError:
                            rel_path_log_err = str(
                                file_proc.resolve()
                            )  # Absolute path if outside root

                        logger.error(
                            f"Exception processing {rel_path_log_err}: {exc}",
                            exc_info=True,  # Include traceback in debug log
                        )
                        # Create a result indicating the processing error
                        lr_err = LintResult()
                        lr_err.file_path = file_proc
                        # Use a specific error code for processing errors
                        lr_err.errors.append((VBL904, f"Processing thread error: {exc}"))
                        temp_results.append(lr_err)
                    finally:
                        progress.update(
                            task_id, advance=1
                        )  # Advance progress regardless of success/failure

                # Sort results only once after all futures are complete
                self.results = sorted(temp_results, key=lambda r: r.file_path)

        processing_time = time.time() - processing_start
        logger.debug(f"File processing took {processing_time:.4f} seconds.")

        # Determine exit code based *only* on the presence of errors in the results
        files_with_errors = sum(1 for r in self.results if r.errors)
        self._final_exit_code = 1 if files_with_errors > 0 else 0
        logger.debug(
            f"LintRunner.run: Vibe Check finished. Found {files_with_errors} files with major failures. Final exit code: {self._final_exit_code}"
        )
        return self._final_exit_code

    def _process_file(self, file_path: Path, ignore_codes_set: Set[str]) -> LintResult:
        """
        Processes a single file for linting issues (Vibe Check), filtering by ignored codes.

        Args:
            file_path: The absolute path to the Python file.
            ignore_codes_set: A set of VBL codes to ignore.

        Returns:
            A LintResult object for the file.

        vibelint/lint.py
        """
        lr = LintResult()
        lr.file_path = file_path
        relative_path_str = file_path.name  # Default
        log_prefix = f"[{file_path.name}]"  # Use simple name for shorter logs initially
        original_content: Optional[str] = None
        collected_errors: List[ValidationIssue] = []
        collected_warnings: List[ValidationIssue] = []

        try:
            # Determine relative path for logging and validation use
            if self.config.project_root:
                try:
                    relative_path = get_relative_path(file_path, self.config.project_root)
                    relative_path_str = str(relative_path).replace("\\", "/")  # Use normalized path
                    log_prefix = f"[{relative_path_str}]"  # Update log prefix
                except ValueError:
                    # File might be outside project root (e.g., explicitly passed)
                    relative_path_str = str(file_path.resolve())
                    log_prefix = f"[{relative_path_str}]"
                    logger.warning(f"{log_prefix} File seems outside project root.")
            else:
                # Should not happen if LintRunner checks config on init, but handle defensively
                relative_path_str = str(file_path.resolve())
                log_prefix = f"[{relative_path_str}]"
                logger.error("Project root missing in config during file processing!")

            logger.debug(f"{log_prefix} --- Starting Vibe Validation ---")

            # 1. Read File Content
            try:
                # Read as bytes first to detect encoding issues if not UTF-8? Maybe too complex. Stick to UTF-8.
                original_content = file_path.read_text(encoding="utf-8")
                logger.debug(f"{log_prefix} Read {len(original_content)} bytes.")
            except Exception as read_e:
                logger.error(f"{log_prefix} Error reading file: {read_e}", exc_info=True)
                # Add a specific error code for file reading issues
                lr.errors.append((VBL901, f"Error reading file: {read_e}"))
                # Can't proceed without content
                return lr

            # 2. Run Validators (Order might matter if one depends on syntax)
            try:
                # Docstring Validation (VBL1xx)
                # Pass the already read content and calculated relative path
                doc_res, _ = validate_every_docstring(original_content, relative_path_str)
                if doc_res:
                    collected_errors.extend(doc_res.errors)
                    collected_warnings.extend(doc_res.warnings)

                # Shebang Validation (VBL4xx)
                allowed_sb: List[str] = self.config.get(
                    "allowed_shebangs", ["#!/usr/bin/env python3"]  # Provide a default
                )
                # Pass content to avoid re-reading; pass path for context/AST parsing if needed
                is_script = file_contains_top_level_main_block(file_path, original_content)
                sb_res = validate_shebang(original_content, is_script, allowed_sb)
                collected_errors.extend(sb_res.errors)
                collected_warnings.extend(sb_res.warnings)

                # Encoding Cookie Validation (VBL2xx)
                enc_res = validate_encoding_cookie(original_content)
                collected_errors.extend(enc_res.errors)
                collected_warnings.extend(enc_res.warnings)

                # Export (__all__) Validation (VBL3xx)
                export_res = validate_exports(original_content, relative_path_str, self.config)
                collected_errors.extend(export_res.errors)
                collected_warnings.extend(export_res.warnings)

                logger.debug(
                    f"{log_prefix} Validation Complete. Found E={len(collected_errors)}, W={len(collected_warnings)} (before filtering)"
                )

            # Handle specific parsing errors
            except SyntaxError as se:
                line = f"line {se.lineno}" if se.lineno is not None else "unk line"
                col = f", col {se.offset}" if se.offset is not None else ""
                err_msg = f"SyntaxError parsing file: {se.msg} ({relative_path_str}, {line}{col})"
                logger.error(f"{log_prefix} {err_msg}")
                # Add a specific error code for syntax errors
                collected_errors.append((VBL902, err_msg))

            # Catch internal errors during the validation phase for this file
            except Exception as val_e:
                logger.error(
                    f"{log_prefix} Error during validation phase: {val_e}",
                    exc_info=True,  # Log traceback for internal errors
                )
                # Add a specific error code for internal validation errors
                collected_errors.append((VBL903, f"Internal validation error: {val_e}"))

            # 3. Filter Results based on `ignore_codes_set`
            lr.errors = [
                (code, msg) for code, msg in collected_errors if code not in ignore_codes_set
            ]
            lr.warnings = [
                (code, msg) for code, msg in collected_warnings if code not in ignore_codes_set
            ]

            if len(collected_errors) != len(lr.errors) or len(collected_warnings) != len(
                lr.warnings
            ):
                ignored_count = (len(collected_errors) - len(lr.errors)) + (
                    len(collected_warnings) - len(lr.warnings)
                )
                logger.debug(
                    f"{log_prefix} Filtered {ignored_count} issue(s) based on ignore config. Final E={len(lr.errors)}, W={len(lr.warnings)}"
                )

        except Exception as e:
            # Catch any other unexpected errors during processing for this file
            logger.error(
                f"{log_prefix} Critical unhandled error in _process_file: {e}\n{traceback.format_exc()}"
            )
            # Use a specific error code for critical processing errors
            # Ensure previous errors aren't lost if this happens late
            lr.errors.append((VBL905, f"Critical processing error: {e}"))

        logger.debug(f"{log_prefix} --- Finished Vibe Validation ---")
        return lr

    def _confirm_large_directory(self, file_count: int) -> bool:
        """
        Asks user for confirmation if many files are found.

        Args:
            file_count: The number of Python files found.

        Returns:
            True if the user confirms, False otherwise.

        vibelint/lint.py
        """
        prompt_console = Console(stderr=True)  # Ensure prompts go to stderr
        prompt_console.print(
            f"[yellow]WARNING:[/yellow] Found {file_count} Python files to check. This might take a moment."
        )
        try:
            # Use Click's confirmation prompt
            return click.confirm("Proceed with Vibe Check?", default=False, err=True)
        except click.Abort:
            # User pressed Ctrl+C or similar during prompt
            prompt_console.print("[yellow]Vibe Check aborted by user.[/yellow]", style="yellow")
            return False
        except RuntimeError as e:
            # Handle cases where prompt cannot be shown (e.g., non-interactive session)
            if "Cannot prompt" in str(e) or "tty" in str(e).lower():
                prompt_console.print(
                    "[yellow]Non-interactive session detected. Use --yes to bypass confirmation. Aborting Vibe Check.[/yellow]",
                    style="yellow",
                )
            else:
                # Log unexpected runtime errors during prompt
                logger.error(f"RuntimeError during confirmation prompt: {e}", exc_info=True)
            return False
        except Exception as e:
            # Catch any other unexpected errors during the prompt
            logger.error(f"Error during confirmation prompt: {e}", exc_info=True)
            return False

    def _print_summary(self) -> None:
        """
        Prints a summary table of the linting results using "Vibe Check" terminology.

        vibelint/lint.py
        """
        summary_console = (
            Console()
        )  # Use a separate console instance if needed, or reuse the global one
        # --- Use Vibe Check Terminology ---
        table = Table(title="vibelint Vibe Check Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")

        total = len(self.results)
        # Count files with actual VBL errors (Major Failures)
        files_with_errors = sum(1 for r in self.results if r.errors)
        # Count files with only warnings (Minor Issues)
        files_with_warnings_only = sum(1 for r in self.results if r.warnings and not r.errors)
        files_ok = total - files_with_errors - files_with_warnings_only

        table.add_row("Files Scanned", str(total))
        table.add_row(
            "Files Vibing", str(files_ok), style="green" if files_ok == total and total > 0 else ""
        )
        table.add_row(
            "Files Not Vibing (Errors)",
            str(files_with_errors),
            style="red" if files_with_errors else "",
        )
        table.add_row(
            "Files In Between (Warnings)",
            str(files_with_warnings_only),
            style="yellow" if files_with_warnings_only else "",
        )

        summary_console.print(table)
        # --- End Vibe Check Terminology ---
