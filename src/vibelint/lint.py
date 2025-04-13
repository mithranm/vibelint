"""
Core linting runner for vibelint.

vibelint/lint.py
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import click
from rich.console import Console
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

console = Console()
logger = logging.getLogger(__name__)

ValidationIssue = Tuple[str, str]


class LintResult:
    """
    Stores the result of a lint operation on a single file.

    vibelint/lint.py
    """

    def __init__(self) -> None:
        """Initializes a LintResult instance."""
        self.file_path: Path = Path()
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    @property
    def has_issues(self) -> bool:
        """Returns True if there are any errors or warnings."""
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner for linting operations. No longer supports fixing.

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

        self.skip_confirmation = skip_confirmation
        self.results: List[LintResult] = []
        self._final_exit_code: int = 0

    def run(self, paths: List[Path]) -> int:
        """
        Runs the linting process, returns the exit code.

        vibelint/lint.py
        """
        logger.debug("LintRunner.run: Starting file discovery...")
        if not self.config.project_root:
            logger.error("Project root not found in config. Cannot run.")
            return 1

        ignore_codes_set = set(self.config.ignore_codes)
        if ignore_codes_set:
            logger.info(f"Ignoring codes: {sorted(list(ignore_codes_set))}")

        all_discovered_files: List[Path] = discover_files(
            paths=paths, config=self.config, explicit_exclude_paths=set()
        )
        python_files: List[Path] = [
            f for f in all_discovered_files if f.is_file() and f.suffix == ".py"
        ]
        logger.debug(f"LintRunner.run: Discovered {len(python_files)} Python files.")

        if not python_files:
            logger.info("No Python files found matching includes/excludes.")
            return 0

        large_dir_threshold = self.config.get("large_dir_threshold", 500)
        if len(python_files) > large_dir_threshold and not self.skip_confirmation:
            logger.debug(
                f"File count {len(python_files)} > threshold {large_dir_threshold}. Requesting confirmation."
            )
            if not self._confirm_large_directory(len(python_files)):
                logger.info("User aborted due to large file count.")
                return 1

        MAX_WORKERS = self.config.get("max_workers")
        logger.debug(f"LintRunner.run: Processing files with max_workers={MAX_WORKERS}")
        progress_console = Console(stderr=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=progress_console,
            transient=True,
        ) as progress:

            task_desc = f"Checking {len(python_files)} Python files..."
            task_id = progress.add_task(task_desc, total=len(python_files))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

                futures = {
                    executor.submit(self._process_file, f, ignore_codes_set): f
                    for f in python_files
                }
                temp_results = []
                for future in futures:
                    file_proc = futures[future]
                    try:
                        res = future.result()
                        temp_results.append(res)
                    except Exception as exc:
                        rel_path_log_err = file_proc.name
                        try:
                            rel_path_log_err = (
                                str(get_relative_path(file_proc, self.config.project_root))
                                if self.config.project_root
                                else file_proc.name
                            )
                        except ValueError:
                            rel_path_log_err = str(file_proc.resolve())
                        logger.error(
                            f"Exception processing {rel_path_log_err}: {exc}",
                            exc_info=True,
                        )
                        lr_err = LintResult()
                        lr_err.file_path = file_proc

                        lr_err.errors.append((VBL904, f"Processing thread error: {exc}"))
                        temp_results.append(lr_err)
                    finally:
                        progress.update(task_id, advance=1)

                self.results = sorted(temp_results, key=lambda r: r.file_path)

        files_with_errors = sum(1 for r in self.results if r.errors)
        self._final_exit_code = 1 if files_with_errors > 0 else 0
        logger.debug(
            f"LintRunner.run: Finished processing. Final exit code: {self._final_exit_code}"
        )
        return self._final_exit_code

    def _process_file(self, file_path: Path, ignore_codes_set: set[str]) -> LintResult:
        """
        Processes a single file for linting issues, filtering by ignored codes.

        vibelint/lint.py
        """
        lr = LintResult()
        lr.file_path = file_path
        relative_path_str = file_path.name
        log_prefix = f"[{file_path.name}]"
        original_content: Optional[str] = None
        collected_errors: List[ValidationIssue] = []
        collected_warnings: List[ValidationIssue] = []

        try:

            if self.config.project_root:
                try:
                    relative_path = get_relative_path(file_path, self.config.project_root)
                    relative_path_str = str(relative_path).replace("\\", "/")
                    log_prefix = f"[{relative_path_str}]"
                except ValueError:
                    relative_path_str = str(file_path.resolve())
            else:
                relative_path_str = str(file_path.resolve())

            logger.debug(f"{log_prefix} --- Starting validation ---")

            try:
                original_content = file_path.read_text(encoding="utf-8")
                logger.debug(f"{log_prefix} Read {len(original_content)} bytes.")
            except Exception as read_e:
                logger.error(f"{log_prefix} Error reading file: {read_e}", exc_info=True)

                lr.errors.append((VBL901, f"Error reading file: {read_e}"))
                return lr

            try:

                doc_res, _ = validate_every_docstring(original_content, relative_path_str)
                if doc_res:
                    collected_errors.extend(doc_res.errors)
                    collected_warnings.extend(doc_res.warnings)

                allowed_sb: List[str] = self.config.get(
                    "allowed_shebangs", ["#!/usr/bin/env python3"]
                )
                is_script = file_contains_top_level_main_block(file_path, original_content)
                sb_res = validate_shebang(original_content, is_script, allowed_sb)
                collected_errors.extend(sb_res.errors)
                collected_warnings.extend(sb_res.warnings)

                enc_res = validate_encoding_cookie(original_content)
                collected_errors.extend(enc_res.errors)
                collected_warnings.extend(enc_res.warnings)

                export_res = validate_exports(original_content, relative_path_str, self.config)
                collected_errors.extend(export_res.errors)
                collected_warnings.extend(export_res.warnings)

                logger.debug(
                    f"{log_prefix} Validation Complete. Found E={len(collected_errors)}, W={len(collected_warnings)} (before filtering)"
                )

            except SyntaxError as se:
                line = f"line {se.lineno}" if se.lineno else "unk line"
                col = f", col {se.offset}" if se.offset is not None else ""
                err_msg = f"SyntaxError parsing file: {se.msg} ({relative_path_str}, {line}{col})"
                logger.error(f"{log_prefix} {err_msg}")

                collected_errors.append((VBL902, err_msg))

            except Exception as val_e:
                logger.error(
                    f"{log_prefix} Error during validation phase: {val_e}",
                    exc_info=True,
                )

                collected_errors.append((VBL903, f"Internal validation error: {val_e}"))

            lr.errors = [
                (code, msg) for code, msg in collected_errors if code not in ignore_codes_set
            ]
            lr.warnings = [
                (code, msg) for code, msg in collected_warnings if code not in ignore_codes_set
            ]

            if len(collected_errors) != len(lr.errors) or len(collected_warnings) != len(
                lr.warnings
            ):
                logger.debug(
                    f"{log_prefix} Filtered issues based on ignore config. Final E={len(lr.errors)}, W={len(lr.warnings)}"
                )

        except Exception as e:
            logger.error(
                f"{log_prefix} Critical unhandled error in _process_file: {e}\n{traceback.format_exc()}"
            )

            lr.errors.append((VBL905, f"Critical processing error: {e}"))

        logger.debug(f"{log_prefix} --- Finished validation ---")
        return lr

    def _confirm_large_directory(self, file_count: int) -> bool:
        """
        Asks user for confirmation if many files are found.

        vibelint/lint.py
        """

        prompt_console = Console(stderr=True)
        prompt_console.print(
            f"[yellow]WARNING:[/yellow] Found {file_count} Python files. This might take time."
        )
        try:
            return click.confirm("Proceed?", default=False, err=True)
        except click.Abort:
            prompt_console.print("[yellow]Aborted.[/yellow]")
            return False
        except RuntimeError as e:
            if "Cannot prompt" in str(e) or "tty" in str(e).lower():
                prompt_console.print(
                    "[yellow]Non-interactive session detected. Use --yes to bypass confirmation. Aborting.[/yellow]"
                )
            else:
                logger.error(f"RuntimeError during confirm: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error during confirm: {e}", exc_info=True)
            return False

    def _print_summary(self) -> None:
        """
        Prints a summary table of the linting results.

        vibelint/lint.py
        """
        summary_console = Console()
        table = Table(title="vibelint Results Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")
        total = len(self.results)

        errors = sum(1 for r in self.results if r.errors)
        warns = sum(1 for r in self.results if r.warnings and not r.errors)
        ok = total - errors - warns

        table.add_row("Files Scanned", str(total))
        table.add_row("Files OK", str(ok), style="green" if ok == total and total > 0 else "")
        table.add_row("Files with Errors", str(errors), style="red" if errors else "")
        table.add_row(
            "Files with Warnings only",
            str(warns),
            style="yellow" if warns else "",
        )

        summary_console.print(table)
