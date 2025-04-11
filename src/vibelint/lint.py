"""
Core linting functionality for vibelint.

vibelint/lint.py
"""

import re
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import fnmatch

import click
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .validators.shebang import validate_shebang, fix_shebang
from .validators.encoding import validate_encoding_cookie, fix_encoding_cookie
from .validators.docstring import validate_module_docstring, fix_module_docstring


console = Console()


class LintResult:
    """
    Class to store the result of a linting operation.

    vibelint/lint.py
    """

    def __init__(self):
        self.file_path: Path = Path()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixed: bool = False

    @property
    def has_issues(self) -> bool:
        """Check if the result has any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0


class LintRunner:
    """
    Runner class for linting operations.

    vibelint/lint.py
    """

    def __init__(
        self,
        config: Dict[str, Any],
        check_only: bool = False,
        skip_confirmation: bool = False,
        include_vcs_hooks: bool = False,
    ):
        self.config = config
        self.check_only = check_only
        self.skip_confirmation = skip_confirmation
        self.include_vcs_hooks = include_vcs_hooks
        self.results: List[LintResult] = []
        self.files_fixed: int = 0
        self.files_with_errors: int = 0
        self.files_with_warnings: int = 0

    def run(self, paths: List[Path]) -> int:
        """Run the linting process on the specified paths."""
        # Get all Python files to check
        python_files = self._collect_python_files(paths)

        if not python_files:
            console.print("[yellow]No Python files found to lint.[/yellow]")
            return 0

        # Check if directory is large and requires confirmation
        if (
            not self.skip_confirmation and len(python_files) > self.config["large_dir_threshold"]
        ):
            if not self._confirm_large_directory(len(python_files)):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return 0

        # Process files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Linting {len(python_files)} Python files...", total=len(python_files)
            )

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                for result in executor.map(self._process_file, python_files):
                    self.results.append(result)
                    progress.advance(task)

        # Update statistics
        for result in self.results:
            if result.fixed:
                self.files_fixed += 1
            if result.errors:
                self.files_with_errors += 1
            elif result.warnings:
                self.files_with_warnings += 1

        # Print summary
        self._print_summary()

        # Determine exit code
        if self.files_with_errors > 0 or (self.check_only and self.files_fixed > 0):
            return 1
        return 0

    def _collect_python_files(self, paths: List[Path]) -> List[Path]:
        """Collect all Python files to lint."""
        python_files: List[Path] = []

        for path in paths:
            if path.is_file() and path.suffix == ".py":
                python_files.append(path)
            elif path.is_dir():
                for include_glob in self.config["include_globs"]:
                    # Generate pattern-matched paths
                    matched_files = path.glob(include_glob)
                    for file_path in matched_files:
                        # Skip if it's not a file or not a Python file
                        if not file_path.is_file() or file_path.suffix != ".py":
                            continue

                        # Skip VCS directories unless explicitly included
                        if not self.include_vcs_hooks and any(
                            part.startswith(".") and part in {".git", ".hg", ".svn"}
                            for part in file_path.parts
                        ):
                            continue

                        # Check exclude patterns
                        if any(
                            fnmatch.fnmatch(str(file_path), str(path / exclude_glob))
                            for exclude_glob in self.config["exclude_globs"]
                        ):
                            continue

                        python_files.append(file_path)

        return python_files

    def _confirm_large_directory(self, file_count: int) -> bool:
        """Ask for confirmation when processing a large directory."""
        console.print(
            f"[yellow]Warning:[/yellow] Found {file_count} Python files to lint, which exceeds the "
            f"large_dir_threshold of {self.config['large_dir_threshold']}."
        )
        return click.confirm("Do you want to continue?", default=True)

    def _process_file(self, file_path: Path) -> LintResult:
        """Process a single Python file."""
        result = LintResult()
        result.file_path = file_path

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create a copy of the content that we'll modify if fixes are needed
            new_content = content

            # Apply validations
            is_script = (
                re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", content)
                is not None
            )
            package_root = self.config["package_root"]
            relative_path = str(file_path)

            # If package_root is specified, compute the relative path
            if package_root:
                try:
                    relative_path = str(file_path.relative_to(package_root))
                except ValueError:
                    # File is not within package_root
                    pass

            # Validate shebang
            shebang_result = validate_shebang(
                content, is_script, self.config["allowed_shebangs"]
            )
            if shebang_result.has_issues():
                result.errors.extend(shebang_result.errors)
                result.warnings.extend(shebang_result.warnings)
                if not self.check_only:
                    new_content = fix_shebang(
                        new_content,
                        shebang_result,
                        is_script,
                        self.config["allowed_shebangs"][0],
                    )

            # Validate encoding cookie
            encoding_result = validate_encoding_cookie(content)
            if encoding_result.has_issues():
                result.errors.extend(encoding_result.errors)
                result.warnings.extend(encoding_result.warnings)
                if not self.check_only:
                    new_content = fix_encoding_cookie(new_content, encoding_result)

            # Validate module docstring
            docstring_result = validate_module_docstring(
                content, relative_path, self.config["docstring_regex"]
            )
            if docstring_result.has_issues():
                result.errors.extend(docstring_result.errors)
                result.warnings.extend(docstring_result.warnings)
                if not self.check_only:
                    new_content = fix_module_docstring(
                        new_content, docstring_result, relative_path
                    )

            # Apply fixes if necessary
            if new_content != content and not self.check_only:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                result.fixed = True

        except Exception as e:
            result.errors.append(f"Error processing file: {str(e)}")

        return result

    def _print_summary(self):
        """Print a summary of the linting operation."""
        table = Table(title="vibelint Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Files processed", str(len(self.results)))
        table.add_row("Files fixed", str(self.files_fixed))
        table.add_row(
            "Files with errors",
            str(self.files_with_errors),
            style="red" if self.files_with_errors else "green",
        )
        table.add_row(
            "Files with warnings",
            str(self.files_with_warnings),
            style="yellow" if self.files_with_warnings else "green",
        )

        console.print(table)

        # Print files with issues
        if self.files_with_errors > 0 or self.files_with_warnings > 0:
            console.print("\n[bold]Files with issues:[/bold]")

            for result in self.results:
                if result.has_issues:
                    status = (
                        "[red]ERROR[/red]"
                        if result.errors
                        else "[yellow]WARNING[/yellow]"
                    )
                    console.print(f"{status} {result.file_path}")

                    for error in result.errors:
                        console.print(f"  - [red]{error}[/red]")
                    for warning in result.warnings:
                        console.print(f"  - [yellow]{warning}[/yellow]")


def lint_files(
    paths: List[Path],
    config: Dict[str, Any],
    check_only: bool = False,
    skip_confirmation: bool = False,
    include_vcs_hooks: bool = False,
) -> int:
    """
    Lint Python files.

    vibelint/lint.py
    """
    runner = LintRunner(
        config=config,
        check_only=check_only,
        skip_confirmation=skip_confirmation,
        include_vcs_hooks=include_vcs_hooks,
    )
    return runner.run(paths)
