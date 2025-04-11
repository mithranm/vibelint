"""
Core linting functionality for vibelint. Uses docstring, shebang, encoding checks.

src/vibelint/lint.py
"""

import re
import fnmatch
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

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
from .validators.docstring import (
    validate_every_docstring,
    fix_every_docstring,
)

console = Console()

class LintResult:
    """
    Stores the result of a lint operation on a single file.

    src/vibelint/lint.py
    """
    def __init__(self) -> None:
        self.file_path: Path = Path()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixed: bool = False

    @property
    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)


class LintRunner:
    """
    Runner for linting operations: checks shebang, encoding, docstrings, etc.

    src/vibelint/lint.py
    """
    def __init__(
        self,
        config: Dict[str, Any],
        check_only: bool = False,
        skip_confirmation: bool = False,
        include_vcs_hooks: bool = False,
    ) -> None:
        self.config = config
        self.check_only = check_only
        self.skip_confirmation = skip_confirmation
        self.include_vcs_hooks = include_vcs_hooks

        self.results: List[LintResult] = []
        self.files_fixed: int = 0
        self.files_with_errors: int = 0
        self.files_with_warnings: int = 0

    def run(self, paths: List[Path]) -> int:
        python_files = self._collect_python_files(paths)
        if not python_files:
            console.print("[yellow]No Python files found to lint.[/yellow]")
            return 0

        threshold = self.config.get("large_dir_threshold", 500)
        if not self.skip_confirmation and len(python_files) > threshold:
            if not self._confirm_large_directory(len(python_files)):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"Linting {len(python_files)} Python files...", total=len(python_files)
            )
            with ThreadPoolExecutor() as executor:
                for result in executor.map(self._process_file, python_files):
                    self.results.append(result)
                    progress.advance(task_id)

        for r in self.results:
            if r.fixed:
                self.files_fixed += 1
            if r.errors:
                self.files_with_errors += 1
            elif r.warnings:
                self.files_with_warnings += 1

        self._print_summary()

        if self.files_with_errors > 0 or (self.check_only and self.files_fixed > 0):
            return 1
        return 0

    def _collect_python_files(self, paths: List[Path]) -> List[Path]:
        python_files: List[Path] = []
        includes = self.config.get("include_globs", ["**/*.py"])
        excludes = self.config.get("exclude_globs", [])

        for p in paths:
            if p.is_file() and p.suffix == ".py":
                python_files.append(p)
            elif p.is_dir():
                for pat in includes:
                    for file_path in p.glob(pat):
                        if not file_path.is_file() or file_path.suffix != ".py":
                            continue
                        if not self.include_vcs_hooks and any(
                            part.startswith(".") and part in {".git", ".hg", ".svn"}
                            for part in file_path.parts
                        ):
                            continue
                        if any(fnmatch.fnmatch(str(file_path), str(p / e)) for e in excludes):
                            continue
                        python_files.append(file_path)
        return python_files

    def _confirm_large_directory(self, file_count: int) -> bool:
        console.print(
            f"[yellow]Warning:[/yellow] Found {file_count} Python files, "
            f"exceeding large_dir_threshold={self.config['large_dir_threshold']}."
        )
        return click.confirm("Do you want to continue?", default=True)

    def _process_file(self, file_path: Path) -> LintResult:
        lr = LintResult()
        lr.file_path = file_path
        try:
            original = file_path.read_text(encoding="utf-8")
            updated = original

            # 1) Shebang
            is_script = bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", updated))
            sb_res = validate_shebang(updated, is_script, self.config["allowed_shebangs"])
            if sb_res.has_issues():
                lr.errors.extend(sb_res.errors)
                lr.warnings.extend(sb_res.warnings)
                if not self.check_only:
                    updated = fix_shebang(updated, sb_res, is_script, self.config["allowed_shebangs"][0])

            # 2) Encoding
            enc_res = validate_encoding_cookie(updated)
            if enc_res.has_issues():
                lr.errors.extend(enc_res.errors)
                lr.warnings.extend(enc_res.warnings)
                if not self.check_only:
                    updated = fix_encoding_cookie(updated, enc_res)

            # 3) Docstring
            doc_res = validate_every_docstring(updated, str(file_path))
            if doc_res.has_issues():
                lr.errors.extend(doc_res.errors)
                lr.warnings.extend(doc_res.warnings)
                if not self.check_only:
                    updated = fix_every_docstring(updated, doc_res, str(file_path))

            if not self.check_only and updated != original:
                file_path.write_text(updated, encoding="utf-8")
                lr.fixed = True

        except Exception as e:
            lr.errors.append(f"Error processing file {file_path}: {e}")
        return lr

    def _print_summary(self) -> None:
        table = Table(title="vibelint Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        table.add_row("Files processed", str(len(self.results)))
        table.add_row("Files fixed", str(self.files_fixed))
        table.add_row("Files with errors", str(self.files_with_errors))
        table.add_row("Files with warnings", str(self.files_with_warnings))
        console.print(table)

        if self.files_with_errors > 0 or self.files_with_warnings > 0:
            console.print("\n[bold]Files with issues:[/bold]")
            for r in self.results:
                if r.has_issues:
                    status = "[red]ERROR[/red]" if r.errors else "[yellow]WARNING[/yellow]"
                    console.print(f"{status} {r.file_path}")
                    for err in r.errors:
                        console.print(f"  - [red]{err}[/red]")
                    for w in r.warnings:
                        console.print(f"  - [yellow]{w}[/yellow]")
