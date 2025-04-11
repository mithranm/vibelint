"""
Report generation functionality for vibelint.

vibelint/report.py
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from rich.console import Console

from .lint import LintRunner
from .namespace import (
    build_namespace_tree_representation,
    detect_namespace_collisions,
    detect_soft_member_collisions,
    get_files_in_namespace_order,
    _build_namespace_tree  # Import the internal function directly
)
from .utils import find_package_root

console = Console()

def generate_markdown_report(
    target_paths: List[Path],
    output_dir: Path,
    config: Dict[str, Any],
    check_only: bool = True,
    include_vcs_hooks: bool = False,
    ignore_inheritance: bool = False,
    output_filename: Optional[str] = None
) -> Path:
    """
    Generate a comprehensive markdown report of linting results, namespace structure,
    and file contents.
    
    Args:
        target_paths: List of paths to analyze
        output_dir: Directory where the report will be saved
        config: Configuration dictionary
        check_only: Only check for issues without suggesting fixes
        include_vcs_hooks: Whether to include version control hooks
        ignore_inheritance: Whether to ignore inheritance when checking for soft collisions
        output_filename: Optional filename for the report (default: vibelint_report_{timestamp}.md)
        
    Returns:
        Path to the generated report file
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"vibelint_report_{timestamp}.md"
    elif not output_filename.endswith('.md'):
        # Ensure the file has a markdown extension
        output_filename = f"{output_filename}.md"
        
    report_path = output_dir / output_filename
    
    # Run linting
    lint_runner = LintRunner(
        config=config,
        check_only=True,  # Always check only for reports
        skip_confirmation=True,
        include_vcs_hooks=include_vcs_hooks,
    )
    
    console.print("[bold blue]Running linting checks...[/bold blue]")
    lint_runner.run(target_paths)
    
    # Build namespace tree representation for display
    console.print("[bold blue]Building namespace structure...[/bold blue]")
    tree_repr = build_namespace_tree_representation(target_paths, config)
    
    # Get the actual namespace tree node for file ordering
    # Use _build_namespace_tree directly instead of trying to extract from tree_repr
    namespace_tree = _build_namespace_tree(target_paths, config, include_vcs_hooks)
    
    # Detect collisions
    console.print("[bold blue]Detecting namespace collisions...[/bold blue]")
    hard_collisions = detect_namespace_collisions(target_paths, config)
    soft_collisions = detect_soft_member_collisions(
        target_paths, config, use_inheritance_check=not ignore_inheritance
    )
    
    # Generate report
    console.print("[bold blue]Generating markdown report...[/bold blue]")
    with open(report_path, "w", encoding="utf-8") as f:
        # Report header
        package_roots = [find_package_root(path) for path in target_paths]
        package_names = []
        for p in package_roots:
            if p is not None and p.exists() and p.name:
                package_names.append(p.name)
        
        f.write("# vibelint Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"**Project(s):** {', '.join(package_names) or 'Unknown'}\n\n")
        f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Linting Results](#linting-results)\n")
        f.write("3. [Namespace Structure](#namespace-structure)\n")
        f.write("4. [Namespace Collisions](#namespace-collisions)\n")
        f.write("5. [File Contents](#file-contents)\n\n")
        
        # Summary section
        f.write("## Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Files analyzed | {len(lint_runner.results)} |\n")
        f.write(f"| Files with errors | {lint_runner.files_with_errors} |\n")
        f.write(f"| Files with warnings | {lint_runner.files_with_warnings} |\n")
        f.write(f"| Hard namespace collisions | {len(hard_collisions)} |\n")
        f.write(f"| Soft namespace collisions | {len(soft_collisions)} |\n\n")
        
        # Linting Results
        f.write("## Linting Results\n\n")
        if not lint_runner.results:
            f.write("*No linting results available.*\n\n")
        else:
            f.write("| File | Errors | Warnings |\n")
            f.write("|------|--------|----------|\n")
            for result in lint_runner.results:
                if result.has_issues:
                    errors = "; ".join(result.errors) or "None"
                    warnings = "; ".join(result.warnings) or "None"
                    f.write(f"| `{result.file_path}` | {errors} | {warnings} |\n")
            f.write("\n")
        
        # Namespace Structure
        f.write("## Namespace Structure\n\n")
        f.write("```\n")
        f.write(str(tree_repr))
        f.write("\n```\n\n")
        
        # Namespace Collisions
        f.write("## Namespace Collisions\n\n")
        
        # Hard collisions
        f.write("### Hard Collisions\n\n")
        if not hard_collisions:
            f.write("*No hard collisions detected.*\n\n")
        else:
            f.write("These collisions can break Python imports:\n\n")
            f.write("| Name | Path 1 | Path 2 |\n")
            f.write("|------|--------|--------|\n")
            for collision in hard_collisions:
                f.write(f"| `{collision.name}` | {collision.path1} | {collision.path2} |\n")
            f.write("\n")
        
        # Soft collisions
        f.write("### Soft Collisions\n\n")
        if not soft_collisions:
            f.write("*No soft collisions detected.*\n\n") 
        else:
            f.write("These don't break Python but may confuse humans and LLMs:\n\n")
            f.write("| Name | Path 1 | Path 2 |\n")
            f.write("|------|--------|--------|\n")
            for collision in soft_collisions:
                f.write(f"| `{collision.name}` | {collision.path1} | {collision.path2} |\n")
            f.write("\n")
        
        # File Contents
        f.write("## File Contents\n\n")
        f.write("Files are ordered by their position in the namespace hierarchy.\n\n")
        
        # Get all Python files from the namespace tree in a logical order
        python_files = get_files_in_namespace_order(namespace_tree)
        
        for file_path in python_files:
            if file_path.is_file() and file_path.suffix == '.py':
                rel_path = get_relative_path(file_path, target_paths)
                f.write(f"### {rel_path}\n\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        content = code_file.read()
                        f.write("```python\n")
                        f.write(content)
                        f.write("\n```\n\n")
                except Exception as e:
                    f.write(f"*Error reading file: {str(e)}*\n\n")
    
    return report_path


def get_relative_path(file_path: Path, base_paths: List[Path]) -> str:
    """
    Get the relative path of a file from the closest base path.
    
    Args:
        file_path: The file path to get the relative path for
        base_paths: List of base paths to use as reference
        
    Returns:
        The relative path as a string
    """
    shortest_path = None
    
    for base_path in base_paths:
        try:
            rel_path = file_path.relative_to(base_path)
            if shortest_path is None or len(str(rel_path)) < len(str(shortest_path)):
                shortest_path = rel_path
        except ValueError:
            continue
            
    return str(shortest_path) if shortest_path else str(file_path)