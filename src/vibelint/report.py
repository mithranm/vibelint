#!/usr/bin/env python3
"""
Report generation functionality for vibelint.

src/vibelint/report.py
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from rich.console import Console

from .lint import LintRunner
from .namespace import (
    detect_namespace_collisions,
    detect_soft_member_collisions,
    _build_namespace_tree  # Import the internal function directly
)
from .utils import find_package_root

console = Console()

# Define functions that appear to be needed but were missing from imports
def build_namespace_tree_representation(target_paths: List[Path], config: Dict[str, Any], 
                                       include_vcs_hooks: bool = False) -> str:
    """
    Build a string representation of the namespace tree.
    
    Args:
        target_paths: List of paths to analyze
        config: Configuration dictionary
        include_vcs_hooks: Whether to include version control hooks
        
    Returns:
        String representation of the namespace tree
    
    src/vibelint/report.py
    """
    # This is a placeholder implementation - you'll need to implement this
    # or import it correctly from the namespace module
    namespace_tree = _build_namespace_tree(target_paths, config, include_vcs_hooks)
    return str(namespace_tree)

def get_files_in_namespace_order(namespace_tree) -> List[Path]:
    """
    Get files in namespace order from the namespace tree.
    
    Args:
        namespace_tree: Namespace tree to extract files from
        
    Returns:
        List of file paths in namespace order
    
    src/vibelint/report.py
    """
    # This is a placeholder implementation - you'll need to implement this
    # or import it correctly from the namespace module
    files = []
    # Logic to traverse the tree and collect files would go here
    return files

def get_relative_path(file_path: Path, target_paths: List[Path]) -> str:
    """
    Get the shortest relative path for a file.
    
    Args:
        file_path: Absolute path to file
        target_paths: List of target paths to get relative paths from
        
    Returns:
        Shortest relative path as a string
    
    src/vibelint/report.py
    """
    shortest_path = None
    
    for target_path in target_paths:
        try:
            rel_path = file_path.relative_to(target_path)
            if shortest_path is None or len(str(rel_path)) < len(str(shortest_path)):
                shortest_path = rel_path
        except ValueError:
            continue
            
    return str(shortest_path) if shortest_path else str(file_path)

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
    
    src/vibelint/report.py
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
    tree_repr = build_namespace_tree_representation(target_paths, config, include_vcs_hooks)
    
    # Get the actual namespace tree node for file ordering
    namespace_tree = _build_namespace_tree(target_paths, config, include_vcs_hooks)
    
    # Detect collisions
    console.print("[bold blue]Detecting namespace collisions...[/bold blue]")
    hard_collisions = detect_namespace_collisions(target_paths, config, include_vcs_hooks)
    
    # Explicitly create the boolean value for clarity and add type hint
    check_inheritance: bool = not ignore_inheritance 
    
    soft_collisions = detect_soft_member_collisions(
        target_paths, 
        config, 
        use_inheritance_check=check_inheritance,  # Pass the variable here
        include_vcs_hooks=include_vcs_hooks
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
        if not lint_runner.results or all(not r.has_issues for r in lint_runner.results):
            f.write("*No linting issues found.*\n\n")
        else:
            f.write("| File | Errors | Warnings |\n")
            f.write("|------|--------|----------|\n")
            for result in lint_runner.results:
                if result.errors or result.warnings:
                    errors = "; ".join(result.errors) or "None"
                    warnings = "; ".join(result.warnings) or "None"
                    rel_path = get_relative_path(result.file_path, target_paths)
                    f.write(f"| `{rel_path}` | {errors} | {warnings} |\n")
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
                    content = file_path.read_text(encoding='utf-8')
                    f.write("```python\n")
                    f.write(content)
                    if not content.endswith('\n'):
                        f.write('\n')
                    f.write("```\n\n")
                except Exception as e:
                    f.write(f"*Error reading file: {e}*\n\n")
    
    console.print(f"Report generated: [bold green]{report_path}[/bold green]")
    return report_path