"""
Report generation functionality for vibelint.

vibelint/report.py
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .config import Config
from .namespace import NamespaceCollision, NamespaceNode
from .utils import get_relative_path

__all__ = ["write_report_content"]
logger = logging.getLogger(__name__)


def _get_files_in_namespace_order(
    node: NamespaceNode, collected_files: set[Path], project_root: Path
) -> None:
    """
    Recursively collects file paths from the namespace tree in DFS order,
    including __init__.py files for packages. Populates the collected_files set.

    Args:
        node: The current NamespaceNode.
        collected_files: A set to store the absolute paths of collected files.
        project_root: The project root path for checking containment.

    vibelint/report.py
    """

    if node.is_package and node.path and node.path.is_dir():
        try:

            node.path.relative_to(project_root)
            init_file = node.path / "__init__.py"

            if init_file.is_file() and init_file not in collected_files:

                init_file.relative_to(project_root)
                logger.debug(f"Report: Adding package init file: {init_file}")
                collected_files.add(init_file)
        except ValueError:
            logger.warning(f"Report: Skipping package node outside project root: {node.path}")
        except Exception as e:
            logger.error(f"Report: Error checking package init file for {node.path}: {e}")

    for child_name in sorted(node.children.keys()):
        child_node = node.children[child_name]

        if child_node.path and child_node.path.is_file() and not child_node.is_package:
            try:

                child_node.path.relative_to(project_root)
                if child_node.path not in collected_files:
                    logger.debug(f"Report: Adding module file: {child_node.path}")
                    collected_files.add(child_node.path)
            except ValueError:
                logger.warning(
                    f"Report: Skipping module file outside project root: {child_node.path}"
                )
            except Exception as e:
                logger.error(f"Report: Error checking module file {child_node.path}: {e}")

        _get_files_in_namespace_order(child_node, collected_files, project_root)

    if not node.children and node.path and node.path.is_file():
        try:
            node.path.relative_to(project_root)
            if node.path not in collected_files:
                logger.debug(f"Report: Adding root file node: {node.path}")
                collected_files.add(node.path)
        except ValueError:
            logger.warning(f"Report: Skipping root file node outside project root: {node.path}")
        except Exception as e:
            logger.error(f"Report: Error checking root file node {node.path}: {e}")


def write_report_content(
    f: TextIO,
    project_root: Path,
    target_paths: list[Path],
    findings: list,  # List of Finding objects from plugin system
    hard_coll: list[NamespaceCollision],
    soft_coll: list[NamespaceCollision],
    root_node: NamespaceNode,
    config: Config,
) -> None:
    """
    Writes the comprehensive markdown report content to the given file handle.

    Args:
    f: The text file handle to write the report to.
    project_root: The root directory of the project.
    target_paths: List of paths that were analyzed.
    findings: List of Finding objects from the plugin validation phase.
    hard_coll: List of hard NamespaceCollision objects.
    soft_coll: List of definition/export (soft) NamespaceCollision objects.
    root_node: The root NamespaceNode of the project structure.
    config: Configuration object.

    vibelint/report.py
    """

    package_name = project_root.name if project_root else "Unknown"

    f.write("# vibelint Report\n\n")
    f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    f.write(f"**Project:** {package_name}\n")

    f.write(f"**Project Root:** `{str(project_root.resolve())}`\n\n")
    f.write(f"**Paths analyzed:** {', '.join(str(p) for p in target_paths)}\n\n")

    f.write("## Table of Contents\n\n")
    f.write("1. [Summary](#summary)\n")
    f.write("2. [Linting Results](#linting-results)\n")
    f.write("3. [Namespace Structure](#namespace-structure)\n")
    f.write("4. [Namespace Collisions](#namespace-collisions)\n")
    f.write("5. [File Contents](#file-contents)\n\n")

    f.write("## Summary\n\n")
    f.write("| Metric | Count |\n")
    f.write("|--------|-------|\n")

    # Get unique files from findings
    analyzed_files = set(f.file_path for f in findings)
    files_analyzed_count = len(analyzed_files)
    f.write(f"| Files analyzed | {files_analyzed_count} |\n")

    # Count findings by severity
    from .plugin_system import Severity
    error_findings = [f for f in findings if f.severity == Severity.BLOCK]
    warn_findings = [f for f in findings if f.severity == Severity.WARN]

    f.write(f"| Findings with errors | {len(error_findings)} |\n")
    f.write(f"| Findings with warnings | {len(warn_findings)} |\n")
    f.write(f"| Hard namespace collisions | {len(hard_coll)} |\n")
    total_soft_collisions = len(soft_coll)
    f.write(f"| Definition/Export namespace collisions | {total_soft_collisions} |\n\n")

    f.write("## Linting Results\n\n")

    if not findings:
        f.write("*No validation issues found.*\n\n")
    else:
        # Group findings by file for better reporting
        from collections import defaultdict
        files_with_findings = defaultdict(list)
        for finding in findings:
            files_with_findings[finding.file_path].append(finding)

        f.write("| File | Rule | Severity | Message |\n")
        f.write("|------|------|----------|---------|\n")

        for file_path in sorted(files_with_findings.keys(), key=str):
            file_findings = files_with_findings[file_path]
            try:
                rel_path = get_relative_path(file_path.resolve(), project_root.resolve())
            except ValueError:
                rel_path = file_path

            for finding in sorted(file_findings, key=lambda f: f.line):
                location = f":{finding.line}" if finding.line > 0 else ""
                f.write(f"| `{rel_path}{location}` | `{finding.rule_id}` | {finding.severity.value} | {finding.message} |\n")
        f.write("\n")

    f.write("## Namespace Structure\n\n")
    f.write("```\n")
    try:

        tree_str = root_node.__str__()
        f.write(tree_str)
    except Exception as e:
        logger.error(f"Report: Error generating namespace tree string: {e}")
        f.write(f"[Error generating namespace tree: {e}]\n")
    f.write("\n```\n\n")

    f.write("## Namespace Collisions\n\n")
    f.write("### Hard Collisions\n\n")
    if not hard_coll:
        f.write("*No hard collisions detected.*\n\n")
    else:
        f.write("These collisions can break Python imports or indicate duplicate definitions:\n\n")
        f.write("| Name | Path 1 | Path 2 | Details |\n")
        f.write("|------|--------|--------|---------|\n")
        for collision in sorted(hard_coll, key=lambda c: (c.name, str(c.path1))):
            try:
                p1_rel = (
                    get_relative_path(collision.path1.resolve(), project_root.resolve())
                    if collision.path1
                    else "N/A"
                )
                p2_rel = (
                    get_relative_path(collision.path2.resolve(), project_root.resolve())
                    if collision.path2
                    else "N/A"
                )
            except ValueError:
                p1_rel = collision.path1 or "N/A"
                p2_rel = collision.path2 or "N/A"
            loc1 = f":{collision.lineno1}" if collision.lineno1 else ""
            loc2 = f":{collision.lineno2}" if collision.lineno2 else ""
            details = (
                "Intra-file duplicate" if str(p1_rel) == str(p2_rel) else "Module/Member clash"
            )
            f.write(f"| `{collision.name}` | `{p1_rel}{loc1}` | `{p2_rel}{loc2}` | {details} |\n")
        f.write("\n")

    f.write("### Definition & Export Collisions (Soft)\n\n")
    if not soft_coll:
        f.write("*No definition or export collisions detected.*\n\n")
    else:
        f.write(
            "These names are defined/exported in multiple files, which may confuse humans and LLMs:\n\n"
        )
        f.write("| Name | Type | Files Involved |\n")
        f.write("|------|------|----------------|\n")
        grouped_soft = defaultdict(lambda: {"paths": set(), "types": set()})
        for collision in soft_coll:
            all_paths = collision.definition_paths or [collision.path1, collision.path2]
            grouped_soft[collision.name]["paths"].update(p for p in all_paths if p)
            grouped_soft[collision.name]["types"].add(collision.collision_type)

        for name, data in sorted(grouped_soft.items()):
            paths_str_list = []
            for p in sorted(list(data["paths"]), key=str):
                try:
                    paths_str_list.append(
                        f"`{get_relative_path(p.resolve(), project_root.resolve())}`"
                    )
                except ValueError:
                    paths_str_list.append(f"`{p}`")
            type_str = (
                " & ".join(sorted([t.replace("_soft", "").upper() for t in data["types"]]))
                or "Unknown"
            )
            f.write(f"| `{name}` | {type_str} | {', '.join(paths_str_list)} |\n")
        f.write("\n")

    f.write("## File Contents\n\n")
    f.write("Files are ordered alphabetically by path.\n\n")

    collected_files_set: set[Path] = set()
    try:
        _get_files_in_namespace_order(root_node, collected_files_set, project_root.resolve())

        python_files_abs = sorted(list(collected_files_set), key=lambda p: str(p))
        logger.info(f"Report: Found {len(python_files_abs)} files for content section.")
    except Exception as e:
        logger.error(f"Report: Error collecting files for content section: {e}", exc_info=True)
        python_files_abs = []

    if not python_files_abs:
        f.write("*No Python files found in the namespace tree to display.*\n\n")
    else:
        for abs_file_path in python_files_abs:

            if abs_file_path and abs_file_path.is_file():
                try:

                    rel_path = get_relative_path(abs_file_path, project_root.resolve())
                    f.write(f"### {rel_path}\n\n")

                    try:
                        lang = "python"
                        content = abs_file_path.read_text(encoding="utf-8", errors="ignore")
                        f.write(f"```{lang}\n")
                        f.write(content)

                        if not content.endswith("\n"):
                            f.write("\n")
                        f.write("```\n\n")
                    except Exception as read_e:
                        logger.warning(
                            f"Report: Error reading file content for {rel_path}: {read_e}"
                        )
                        f.write(f"*Error reading file content: {read_e}*\n\n")

                except ValueError:

                    logger.warning(
                        f"Report: Skipping file outside project root in content section: {abs_file_path}"
                    )
                    f.write(f"### {abs_file_path} (Outside Project Root)\n\n")
                    f.write("*Skipping content as file is outside the detected project root.*\n\n")
                except Exception as e_outer:
                    logger.error(
                        f"Report: Error processing file entry for {abs_file_path}: {e_outer}",
                        exc_info=True,
                    )
                    f.write(f"### Error Processing Entry for {abs_file_path}\n\n")
                    f.write(f"*An unexpected error occurred: {e_outer}*\n\n")
            elif abs_file_path:
                logger.warning(
                    f"Report: Skipping non-file path found during content writing: {abs_file_path}"
                )
                f.write(f"### {abs_file_path} (Not a File)\n\n")
                f.write("*Skipping entry as it is not a file.*\n\n")

            f.write("---\n\n")
