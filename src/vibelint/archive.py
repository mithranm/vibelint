"""
Tape archive generation in markdown for the Python codebase.

src/vibelint/archive.py
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any

def tape_archive(
    paths: List[Path],
    config: Dict[str, Any],
    output_path: Path,
    include_vcs_hooks: bool = False
) -> None:
    """
    Generate a single markdown "tape archive" of the codebase, 
    with a filesystem tree at the top, followed by file contents 
    (unless they're 'peeked').

    src/vibelint/archive.py
    """
    # 1) Collect files
    include_globs = config.get("include_globs", ["**/*.py"])
    exclude_globs = config.get("exclude_globs", [])
    peek_globs = config.get("peek_globs", [])

    file_infos = []  # list of (Path, category)
    for root_path in paths:
        if root_path.is_file():
            file_infos.append((root_path, "FULL"))
        else:
            for pat in include_globs:
                for f in root_path.glob(pat):
                    if not f.is_file():
                        continue
                    if not include_vcs_hooks and any(
                        part.startswith(".") and part in {".git", ".hg", ".svn"}
                        for part in f.parts
                    ):
                        continue
                    if any(fnmatch.fnmatch(str(f), str(root_path / e)) for e in exclude_globs):
                        continue
                    cat = "FULL"
                    for pk in peek_globs:
                        if fnmatch.fnmatch(str(f), str(root_path / pk)):
                            cat = "PEEK"
                            break
                    file_infos.append((f.resolve(), cat))

    # 2) Sort them
    file_infos = sorted(file_infos, key=lambda x: str(x[0]))

    # 3) Build a small filesystem tree. We'll do a naive approach: 
    # grouping by directories using a dict => { parent: [ (basename, cat), ... ] }
    # Then we recursively build.
    from collections import defaultdict
    root_map = defaultdict(list)

    # We'll pick the "common prefix" across all paths so we can create a relative tree
    all_files_str = [str(f) for (f, _) in file_infos]
    if not all_files_str:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Tape Archive\n\nNo files found.\n")
        return

    common_prefix = os.path.commonpath(all_files_str)
    # We'll store each path as relative to common_prefix
    rel_entries = []
    for f, cat in file_infos:
        relpath = os.path.relpath(str(f), common_prefix).replace("\\", "/")
        rel_entries.append((relpath, cat))

    # Build a naive tree structure
    tree = {}
    for rp, cat in rel_entries:
        parts = rp.split("/")
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # file
                if "__FILES__" not in node:
                    node["__FILES__"] = []
                node["__FILES__"].append((part, cat))
            else:
                if part not in node:
                    node[part] = {}
                node = node[part]

    # We'll define a function to pretty print the tree
    def print_tree(node, prefix=""):
        lines = []
        files = node.get("__FILES__", [])
        subdirs = sorted([k for k in node.keys() if k != "__FILES__"])

        for sd in subdirs:
            lines.append(prefix + sd + "/")
            subsub = print_tree(node[sd], prefix + "    ")
            lines.extend(subsub)

        for fn, cat in sorted(files, key=lambda x: x[0]):
            if cat == "PEEK":
                lines.append(prefix + fn + " (peek)")
            else:
                lines.append(prefix + fn)
        return lines

    tree_lines = print_tree(tree)

    with open(output_path, "w", encoding="utf-8") as outf:
        outf.write("# Tape Archive\n\n")
        outf.write("## Filesystem Tree\n\n")
        outf.write("```\n")
        for line in tree_lines:
            outf.write(line + "\n")
        outf.write("```\n\n")

        outf.write("## File Contents\n\n")
        for rp, cat in rel_entries:
            if cat == "PEEK":
                # We just mention them as omitted
                outf.write(f"### File: {rp}\n\n*(omitted due to peek_globs)*\n\n")
                continue
            # read
            fullp = Path(common_prefix, *rp.split("/"))
            if not fullp.exists():
                continue
            outf.write(f"### File: {rp}\n\n")
            try:
                content = fullp.read_text(encoding="utf-8")
                outf.write("```python\n")
                outf.write(content)
                outf.write("\n```\n\n")
            except Exception as e:
                outf.write(f"*Error reading file: {e}*\n\n")
