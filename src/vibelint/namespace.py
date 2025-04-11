"""
Namespace representation and collision detection for vibelint.

vibelint/namespace.py
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import fnmatch

from rich.tree import Tree
from rich.console import Console


class NamespaceCollision:
    """
    Class to store information about a namespace collision.

    vibelint/namespace.py
    """

    def __init__(self, name: str, path1: Path, path2: Path):
        self.name = name
        self.path1 = path1
        self.path2 = path2

    def __str__(self) -> str:
        return f"Namespace collision: '{self.name}' in {self.path1} and {self.path2}"


class NamespaceNode:
    """
    Class to represent a node in the namespace tree.

    vibelint/namespace.py
    """

    def __init__(
        self, name: str, path: Optional[Path] = None, is_package: bool = False
    ):
        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: Dict[str, NamespaceNode] = {}
        self.members: Dict[str, Path] = {}  # Stores names defined at this level

    def add_child(
        self, name: str, path: Path, is_package: bool = False
    ) -> "NamespaceNode":
        """Add a child node to this node."""
        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)
        return self.children[name]

    def add_member(self, name: str, path: Path) -> None:
        """Add a member (variable, function, class) to this node."""
        self.members[name] = path

    def get_collisions(self) -> List[NamespaceCollision]:
        """Get all namespace collisions in this node and its children."""
        collisions: List[NamespaceCollision] = []

        # Check for collisions between children and members
        for name, path in self.members.items():
            if name in self.children:
                child = self.children[name]
                if child.path is None:
                    continue
                collisions.append(NamespaceCollision(name, path, child.path))

        # Check for collisions in children
        for child in self.children.values():
            collisions.extend(child.get_collisions())

        return collisions

    def to_tree(self, parent_tree: Optional[Tree] = None) -> Tree:
        """Convert this node to a rich.Tree for display."""
        # Create a new tree if this is the root
        if parent_tree is None:
            tree = Tree(f":package: {self.name}" if self.is_package else self.name)
        else:
            # Add this node as a branch to the parent tree
            tree = parent_tree.add(
                f":package: {self.name}" if self.is_package else self.name
            )

        # Add members
        if self.members:
            members_branch = tree.add(":page_facing_up: Members")
            for name in sorted(self.members.keys()):
                members_branch.add(name)

        # Add children
        for name, child in sorted(self.children.items()):
            child.to_tree(tree)

        return tree


def _extract_module_members(file_path: Path) -> List[str]:
    """
    Extract all top-level members from a Python module.

    vibelint/namespace.py
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        members = []
        module = ast.parse(content)

        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                members.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        members.append(target.id)

        return members
    except Exception:
        # If we can't parse the file, return an empty list
        return []


def _build_namespace_tree(
    paths: List[Path], config: Dict[str, Any], include_vcs_hooks: bool = False
) -> NamespaceNode:
    """
    Build a namespace tree from a list of paths.

    vibelint/namespace.py
    """
    # Create the root node
    root = NamespaceNode("root")

    # Keep track of all Python files
    python_files: List[Path] = []

    # Collect all Python files
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            python_files.append(path)
        elif path.is_dir():
            for include_glob in config["include_globs"]:
                # Generate pattern-matched paths
                matched_files = path.glob(include_glob)
                for file_path in matched_files:
                    # Skip if it's not a file or not a Python file
                    if not file_path.is_file() or file_path.suffix != ".py":
                        continue

                    # Skip VCS directories unless explicitly included
                    if not include_vcs_hooks and any(
                        part.startswith(".") and part in {".git", ".hg", ".svn"}
                        for part in file_path.parts
                    ):
                        continue

                    # Check exclude patterns
                    if any(
                        fnmatch.fnmatch(str(file_path), str(path / exclude_glob))
                        for exclude_glob in config["exclude_globs"]
                    ):
                        continue

                    python_files.append(file_path)

    # Find the common root of all files
    if python_files:
        # Convert to strings for easier manipulation
        file_paths_str = [str(p) for p in python_files]

        # Find common prefix
        common_prefix = os.path.commonpath(file_paths_str)

        # Build the namespace tree
        for file_path in python_files:
            # Get the relative path from the common root
            rel_path = str(file_path).replace(common_prefix, "").lstrip(os.sep)
            parts = rel_path.split(os.sep)

            # The last part is the file name
            file_name = parts[-1]

            # Navigate the tree and add packages/modules
            current = root
            for i, part in enumerate(parts[:-1]):
                # Determine if this directory is a package (contains __init__.py)
                package_path = Path(common_prefix, *parts[: i + 1], "__init__.py")
                is_package = package_path.exists()

                # Add this part to the tree
                current = current.add_child(
                    part, Path(common_prefix, *parts[: i + 1]), is_package
                )

            # Add the file as a module
            module_name = file_name[:-3]  # Remove .py extension
            is_package = module_name == "__init__"

            if is_package:
                # For __init__.py files, the members belong to the parent package
                members = _extract_module_members(file_path)
                for member in members:
                    current.add_member(member, file_path)
            else:
                # Add the module to the tree
                module_node = current.add_child(module_name, file_path)

                # Extract and add members from the module
                members = _extract_module_members(file_path)
                for member in members:
                    module_node.add_member(member, file_path)

    return root


def generate_namespace_representation(paths: List[Path], config: Dict[str, Any]) -> str:
    """
    Generate a text representation of the namespace.

    vibelint/namespace.py
    """
    # Build the namespace tree
    namespace_tree = _build_namespace_tree(paths, config)

    # Create a rich console to capture the output
    console = Console(width=100, record=True)

    # Print the tree representation
    tree = namespace_tree.to_tree()
    console.print(tree)

    # Check for collisions
    collisions = namespace_tree.get_collisions()
    if collisions:
        console.print("\n[bold red]Namespace Collisions:[/bold red]")
        for collision in collisions:
            console.print(
                f"- [red]'{collision.name}'[/red] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
            )

    # Return the captured output
    return console.export_text()


def detect_namespace_collisions(
    paths: List[Path], config: Dict[str, Any]
) -> List[NamespaceCollision]:
    """
    Detect namespace collisions in the given paths.

    vibelint/namespace.py
    """
    # Build the namespace tree
    namespace_tree = _build_namespace_tree(paths, config)

    # Return the collisions
    return namespace_tree.get_collisions()
