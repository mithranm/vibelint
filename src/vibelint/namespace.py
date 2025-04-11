"""
Namespace representation and collision detection for vibelint.

vibelint/namespace.py
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import fnmatch

from rich.tree import Tree
from rich.console import Console


class CollisionType:
    """Enum-like class for collision types."""
    HARD = "hard"  # Name conflicts that break Python imports
    SOFT = "soft"  # Same name in different modules, potentially confusing


class NamespaceCollision:
    """
    Class to store information about a namespace collision.

    vibelint/namespace.py
    """

    def __init__(self, name: str, path1: Path, path2: Path, collision_type: str = CollisionType.HARD):
        self.name = name
        self.path1 = path1
        self.path2 = path2
        self.collision_type = collision_type

    def __str__(self) -> str:
        type_str = "Hard" if self.collision_type == CollisionType.HARD else "Soft"
        return f"{type_str} collision: '{self.name}' in {self.path1} and {self.path2}"


class ClassInheritanceTracker:
    """Track inheritance relationships between classes to determine if name reuse is legitimate."""
    
    def __init__(self):
        self.inheritance_map: Dict[str, List[str]] = {}  # class -> parent classes
        self.class_locations: Dict[str, Path] = {}  # Fully qualified class name -> file path
    
    def add_class(self, class_name: str, parent_classes: List[str], file_path: Path, module_path: List[str]) -> None:
        """Add a class and its inheritance information."""
        qualified_name = ".".join([*module_path, class_name])
        
        if qualified_name not in self.inheritance_map:
            self.inheritance_map[qualified_name] = []
            self.class_locations[qualified_name] = file_path
        
        for parent in parent_classes:
            self.inheritance_map[qualified_name].append(parent)
    
    def is_related_through_inheritance(self, class1: str, class2: str) -> bool:
        """Check if two classes are related through inheritance."""
        if class1 == class2:
            return True
            
        # Check if class1 inherits from class2
        if class1 in self.inheritance_map:
            if class2 in self.inheritance_map[class1]:
                return True
            
            # Check recursively through all parent classes
            for parent in self.inheritance_map[class1]:
                if self.is_related_through_inheritance(parent, class2):
                    return True
        
        # Check if class2 inherits from class1
        if class2 in self.inheritance_map:
            if class1 in self.inheritance_map[class2]:
                return True
            
            # Check recursively through all parent classes
            for parent in self.inheritance_map[class2]:
                if self.is_related_through_inheritance(parent, class1):
                    return True
        
        return False


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
        # Track the file path where this node is defined
        self.file_path = path if path and path.is_file() else None

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


def _extract_imports_and_all(file_path: Path) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Extract imports and __all__ from a Python module.
    
    Returns:
    - import_map: Dict mapping imported name to its source
    - imported_modules: List of modules imported with 'import module'
    - all_names: List of names in __all__
    """
    import_map = {}
    imported_modules = []
    all_names = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        module = ast.parse(content)
        
        for node in ast.walk(module):
            # Handle 'from X import Y' statements
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""  # Handle 'from . import X'
                for name in node.names:
                    import_name = name.asname or name.name
                    import_source = f"{module_name}.{name.name}" if module_name else name.name
                    import_map[import_name] = import_source
            
            # Handle 'import X' and 'import X.Y' statements
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.asname:
                        import_map[name.asname] = name.name
                    else:
                        imported_modules.append(name.name)
                        import_map[name.name.split(".")[-1]] = name.name
            
            # Extract __all__ list
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_names.append(elt.value)
                                elif isinstance(elt, ast.Str):  # For Python < 3.8
                                    all_names.append(elt.s)
    
    except Exception:
        # If we can't parse the file, return empty collections
        pass
        
    return import_map, imported_modules, all_names


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


def build_namespace_tree_representation(paths: List[Path], config: Dict[str, Any]) -> Tree:
    """
    Build and return the Rich Tree representation of the namespace without printing it.
    
    vibelint/namespace.py
    """
    # Build the namespace tree
    namespace_tree = _build_namespace_tree(paths, config)
    
    # Create and return the tree representation without printing
    return namespace_tree.to_tree()


def get_namespace_collisions_str(paths: List[Path], config: Dict[str, Any]) -> str:
    """
    Get a formatted string of namespace collisions if any exist.
    
    vibelint/namespace.py
    """
    # Build the namespace tree
    namespace_tree = _build_namespace_tree(paths, config)
    
    # Check for collisions
    collisions = namespace_tree.get_collisions()
    if not collisions:
        return ""
        
    # Format collisions as a string
    console = Console(width=100, record=True)
    console.print("\n[bold red]Namespace Collisions:[/bold red]")
    for collision in collisions:
        console.print(
            f"- [red]'{collision.name}'[/red] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
        )
    return console.export_text()


def generate_namespace_representation(paths: List[Path], config: Dict[str, Any]) -> str:
    """
    Generate a text representation of the namespace.
    
    vibelint/namespace.py
    """
    # Return the tree and collision information
    tree = build_namespace_tree_representation(paths, config)
    collision_str = get_namespace_collisions_str(paths, config)
    
    # Format the final output without double-printing
    console = Console(width=100, record=True)
    console.print(tree)
    if collision_str:
        console.print(collision_str)
    
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
    
    # Get basic collisions from the namespace tree
    collisions = namespace_tree.get_collisions()
    
    # Now check for additional collision types
    # 1. Find all init files to check for import/module collisions
    init_files = []
    python_modules = {}
    
    for path in paths:
        if path.is_file():
            if path.name == "__init__.py":
                init_files.append(path)
            elif path.suffix == ".py":
                python_modules[path.stem] = path
        elif path.is_dir():
            for file_path in path.rglob("*.py"):
                if file_path.name == "__init__.py":
                    init_files.append(file_path)
                else:
                    python_modules[file_path.stem] = file_path
    
    # 2. For each init file, check for module/import collisions
    for init_file in init_files:
        package_dir = init_file.parent
        import_map, imported_modules, all_names = _extract_imports_and_all(init_file)
        
        # Check for name conflicts between imports and sibling modules
        for module_name, module_path in python_modules.items():
            # Only consider modules in the same directory as this __init__.py
            if module_path.parent != package_dir:
                continue
                
            # Check if this module name conflicts with an imported name
            if module_name in import_map:
                collisions.append(
                    NamespaceCollision(
                        name=module_name,
                        path1=module_path,
                        path2=init_file,
                        collision_type=CollisionType.HARD
                    )
                )
                
        # Check for duplicate names in __all__
        name_counts = {}
        for name in all_names:
            if name not in name_counts:
                name_counts[name] = 0
            name_counts[name] += 1
            
        for name, count in name_counts.items():
            if count > 1:
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        path1=init_file,
                        path2=init_file,  # Same file for duplicates in __all__
                        collision_type=CollisionType.HARD
                    )
                )
        
        # Check if any name in __all__ appears as both an import and a module
        for name in all_names:
            # Check if this name is both a module file and imported
            module_path = package_dir / f"{name}.py"
            if module_path.exists() and name in import_map:
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        path1=module_path,
                        path2=init_file,
                        collision_type=CollisionType.HARD
                    )
                )

    return collisions


def _extract_class_inheritance(file_path: Path, module_path: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Extract class inheritance information from a Python file.
    
    Returns a list of tuples (class_name, [parent_classes])
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        classes = []
        module = ast.parse(content)

        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                parent_classes = []
                
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_classes.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        # Handle cases like 'module.Class'
                        attr_parts = []
                        current = base
                        
                        while isinstance(current, ast.Attribute):
                            attr_parts.append(current.attr)
                            current = current.value
                            
                        if isinstance(current, ast.Name):
                            attr_parts.append(current.id)
                            parent_classes.append(".".join(reversed(attr_parts)))
                
                classes.append((class_name, parent_classes))
        
        return classes
    except Exception:
        # If we can't parse the file, return an empty list
        return []


def detect_soft_member_collisions(
    paths: List[Path], config: Dict[str, Any], use_inheritance_check: bool = True
) -> List[NamespaceCollision]:
    """
    Find member names that appear in multiple modules without inheritance relationships.
    
    These are "soft collisions" - they don't break Python but can confuse humans and LLMs.
    """
    # First, build the namespace tree
    namespace_tree = _build_namespace_tree(paths, config)
    
    # Track all member definitions and their locations
    member_definitions: Dict[str, List[Tuple[Path, List[str]]]] = {}
    
    # Track inheritance relationships if needed
    inheritance_tracker = ClassInheritanceTracker() if use_inheritance_check else None
    
    def traverse_for_members(node: NamespaceNode, module_path: List[str]) -> None:
        """Traverse the namespace tree and collect member definitions."""
        # Process this node's members
        for member_name, file_path in node.members.items():
            if member_name not in member_definitions:
                member_definitions[member_name] = []
            member_definitions[member_name].append((file_path, module_path))
            
        # Process children
        for child_name, child_node in node.children.items():
            # Skip "__init__" since it's a special case
            if child_name == "__init__":
                continue
                
            # Add this child to the module path when traversing
            new_path = module_path + [child_name]
            traverse_for_members(child_node, new_path)
            
            # If we're tracking inheritance, extract class information
            if use_inheritance_check and inheritance_tracker and child_node.path and child_node.path.is_file():
                for class_name, parents in _extract_class_inheritance(child_node.path, new_path):
                    inheritance_tracker.add_class(class_name, parents, child_node.path, new_path)
    
    # Start traversal from the root
    traverse_for_members(namespace_tree, [])
    
    # Find soft collisions
    soft_collisions: List[NamespaceCollision] = []
    
    for member_name, locations in member_definitions.items():
        if len(locations) <= 1:
            continue
            
        # Check all pairs of locations
        for i in range(len(locations)):
            file_path1, module_path1 = locations[i]
            
            for j in range(i + 1, len(locations)):
                file_path2, module_path2 = locations[j]
                
                # Skip comparing a member to itself (same file)
                if file_path1 == file_path2:
                    continue
                
                # If inheritance checking is enabled and we find these are related by inheritance, it's not a collision
                is_related = False
                if use_inheritance_check and inheritance_tracker:
                    qualified_name1 = ".".join(module_path1)
                    qualified_name2 = ".".join(module_path2)
                    is_related = inheritance_tracker.is_related_through_inheritance(qualified_name1, qualified_name2)
                
                if not is_related:
                    soft_collisions.append(
                        NamespaceCollision(
                            name=member_name,
                            path1=file_path1,
                            path2=file_path2,
                            collision_type=CollisionType.SOFT
                        )
                    )
    
    return soft_collisions


def get_soft_collisions_str(paths: List[Path], config: Dict[str, Any]) -> str:
    """
    Get a formatted string of soft namespace collisions if any exist.
    
    vibelint/namespace.py
    """
    # Detect soft collisions
    soft_collisions = detect_soft_member_collisions(paths, config)
    
    if not soft_collisions:
        return ""
        
    # Format collisions as a string
    console = Console(width=100, record=True)
    console.print("\n[bold yellow]Soft Namespace Collisions:[/bold yellow]")
    console.print("[dim](These don't break Python but may confuse humans and LLMs)[/dim]")
    
    for collision in soft_collisions:
        console.print(
            f"- [yellow]'{collision.name}'[/yellow] in [cyan]{collision.path1}[/cyan] and [cyan]{collision.path2}[/cyan]"
        )
    return console.export_text()


def get_files_in_namespace_order(namespace_tree: NamespaceNode) -> List[Path]:
    """
    Get all Python files from the namespace tree in a logical order.
    
    Args:
        namespace_tree: The namespace tree root node
        
    Returns:
        List of file paths ordered by namespace hierarchy
    """
    files = []
    visited = set()
    
    def traverse(node: NamespaceNode):
        # First add the node's file if it exists
        if node.file_path and node.file_path not in visited:
            files.append(node.file_path)
            visited.add(node.file_path)
            
        # Then traverse children in alphabetical order
        for name in sorted(node.children.keys()):
            traverse(node.children[name])
    
    traverse(namespace_tree)
    return files
