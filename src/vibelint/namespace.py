"""
Namespace representation & collision detection for Python code.

vibelint/namespace.py
"""

import ast
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .discovery import discover_files
from .utils import find_project_root, get_relative_path

__all__ = [
    "CollisionType",
    "NamespaceCollision",
    "NamespaceNode",
    "detect_hard_collisions",
    "detect_global_definition_collisions",
    "detect_local_export_collisions",
    "build_namespace_tree",
    "get_namespace_collisions_str",
]

logger = logging.getLogger(__name__)


class CollisionType:
    """
    Enum-like class for collision types.

    vibelint/namespace.py
    """

    HARD = "hard"
    LOCAL_SOFT = "local_soft"
    GLOBAL_SOFT = "global_soft"


class NamespaceCollision:
    """
    Represents a collision between two or more same-named entities.

    vibelint/namespace.py
    """

    def __init__(
        self,
        name: str,
        collision_type: str,
        paths: List[Path],
        linenos: Optional[List[Optional[int]]] = None,
    ) -> None:
        """
        Initializes a NamespaceCollision instance.

        Args:
        name: The name of the colliding entity.
        collision_type: The type of collision (HARD, LOCAL_SOFT, GLOBAL_SOFT).
        paths: A list of Path objects for all files involved in the collision.
        linenos: An optional list of line numbers corresponding to each path.

        vibelint/namespace.py
        """

        if not paths:
            raise ValueError("At least one path must be provided for a collision.")

        self.name = name
        self.collision_type = collision_type

        self.paths = sorted(list(set(paths)), key=str)

        self.linenos = (
            linenos if linenos and len(linenos) == len(self.paths) else [None] * len(self.paths)
        )

        self.path1: Path = self.paths[0]
        self.path2: Path = self.paths[1] if len(self.paths) > 1 else self.paths[0]
        self.lineno1: Optional[int] = self.linenos[0] if self.linenos else None
        self.lineno2: Optional[int] = self.linenos[1] if len(self.linenos) > 1 else self.lineno1

        self.definition_paths: List[Path] = (
            self.paths
            if self.collision_type in [CollisionType.GLOBAL_SOFT, CollisionType.LOCAL_SOFT]
            else []
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for debugging.

        vibelint/namespace.py
        """

        return (
            f"NamespaceCollision(name='{self.name}', type='{self.collision_type}', "
            f"paths={self.paths}, linenos={self.linenos})"
        )

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the collision.

        vibelint/namespace.py
        """

        proj_root = find_project_root(Path(".").resolve())
        base_path = proj_root if proj_root else Path(".")

        paths_str_list = []
        for i, p in enumerate(self.paths):
            loc = f":{self.linenos[i]}" if self.linenos and self.linenos[i] is not None else ""
            try:
                paths_str_list.append(f"{get_relative_path(p, base_path)}{loc}")
            except ValueError:
                paths_str_list.append(f"{p}{loc}")
        paths_str = ", ".join(paths_str_list)

        if self.collision_type == CollisionType.HARD:
            if len(self.paths) == 2 and self.paths[0] == self.paths[1]:

                line_info = ""
                if self.lineno1 is not None and self.lineno2 is not None:
                    line_info = f" (lines ~{self.lineno1} and ~{self.lineno2})"
                elif self.lineno1 is not None:
                    line_info = f" (line ~{self.lineno1})"

                return (
                    f"{self.collision_type.upper()} Collision: Duplicate definition/import of '{self.name}' in "
                    f"{paths_str_list[0]}{line_info}"
                )
            else:
                return f"{self.collision_type.upper()} Collision: Name '{self.name}' used by conflicting entities in: {paths_str}"
        elif self.collision_type == CollisionType.LOCAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' exported via __all__ in multiple sibling modules: {paths_str}"
        elif self.collision_type == CollisionType.GLOBAL_SOFT:
            return f"{self.collision_type.upper()} Collision: '{self.name}' defined in multiple modules: {paths_str}"
        else:
            return f"Unknown Collision: '{self.name}' involving paths: {paths_str}"


def detect_hard_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect HARD collisions: member vs. submodule, or duplicate definitions within a file.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected HARD NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, intra_file_collisions = build_namespace_tree(paths, config)

    inter_file_collisions = root_node.get_hard_collisions()

    all_collisions = intra_file_collisions + inter_file_collisions
    for c in all_collisions:
        c.collision_type = CollisionType.HARD
    return all_collisions


def detect_global_definition_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect GLOBAL SOFT collisions: the same name defined/assigned at the top level
    in multiple different modules across the project.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected GLOBAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)

    definition_collisions = root_node.detect_global_definition_collisions()

    return definition_collisions


def detect_local_export_collisions(
    paths: List[Path],
    config: Config,
) -> List[NamespaceCollision]:
    """
    Detect LOCAL SOFT collisions: the same name exported via __all__ by multiple
    sibling modules within the same package.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns:
    A list of detected LOCAL_SOFT NamespaceCollision objects.

    vibelint/namespace.py
    """

    root_node, _ = build_namespace_tree(paths, config)
    collisions: List[NamespaceCollision] = []
    root_node.find_local_export_collisions(collisions)
    return collisions


def get_namespace_collisions_str(
    paths: List[Path],
    config: Config,
    console=None,
) -> str:
    """
    Return a string representation of all collision types for quick debugging.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.
    console: Optional console object (unused).

    Returns:
    A string summarizing all detected collisions.

    vibelint/namespace.py
    """

    from io import StringIO

    buf = StringIO()

    hard_collisions = detect_hard_collisions(paths, config)
    global_soft_collisions = detect_global_definition_collisions(paths, config)
    local_soft_collisions = detect_local_export_collisions(paths, config)

    proj_root = find_project_root(Path(".").resolve())
    base_path = proj_root if proj_root else Path(".")

    if hard_collisions:
        buf.write("Hard Collisions:\n")
        for c in sorted(hard_collisions, key=lambda x: (x.name, str(x.paths[0]))):
            buf.write(f"- {str(c)}\n")

    if local_soft_collisions:
        buf.write("\nLocal Soft Collisions (__all__):\n")

        grouped = defaultdict(list)
        for c in local_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': exported by {paths_str}\n")

    if global_soft_collisions:
        buf.write("\nGlobal Soft Collisions (Definitions):\n")

        grouped = defaultdict(list)
        for c in global_soft_collisions:
            grouped[c.name].extend(c.paths)
        for name, involved_paths in sorted(grouped.items()):
            try:
                paths_str = ", ".join(
                    sorted(str(get_relative_path(p, base_path)) for p in set(involved_paths))
                )
            except ValueError:
                paths_str = ", ".join(sorted(str(p) for p in set(involved_paths)))
            buf.write(f"- '{name}': defined in {paths_str}\n")

    return buf.getvalue()


class NamespaceNode:
    """
    A node in the "module" hierarchy (like package/subpackage, or file-level).
    Holds child nodes and top-level members (functions/classes).

    vibelint/namespace.py
    """

    def __init__(self, name: str, path: Optional[Path] = None, is_package: bool = False) -> None:
        """
        Initializes a NamespaceNode.

        Args:
        name: The name of the node (e.g., module name, package name).
        path: The filesystem path associated with this node (optional).
        is_package: True if this node represents a package (directory).

        vibelint/namespace.py
        """

        self.name = name
        self.path = path
        self.is_package = is_package
        self.children: Dict[str, "NamespaceNode"] = {}

        self.members: Dict[str, Tuple[Path, Optional[int]]] = {}

        self.member_collisions: List[NamespaceCollision] = []

        self.exported_names: Optional[List[str]] = None

    def set_exported_names(self, names: List[str]):
        """
        Sets the list of names found in __all__.

        vibelint/namespace.py
        """

        self.exported_names = names

    def add_child(self, name: str, path: Path, is_package: bool = False) -> "NamespaceNode":
        """
        Adds a child node, creating if necessary.

        vibelint/namespace.py
        """

        if name not in self.children:
            self.children[name] = NamespaceNode(name, path, is_package)

        elif path:

            if not (self.children[name].is_package and not is_package):
                self.children[name].path = path
            self.children[name].is_package = is_package or self.children[name].is_package
        return self.children[name]

    def get_hard_collisions(self) -> List[NamespaceCollision]:
        """
        Detect HARD collisions recursively: members vs. child modules.

        vibelint/namespace.py
        """

        collisions: List[NamespaceCollision] = []

        member_names_with_info = {}
        if self.is_package and self.path:
            init_path = (self.path / "__init__.py").resolve()
            member_names_with_info = {
                name: (def_path, lineno)
                for name, (def_path, lineno) in self.members.items()
                if def_path.resolve() == init_path
            }

        child_names = set(self.children.keys())
        common_names = set(member_names_with_info.keys()).intersection(child_names)

        for name in common_names:

            member_def_path, member_lineno = member_names_with_info.get(name, (None, None))
            cnode = self.children[name]
            child_path = cnode.path

            if member_def_path and child_path:

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[member_def_path, child_path],
                        linenos=[member_lineno, None],
                    )
                )

        for cnode in self.children.values():
            collisions.extend(cnode.get_hard_collisions())
        return collisions

    def collect_defined_members(self, all_dict: Dict[str, List[Tuple[Path, Optional[int]]]]):
        """
        Recursively collects defined members (path, lineno) for global definition collision check.

        vibelint/namespace.py
        """

        if self.path and self.members:

            for mname, (mpath, mlineno) in self.members.items():
                all_dict.setdefault(mname, []).append((mpath, mlineno))

        for cnode in self.children.values():
            cnode.collect_defined_members(all_dict)

    def detect_global_definition_collisions(self) -> List[NamespaceCollision]:
        """
        Detects GLOBAL SOFT collisions across the whole tree starting from this node.

        vibelint/namespace.py
        """

        all_defined_members: Dict[str, List[Tuple[Path, Optional[int]]]] = defaultdict(list)
        self.collect_defined_members(all_defined_members)

        collisions: List[NamespaceCollision] = []
        for name, path_lineno_list in all_defined_members.items():

            unique_paths_map: Dict[Path, Optional[int]] = {}
            for path, lineno in path_lineno_list:
                resolved_p = path.resolve()

                if resolved_p not in unique_paths_map:
                    unique_paths_map[resolved_p] = lineno

            if len(unique_paths_map) > 1:

                sorted_paths = sorted(unique_paths_map.keys(), key=str)

                sorted_linenos = [unique_paths_map[p] for p in sorted_paths]

                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.GLOBAL_SOFT,
                        paths=sorted_paths,
                        linenos=sorted_linenos,
                    )
                )
        return collisions

    def find_local_export_collisions(self, collisions_list: List[NamespaceCollision]):
        """
        Recursively finds LOCAL SOFT collisions (__all__) within packages.

        Args:
        collisions_list: A list to append found collisions to.

        vibelint/namespace.py
        """

        if self.is_package:
            exports_in_package: Dict[str, List[Path]] = defaultdict(list)

            if self.path and self.path.is_dir() and self.exported_names:

                init_path = (self.path / "__init__.py").resolve()

                if init_path.exists() and any(
                    p.resolve() == init_path for p, _ in self.members.values()
                ):
                    for name in self.exported_names:
                        exports_in_package[name].append(init_path)

            for child in self.children.values():

                if (
                    child.path
                    and child.path.is_file()
                    and not child.is_package
                    and child.name != "__init__"
                    and child.exported_names
                ):
                    for name in child.exported_names:
                        exports_in_package[name].append(child.path.resolve())

            for name, paths in exports_in_package.items():
                unique_paths = sorted(list(set(paths)), key=str)
                if len(unique_paths) > 1:
                    collisions_list.append(
                        NamespaceCollision(
                            name=name,
                            collision_type=CollisionType.LOCAL_SOFT,
                            paths=unique_paths,
                            linenos=[None for _ in unique_paths],
                        )
                    )

        for child in self.children.values():
            if child.is_package:
                child.find_local_export_collisions(collisions_list)

    def __str__(self) -> str:
        """
        Provides a string representation of the node and its subtree, including members.
        Uses a revised formatting approach for better clarity relative to project root.

        vibelint/namespace.py
        """

        lines = []

        proj_root = find_project_root(Path(".").resolve())
        base_path_for_display = proj_root if proj_root else Path(".")

        def build_tree_lines(node: "NamespaceNode", prefix: str = "", base: Path = Path(".")):
            """
            Docstring for function 'build_tree_lines'.

            vibelint/namespace.py
            """

            child_items = sorted(node.children.items())

            direct_members = []
            if node.path and node.members:

                expected_def_path = None
                node_path_resolved = node.path.resolve()
                if node.is_package and node_path_resolved.is_dir():
                    expected_def_path = (node_path_resolved / "__init__.py").resolve()
                elif node_path_resolved.is_file():
                    expected_def_path = node_path_resolved

                if expected_def_path:
                    direct_members = sorted(
                        [
                            name
                            for name, (def_path, _) in node.members.items()
                            if def_path.resolve() == expected_def_path
                        ]
                    )

            all_items = child_items + [(name, "member") for name in direct_members]
            total_items = len(all_items)

            for i, (name, item) in enumerate(all_items):
                is_last = i == total_items - 1
                connector = "└── " if is_last else "├── "
                next_level_prefix = prefix + ("    " if is_last else "│   ")

                if item == "member":

                    lines.append(f"{prefix}{connector}{name} (member)")
                else:

                    child: "NamespaceNode" = item
                    child_path_str = ""
                    indicator = ""
                    if child.path:
                        try:
                            rel_p = get_relative_path(child.path, base)

                            if child.is_package:
                                indicator = " (P)"
                            elif child.name == "__init__":
                                indicator = " (I)"
                            else:
                                indicator = " (M)"
                            child_path_str = f"  [{rel_p}{indicator}]"
                        except ValueError:
                            indicator = (
                                " (P)"
                                if child.is_package
                                else (" (I)" if child.name == "__init__" else " (M)")
                            )
                            child_path_str = f"  [{child.path.resolve()}{indicator}]"
                    else:
                        child_path_str = "  [No Path]"

                    lines.append(f"{prefix}{connector}{name}{child_path_str}")

                    if child.children or (
                        child.members
                        and any(
                            m_path.resolve() == (child.path.resolve() if child.path else None)
                            for m, (m_path, _) in child.members.items()
                        )
                    ):
                        build_tree_lines(child, next_level_prefix, base)

        root_path_str = ""
        root_indicator = ""

        if self.path:
            root_path_resolved = self.path.resolve()
            try:

                rel_p = get_relative_path(root_path_resolved, base_path_for_display.parent)

                if rel_p == Path("."):
                    rel_p = Path(self.name)

                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{rel_p}{root_indicator}]"
            except ValueError:
                root_indicator = (
                    " (P)" if self.is_package else (" (M)" if root_path_resolved.is_file() else "")
                )
                root_path_str = f"  [{root_path_resolved}{root_indicator}]"
        else:
            root_path_str = "  [No Path]"

        lines.append(f"{self.name}{root_path_str}")
        build_tree_lines(self, prefix="", base=base_path_for_display)
        return "\n".join(lines)


def _extract_module_members(
    file_path: Path,
) -> Tuple[Dict[str, Tuple[Path, Optional[int]]], List[NamespaceCollision], Optional[List[str]]]:
    """
    Parses a Python file and extracts top-level member definitions/assignments,
    intra-file hard collisions, and the contents of __all__ if present.

    Returns:
    - A dictionary mapping defined/assigned names to a tuple of (file path, line number).
    - A list of intra-file hard collisions (NamespaceCollision objects).
    - A list of names in __all__, or None if __all__ is not found or invalid.

    vibelint/namespace.py
    """

    try:
        source = file_path.read_text(encoding="utf-8")

        tree = ast.parse(source, filename=str(file_path))
    except Exception as e:
        logger.warning(f"Could not parse file {file_path} for namespace analysis: {e}")

        return {}, [], None

    defined_members_map: Dict[str, Tuple[Path, Optional[int]]] = {}
    collisions: List[NamespaceCollision] = []
    exported_names: Optional[List[str]] = None

    defined_names_nodes: Dict[str, ast.AST] = {}

    for node in tree.body:
        current_node = node
        name: Optional[str] = None
        is_definition = False
        is_all_assignment = False
        lineno = getattr(current_node, "lineno", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            is_definition = True
        elif isinstance(node, ast.Assign):

            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                is_all_assignment = True

                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exported_names = []
                    for elt in node.value.elts:

                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exported_names.append(elt.value)

                if "__all__" not in defined_names_nodes:
                    defined_names_nodes["__all__"] = current_node
                else:
                    first_node = defined_names_nodes["__all__"]
                    collisions.append(
                        NamespaceCollision(
                            name="__all__",
                            collision_type=CollisionType.HARD,
                            paths=[file_path, file_path],
                            linenos=[getattr(first_node, "lineno", None), lineno],
                        )
                    )

            else:

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        is_definition = True

                        if name:
                            if name in defined_names_nodes:

                                first_node = defined_names_nodes[name]
                                collisions.append(
                                    NamespaceCollision(
                                        name=name,
                                        collision_type=CollisionType.HARD,
                                        paths=[file_path, file_path],
                                        linenos=[
                                            getattr(first_node, "lineno", None),
                                            lineno,
                                        ],
                                    )
                                )
                            else:

                                defined_names_nodes[name] = current_node
                                defined_members_map[name] = (
                                    file_path,
                                    lineno,
                                )
                            name = None

        if name and is_definition and not is_all_assignment:
            if name in defined_names_nodes:

                first_node = defined_names_nodes[name]
                collisions.append(
                    NamespaceCollision(
                        name=name,
                        collision_type=CollisionType.HARD,
                        paths=[file_path, file_path],
                        linenos=[getattr(first_node, "lineno", None), lineno],
                    )
                )
            else:

                defined_names_nodes[name] = current_node
                defined_members_map[name] = (file_path, lineno)

    return defined_members_map, collisions, exported_names


def build_namespace_tree(
    paths: List[Path], config: Config
) -> Tuple[NamespaceNode, List[NamespaceCollision]]:
    """
    Builds the namespace tree, collects intra-file collisions, and stores members/__all__.

    Args:
    paths: List of target paths (files or directories).
    config: The loaded vibelint configuration object.

    Returns a tuple: (root_node, all_intra_file_collisions)

    vibelint/namespace.py
    """

    project_root_found = config.project_root or find_project_root(
        paths[0].resolve() if paths else Path(".")
    )
    if not project_root_found:

        project_root_found = Path(".")
        root_node_name = "root"
        logger.warning(
            "Could not determine project root. Using '.' as root for namespace analysis."
        )
    else:
        root_node_name = project_root_found.name

    root = NamespaceNode(root_node_name, path=project_root_found.resolve(), is_package=True)
    root_path_for_rel = project_root_found.resolve()
    all_intra_file_collisions: List[NamespaceCollision] = []

    python_files = [
        f
        for f in discover_files(
            paths,
            config,
        )
        if f.suffix == ".py"
    ]

    if not python_files:
        logger.info("No Python files found for namespace analysis based on configuration.")
        return root, all_intra_file_collisions

    for f in python_files:
        try:

            rel_path = f.relative_to(root_path_for_rel)
            rel_parts = list(rel_path.parts)
        except ValueError:

            rel_parts = [f.name]
            logger.warning(
                f"File {f} is outside the determined project root {root_path_for_rel}. Adding directly under root."
            )

        current = root

        for i, part in enumerate(rel_parts[:-1]):

            dir_path = root_path_for_rel.joinpath(*rel_parts[: i + 1])
            current = current.add_child(part, dir_path, is_package=True)

        file_name = rel_parts[-1]
        mod_name = Path(file_name).stem
        file_abs_path = f

        members, intra_collisions, exported_names = _extract_module_members(file_abs_path)
        all_intra_file_collisions.extend(intra_collisions)

        if mod_name == "__init__":

            package_node = current
            package_node.is_package = True
            package_node.path = file_abs_path.parent

            for m_name, m_info in members.items():
                if m_name not in package_node.members:
                    package_node.members[m_name] = m_info

            if exported_names is not None:
                package_node.set_exported_names(exported_names)

        else:

            module_node = current.add_child(mod_name, file_abs_path, is_package=False)
            module_node.members = members
            if exported_names is not None:
                module_node.set_exported_names(exported_names)
            module_node.member_collisions.extend(intra_collisions)

    return root, all_intra_file_collisions
