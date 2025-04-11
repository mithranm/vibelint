"""
Namespace representation & collision detection for Python code.

src/vibelint/namespace.py
"""

import os
import ast
import fnmatch
from pathlib import Path
from typing import Dict, List, Any, Optional

class CollisionType:
    """
    Docstring for class 'CollisionType'.
    
    vibelint/namespace.py
    """
    HARD = "hard"
    SOFT = "soft"


class NamespaceCollision:
    """
    Represents a collision between two same-named entities.

    src/vibelint/namespace.py
    """
    def __init__(
        self,
        name: str,
        path1: Path,
        path2: Path,
        collision_type: str = CollisionType.HARD
    ) -> None:
        """
        Docstring for method 'NamespaceCollision.__init__'.
        
        vibelint/namespace.py
        """
        self.name = name
        self.path1 = path1
        self.path2 = path2
        self.collision_type = collision_type


def detect_namespace_collisions(
    paths: List[Path],
    config: Dict[str, Any],
    include_vcs_hooks: bool = False
) -> List[NamespaceCollision]:
    """
    Detect "hard collisions" e.g. a member vs. a submodule with the same name, repeated exports.

    src/vibelint/namespace.py
    """
    # We'll build a naive "module tree" to find collisions.
    root_node = _build_namespace_tree(paths, config, include_vcs_hooks)
    collisions = root_node.get_collisions()
    # Possibly also do extra checks for __init__.py
    # ...
    return collisions


def detect_soft_member_collisions(
    paths: List[Path],
    config: Dict[str, Any],
    use_inheritance_check: bool = True,
    include_vcs_hooks: bool = False
) -> List[NamespaceCollision]:
    """
    Soft collisions: same name repeated in different modules, not obviously related.

    src/vibelint/namespace.py
    """
    # We'll gather all top-level names -> [ (file_path, is_class?), ... ]
    root_node = _build_namespace_tree(paths, config, include_vcs_hooks)
    all_members = {}
    root_node.collect_all_members(all_members)

    # If a name appears in multiple distinct files, that's a potential collision
    collisions: List[NamespaceCollision] = []
    for name, occurrences in all_members.items():
        if len(occurrences) > 1:
            # naive approach: pairwise
            # skip if same file
            seen_pairs = set()
            for i in range(len(occurrences)):
                for j in range(i+1, len(occurrences)):
                    p1 = occurrences[i]
                    p2 = occurrences[j]
                    if p1 != p2:
                        pair = tuple(sorted([str(p1), str(p2)]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            collisions.append(NamespaceCollision(name, p1, p2, CollisionType.SOFT))
    return collisions


def get_namespace_collisions_str(
    paths: List[Path],
    config: Dict[str, Any],
    console=None,
    include_vcs_hooks: bool = False
) -> str:
    """
    Return a string representation of collisions for quick debugging.

    src/vibelint/namespace.py
    """
    from io import StringIO
    buf = StringIO()
    collisions = detect_namespace_collisions(paths, config, include_vcs_hooks)
    if collisions:
        buf.write("Hard Collisions:\n")
        for c in collisions:
            buf.write(f"- {c.name}: {c.path1} vs {c.path2}\n")
    return buf.getvalue()


class NamespaceNode:
    """
    A node in the "module" hierarchy (like package/subpackage, or file-level).
    Holds child nodes and top-level members (functions/classes).
    
    src/vibelint/namespace.py
    """
    def __init__(self, name: str, path: Optional[Path] = None) -> None:
        """
        Docstring for method 'NamespaceNode.__init__'.
        
        vibelint/namespace.py
        """
        self.name = name
        self.path = path
        self.children: Dict[str, "NamespaceNode"] = {}
        self.members: Dict[str, Path] = {}

    def add_member(self, name: str, path: Path):
        """
        Docstring for method 'NamespaceNode.add_member'.
        
        vibelint/namespace.py
        """
        self.members[name] = path

    def add_child(self, name: str, path: Path) -> "NamespaceNode":
        """
        Docstring for method 'NamespaceNode.add_child'.
        
        vibelint/namespace.py
        """
        if name not in self.children:
            self.children[name] = NamespaceNode(name, path)
        return self.children[name]

    def get_collisions(self) -> List[NamespaceCollision]:
        """
        Docstring for method 'NamespaceNode.get_collisions'.
        
        vibelint/namespace.py
        """
        collisions: List[NamespaceCollision] = []
        # check if a child node name is also in members
        for mname, mpath in self.members.items():
            if mname in self.children:
                cnode = self.children[mname]
                if cnode.path:
                    collisions.append(NamespaceCollision(mname, mpath, cnode.path, CollisionType.HARD))
        # Recurse
        for cnode in self.children.values():
            collisions.extend(cnode.get_collisions())
        return collisions

    def collect_all_members(self, all_dict: Dict[str, List[Path]]):
        """
        Docstring for method 'NamespaceNode.collect_all_members'.
        
        vibelint/namespace.py
        """
        for mname, mpath in self.members.items():
            all_dict.setdefault(mname, []).append(mpath)
        for cnode in self.children.values():
            cnode.collect_all_members(all_dict)


def _extract_module_members(file_path: Path) -> List[str]:
    """
    Extract top-level function/class/assignment names.

    src/vibelint/namespace.py
    """
    try:
        txt = file_path.read_text(encoding="utf-8")
        tree = ast.parse(txt)
    except Exception:
        return []
    members = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            members.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    members.append(t.id)
    return members


def _build_namespace_tree(paths: List[Path], config: Dict[str, Any], include_vcs_hooks: bool) -> NamespaceNode:
    """
    Docstring for function '_build_namespace_tree'.
    
    vibelint/namespace.py
    """
    root = NamespaceNode("root")
    includes = config.get("include_globs", ["**/*.py"])
    excludes = config.get("exclude_globs", [])

    python_files: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            python_files.append(p)
        elif p.is_dir():
            for pat in includes:
                for f in p.glob(pat):
                    if not f.is_file() or f.suffix != ".py":
                        continue
                    if not include_vcs_hooks and any(
                        part.startswith(".") and part in {".git", ".hg", ".svn"}
                        for part in f.parts
                    ):
                        continue
                    if any(fnmatch.fnmatch(str(f), str(p / e)) for e in excludes):
                        continue
                    python_files.append(f)

    # build a naive tree
    if not python_files:
        return root

    files_str = [str(x) for x in python_files]
    common_prefix = os.path.commonpath(files_str)

    for f in python_files:
        relp = str(f).replace(common_prefix, "").lstrip(os.sep)
        parts = relp.split(os.sep)
        file_name = parts[-1]
        current = root
        for i, part in enumerate(parts[:-1]):
            if part not in current.children:
                current.children[part] = NamespaceNode(part)
            current = current.children[part]

        # add a child for the .py
        mod_name = file_name[:-3]  # remove .py
        if mod_name not in current.children:
            current.children[mod_name] = NamespaceNode(mod_name, f)
        node = current.children[mod_name]

        # extract top-level members
        mm = _extract_module_members(f)
        for m in mm:
            node.add_member(m, f)

    return root
