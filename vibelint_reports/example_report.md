# vibelint Report

*Generated on: 2025-04-11 17:09:03*

**Project(s):** vibelint

**Paths analyzed:** /Users/mithranmohanraj/Documents/vibelint

## Table of Contents

1. [Summary](#summary)
2. [Linting Results](#linting-results)
3. [Namespace Structure](#namespace-structure)
4. [Namespace Collisions](#namespace-collisions)
5. [File Contents](#file-contents)

## Summary

| Metric | Count |
|--------|-------|
| Files analyzed | 12 |
| Files with errors | 10 |
| Files with warnings | 0 |
| Hard namespace collisions | 0 |
| Soft namespace collisions | 4 |

## Linting Results

| File | Errors | Warnings |
|------|--------|----------|
| `setup.py` | Docstring for module 'module' must include file path: /Users/mithranmohanraj/Documents/vibelint/setup.py | None |
| `src/vibelint/__init__.py` | Missing docstring for module 'module'. | None |
| `src/vibelint/cli.py` | Docstring for function 'report' must include file path: vibelint/cli.py | None |
| `src/vibelint/utils.py` | File has shebang #!/usr/bin/env python3 but no '__main__' block. | None |
| `src/vibelint/namespace.py` | Missing docstring for class 'CollisionType'.; Missing docstring for method 'NamespaceCollision.__init__'.; Missing docstring for method 'NamespaceNode.__init__'.; Missing docstring for method 'NamespaceNode.add_member'.; Missing docstring for method 'NamespaceNode.add_child'.; Missing docstring for method 'NamespaceNode.get_collisions'.; Missing docstring for method 'NamespaceNode.collect_all_members'.; Missing docstring for function '_build_namespace_tree'. | None |
| `src/vibelint/report.py` | File has shebang #!/usr/bin/env python3 but no '__main__' block. | None |
| `src/vibelint/lint.py` | Missing docstring for method 'LintResult.__init__'.; Missing docstring for method 'LintResult.has_issues'.; Missing docstring for method 'LintRunner.__init__'.; Missing docstring for method 'LintRunner.run'.; Missing docstring for method 'LintRunner._collect_python_files'.; Missing docstring for method 'LintRunner._confirm_large_directory'.; Missing docstring for method 'LintRunner._process_file'.; Missing docstring for method 'LintRunner._print_summary'. | None |
| `src/vibelint/validators/encoding.py` | Missing docstring for method 'EncodingValidationResult.__init__'. | None |
| `src/vibelint/validators/shebang.py` | Missing docstring for method 'ShebangValidationResult.__init__'. | None |
| `src/vibelint/validators/docstring.py` | Missing docstring for method 'DocstringValidationResult.__init__'.; Missing docstring for method 'DocstringValidationResult.has_issues'. | None |

## Namespace Structure

```
<vibelint.namespace.NamespaceNode object at 0x1014215d0>
```

## Namespace Collisions

### Hard Collisions

*No hard collisions detected.*

### Soft Collisions

These don't break Python but may confuse humans and LLMs:

| Name | Path 1 | Path 2 |
|------|--------|--------|
| `console` | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/cli.py | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/report.py |
| `console` | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/cli.py | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/lint.py |
| `console` | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/report.py | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/lint.py |
| `get_relative_path` | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/utils.py | /Users/mithranmohanraj/Documents/vibelint/src/vibelint/report.py |

## File Contents

Files are ordered by their position in the namespace hierarchy.

