# tests/test_cli.py
"""
Baseline tests for the vibelint CLI interface.

tests/test_cli.py
"""
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from click.testing import CliRunner, Result

# Conditional TOML library import
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None

from vibelint import __version__
from vibelint.cli import cli  # Import the main Click group

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# --- Helper Functions ---


def clean_output(output: str) -> str:
    """Removes ANSI escape codes and strips leading/trailing whitespace from each line."""
    cleaned = re.sub(r"\x1b\[.*?m", "", output)  # Remove ANSI codes
    cleaned = re.sub(r"\r\n?", "\n", cleaned)  # Normalize line endings
    # Process line by line to strip, then rejoin. Filter removes empty lines.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]  # Filter empty lines
    return "\n".join(lines)


def assert_output_contains(result: Result, substring: str, msg: str = ""):
    """Asserts substring is in cleaned output."""
    cleaned = clean_output(result.output)
    assert (
        substring in cleaned
    ), f"{msg}\nSubstring '{substring}' not found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_does_not_contain(result: Result, substring: str, msg: str = ""):
    """Asserts substring is NOT in cleaned output."""
    cleaned = clean_output(result.output)
    assert (
        substring not in cleaned
    ), f"{msg}\nSubstring '{substring}' unexpectedly found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_matches(result: Result, pattern: str, msg: str = ""):
    """Asserts regex pattern matches cleaned output (multiline, dotall)."""
    cleaned = clean_output(result.output)
    # Using DOTALL means '.' matches newline characters as well
    # Using MULTILINE ensures ^/$ match line beginnings/ends if needed
    assert re.search(
        pattern, cleaned, re.MULTILINE | re.DOTALL
    ), f"{msg}\nPattern '{pattern}' not found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


def assert_output_does_not_match(result: Result, pattern: str, msg: str = ""):
    """Asserts regex pattern does NOT match cleaned output (multiline, dotall)."""
    cleaned = clean_output(result.output)
    assert not re.search(
        pattern, cleaned, re.MULTILINE | re.DOTALL
    ), f"{msg}\nPattern '{pattern}' unexpectedly found in cleaned output:\n---\n{cleaned}\n---\nOriginal output:\n{result.output}"


# --- Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner()


@pytest.fixture
def setup_test_project(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """
    Copies a fixture project structure (e.g., fixtures/check_success/myproject/*)
    into a temporary directory, changes the CWD to the identified project root
    within that temp structure, ensures pyproject.toml exists there, and yields the path.
    """
    fixture_name = request.param
    source_fixture_path = FIXTURES_DIR / fixture_name
    if not source_fixture_path.is_dir():
        pytest.fail(f"Fixture directory not found: {source_fixture_path}")

    # Target directory in tmp_path named after the fixture
    target_base_dir = tmp_path / fixture_name
    target_base_dir.mkdir(parents=True, exist_ok=True)

    # Copy the *contents* of the source fixture directory
    for item in source_fixture_path.iterdir():
        source_item = source_fixture_path / item.name
        target_item = target_base_dir / item.name
        if source_item.is_dir():
            # Ensure target directory exists before copying into it
            target_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            # Ensure target directory exists before copying file
            target_item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_item, target_item)  # copy2 preserves metadata
    print(f"DEBUG: Copied contents of {source_fixture_path} to {target_base_dir}")

    # --- Determine the actual project root within the copied structure ---
    target_project_root = None
    if (target_base_dir / "pyproject.toml").is_file() or (target_base_dir / ".git").is_dir():
        target_project_root = target_base_dir
        print(f"DEBUG: Project marker found in target base directory: {target_project_root}")
    else:
        potential_project_dirs = [
            d for d in target_base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        found_marker_in_subdir = False
        if potential_project_dirs:
            for potential_dir in potential_project_dirs:
                if (potential_dir / "pyproject.toml").is_file() or (
                    potential_dir / ".git"
                ).is_dir():
                    target_project_root = potential_dir
                    print(f"DEBUG: Project marker found in subdirectory: {target_project_root}")
                    found_marker_in_subdir = True
                    break
            if not found_marker_in_subdir and len(potential_project_dirs) == 1:
                target_project_root = potential_project_dirs[0]
                print(
                    f"DEBUG: Assuming single subdirectory is project root (no marker found): {target_project_root}"
                )
        if target_project_root is None:
            target_project_root = target_base_dir
            print(
                f"DEBUG: WARNING - No definitive project root identified via markers or single subdir heuristic. Defaulting to base: {target_project_root}"
            )

    # --- Ensure pyproject.toml exists in the determined root ---
    pyproject_path_in_root = target_project_root / "pyproject.toml"
    if not pyproject_path_in_root.is_file():
        print(f"DEBUG: Creating dummy pyproject.toml in identified root {target_project_root}")
        pyproject_path_in_root.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the dummy toml is valid
        pyproject_path_in_root.write_text(
            "[tool.vibelint]\ninclude_globs = [] # Dummy\n", encoding="utf-8"
        )
    else:
        print(f"DEBUG: Found existing pyproject.toml in identified root: {pyproject_path_in_root}")
        # Ensure the existing file is valid TOML if modifying later
        try:
            with open(pyproject_path_in_root, "rb") as f:
                content = f.read()
                if content.strip():
                    tomllib.loads(content.decode("utf-8", errors="replace"))
        except Exception as e:
            print(
                f"DEBUG: Warning - Existing pyproject.toml at {pyproject_path_in_root} may be invalid: {e}"
            )
            # Overwrite with a basic valid one if parsing fails? Risky. Let modify handle it.
            pass

    # --- Change CWD to the determined project root and Yield ---
    original_cwd = Path.cwd()
    resolved_root = target_project_root.resolve()
    print(f"DEBUG: Original CWD: {original_cwd}")
    print(f"DEBUG: Changing CWD to identified project root: {resolved_root}")
    os.chdir(resolved_root)
    try:
        yield resolved_root
    finally:
        print(f"DEBUG: Restoring CWD to: {original_cwd}")
        os.chdir(original_cwd)


def modify_pyproject(project_path: Path, updates: Dict[str, Any]):
    """Modifies the [tool.vibelint] section of pyproject.toml."""
    if tomllib is None or tomli_w is None:
        pytest.skip("Skipping test: 'tomli'/'tomli-w' not available for modifying pyproject.toml.")

    pyproject_file = project_path / "pyproject.toml"
    if not pyproject_file.is_file():
        pytest.fail(f"pyproject.toml unexpectedly missing in {project_path} despite fixture logic.")

    try:
        with open(pyproject_file, "rb") as f:
            content = f.read()
            if not content.strip():
                data = {}
                print(f"DEBUG: Reading empty or whitespace-only {pyproject_file}")
            else:
                try:
                    data = tomllib.loads(content.decode("utf-8", errors="replace"))
                    print(f"DEBUG: Successfully loaded TOML from {pyproject_file}")
                except tomllib.TOMLDecodeError as e:
                    pytest.fail(
                        f"Failed to parse non-empty {pyproject_file}: {e}\nContent:\n>>>\n{content.decode('utf-8', errors='replace')}\n<<<"
                    )
    except Exception as e:
        pytest.fail(f"Failed to read {pyproject_file}: {e}")

    data.setdefault("tool", {}).setdefault("vibelint", {}).update(updates)

    try:
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(data, f)
        print(f"DEBUG: Successfully modified {pyproject_file} with updates: {updates}")
    except Exception as e:
        pytest.fail(f"Failed to write modified {pyproject_file}: {e}")


# --- Test Cases ---


def test_cli_version(runner: CliRunner):
    """Test the --version flag."""
    result = runner.invoke(cli, ["--version"], prog_name="vibelint", catch_exceptions=False)
    assert result.exit_code == 0
    assert f"vibelint, version {__version__}" in result.output


def test_cli_help(runner: CliRunner):
    """Test the --help flag."""
    result = runner.invoke(cli, ["--help"], prog_name="vibelint", catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "check" in result.output
    assert "namespace" in result.output
    assert "snapshot" in result.output
    assert "Vibe Check" in result.output
    assert "Visualize" in result.output
    assert "snapshot" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_cli_no_command_shows_art_and_help(runner: CliRunner, setup_test_project: Path):
    """Test running vibelint with no command shows ASCII art and help hint."""
    vibe_file = setup_test_project / "VIBECHECKER.txt"
    if not vibe_file.is_file():
        vibe_file.write_text(":-)", encoding="utf-8")

    result = runner.invoke(cli, [], catch_exceptions=False)
    assert result.exit_code == 0
    assert_output_matches(result, r"[^\s]")
    assert_output_contains(result, "Run vibelint --help for available commands.")
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." not in result.output


def test_cli_no_project_root(runner: CliRunner, tmp_path: Path):
    """Test CLI behavior when no project root (pyproject.toml/.git) is found."""
    original_cwd = Path.cwd()
    empty_dir = tmp_path / "empty_test_dir_no_root_v5"  # Use different name
    empty_dir.mkdir()
    os.chdir(empty_dir)
    try:
        result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
        print(f"Output (no project root):\n{result.output}")
        assert (
            result.exit_code == 1
        ), f"Expected exit code 1 when no root found, got {result.exit_code}"
        assert_output_contains(result, "Error: Could not find project root.")
        assert_output_contains(result, "pyproject.toml")
        assert_output_contains(result, ".git")
    finally:
        os.chdir(original_cwd)


# --- 'check' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_success(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check` on a project fixture expected to pass."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    print(f"DEBUG: CWD during test_check_success: {Path.cwd()}")
    print(f"DEBUG: setup_test_project path (yielded root): {setup_test_project}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Vibe Check Summary")
    assert_output_matches(
        result, r"(Immaculate vibes|Vibes confirmed|Vibe on brother)", msg="Missing success message"
    )
    assert_output_does_not_match(result, r"\[VBL\d{3}\]", msg="Unexpected VBL codes found")
    assert_output_does_not_contain(result, "Collision Summary", msg="Unexpected collision summary")


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_failure(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check` on a project fixture expected to fail."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["*.py"]})

    result = runner.invoke(cli, ["--debug", "check"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    print(f"DEBUG: CWD during test_check_failure: {Path.cwd()}")
    print(f"DEBUG: setup_test_project path (yielded root): {setup_test_project}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Vibe Check Summary")
    assert_output_matches(result, r"Vibe Check Failed", msg="Missing failure message")
    assert_output_matches(result, r"\[VBL101\]", msg="Missing VBL101 code")
    assert_output_matches(result, r"\[VBL102\]", msg="Missing VBL102 code")
    assert_output_matches(result, r"\[VBL301\]", msg="Missing VBL301 code")
    assert_output_contains(result, "VBL101] Missing docstring for function 'func_one'")
    assert_output_matches(
        result, r"VBL102\].*\(expected '[^']*?/?another\.py'\)", msg="Missing VBL102 for another.py"
    )
    assert_output_matches(
        result,
        r"VBL102\].*\(expected '[^']*?/?needs_fix\.py'\)",
        msg="Missing VBL102 for needs_fix.py",
    )
    assert_output_matches(
        result,
        r"VBL301] __all__ definition not found in [^']*/?needs_fix\.py",
        msg="Missing VBL301 message for needs_fix.py",
    )
    assert_output_does_not_contain(result, "Collision Summary", msg="Unexpected collision summary")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_output_report(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint check -o report.md` generates a report."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    report_file = setup_test_project / "vibelint_report.md"
    assert not report_file.exists()

    result = runner.invoke(
        cli, ["--debug", "check", "-o", str(report_file)], catch_exceptions=False
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 for check, got {result.exit_code}. Output:\n{result.output}"
    assert report_file.is_file(), f"Report file was not created at {report_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Detailed Vibe Report generated at" in cleaned_output
    ), f"Report text missing. Output:\n{cleaned_output}"
    # Check if the filename exists somewhere in the cleaned output
    assert (
        report_file.name in cleaned_output
    ), f"Report filename missing. Filename: {report_file.name}\nOutput:\n{cleaned_output}"

    report_content = report_file.read_text()
    assert "# vibelint Report" in report_content
    assert "## Linting Results" in report_content
    assert "*No linting issues found.*" in report_content
    assert "## Namespace Structure" in report_content
    assert setup_test_project.name in report_content
    assert "hello (member)" in report_content
    assert "## Namespace Collisions" in report_content
    assert "*No hard collisions detected.*" in report_content
    assert "## File Contents" in report_content
    assert "src/mypkg/module.py" in report_content.replace("\\", "/")


# --- 'namespace' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_basic(runner: CliRunner, setup_test_project: Path):
    """Test basic `vibelint namespace` output."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    result = runner.invoke(cli, ["--debug", "namespace"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_output_matches(result, r"Namespace Structure Visualization")
    assert_output_contains(result, setup_test_project.name)
    assert_output_contains(result, "src")
    assert_output_contains(result, "mypkg")
    assert_output_contains(result, "module")
    assert_output_contains(result, "hello (member)")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_output_file(runner: CliRunner, setup_test_project: Path):
    """Test `vibelint namespace -o tree.txt` saves the tree."""
    # Add default includes if missing from fixture's pyproject.toml
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    tree_file = setup_test_project / "namespace_tree.txt"
    assert not tree_file.exists()

    result = runner.invoke(
        cli, ["--debug", "namespace", "-o", str(tree_file)], catch_exceptions=False
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert tree_file.is_file(), f"Namespace tree file not created at {tree_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Namespace tree saved to" in cleaned_output
    ), f"Namespace save text missing. Output:\n{cleaned_output}"
    assert (
        tree_file.name in cleaned_output
    ), f"Namespace save filename missing. Filename: {tree_file.name}\nOutput:\n{cleaned_output}"

    tree_content = tree_file.read_text()
    assert setup_test_project.name in tree_content
    assert "hello (member)" in tree_content


# --- 'snapshot' Command Tests ---


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_basic(runner: CliRunner, setup_test_project: Path):
    """Test basic `vibelint snapshot` default output."""
    modify_pyproject(setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml"]})

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(cli, ["--debug", "snapshot"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.is_file(), f"Snapshot file not created at {snapshot_file}"

    # Check for the message parts separately due to potential wrapping
    cleaned_output = clean_output(result.output)
    assert (
        "Codebase snapshot created at" in cleaned_output
    ), f"Snapshot creation text missing. Output:\n{cleaned_output}"
    assert (
        snapshot_file.name in cleaned_output
    ), f"Snapshot creation filename missing. Filename: {snapshot_file.name}\nOutput:\n{cleaned_output}"

    snapshot_content = snapshot_file.read_text()
    normalized_content = snapshot_content.replace("\\", "/")  # Normalize paths in content
    assert "# Snapshot" in normalized_content
    assert "## Filesystem Tree" in normalized_content
    # Check for the project root name (should be the directory name vibelint ran in)
    assert setup_test_project.name + "/" in normalized_content
    assert "pyproject.toml" in normalized_content
    assert "src/" in normalized_content
    assert "__init__.py" in normalized_content
    assert "module.py" in normalized_content
    assert "## File Contents" in normalized_content
    assert "### File: pyproject.toml" in normalized_content
    assert "[tool.vibelint]" in normalized_content
    assert "### File: src/mypkg/__init__.py" in normalized_content
    assert "### File: src/mypkg/module.py" in normalized_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude(runner: CliRunner, setup_test_project: Path):
    """Test snapshot respects exclude_globs from config."""
    modify_pyproject(
        setup_test_project,
        {
            "include_globs": ["src/**/*.py", "pyproject.toml"],
            "exclude_globs": ["src/mypkg/module.py"],
        },
    )

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    result = runner.invoke(cli, ["--debug", "snapshot"], catch_exceptions=False)
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.is_file()

    snapshot_content = snapshot_file.read_text()
    tree_match = re.search(r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL)
    assert tree_match, "Filesystem Tree section not found"
    tree_content = tree_match.group(1)
    print(f"DEBUG Tree Content:\n{tree_content}")

    assert "module.py" not in tree_content
    # Simpler string check for presence in tree
    assert "__init__.py" in tree_content, "__init__.py missing from tree"
    assert "pyproject.toml" in tree_content, "pyproject.toml missing from tree"

    normalized_content = snapshot_content.replace("\\", "/")
    assert "## File Contents" in normalized_content
    assert "### File: src/mypkg/module.py" not in normalized_content
    assert "### File: src/mypkg/__init__.py" in normalized_content
    assert "### File: pyproject.toml" in normalized_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude_output_file(runner: CliRunner, setup_test_project: Path):
    """Test snapshot doesn't include its own output file."""
    modify_pyproject(
        setup_test_project, {"include_globs": ["src/**/*.py", "pyproject.toml", "*.md"]}
    )

    snapshot_file = setup_test_project / "my_own_snapshot.md"

    # First run creates the file
    result1 = runner.invoke(
        cli, ["--debug", "snapshot", "-o", str(snapshot_file)], catch_exceptions=False
    )
    assert result1.exit_code == 0
    assert snapshot_file.is_file()
    print(f"Snapshot 1 Output:\n{result1.output}")
    snapshot_content_1 = snapshot_file.read_text()
    assert "my_own_snapshot.md" not in snapshot_content_1

    # Second run should explicitly exclude the existing snapshot file
    result2 = runner.invoke(
        cli, ["--debug", "snapshot", "-o", str(snapshot_file)], catch_exceptions=False
    )
    assert result2.exit_code == 0
    print(f"Snapshot 2 Output:\n{result2.output}")

    snapshot_content_2 = snapshot_file.read_text()
    cleaned_content_tree = ""
    tree_match = re.search(r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content_2, re.DOTALL)
    if tree_match:
        cleaned_content_tree = "\n".join(
            [line.strip() for line in tree_match.group(1).splitlines() if line.strip()]
        )
    print(f"DEBUG Tree Content (Run 2):\n{cleaned_content_tree}")

    cleaned_content_files = ""
    files_match = re.search(r"## File Contents\s*(.*)", snapshot_content_2, re.DOTALL)
    if files_match:
        cleaned_content_files = "\n".join(
            line.strip() for line in files_match.group(1).splitlines() if line.strip()
        )
    print(f"DEBUG File Content (Run 2):\n{cleaned_content_files}")

    # Check exclusion from tree
    assert (
        "my_own_snapshot.md" not in cleaned_content_tree
    ), "Snapshot file included in tree section on second run"
    # Check exclusion from content
    assert (
        "### File: my_own_snapshot.md" not in cleaned_content_files
    ), "Snapshot file content included on second run"

    # Verify other expected files are still present using simpler string checks
    assert "pyproject.toml" in cleaned_content_tree, "pyproject.toml missing from tree (run 2)"
    assert "### File: pyproject.toml" in cleaned_content_files
    assert "__init__.py" in cleaned_content_tree, "__init__.py missing from tree (run 2)"
    assert "### File: src/mypkg/__init__.py" in cleaned_content_files.replace("\\", "/")
    assert "module.py" in cleaned_content_tree, "module.py missing from tree (run 2)"
    assert "### File: src/mypkg/module.py" in cleaned_content_files.replace("\\", "/")
