import pytest
from click.testing import CliRunner
from pathlib import Path
import shutil
import sys
import os
import io
import re


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


from vibelint.cli import cli
from vibelint import __version__


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    """Provides a Click CliRunner instance."""
    return CliRunner()


@pytest.fixture
def setup_test_project(tmp_path, request):
    """
    Copies a fixture directory into a temporary directory and changes
    the current working directory to the *actual project root* within
    that temporary directory for the duration of the test.

    Yields the Path to the temporary project root.
    """
    fixture_name = request.param
    source_fixture_path = FIXTURES_DIR / fixture_name
    if not source_fixture_path.is_dir():
        raise ValueError(f"Fixture directory not found: {source_fixture_path}")

    project_dirs = [d for d in source_fixture_path.iterdir() if d.is_dir()]
    if len(project_dirs) != 1:
        if (source_fixture_path / "pyproject.toml").exists():
            project_dir_name = None
            target_project_root = tmp_path / fixture_name
            shutil.copytree(
                source_fixture_path, target_project_root, dirs_exist_ok=True
            )
        else:
            raise ValueError(
                f"Fixture '{fixture_name}' must contain exactly one project subdirectory "
                f"or have a pyproject.toml in its root."
            )
    else:
        project_dir_name = project_dirs[0].name
        target_fixture_path = tmp_path / fixture_name
        shutil.copytree(source_fixture_path, target_fixture_path, dirs_exist_ok=True)
        target_project_root = target_fixture_path / project_dir_name

    original_cwd = Path.cwd()
    os.chdir(target_project_root)
    print(f"DEBUG: Changed CWD to: {Path.cwd()}")
    try:
        yield target_project_root
    finally:
        os.chdir(original_cwd)
        print(f"DEBUG: Restored CWD to: {Path.cwd()}")


def modify_pyproject(project_path: Path, updates: dict):
    """Modifies the [tool.vibelint] section of pyproject.toml."""
    if tomllib is None:
        pytest.fail("TOML reading library (tomllib/tomli) could not be imported.")
    if tomli_w is None:
        pytest.fail(
            "TOML writing library (tomli_w) could not be imported. Is it installed?"
        )

    pyproject_file = project_path / "pyproject.toml"
    print(f"DEBUG: Attempting to modify pyproject.toml at: {pyproject_file}")
    if not pyproject_file.is_file():
        print(f"DEBUG: Contents of {project_path}: {list(project_path.iterdir())}")
        raise FileNotFoundError(f"pyproject.toml not found in {project_path}")

    with open(pyproject_file, "rb") as f:
        data = tomllib.load(f)

    if "tool" not in data:
        data["tool"] = {}
    if "vibelint" not in data["tool"]:
        data["tool"]["vibelint"] = {}
    data["tool"]["vibelint"].update(updates)

    with open(pyproject_file, "wb") as f:
        tomli_w.dump(data, f)
    print(f"DEBUG: Successfully modified {pyproject_file}")


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"], prog_name="vibelint")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Usage: vibelint [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "check" in result.output
    assert "namespace" in result.output
    assert "snapshot" in result.output


def test_cli_no_project_root(runner, tmp_path):
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["check"], prog_name="vibelint")
        assert result.exit_code == 1
        assert "Error: Could not find project root." in result.output
        assert "a 'pyproject.toml' file or a '.git' directory" in result.output
    finally:
        os.chdir(original_cwd)


def assert_in_output(substring: str, full_output: str, msg: str = ""):
    """Asserts substring is in full_output, handling potential formatting chars."""
    if substring in full_output:
        return

    escaped_substring = re.escape(substring)

    pattern_basic = rf"\x1b\[.*?m{escaped_substring}\x1b\[.*?m|{escaped_substring}"
    if re.search(pattern_basic, full_output):
        return

    cleaned_output = re.sub(r"\x1b\[.*?m", "", full_output)
    cleaned_output = cleaned_output.replace("\r", "")

    cleaned_output = (
        cleaned_output.replace("│", "|")
        .replace("─", "-")
        .replace("┌", "+")
        .replace("┐", "+")
        .replace("└", "+")
        .replace("┘", "+")
        .replace("├", "|")
        .replace("┤", "|")
        .replace("┬", "+")
        .replace("┴", "+")
    )
    cleaned_output = re.sub(r"\s+", " ", cleaned_output)

    cleaned_substring = (
        substring.replace("│", "|")
        .replace("─", "-")
        .replace("┌", "+")
        .replace("┐", "+")
        .replace("└", "+")
        .replace("┘", "+")
        .replace("├", "|")
        .replace("┤", "|")
        .replace("┬", "+")
        .replace("┴", "+")
    )
    cleaned_substring = re.sub(r"\s+", " ", cleaned_substring).strip()

    assert (
        cleaned_substring in cleaned_output
    ), f"{msg} Substring '{substring}' not found in cleaned output:\n{cleaned_output}\nOriginal output:\n{full_output}"


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_success(runner, setup_test_project):
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files OK", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 0 ", result.output)
    assert_in_output("Files with Warnings only", result.output)
    assert "Check finished successfully" in result.output
    assert "Namespace Collision Results Summary" not in result.output


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_errors_missing_all(runner, setup_test_project):
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files OK", result.output)
    assert_in_output(" 0 ", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 2 ", result.output)
    assert "[VBL301] __all__ definition not found" in result.output
    assert "[VBL101] Missing docstring for function 'func_one'" in result.output
    assert "[VBL101] Missing docstring for function 'something'" in result.output
    assert (
        "[VBL102] Docstring for module 'module' missing/incorrect path reference"
        in result.output
    )
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["fix_missing_all"], indirect=True)
def test_check_ignore_codes(runner, setup_test_project):
    modify_pyproject(setup_test_project, {"ignore": ["VBL301"]})
    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("vibelint Results Summary", result.output)
    assert_in_output("Files Scanned", result.output)
    assert_in_output(" 2 ", result.output)
    assert_in_output("Files with Errors", result.output)
    assert_in_output(" 2 ", result.output)
    assert "[VBL301]" not in result.output
    assert "[VBL101] Missing docstring for function 'func_one'" in result.output
    assert "[VBL101] Missing docstring for function 'something'" in result.output
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_output_report(runner, setup_test_project):
    """Test `vibelint check -o report.md`."""
    report_file = setup_test_project / "vibelint_report.md"
    assert not report_file.exists()

    result = runner.invoke(
        cli, ["check", "-o", "vibelint_report.md"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert report_file.exists()
    assert "Report generated at" in result.output
    assert str(report_file.resolve()) in result.output.replace("\n", "")

    report_content = report_file.read_text()
    assert "# vibelint Report" in report_content
    assert "## Summary" in report_content
    assert "## Linting Results" in report_content
    assert "*No linting issues found.*" in report_content
    assert "## Namespace Structure" in report_content

    assert "myproject" in report_content
    assert "src" in report_content
    assert "mypkg" in report_content
    assert "module" in report_content
    assert "hello (member)" in report_content
    assert "## Namespace Collisions" in report_content
    assert "*No hard collisions detected.*" in report_content
    assert "## File Contents" in report_content
    assert "### src/mypkg/__init__.py" in report_content
    assert "### src/mypkg/module.py" in report_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_check_with_hard_collision(runner, setup_test_project):
    """Create and test a hard collision: object in parent __init__ vs child dir."""
    src_dir = setup_test_project / "src"
    src_init_file = src_dir / "__init__.py"
    mypkg_dir = src_dir / "mypkg"

    print(f"DEBUG: Creating file: {src_init_file}")
    src_init_file.touch()
    assert src_init_file.is_file(), f"{src_init_file} not created!"

    print(f"DEBUG: Modifying file: {src_init_file} to add collision")
    src_init_file.write_text("mypkg = 123 # This clashes with the mypkg directory\n")
    print(f"DEBUG: Successfully modified {src_init_file}")

    result = runner.invoke(cli, ["check"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1 (hard collision), got {result.exit_code}. Output:\n{result.output}"
    assert_in_output("Namespace Collision Results Summary", result.output)
    assert_in_output("Hard Collisions", result.output)

    collision_summary_match = re.search(r"Hard Collisions\s*│\s*1\s*│", result.output)
    assert (
        collision_summary_match is not None
    ), "Could not find 'Hard Collisions | 1 |' row in summary table"

    assert_in_output("Hard Collisions:", result.output)
    assert "- 'mypkg': Conflicting definitions/imports" in result.output
    assert str(src_init_file.relative_to(setup_test_project)) in result.output
    assert str(mypkg_dir.relative_to(setup_test_project)) in result.output
    assert "Check finished with errors (exit code 1)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_basic(runner, setup_test_project):
    result = runner.invoke(cli, ["namespace"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert "Namespace Structure:" in result.output
    assert "myproject" in result.output
    assert "└── src" in result.output
    assert "└── mypkg" in result.output
    assert "└── module" in result.output
    assert "hello (member)" in result.output


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_output_file(runner, setup_test_project):
    tree_file = setup_test_project / "namespace_tree.txt"
    assert not tree_file.exists()

    result = runner.invoke(
        cli, ["namespace", "-o", "namespace_tree.txt"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")

    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert tree_file.exists()
    assert "Namespace tree saved to" in result.output
    assert str(tree_file.resolve()) in result.output.replace("\n", "")

    tree_content = tree_file.read_text()
    assert "myproject" in tree_content
    assert "└── src" in tree_content
    assert "└── mypkg" in tree_content
    assert "hello (member)" in tree_content


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_namespace_intra_file_collision(runner, setup_test_project):
    module_file = setup_test_project / "src" / "mypkg" / "module.py"
    print(f"DEBUG: Modifying file: {module_file}")
    assert module_file.is_file(), f"{module_file} not found!"
    content = module_file.read_text()
    content += "\nhello = 123 # Duplicate definition\n"
    module_file.write_text(content)
    print(f"DEBUG: Successfully modified {module_file}")

    result = runner.invoke(cli, ["namespace"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert "Intra-file Collisions Found:" in result.output
    assert (
        "- 'hello': Duplicate definition/import in src/mypkg/module.py" in result.output
    )
    assert "Namespace Structure:" in result.output
    assert "myproject" in result.output


@pytest.fixture
def setup_snapshot_project(tmp_path):
    """Fixture specifically for snapshot tests, ensuring pyproject.toml is included."""
    fixture_name = "check_success"
    source_fixture_path = FIXTURES_DIR / fixture_name
    project_dir_name = "myproject"
    target_fixture_path = tmp_path / fixture_name
    shutil.copytree(source_fixture_path, target_fixture_path, dirs_exist_ok=True)
    target_project_root = target_fixture_path / project_dir_name

    pyproject_file = target_project_root / "pyproject.toml"
    if tomllib and tomli_w and pyproject_file.is_file():
        with open(pyproject_file, "rb") as f:
            data = tomllib.load(f)
        if "tool" not in data:
            data["tool"] = {}
        if "vibelint" not in data["tool"]:
            data["tool"]["vibelint"] = {}
        data["tool"]["vibelint"]["include_globs"] = ["src/**/*.py", "pyproject.toml"]
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(data, f)
    else:
        print("WARN: Could not modify pyproject.toml for snapshot include test.")

    original_cwd = Path.cwd()
    os.chdir(target_project_root)
    print(f"DEBUG: Snapshot Test Changed CWD to: {Path.cwd()}")
    try:
        yield target_project_root
    finally:
        os.chdir(original_cwd)
        print(f"DEBUG: Snapshot Test Restored CWD to: {Path.cwd()}")


def test_snapshot_basic(runner, setup_snapshot_project):
    """Test `vibelint snapshot` default behavior (using modified fixture)."""
    snapshot_file = setup_snapshot_project / "codebase_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(cli, ["snapshot"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()
    assert "Codebase snapshot created at" in result.output
    assert str(snapshot_file.resolve()) in result.output.replace("\n", "")

    snapshot_content = snapshot_file.read_text()

    tree_match = re.search(
        r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL
    )
    assert tree_match, "Filesystem Tree section not found in snapshot"
    tree_block = tree_match.group(1)

    assert "# Snapshot" in snapshot_content
    assert "## Filesystem Tree" in snapshot_content
    assert "myproject/" in tree_block

    assert "pyproject.toml" in tree_block
    assert "src/" in tree_block
    assert "mypkg/" in tree_block
    assert "__init__.py" in tree_block
    assert "module.py" in tree_block

    assert "## File Contents" in snapshot_content
    assert "### File: pyproject.toml" in snapshot_content
    assert "[tool.vibelint]" in snapshot_content
    assert "### File: src/mypkg/__init__.py" in snapshot_content
    assert "### File: src/mypkg/module.py" in snapshot_content


def test_snapshot_output_file(runner, setup_snapshot_project):
    snapshot_file = setup_snapshot_project / "custom_snapshot.md"
    assert not snapshot_file.exists()

    result = runner.invoke(
        cli, ["snapshot", "-o", "custom_snapshot.md"], prog_name="vibelint"
    )
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()
    assert "Codebase snapshot created at" in result.output
    assert str(snapshot_file.resolve()) in result.output.replace("\n", "")


@pytest.mark.parametrize("setup_test_project", ["check_success"], indirect=True)
def test_snapshot_exclude(runner, setup_test_project):
    """Test snapshot respects exclude_globs from config ('check_success' fixture)."""
    modify_pyproject(setup_test_project, {"exclude_globs": ["src/mypkg/module.py"]})

    snapshot_file = setup_test_project / "codebase_snapshot.md"
    result = runner.invoke(cli, ["snapshot"], prog_name="vibelint")
    print(f"Output:\n{result.output}")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    assert snapshot_file.exists()

    snapshot_content = snapshot_file.read_text()
    tree_match = re.search(
        r"## Filesystem Tree\s*```\s*(.*?)\s*```", snapshot_content, re.DOTALL
    )
    assert tree_match, "Filesystem Tree section not found in snapshot"
    tree_block = tree_match.group(1)

    assert "module.py" not in tree_block
    assert "### File: src/mypkg/module.py" not in snapshot_content

    assert "__init__.py" in tree_block
    assert "### File: src/mypkg/__init__.py" in snapshot_content

    assert "pyproject.toml" not in tree_block
    assert "### File: pyproject.toml" not in snapshot_content


def test_snapshot_exclude_output_file(runner, setup_snapshot_project):
    """Test snapshot doesn't include its own output file (using modified fixture)."""
    snapshot_file = setup_snapshot_project / "mysnapshot.md"

    result1 = runner.invoke(
        cli, ["snapshot", "-o", "mysnapshot.md"], prog_name="vibelint"
    )
    assert (
        result1.exit_code == 0
    ), f"Snapshot creation failed (1st run). Output:\n{result1.output}"
    assert snapshot_file.exists()

    result2 = runner.invoke(
        cli, ["snapshot", "-o", "mysnapshot.md"], prog_name="vibelint"
    )
    assert (
        result2.exit_code == 0
    ), f"Snapshot creation failed (2nd run). Output:\n{result2.output}"

    snapshot_content = snapshot_file.read_text()
    assert "mysnapshot.md" not in snapshot_content
    assert "### File: pyproject.toml" in snapshot_content
