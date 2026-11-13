"""The conftest guard that fails a test leaving the cwd moved or repo-root files gone.

A leaked ``os.chdir`` broke every later relative-path test in the same worker
(``FileNotFoundError: 'spacr/qt/widgets/fast_plots.py'``), and a cleanup run in
the wrong cwd can delete ``.coveragerc``. ``tests/conftest.py`` holds the two
hooks that notice; these tests run the real hooks in a throwaway pytest
project and check what they report.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

CONFTEST = Path(__file__).with_name("conftest.py")


def _guard_source() -> str:
    """The guard's own source lines, cut out of the real conftest."""
    text = CONFTEST.read_text(encoding="utf-8")
    tree = ast.parse(text)
    wanted = {"_REPO_ROOT_FILES", "_CWD_BEFORE"}
    names = {
        "_remember_the_working_directory",
        "_no_test_leaves_the_working_directory_or_repo_root_files_changed",
    }
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            pieces.append(ast.get_source_segment(text, node))
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            pieces.append("\n".join(text.splitlines()[start - 1:node.end_lineno]))
    assert len(pieces) == 4, "the guard is no longer where this test expects it"
    hooks = '''
def pytest_runtest_setup(item):
    _remember_the_working_directory(item)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    outcome = yield
    _no_test_leaves_the_working_directory_or_repo_root_files_changed(item)
    return outcome
'''
    return "\n\n".join(pieces) + "\n\n" + hooks


def _run(tmp_path: Path, body: str, env_extra=None):
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / ".coveragerc").write_text("[run]\n", encoding="utf-8")
    (project / "tests" / "conftest.py").write_text(
        "import os\nfrom pathlib import Path\nimport pytest\n\n" + _guard_source() + "\n",
        encoding="utf-8")
    (project / "tests" / "test_probe.py").write_text(
        "import os\nfrom pathlib import Path\n" + textwrap.dedent(body), encoding="utf-8")
    import os
    env = {k: v for k, v in os.environ.items() if k != "SPACR_CWD_GUARD_LOG"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "no:randomly", "-p", "no:xdist", "-c", "/dev/null",
         "--rootdir", str(project), str(project / "tests")],
        cwd=project, env=env, capture_output=True, text=True, timeout=120)


def test_a_test_that_changes_directory_and_never_returns_fails(tmp_path):
    result = _run(tmp_path, """
        def test_leaks(tmp_path):
            os.chdir(tmp_path)
    """)
    assert result.returncode != 0, result.stdout
    assert "working directory changed" in result.stdout + result.stderr


def test_a_test_that_deletes_a_repo_root_file_fails(tmp_path):
    result = _run(tmp_path, """
        def test_deletes():
            (Path(__file__).resolve().parent.parent / ".coveragerc").unlink()
    """)
    assert result.returncode != 0, result.stdout
    assert ".coveragerc" in result.stdout + result.stderr


def test_the_cwd_is_put_back_so_one_leak_does_not_cascade(tmp_path):
    result = _run(tmp_path, """
        START = os.getcwd()

        def test_leaks(tmp_path):
            os.chdir(tmp_path)

        def test_next_one_starts_where_the_session_started():
            assert os.getcwd() == START
    """)
    assert "2 passed" in result.stdout
    assert "1 error" in result.stdout


def test_monkeypatch_chdir_is_not_a_leak(tmp_path):
    result = _run(tmp_path, """
        def test_fine(monkeypatch, tmp_path):
            monkeypatch.chdir(tmp_path)
    """)
    assert result.returncode == 0, result.stdout + result.stderr


def test_survey_mode_writes_the_offender_and_does_not_fail(tmp_path):
    log = tmp_path / "offenders.tsv"
    result = _run(tmp_path, """
        def test_leaks(tmp_path):
            os.chdir(tmp_path)
    """, {"SPACR_CWD_GUARD_LOG": str(log)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_probe.py::test_leaks" in log.read_text(encoding="utf-8")


def test_the_real_conftest_calls_the_guard_from_its_hooks():
    """The guard only works if the real hooks call it; the probe above cannot see that."""
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    called = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "pytest_runtest_setup", "pytest_runtest_teardown"):
            called[node.name] = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_remember_the_working_directory" in called["pytest_runtest_setup"]
    assert ("_no_test_leaves_the_working_directory_or_repo_root_files_changed"
            in called["pytest_runtest_teardown"])
