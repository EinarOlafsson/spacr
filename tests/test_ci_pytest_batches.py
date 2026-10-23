"""Contracts for the memory-bounded CI pytest runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_pytest_batches as runner


def test_qt_precollection_preserves_execution_failures_and_later_batches(
    tmp_path, monkeypatch, capsys,
):
    for index in range(4):
        (tmp_path / f"test_{index}.py").write_text("", encoding="utf-8")
    calls = []
    statuses = iter((5, 0, 1, 2, runner.BATCH_TIMED_OUT))
    monkeypatch.setenv("SPACR_PYTEST_FILE_SHARD_INDEX", "1")
    monkeypatch.setenv("SPACR_PYTEST_FILE_SHARD_COUNT", "3")
    monkeypatch.setattr(runner, "_timeout_plugin_available", lambda: True)

    def run(command, *, timeout):
        import os
        assert os.environ["SPACR_PYTEST_FILE_SHARD_INDEX"] == "1"
        assert os.environ["SPACR_PYTEST_FILE_SHARD_COUNT"] == "3"
        assert timeout == 2700
        calls.append(command)
        return SimpleNamespace(returncode=next(statuses))

    monkeypatch.setattr(runner, "_run_bounded", run)
    assert runner.main([
        str(tmp_path), "--marker", "qt and not slow", "--batch-size", "1",
        "--workers", "2", "--precollect-nonqt-batches",
        "--batch-timeout", "2700", "--faulthandler-timeout", "900",
        "--per-test-timeout", "1200",
    ]) == 1
    assert len(calls) == 5
    collected = [command for command in calls if "--collect-only" in command]
    assert len(collected) == 4
    assert all(command[command.index("-n") + 1] == "0" for command in collected)
    assert all("faulthandler_timeout=900" in command for command in calls)
    [executed] = [command for command in calls if "--collect-only" not in command]
    assert executed[3:executed.index("-m", 3)] == [str(tmp_path / "test_1.py")]
    assert executed[executed.index("-n") + 1] == "2"
    assert executed[executed.index("--timeout") + 1] == "1200"
    assert "--max-worker-restart=0" in executed
    assert "3 of 4 batches failed" in capsys.readouterr().out


@pytest.mark.parametrize("marker", ["not qt", "qt or slow", "qt and not qt", "slow"])
def test_qt_precollection_refuses_nonconjunctive_selection(tmp_path, marker):
    with pytest.raises(ValueError, match="positive Qt conjunction"):
        runner.main([str(tmp_path), "--marker", marker,
                     "--precollect-nonqt-batches"])


@pytest.mark.parametrize("collection_error", [False, True])
def test_real_qt_precollection_keeps_explicit_and_automatic_tests(
    tmp_path, collection_error,
):
    import json
    import os
    import subprocess
    import sys

    tests = tmp_path / "tests"
    (tests / "qt").mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    qt: Qt tests\n    slow: slow tests\n")
    (tests / "conftest.py").write_text('''import json
from pathlib import Path
import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "qt" in item.path.parts:
            item.add_marker(pytest.mark.qt)
    with Path("collection.jsonl").open("a") as stream:
        stream.write(json.dumps({"worker": getattr(config, "workerinput", {}).get("workerid"), "n": config.getoption("numprocesses"), "files": sorted({item.path.name for item in items})}) + "\\n")

def pytest_runtest_logreport(report):
    if report.when == "call":
        with Path("executed.txt").open("a") as stream:
            stream.write(report.nodeid + "\\n")
''')
    (tests / "test_empty.py").write_text("def test_not_qt(): pass\n")
    (tests / "test_explicit.py").write_text(
        "import pytest\n@pytest.mark.qt\ndef test_selected(): pass\n"
        "@pytest.mark.qt\n@pytest.mark.slow\ndef test_excluded(): pass\n")
    (tests / "qt/test_auto.py").write_text("def test_automatic(): pass\n")
    if collection_error:
        (tests / "test_0_fault.py").write_text(
            "raise RuntimeError('collection remains a failure')\n")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTEST_ADDOPTS="-p xdist.plugin -n 2")
    script = Path(runner.__file__).resolve()
    result = subprocess.run([
        sys.executable, str(script), "tests", "--marker", "qt and not slow",
        "--batch-size", "1", "--workers", "2",
        "--precollect-nonqt-batches", "--batch-timeout", "30",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == (2 if collection_error else 0), result.stdout + result.stderr
    executed = set((tmp_path / "executed.txt").read_text().splitlines())
    assert executed == {"tests/qt/test_auto.py::test_automatic",
                        "tests/test_explicit.py::test_selected"}
    records = [json.loads(line) for line in
               (tmp_path / "collection.jsonl").read_text().splitlines()]
    serial = [record for record in records if record["worker"] is None]
    expected = {("test_empty.py",), ("test_explicit.py",)}
    if collection_error:
        expected.add(())
        assert "1 of 4 batches failed" in result.stdout
    assert {tuple(record["files"]) for record in serial} == expected
    assert all(record["n"] == 0 for record in serial)
    assert "collection selected no tests; no worker pool started" in result.stdout


def test_test_files_are_recursive_sorted_and_deduplicated(tmp_path):
    first = tmp_path / "test_a.py"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "test_b.py"
    ignored = nested / "helper.py"
    for path in (first, second, ignored):
        path.write_text("", encoding="utf-8")

    assert runner._test_files([str(tmp_path), str(first)]) == sorted([
        str(first), str(second),
    ])


def test_main_recycles_workers_and_accepts_an_empty_marker_batch(
    tmp_path, monkeypatch,
):
    for index in range(3):
        (tmp_path / f"test_{index}.py").write_text("", encoding="utf-8")

    commands = []
    statuses = iter((0, runner.NO_TESTS_COLLECTED))

    def run(command, check):
        commands.append(command)
        assert check is False
        return SimpleNamespace(returncode=next(statuses))

    monkeypatch.setattr(runner.subprocess, "run", run)

    assert runner.main([
        str(tmp_path), "--marker", "not slow",
        "--batch-size", "2", "--workers", "2",
    ]) == 0
    assert [len(command[3:command.index("-m", 3)]) for command in commands] == [
        2, 1,
    ]
    assert all(command[-6:] == [
        "-n", "2", "--dist", "loadfile", "-v", "--tb=short",
    ] for command in commands)


def test_qt_only_skip_preserves_global_batches_and_every_eligible_command(
    tmp_path, monkeypatch, capsys,
):
    tests = tmp_path / "tests"
    qt = tests / "qt"
    qt.mkdir(parents=True)
    for name in ("a", "b", "c"):
        (qt / f"test_{name}.py").write_text("def test_x(): pass\n", encoding="utf-8")
    for name in ("d", "e"):
        (tests / f"test_{name}.py").write_text("def test_x(): pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "_timeout_plugin_available", lambda: True)
    calls = []

    def run(command, check):
        assert check is False
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    args = [str(tests), "--marker", "not qt and not slow",
            "--batch-size", "2", "--workers", "2",
            "--per-test-timeout", "300"]
    assert runner.main(args) == 0
    original = list(calls)
    calls.clear()
    assert runner.main([*args, "--skip-qt-only-batches"]) == 0
    assert calls == original[1:]
    assert [command[3:command.index("-m", 3)] for command in calls] == [
        (str(qt / "test_c.py"), str(tests / "test_d.py")),
        (str(tests / "test_e.py"),),
    ]
    assert "pytest batch 1/3: 2 files; all automatically qt-marked" in (
        capsys.readouterr().out)


@pytest.mark.parametrize("marker", ["not slow", "qt or not slow", "not qt or slow"])
def test_qt_only_skip_refuses_a_marker_that_can_select_qt(
    tmp_path, monkeypatch, marker,
):
    (tmp_path / "test_one.py").write_text("def test_x(): pass\n", encoding="utf-8")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_kw: pytest.fail(
        "a rejected marker must not start pytest"))
    with pytest.raises(ValueError, match="requires a conjunction"):
        runner.main([str(tmp_path), "--marker", marker,
                     "--skip-qt-only-batches"])


def test_fast_and_minimum_opt_in_only_for_excluded_qt_batches():
    from tests.conftest import _automatic_ci_markers

    assert "qt" in _automatic_ci_markers(Path("tests/qt/test_example.py"))
    workflow = (Path(__file__).resolve().parents[1]
                / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert workflow.count("--skip-qt-only-batches") == 2


def test_ignored_paths_are_removed_before_batching_and_every_file_runs_once(
    tmp_path, monkeypatch,
):
    ignored_file = tmp_path / "test_ignored.py"
    ignored_directory = tmp_path / "ignored"
    ignored_directory.mkdir()
    ignored_nested = ignored_directory / "test_nested.py"
    selected = [tmp_path / f"test_selected_{index}.py" for index in range(5)]
    for path in [ignored_file, ignored_nested, *selected]:
        path.write_text("", encoding="utf-8")

    commands = []

    def run(command, check):
        commands.append(command)
        assert check is False
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([
        str(tmp_path), str(selected[0]), "--marker", "qt and not slow",
        "--batch-size", "2", "--workers", "1",
        "--ignore", str(ignored_file), "--ignore", str(ignored_directory),
    ]) == 1

    batches = [command[3:command.index("-m", 3)] for command in commands]
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [path for batch in batches for path in batch] == sorted(
        str(path) for path in selected)
    assert all(command[command.index("-m", 3) + 1] == "qt and not slow"
               for command in commands)


def test_batch_timeouts_reach_every_fresh_pytest_process(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / f"test_{index}.py").write_text("", encoding="utf-8")
    commands = []
    monkeypatch.setattr(runner, "_timeout_plugin_available", lambda: True)

    def run(command, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main([
        str(tmp_path), "--marker", "qt", "--batch-size", "2",
        "--workers", "2", "--per-test-timeout", "1200",
        "--faulthandler-timeout", "900",
    ]) == 0
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--timeout") + 1] == "1200"
        assert command[command.index("--timeout-method") + 1] == "thread"
        assert "--max-worker-restart=0" in command
        assert command[command.index("-o") + 1] == "faulthandler_timeout=900"


def test_main_stops_at_the_first_real_failure(tmp_path, monkeypatch):
    (tmp_path / "test_one.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda _command, check: SimpleNamespace(returncode=2),
    )

    assert runner.main([str(tmp_path), "--marker", "not slow"]) == 2


@pytest.mark.parametrize("option,value", [
    ("--batch-size", "0"),
    ("--workers", "0"),
])
def test_main_rejects_non_positive_limits(tmp_path, option, value):
    (tmp_path / "test_one.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.main([str(tmp_path), "--marker", "not slow", option, value])


def test_every_batch_runs_even_after_one_fails(monkeypatch, tmp_path):
    """A job that stops at the first failing batch reports a PREFIX.

    The batches partition the suite, so returning early means the batches
    after the failure never execute. Measured on one commit: the run
    stopped at batch 19 of 54, thirty-five batches never ran, and the job
    reported "one failure" -- while a file in batch 39 had three real
    failures that had gone unreported for days because no run reached it.
    """
    import subprocess

    from tools import run_pytest_batches as runner

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"test_{name}.py").write_text("def test_x():\n    pass\n", encoding="utf-8")

    ran = []

    class _Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, check=False):
        batch = [c for c in command if c.endswith(".py")]
        ran.append(tuple(sorted(batch)))
        # The second batch fails; the rest must still be attempted.
        return _Result(1 if len(ran) == 2 else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = runner.main([str(tmp_path), "--batch-size", "1", "--marker", "not slow"])

    assert len(ran) == 4, f"only {len(ran)} of 4 batches ran"
    assert status == 1, "the failing status must still be what the job exits with"


def test_the_summary_names_every_failing_batch(monkeypatch, tmp_path, capsys):
    """One line a reader can act on, rather than a count to go hunting for."""
    import subprocess

    from tools import run_pytest_batches as runner

    for name in ("a", "b", "c"):
        (tmp_path / f"test_{name}.py").write_text("def test_x():\n    pass\n", encoding="utf-8")

    seen = []

    class _Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, check=False):
        seen.append(1)
        return _Result(0 if len(seen) == 1 else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.main([str(tmp_path), "--batch-size", "1", "--marker", "not slow"])

    out = capsys.readouterr().out
    assert "2 of 3 batches failed" in out
    assert "batch 2" in out and "batch 3" in out


@pytest.mark.posix_only
def test_a_batch_past_its_ceiling_is_killed_with_its_children(tmp_path):
    """--batch-timeout ends the controller AND anything it started."""
    import sys
    import time

    marker = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    started = time.monotonic()
    result = runner._run_bounded([sys.executable, "-c", script], timeout=3)
    assert result.returncode == runner.BATCH_TIMED_OUT
    assert time.monotonic() - started < 60
    child = int(marker.read_text(encoding="utf-8"))
    import os
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        try:
            if os.waitpid(child, os.WNOHANG) != (0, 0):
                break
        except ChildProcessError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("the batch's child process outlived the batch timeout")


def test_a_batch_inside_its_ceiling_reports_its_own_status():
    import sys

    result = runner._run_bounded([sys.executable, "-c", "raise SystemExit(3)"],
                                 timeout=60)
    assert result.returncode == 3


def test_main_passes_the_batch_timeout_through(tmp_path, monkeypatch):
    (tmp_path / "test_0.py").write_text("", encoding="utf-8")
    seen = []

    def bounded(command, *, env=None, timeout=0):
        seen.append(timeout)
        return SimpleNamespace(returncode=runner.BATCH_TIMED_OUT)

    monkeypatch.setattr(runner, "_run_bounded", bounded)
    status = runner.main([str(tmp_path), "--marker", "", "--workers", "1",
                          "--batch-timeout", "12"])
    assert seen == [12.0]
    assert status == runner.BATCH_TIMED_OUT
