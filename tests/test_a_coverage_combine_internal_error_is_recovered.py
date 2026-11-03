"""A batch whose tests all passed is not failed by a half-written data file.

Run 41646358629, coverage shard 6, batch 1: every test passed, then
pytest-cov's combine at session end read a child process's coverage data
file before that process had written its schema. coverage.py's read of it
escaped as an INTERNALERROR (``no such table: coverage_schema``, then
``attempt to write a readonly database``), pytest exited 3 and the shard
failed with no test failure behind it.

``tools/run_coverage_batches.py`` now takes that exact signature, as the
ledger plugin records it, through the serial re-run that recovers lost
coverage: the batch passes only when every re-run passed and saved its data.
A failed test, a failed re-run or an INTERNALERROR from anything other than
coverage data reading still fails it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_a_crashed_coverage_worker_cannot_hide_lost_coverage import (
    _lines,
    _main,
    _project_with_tests,
    _record,
    _worker,
)
from tests.test_module_coverage_ratchet import (
    RUNNER_SCRIPT,
    _load_coverage_runner,
    ratchet,
)

PLUGIN_FILE = (Path(__file__).resolve().parents[1] / "tools" / "pytest_plugins"
               / "xdist_coverage_ledger.py")

COVERAGE_CHAIN = [
    "sqlite3.OperationalError: attempt to write a readonly database",
    "coverage.exceptions.DataError: Couldn't use data file "
    "'.coverage.shard-00.batch-001.host.pid5717.X': no such table: "
    "coverage_schema",
    "sqlite3.OperationalError: no such table: coverage_schema",
]


def _load_plugin():
    spec = importlib.util.spec_from_file_location("xdist_coverage_ledger", PLUGIN_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger_writer(runner, *, failed=(), testsfailed=0, coverage_data=True,
                   chain=COVERAGE_CHAIN, internal=True):
    def write(env):
        Path(env[runner.LEDGER_ENV]).write_text(json.dumps({
            "schema": runner.LEDGER_SCHEMA,
            "exitstatus": 3,
            "workers": {
                "gw0": _worker(["tests/test_one.py"], returned=True),
                "gw1": _worker(["tests/test_two.py"], returned=True),
            },
            "crashes": [],
            "unreported_files": [],
            "failed": list(failed),
            "testsfailed": testsfailed,
            "internal_error": (
                {"chain": list(chain), "coverage_data": coverage_data}
                if internal else None
            ),
        }), encoding="utf-8")
    return write


def _shard(tmp_path, monkeypatch, write_ledger, recovery_status=0):
    runner = _load_coverage_runner()
    project = _project_with_tests(tmp_path / "project", "test_one", "test_two")
    data_dir = tmp_path / "coverage-data"
    recoveries = []

    def fake_run(command, *, env, check, timeout=None):
        if runner.LEDGER_ENV in env:
            write_ledger(env)
            return subprocess.CompletedProcess(command, 3)
        recoveries.append(command[3])
        return subprocess.CompletedProcess(command, recovery_status)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    status = _main(runner, data_dir)
    return runner, status, recoveries, _record(runner, data_dir)["batches"][0], data_dir


def test_every_test_passed_and_the_combine_failed_is_re_run_and_passes(
        tmp_path, monkeypatch):
    runner = _load_coverage_runner()
    runner, status, recoveries, batch, data_dir = _shard(
        tmp_path, monkeypatch, _ledger_writer(runner))

    assert status == 0
    assert recoveries == ["tests/test_one.py", "tests/test_two.py"]
    assert batch["exit_code"] == 3
    assert batch["status_after_recovery"] == 0
    assert batch["lost"][0]["exit_kind"] == "coverage-internal-error"
    assert "no such table: coverage_schema" in batch["lost"][0]["error"]
    assert batch["unrecovered_files"] == []
    integrity = ratchet.check_shard_integrity(data_dir, 1)
    assert integrity["status"] == "complete", integrity
    assert "re-run serially and their coverage recovered" in integrity["recovered"][0]


def test_a_re_run_that_fails_a_test_fails_the_batch(tmp_path, monkeypatch):
    runner = _load_coverage_runner()
    runner, status, recoveries, batch, _data = _shard(
        tmp_path, monkeypatch, _ledger_writer(runner), recovery_status=1)

    assert status == 1
    assert recoveries == ["tests/test_one.py", "tests/test_two.py"]
    assert batch["status_after_recovery"] == 1


def test_a_re_run_that_cannot_save_its_data_keeps_the_internal_error(
        tmp_path, monkeypatch):
    runner = _load_coverage_runner()
    runner, status, _recoveries, batch, data_dir = _shard(
        tmp_path, monkeypatch, _ledger_writer(runner), recovery_status=3)

    assert status == 3
    assert batch["status_after_recovery"] == 3
    assert ratchet.check_shard_integrity(data_dir, 1)["status"] == "incomplete"


@pytest.mark.parametrize("ledger", [
    {"failed": ["tests/test_one.py::test_one"], "testsfailed": 1},
    {"failed": [], "testsfailed": 1},
    {"coverage_data": False,
     "chain": ["builtins.RuntimeError: a plugin broke at session end"]},
    {"internal": False},
], ids=["a-failed-test", "pytest-counted-a-failure", "not-coverage-data",
        "no-internal-error"])
def test_anything_but_the_exact_signature_keeps_exit_3_and_re_runs_nothing(
        tmp_path, monkeypatch, ledger):
    runner = _load_coverage_runner()
    runner, status, recoveries, batch, _data = _shard(
        tmp_path, monkeypatch, _ledger_writer(runner, **ledger))

    assert status == 3
    assert recoveries == []
    assert "status_after_recovery" not in batch
    assert batch["lost_files"] == []


def _synthetic_half_written_file(path: Path) -> Path:
    """An empty data file the reader cannot initialise: read only."""
    path.write_bytes(b"")
    path.chmod(0o444)
    return path


needs_a_read_only_file = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions that bind the current user",
)


@needs_a_read_only_file
def test_the_plugin_names_an_empty_data_file_read_as_coverage_data(tmp_path):
    coverage = pytest.importorskip("coverage")
    plugin = _load_plugin()
    shell = _synthetic_half_written_file(tmp_path / ".coverage.child.pid1.x")
    with pytest.raises(Exception) as raised:
        coverage.CoverageData(basename=str(shell)).read()
    described = plugin.describe_internal_error(raised.value)
    assert described["coverage_data"] is True, described
    assert any("coverage_schema" in item for item in described["chain"])
    runner = _load_coverage_runner()
    assert "readonly" in runner._unreadable(shell)
    assert plugin.describe_internal_error(RuntimeError("x"))["coverage_data"] is False


TOY_HALF_WRITTEN = '''
import os
from pathlib import Path

from toypkg import alpha


def test_passes_then_a_child_leaves_half_written_data():
    assert alpha.run(True) == 1
    marker = Path(os.environ["TOY_MARKER"])
    if not marker.exists():
        marker.write_text("left once", encoding="utf-8")
        shell = Path(os.environ["COVERAGE_FILE"] + ".child.pid1.halfwritten")
        shell.write_bytes(b"")
        shell.chmod(0o444)
'''

TOY_BETA = '''
import os

from toypkg import beta


def test_beta():
    assert beta.run(True) == 1
    assert os.environ.get("TOY_FAIL") != "1"
'''


def _toy(root):
    package = root / "toypkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("alpha", "beta"):
        (package / f"{name}.py").write_text(
            "def run(flag):\n    if flag:\n        return 1\n    return 0\n",
            encoding="utf-8",
        )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_0_half_written.py").write_text(TOY_HALF_WRITTEN, encoding="utf-8")
    (tests / "test_1_beta.py").write_text(TOY_BETA, encoding="utf-8")
    (root / ".coveragerc").write_text(
        "[run]\nbranch = True\nparallel = True\nrelative_files = True\n"
        "source = toypkg\n",
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return root


def _run_toy(project, data_dir, *, fail=False):
    marker = project / "left.marker"
    marker.unlink(missing_ok=True)
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("COV", "PYTEST_"))
    }
    environment.update({
        "PYTHONPATH": str(project),
        "TOY_MARKER": str(marker),
        "TOY_FAIL": "1" if fail else "0",
        "RUNNER_TEMP": str(data_dir.parent / f"{data_dir.name}-runner-temp"),
        "PYTEST_ADDOPTS": "--max-worker-restart=0 -p no:randomly -p no:cacheprovider",
    })
    return subprocess.run(
        [
            sys.executable, str(RUNNER_SCRIPT), "tests",
            "--marker", "not gui", "--shard-index", "0", "--shard-count", "1",
            "--workers", "2", "--data-dir", str(data_dir),
            "--cov-source", "toypkg",
            "--recovery-attempts", "2", "--recovery-timeout", "120",
        ],
        cwd=project, env=environment, capture_output=True, text=True,
        timeout=300, check=False,
    )


@needs_a_read_only_file
def test_a_real_combine_internal_error_after_passing_tests_is_recovered(tmp_path):
    pytest.importorskip("pytest_cov")
    pytest.importorskip("xdist")
    runner = _load_coverage_runner()
    project = _toy(tmp_path / "toy")

    data_dir = tmp_path / "data"
    result = _run_toy(project, data_dir)
    output = result.stdout + result.stderr
    assert "INTERNALERROR" in output and "coverage_schema" in output, output[-4000:]
    assert result.returncode == 0, output[-4000:]
    batch = _record(runner, data_dir)["batches"][0]
    assert batch["exit_code"] == 3
    assert batch["status_after_recovery"] == 0
    assert sorted(outcome["file"] for outcome in batch["recovered_files"]) == [
        "tests/test_0_half_written.py", "tests/test_1_beta.py",
    ]
    ledger = json.loads((data_dir / batch["ledger"]).read_text(encoding="utf-8"))
    assert ledger["internal_error"]["coverage_data"] is True
    assert ledger["failed"] == [] and ledger["testsfailed"] == 0
    assert ratchet.check_shard_integrity(data_dir, 1)["status"] == "complete"
    assert {2, 3} <= _lines(data_dir, project, "alpha")

    # NEGATIVE CONTROL: the same internal error with a real test failure.
    failing = tmp_path / "failing"
    result = _run_toy(project, failing, fail=True)
    output = result.stdout + result.stderr
    assert "INTERNALERROR" in output, output[-4000:]
    assert result.returncode == 3, output[-4000:]
    batch = _record(runner, failing)["batches"][0]
    assert "status_after_recovery" not in batch
    assert all(reason["exit_kind"] != "coverage-internal-error"
               for reason in batch["lost"])
