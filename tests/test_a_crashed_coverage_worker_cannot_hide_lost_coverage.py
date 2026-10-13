"""A coverage worker killed by a signal cannot silently take coverage with it.

Items 288 and 43, 2026-09-15. pytest-cov writes a process's coverage data when
its session finishes, so a worker that dies by a signal -- the segfault family
item 43 records -- takes every line it measured with it, including the lines
of tests it had already reported as PASSED. On dispatch 34989909231 two
segfaulted coverage workers became thirteen false ratchet regressions
(pivot_spec.py at 24 uncovered statements, from 100%).

``tools/run_coverage_batches.py`` loads
``tools/pytest_plugins/xdist_coverage_ledger.py``, re-runs the files of every
worker whose data did not come back -- serially, one file per process -- and
writes what it recovered and what it could not for the gate. The real-crash
test segfaults an xdist worker after a test on that worker has PASSED, with a
NEGATIVE CONTROL (recovery off) that shows the lines really are lost, so the
recovery is what restores them.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.test_module_coverage_ratchet import (
    RUNNER_SCRIPT,
    _load_coverage_runner,
    ratchet,
)

#: ``def run(flag)`` is line 1 and runs on import; 2-4 run only when called.
BODY = {2, 3, 4}
CRASHING_TEST = "tests/test_0_crash.py::test_b_segfaults_its_worker_the_first_time"


def _worker(files, *, returned, error=None, down=True, exit_status=None):
    return {
        "files": list(files),
        "down": down,
        "error": error,
        "coverage_returned": returned,
        "exit_status": exit_status,
    }


def _ledger(workers, *, crashes=(), unreported=()):
    return {
        "workers": workers,
        "crashes": list(crashes),
        "unreported_files": list(unreported),
    }


def _write_ledger(runner, env, workers, **extra):
    Path(env[runner.LEDGER_ENV]).write_text(
        json.dumps({
            "schema": runner.LEDGER_SCHEMA,
            "exitstatus": 1,
            **_ledger(workers, **extra),
        }),
        encoding="utf-8",
    )


def _project_with_tests(root, *names):
    tests = root / "tests"
    tests.mkdir(parents=True)
    for name in names:
        (tests / f"{name}.py").write_text(f"def {name}(): pass\n", encoding="utf-8")
    return root


def _main(runner, data_dir, *extra):
    return runner.main([
        "tests", "--marker", "not gui", "--shard-index", "0",
        "--shard-count", "1", "--workers", "2", "--data-dir", str(data_dir),
        *extra,
    ])


def _record(runner, data_dir):
    return json.loads(
        (data_dir / runner.integrity_record_name(0)).read_text(encoding="utf-8")
    )


# -- what counts as lost ----------------------------------------------------


def test_only_a_worker_whose_coverage_came_back_counts_as_measured():
    runner = _load_coverage_runner()
    files = ["tests/test_a.py", "tests/test_b.py"]

    healthy = _ledger({"gw0": _worker(["tests/test_a.py"], returned=True)})
    serial = _ledger({
        "controller": _worker(["tests/test_a.py"], returned=None, down=False),
    })
    assert runner.lost_coverage(healthy, files) == ([], [])
    assert runner.lost_coverage(serial, files) == ([], [])

    crashed = _ledger(
        {
            "gw0": _worker(
                ["tests/test_a.py"], returned=False,
                error="Not properly terminated", exit_status=-signal.SIGSEGV,
            ),
            "gw1": _worker(["tests/test_b.py"], returned=True),
        },
        crashes=[{"worker": "gw0", "nodeid": "tests/test_c.py::test_c"}],
    )
    assert runner.lost_coverage(crashed, files) == (
        ["tests/test_a.py", "tests/test_c.py"],
        [{
            "worker": "gw0",
            "error": "Not properly terminated",
            "exit_status": -signal.SIGSEGV,
            "exit_kind": "segfault",
            "exit": "was killed by SIGSEGV (a segfault: the crash family of "
                    "item 43)",
            "running": ["tests/test_c.py::test_c"],
            "files": ["tests/test_a.py", "tests/test_c.py"],
        }],
    )

    never_down = _ledger({
        "gw2": _worker(["tests/test_d.py"], returned=None, down=False),
    })
    lost, reasons = runner.lost_coverage(never_down, files)
    assert lost == ["tests/test_d.py"]
    assert reasons[0]["error"] == "never went down"

    stopped = _ledger({}, unreported=["tests/test_e.py"])
    assert runner.lost_coverage(stopped, files)[0] == ["tests/test_e.py"]

    lost, reasons = runner.lost_coverage(
        None, list(reversed(files)), -signal.SIGKILL,
    )
    assert lost == files
    assert "no readable ledger" in reasons[0]["error"]
    assert reasons[0]["exit_kind"] == "signal"
    assert "SIGKILL" in reasons[0]["exit"]


@pytest.mark.parametrize(
    ("status", "kind", "words"),
    [
        (-signal.SIGSEGV, "segfault", "the crash family of item 43"),
        (3, "memory-guard", "SPACR_TEST_MEMORY_GB"),
        (-signal.SIGKILL, "signal", "kernel OOM killer"),
        (-signal.SIGABRT, "signal", "SIGABRT"),
        (1, "exit", "exited with status 1"),
        (None, "unknown", "could not be read"),
    ],
)
def test_how_a_process_ended_is_named_by_its_exit_status(status, kind, words):
    """xdist says "Not properly terminated" for all of these."""
    runner = _load_coverage_runner()

    named_kind, named = runner.describe_exit(status, worker=True)

    assert named_kind == kind
    assert words in named


def test_a_memory_guard_exit_and_a_segfault_are_told_apart_in_the_log(
    tmp_path, monkeypatch, capsys,
):
    runner = _load_coverage_runner()
    project = _project_with_tests(tmp_path / "project", "test_one", "test_two")
    data_dir = tmp_path / "coverage-data"

    def fake_run(command, *, env, check, timeout=None):
        if runner.LEDGER_ENV in env:
            _write_ledger(runner, env, {
                "gw0": _worker(
                    ["tests/test_one.py"], returned=False,
                    error="Not properly terminated", exit_status=3,
                ),
                "gw1": _worker(
                    ["tests/test_two.py"], returned=False,
                    error="Not properly terminated",
                    exit_status=-signal.SIGSEGV,
                ),
            })
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    _main(runner, data_dir)

    output = capsys.readouterr().out
    assert (
        "worker gw0 was ended by the test memory guard (exit 3: its RSS "
        "passed SPACR_TEST_MEMORY_GB, default 6 GB) [Not properly terminated]"
    ) in output
    assert (
        "worker gw1 was killed by SIGSEGV (a segfault: the crash family of "
        "item 43) [Not properly terminated]"
    ) in output
    integrity = ratchet.check_shard_integrity(data_dir, 1)
    assert integrity["loss_kinds"] == {"memory-guard": 1, "segfault": 1}
    assert integrity["status"] == "complete"


# -- what the runner does about it (subprocess replaced) --------------------


def test_a_lost_workers_files_are_re_run_one_per_process_into_their_own_data(
    tmp_path, monkeypatch,
):
    runner = _load_coverage_runner()
    project = _project_with_tests(
        tmp_path / "project", "test_one", "test_two", "test_three",
    )
    data_dir = tmp_path / "coverage-data"
    calls = []

    def fake_run(command, *, env, check, timeout=None):
        calls.append((command, dict(env), timeout))
        if runner.LEDGER_ENV in env:
            _write_ledger(
                runner, env,
                {
                    "gw0": _worker(
                        ["tests/test_one.py"], returned=False,
                        error="Not properly terminated",
                    ),
                    "gw1": _worker(["tests/test_three.py"], returned=True),
                },
                crashes=[{"worker": "gw0", "nodeid": "tests/test_two.py::test_two"}],
            )
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    status = _main(runner, data_dir)

    # Recovering the coverage does not hide the crash: the batch still failed.
    assert status == 1
    batch_command, batch_env, _timeout = calls[0]
    assert batch_command[batch_command.index("-p") + 1] == runner.PLUGIN
    assert str(runner.PLUGIN_DIR) in batch_env["PYTHONPATH"].split(os.pathsep)
    recovery = calls[1:]
    assert [command[3] for command, _env, _t in recovery] == [
        "tests/test_one.py", "tests/test_two.py",
    ]
    assert [Path(env["COVERAGE_FILE"]).name for _c, env, _t in recovery] == [
        ".coverage.shard-00.batch-001-recovery-001",
        ".coverage.shard-00.batch-001-recovery-002",
    ]
    for command, env, timeout in recovery:
        assert "-n" not in command
        assert "--cov-append" in command
        assert "--cov=spacr" in command and "--cov-branch" in command
        assert runner.LEDGER_ENV not in env
        assert timeout == 1800.0

    batch = _record(runner, data_dir)["batches"][0]
    assert batch["lost_files"] == ["tests/test_one.py", "tests/test_two.py"]
    assert [outcome["file"] for outcome in batch["recovered_files"]] == (
        batch["lost_files"]
    )
    assert batch["unrecovered_files"] == []
    assert batch["lost"][0]["running"] == ["tests/test_two.py::test_two"]
    assert ratchet.check_shard_integrity(data_dir, 1)["status"] == "complete"


def test_a_file_that_crashes_again_is_unrecovered_and_fails_the_shard(
    tmp_path, monkeypatch, capsys,
):
    runner = _load_coverage_runner()
    project = _project_with_tests(tmp_path / "project", "test_one")
    data_dir = tmp_path / "coverage-data"
    attempts = []

    def fake_run(command, *, env, check, timeout=None):
        if runner.LEDGER_ENV in env:
            # pytest itself said the batch PASSED: the worker died after its
            # last report, which xdist does not count as a failed test.
            _write_ledger(runner, env, {
                "gw0": _worker(
                    ["tests/test_one.py"], returned=False,
                    error="Not properly terminated",
                ),
            })
            return subprocess.CompletedProcess(command, 0)
        attempts.append(command)
        return subprocess.CompletedProcess(command, -signal.SIGSEGV)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    status = _main(runner, data_dir)

    assert status == runner.UNRECOVERED_STATUS == 1
    assert len(attempts) == 2
    batch = _record(runner, data_dir)["batches"][0]
    assert batch["recovered_files"] == []
    [outcome] = batch["unrecovered_files"]
    assert outcome["file"] == "tests/test_one.py"
    assert [attempt["exit_code"] for attempt in outcome["attempts"]] == [
        -signal.SIGSEGV, -signal.SIGSEGV,
    ]
    assert all("SIGSEGV" in attempt["exit"] for attempt in outcome["attempts"])
    assert "::error title=Coverage data lost in shard 0::" in capsys.readouterr().out
    integrity = ratchet.check_shard_integrity(data_dir, 1)
    assert integrity["status"] == "incomplete"
    assert "could not be recovered: tests/test_one.py" in integrity["issues"][0]
    assert "worker gw0 exited with status" not in integrity["issues"][0]
    assert "recovery attempts -- tests/test_one.py: was killed by SIGSEGV" in (
        integrity["issues"][0]
    )


def test_a_batch_that_left_no_ledger_has_every_file_re_run(tmp_path, monkeypatch):
    runner = _load_coverage_runner()
    project = _project_with_tests(tmp_path / "project", "test_one", "test_two")
    data_dir = tmp_path / "coverage-data"

    def fake_run(command, *, env, check, timeout=None):
        if runner.LEDGER_ENV in env:
            # The controller itself was killed: no ledger, no summary.
            return subprocess.CompletedProcess(command, -signal.SIGKILL)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert _main(runner, data_dir) == -signal.SIGKILL
    batch = _record(runner, data_dir)["batches"][0]
    assert batch["ledger"] is None
    assert batch["lost_files"] == ["tests/test_one.py", "tests/test_two.py"]
    assert "no readable ledger" in batch["lost"][0]["error"]
    assert [outcome["file"] for outcome in batch["recovered_files"]] == (
        batch["lost_files"]
    )


def test_a_shard_stopped_part_way_leaves_a_record_the_gate_calls_incomplete(
    tmp_path, monkeypatch,
):
    runner = _load_coverage_runner()
    project = _project_with_tests(tmp_path / "project", "test_one", "test_two")
    data_dir = tmp_path / "coverage-data"

    def fake_run(command, *, env, check, timeout=None):
        if "tests/test_two.py" in command:
            raise RuntimeError("the runner was taken away")
        _write_ledger(runner, env, {
            "gw0": _worker(["tests/test_one.py"], returned=True),
        })
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        _main(runner, data_dir, "--batch-size", "1")

    record = _record(runner, data_dir)
    assert (record["batches_finished"], record["batches_total"]) == (1, 2)
    assert "coverage shard 0 finished 1 of 2 batches" in (
        ratchet.check_shard_integrity(data_dir, 1)["issues"]
    )


# -- a real segfault under real xdist and pytest-cov -------------------------


TOY_CRASH = textwrap.dedent('''
    import os
    import signal
    from pathlib import Path

    from toypkg import alpha


    def test_a_passes_and_covers_alpha():
        assert alpha.run(True) == 1
        assert alpha.run(False) == 0


    def test_b_segfaults_its_worker_the_first_time():
        marker = Path(os.environ["TOY_CRASH_MARKER"])
        if marker.exists():
            return
        marker.write_text("crashed", encoding="utf-8")
        if os.environ.get("TOY_EXIT") == "memory-guard":
            # What tests/conftest.py's memory guard does at its ceiling.
            os._exit(3)
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ImportError, OSError, ValueError):
            pass
        os.kill(os.getpid(), signal.SIGSEGV)
''')

TOY_BETA = textwrap.dedent('''
    import time

    from toypkg import beta


    def test_beta():
        time.sleep(1)
        assert beta.run(True) == 1
        assert beta.run(False) == 0
''')


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
    (tests / "test_0_crash.py").write_text(TOY_CRASH, encoding="utf-8")
    (tests / "test_1_beta.py").write_text(TOY_BETA, encoding="utf-8")
    (root / ".coveragerc").write_text(
        "[run]\nbranch = True\nparallel = True\nrelative_files = True\n"
        "source = toypkg\n",
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return root


def _run_toy_shard(project, data_dir, *, attempts, crash=True, how="segfault"):
    marker = project / "crashed.marker"
    marker.unlink(missing_ok=True)
    if not crash:
        marker.write_text("no crash this time", encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        # An outer coverage run or xdist worker must not leak into the toy.
        if not key.startswith(("COV", "PYTEST_"))
    }
    environment.update({
        "PYTHONPATH": str(project),
        "TOY_CRASH_MARKER": str(marker),
        "TOY_EXIT": how,
        # A replacement worker that arrives after the toy's two short files
        # are done can leave xdist waiting forever (seen on this toy, and
        # without the ledger plugin too). Not restarting keeps the toy
        # deterministic; a test it never ran is an unreported file, which
        # the ledger must catch as well.
        "PYTEST_ADDOPTS": "--max-worker-restart=0 -p no:randomly -p no:cacheprovider",
    })
    return subprocess.run(
        [
            sys.executable, str(RUNNER_SCRIPT), "tests",
            "--marker", "not gui", "--shard-index", "0", "--shard-count", "1",
            "--workers", "2", "--data-dir", str(data_dir),
            "--cov-source", "toypkg",
            "--recovery-attempts", str(attempts), "--recovery-timeout", "120",
        ],
        cwd=project, env=environment, capture_output=True, text=True,
        timeout=300, check=False,
    )


def _lines(data_dir, project, module):
    coverage = pytest.importorskip("coverage")
    combined = data_dir.parent / f"{data_dir.name}-combined" / ".coverage"
    combined.parent.mkdir()
    measurement = coverage.Coverage(
        data_file=str(combined), config_file=str(project / ".coveragerc"),
    )
    try:
        measurement.combine([str(data_dir)], keep=True)
    except coverage.exceptions.NoDataError:
        return set()
    data = measurement.get_data()
    return {
        line
        for name in data.measured_files()
        if name.replace("\\", "/").endswith(f"toypkg/{module}.py")
        for line in (data.lines(name) or ())
    }


needs_a_real_segfault = pytest.mark.skipif(
    sys.platform == "win32", reason="needs a POSIX SIGSEGV",
)


@needs_a_real_segfault
def test_a_segfaulted_worker_loses_passed_coverage_and_the_runner_recovers_it(
    tmp_path,
):
    pytest.importorskip("pytest_cov")
    pytest.importorskip("xdist")
    runner = _load_coverage_runner()
    project = _toy(tmp_path / "toy")

    # NEGATIVE CONTROL, recovery off: a test PASSED on the worker, the worker
    # then segfaulted, and that passed test's lines are not in the data.
    control = tmp_path / "control"
    lost = _run_toy_shard(project, control, attempts=0)
    output = lost.stdout + lost.stderr
    assert "Segmentation fault" in output, output[-4000:]
    assert "PASSED tests/test_0_crash.py::test_a_passes_and_covers_alpha" in output
    assert lost.returncode != 0
    assert not BODY <= _lines(control, project, "alpha")
    control_batch = _record(runner, control)["batches"][0]
    # The exit status came through execnet: named a segfault, not "unknown".
    assert any(
        reason["worker"] and reason["running"] == [CRASHING_TEST]
        and reason["exit_kind"] == "segfault"
        for reason in control_batch["lost"]
    ), control_batch["lost"]
    assert "tests/test_0_crash.py" in [
        outcome["file"] for outcome in control_batch["unrecovered_files"]
    ]
    assert ratchet.check_shard_integrity(control, 1)["status"] == "incomplete"

    # The same crash with recovery on: the lost files re-run, the lines back.
    recovered = tmp_path / "recovered"
    result = _run_toy_shard(project, recovered, attempts=2)
    output = result.stdout + result.stderr
    assert result.returncode != 0, "a recovered crash still fails its batch"
    assert BODY <= _lines(recovered, project, "alpha"), output[-4000:]
    batch = _record(runner, recovered)["batches"][0]
    assert "tests/test_0_crash.py" in [
        outcome["file"] for outcome in batch["recovered_files"]
    ]
    assert batch["unrecovered_files"] == []
    integrity = ratchet.check_shard_integrity(recovered, 1)
    assert integrity["status"] == "complete", integrity
    assert integrity["recovered"]


@needs_a_real_segfault
def test_a_worker_ended_by_the_memory_guard_is_named_so_and_recovered(tmp_path):
    """Same lost data, different cause: the report must not call it a crash."""
    pytest.importorskip("pytest_cov")
    pytest.importorskip("xdist")
    runner = _load_coverage_runner()
    project = _toy(tmp_path / "toy")
    data_dir = tmp_path / "data"

    result = _run_toy_shard(project, data_dir, attempts=2, how="memory-guard")

    output = result.stdout + result.stderr
    assert "Segmentation fault" not in output
    batch = _record(runner, data_dir)["batches"][0]
    guarded = [
        reason for reason in batch["lost"]
        if reason["worker"] and reason["running"] == [CRASHING_TEST]
    ]
    assert [reason["exit_kind"] for reason in guarded] == ["memory-guard"], (
        batch["lost"]
    )
    assert "was ended by the test memory guard" in output
    assert BODY <= _lines(data_dir, project, "alpha"), output[-4000:]
    integrity = ratchet.check_shard_integrity(data_dir, 1)
    assert integrity["status"] == "complete", integrity
    assert integrity["loss_kinds"].get("memory-guard") == 1


@needs_a_real_segfault
def test_a_healthy_xdist_batch_reports_every_worker_returned_and_re_runs_nothing(
    tmp_path,
):
    """The ledger's positive signal, on real pytest-cov: without it every
    batch would look lost and every file would run twice."""
    pytest.importorskip("pytest_cov")
    pytest.importorskip("xdist")
    runner = _load_coverage_runner()
    project = _toy(tmp_path / "toy")
    data_dir = tmp_path / "data"

    result = _run_toy_shard(project, data_dir, attempts=2, crash=False)

    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    batch = _record(runner, data_dir)["batches"][0]
    assert batch["lost_files"] == []
    assert batch["recovered_files"] == []
    assert "coverage recovery:" not in result.stdout
    workers = json.loads(
        (data_dir / batch["ledger"]).read_text(encoding="utf-8")
    )["workers"]
    assert len(workers) == 2
    assert all(entry["coverage_returned"] is True for entry in workers.values())
    assert sorted(name for entry in workers.values() for name in entry["files"]) == [
        "tests/test_0_crash.py", "tests/test_1_beta.py",
    ]
    assert BODY <= _lines(data_dir, project, "alpha")
