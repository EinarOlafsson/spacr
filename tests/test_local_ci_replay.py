"""Local replays preserve CI partitions, source provenance and failing exits."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import replay_ci_batch as replay

ROOT = Path(__file__).resolve().parents[1]


def test_pytest_plugin_inventory_accepts_python39_and_selectable_metadata(monkeypatch):
    plugins = [SimpleNamespace(name="fixture", value="package:plugin")]
    unrelated = [SimpleNamespace(name="other", value="package:other")]
    monkeypatch.setattr(replay.importlib.metadata, "entry_points",
                        lambda: {"pytest11": plugins, "console_scripts": unrelated})
    assert replay._pytest_plugin_entry_points() == {"fixture": "package:plugin"}

    class SelectableEntries:
        """Provide the selectable entry-point interface of newer Python."""

        def select(self, *, group):
            assert group == "pytest11"
            return plugins

    monkeypatch.setattr(replay.importlib.metadata, "entry_points", SelectableEntries)
    assert replay._pytest_plugin_entry_points() == {"fixture": "package:plugin"}


@pytest.mark.parametrize("suite,count,size,timeout,ignored", [
    ("fast", 3, 32, 300, 0),
    ("qt", 3, 16, 1200, 7),
    ("coverage", 12, 32, 600, 5),
])
def test_profile_comes_from_the_actual_workflow(suite, count, size, timeout, ignored, tmp_path):
    _, args, _, actual_count, python = replay._configuration(ROOT, suite, 1, tmp_path)
    assert (actual_count, args.batch_size, args.per_test_timeout) == (count, size, timeout)
    assert (args.workers, args.batch_timeout, python) == (2, 2700, "3.12")
    assert len(args.exclude_file if suite == "coverage" else args.ignore) == ignored
    assert args.marker == {
        "fast": "not integration and not slow and not heavy and not qt and not gpu and not network and not nas and not gui",
        "qt": "qt and not slow and not heavy and not gpu and not network and not nas and not gui",
        "coverage": "not gui",
    }[suite]


def test_fast_and_qt_do_not_rebucket_files_before_batching(tmp_path, monkeypatch):
    tests = tmp_path / "tests"
    tests.mkdir()
    for index in range(83):
        (tests / f"test_{index:03d}.py").touch()
    monkeypatch.chdir(tmp_path)
    for suite in ("fast", "qt"):
        runner, args, _, count, _ = replay._configuration(ROOT, suite, 0, tmp_path)
        left = replay._partition(tmp_path, suite, runner, args, 0, count)
        right = replay._partition(tmp_path, suite, runner, args, count - 1, count)
        assert left == right
        assert [path for batch in left for path in batch] == [f"tests/test_{i:03d}.py" for i in range(83)]


def test_fast_replay_names_and_refuses_a_batch_ci_skips(tiny_repo, tmp_path, capsys):
    workflow = tiny_repo / ".github/workflows/tests.yml"
    source = workflow.read_text()
    workflow.write_text(source.replace(
        '--marker "not slow" --batch-size',
        '--marker "not qt and not slow" --skip-qt-only-batches --batch-size',
    ))
    qt = tiny_repo / "tests/qt"
    qt.mkdir()
    for name in ("a", "b"):
        (qt / f"test_{name}.py").write_text("def test_x(): pass\n")
    base = ["--repo", str(tiny_repo), "--suite", "fast", "--shard", "0"]
    assert replay.main([*base, "--batch", "1", "--list"]) == 0
    assert "CI skips this automatically Qt-marked batch" in capsys.readouterr().out
    output = tmp_path / "evidence"
    with pytest.raises(SystemExit) as stopped:
        replay.main([*base, "--batch", "1", "--output", str(output)])
    assert stopped.value.code == 2
    assert "CI skips this automatically Qt-marked batch" in capsys.readouterr().err
    assert not output.exists()


def test_coverage_hashes_relative_paths_before_making_batches(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    labels = [f"tests/test_{index:03d}.py" for index in range(260)]
    for label in labels:
        (tmp_path / label).touch()
    runner, args, _, count, _ = replay._configuration(ROOT, "coverage", 4, tmp_path)
    args.batch_size = 7
    args.exclude_file = [labels[0]]
    actual = replay._partition(tmp_path, "coverage", runner, args, 4, count)
    expected = [label for label in labels if label != labels[0]
                and int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % count == 4]
    assert [path for batch in actual for path in batch] == expected
    assert all(len(batch) == 7 for batch in actual[:-1])


def test_coverage_runner_imports_and_plans_without_installed_coverage():
    """The complete coverage runner imports and plans without coverage.py."""
    result = subprocess.run([
        sys.executable, "-S", "-c",
        "from pathlib import Path; from tools import replay_ci_batch as replay; "
        "runner = replay._load_runner(Path.cwd(), True); "
        "print(Path(runner.__file__).name, hasattr(runner, '_run_batch')); "
        "print(runner._shard(Path.cwd() / 'tests/test_local_ci_replay.py', "
        "Path.cwd(), 12)); print(runner.build_parser().parse_args("
        "['tests', '--marker', 'not gui', '--shard-index', '4', "
        "'--shard-count', '12', '--data-dir', '/tmp']).shard_count)",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "run_coverage_batches.py True"
    assert result.stdout.splitlines()[-1] == "12"


def test_coverage_execution_refuses_a_missing_measurement_dependency(
        tmp_path, monkeypatch, capsys):
    """A dry plan can work without coverage while execution fails clearly."""
    original = replay.importlib.util.find_spec
    monkeypatch.setattr(replay.importlib.util, "find_spec",
                        lambda name: None if name == "coverage" else original(name))
    output = tmp_path / "evidence"
    assert replay.main(["--repo", str(ROOT), "--suite", "coverage", "--shard", "0",
                        "--batch", "1", "--list"]) == 0
    assert any(line.startswith("tests/") for line in capsys.readouterr().out.splitlines())
    with pytest.raises(SystemExit) as stopped:
        replay.main(["--repo", str(ROOT), "--suite", "coverage", "--shard", "0",
                     "--batch", "1", "--output", str(output)])
    assert stopped.value.code == 2
    assert "coverage.py is required" in capsys.readouterr().err
    assert not output.exists()


def test_environment_cannot_inherit_gpu_sharding_or_extra_pytest_options(tmp_path, monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "SPACR_PYTEST_FILE_SHARD_COUNT", "PYTEST_ADDOPTS", "COVERAGE_FILE", "SPACR_HF_E2E_STUB"):
        monkeypatch.setenv(key, "inherited")
    env = replay._environment(ROOT, tmp_path, "coverage", 4, 12)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["SPACR_PYTEST_FILE_SHARD_COUNT"] == "1"
    assert env["SPACR_HF_E2E_STUB"] == "1"
    assert env["SPACR_TEST_MEMORY_GB"] == "6"
    assert "PYTEST_ADDOPTS" not in env and "COVERAGE_FILE" not in env
    fast = replay._environment(ROOT, tmp_path, "fast", 2, 3)
    assert fast["SPACR_PYTEST_FILE_SHARD_INDEX"] == "2"
    assert fast["SPACR_PYTEST_FILE_SHARD_COUNT"] == "3"
    assert "SPACR_HF_E2E_STUB" not in fast
    assert fast["CI"] == fast["GITHUB_ACTIONS"] == "true"
    selected = replay._environment(ROOT, tmp_path, "coverage", 0, 12, ["randomly"])
    assert selected["PYTEST_ADDOPTS"] == "-p no:randomly"


@pytest.fixture
def tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / ".github/workflows").mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/run_pytest_batches.py", repo / "tools/run_pytest_batches.py")
    (repo / "tools/run_capped.sh").write_text('''#!/usr/bin/env bash
test "$1" = "8G" || exit 97
shift
exec "$@"
''')
    (repo / ".github/workflows/tests.yml").write_text('''jobs:
  fast:
    steps:
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - name: Run fast tests
        env:
          SPACR_PYTEST_FILE_SHARD_COUNT: "1"
        run: |
          python tools/run_pytest_batches.py --marker "not slow" --batch-size 2 --workers 1 --per-test-timeout 0 --batch-timeout 60
''')
    (repo / "tests/test_a.py").write_text("def test_pass():\n    pass\n")
    (repo / "tests/test_b.py").write_text("def test_failure_is_reported():\n    assert False, 'local diagnostic failure'\n")
    (repo / "tests/test_c.py").write_text("def test_outside_selected_batch():\n    raise RuntimeError('must not run')\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Einar Olafsson", "-c", "user.email=test@localhost", "commit", "-qm", "Fixture"], check=True)
    return repo


def test_real_failing_child_keeps_its_exit_live_log_and_exact_manifest(tiny_repo, tmp_path):
    output = tmp_path / "evidence"
    result = subprocess.run([
        sys.executable, str(ROOT / "tools/replay_ci_batch.py"), "--repo", str(tiny_repo),
        "--suite", "fast", "--shard", "0", "--batch", "1", "--output", str(output),
    ], capture_output=True, text=True,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    assert result.returncode == 1, result.stdout + result.stderr
    evidence = json.loads((output / "result.json").read_text())
    assert evidence["exit_code"] == 1 and not evidence["source_changed"]
    assert evidence["failed_test_ids"] == ["tests/test_b.py::test_failure_is_reported"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert [entry["path"] for entry in manifest["source"]["files"]] == ["tests/test_a.py", "tests/test_b.py"]
    assert manifest["scope"].startswith("one batch")
    log = (output / "pytest.log").read_text()
    assert "local diagnostic failure" in log
    assert "must not run" not in log
    assert "FAILED tests/test_b.py" in result.stdout


def test_wrong_sha_and_missing_batch_refuse_to_run(tiny_repo, tmp_path):
    output = tmp_path / "evidence"
    base = [sys.executable, str(ROOT / "tools/replay_ci_batch.py"), "--repo", str(tiny_repo), "--suite", "fast", "--shard", "0"]
    result = subprocess.run([*base, "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2 and not output.exists()
    result = subprocess.run([*base, "--list", "--expect-sha", "0" * 40], capture_output=True, text=True)
    assert result.returncode == 2 and not output.exists()


def test_exact_sha_refuses_dirty_source_and_output_inside_repo(tiny_repo, tmp_path):
    sha = subprocess.check_output(["git", "-C", str(tiny_repo), "rev-parse", "HEAD"], text=True).strip()
    base = [sys.executable, str(ROOT / "tools/replay_ci_batch.py"), "--repo", str(tiny_repo), "--suite", "fast", "--shard", "0"]
    (tiny_repo / "tests/test_a.py").write_text("def test_changed():\n    pass\n")
    result = subprocess.run([*base, "--list", "--expect-sha", sha], capture_output=True, text=True)
    assert result.returncode == 2 and "clean source" in result.stderr
    result = subprocess.run([*base, "--list", "--expect-sha", sha, "--allow-dirty"], capture_output=True, text=True)
    assert result.returncode == 0
    result = subprocess.run([*base, "--batch", "1", "--output", str(tiny_repo / "evidence")], capture_output=True, text=True)
    assert result.returncode == 2 and "outside the repository" in result.stderr


def test_cleanup_timeout_keeps_an_interrupted_receipt(monkeypatch):
    from types import SimpleNamespace

    def wedged(timeout):
        raise subprocess.TimeoutExpired("child", timeout)

    monkeypatch.setattr(replay, "_owned_scope", lambda pid: None)
    monkeypatch.setattr(replay.os, "kill", lambda *args: None)
    assert replay._stop_child(SimpleNamespace(pid=999999999, wait=wedged), None) is False


def test_single_coverage_batch_uses_recovery_and_never_claims_shard_integrity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    output = tmp_path / "evidence"
    output.mkdir()
    source = {"sha": "pinned"}
    (output / "manifest.json").write_text(json.dumps({
        "source": source, "helper_sha256": replay._hash(Path(replay.__file__)),
    }))
    monkeypatch.setattr(replay, "_identity", lambda *args: source)
    calls = []

    def run(args, number, files):
        calls.append((number, files))
        return {"exit_code": 0, "unrecovered_files": [{"file": files[0]}]}

    cli = SimpleNamespace(output=output, suite="coverage", batch=3)
    args = SimpleNamespace(data_dir=output / "coverage")
    assert replay._execute(cli, ROOT, SimpleNamespace(_run_batch=run), args, [], ["tests/test_a.py"]) == 1
    assert calls == [(3, ["tests/test_a.py"])]
    assert (output / "coverage-batch.json").is_file()
    assert not list(output.rglob("*integrity*"))


def test_interrupt_stops_a_child_even_in_a_separate_process_session(tiny_repo, tmp_path):
    marker = tmp_path / "sleeper.pid"
    (tiny_repo / "tests/test_a.py").write_text(
        "import subprocess, sys, time\n"
        "def test_wait():\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
        f"    open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "    time.sleep(120)\n"
    )
    output = tmp_path / "interrupt-evidence"
    process = subprocess.Popen([
        sys.executable, str(ROOT / "tools/replay_ci_batch.py"), "--repo", str(tiny_repo),
        "--suite", "fast", "--shard", "0", "--batch", "1", "--output", str(output),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "child test never started"
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=20)
        assert process.returncode == 130, stdout
        receipt = json.loads((output / "result.json").read_text())
        assert receipt["exit_code"] == 130 and receipt["cleanup_complete"] is True
        status = Path(f"/proc/{marker.read_text()}/status")
        assert not status.exists() or "State:\tZ" in status.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
