"""Hosted numeric badges preserve failures and count unique selected tests."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from tools.test_count_badge import endpoint


def records(tmp_path, rows):
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(dict(session=s, nodeid=n, outcome=o)) + "\n"
                for s, n, o in rows), encoding="utf-8")


@pytest.mark.parametrize("passed,color", [(100,"green"),(91,"green"),
    (90,"yellow"),(80,"yellow"),(79,"orange"),(70,"orange"),(69,"red"),(0,"red")])
def test_exact_requested_thresholds(tmp_path, passed, color):
    records(tmp_path, [("run", str(i), "passed" if i < passed else "failed")
                       for i in range(100)])
    badge = endpoint(tmp_path)
    assert badge["message"] == f"{passed}/100"
    assert badge["color"] == color


def test_worker_duplicates_and_teardown_failure(tmp_path):
    records(tmp_path, [("one","a","not_run"),("one","a","passed"),
        ("one","a","passed"),("one","a","failed"),
        ("one","b","not_run"),("one","b","passed"),
        ("two","b","passed"),("one","c","skipped"),
        ("one","d","not_run")])
    assert endpoint(tmp_path)["message"] == "1/4"


def test_failure_in_another_dependency_profile_is_retained(tmp_path):
    records(tmp_path, [("one","a","passed"),("two","a","failed")])
    assert endpoint(tmp_path)["message"] == "0/1"


def test_no_results_are_pending(tmp_path):
    assert endpoint(tmp_path)["message"] == "pending"
    assert endpoint(tmp_path)["color"] == "lightgrey"


@pytest.fixture
def collector():
    path = Path(__file__).resolve().parents[1] / "tools/test_count_badge.py"
    spec = importlib.util.spec_from_file_location("isolated_count_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_collector(collector, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPACR_TEST_COUNT_OWNER_PID", raising=False)
    monkeypatch.setenv("SPACR_TEST_COUNT_DIR", "counts")
    monkeypatch.setenv("SPACR_TEST_COUNT_SESSION", "previous-session")
    collector.pytest_configure(SimpleNamespace())
    return collector._RECORD


def test_outcomes_stay_in_original_folder_when_a_test_changes_directory(collector, tmp_path, monkeypatch):
    record = configure_collector(collector, tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    collector._emit("selected", "not_run")
    collector._emit("selected", "passed")
    assert record.is_absolute()
    assert endpoint(tmp_path / "counts")["message"] == "1/1"
    assert not (other / "counts").exists()


def test_outcome_writes_survive_mocked_application_file_access(collector, tmp_path, monkeypatch):
    configure_collector(collector, tmp_path, monkeypatch)

    def locked(*args, **kwargs):
        raise OSError("locked")

    with monkeypatch.context() as mocked:
        mocked.setattr(Path, "open", locked)
        mocked.setattr("builtins.open", locked)
        collector._emit("selected", "not_run")
        collector._emit("selected", "passed")
        collector._emit("selected", "failed")
    assert endpoint(tmp_path / "counts")["message"] == "0/1"


@pytest.mark.parametrize("disabled", ["directory", "inherited_owner"])
def test_reconfiguration_drops_an_old_record_when_recording_is_disabled(
        collector, tmp_path, monkeypatch, disabled):
    record = configure_collector(collector, tmp_path, monkeypatch)
    collector._emit("primary", "passed")
    if disabled == "directory":
        monkeypatch.delenv("SPACR_TEST_COUNT_DIR")
    else:
        monkeypatch.setenv("SPACR_TEST_COUNT_OWNER_PID", str(os.getpid() + 1))
    collector.pytest_configure(SimpleNamespace())
    collector._emit("nested", "failed")
    assert collector._RECORD is None
    assert collector._SESSION is None
    assert "SPACR_TEST_COUNT_DIR" not in os.environ
    assert [json.loads(line)["nodeid"] for line in record.read_text().splitlines()] == ["primary"]


def test_an_unmarked_primary_and_same_process_reconfiguration_keep_fresh_sessions(
        collector, tmp_path, monkeypatch):
    monkeypatch.delenv("SPACR_TEST_COUNT_OWNER_PID", raising=False)
    configure_collector(collector, tmp_path, monkeypatch)
    first = collector._SESSION
    assert first != "previous-session"
    collector._emit("primary", "passed")
    collector.pytest_configure(SimpleNamespace())
    assert collector._SESSION != first
    collector._emit("primary", "failed")
    assert endpoint(tmp_path / "counts")["message"] == "0/1"


@pytest.mark.parametrize("workers", [0, 2])
def test_nested_pytest_probes_do_not_enter_primary_results(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(
        "def test_intentional_negative_probe():\n    assert False\n", encoding="utf-8")
    (tmp_path / "test_parent.py").write_text('''import json
import os
from pathlib import Path
import subprocess
import sys

def test_primary():
    if os.environ.get("BADGE_PRIMARY_FAIL"):
        assert False, "a real primary failure must remain recorded"
    Path("primary-process.json").write_text(json.dumps({
        "pid": os.getpid(), "owner": os.environ["SPACR_TEST_COUNT_OWNER_PID"],
        "worker": os.environ.get("PYTEST_XDIST_WORKER"),
    }))
    command = [sys.executable, "-m", "pytest", "-p", "tools.test_count_badge", "-q"]
    if os.environ.get("PYTEST_XDIST_WORKER"):
        command += ["-n", "2"]
    collected = subprocess.run(command + ["--collect-only", "test_probe.py"],
                               capture_output=True, text=True, timeout=60)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    failed = subprocess.run(command + ["test_probe.py"],
                            capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1, failed.stdout + failed.stderr
''', encoding="utf-8")
    env = dict(os.environ)
    for marker in ("SPACR_TEST_COUNT_OWNER_PID", "SPACR_TEST_COUNT_SESSION",
                   "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
                   "PYTEST_XDIST_TESTRUNUID"):
        env.pop(marker, None)
    env["SPACR_TEST_COUNT_DIR"] = str(tmp_path / "counts")
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    command = [sys.executable, "-m", "pytest", "-p", "tools.test_count_badge",
               "-q", "test_parent.py"]
    if workers:
        command += ["-n", str(workers)]
    primary = subprocess.run(command, cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=120)
    assert primary.returncode == 0, primary.stdout + primary.stderr
    rows = [json.loads(line) for path in (tmp_path / "counts").glob("*.jsonl")
            for line in path.read_text().splitlines()]
    assert {row["nodeid"] for row in rows} == {"test_parent.py::test_primary"}
    assert len({row["session"] for row in rows}) == 1
    assert endpoint(tmp_path / "counts")["message"] == "1/1"
    process = json.loads((tmp_path / "primary-process.json").read_text())
    assert (process["worker"] is not None) == bool(workers)
    assert (str(process["pid"]) != process["owner"]) == bool(workers)

    env["BADGE_PRIMARY_FAIL"] = "1"
    second = subprocess.run(command, cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert second.returncode == 1, second.stdout + second.stderr
    assert endpoint(tmp_path / "counts")["message"] == "0/1"


def test_readme_install_retires_old_badge_and_is_idempotent(tmp_path):
    from tools.test_count_badge import install_readme_badges
    path = tmp_path / "README.rst"
    path.write_text("|Tests|\n\n.. |Tests| image:: status.svg\n"
                    ".. |Qt| image:: qt.svg\n", encoding="utf-8")
    install_readme_badges(tmp_path)
    before = path.read_bytes()
    install_readme_badges(tmp_path)
    assert path.read_bytes() == before
    assert b"status.svg" not in before
    assert b"|Tests|" not in before
    assert before.count(b".. |Test counts| image::") == 1
    assert b"test-counts.json" in before
