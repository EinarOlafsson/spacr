"""Hosted numeric badges preserve failures and count unique selected tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.test_count_badge import endpoint


def records(tmp_path, rows):
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(dict(session=s, nodeid=n, outcome=o)) + "\n"
                for s, n, o in rows))


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


def test_readme_install_retires_old_badge_and_is_idempotent(tmp_path):
    from tools.test_count_badge import install_readme_badges
    path = tmp_path / "README.rst"
    path.write_text("|Tests|\n\n.. |Tests| image:: status.svg\n"
                    ".. |Qt| image:: qt.svg\n")
    install_readme_badges(tmp_path)
    before = path.read_bytes()
    install_readme_badges(tmp_path)
    assert path.read_bytes() == before
    assert b"status.svg" not in before
    assert b"|Tests|" not in before
    assert before.count(b".. |Test counts| image::") == 1
    assert b"test-counts.json" in before
