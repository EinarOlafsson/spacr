"""Automatic publication must preserve unreviewed edits and failing checks."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepared_publisher", Path(__file__).resolve().parents[1] / "tools/publish_prepared_items.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def test_shared_checkout_is_never_eligible():
    with pytest.raises(ValueError, match="isolated worktree"):
        publisher.validate({"root": "/mnt/firecuda2/Claude/repo/spacr"})


def test_prepared_scope_requires_explicit_review(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "WORKTREES", tmp_path)
    root = tmp_path / "private"
    root.mkdir()
    with pytest.raises(ValueError, match="reviewed"):
        publisher.validate({"root": str(root), "completion_approved": False})


@pytest.mark.parametrize("tests,failed,errors,skipped", [(0,0,0,0),(2,1,0,0),(2,0,1,0),(2,0,0,1)])
def test_empty_failed_or_skipped_checks_never_finish_an_item(tmp_path, tests, failed, errors, skipped):
    path = tmp_path / "result.xml"
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failed}" errors="{errors}" skipped="{skipped}"/></testsuites>')
    with pytest.raises(ValueError):
        publisher.counts(path)


def test_real_passing_counts_are_retained(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    assert publisher.counts(path) == dict(tests=3, failures=0, errors=0, skipped=0)


def test_source_drift_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "WORKTREES", tmp_path)
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(publisher, "git", lambda *args: "changed")
    with pytest.raises(ValueError, match="no longer matches"):
        publisher.validate(dict(root=str(root), completion_approved=True, base="prepared"))


def test_unrelated_work_is_never_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "WORKTREES", tmp_path)
    root = tmp_path / "private"
    root.mkdir()
    item = root / "item.txt"
    item.write_text("Status: COMPLETE 100%\n")
    def fake_git(root, *args):
        return {"rev-parse": "base", "branch": "private", "status": " M item.txt\n M unrelated.txt"}[args[0]]
    monkeypatch.setattr(publisher, "git", fake_git)
    monkeypatch.setattr(publisher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="allowlist"):
        publisher.validate(dict(root=str(root), completion_approved=True, base="base",
                                files={"item.txt": publisher.digest(item)}))


def test_failed_test_command_never_commits_or_pushes(tmp_path, monkeypatch):
    manifest = tmp_path / "failure.job.json"
    manifest.write_text(json.dumps(dict(tests=["tests/test_example.py"])))
    monkeypatch.setattr(publisher, "validate", lambda job: tmp_path)
    monkeypatch.setattr(publisher.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=12345, wait=lambda timeout: 1))
    mutations = []
    monkeypatch.setattr(publisher, "git", lambda *args: mutations.append(args))
    output = tmp_path / "results"
    output.mkdir()
    publisher.process(manifest, output)
    result = json.loads((output / "failure.job.json").read_text())
    assert result["status"] == "blocked"
    assert "Prepared checks failed" in result["error"]
    assert mutations == []
