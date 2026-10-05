"""Home-folder pruning and cache management at their edges."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from spacr import run_journal as rj
from tests.test_home_storage_prune_caches_purge import _aged, home  # noqa: F401


def test_quiet_commands_that_fail_or_print_nothing_answer_none(monkeypatch):
    def broken(*a, **k):
        raise OSError("no such program")

    monkeypatch.setattr(subprocess, "run", broken)
    assert rj._run_quiet(["conda", "list"]) is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, stdout="", stderr=""))
    assert rj._run_quiet(["conda", "list"]) is None


def test_a_conda_prefix_without_conda_has_no_lock(tmp_path, monkeypatch):
    import shutil
    import sys

    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert rj._conda_lock_text() is None
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/conda" if name == "conda" else None)
    monkeypatch.setattr(rj, "_run_quiet", lambda command: "lock")
    assert rj._conda_lock_text() == "lock"


def test_a_failed_environment_snapshot_is_only_logged(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no metadata")

    monkeypatch.setattr(rj, "_environment_lock_texts", broken)
    assert rj._warm_env_snapshot() is None


def test_tree_stats_of_missing_or_unreadable_entries(tmp_path, monkeypatch):
    assert rj._tree_stats(tmp_path / "gone") == (0, 0.0)
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "a").write_bytes(b"12")
    real = os.scandir

    def refuse(path):
        raise OSError("permission")

    monkeypatch.setattr(rj.os, "scandir", refuse)
    size, _newest = rj._tree_stats(folder)
    assert size >= 0
    monkeypatch.setattr(rj.os, "scandir", real)


def test_an_unknown_storage_kind_and_cache_are_refused():
    with pytest.raises(ValueError, match="unknown storage kind"):
        rj._prune_root("nonsense")
    with pytest.raises(ValueError, match="unknown cache"):
        rj._cache_folder("nonsense")


def test_the_running_run_is_protected(home, monkeypatch):  # noqa: F811
    run_dir = home / ".spacr" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(rj, "current_run", lambda: type("R", (), {"dir": str(run_dir)})())
    protected = rj._protected_storage()
    assert "r1" in protected
    assert rj._is_protected(run_dir, protected) is True


def test_an_unreadable_storage_root_plans_nothing(home, monkeypatch):  # noqa: F811
    monkeypatch.setattr(rj, "_prune_root", lambda kind: home / "not-a-folder")
    plan = rj._prune_plan("logs", 7, 0)
    assert plan["delete"] == []


def test_entries_that_vanish_or_cannot_be_deleted_are_refused(home, monkeypatch):  # noqa: F811
    logs = Path(os.environ["SPACR_LOG_DIR"])
    old = _aged(logs / "spacr-20200101.log", 400)
    plan = rj._prune_plan("logs", 7, 0)
    assert plan["delete"]
    real_unlink = Path.unlink

    def refuse(self, *a, **k):
        if self.name == old.name:
            raise OSError("busy")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", refuse)
    deleted, _freed, refused = rj._prune(plan)
    assert deleted == 0 and any("busy" in why for why in refused)


def test_cache_refusals_and_relocation_guards(home, tmp_path, monkeypatch):  # noqa: F811
    assert "spaCR's own home" in rj._cache_refusal(rj._spacr_home())
    assert rj._clear_cache("news") == (0, [])
    target_parent = tmp_path / "elsewhere"
    full = target_parent / "spacr-news"
    full.mkdir(parents=True)
    (full / "keep").write_text("x")
    with pytest.raises(ValueError, match="already holds files"):
        rj._relocate_cache("news", target_parent)
    moved = rj._relocate_cache("torch", tmp_path / "fresh")
    assert moved.is_dir()


def test_a_failed_shared_environment_lock_still_returns_this_runs_texts(
        home, monkeypatch, caplog):  # noqa: F811
    """A read-only shared cache must not leave a run without a lockfile."""
    digest = "unwritable-environment"
    rj._ENV_LOCK_MEMO.pop(digest, None)
    monkeypatch.setattr(rj, "_pip_lock_text", lambda packages: "spacr==1.5\n")
    monkeypatch.setattr(rj, "_conda_lock_text", lambda: None)

    def refuse(path, text):
        raise PermissionError("shared store is read-only")

    monkeypatch.setattr(rj, "_atomic_write_text", refuse)
    try:
        with caplog.at_level("WARNING", logger="spacr.run_journal"):
            texts = rj._environment_lock_texts(digest, {"spacr": "1.5"})
        assert texts == {"pip": "spacr==1.5\n", "conda": None}
        assert "Could not save the environment lock" in caplog.text
        assert rj._environment_lock_texts(digest, {}) is texts
    finally:
        rj._ENV_LOCK_MEMO.pop(digest, None)


def test_tree_stats_skips_a_file_removed_during_the_scan(tmp_path,
                                                          monkeypatch):
    """A disappearing cache entry must not hide the readable siblings."""
    folder = tmp_path / "cache"
    folder.mkdir()
    (folder / "kept.bin").write_bytes(b"1234")
    real_scandir = os.scandir

    class Vanished:
        def stat(self, follow_symlinks=False):
            raise FileNotFoundError("removed while scanning")

    class Listing:
        def __init__(self, path):
            self.entries = real_scandir(path)

        def __enter__(self):
            return list(self.entries) + [Vanished()]

        def __exit__(self, *_exc):
            self.entries.close()
            return False

    monkeypatch.setattr(rj.os, "scandir", Listing)
    size, _newest = rj._tree_stats(folder)
    assert size >= 4


def test_clear_logs_refuses_a_file_replaced_by_a_link(home, tmp_path):  # noqa: F811
    """A plan cannot follow a changed log path outside spaCR's folder."""
    logs = Path(os.environ["SPACR_LOG_DIR"])
    original = _aged(logs / "spacr-20200101.log", 400)
    outside = tmp_path / "precious.log"
    outside.write_text("keep me", encoding="utf-8")
    plan = rj._clear_logs_plan()
    original.unlink()
    original.symlink_to(outside)

    deleted, emptied, _freed, refused = rj._clear_logs(plan)
    assert (deleted, emptied) == (0, 0)
    assert "not a log any more" in refused[0]
    assert outside.read_text(encoding="utf-8") == "keep me"


def test_clear_logs_keeps_a_run_log_that_became_active(home, monkeypatch):  # noqa: F811
    """A run started after the dialog opened keeps its log."""
    runs = Path(os.environ["SPACR_LOG_DIR"]) / "runs"
    active = _aged(runs / "live123.jsonl", 100)
    plan = rj._clear_logs_plan()
    monkeypatch.setenv("SPACR_RUN_ID", "live123")

    deleted, emptied, _freed, refused = rj._clear_logs(plan)
    assert (deleted, emptied) == (0, 0)
    assert active.exists()
    assert refused == ["live123.jsonl: in use"]


def test_clear_logs_reports_a_file_that_cannot_be_unlinked(home,
                                                             monkeypatch):  # noqa: F811
    """A busy log is left in place and named in the result."""
    logs = Path(os.environ["SPACR_LOG_DIR"])
    busy = _aged(logs / "spacr-20200101.log", 400)
    plan = rj._clear_logs_plan()
    real_unlink = Path.unlink

    def refuse(self, *args, **kwargs):
        if self == busy:
            raise PermissionError("log file is busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse)
    deleted, emptied, _freed, refused = rj._clear_logs(plan)
    assert (deleted, emptied) == (0, 0)
    assert busy.exists()
    assert any("log file is busy" in message for message in refused)
