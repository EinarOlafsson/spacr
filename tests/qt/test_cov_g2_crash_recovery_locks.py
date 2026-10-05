"""Project and instance locks when Qt's lock file misbehaves."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt import crash_recovery as cr  # noqa: E402


class _Lock:
    def __init__(self, info):
        self.info = info

    def setStaleLockTime(self, ms):
        pass

    def tryLock(self, ms):
        return False

    def getLockInfo(self):
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


@pytest.fixture
def fake_lock(monkeypatch):
    import PySide6.QtCore as core

    box = {}
    monkeypatch.setattr(core, "QLockFile", lambda path: _Lock(box["info"]))
    monkeypatch.setattr(cr, "_HELD_LOCKS", {})
    return box


@pytest.mark.parametrize("info, expected", [
    (RuntimeError("no info"), (False, {})),
    ((True, "not-a-pid", "host", "app"), (False, {})),
    ((True, os.getpid(), "host", "app"), (True, None)),
])
def test_lock_holders_are_read_defensively(fake_lock, tmp_path, info, expected):
    fake_lock["info"] = info
    assert cr._try_lock(str(tmp_path / "x.lock")) == expected


def test_a_lock_that_will_not_unlock_is_still_forgotten(monkeypatch):
    class _Stuck:
        def unlock(self):
            raise RuntimeError("gone")

    monkeypatch.setattr(cr, "_HELD_LOCKS", {"/x": _Stuck()})
    cr._release_all_locks()
    assert cr._HELD_LOCKS == {}


def test_an_instance_lock_that_cannot_be_taken_reports_nothing(monkeypatch):
    def broken(path):
        raise OSError("read-only home")

    monkeypatch.setattr(cr, "_try_lock", broken)
    assert cr._claim_the_instance() == {}


def test_projects_without_a_path_or_a_lockable_folder_count_as_locked(
        monkeypatch):
    assert cr._claim_project(None, "") == "locked"

    def broken(path):
        raise OSError("no locks folder")

    monkeypatch.setattr(cr, "_try_lock", broken)
    assert cr._claim_project(None, "/data/plate") == "locked"


def test_a_cancelled_answer_is_not_remembered(monkeypatch):
    monkeypatch.setattr(cr, "_try_lock", lambda path: (False, {"pid": 1}))
    monkeypatch.setattr(cr, "_PROJECT_CHOICES", {})
    monkeypatch.setattr(cr, "_ask_about_a_locked_project",
                        lambda parent, path, holder: "cancel")
    assert cr._claim_project(None, "/data/plate") == "cancel"
    assert cr._PROJECT_CHOICES == {}
