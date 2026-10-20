"""A preference store that cannot be watched is opened afresh for each read.

The kept per-thread store is reread when its file changes on disk, which
it looks for once a second. Where
there is no file to watch -- the Windows registry, the macOS preferences
daemon, a file name the OS refuses -- the store is not kept, so a write by
another process is never missed. Moving the settings folder counts as a
move only when the folder actually changes.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.qt


def _store(name, fmt=QSettings.IniFormat):
    return SimpleNamespace(format=lambda: fmt, fileName=lambda: name)


def test_a_native_store_on_windows_or_macos_cannot_be_watched(monkeypatch, tmp_path):
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(prefs.sys, "platform", platform)
        assert prefs._store_file_state(_store(str(tmp_path / "x.ini"), QSettings.NativeFormat)) is None
    monkeypatch.setattr(prefs.sys, "platform", "darwin")
    assert prefs._store_file_state(_store(str(tmp_path / "x.ini"))) == ()


def test_a_store_file_the_os_refuses_cannot_be_watched(tmp_path):
    assert prefs._store_file_state(_store("bad\0name.ini")) is None
    assert prefs._store_file_state(_store(str(tmp_path / "unwritten.ini"))) == ()
    written = tmp_path / "written.ini"
    written.write_text("[General]\n")
    assert len(prefs._store_file_state(_store(str(written)))) == 3


def test_an_unwatchable_store_is_never_kept(monkeypatch):
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    monkeypatch.setattr(prefs, "_store_file_state", lambda store: None)
    first = prefs._real_store()
    second = prefs._real_store()
    assert isinstance(first, QSettings) and first is not second
    assert prefs._STORE.store is None


def test_a_kept_store_that_becomes_unwatchable_is_replaced(monkeypatch):
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    states = {"value": ()}
    monkeypatch.setattr(prefs, "_store_file_state", lambda store: states["value"])
    now = [1000.0]
    monkeypatch.setattr(prefs, "_store_clock", lambda: now[0])
    kept = prefs._real_store()
    assert prefs._real_store() is kept
    states["value"] = None
    now[0] += prefs._STORE_WATCH_SECONDS
    fresh = prefs._real_store()
    assert fresh is not kept and isinstance(fresh, QSettings)
    assert prefs._STORE.store is None


def test_only_a_change_of_folder_counts_as_a_move(monkeypatch):
    calls = []

    def recorder(fmt, scope, path):
        calls.append(path)

    monkeypatch.setattr(QSettings, "setPath", staticmethod(recorder))
    monkeypatch.setattr(prefs, "_STORE_FOLDERS", {})
    monkeypatch.setattr(prefs, "_STORE_MOVES", [0])
    prefs._count_store_moves()
    counting = QSettings.setPath
    prefs._count_store_moves()
    assert QSettings.setPath is counting, "the counter wrapped itself"
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, "/portable/settings")
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, "/portable/settings")
    assert calls == ["/portable/settings", "/portable/settings"]
    assert prefs._STORE_MOVES == [1]
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, "/elsewhere")
    assert prefs._STORE_MOVES == [2]
