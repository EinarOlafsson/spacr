"""The GUI thread's look for another process's preference write makes no system call.

Once a watch interval the first preference read on a thread stats the store
file to see whether another process wrote it. On the GUI thread that read
can be anything -- a drop onto a module, a paint, a translation lookup --
and test_a_folder_drop_makes_no_filesystem_call_on_the_gui_thread caught it
inside a drop (metadata_table -> theme -> preferences). A daemon thread now
makes that look for the GUI thread, and a write from another process is
still seen within one interval. Counted, not timed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the native store is an INI file only on Linux")


@pytest.fixture
def fresh_store(monkeypatch):
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    monkeypatch.setattr(prefs, "_REMEMBERED", {})
    store = prefs._real_store()
    store.setValue("tests/gui_watch", 1)
    store.sync()
    path = store.fileName()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        seen = prefs._GUI_WATCH["seen"]
        if seen is not None and seen[0] == path:
            break
        time.sleep(0.02)
    return store, path


def _gui_stats(monkeypatch):
    calls = []
    gui = threading.get_ident()
    real = os.stat

    def stat(*args, **kwargs):
        if threading.get_ident() == gui:
            calls.append(args[0] if args else kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    return calls


def _due():
    prefs._STORE.checked = prefs._store_clock() - prefs._STORE_WATCH_SECONDS


def test_a_due_look_on_the_gui_thread_reads_the_watcher(fresh_store,
                                                        monkeypatch):
    _store, path = fresh_store
    assert prefs._GUI_WATCH["seen"][0] == path
    calls = _gui_stats(monkeypatch)
    for _ in range(3):
        _due()
        prefs._real_store()
        prefs.get_reduce_motion()
    assert calls == []


def test_another_process_write_is_seen_through_the_watcher(fresh_store,
                                                           monkeypatch):
    store, path = fresh_store
    before = prefs._GUI_WATCH["seen"][1]
    prefs.set_reduce_motion(False)
    store.sync()
    subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from PySide6.QtCore import QSettings\n"
         "store = QSettings(sys.argv[1], QSettings.IniFormat)\n"
         "store.setValue('prefs/reduce_motion', True)\n"
         "store.sync()\n",
         path],
        check=True, timeout=60)
    deadline = time.monotonic() + 10
    while (time.monotonic() < deadline
           and prefs._GUI_WATCH["seen"][1] == before):
        time.sleep(0.02)
    calls = _gui_stats(monkeypatch)
    _due()
    assert prefs.get_reduce_motion() is True
    assert path not in calls


def test_a_worker_thread_still_watches_its_own_store(monkeypatch):
    seen = {}

    def work():
        monkeypatch.setattr(prefs, "_REMEMBERED", {})
        store = prefs._real_store()
        seen["watched"] = getattr(prefs._STORE, "watched", None)
        seen["file"] = store.fileName()

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(30)
    assert seen["file"] and seen["watched"] is None
    assert QSettings is not None
