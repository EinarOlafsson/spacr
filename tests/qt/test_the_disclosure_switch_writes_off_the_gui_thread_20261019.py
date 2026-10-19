"""Pressing Essentials/All does not write the preference file on the GUI thread.

Lesson 24's "set Source" clip opens All settings first. Remembering that
choice ended a ``QSettings`` holding a change on the GUI thread, which
rewrites the whole preference file there (a lock file and a temporary file,
each with ``fdatasync``, then a rename): 40-370 ms on the recording disk.
The value is now stored by the preference writer thread, and read back at
once from what is waiting. Counted by thread, not timed.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import prefs, settings_search  # noqa: E402
from spacr.qt.settings_search import (ALL, ESSENTIALS, disclosure_for,  # noqa: E402
                                      forget_disclosure, remember_disclosure)

pytestmark = pytest.mark.qt


@pytest.fixture()
def watched(monkeypatch):
    """Record which thread changes or syncs a QSettings object."""
    seen = {"changed": [], "synced": []}

    class Watched(QSettings):
        def setValue(self, key, value):
            seen["changed"].append(threading.current_thread())
            super().setValue(key, value)

        def sync(self):
            seen["synced"].append(threading.current_thread())
            super().sync()

    monkeypatch.setattr(settings_search, "_settings", lambda: Watched(
        *prefs._store_args(settings_search._QSETTINGS_ORG,
                           settings_search._QSETTINGS_APP)))
    monkeypatch.setattr(prefs, "QSettings", Watched)
    yield seen
    prefs._flush_stores()


def test_the_choice_is_stored_by_the_writer_and_read_back_at_once(watched):
    main = threading.main_thread()
    remember_disclosure("lesson24_probe", ALL)

    assert disclosure_for("lesson24_probe") == ALL
    assert prefs._flush_stores()
    assert main not in watched["changed"]
    assert main not in watched["synced"]
    assert watched["synced"]

    stored = QSettings(*prefs._store_args("spacr", "qt"))
    assert stored.value(f"{settings_search._KEY_DISCLOSURE}/lesson24_probe") == ALL
    assert disclosure_for("lesson24_probe") == ALL


def test_the_newest_choice_wins_and_forgetting_waits_for_the_writer(watched):
    for level in (ALL, ESSENTIALS, ALL, ESSENTIALS):
        remember_disclosure("lesson24_probe", level)
    assert disclosure_for("lesson24_probe") == ESSENTIALS

    remember_disclosure("lesson24_probe", ALL)
    forget_disclosure("lesson24_probe")

    assert disclosure_for("lesson24_probe") == ESSENTIALS
    stored = QSettings(*prefs._store_args("spacr", "qt"))
    assert not stored.contains(f"{settings_search._KEY_DISCLOSURE}/lesson24_probe")


def test_a_value_left_waiting_is_stored_by_the_flush(monkeypatch):
    held = threading.Event()
    real = prefs._write_stores

    def slow(batch):
        held.wait(5)
        real(batch)

    monkeypatch.setattr(prefs, "_write_stores", slow)
    remember_disclosure("lesson24_flush", ALL)
    assert disclosure_for("lesson24_flush") == ALL
    held.set()
    assert prefs._flush_stores()
    stored = QSettings(*prefs._store_args("spacr", "qt"))
    assert stored.value(f"{settings_search._KEY_DISCLOSURE}/lesson24_flush") == ALL
