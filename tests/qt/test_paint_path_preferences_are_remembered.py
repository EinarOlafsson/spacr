"""A paint reads its preferences from memory, and looks at the disk once a second.

The tutorial host recorded a 0.37 s stall in a paint at lesson 13 while a
Regression worker held the interpreter: ``get_theme`` reached
``_store_file_state``, which stat'ed the preference file on every read. The
kept store now looks for another process's write at most once a second per
thread, and the getters a paint or a backdrop tick calls answer from memory
until a preference is written. These tests drive a fake clock and count
stats and store reads instead of timing frames.
"""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt

FPS = 60
SECONDS = 10


@pytest.fixture
def clock(monkeypatch):
    """A clock the test moves by hand, and a fresh store and memory."""
    now = [1000.0]
    monkeypatch.setattr(prefs, "_store_clock", lambda: now[0])
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    monkeypatch.setattr(prefs, "_REMEMBERED", {})
    return now


@pytest.fixture
def stats(monkeypatch):
    """Count each look at the preference file."""
    count = [0]
    real = prefs._store_file_state

    def counted(store):
        count[0] += 1
        return real(store)

    monkeypatch.setattr(prefs, "_store_file_state", counted)
    return count


@pytest.fixture
def values(monkeypatch):
    """Count each value read from any ``QSettings``."""
    count = [0]
    real = QSettings.value

    def counted(self, *args, **kwargs):
        count[0] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(QSettings, "value", counted)
    return count


def _paint():
    """What one frame of the backdrop and a themed paint read."""
    return (prefs.get_theme(), prefs.resolve_effective_theme(),
            prefs.get_reduce_motion(), prefs.get_ambient_enabled(),
            prefs.get_font_scale(), prefs.get_language(),
            prefs.get_pane_opacity())


def test_paint_ticks_stat_once_a_second_and_read_nothing(clock, stats,
                                                         values):
    first = _paint()
    stats_before, values_before = stats[0], values[0]
    for _ in range(FPS * SECONDS):
        clock[0] += 1.0 / FPS
        assert _paint() == first
    assert stats[0] - stats_before <= SECONDS + 1
    assert values[0] - values_before == 0


def test_remembered_getters_answer_as_the_store_does(clock):
    remembered = _paint()
    for getter in (prefs.get_theme, prefs.get_reduce_motion,
                   prefs.get_font_scale, prefs.get_language,
                   prefs.get_pane_opacity):
        assert getter() == getter.__wrapped__()
    assert remembered[3] == prefs._stored_ambient_enabled.__wrapped__()


def test_a_write_in_this_process_is_seen_at_once(clock):
    prefs.set_reduce_motion(False)
    assert prefs.get_reduce_motion() is False
    prefs.set_reduce_motion(True)
    assert prefs.get_reduce_motion() is True
    prefs._settings().setValue(prefs._KEY_REDUCE_MOTION, False)
    assert prefs.get_reduce_motion() is False
    prefs.set_font_scale(1.25)
    assert prefs.get_font_scale() == pytest.approx(1.25)


def test_safe_mode_is_not_answered_from_memory(clock, monkeypatch):
    prefs.set_reduce_motion(True)
    assert prefs.get_reduce_motion() is True
    monkeypatch.setattr(prefs, "_SAFE_MODE", True)
    assert prefs.get_reduce_motion() is False


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="the native store is an INI file only on Linux")
def test_another_process_write_is_seen_within_a_second(clock):
    prefs.set_reduce_motion(False)
    prefs._settings().sync()
    assert prefs.get_reduce_motion() is False
    path = prefs._settings().fileName()
    subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from PySide6.QtCore import QSettings\n"
         "store = QSettings(sys.argv[1], QSettings.IniFormat)\n"
         "store.setValue('prefs/reduce_motion', True)\n"
         "store.sync()\n",
         path],
        check=True, timeout=60)
    clock[0] += 0.5
    assert prefs.get_reduce_motion() is False
    clock[0] += prefs._STORE_WATCH_SECONDS
    assert prefs.get_reduce_motion() is True


def test_a_write_by_an_unrelated_store_is_seen_by_the_next_look(clock):
    """Only the preference store counts its writes; Qt's type is left alone.

    Another ``QSettings`` on the same file in this process is seen once it
    has reached the file, at the next once-a-second look.
    """
    prefs.set_reduce_motion(True)
    assert prefs.get_reduce_motion() is True
    other = QSettings(*prefs._store_args(prefs._ORG, prefs._APP))
    other.setValue(prefs._KEY_REDUCE_MOTION, False)
    other.sync()
    clock[0] += prefs._STORE_WATCH_SECONDS
    assert prefs.get_reduce_motion() is False
