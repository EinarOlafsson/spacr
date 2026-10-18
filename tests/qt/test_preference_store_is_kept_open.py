"""Reading a preference on a frame or a tick opens no ``QSettings``.

The backdrop dipped below 30 fps during tutorial captures, and the late
frames sat in `spacr.qt.preferences._settings`: it opened a new
``QSettings`` for every read, which takes Qt's process-wide settings lock
and checks the file on disk, and the backdrop tick, the dock's paint and
every theme lookup each read a preference. The store is now kept open per
thread. These tests count openings instead of timing frames, and check that
a kept store still sees every write: its own, another process's, and a
store moved with ``QSettings.setPath``.
"""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture
def openings(monkeypatch):
    """Count each ``QSettings`` the preference store opens."""
    count = [0]

    class Counted(QSettings):
        def __init__(self, *args):
            count[0] += 1
            super().__init__(*args)

    monkeypatch.setattr(prefs, "QSettings", Counted)
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    return count


@pytest.fixture
def reads(monkeypatch):
    """Count each read of the preference store, opened or kept."""
    count = [0]
    real = prefs._real_store

    def counted():
        count[0] += 1
        return real()

    monkeypatch.setattr(prefs, "_real_store", counted)
    return count


def test_hot_preference_reads_open_the_store_once(openings):
    prefs.get_reduce_motion()
    prefs.get_ambient_enabled()
    prefs.resolve_effective_theme()
    assert openings[0] == 1
    for _ in range(100):
        prefs.get_reduce_motion()
        prefs.get_ambient_enabled()
        prefs.resolve_effective_theme()
    assert openings[0] == 1


def test_a_backdrop_tick_and_repaint_open_no_store(qtbot, openings, reads):
    from spacr.qt.widgets.ambient import AmbientWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    widget = AmbientWidget(host, theme="data_art_impulse_lens")
    host.show()
    widget._on_tick()
    widget.repaint()
    before, read_before = openings[0], reads[0]
    for _ in range(20):
        assert widget._should_run() == widget._should_run()
        widget._on_tick()
        widget.repaint()
    assert reads[0] - read_before >= 20
    assert openings[0] == before


def test_painting_the_dock_edge_opens_no_store(qtbot, openings, reads):
    from spacr.qt.widgets.dock import Dock, DockEdge

    host = QWidget()
    qtbot.addWidget(host)
    row = QHBoxLayout(host)
    slot = QWidget(host)
    dock = Dock([("__home__", "Home", "Home", "")])
    QVBoxLayout(slot).addWidget(dock)
    edge = DockEdge(dock, host)
    row.addWidget(slot)
    row.addWidget(edge)
    host.show()
    host.resize(400, 300)
    edge.grab()
    before, read_before = openings[0], reads[0]
    for _ in range(20):
        edge.grab()
    assert reads[0] - read_before >= 20
    assert openings[0] == before


def test_a_kept_store_sees_its_own_writes(openings):
    prefs.set_reduce_motion(True)
    assert prefs.get_reduce_motion() is True
    prefs.set_reduce_motion(False)
    assert prefs.get_reduce_motion() is False


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="the native store is an INI file only on Linux")
def test_a_kept_store_sees_another_process_write(openings):
    prefs.set_reduce_motion(False)
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
    assert prefs.get_reduce_motion() is True


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="setPath moves the native store only on Linux")
def test_a_moved_store_is_reopened(tmp_path, openings):
    first = prefs._settings().fileName()
    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path))
    moved = prefs._settings().fileName()
    assert moved != first
    assert moved.startswith(str(tmp_path))
