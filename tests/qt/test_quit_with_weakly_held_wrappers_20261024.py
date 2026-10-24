"""Quitting leaves no weak reference to a freed Qt wrapper.

Shiboken frees a wrapper without clearing the weak references to it when
the wrapper is deallocated during interpreter finalization, so a
``WeakSet`` or a watcher list torn down after it reads freed memory. In an
ordinary run that is a rare SIGSEGV after the event loop has returned (the
embedded-overview child of ``test_overview_expanded_workflow.py`` ended
with rc -11 in CI and 139 locally); under ``PYTHONMALLOC=debug`` the freed
memory is poisoned and the same exit crashes every time, so each child here
runs with it. :func:`spacr.qt._keep_weakly_held_qt_wrappers_to_the_end`
keeps such wrappers alive through finalization.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

RUNS = 4

_HOLDERS = r'''
import sys, weakref
import spacr.qt
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QWidget
import json
app = QApplication([])
class Watcher(QObject):
    pass
watchers = [Watcher() for _ in range(64)]
widgets = [QWidget() for _ in range(64)]
json._spacr_exit_refs = [weakref.ref(item) for item in watchers + widgets]
json._spacr_exit_set = weakref.WeakSet(watchers + widgets)
class Hub(QObject):
    pass
hub = Hub(app)
hub.entries = [[weakref.ref(item), True] for item in watchers]
app.spacr_exit_hub = hub
sys.exit(0)
'''

_APPLICATION = r'''
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from spacr.qt.app import MainWindow
from spacr.qt import preferences
from spacr.qt.first_run import mark_tour_seen
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
mark_tour_seen()
preferences.apply_preferences_to_app(app)
window = MainWindow()
window.resize(1180, 820)
window.show()
def finish():
    window.close()
    app.exit(0)
QTimer.singleShot(0, lambda: window.open_module("mask"))
QTimer.singleShot(1500, finish)
sys.exit(app.exec())
'''


def _run(code, tmp_path, index):
    """Run ``code`` in a fresh interpreter with poisoned freed memory."""
    home = tmp_path / f"run{index}"
    environment = dict(
        os.environ, PYTHONMALLOC="debug", QT_QPA_PLATFORM="offscreen",
        CUDA_VISIBLE_DEVICES="", XDG_CONFIG_HOME=str(home / "config"),
        XDG_CACHE_HOME=str(home / "cache"), SPACR_HOME=str(home / "profile"),
        SPACR_NO_NETWORK="1", SPACR_PREWARM="0", SPACR_ISOLATED_PROFILE="1")
    return subprocess.run([sys.executable, "-c", code], env=environment,
                          capture_output=True, text=True, timeout=240)


def test_weakly_held_wrappers_do_not_crash_the_exit(tmp_path):
    """Weak references in a module, a WeakSet and an application attribute."""
    for index in range(RUNS):
        result = _run(_HOLDERS, tmp_path, index)
        assert result.returncode == 0, (
            f"run {index + 1} of {RUNS} ended with {result.returncode}:\n"
            f"{result.stderr[-4000:]}")


def test_the_application_exits_cleanly_after_opening_mask(tmp_path):
    """The real window, with Mask open, quits with status 0 every time."""
    for index in range(2):
        result = _run(_APPLICATION, tmp_path, index)
        assert result.returncode == 0, (
            f"run {index + 1} of 2 ended with {result.returncode}:\n"
            f"{result.stderr[-4000:]}")
