"""Quitting while a background import is in native code exits cleanly.

The occasional SIGSEGV after a clean quit came from daemon warm threads still
inside an extension import (Matplotlib building its font cache, torch) when
the interpreter finalised: ``pthread_exit`` unwound their C++ frames without
the interpreter lock (gdb under ``PYTHONMALLOC=debug``: a pybind11 destructor
in ``PyFT2Font_init`` on an exiting thread). With the daemon threads this
child crashed on three of four quits made during the font scan.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

CHILD = r"""
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication([])
from spacr.qt import app as appmod
appmod._DATA_LIBRARIES = ("matplotlib.figure",)
thread = appmod._import_the_data_libraries_off_the_gui_thread()
assert thread is not None and not thread.daemon
QTimer.singleShot(int(sys.argv[1]), app.quit)
sys.exit(app.exec())
"""


@pytest.mark.parametrize("quit_after_ms", (600, 800, 1000, 1200))
def test_quitting_while_matplotlib_scans_fonts_exits_zero(tmp_path, quit_after_ms):
    environment = dict(os.environ, PYTHONMALLOC="debug", QT_QPA_PLATFORM="offscreen",
                       MPLCONFIGDIR=str(tmp_path / "mpl"),
                       XDG_CACHE_HOME=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, "-c", CHILD, str(quit_after_ms)],
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]


def test_the_entry_import_is_not_a_daemon(monkeypatch):
    from spacr.qt import bridge

    started = []

    class Recording:
        def __init__(self, *, target, args, name, daemon):
            started.append(daemon)

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(bridge, "_ENTRY_IMPORTS", {})
    monkeypatch.setattr(bridge, "_module_is_imported", lambda name: False)
    monkeypatch.setattr(bridge.threading, "Thread", Recording)
    assert bridge._import_the_entry_off_the_gui_thread("mask") is not None
    assert started == [False]
