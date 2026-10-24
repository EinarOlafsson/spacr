"""The recorder leaves no live sampler timers behind: 20 start/end_scene cycles (lesson 14 segfault)."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('SPACR_CAPTURE_HOST_SIZE', '2560x1440')
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import capture_host_mode  # noqa: E402


class _Pump:
    def run(self, seconds=0):
        QApplication.processEvents()

    def once(self):
        QApplication.processEvents()


class _Adapter:
    def __init__(self):
        self.pump = _Pump()
        self.bypasses = []

    def call(self, op, *args, **kwargs):
        return [] if op == 'trace_since' else {'index': 0} if op == 'trace_mark' else {}


def test_twenty_scenes_leave_no_sampler_timers(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.show()
    runtime = capture_host_mode.HostRuntime(window, _Adapter(), stage=tmp_path)
    for index in range(20):
        runtime.start_scene(tmp_path, f'scene_{index:02d}', 0.0)
        assert len(window.findChildren(QTimer)) == 1
        runtime.end_scene(at_least=0.0)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert window.findChildren(QTimer) == [], 'a sampler timer outlived its scene'
    assert all('sampler' not in scene for scene in runtime.pending)
    window.close()
