"""Mask generation maps PyTorch's libraries before its Overview opens.

The Overview's first thumbnail job mapped them (0.3-0.4 s with the dynamic
linker's lock held) while the plate painted: the thumbnails started late and a
library the GUI thread loaded meanwhile waited with the interpreter lock held.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

pytest.importorskip("PySide6")


def test_mask_open_maps_the_torch_libraries_once_on_a_worker(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import app as app_module
    from spacr.qt import gil_priority
    from spacr.qt.app import MainWindow

    calls = []

    def fake_open():
        calls.append((threading.current_thread() is threading.main_thread(),
                      app_module.HEAVY_IMPORT_LOCK.locked()))
        return True

    monkeypatch.setattr(gil_priority, "_open_torch_libraries", fake_open)
    monkeypatch.setattr(app_module, "_TORCH_LIBRARIES_STARTED", False)
    monkeypatch.setattr(app_module, "_TORCH_LIBRARIES_AFTER_S", 0.05)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.open_module("mask")
    qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    assert calls == [(False, True)]
    assert app_module._map_the_torch_libraries_off_the_gui_thread() is None
    assert len(calls) == 1


def test_an_overview_load_postpones_the_mapping(monkeypatch):
    from spacr.qt import app as app_module
    from spacr.qt import gil_priority

    class Window:
        _closing = False

    owner = Window()
    started = []
    later = []
    monkeypatch.setattr(app_module, "_after", lambda msec, window, callback: later.append(msec))
    gil_priority.hold_background_imports(owner)
    try:
        assert app_module._start_unless_held(Window(), lambda: started.append(1)) is None
    finally:
        gil_priority.release_background_imports(owner)
    assert started == [] and later == [app_module._HELD_IMPORT_RETRY_MS]


def test_the_mapping_opens_no_gpu_device_without_cuda_devices():
    pytest.importorskip("torch")
    if not sys.platform.startswith("linux"):
        pytest.skip("the mapping is Linux only")
    code = (
        "import os\n"
        "from spacr.qt import app\n"
        "thread = app._map_the_torch_libraries_off_the_gui_thread()\n"
        "thread.join(60)\n"
        "import sys\n"
        "assert 'torch' not in sys.modules\n"
        "maps = open('/proc/self/maps').read()\n"
        "handles = []\n"
        "for fd in os.listdir('/proc/self/fd'):\n"
        "    try:\n"
        "        handles.append(os.readlink('/proc/self/fd/' + fd))\n"
        "    except OSError:\n"
        "        pass\n"
        "print('libtorch_cpu' in maps, [h for h in handles if '/dev/nvidia' in h])\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True []"
