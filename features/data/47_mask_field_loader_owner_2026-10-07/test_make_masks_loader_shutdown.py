"""A timed-out field decoder outlives its closing screen safely."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np

from spacr.qt import bridge
from spacr.qt.screens import make_masks as module


def test_timed_out_decoder_drops_pending_load_and_cannot_publish_after_close(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = tmp_path / 'images'
    folder.mkdir()
    module.engine.imageio.imwrite(folder / 'one.tif', np.ones((10, 10), np.uint16))
    module.engine.imageio.imwrite(folder / 'two.tif', np.full((10, 10), 2, np.uint16))
    entered, release = threading.Event(), threading.Event()
    original_load, original_drain = module.engine.load_image_and_mask, bridge.drain_thread
    calls = []

    def blocked(*args, **kwargs):
        calls.append(args)
        entered.set()
        assert release.wait(10)
        return original_load(*args, **kwargs)

    def immediate_timeout(worker, *args, timeout_ms):
        assert timeout_ms == 5000
        return original_drain(worker, *args, timeout_ms=0)

    monkeypatch.setattr(module.engine, 'load_image_and_mask', blocked)
    monkeypatch.setattr(module.MakeMasksScreen, '_should_background_load',
                        staticmethod(lambda _path: True))
    monkeypatch.setattr(bridge, 'drain_thread', immediate_timeout)
    screen = module.MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(entered.is_set)
    worker = screen._load_worker
    try:
        screen._current_index = 1
        screen._load_current()
        assert screen._pending_load is not None
        screen.close()
        assert worker.isRunning() and worker.parent() is None
        assert screen._load_worker is None and screen._pending_load is None
        assert not screen._loading and screen._canvas.image is None
        assert bridge.parked_thread_count() >= 1
        status = screen._status_label.text()
        release.set()
        assert worker.wait(2000)
        qtbot.wait(10)
        assert len(calls) == 1 and screen._canvas.image is None
        assert screen._status_label.text() == status
        assert screen._load_worker is None and not screen._loading
    finally:
        release.set()
        assert worker.wait(2000)
        bridge.prune_parked_threads()
        worker.deleteLater()


def test_completed_decoder_keeps_normal_parent_ownership(
        qtbot, qt_theme_applied, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return np.ones((10, 10), np.uint16), np.zeros((10, 10), np.uint16)

    monkeypatch.setattr(module.engine, 'load_image_and_mask', blocked)
    screen = module.MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._start_background_load('/controlled', 'one.tif', screen._load_token)
    qtbot.waitUntil(entered.is_set)
    worker = screen._load_worker
    try:
        release.set()
        screen.close()
        assert not worker.isRunning() and worker.parent() is screen
        assert screen._load_worker is None and screen._pending_load is None
        qtbot.wait(10)
        assert screen._canvas.image is None and not screen._loading
    finally:
        release.set()
        assert worker.wait(2000)


def test_native_screen_deletion_preserves_decoder_after_real_drain_timeout(tmp_path):
    code = textwrap.dedent('''
        import threading
        import numpy as np
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from spacr.qt import bridge
        from spacr.qt.screens import make_masks as module
        try:
            import resource
        except ImportError:
            pass
        else:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        app = QApplication([])
        entered, release = threading.Event(), threading.Event()
        def blocked(*_args, **_kwargs):
            entered.set()
            assert release.wait(25)
            return np.ones((8, 8), np.uint16), np.zeros((8, 8), np.uint16)
        module.engine.load_image_and_mask = blocked
        screen = module.MakeMasksScreen()
        screen._start_background_load('/controlled', 'one.tif', screen._load_token)
        assert entered.wait(2)
        worker = screen._load_worker
        screen.close()
        assert worker.isRunning() and worker.parent() is None
        assert bridge.parked_thread_count() == 1
        screen.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(screen) and isValid(worker) and worker.isRunning()
        release.set()
        assert worker.wait(2000)
        assert bridge.prune_parked_threads() == 0
        worker.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(worker)
        print('native-screen-deleted-decoder-finished')
    ''')
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen',
                       PYTHONPATH=str(root), XDG_CONFIG_HOME=str(tmp_path / 'child-config'))
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                            cwd=root, env=environment, capture_output=True,
                            text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'native-screen-deleted-decoder-finished' in result.stdout
    assert 'QThread: Destroyed while thread' not in result.stderr
