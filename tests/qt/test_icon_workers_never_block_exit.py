"""The icon prefetch pool must not outlive its work or stall interpreter exit."""
import os
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("PySide6")


def _icon_threads():
    return [t for t in threading.enumerate() if t.name.startswith("spacr-icons") and t.is_alive()]


def _wait_gone(timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and _icon_threads():
        time.sleep(0.05)
    return not _icon_threads()


def test_workers_exit_once_the_prefetch_is_done(tmp_path, monkeypatch):
    from spacr.qt import iconset

    monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(tmp_path / "icons"))
    iconset._themed_array.cache_clear()
    monkeypatch.setattr(iconset, "_PREFETCHED", {})
    assert iconset._prefetch_themed_arrays(iconset.bundled_icon_paths()[:4], "dark") == 4
    assert _wait_gone()
    assert iconset._PREFETCH_POOL is None


def test_closing_the_main_window_leaves_no_icon_thread(qtbot):
    from spacr.qt import app as app_module

    window = app_module.MainWindow()
    qtbot.addWidget(window)
    window.close()
    assert _wait_gone()


def test_a_process_that_builds_and_closes_the_window_exits(tmp_path):
    code = (
        "from PySide6.QtWidgets import QApplication\n"
        "from spacr.qt import app\n"
        "qa = QApplication.instance() or QApplication([])\n"
        "w = app.MainWindow()\n"
        "w.close()\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               SPACR_HOME=str(tmp_path))
    subprocess.run([sys.executable, "-c", code], env=env, timeout=120, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
