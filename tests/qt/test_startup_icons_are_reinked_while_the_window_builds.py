"""The dock's and Home's icons are re-inked on two workers while the main window builds.

A cold start decoded, downsized and re-inked about thirty icons one after
another on the GUI thread. They are now started on two workers as the window
starts building; asking for one waits only for that one, and the pixels are
those the GUI thread would have made.
"""
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def cold_icons(tmp_path, monkeypatch):
    from spacr.qt import iconset

    monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(tmp_path / "icons"))
    iconset._themed_array.cache_clear()
    monkeypatch.setattr(iconset, "_PREFETCHED", {})
    yield iconset
    iconset._themed_array.cache_clear()


def test_prefetched_icons_are_the_icons_the_gui_thread_would_make(cold_icons, monkeypatch, tmp_path):
    iconset = cold_icons
    paths = [path for path in iconset.bundled_icon_paths()[:6]]
    workers = []
    real = iconset._themed_array.__wrapped__

    def record(stamp, theme):
        workers.append(threading.current_thread().name)
        return real(stamp, theme)

    monkeypatch.setattr(iconset, "_themed_array", iconset.lru_cache(maxsize=192)(record))
    assert iconset._prefetch_themed_arrays(paths + [None, "art.svg"], "dark") == len(paths)
    assert iconset._prefetch_themed_arrays(paths, "dark") == 0
    prefetched = [iconset.themed_array(path, "dark") for path in paths]
    assert len(workers) == len(paths) and all(name.startswith("spacr-icons") for name in workers)
    assert iconset._PREFETCHED == {}
    monkeypatch.setenv(iconset.ENV_ICON_CACHE, str(tmp_path / "elsewhere"))
    for path, array in zip(paths, prefetched):
        fresh = real(iconset._file_stamp(path), "dark")
        assert array is not None and np.array_equal(array, fresh)


def test_the_main_window_starts_the_startup_icons_first(cold_icons, qtbot, monkeypatch):
    from spacr.qt import app as app_module

    started = []
    monkeypatch.setattr(cold_icons, "_prefetch_themed_arrays", lambda paths, theme=None: started.append(list(paths)))
    paths = app_module._startup_icon_paths()
    assert paths[0].endswith("home.png") and paths[-1].endswith("logo_spacr.png")
    assert len(paths) > 10
    window = app_module.MainWindow()
    qtbot.addWidget(window)
    assert started == [paths]
