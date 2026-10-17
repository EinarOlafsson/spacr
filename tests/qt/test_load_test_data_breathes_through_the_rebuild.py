"""Mask "Load test data" keeps the event loop running while the screen is rebuilt.

The example plate ships a settings pack that names the object channels, which
changes the shape of the Mask form, so applying it rebuilds the whole screen.
That rebuild ran in one piece: one GUI-thread block of about 0.4 s on the
tutorial capture host, with the backdrop stopped (tutorial 07 recorded it as
an app freeze in every take). It now takes the breaths a module open takes,
so the longest block is one step of the rebuild, not all of them together.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

PACK = """Key,Value
channels,"[0, 1, 2, 3]"
nucleus_channel,0
cell_channel,1
pathogen_channel,2
"""


@pytest.fixture
def mask_window(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    window.open_module("mask")
    qtbot.wait(20)
    screen = window._screens.get("mask")
    assert screen is not None, "Mask did not open"
    return window, screen


@pytest.fixture
def example_plate(tmp_path):
    plate = tmp_path / "plate1"
    (plate / "settings").mkdir(parents=True)
    (plate / "plate1_E01_T0001F001L01A01Z01C01.tif").write_bytes(b"II*\x00")
    (plate / "settings" / "gen_masks_settings.csv").write_text(PACK)
    return plate


def _ticks():
    """A 1 ms timer: its timestamps show when the event loop got to run."""
    from PySide6.QtCore import QTimer

    stamps = []
    timer = QTimer()
    timer.setInterval(1)
    timer.timeout.connect(lambda: stamps.append(time.perf_counter()))
    timer.start()
    return timer, stamps


def test_load_test_data_rebuild_is_not_one_block(mask_window, example_plate,
                                                 monkeypatch, qtbot):
    from spacr.qt import screens

    window, screen = mask_window
    monkeypatch.setattr(type(screen), "example_images_destination",
                        lambda self: example_plate)
    timer, stamps = _ticks()
    qtbot.wait(30)
    started = time.perf_counter()
    screen.load_the_example_images()
    ended = time.perf_counter()
    timer.stop()

    fresh = window._screens["mask"]
    assert fresh is not screen, "the pack did not change the form's shape"
    assert fresh._settings_model.collect().get("nucleus_channel") == 0
    inside = [started] + [t for t in stamps if started < t < ended] + [ended]
    longest = max(b - a for a, b in zip(inside, inside[1:]))
    assert len(inside) > 3, "the event loop never ran during the rebuild"
    assert longest < 0.6 * (ended - started), (
        f"one block of {longest * 1000:.0f} ms in a "
        f"{(ended - started) * 1000:.0f} ms load")
    assert not screens._a_window_is_opening_a_screen
    assert not window._opening_a_screen


def test_a_rebuild_inside_an_open_leaves_the_open_breathing(mask_window):
    from spacr.qt import screens

    window, _screen = mask_window
    window._opening_a_screen = True
    screens._start_breathing_while_a_window_opens(time.perf_counter())
    try:
        window.rebuild_app_screen("mask", {"nucleus_channel": 0})
        assert screens._a_window_is_opening_a_screen
        assert window._opening_a_screen
    finally:
        window._opening_a_screen = False
        screens._stop_breathing_while_a_window_opens()
