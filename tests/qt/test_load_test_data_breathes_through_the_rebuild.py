"""Mask "Load test data" keeps the event loop running.

The example plate ships a settings pack that names the object channels. That
rebuilt the whole Mask screen in one GUI-thread block of about 0.4 s on the
tutorial capture host, with the backdrop stopped (tutorial 07 recorded it as
an app freeze in every take). Item 356, extended 2026-10-17: a channel in a
pack is shown in place, so the load no longer rebuilds; a rebuild that is
still needed (a new organelle count) takes the breaths of a module open.
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


def test_load_test_data_shows_the_channels_in_place(mask_window, example_plate,
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

    assert window._screens["mask"] is screen, "a channel pack rebuilt the screen"
    model = screen._settings_model
    assert model.collect().get("nucleus_channel") == 0
    assert model.collect().get("pathogen_channel") == 2
    assert screen._form_shape_on_screen == screen._form_shape()
    assert not screens._a_window_is_opening_a_screen
    assert not window._opening_a_screen


def test_a_new_organelle_count_rebuilds_with_breaths(mask_window, qtbot):
    from spacr.organelle_types import NUMBER_OF_ORGANELLES

    window, screen = mask_window
    count = screen._settings_model.collect().get(NUMBER_OF_ORGANELLES) or 0
    timer, stamps = _ticks()
    qtbot.wait(30)
    started = time.perf_counter()
    screen.apply_settings_dict({NUMBER_OF_ORGANELLES: int(count) + 1})
    ended = time.perf_counter()
    timer.stop()

    assert window._screens["mask"] is not screen, "the count did not rebuild"
    inside = [started] + [t for t in stamps if started < t < ended] + [ended]
    longest = max(b - a for a, b in zip(inside, inside[1:]))
    assert len(inside) > 3, "the event loop never ran during the rebuild"
    assert longest < 0.6 * (ended - started), (
        f"one block of {longest * 1000:.0f} ms in a "
        f"{(ended - started) * 1000:.0f} ms rebuild")


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
