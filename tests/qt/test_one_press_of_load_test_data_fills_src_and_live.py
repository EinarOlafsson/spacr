"""One press of Load test data fills `src` and Live shows it (item 514).

Reported 2026-09-25: in Mask generation one press of "Load test data" left
the Source field showing `<src>` and Live showed nothing; a second press
worked.

THE ROOT CAUSE WAS A RETIRED SCREEN. The shipped settings pack changes the
form's shape (object channels), so `apply_settings_dict` has the window build
a new screen and destroy the one the button was on. Every route then wrote
`src` into the field it had looked up BEFORE the load -- the retired screen's
-- so the screen on view kept the pack's `<src>` and its Live panel was never
told. The second press rebuilt nothing, so its field was the live one.

These tests drive a real window, press once with the data already cached in
a temporary folder (no network, nothing under the real home), and assert on
the screen the window shows AFTERWARDS: `src` holds the real path and the
Live panel was handed it.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: What the published Mask pack carries that matters here: the template
#: token for `src`, and object channels that reshape the form.
MASK_PACK = [("src", "<src>"), ("channels", "[0, 1, 2, 3]"),
             ("cell_channel", "1"), ("nucleus_channel", "0"),
             ("pathogen_channel", "2")]


def _write_pack(folder, name, rows):
    """Write ``rows`` as a shipped settings CSV under ``folder/settings``."""
    settings = folder / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    with (settings / name).open("w", newline="") as handle:
        csv.writer(handle).writerows([("Key", "Value")] + list(rows))


@pytest.fixture
def loads(monkeypatch):
    """Record every source a preview panel is asked to load, without loading."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    from spacr.qt.widgets.plaque_preview import PlaquePreviewPanel

    seen = []

    def record(_panel, source, *_a, **_k):
        seen.append(str(source))
        return True

    monkeypatch.setattr(LivePreviewPanel, "load_source_async", record)
    monkeypatch.setattr(PlaquePreviewPanel, "load_source_async", record)
    monkeypatch.setattr(MeasurePreviewPanel, "load_array_async", record)
    return seen


def _open(qtbot, monkeypatch, app_key):
    """A real window on ``app_key``, with its rebuilds counted."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    assert window.open_module(app_key) == app_key
    qtbot.wait(20)
    rebuilds = []
    real = window.rebuild_app_screen

    def counted(key, values=None):
        rebuilds.append(key)
        return real(key, values)

    monkeypatch.setattr(window, "rebuild_app_screen", counted)
    return window, rebuilds


def _src(window, app_key):
    """What the Source field on the screen now on view holds."""
    return window._screens[app_key]._settings_model._widgets["src"].text()


def _live(window, app_key, on):
    """Switch Live on or off on the screen now on view."""
    window._screens[app_key]._preview_switch.setChecked(on)


def _mask_plate(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    (plate / "plate1_E01_T0001F001L01A01Z01C01.tif").write_bytes(b"II*\0")
    _write_pack(plate, "gen_mask_settings.csv", MASK_PACK)
    return plate


@pytest.mark.parametrize("live_first", [True, False],
                         ids=["live-already-on", "live-turned-on-after"])
def test_mask_one_press(qtbot, monkeypatch, tmp_path, loads, live_first):
    from spacr.qt.screens.app_screen import AppScreen

    plate = _mask_plate(tmp_path)
    monkeypatch.setattr(AppScreen, "example_images_destination",
                        lambda self: plate)
    window, rebuilds = _open(qtbot, monkeypatch, "mask")
    if live_first:
        _live(window, "mask", True)
    loads.clear()

    window._screens["mask"]._example_images_button.click()

    assert rebuilds, "the pack must reshape the form for this to test 514"
    assert _src(window, "mask") == str(plate)
    screen = window._screens["mask"]
    assert screen._preview_switch.isChecked() is live_first
    if not live_first:
        _live(window, "mask", True)
    assert str(plate) in loads


def test_measure_one_press(qtbot, monkeypatch, tmp_path, loads):
    from spacr.qt.screens.app_screen import AppScreen

    plate = tmp_path / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "plate1_E01_1.npy", np.zeros((8, 8, 2), np.uint16))
    _write_pack(plate, "crop_measure_settings.csv", [("src", "<src>")])
    monkeypatch.setattr(AppScreen, "measure_example_destination",
                        lambda self: plate)
    window, _rebuilds = _open(qtbot, monkeypatch, "measure")
    _live(window, "measure", True)
    loads.clear()

    window._screens["measure"]._measure_example_button.click()

    assert _src(window, "measure") == str(plate)
    assert any(seen.startswith(str(plate)) for seen in loads)


def test_classify_one_press(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets import test_data_chooser

    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    (plate / "measurements" / "measurements.db").write_bytes(b"")
    _write_pack(plate, "classify_settings.csv", [("src", "<src>")])
    monkeypatch.setattr(AppScreen, "annotate_example_destination",
                        lambda self: plate)

    class _Chooser:
        chosen = "crops"

        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(test_data_chooser, "TestDataChooser", _Chooser)
    window, _rebuilds = _open(qtbot, monkeypatch, "classify_merged")

    window._screens["classify_merged"]._annotate_example_button.click()

    assert _src(window, "classify_merged") == str(plate)


def test_replication_one_press_after_the_download(qtbot, monkeypatch,
                                                  tmp_path):
    from spacr.qt.assay_examples import load_the_assay_example

    folder = tmp_path / "replication"
    folder.mkdir()
    _write_pack(folder, "replication_settings.csv", [("src", "<src>")])
    window, _rebuilds = _open(qtbot, monkeypatch, "replication")

    def finished(_parent, _dest, on_done):
        on_done(object(), "")

    load_the_assay_example(window._screens["replication"], ask=finished,
                           folder=folder)

    assert _src(window, "replication") == str(folder)


def test_plaques_live_shows_the_test_data_at_once(qtbot, monkeypatch,
                                                  tmp_path, loads):
    fields = tmp_path / "plaque_fields"
    fields.mkdir()
    window, _rebuilds = _open(qtbot, monkeypatch, "analyze_plaques")
    _live(window, "analyze_plaques", True)
    loads.clear()

    assert window._screens["analyze_plaques"].point_src_at(fields)

    assert _src(window, "analyze_plaques") == str(fields)
    assert str(fields) in loads
