"""One "Load test data" button, opening a chooser that explains both routes.

Asked for on 2026-09-01: rename the two example buttons, move the control to
the left of Generate annotation database, and put the choice in a popup with a
Load and a Stream button, a description that fills in on hover, and a Close
button. Clicking either must apply the right settings AND reset the source to
the local folder the files were downloaded to.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.annotate import AnnotateScreen
    s = AnnotateScreen()
    qtbot.addWidget(s)
    return s


@pytest.fixture
def chooser(qtbot):
    from spacr.qt.screens.annotate import TestDataChooser
    d = TestDataChooser()
    qtbot.addWidget(d)
    return d


def test_the_chooser_offers_load_and_stream(chooser):
    labels = {b.text() for b in chooser._buttons.values()}
    assert labels == {"Load", "Stream"}, labels


def test_the_description_starts_at_rest_and_fills_in_on_hover(chooser):
    """The pane is the point: the old tooltip had to be hunted for."""
    assert chooser.description_text() == chooser.RESTING_TEXT

    load = chooser._buttons["load"]
    chooser.eventFilter(load, QEvent(QEvent.Enter))
    hovered = chooser.description_text()
    assert hovered != chooser.RESTING_TEXT
    assert "280 MB" in hovered, "the size is the fact users decide on"

    chooser.eventFilter(load, QEvent(QEvent.Leave))
    assert chooser.description_text() == chooser.RESTING_TEXT


def test_each_route_describes_itself_differently(chooser):
    seen = {}
    for key, button in chooser._buttons.items():
        chooser.eventFilter(button, QEvent(QEvent.Enter))
        seen[key] = chooser.description_text()
    assert seen["load"] != seen["stream"]
    assert "390 MB" in seen["stream"]


def test_pressing_a_button_records_the_route_and_closes(chooser):
    chooser._buttons["stream"].click()
    assert chooser.chosen == "stream"


def test_closing_without_choosing_reports_nothing(chooser):
    chooser.reject()
    assert chooser.chosen == ""


def test_the_screen_button_sits_left_of_generate(screen):
    """Asked for explicitly, and it is a layout fact rather than a preference."""
    row = screen._btn_generate.parentWidget().layout()
    order = []
    for i in range(row.count()):
        item = row.itemAt(i)
        w = item.widget() if item is not None else None
        if w is not None:
            order.append(w)
    assert screen._btn_test_data in order and screen._btn_generate in order
    assert order.index(screen._btn_test_data) < order.index(screen._btn_generate)


def test_choosing_load_sets_the_local_source_and_the_load_mode(screen, tmp_path,
                                                               monkeypatch):
    """The reported defect: the source must be LOCAL, not the publisher's."""
    from spacr.qt.screens import annotate as mod

    plate = tmp_path / "plate1"
    (plate / "measurements").mkdir(parents=True)
    (plate / "measurements" / "measurements.db").write_text("")
    monkeypatch.setattr(mod, "example_plate_folder", lambda: plate,
                        raising=False)
    import spacr.qt.hf_download as hf
    monkeypatch.setattr(hf, "example_plate_folder", lambda: plate)

    class Chose:
        chosen = "load"
        def exec(self): return 1

    source = screen._choose_the_test_data(chooser=Chose())

    assert str(plate) in source, source
    assert "carruthers" not in source
    assert screen._settings.crop_source == "load_images"


def test_choosing_stream_sets_the_stream_mode(screen, tmp_path, monkeypatch):
    import numpy as np

    import spacr.qt.hf_download as hf

    plate = tmp_path / "plate1"
    (plate / "merged").mkdir(parents=True)
    np.save(plate / "merged" / "f.npy", np.zeros((2, 2)))
    monkeypatch.setattr(hf, "example_plate_folder", lambda: plate)

    class Chose:
        chosen = "stream"
        def exec(self): return 1

    screen._choose_the_test_data(chooser=Chose())
    assert screen._settings.crop_source == "stream_images"


def test_cancelling_the_chooser_downloads_nothing(screen, monkeypatch):
    called = []

    class Chose:
        chosen = ""
        def exec(self): return 0

    assert screen._choose_the_test_data(
        chooser=Chose(), ask=lambda *a: called.append(a)) == ""
    assert not called
