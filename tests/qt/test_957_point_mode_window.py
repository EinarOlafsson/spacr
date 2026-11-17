"""Item 957: the point-mode window, driven offscreen with real mouse events.

Left click adds a point on the clicked pixel, right click removes the
nearest, ``U`` marks unsure, ``Ctrl+Z`` undoes, Enter marks the image done
and opens the next. The overlays stay hidden until every image is done, and
the points cannot be changed while one is on screen. The Make Masks button
is an alpha widget, ``MakeMasksPointModeButton``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402


def _crops(tmp_path, n=2, size=64):
    root = tmp_path / "handcheck_crops"
    clean = root / "annotate_clean"
    clean.mkdir(parents=True)
    overlays = root / "overlays"
    overlays.mkdir()
    for i in range(n):
        Image.fromarray(np.full((size, size), 40 * (i + 1), np.uint8)).save(
            clean / f"crop{i:02d}_clean.png")
        Image.fromarray(np.full((size, size, 3), 200, np.uint8)).save(
            overlays / f"crop{i:02d}_spaCR_outlines.png")
    return clean, overlays


@pytest.fixture
def window(qtbot, qt_theme_applied, tmp_path):
    from spacr.point_annotation import build_point_queue
    from spacr.qt.point_annotator import PointAnnotatorDialog

    clean, overlays = _crops(tmp_path)
    dialog = PointAnnotatorDialog(build_point_queue(clean), overlays=[overlays])
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog.view.resetTransform()
    dialog.view.scale(4.0, 4.0)
    dialog.view.centerOn(32, 32)
    return dialog


def _at(window, x, y):
    """The viewport position of the centre of image pixel (x, y)."""
    return window.view.mapFromScene(x + 0.5, y + 0.5)


def _click(qtbot, window, x, y, button=Qt.LeftButton):
    qtbot.mouseClick(window.view.viewport(), button, Qt.NoModifier, _at(window, x, y))


def test_clicks_add_remove_unsure_undo_and_save(qtbot, window):
    _click(qtbot, window, 30, 25)
    _click(qtbot, window, 34, 40)
    assert [(p.x, p.y) for p in window.session.points] == [(30.0, 25.0), (34.0, 40.0)]
    frame = pd.read_csv(window.session.path)
    assert np.round(frame[["Y", "X"]].to_numpy()).astype(int).tolist() == [[25, 30], [40, 34]]
    _click(qtbot, window, 34, 41, Qt.RightButton)
    assert window.session.count == 1
    qtbot.mouseMove(window.view.viewport(), _at(window, 30, 25))
    qtbot.keyClick(window, Qt.Key_U)
    assert window.session.points[0].unsure
    assert "1 points, 1 unsure" in window._count.text()
    qtbot.keyClick(window, Qt.Key_Z, Qt.ControlModifier)
    assert not window.session.points[0].unsure
    assert pd.read_csv(window.session.path)["unsure"].tolist() == [0]


def test_shift_drag_pans_and_adds_nothing(qtbot, window):
    window.view.scale(4.0, 4.0)
    bar = window.view.horizontalScrollBar()
    before = bar.value()
    viewport = window.view.viewport()
    start = QPoint(200, 200)
    qtbot.mousePress(viewport, Qt.LeftButton, Qt.ShiftModifier, start)
    qtbot.mouseMove(viewport, QPoint(150, 200))
    qtbot.mouseRelease(viewport, Qt.LeftButton, Qt.ShiftModifier, QPoint(150, 200))
    assert bar.value() != before
    assert window.session.count == 0


def test_overlays_wait_for_every_image_and_lock_the_points(qtbot, window):
    from spacr.curation_queue import read_status

    assert not window._btn_overlay.isEnabled()
    _click(qtbot, window, 10, 10)
    qtbot.keyClick(window, Qt.Key_Return)
    assert window.index == 1
    assert not window._btn_overlay.isEnabled()
    qtbot.keyClick(window, Qt.Key_Return)
    status = read_status(window.queue.folder)
    assert {row.state for row in status.values()} == {"done"}
    assert status["crop00_clean"].n_objects == 1
    assert window._btn_overlay.isEnabled()
    window._btn_overlay.setChecked(True)
    count = window.session.count
    _click(qtbot, window, 20, 20)
    assert window.session.count == count
    window._btn_overlay.setChecked(False)
    _click(qtbot, window, 20, 20)
    assert window.session.count == count + 1


def test_reopening_resumes_on_the_first_image_not_done(qtbot, qt_theme_applied, tmp_path):
    from spacr.point_annotation import build_point_queue
    from spacr.qt.point_annotator import PointAnnotatorDialog

    clean, overlays = _crops(tmp_path, n=3)
    first = PointAnnotatorDialog(build_point_queue(clean), overlays=[overlays])
    qtbot.addWidget(first)
    first.session.add(3, 4)
    first.mark_done()
    first.close()
    again = PointAnnotatorDialog(build_point_queue(clean), overlays=[overlays])
    qtbot.addWidget(again)
    assert again.index == 1
    again.open_index(0)
    assert [(p.x, p.y) for p in again.session.points] == [(3.0, 4.0)]


def test_the_button_is_alpha_and_the_handover_opens_the_window(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    from spacr import cli_make_masks
    from spacr.point_annotation import build_point_queue
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        widget = MakeMasksScreen()
        qtbot.addWidget(widget)
        assert widget._btn_point_mode.objectName() == "MakeMasksPointModeButton"
        assert widget._btn_point_mode.isHidden() is not shown
    clean, overlays = _crops(tmp_path)
    cli_make_masks._hand_over_points({"queue": build_point_queue(clean),
                                      "overlays": [str(overlays)],
                                      "hide_overlays": True})
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    qtbot.waitUntil(lambda: getattr(widget, "_point_dialog", None) is not None)
    dialog = widget._point_dialog
    assert dialog.queue.folder == clean and dialog.hide_overlays
    assert cli_make_masks._take_points() is None
    dialog.close()
