"""The ROI pen ignores what it was not given, and the panel stays consistent.

Drawing is a modal thing bolted onto a canvas that has other jobs. Every
branch here is the pen declining to act -- a mouse button that is not the
draw or undo button, a key that belongs to the canvas underneath -- so that
declining REPORTS itself, letting the event fall through to whatever else
wanted it. The panel side is the same idea: clearing ROIs while half of one
is drawn must not leave the pen holding vertices for a layer that is empty.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.layers import LayerStack, ShapesLayer, Spacing
from spacr.qt import layer_viewer as lv
from spacr.qt import roi_tool as rt


class _Event:
    """A mouse or key event with only the accessor the tool reads."""

    def __init__(self, *, button=None, key=None):
        self._button = button
        self._key = key

    def button(self):
        return self._button

    def key(self):
        return self._key


def _stack(size=64):
    stack = LayerStack()
    stack.add_image(np.arange(size * size, dtype=np.uint16).reshape(size, size),
                    name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    return stack


def _canvas(qtbot, stack=None):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, tmp_path, canvas=None):
    panel = rt.RoiPanel(canvas if canvas is not None else _canvas(qtbot),
                        roi_path=str(tmp_path / "roi.json"))
    qtbot.addWidget(panel)
    return panel


# ---------------------------------------------------------------------------
# The pen says what it is
# ---------------------------------------------------------------------------

def test_the_pen_reports_the_shape_it_is_drawing(qtbot, tmp_path):
    """The kind decides how many clicks close the shape, so it is readable.

    A rectangle closes itself on the second click; a polygon does not, and a
    caller has to be able to tell which pen it is holding.
    """
    panel = _panel(qtbot, tmp_path)
    panel.kind_combo.setCurrentText("rectangle")
    pen = panel.start_drawing()

    assert pen.kind == "rectangle"
    assert pen.layer.name == rt.ROI_LAYER_NAME


# ---------------------------------------------------------------------------
# Events the pen does not want
# ---------------------------------------------------------------------------

def test_a_middle_click_is_not_a_vertex(qtbot, tmp_path):
    """Only the left button places a point; the pen declines the rest.

    Returning False is how the canvas underneath still gets its pan, so
    swallowing the event would break middle-drag while the pen is on.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, tmp_path, canvas)
    pen = panel.start_drawing()

    handled = pen.press(canvas, {"y": 4.0, "x": 4.0},
                        _Event(button=Qt.MiddleButton))

    assert handled is False
    assert len(pen.pending) == 0


def test_a_right_click_takes_the_last_vertex_back(qtbot, tmp_path):
    """The other half of the same dispatch, so False means something."""
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, tmp_path, canvas)
    pen = panel.start_drawing()
    pen.add_world({"y": 4.0, "x": 4.0})
    pen.add_world({"y": 4.0, "x": 40.0})

    handled = pen.press(canvas, {"y": 0.0, "x": 0.0},
                        _Event(button=Qt.RightButton))

    assert handled is True
    assert len(pen.pending) == 1


def test_a_key_the_pen_does_not_use_falls_through(qtbot, tmp_path):
    """Escape, Return and Backspace are the pen's; everything else is not.

    Claiming an unhandled key would take the canvas's own arrow-key panning
    away for as long as drawing is switched on.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, tmp_path, canvas)
    pen = panel.start_drawing()
    pen.add_world({"y": 4.0, "x": 4.0})

    assert pen.key(canvas, _Event(key=Qt.Key_Left)) is False
    assert len(pen.pending) == 1


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------

def test_clearing_before_anything_is_drawn_removes_nothing(qtbot, tmp_path):
    """No ROI layer means no ROIs, and no layer created just to empty it."""
    panel = _panel(qtbot, tmp_path)

    assert panel.clear_rois() == 0
    assert panel.roi_layer(create=False) is None


def test_clearing_while_drawing_abandons_the_half_drawn_outline(qtbot,
                                                                tmp_path):
    """The pen's pending vertices go with the shapes it would have joined.

    Left in place they would close into a shape after the user pressed
    Clear, which reads as the button not having worked.
    """
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, tmp_path, canvas)
    pen = panel.start_drawing()
    pen.add_world({"y": 4.0, "x": 4.0})
    pen.add_world({"y": 4.0, "x": 40.0})
    pen.add_world({"y": 40.0, "x": 40.0})
    pen.close_shape()
    pen.add_world({"y": 50.0, "x": 50.0})

    removed = panel.clear_rois()

    assert removed == 1
    assert len(pen.pending) == 0
    assert len(panel.roi_layer(create=False)) == 0


# ---------------------------------------------------------------------------
# Choosing where the ROI is written
# ---------------------------------------------------------------------------

def test_choosing_a_file_moves_where_the_roi_will_be_written(qtbot, tmp_path,
                                                             monkeypatch):
    """A worker reads the ROI from this path, so the panel must follow it."""
    panel = _panel(qtbot, tmp_path)
    chosen = str(tmp_path / "elsewhere" / "picked.json")
    monkeypatch.setattr(rt.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (chosen, "")))

    panel._on_choose_path()

    assert panel.roi_path == os.path.abspath(chosen)


def test_cancelling_the_file_dialog_keeps_the_path_it_had(qtbot, tmp_path,
                                                          monkeypatch):
    """A cancelled dialog must not blank the destination."""
    panel = _panel(qtbot, tmp_path)
    before = panel.roi_path
    monkeypatch.setattr(rt.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    panel._on_choose_path()

    assert panel.roi_path == before
