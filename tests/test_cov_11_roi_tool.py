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


from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv
from spacr.qt import roi_tool as rt


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


# ---------------------------------------------------------------------------
# Events the pen does not want
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Clearing
# ---------------------------------------------------------------------------

def test_clearing_before_anything_is_drawn_removes_nothing(qtbot, tmp_path):
    """No ROI layer means no ROIs, and no layer created just to empty it."""
    panel = _panel(qtbot, tmp_path)

    assert panel.clear_rois() == 0
    assert panel.roi_layer(create=False) is None


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
