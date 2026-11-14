"""The ROI tool's edges: nothing to undo, a preview somebody else removed, a
volume in the stack, and a pen that has already been put down.

Each of these is a path :mod:`tests.qt.test_roi_tool` never walks because it
only ever drives the happy sequence -- click, click, close. What is pinned
here is what the tool does when the sequence is interrupted.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv
from spacr.qt import roi_tool as rt


def _stack(size=48, step=1.0, units="px"):
    stack = LayerStack()
    stack.add_image(np.arange(size * size, dtype=np.uint16).reshape(size, size),
                    name="image",
                    spacing=Spacing.isotropic(2, step, units=units))
    return stack


def _canvas(qtbot, stack, width=160, height=160):
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(width, height)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, canvas, tmp_path):
    panel = rt.RoiPanel(canvas, roi_path=str(tmp_path / "roi.json"))
    qtbot.addWidget(panel)
    return panel


def test_the_roi_layer_takes_its_spacing_from_the_plane_not_the_volume(
        qtbot, qt_theme_applied, tmp_path):
    """An ROI is two-dimensional. A stack that opens with a z-stack has to be
    walked past to find the plane whose pixel size the ROI is measured in."""
    stack = LayerStack()
    stack.add_image(np.zeros((3, 32, 32), dtype=np.uint16), name="volume",
                    spacing=Spacing.isotropic(3, 5.0, units="um"))
    stack.add_image(np.arange(32 * 32, dtype=np.uint16).reshape(32, 32),
                    name="plane", spacing=Spacing.isotropic(2, 0.65,
                                                            units="um"))
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas, tmp_path)

    layer = panel.roi_layer(create=True)

    assert layer.ndim == 2
    assert layer.spacing.scale == (0.65, 0.65), (
        "the ROI took the volume's 5 um step instead of the plane's")
    assert layer.spacing.units == "um"


def test_stopping_the_pen_does_not_take_a_tool_that_replaced_it(
        qtbot, qt_theme_applied, tmp_path):
    """Turning the Draw button off after another tool has taken the canvas
    must forget the pen without evicting the tool now in charge."""
    canvas = _canvas(qtbot, _stack())
    panel = _panel(qtbot, canvas, tmp_path)
    pen = panel.start_drawing()

    successor = rt.RoiPen(panel.roi_layer(), kind="rectangle", parent=panel)
    canvas.set_tool(successor)
    assert canvas.tool is successor and panel.pen is pen

    panel.stop_drawing()
    assert panel.pen is None
    assert canvas.tool is successor, "stop_drawing evicted somebody else's tool"

    # And while the pen IS the canvas's tool, stopping does take it off.
    canvas.set_tool(None)
    panel.start_drawing()
    panel.stop_drawing()
    assert panel.pen is None and canvas.tool is None
