"""A layer canvas composites its stack once per change, not once per paint.

Lesson 42 (Make Masks, Curate) painted at 6 to 8 frames a second: the window
repaints everything over its animated backdrop on every backdrop frame, and
each paint of the Curate canvas composited the whole stack again, about
150 ms at 1920x1080. These tests count composites rather than timing them.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.layers import LayerStack, Spacing
from spacr.qt import curation_tool as ct
from spacr.qt import layer_viewer as lv

pytestmark = pytest.mark.qt


def _stack(size=64):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, 1.0, units="px")
    stack.add_image(np.arange(size * size, dtype=np.uint16).reshape(size, size),
                    name="image", spacing=spacing)
    labels = np.zeros((size, size), np.int64)
    labels[10:20, 10:20] = 3
    stack.add_labels(labels, name="mask", spacing=spacing)
    return stack


@pytest.fixture()
def counted(qtbot, monkeypatch):
    stack = _stack()
    calls = []
    original = stack.render_uint8
    monkeypatch.setattr(stack, "render_uint8",
                        lambda canvas: calls.append(canvas) or original(canvas))
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.show()
    canvas.repaint()
    return canvas, stack, calls


def test_repaints_with_nothing_changed_composite_nothing(counted):
    canvas, _stack_, calls = counted
    assert len(calls) == 1
    for _ in range(25):
        canvas.repaint()
    assert len(calls) == 1


def test_a_stack_change_composites_once_however_often_it_paints(counted):
    canvas, stack, calls = counted
    stack["image"].opacity = 0.5
    for _ in range(5):
        canvas.repaint()
    assert len(calls) == 2
    stack["mask"].visible = False
    stack["mask"].visible = True
    canvas.repaint()
    canvas.repaint()
    assert len(calls) == 3


def test_a_new_world_window_composites_again(counted):
    canvas, _stack_, calls = counted
    canvas.resize(260, 200)
    canvas.repaint()
    canvas.repaint()
    assert len(calls) == 2
    QTest.mouseMove(canvas, QPoint(100, 100))
    canvas._canvas = canvas._canvas.zoomed(1.2, centre=(50.0, 50.0))
    canvas.repaint()
    assert len(calls) == 3
    assert calls[-1] is canvas._canvas


def test_a_brush_dab_shows_and_then_stays_kept(counted, qtbot):
    canvas, stack, calls = counted
    panel = ct.BrushPanel(canvas, layer=stack["mask"])
    qtbot.addWidget(panel)
    panel.start_painting()
    canvas.repaint()
    before = len(calls)
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(110, 100))
    canvas.repaint()
    assert len(calls) == before + 1
    painted = canvas.grab().toImage()
    for _ in range(10):
        canvas.repaint()
    assert len(calls) == before + 1
    panel.undo()
    canvas.repaint()
    assert len(calls) == before + 2
    assert canvas.grab().toImage() != painted
