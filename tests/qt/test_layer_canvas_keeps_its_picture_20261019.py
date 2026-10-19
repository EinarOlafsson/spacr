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
    qtbot.waitExposed(canvas)
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


def _full(canvas):
    from PySide6.QtGui import QImage
    from spacr.layers import LayerStack
    rgb = np.ascontiguousarray(LayerStack.render_uint8(canvas.stack, canvas.canvas))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).convertToFormat(
        QImage.Format_RGB32)


def _kept(canvas):
    from PySide6.QtGui import QImage
    return canvas._picture.toImage().convertToFormat(QImage.Format_RGB32)


def test_a_brush_dab_redraws_only_its_box(counted, qtbot, monkeypatch):
    canvas, stack, calls = counted
    panel = ct.BrushPanel(canvas, layer=stack["mask"])
    qtbot.addWidget(panel)
    panel.start_painting()
    canvas.repaint()
    before = len(calls)
    pieces = []
    original = stack._render_uint8_gathered

    def counted_piece(view, rows=slice(None), cols=slice(None)):
        piece = original(view, rows, cols)
        pieces.append(piece.shape[:2])
        return piece
    monkeypatch.setattr(stack, "_render_uint8_gathered", counted_piece)
    QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(110, 100))
    canvas.repaint()
    assert len(calls) == before
    assert len(pieces) == 1
    h, w = canvas.canvas.shape
    assert pieces[0][0] * pieces[0][1] < h * w // 4
    assert _kept(canvas) == _full(canvas)
    for _ in range(10):
        canvas.repaint()
    assert len(calls) == before
    panel.undo()
    canvas.repaint()
    assert len(calls) == before
    assert _kept(canvas) == _full(canvas)


def test_a_dab_on_another_grid_composites_everything(qtbot):
    from spacr.layers import Spacing
    stack = _stack()
    stack.add_labels(np.zeros((8, 8), np.int64), name="coarse",
                     spacing=Spacing.isotropic(2, 8.0, units="px"))
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.grab()
    assert canvas._picture is not None
    layer = stack["mask"]
    layer.set_labels_at((np.array([5, 6]), np.array([5, 5])), 9)
    assert canvas._picture is None
    canvas.grab()
    assert _kept(canvas) == _full(canvas)
