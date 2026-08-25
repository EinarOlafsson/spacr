"""A canvas tool sees the events it consumes, and the canvas keeps the rest.

The tool protocol is what lets a lasso, a brush and a polygon share one
canvas. Its base class is deliberately inert so a tool that only wants clicks
does not have to write four no-ops, and the canvas has to honour "I consumed
this" both ways: a consumed key must not also reach the surrounding screen,
and an unconsumed one must.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest

from spacr.layers import FieldKey, LayerStack
from spacr.qt import layer_viewer as lv


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(),
                                    ("plate1", "A", "1", "1"))))


def _stack():
    stack = LayerStack()
    image = np.full((40, 40), 1000, dtype=np.uint16)
    image[20:30, 20:30] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[20:30, 20:30] = 17
    stack.add_labels(mask, name="mask", field=_field_key())
    return stack


def _sized(qtbot, widget, width=320, height=320):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


def _move_event(x=10.0, y=10.0):
    return QMouseEvent(QMouseEvent.MouseMove, QPointF(x, y), QPointF(x, y),
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)


# -- the inert base ----------------------------------------------------------

def test_the_base_tool_consumes_nothing(qtbot):
    """A tool that only wants clicks must not have to write four no-ops."""
    tool = lv.CanvasTool()
    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    world = {"y": 25.0, "x": 25.0}
    event = _move_event()
    assert tool.press(canvas, world, event) is False
    assert tool.move(canvas, world, event) is False
    assert tool.release(canvas, world, event) is False
    assert tool.double_click(canvas, world, event) is False
    assert tool.key(canvas, event) is False
    assert tool.detach() is None


# -- attaching ---------------------------------------------------------------

def test_attaching_the_tool_already_attached_does_not_detach_it(qtbot):
    """A screen re-applying its own tool must not lose a half-drawn shape."""
    class _Tool(lv.CanvasTool):
        def __init__(self):
            self.detached = 0

        def detach(self):
            self.detached += 1

    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    tool = _Tool()
    canvas.set_tool(tool)
    assert canvas.set_tool(tool) is tool
    assert tool.detached == 0
    canvas.set_tool(None)
    assert tool.detached == 1


# -- where an event landed ---------------------------------------------------

def test_an_event_over_a_canvas_resolves_to_a_world_point(qtbot):
    """A tool is handed world coordinates, never widget pixels."""
    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    canvas._ensure_canvas()
    world = canvas._tool_world(_move_event(50.0, 50.0))
    assert world is not None
    assert set(world) >= {"y", "x"}


def test_an_event_over_an_empty_canvas_resolves_to_nothing(qtbot):
    """With no layers there is no world to name a point in."""
    canvas = _sized(qtbot, lv.LayerCanvas())
    assert canvas._tool_world(_move_event()) is None


# -- keys and moves ----------------------------------------------------------

def test_a_key_no_tool_wanted_is_passed_on(qtbot):
    """The surrounding screen's shortcuts must still reach it."""
    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    event.accept()
    canvas.keyPressEvent(event)
    assert canvas._tool is None


def test_a_key_the_tool_consumed_is_not_passed_on(qtbot):
    """Escape belongs to the tool while one is attached."""
    class _Eats(lv.CanvasTool):
        def key(self, view, event):
            return True

    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    canvas.set_tool(_Eats())
    event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    canvas.keyPressEvent(event)
    assert event.isAccepted()


def test_a_move_the_tool_consumed_does_not_also_pan(qtbot):
    """A brush dragging over the image must not drag the image with it."""
    class _Tracks(lv.CanvasTool):
        def __init__(self):
            self.seen = []

        def move(self, view, world, event):
            self.seen.append(world)
            return True

    canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
    canvas._ensure_canvas()
    before = canvas.canvas
    tool = _Tracks()
    canvas.set_tool(tool)
    canvas.mouseMoveEvent(_move_event(40.0, 40.0))
    assert tool.seen, "the tool was never asked"
    assert canvas.canvas is before


# -- double click ------------------------------------------------------------

def test_double_clicking_nothing_opens_nothing(qtbot):
    """A double click on the background must not raise out of a handler."""
    viewer = _sized(qtbot, lv.LayerViewer(_stack()), 400, 400)
    opened = []
    viewer.open_objects = lambda keys, reason="": opened.append(keys)
    viewer._on_activated(None, {"y": 2.0, "x": 2.0}, 0)
    assert opened == []


# -- companion registration --------------------------------------------------

def test_a_companion_that_cannot_register_costs_only_itself(monkeypatch):
    """An optional screen must never stop the window opening."""
    monkeypatch.setattr(lv, "COMPANION_APPS", (
        ("spacr.qt.screens.no_such_companion", "register"),
        ("spacr.qt.screens.image_scatter", "register"),
    ))
    registered = lv.register_companion_apps()
    assert registered == ("spacr.qt.screens.image_scatter",)
