"""The ROI pen's event dispatch, its undo and cancel, and the panel's Clear.

The pen claims left (vertex), right (undo), Escape, Return/Enter and
Backspace/Delete and declines everything else so the canvas keeps its pan and
arrow keys; undo and cancel stay consistent when there is nothing to take
back or when the preview was removed by another route; Clear abandons a
half-drawn outline together with the closed shapes.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv
from spacr.qt import roi_tool as rt

pytestmark = pytest.mark.qt


class _Event:
    """A mouse or key event with only the accessor the pen reads."""

    def __init__(self, *, button=None, key=None):
        self._button = button
        self._key = key

    def button(self):
        return self._button

    def key(self):
        return self._key


def _panel(qtbot, tmp_path, size=64):
    stack = LayerStack()
    stack.add_image(np.arange(size * size, dtype=np.uint16).reshape(size, size),
                    name="image",
                    spacing=Spacing.isotropic(2, 1.0, units="px"))
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    panel = rt.RoiPanel(canvas, roi_path=str(tmp_path / "roi.json"))
    qtbot.addWidget(panel)
    return canvas, panel


def _pending(pen):
    return pen.pending.tolist()


def test_roi_pen_claims_only_its_own_buttons_and_keys(subtests, qtbot,
                                                      tmp_path):
    """Each event's claim and the vertices it leaves, case by case."""
    with subtests.test(msg="the pen reports its kind and layer"):
        _canvas, panel = _panel(qtbot, tmp_path)
        panel.kind_combo.setCurrentText("rectangle")
        pen = panel.start_drawing()
        assert pen.kind == "rectangle"
        assert pen.layer.name == rt.ROI_LAYER_NAME
        assert _canvas.tool is pen

    cases = [
        ("left click adds a vertex", "press", Qt.LeftButton, True,
         [[4.0, 4.0], [4.0, 40.0], [20.0, 30.0]]),
        ("middle click is not a vertex", "press", Qt.MiddleButton, False,
         [[4.0, 4.0], [4.0, 40.0]]),
        ("side button is not a vertex", "press", Qt.BackButton, False,
         [[4.0, 4.0], [4.0, 40.0]]),
        ("right click takes the last vertex back", "press", Qt.RightButton,
         True, [[4.0, 4.0]]),
        ("Backspace undoes", "key", Qt.Key_Backspace, True, [[4.0, 4.0]]),
        ("Delete undoes", "key", Qt.Key_Delete, True, [[4.0, 4.0]]),
        ("Escape cancels", "key", Qt.Key_Escape, True, []),
        ("an arrow key falls through", "key", Qt.Key_Left, False,
         [[4.0, 4.0], [4.0, 40.0]]),
        ("a letter falls through", "key", Qt.Key_A, False,
         [[4.0, 4.0], [4.0, 40.0]]),
    ]
    for case_id, kind, code, claimed, pending in cases:
        with subtests.test(msg=case_id):
            canvas, panel = _panel(qtbot, tmp_path)
            pen = panel.start_drawing()
            pen.add_world({"y": 4.0, "x": 4.0})
            pen.add_world({"y": 4.0, "x": 40.0})
            if kind == "press":
                got = pen.press(canvas, {"y": 20.0, "x": 30.0},
                                _Event(button=code))
            else:
                got = pen.key(canvas, _Event(key=code))
            assert got is claimed
            assert _pending(pen) == pending

    for case_id, code in [("Return closes the shape", Qt.Key_Return),
                          ("Enter closes the shape", Qt.Key_Enter)]:
        with subtests.test(msg=case_id):
            canvas, panel = _panel(qtbot, tmp_path)
            pen = panel.start_drawing()
            finished = []
            pen.roi_finished.connect(finished.append)
            for world in ({"y": 4.0, "x": 4.0}, {"y": 4.0, "x": 40.0},
                          {"y": 40.0, "x": 40.0}):
                pen.add_world(world)
            assert pen.key(canvas, _Event(key=code)) is True
            assert finished == [0]
            assert _pending(pen) == []
            assert len(pen.layer) == 1


def test_roi_pen_undo_cancel_and_clear_stay_consistent(subtests, qtbot,
                                                       tmp_path):
    """Nothing to undo announces nothing; a lost preview still cancels; Clear drops the outline."""
    with subtests.test(msg="undo past the end is a no-op, not a pop"):
        _canvas, panel = _panel(qtbot, tmp_path)
        pen = panel.start_drawing()
        changes = []
        pen.roi_changed.connect(lambda: changes.append(1))
        pen.add_world({"y": 4.0, "x": 4.0})
        assert len(changes) == 1
        assert pen.undo() == 0
        assert len(changes) == 2
        assert pen.undo() == 0
        assert len(changes) == 2
        pen.cancel()
        assert len(changes) == 2

    with subtests.test(msg="a preview removed by another route still cancels"):
        _canvas, panel = _panel(qtbot, tmp_path)
        pen = panel.start_drawing()
        changes = []
        pen.roi_changed.connect(lambda: changes.append(1))
        pen.add_world({"y": 4.0, "x": 4.0})
        pen.add_world({"y": 4.0, "x": 30.0})
        assert len(pen.layer) == 1
        pen.layer.remove(0)
        pen.cancel()
        assert len(pen.layer) == 0
        assert pen.pending.shape == (0, 2)
        assert len(changes) == 3
        pen.add_world({"y": 8.0, "x": 8.0})
        pen.add_world({"y": 8.0, "x": 30.0})
        assert len(pen.layer) == 1
        pen.cancel()
        assert len(pen.layer) == 0
        assert pen.pending.shape == (0, 2)

    with subtests.test(msg="Clear while drawing abandons the half-drawn outline"):
        _canvas, panel = _panel(qtbot, tmp_path)
        pen = panel.start_drawing()
        for world in ({"y": 4.0, "x": 4.0}, {"y": 4.0, "x": 40.0},
                      {"y": 40.0, "x": 40.0}):
            pen.add_world(world)
        assert pen.close_shape() == 0
        pen.add_world({"y": 50.0, "x": 50.0})
        assert _pending(pen) == [[50.0, 50.0]]
        assert panel.clear_rois() == 1
        assert pen.pending.shape == (0, 2)
        assert len(panel.roi_layer(create=False)) == 0
        assert panel.clear_rois() == 0
