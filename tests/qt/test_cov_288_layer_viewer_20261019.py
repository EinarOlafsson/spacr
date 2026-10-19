"""A brush dab the canvas cannot see leaves the kept picture alone."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import LayerStack, Spacing
from spacr.qt import layer_viewer as lv

pytestmark = pytest.mark.qt


def _shown(qtbot, stack):
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(220, 200)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.grab()
    assert canvas._picture is not None
    return canvas


def _spy(canvas, monkeypatch):
    answers = []
    original = canvas._patch

    def patch(event):
        answers.append(original(event))
        return answers[-1]

    monkeypatch.setattr(canvas, "_patch", patch)
    return answers


def test_a_dab_outside_the_zoomed_window_is_handled_without_a_redraw(
        qtbot, monkeypatch):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, 1.0, units="px")
    stack.add_image(np.arange(4096, dtype=np.uint16).reshape(64, 64),
                    name="image", spacing=spacing)
    stack.add_labels(np.zeros((64, 64), np.int64), name="mask", spacing=spacing)
    canvas = _shown(qtbot, stack)
    canvas._canvas = canvas._canvas.zoomed(8.0, centre=(100.0, 110.0))
    canvas.grab()
    picture = canvas._picture
    answers = _spy(canvas, monkeypatch)
    changed = stack["mask"].set_labels_at(
        (np.array([0, 1]), np.array([0, 0])), 5)
    assert changed == 2
    assert answers == [True]
    assert canvas._picture is picture, "nothing visible changed, nothing recomposited"


def test_a_dab_on_another_plane_of_a_volume_is_handled_without_a_redraw(
        qtbot, monkeypatch):
    stack = LayerStack()
    spacing = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0})
    stack.add_image(np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32),
                    name="image", spacing=spacing)
    stack.add_labels(np.zeros((3, 32, 32), np.int64), name="mask",
                     spacing=spacing)
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.set_plane(("y", "x"), depth={"z": 0.0})
    canvas.resize(220, 200)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.grab()
    picture = canvas._picture
    assert picture is not None
    answers = _spy(canvas, monkeypatch)
    changed = stack["mask"].set_labels_at(
        (np.array([2, 2]), np.array([5, 6]), np.array([5, 5])), 4)
    assert changed == 2
    assert answers == [True]
    assert canvas._picture is picture


def test_a_dab_on_the_plane_being_shown_recomposites_only_that_box(
        qtbot, monkeypatch):
    stack = LayerStack()
    spacing = Spacing.from_map({"z": 1.0, "y": 1.0, "x": 1.0})
    stack.add_image(np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32),
                    name="image", spacing=spacing)
    stack.add_labels(np.zeros((3, 32, 32), np.int64), name="mask",
                     spacing=spacing)
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.set_plane(("y", "x"), depth={"z": 0.0})
    canvas.resize(220, 200)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.grab()
    picture = canvas._picture
    answers = _spy(canvas, monkeypatch)
    stack["mask"].set_labels_at(
        (np.array([0, 0]), np.array([5, 6]), np.array([5, 5])), 4)
    assert answers == [True]
    assert canvas._picture is picture
    rgb = np.ascontiguousarray(lv.LayerStack.render_uint8(stack, canvas.canvas))
    height, width = rgb.shape[:2]
    full = lv.QImage(rgb.data, width, height, 3 * width,
                     lv.QImage.Format_RGB888).convertToFormat(
                         lv.QImage.Format_RGB32)
    assert picture.toImage().convertToFormat(lv.QImage.Format_RGB32) == full
