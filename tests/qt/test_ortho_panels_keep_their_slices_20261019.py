"""Orthogonal panels composite each slice once per change, not once per paint.

Every panel repaints on each frame of the animated backdrop; compositing the
stack on each of those paints is what held the layer canvases to a few
frames a second. These tests count composites instead of timing them.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import LayerStack, Spacing
from spacr.qt import ortho_view as ov

pytestmark = pytest.mark.qt


@pytest.fixture()
def counted(qtbot, monkeypatch):
    stack = LayerStack()
    spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
    data = np.zeros((6, 40, 40), np.uint16)
    data[3, 10:20, 10:20] = 4000
    stack.add_image(data, name="volume", spacing=spacing,
                    contrast_limits=(0.0, 4000.0))
    calls = []
    original = stack.render_uint8
    monkeypatch.setattr(stack, "render_uint8",
                        lambda canvas: calls.append(canvas) or original(canvas))
    view = ov.OrthoView(stack, width=96)
    qtbot.addWidget(view)
    view.resize(420, 420)
    view.show()
    for panel in view.panels.values():
        panel.repaint()
    return view, stack, calls


def test_repaints_with_nothing_changed_composite_nothing(counted):
    view, _stack, calls = counted
    first = len(calls)
    assert first == len(view.panels)
    for _ in range(10):
        for panel in view.panels.values():
            panel.repaint()
    assert len(calls) == first


def test_a_layer_change_composites_every_panel_once(counted):
    view, stack, calls = counted
    first = len(calls)
    stack["volume"].opacity = 0.5
    for _ in range(4):
        for panel in view.panels.values():
            panel.repaint()
    assert len(calls) == first + len(view.panels)


def test_moving_the_crosshair_composites_the_slices_it_moved(counted):
    view, _stack, calls = counted
    first = len(calls)
    axis = next(iter(view._sliders))
    view.move_to(**{axis: view._world(axis, 1)})
    for panel in view.panels.values():
        panel.repaint()
        panel.repaint()
    assert first < len(calls) <= first + len(view.panels)


def test_closing_stops_listening(counted):
    view, stack, _calls = counted
    view.close()
    assert view._listening is None
    stack["volume"].opacity = 0.25
