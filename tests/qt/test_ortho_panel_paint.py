"""An orthogonal panel paints its slice and crosshair, or logs why it cannot.

``OrthoPanel.show_canvas`` places the crosshair through the canvas: a point
the canvas spans becomes a (row, column) pixel and is drawn as two yellow
lines over the slice at a one-pixel inset; a point in axes the canvas does
not span has no pixel and draws no crosshair (a defaulted (0, 0) would put
a cross in every panel's corner).  A paint handler is the one place where
an exception takes the window with it, so a failed render is logged and
the paint still returns.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

CONFOCAL_STEPS = {"z": 2.0, "y": 0.65, "x": 0.65}
_YELLOW = (255, 255, 0)


def _volume(shape=(10, 64, 64)):
    from spacr.layers import LayerStack, Spacing

    stack = LayerStack()
    data = np.zeros(shape, np.uint16)
    data[shape[0] // 2, 30:34, 30:34] = 4000
    stack.add_image(data, name="volume",
                    spacing=Spacing.from_map(CONFOCAL_STEPS, units="um"),
                    contrast_limits=(0.0, 4000.0))
    return stack


def _rgb(image, x, y):
    colour = image.pixelColor(x, y)
    return colour.red(), colour.green(), colour.blue()


def test_orthopanel_draws_the_crosshair_only_where_the_canvas_places_it(
        qtbot, qapp, monkeypatch, caplog, subtests):
    """Each crosshair point gives its exact pixel and yellow lines, or none."""
    from spacr.layers import OrthoViews
    from spacr.qt import ortho_view as ov

    stack = _volume()
    views = OrthoViews.covering(stack, width=128)
    placed = views.xy.pixel_at(views.point)
    # case, crosshair point, expected crosshair pixel
    cases = [
        ("the selected point", views.point, placed),
        ("a point in an axis the canvas lacks", {"t": 3.0}, None),
        ("no point", None, None),
    ]
    for case, point, expected in cases:
        with subtests.test(msg=case):
            panel = ov.OrthoPanel("xy")
            qtbot.addWidget(panel)
            panel.resize(200, 200)
            panel.show_canvas(stack, views.xy, crosshair=point)
            assert panel._crosshair == expected
            assert panel.canvas is views.xy and panel.name == "xy"

            image = panel.grab().toImage()
            row = int(placed[0]) + 1
            column = int(placed[1]) + 1
            # sample the lines away from the bright object in the slice
            on_row = _rgb(image, 3, row)
            on_column = _rgb(image, column, 3)
            if expected is None:
                assert on_row != _YELLOW and on_column != _YELLOW
            else:
                assert on_row == _YELLOW and on_column == _YELLOW
                assert _rgb(image, 3, row + 3) != _YELLOW

    with subtests.test(msg="a render failure is logged, not raised"):
        panel = ov.OrthoPanel("zx")
        qtbot.addWidget(panel)
        panel.resize(200, 200)
        panel.show_canvas(stack, views.xy, crosshair=views.point)

        def explode(self, canvas):
            raise RuntimeError("the volume went away mid-paint")

        with monkeypatch.context() as patch:
            patch.setattr(type(stack), "render_uint8", explode)
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger=ov.LOG.name):
                grabbed = panel.grab()
        assert not grabbed.isNull()
        messages = [record.getMessage() for record in caplog.records
                    if record.name == ov.LOG.name]
        assert messages == ["Could not paint the zx panel"]
        assert caplog.records[-1].exc_info[0] is RuntimeError
