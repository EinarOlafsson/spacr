"""The orthogonal panels paint, and refuse to paint nonsense.

A paint handler is the one place in a Qt application where an exception takes
the window with it, so the panel draws its pixmap and its crosshair inside a
guard and reports a failure to the log instead.

The crosshair itself is only drawn when the world point it names is one this
canvas can place. A point given in axes the canvas does not span has no pixel,
and drawing it at a defaulted (0, 0) would put a yellow cross in the corner of
every panel and call it the selection.

The slider arithmetic has the same shape: a degenerate extent -- one plane, or
a stack whose voxel size along an axis is zero -- has no scale to divide by,
and the panels answer 0 rather than dividing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

CONFOCAL_STEPS = {"z": 2.0, "y": 0.65, "x": 0.65}


def _volume(shape=(10, 64, 64)):
    from spacr.layers import LayerStack, Spacing

    stack = LayerStack()
    data = np.zeros(shape, np.uint16)
    data[shape[0] // 2, 30:34, 30:34] = 4000
    stack.add_image(data, name="volume",
                    spacing=Spacing.from_map(CONFOCAL_STEPS, units="um"),
                    contrast_limits=(0.0, 4000.0))
    return stack


def _view(qtbot, **kwargs):
    from spacr.qt import ortho_view as ov

    view = ov.OrthoView(_volume(), **kwargs)
    qtbot.addWidget(view)
    view.resize(520, 520)
    return view


def test_a_panel_with_no_volume_says_so_rather_than_painting(qtbot, qapp):
    """An empty panel draws its own caption instead of an empty pixmap."""
    from spacr.qt.ortho_view import OrthoPanel

    panel = OrthoPanel("xy")
    qtbot.addWidget(panel)
    panel.resize(120, 90)

    grabbed = panel.grab()

    assert not grabbed.isNull()
    assert panel.canvas is None


def test_a_panel_paints_its_canvas_and_its_crosshair(qtbot, qapp):
    """The panel renders the volume and puts the crosshair on it."""
    from spacr.qt.ortho_view import OrthoPanel

    view_model_stack = _volume()
    from spacr.layers import OrthoViews

    views = OrthoViews.covering(view_model_stack, width=128)
    panel = OrthoPanel("xy")
    qtbot.addWidget(panel)
    panel.resize(200, 200)
    panel.show_canvas(view_model_stack, views.xy, crosshair=views.point)

    assert panel._crosshair is not None

    grabbed = panel.grab()

    assert not grabbed.isNull()
    assert grabbed.width() >= 1


def test_a_crosshair_the_canvas_cannot_place_is_not_drawn(qtbot, qapp):
    """A point in axes this canvas does not span leaves the crosshair off.

    Defaulting it to the canvas origin would draw a cross in the corner of
    every panel and present it as the selected point.
    """
    from spacr.layers import OrthoViews
    from spacr.qt.ortho_view import OrthoPanel

    stack = _volume()
    views = OrthoViews.covering(stack, width=128)
    panel = OrthoPanel("xy")
    qtbot.addWidget(panel)
    panel.resize(200, 200)

    panel.show_canvas(stack, views.xy, crosshair={"t": 3.0})

    assert panel._crosshair is None
    assert not panel.grab().isNull()


def test_a_panel_that_cannot_paint_logs_instead_of_raising(qtbot, qapp,
                                                           monkeypatch,
                                                           caplog):
    """A render failure is logged; the paint handler still returns cleanly.

    An exception out of ``paintEvent`` takes the whole window down.
    """
    import logging

    from spacr.layers import OrthoViews
    from spacr.qt import ortho_view as ov

    stack = _volume()
    views = OrthoViews.covering(stack, width=128)
    panel = ov.OrthoPanel("xy")
    qtbot.addWidget(panel)
    panel.resize(200, 200)
    panel.show_canvas(stack, views.xy, crosshair=views.point)

    def _explode(self, canvas):
        raise RuntimeError("the volume went away mid-paint")

    monkeypatch.setattr(type(stack), "render_uint8", _explode)

    with caplog.at_level(logging.ERROR, logger=ov.LOG.name):
        grabbed = panel.grab()

    assert not grabbed.isNull()
    assert any("Could not paint" in record.message
               for record in caplog.records)


def test_only_a_left_click_moves_the_crosshair(qtbot, qapp):
    """A right-click on a panel reports nothing."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from spacr.layers import OrthoViews
    from spacr.qt.ortho_view import OrthoPanel

    stack = _volume()
    views = OrthoViews.covering(stack, width=128)
    panel = OrthoPanel("xy")
    qtbot.addWidget(panel)
    panel.show_canvas(stack, views.xy, crosshair=views.point)
    seen = []
    panel.clicked.connect(lambda *args: seen.append(args))

    right = QMouseEvent(QEvent.MouseButtonPress, QPointF(5.0, 5.0),
                        Qt.RightButton, Qt.RightButton, Qt.NoModifier)
    panel.mousePressEvent(right)

    assert seen == []

    left = QMouseEvent(QEvent.MouseButtonPress, QPointF(5.0, 5.0),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    panel.mousePressEvent(left)

    assert seen == [("xy", 4.0, 4.0)]


def test_a_click_on_a_panel_with_no_canvas_reports_nothing(qtbot, qapp):
    """A panel that has never been given a canvas has no pixel to report."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from spacr.qt.ortho_view import OrthoPanel

    panel = OrthoPanel("xy")
    qtbot.addWidget(panel)
    seen = []
    panel.clicked.connect(lambda *args: seen.append(args))

    panel.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress,
                                      QPointF(5.0, 5.0), Qt.LeftButton,
                                      Qt.LeftButton, Qt.NoModifier))

    assert seen == []


def test_rebuilding_the_sliders_empties_the_box_however_it_was_filled(qtbot,
                                                                     qapp):
    """A bare widget in the slider box is drained like a slider row is.

    ``set_stack`` rebuilds the box on every call; anything left behind
    accumulates one stale row per volume the user opens.
    """
    from PySide6.QtWidgets import QLabel

    view = _view(qtbot, width=128)
    stray = QLabel("left over")
    view.slider_box.addWidget(stray)

    view._build_sliders()

    assert view.slider_box.indexOf(stray) == -1


def test_a_degenerate_extent_answers_zero_rather_than_dividing(qtbot, qapp,
                                                               monkeypatch):
    """An axis with no span and no voxel size has no scale to divide by.

    A stack whose metadata reports a zero step is malformed, and the panels
    have to stay usable rather than raising ``ZeroDivisionError`` out of a
    slider callback.
    """
    from spacr.layers import OrthoViews

    view = _view(qtbot, width=128)
    monkeypatch.setattr(OrthoViews, "slider",
                        lambda self, axis: (5.0, 5.0, 0.0))

    assert view._tick("z", 12.0) == 0
    assert view._snap("z", 12.0) == 12.0
    assert view.slice_index("z") == 0


def test_an_axis_with_no_slider_has_no_readout_to_update(qtbot, qapp):
    """Asking for a readout on an axis that has no slider is a no-op."""
    view = _view(qtbot, width=128)

    assert view._update_readout("not_an_axis") is None
