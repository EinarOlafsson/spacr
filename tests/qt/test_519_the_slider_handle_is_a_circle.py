"""519: the plaque preview's size slider has a round handle at every scale.

Reported as "the size slider in the plaque modual has a square on the
slider, this should be a circle". The handle's length and thickness were
three separately rounded sizes, so at a 70 or 75 % GUI scale they came out
one pixel apart and Qt dropped the radii that no longer fitted: a square.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QSlider, QStyle, QStyleOptionSlider,
                               QVBoxLayout, QWidget)

from spacr.qt import gui_scale
from spacr.qt.theme import palette_for
from spacr.qt.widgets import preview_scale as ps

from .test_the_gui_scale_fits_a_laptop import (
    _never_the_real_preferences,  # noqa: F401
    _the_layer_is_in_and_the_scale_goes_back,  # noqa: F401
)

pytestmark = pytest.mark.qt


def _handle_is_round(slider, qapp):
    """Grab ``slider`` and say whether its handle is a circle."""
    qapp.processEvents()
    image = slider.grab().toImage()
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    rect = slider.style().subControlRect(
        QStyle.CC_Slider, option, QStyle.SC_SliderHandle, slider)
    assert image.rect().contains(rect), (rect, image.rect())
    handle = QColor(image.pixel(rect.center())).name()
    corners = [QColor(image.pixel(point)).name() for point in (
        rect.topLeft() + QPoint(1, 1), rect.topRight() + QPoint(-1, 1),
        rect.bottomLeft() + QPoint(1, -1), rect.bottomRight() + QPoint(-1, -1))]
    return rect.width() == rect.height(), handle, corners


@pytest.fixture
def plaque(qtbot, qt_theme_applied):
    """The plaque preview, and a plain slider beside it, in one window."""
    from spacr.qt.widgets.plaque_preview import PlaquePreviewPanel

    window = QWidget()
    column = QVBoxLayout(window)
    panel = PlaquePreviewPanel()
    column.addWidget(panel)
    plain = QSlider(Qt.Horizontal)
    plain.setFixedWidth(72)
    column.addWidget(plain)
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    control = panel._scale_control
    control.show()
    yield control, plain
    control.set_percent(100)


@pytest.mark.parametrize("gui", [1.0, 0.75, 0.7, 1.25])
@pytest.mark.parametrize("percent", [100, 75])
def test_the_size_slider_handle_is_a_circle(plaque, qt_theme_applied,
                                            gui, percent):
    control, plain = plaque
    gui_scale.set_gui_scale_live(gui)
    control.set_percent(percent)
    for slider in (control.slider, plain):
        square_box, handle, corners = _handle_is_round(slider, qt_theme_applied)
        assert square_box, (gui, percent, slider.objectName())
        assert handle == QColor(palette_for("dark")["accent"]).name(), handle
        assert all(corner != handle for corner in corners), (
            gui, percent, slider.objectName(), handle, corners)


@pytest.mark.parametrize("factor", [0.7, 0.75, 0.95, 1.1, 1.3])
def test_a_scaled_radius_never_outgrows_half_its_side(factor):
    sheet = "QSlider::handle { width: 16px; border-radius: 8px }"
    for scaled in (gui_scale.scale_qss_text(sheet, factor),
                   ps.scale_qss(sheet, factor)):
        width = int(scaled.split("width:")[1].split("px")[0])
        radius = int(scaled.split("border-radius:")[1].split("px")[0])
        assert 2 * radius <= width, scaled
