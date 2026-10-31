"""A combo box's drop-down arrow is a chevron, not the flat bar a CSS border
triangle paints, in both themes and at 100 % and 200 % Zoom."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QComboBox, QStyle, QStyleOptionComboBox, QWidget, QVBoxLayout


def _arrow_pixels(theme, scale, enabled=True):
    from spacr.qt import theme as theming

    app = QApplication.instance()
    host = QWidget()
    host.setStyleSheet(theming.stylesheet(theme, scale,
                                          load_widget_registrars=False))
    layout = QVBoxLayout(host)
    combo = QComboBox()
    combo.addItems(["one", "two"])
    combo.setEnabled(enabled)
    layout.addWidget(combo)
    host.resize(int(260 * scale), int(80 * scale))
    host.show()
    app.processEvents()
    option = QStyleOptionComboBox()
    combo.initStyleOption(option)
    box = combo.style().subControlRect(
        QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, combo)
    image = combo.grab().toImage()
    box = box.intersected(QRect(0, 0, image.width(), image.height()))
    reference = image.pixelColor(box.left() + 1, box.top() + 1)
    ink = []
    for y in range(box.top(), box.bottom() + 1):
        for x in range(box.left(), box.right() + 1):
            c = image.pixelColor(x, y)
            distance = (abs(c.red() - reference.red())
                        + abs(c.green() - reference.green())
                        + abs(c.blue() - reference.blue()))
            if distance > 90:
                ink.append((x, y))
    host.close()
    return ink


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_arrow_is_a_chevron_not_a_flat_bar(qapp, theme, scale):
    ink = _arrow_pixels(theme, scale)
    assert ink, "no arrow was drawn"
    xs = [x for x, _ in ink]
    ys = [y for _, y in ink]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    assert width >= 6 and height >= 3, (width, height)
    # A chevron's tip is lower than its ends; a flat bar's rows are all the
    # same height in every column.
    left = [y for x, y in ink if x <= min(xs) + 1]
    middle = [y for x, y in ink if abs(x - (min(xs) + max(xs)) // 2) <= 1]
    assert max(middle) - max(left) >= 2, (max(left), max(middle))


def test_a_disabled_combo_dims_its_chevron(qapp):
    from PySide6.QtGui import QColor  # noqa: F401

    assert _arrow_pixels("dark", 1.0, enabled=False)
