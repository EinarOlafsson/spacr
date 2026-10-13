"""A divider draws along the axis it was asked for.

The separator is one pixel thick in the direction it does not run. Getting
the orientation wrong gives a one-pixel-tall horizontal rule where a full
height vertical rule was wanted, which reads as a missing divider rather
than as a wrong one.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame

from spacr.qt.widgets.divider import Divider


def test_a_horizontal_divider_is_one_pixel_tall(qtbot):
    """The default runs across, so height is what is pinned."""
    line = Divider()
    qtbot.addWidget(line)
    assert line.frameShape() == QFrame.HLine
    assert line.height() == 1
    assert line.objectName() == "Divider"


def test_a_vertical_divider_is_one_pixel_wide(qtbot):
    """Asked for a column rule, it pins width and leaves height free."""
    line = Divider(Qt.Vertical)
    qtbot.addWidget(line)
    assert line.frameShape() == QFrame.VLine
    assert line.width() == 1
    assert line.maximumHeight() > 1
