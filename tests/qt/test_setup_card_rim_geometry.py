"""The setup card's rim: where the pointer puts the accent, and the shape of the lit run.

The pointer is read as a RAY from the card's centre (so it moves continuously
across the diagonals and answers for a pointer outside the window), and the lit
run is one path sampled along the whole rim (so it crosses a corner without a
seam and wraps past the start of the perimeter).
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QHoverEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.setup_card import SetupCard


@pytest.fixture
def card(qapp):
    """A 400 x 300 card with an 18 px corner radius and a 280 px arc."""
    widget = SetupCard(radius=18, arc=280)
    widget.resize(400, 300)
    yield widget
    widget._timer.stop()
    widget.deleteLater()


def test_the_pointer_names_the_place_on_the_rim_along_the_ray_from_the_centre(
        card, qapp, subtests):
    """perimeter_position is the ray's exit point, clockwise from the top-left; events steer it."""
    # 400 x 300: perimeter 1400, top edge 400 -> the top-right corner is 400/1400.
    for case, point, expected in (
            ("top-left corner", (0, 0), 0.0),
            ("middle of the top", (200, 0), 200 / 1400),
            ("top-right corner", (400, 0), 400 / 1400),
            ("bottom-right corner", (400, 300), 0.5),
            ("bottom-left corner", (0, 300), 1100 / 1400),
            ("far outside on the right", (9999, 150), 550 / 1400),
    ):
        with subtests.test(msg=case):
            assert card.perimeter_position(QPointF(*point)) == pytest.approx(
                expected)

    with subtests.test(msg="the exact centre names no direction and steers nothing"):
        centre = QPointF(200, 150)
        assert card.perimeter_position(centre) is None
        card._towards = 0.4
        card.flow_towards(centre)
        assert card._towards == 0.4

    card.show()
    qapp.processEvents()
    with subtests.test(msg="a mouse move over the card steers the rim"):
        point = QPointF(card.width() - 2, 2)
        before = card._towards
        QApplication.sendEvent(card, QMouseEvent(
            QMouseEvent.MouseMove, point, card.mapToGlobal(point.toPoint()),
            Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        assert card._towards != before
        assert card._towards == pytest.approx(card.perimeter_position(point))
        assert card._towards == pytest.approx(0.28607503607503604)
    with subtests.test(msg="a hover move steers it too"):
        point = QPointF(2, card.height() - 2)
        QApplication.sendEvent(card, QHoverEvent(
            QEvent.HoverMove, point, QPointF(0, 0), point))
        assert card._towards == pytest.approx(0.7860750360750359)


def test_the_lit_run_is_one_unbroken_path_that_follows_the_accent(card, subtests):
    """_accent_path is one moveTo plus 48 lineTo on the rim, wherever the accent is."""
    rect = QRectF(card.rect()).adjusted(1, 1, -1, -1)
    grown = rect.adjusted(-2, -2, 2, 2)
    for case, at, middle in (
            ("at the start", 0.0, (1.0, 19.0)),
            ("half way round", 0.5, (399.0, 281.0)),
            ("wrapping past the start", 0.99, (1.0, 32.612)),
    ):
        with subtests.test(msg=case):
            card._at = at
            path = card._accent_path(rect)
            assert path.elementCount() == 49
            assert path.elementAt(0).isMoveTo()
            assert all(path.elementAt(i).isLineTo() for i in range(1, 49))
            points = [QPointF(path.elementAt(i).x, path.elementAt(i).y)
                      for i in range(49)]
            assert all(grown.contains(p) for p in points)
            mid = path.elementAt(24)
            assert (round(mid.x, 3), round(mid.y, 3)) == middle
