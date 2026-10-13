"""A window's resize guide is a line over the middle half of the active side.

    "The lines indicating where to expand/shrink windows should span 50% of
     each side and be centered on that side."

The overlay is rendered into an image and the painted pixels along each edge
are measured, so the test checks what is drawn rather than the arithmetic.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                       # noqa: E402
from PySide6.QtGui import QColor, QImage            # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _painted_run(image, coords):
    """Return the first and last index along ``coords`` whose pixel is painted."""
    hits = [i for i, (x, y) in enumerate(coords) if QColor(image.pixel(x, y)).alpha() > 0
            and QColor(image.pixel(x, y)).blue() > 200]
    return (hits[0], hits[-1]) if hits else None


@pytest.mark.parametrize("size", [(640, 420), (920, 620)])
def test_each_active_edge_shows_a_centred_half_length_line(qapp, size):
    from spacr.qt.widgets.glass import _ResizeEdgeHint

    w, h = size
    window = QWidget()
    window.resize(w, h)
    hint = _ResizeEdgeHint(window)
    edges = Qt.LeftEdge | Qt.RightEdge | Qt.TopEdge | Qt.BottomEdge
    hint.show_edges(edges)
    assert hint.isVisibleTo(window)
    image = QImage(w, h, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    hint.render(image)

    span_x = w - 3
    span_y = h - 3
    left = _painted_run(image, [(1, y) for y in range(h)])
    top = _painted_run(image, [(x, 1) for x in range(w)])
    right = _painted_run(image, [(w - 2, y) for y in range(h)])
    bottom = _painted_run(image, [(x, h - 2) for x in range(w)])
    for run, side, span in ((left, h, span_y), (right, h, span_y),
                            (top, w, span_x), (bottom, w, span_x)):
        assert run is not None
        length = run[1] - run[0]
        assert abs(length - span / 2) <= 3
        centre = (run[0] + run[1]) / 2
        assert abs(centre - (side - 1) / 2) <= 2
    hint.show_edges(Qt.Edge(0))
    assert not hint.isVisibleTo(window)
    window.deleteLater()


def test_only_the_active_edge_is_painted(qapp):
    from spacr.qt.widgets.glass import _ResizeEdgeHint

    window = QWidget()
    window.resize(400, 300)
    hint = _ResizeEdgeHint(window)
    hint.show_edges(Qt.LeftEdge)
    image = QImage(400, 300, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    hint.render(image)
    assert _painted_run(image, [(1, y) for y in range(300)]) is not None
    assert _painted_run(image, [(x, 1) for x in range(400)]) is None
    assert _painted_run(image, [(398, y) for y in range(300)]) is None
    window.deleteLater()
