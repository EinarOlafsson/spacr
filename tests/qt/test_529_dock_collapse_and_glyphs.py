"""Dock contents/width survive repeated collapse and glyphs scale alone."""
import pytest
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from spacr.qt.widgets.dock import Dock, DockEdge
from spacr.qt.widgets import collapse_arrow


@pytest.mark.parametrize('size', [(760, 480), (1366, 768)])
def test_collapse_preserves_content_and_width_and_keyboard_restore(qtbot, size):
    host = QWidget()
    host.resize(*size)
    qtbot.addWidget(host)
    row = QHBoxLayout(host)
    slot = QWidget(host)
    dock = Dock([('__home__', 'Home', 'Home', ''),
                 ('measure', 'Measure', 'Objects', '')])
    QVBoxLayout(slot).addWidget(dock)
    edge = DockEdge(dock, host)
    edge.collapsedChanged.connect(lambda collapsed: slot.setVisible(not collapsed))
    row.addWidget(slot)
    row.addWidget(edge)
    row.addWidget(QWidget(), 1)
    edge.show()
    host.show()
    dock.set_column_width(280)
    qtbot.wait(20)
    original = dock.width()
    objects = tuple(dock.rows())
    for _ in range(3):
        qtbot.mouseClick(edge, Qt.LeftButton)
        assert not slot.isVisible()
        assert edge.isVisible() and edge.is_collapsed()
        assert 'show' in edge.toolTip()
        qtbot.keyClick(edge, Qt.Key_Space)
        qtbot.wait(10)
        assert slot.isVisible() and not edge.is_collapsed()
        assert dock.width() == original
        assert tuple(dock.rows()) == objects
        assert 'hide' in edge.toolTip()


def _glyph_bounds(orientation, factor, monkeypatch):
    monkeypatch.setattr(collapse_arrow, 'ARROW_SCALE', factor)
    image = QImage(400, 400, QImage.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(4)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    rect = QRect(0, 0, 16, 100) if orientation == Qt.Horizontal else QRect(0, 0, 100, 16)
    collapse_arrow.paint_collapse_arrow(painter, rect, orientation, True,
        {'surface_alt': '#555555', 'border': '#555555', 'text_muted': '#ff00ff'})
    painter.end()
    points = [(x, y) for x in range(400) for y in range(400)
              if (c := image.pixelColor(x, y)).red() > 150
              and c.blue() > 150 and c.green() < 50 and c.alpha() > 100]
    return (max(x for x, y in points)-min(x for x, y in points)+1,
            max(y for x, y in points)-min(y for x, y in points)+1)


@pytest.mark.parametrize('orientation', [Qt.Horizontal, Qt.Vertical])
def test_rendered_glyph_is_three_quarters_of_baseline(orientation, monkeypatch):
    baseline = _glyph_bounds(orientation, 1.0, monkeypatch)
    for _ in range(3):
        smaller = _glyph_bounds(orientation, 0.75, monkeypatch)
        assert all(abs(actual - old * .75) <= 2 for actual, old in zip(smaller, baseline)), (baseline, smaller)
