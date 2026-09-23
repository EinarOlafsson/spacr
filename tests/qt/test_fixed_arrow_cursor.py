"""Interactive contexts change cursor color without changing its silhouette."""
import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QWidget
from spacr.qt.widgets import cursor_policy as cp

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('shape', [Qt.SizeHorCursor, Qt.SizeFDiagCursor, Qt.PointingHandCursor,
                                  Qt.WhatsThisCursor, Qt.IBeamCursor, Qt.ClosedHandCursor])
def test_interactive_shapes_are_the_same_blue_arrow(qapp, qtbot, shape):
    cp.install_cursor_policy(qapp)
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setCursor(shape)
    assert widget.cursor().pixmap().cacheKey() == cp.arrow_cursor(True).pixmap().cacheKey()
    widget.unsetCursor()
    assert widget.cursor().pixmap().cacheKey() == cp.arrow_cursor(False).pixmap().cacheKey()


@pytest.mark.parametrize('light', [False, True])
def test_idle_and_blue_have_identical_geometry_and_theme_outline(qapp, monkeypatch, light):
    monkeypatch.setattr(cp, 'active_palette', lambda: {'fg': '#111111' if light else '#eeeeee'})
    active, idle = (cp.arrow_cursor(flag).pixmap().toImage() for flag in (True, False))
    assert active.size() == idle.size()
    assert all(active.pixelColor(x,y).alpha() == idle.pixelColor(x,y).alpha()
               for x in range(active.width()) for y in range(active.height()))
    colour = active.pixelColor(8, 14)
    assert colour.blue() > 220 and colour.red() < 60 and colour.alpha() == 255
    rim = active.pixelColor(4, 14)
    assert rim.lightness() < 100 if light else rim.lightness() > 180


def test_corner_resize_uses_press_origin_and_respects_minimum(qapp, qtbot):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent
    from spacr.qt.widgets.workflow_diagram import DiagramDialog
    window = DiagramDialog()
    qtbot.addWidget(window)
    window.resize(400, 300)
    window.setMinimumSize(250, 180)
    window.move(100, 100)
    window.show()
    qtbot.waitExposed(window)
    start = window.geometry()
    origin = window.mapToGlobal(QPoint(1, 1))
    filter_ = window._spacr_resizer
    press = QMouseEvent(QEvent.MouseButtonPress, QPointF(1,1), QPointF(origin), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert filter_.eventFilter(window, press)
    for dx, dy in ((20, 10), (40, 30), (500, 500)):
        move = QMouseEvent(QEvent.MouseMove, QPointF(1,1), QPointF(origin + QPoint(dx,dy)), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        assert filter_.eventFilter(window, move)
        assert window.width() == max(250, start.width() - dx)
        assert window.height() == max(180, start.height() - dy)
        assert window.geometry().bottomRight() == start.bottomRight()
    assert window.cursor().pixmap().cacheKey() == cp.arrow_cursor(True).pixmap().cacheKey()
