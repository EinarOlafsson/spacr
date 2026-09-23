"""Native OS arrow remains unchanged; resizing is signalled on the surface."""
import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QWidget
from spacr.qt.widgets import cursor_policy as cp

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('shape', [Qt.SizeHorCursor, Qt.SizeFDiagCursor, Qt.PointingHandCursor,
                                  Qt.WhatsThisCursor, Qt.IBeamCursor, Qt.ClosedHandCursor])
def test_interactive_shapes_are_the_native_os_arrow(qapp, qtbot, shape):
    cp.install_cursor_policy(qapp)
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setCursor(shape)
    assert widget.cursor().shape() == Qt.ArrowCursor
    assert widget.cursor().pixmap().isNull()
    widget.unsetCursor()
    assert widget.cursor().shape() == Qt.ArrowCursor


@pytest.mark.parametrize('active', [False, True])
def test_arrow_uses_no_custom_artwork(qapp, active):
    cursor = cp.arrow_cursor(active)
    assert cursor.shape() == Qt.ArrowCursor
    assert cursor.pixmap().isNull()


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
    assert window.cursor().shape() == Qt.ArrowCursor
    assert filter_._hint.edges == Qt.LeftEdge | Qt.TopEdge


def test_main_window_edge_hint_over_child_and_fixed_size(qapp, qtbot):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QMainWindow, QLabel
    from spacr.qt.widgets import glass
    glass.install_glass_everywhere(qapp)
    window = QMainWindow()
    qtbot.addWidget(window)
    child = QLabel('Main window contents')
    window.setCentralWidget(child)
    window.resize(400, 300)
    glass.let_the_user_resize(window)
    window.show()
    qtbot.waitExposed(window)
    resizer = window._spacr_resizer
    for point, expected in ((QPoint(399, 100), Qt.RightEdge),
                            (QPoint(399, 299), Qt.RightEdge | Qt.BottomEdge),
                            (QPoint(200, 100), Qt.Edge(0))):
        position = child.mapFrom(window, point)
        event = QMouseEvent(QEvent.MouseMove, QPointF(position),
                            QPointF(window.mapToGlobal(point)), Qt.NoButton,
                            Qt.NoButton, Qt.NoModifier)
        qapp.sendEvent(child, event)
        assert resizer._hint.edges == expected
        assert resizer._hint.isVisible() == bool(expected)
        assert window.cursor().shape() == Qt.ArrowCursor
        if expected:
            image = resizer._hint.grab().toImage()
            pixel = image.pixelColor(398, 100)
            assert pixel.name() == '#168cff'
            assert image.pixelColor(397, 100).alpha() == 0
    window.setFixedSize(window.size())
    assert not glass._edges_at(window, QPoint(399, 299))
