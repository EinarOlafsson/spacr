"""Wheel zoom retains the actual image point, including linked previews."""
import pytest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.live_preview import _ZoomView
from spacr.qt.widgets.zoom_view import ZoomableImageView

pytestmark = pytest.mark.qt


def wheel(view, position, delta):
    event = QWheelEvent(QPointF(position), QPointF(view.viewport().mapToGlobal(position)),
                        QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                        Qt.ScrollUpdate, False)
    QApplication.sendEvent(view.viewport(), event)


@pytest.mark.parametrize('kind', [_ZoomView, ZoomableImageView])
@pytest.mark.parametrize('fraction', [(0.18, 0.2), (0.78, 0.22), (0.23, 0.77), (0.8, 0.8)])
def test_zoom_uses_event_position_without_prior_mouse_move(qtbot, kind, fraction):
    views = [kind(), kind()]
    for view in views:
        qtbot.addWidget(view)
        view.resize(480, 360)
        view.set_pixmap(QPixmap(1000, 750))
        view.show()
    QApplication.processEvents()
    a, b = views
    if kind is _ZoomView:
        a.set_peer(b)
        b.set_peer(a)
    else:
        a.link_to(b)
    original_scale = a.transform().m11()
    for active in (a, b):
        point = QPoint(int(active.viewport().width() * fraction[0]),
                       int(active.viewport().height() * fraction[1]))
        for delta in [120] * 5 + [-120] * 7 + [120] * 2:
            target = active.mapToScene(point)
            wheel(active, point, delta)
            QApplication.processEvents()
            moved = active.mapFromScene(target)
            assert abs(moved.x() - point.x()) <= 2
            assert abs(moved.y() - point.y()) <= 2
            assert a.transform() == b.transform()
            assert a.horizontalScrollBar().value() == b.horizontalScrollBar().value()
            assert a.verticalScrollBar().value() == b.verticalScrollBar().value()
        assert active.transform().m11() == pytest.approx(original_scale)


@pytest.mark.parametrize('kind', [_ZoomView, ZoomableImageView])
def test_horizontal_wheel_does_not_zoom_and_fit_clears_padding(qtbot, kind):
    view = kind()
    qtbot.addWidget(view)
    view.resize(500, 360)
    view.set_pixmap(QPixmap(1200, 700))
    view.show()
    QApplication.processEvents()
    original = view.transform()
    wheel(view, QPoint(70, 80), 0)
    assert view.transform() == original
    wheel(view, QPoint(70, 80), 120)
    if kind is _ZoomView:
        view.reset_zoom()
    else:
        view.fit()
    assert view.sceneRect() == view.scene().sceneRect()


def test_make_masks_zoom_preserves_an_off_center_image_point(qtbot):
    import numpy as np
    from spacr.qt.screens.make_masks import _MaskCanvas

    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_image_and_mask(np.zeros((750, 1000), np.uint8),
                              np.zeros((750, 1000), np.uint16))
    QApplication.processEvents()
    position = QPoint(180, 140)
    for delta in [120] * 4 + [-120] * 4:
        target = canvas._canvas_to_image(position.x(), position.y())
        event = QWheelEvent(QPointF(position), QPointF(canvas.mapToGlobal(position)),
                            QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                            Qt.ScrollUpdate, False)
        QApplication.sendEvent(canvas, event)
        mapped = canvas._image_to_canvas(*target)
        assert abs(mapped.x() - position.x()) <= 2
        assert abs(mapped.y() - position.y()) <= 2


@pytest.mark.parametrize('name', ['xy', 'zx', 'yz'])
def test_orthogonal_planes_zoom_at_pointer_without_moving_crosshair(qtbot, name):
    import numpy as np
    from spacr.layers import LayerStack
    from spacr.qt.ortho_view import OrthoView

    stack = LayerStack()
    stack.add_image(np.zeros((20, 64, 80), np.uint16), name='volume')
    view = OrthoView(stack, width=128)
    qtbot.addWidget(view)
    view.resize(520, 520)
    view.show()
    QApplication.processEvents()
    panel = view.panels[name]
    point = QPoint(int(panel.canvas.width * .3) + 1, int(panel.canvas.height * .7) + 1)
    crosshair = dict(view.views.point)
    world = panel.canvas.world_at(point.y() - 1, point.x() - 1)
    for delta in [120, -120]:
        global_point = panel.mapToGlobal(point)
        event = QWheelEvent(QPointF(view.mapFromGlobal(global_point)), QPointF(global_point),
                            QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                            Qt.ScrollUpdate, False)
        view.wheelEvent(event)
        mapped = panel.canvas.pixel_at(world)
        assert mapped == pytest.approx((point.y() - 1, point.x() - 1))
        assert view.views.point == crosshair
