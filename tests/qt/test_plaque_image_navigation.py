"""Plaque images remain stable until the user explicitly zooms or fits them."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget
from spacr.qt.widgets.plaque_preview import _ImageView


def test_large_image_does_not_grow_layout_or_autofit_on_resize(qtbot):
    host = QWidget(); layout = QVBoxLayout(host); view = _ImageView(host); layout.addWidget(view)
    qtbot.addWidget(host)
    host.resize(540, 360); host.show(); qtbot.waitExposed(host)
    view.set_image(np.zeros((1000, 2000, 3), np.uint8))
    scale, size = view._scale, host.size()
    assert view.image_rect().width() <= 480 and view.image_rect().height() <= 260
    for _ in range(12):
        qtbot.wait(10)
        assert host.size() == size and view._scale == scale
    host.resize(800, 600); qtbot.wait(10)
    assert view._scale == scale
    view.fit_image()
    assert view._scale > scale
    assert view.image_rect().width() <= view.width()


def test_zoom_keeps_image_point_under_pointer_and_overlay_refresh_keeps_zoom(qtbot):
    view = _ImageView(); qtbot.addWidget(view); view.resize(500, 400); view.show()
    image = np.zeros((400, 800, 3), np.uint8); view.set_image(image)
    position = QPointF(200, 180)
    before = view.image_point(position.x(), position.y())
    scale = view._scale
    wheel = QWheelEvent(position, view.mapToGlobal(position.toPoint()), QPoint(), QPoint(0,120),
                        Qt.NoButton, Qt.ControlModifier, Qt.NoScrollPhase, False)
    view.wheelEvent(wheel)
    assert view._scale == pytest.approx(scale*1.2)
    assert view.image_point(position.x(), position.y()) == pytest.approx(before)
    zoomed, pan = view._scale, QPointF(view._pan)
    view.set_image(image.copy())
    assert view._scale == zoomed and view._pan == pan
    assert view.sizeHint().width() == 480


def test_pan_does_not_select_well_but_click_does(qtbot):
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent
    view = _ImageView(); qtbot.addWidget(view); view.resize(500, 400); view.show()
    view.set_image(np.zeros((400,800,3),np.uint8)); selected=[];view.clicked.connect(lambda *p:selected.append(p))
    qtbot.mouseClick(view,Qt.LeftButton,pos=QPoint(250,200))
    assert selected == [pytest.approx((400,200))]
    selected.clear()
    qtbot.mousePress(view,Qt.LeftButton,pos=QPoint(250,200))
    move=QMouseEvent(QEvent.MouseMove,QPointF(290,220),QPointF(290,220),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
    view.mouseMoveEvent(move)
    qtbot.mouseRelease(view,Qt.LeftButton,pos=QPoint(290,220))
    assert not selected
    assert view._pan == QPointF(40,20)
    assert view.cursor().shape() == Qt.ArrowCursor
