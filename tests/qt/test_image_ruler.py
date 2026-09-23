"""Line measurements remain in image pixels through zoom/pan and never edit masks."""
from math import hypot

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets.image_ruler import ImageRuler
from spacr.qt.widgets.live_preview import LivePreviewPanel, _ZoomView
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    canvas_xy, fields, screen,
)


def draw(widget, start, end):
    QTest.mousePress(widget, Qt.LeftButton, pos=start)
    QTest.mouseMove(widget, end)
    QTest.mouseRelease(widget, Qt.LeftButton, pos=end)


def test_make_masks_ruler_measures_without_mask_or_history_edits(screen):
    canvas = screen._canvas
    canvas.mask[:] = 7
    original = canvas.image.copy()
    mask = canvas.mask.copy()
    count = len(screen._log.edits)
    screen._mode_buttons[mm.MODE_RULER].click()
    draw(canvas, QPoint(*map(round, canvas_xy(10, 10))),
         QPoint(*map(round, canvas_xy(13, 14))))
    assert canvas.ruler.length() == 5
    assert canvas.ruler.label() == '5.00 px'
    endpoints = canvas.ruler.start, canvas.ruler.end
    canvas.zoom_at(12, 12, 2)
    assert canvas.ruler.length() == 5
    assert (canvas.ruler.start, canvas.ruler.end) == endpoints
    screen._set_mode(mm.MODE_BRUSH)
    assert canvas.ruler.length() == 5
    screen._set_mode(mm.MODE_RULER)
    QTest.mouseClick(canvas, Qt.RightButton, pos=QPoint(200, 200))
    assert canvas.ruler.length() is None
    np.testing.assert_array_equal(canvas.mask, mask)
    np.testing.assert_array_equal(canvas.image, original)
    assert len(screen._log.edits) == count


def test_ruler_and_magnifier_do_not_steal_each_others_gestures(screen):
    screen._btn_magnifier.setChecked(True)
    screen._set_mode(mm.MODE_RULER)
    assert screen._canvas.ruler.active and not screen._magnifier.enabled
    screen._btn_magnifier.setChecked(True)
    assert screen._magnifier.enabled and not screen._canvas.ruler.active


def test_ruler_calibration_is_explicit_and_supports_rectangular_pixels(qapp):
    ruler = ImageRuler()
    ruler.start, ruler.end = (0, 0), (3, 4)
    assert ruler.length() == 5 and ruler.length(True) is None
    ruler.set_spacing(2, 3)
    assert ruler.length(True) == hypot(6, 12)
    assert '5.00 px' in ruler.label() and '13.42 µm' in ruler.label()
    for invalid in (0, -1, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            ruler.set_spacing(invalid)
    ruler.set_spacing()
    assert ruler.label() == '5.00 px'


def test_preview_ruler_is_shared_preserved_by_render_and_cleared_on_new_field(qtbot, tmp_path):
    image = np.arange(64*64, dtype=np.uint16).reshape(64, 64)
    path = tmp_path/'field.tif'
    imageio.imwrite(path, image)
    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    panel.resize(1000, 760)
    panel.show()
    assert panel.load_image(path)
    qtbot.wait(20)
    panel._ruler_btn.click()
    view = panel._src_view
    clicks = []
    view.clicked.connect(lambda: clicks.append(True))
    start = view.mapFromScene(QPointF(10, 10))
    end = view.mapFromScene(QPointF(13, 14))
    draw(view.viewport(), start, end)
    length = view.ruler.length()
    assert length == pytest.approx(5, abs=0.4)
    assert panel._mask_view.ruler is view.ruler
    assert not clicks
    view._apply_zoom(3, broadcast=True)
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().maximum())
    assert view.ruler.length() == length
    panel._refresh_canvases()
    assert view.ruler.length() == length
    np.testing.assert_array_equal(panel._image, image)
    assert panel._masks == {}
    view.ruler.set_spacing(0.5)
    assert panel.load_image(path)
    assert view.ruler.length() is None and view.ruler.spacing is None
    panel.close()


def test_empty_and_letterbox_clicks_do_not_measure(qtbot):
    from PySide6.QtGui import QPixmap
    view = _ZoomView()
    qtbot.addWidget(view)
    view.resize(600, 200)
    view.show()
    view.ruler.set_active(True)
    draw(view.viewport(), QPoint(10, 10), QPoint(30, 30))
    assert view.ruler.length() is None
    view.set_pixmap(QPixmap(100, 100))
    draw(view.viewport(), QPoint(5, 20), QPoint(20, 40))
    assert view.ruler.length() is None


def test_shortcuts_scroll_instead_of_compressing_text(screen, qtbot):
    scroll = screen._shortcut_scroll
    scroll.setParent(None)
    qtbot.addWidget(scroll)
    scroll.resize(220, 350)
    scroll.show()
    qtbot.wait(20)
    assert scroll.verticalScrollBar().maximum() > 0
    for key, description in screen._shortcut_rows.values():
        assert key.height() >= key.fontMetrics().height()
        assert description.height() >= description.fontMetrics().height()
    last = list(screen._shortcut_rows.values())[-1][1]
    scroll.ensureWidgetVisible(last)
    qtbot.wait(20)
    top = last.mapTo(scroll.viewport(), QPoint()).y()
    assert 0 <= top and top + last.height() <= scroll.viewport().height()
