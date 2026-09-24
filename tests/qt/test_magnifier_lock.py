"""Lock gestures preserve the selected region without deleting mask objects."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from spacr.qt.screens import make_masks as mm


@pytest.fixture
def lens(qtbot):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 400)
    canvas.set_image_and_mask(np.arange(10000, dtype=np.uint16).reshape(100, 100),
                              np.ones((100, 100), dtype=np.uint16))
    magnifier = mm._LiveMagnifier(canvas, canvas)
    canvas.magnifier = magnifier
    magnifier.segment = lambda request: np.zeros(request.crop.shape, dtype=np.int32)
    magnifier.set_size(32)
    canvas.show()
    canvas.refresh()
    magnifier.set_enabled(True)
    yield canvas, magnifier
    magnifier.close()
    canvas.close_enhancer()


def chord(canvas, pos=QPoint(160, 160)):
    QTest.keyPress(canvas, Qt.Key_L, Qt.ControlModifier)
    QTest.mouseClick(canvas, Qt.RightButton, Qt.ControlModifier, pos)
    QTest.keyRelease(canvas, Qt.Key_L, Qt.ControlModifier)


@pytest.mark.parametrize('scope', ['region', 'image'])
def test_chord_pins_region_size_zoom_and_never_deletes(lens, scope):
    canvas, mag = lens
    mag.set_scope(scope)
    before = canvas.mask.copy()
    chord(canvas)
    assert mag.locked
    cursor, anchor, size, zoom = mag._cursor, QPointF(mag._anchor), mag.size, mag.zoom
    box = mag.build_request().box
    QTest.mouseMove(canvas, QPoint(290, 290))
    mag.hover(None)
    for modifiers in (Qt.NoModifier, Qt.ShiftModifier):
        pos = QPointF(280, 280)
        QApplication.sendEvent(canvas, QWheelEvent(
            pos, pos, QPoint(), QPoint(0, 120), Qt.NoButton, modifiers,
            Qt.NoScrollPhase, False))
    mag.set_size(64)
    mag.set_zoom(8)
    assert (mag._cursor, mag._anchor, mag.size, mag.zoom) == (cursor, anchor, size, zoom)
    assert mag.build_request().box == box
    np.testing.assert_array_equal(canvas.mask, before)
    chord(canvas, QPoint(280, 280))
    assert not mag.locked
    mag.hover(QPointF(280, 280))
    mag.wheel(True)
    assert mag._cursor != cursor and mag.zoom > zoom
    np.testing.assert_array_equal(canvas.mask, before)


@pytest.mark.parametrize('reset', ['disable', 'field'])
def test_lock_is_released_when_lens_or_field_goes_away(lens, reset):
    canvas, mag = lens
    chord(canvas)
    assert mag.locked
    if reset == 'disable':
        mag.set_enabled(False)
    else:
        canvas.set_image_and_mask(np.zeros((80, 80), dtype=np.uint16), np.zeros((80, 80), dtype=np.uint16))
    assert not mag.locked and not mag._lock_key_down
    assert mag._cursor is None


def test_ctrl_right_without_held_l_still_deletes(lens):
    canvas, mag = lens
    QTest.keyPress(canvas, Qt.Key_L, Qt.ControlModifier)
    QTest.keyRelease(canvas, Qt.Key_L, Qt.ControlModifier)
    QTest.mouseClick(canvas, Qt.RightButton, Qt.ControlModifier, QPoint(160, 160))
    assert not mag.locked
    assert not canvas.mask.any()


def test_window_deactivation_clears_held_key(lens):
    canvas, mag = lens
    QTest.keyPress(canvas, Qt.Key_L, Qt.ControlModifier)
    assert mag._lock_key_down
    QApplication.sendEvent(canvas, QEvent(QEvent.WindowDeactivate))
    assert not mag._lock_key_down
