"""Histogram cutoffs change display and opt-in detector input, never source data."""
import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, tmp_path):
    for name in ('a', 'b'):
        imageio.imwrite(tmp_path / (name + '.tif'),
                        np.arange(128*128, dtype=np.uint16).reshape(128, 128))
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(tmp_path))
    made._canvas.resize(400, 400)
    made._canvas.refresh()
    yield made
    made._close_levels()
    made._magnifier.close()
    made._canvas.close_enhancer()


def open_levels(screen, qtbot):
    screen._btn_levels.click()
    dialog = screen._levels_dialog
    qtbot.waitUntil(lambda: dialog.ready, timeout=10000)
    return dialog


def test_levels_preserve_source_and_masks_and_detection_is_explicit(screen, qtbot):
    canvas = screen._canvas
    original, mask = canvas.image.copy(), canvas.mask.copy()
    dialog = open_levels(screen, qtbot)
    before = canvas.pixmap().toImage()
    dialog.black.setValue(4000)
    dialog.white.setValue(12000)
    assert canvas.pixmap().toImage() != before
    np.testing.assert_allclose(np.percentile(original, [canvas.norm_lo, canvas.norm_hi]),
                               [4000, 12000], atol=0.001)
    np.testing.assert_array_equal(canvas.detection_source(), original)
    dialog.detect.setChecked(True)
    assert screen._detect_normalized.isChecked()
    detected = canvas.detection_source()
    assert detected.flat[3999] == 0 and detected.flat[12001] == 65535
    assert 0 < detected.flat[8000] < 65535
    screen._detect_normalized.setChecked(False)
    assert not dialog.detect.isChecked()
    np.testing.assert_array_equal(canvas.detection_source(), original)
    np.testing.assert_array_equal(canvas.image, original)
    np.testing.assert_array_equal(canvas.mask, mask)


def test_histogram_drag_and_reset_follow_actual_intensity_axis(screen, qtbot):
    dialog = open_levels(screen, qtbot)
    plot = dialog.plot
    start = QPoint(round(plot.level_x(plot.levels[0])), 80)
    end = QPoint(round(plot.width()*0.25), 80)
    QTest.mousePress(plot, Qt.LeftButton, pos=start)
    QTest.mouseMove(plot, end)
    QTest.mouseRelease(plot, Qt.LeftButton, pos=end)
    assert 24 < screen._norm_lo.value() < 26
    assert plot.levels[0] == pytest.approx(dialog.black.value(), abs=1e-5)
    dialog.reset.click()
    assert screen._norm_lo.value() == 0 and screen._norm_hi.value() == 100
    screen._norm_lo.setValue(10)
    assert dialog.black.value() == pytest.approx(1638.3)


def test_reopening_reuses_and_new_field_or_inversion_closes_snapshot(screen, qtbot):
    dialog = open_levels(screen, qtbot)
    assert open_levels(screen, qtbot) is dialog
    screen._on_next()
    qtbot.waitUntil(lambda: not screen._loading, timeout=10000)
    assert dialog.closed and screen._levels_dialog is None
    dialog = open_levels(screen, qtbot)
    screen._invert_display.setChecked(True)
    assert dialog.closed and screen._levels_dialog is None
    inverted = open_levels(screen, qtbot)
    assert inverted.values[0] == 0 and inverted.values[-1] == 65535


def test_constant_and_nonfinite_fields_are_explained(qtbot):
    for image, ready in [(np.ones((8, 8), dtype=np.uint16), True),
                         (np.full((8, 8), np.nan), False)]:
        dialog = mm._LevelsDialog(image, (1, 99.9))
        qtbot.addWidget(dialog)
        dialog.show()
        if ready:
            qtbot.waitUntil(lambda: dialog.ready)
            assert not dialog.black.isEnabled() and not dialog.plot.isEnabled()
            assert 'one intensity' in dialog.caption.text()
        else:
            qtbot.waitUntil(lambda: 'no finite' in dialog.caption.text())
            assert not dialog.ready
        dialog.close()
        assert dialog.values is None and dialog.closed
