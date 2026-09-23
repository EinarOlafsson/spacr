"""Explicit enhancement application and Adaptive error/retry behavior."""
import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import detect_chain as dc
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, tmp_path):
    image = np.arange(128*128, dtype=np.uint16).reshape(128, 128)
    imageio.imwrite(tmp_path/'field.tif', image)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(tmp_path))
    made._canvas.resize(500, 500)
    made._canvas.refresh()
    yield made
    made._magnifier.close()
    made._canvas.close_enhancer()


def test_apply_is_explicit_reversible_and_keeps_original_pixels_and_settings(screen):
    canvas = screen._canvas
    original = canvas.image.copy()
    screen._enh_gamma.setValue(0.6)
    assert not screen._btn_apply.isChecked()
    assert screen._detect_chain() == dc.NO_CHAIN
    np.testing.assert_array_equal(canvas.detection_source(), original)
    screen._btn_apply.click()
    assert screen._btn_apply.isChecked() and canvas.enhance_display
    assert screen._detect_chain().gamma == 0.6
    assert not np.array_equal(canvas.detection_source(), original)
    np.testing.assert_array_equal(canvas.image, original)
    screen._btn_apply.click()
    assert screen._detect_chain() == dc.NO_CHAIN
    assert screen._enh_gamma.value() == 0.6
    assert not canvas.enhance_display
    np.testing.assert_array_equal(canvas.detection_source(), original)


def test_compare_uses_configured_steps_without_applying_them(screen, monkeypatch):
    captured = []
    class Comparison:
        def __init__(self, raw, enhanced, *args):
            captured.append((raw, enhanced))
        def show(self):
            pass
    monkeypatch.setattr(mm, '_ComparePreview', Comparison)
    screen._enh_gamma.setValue(0.6)
    screen._btn_compare.click()
    assert captured and not np.array_equal(*captured[0])
    assert not screen._btn_apply.isChecked()
    assert screen._detect_chain() == dc.NO_CHAIN


@pytest.mark.parametrize('scope', ['region', 'image'])
def test_adaptive_error_does_not_latch_otsu_and_block_change_retries(screen, qtbot, monkeypatch, scope):
    real = mm._MAGNIFIER_SEGMENTERS['adaptive']
    seen = []
    def segment(request, load_model=None):
        seen.append(request.method_params.adaptive_block)
        if request.method_params.adaptive_block == 3:
            raise ValueError('controlled invalid local window')
        return real(request, load_model)
    monkeypatch.setitem(mm._MAGNIFIER_SEGMENTERS, 'adaptive', segment)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData('adaptive'))
    block = screen._method_widgets['adaptive_block']
    block.setValue(3)
    mag = screen._magnifier
    messages = []
    mag.status.connect(messages.append)
    mag.set_scope(scope)
    mag.set_enabled(True)
    mag.hover(QPointF(250, 250))
    qtbot.waitUntil(lambda: any('controlled invalid local window' in m for m in messages), timeout=15000)
    assert screen._mag_mode.currentData() == 'adaptive'
    assert mag._model_settings()[0] == 'adaptive'
    assert not mag._unavailable
    block.setValue(31)
    qtbot.waitUntil(lambda: (mag._shown if scope == 'region' else mag._image_result) is not None, timeout=15000)
    result = mag._shown if scope == 'region' else mag._image_result
    assert result.mode == 'adaptive' and not result.note
    assert result.request.method_params.adaptive_block == 31
    assert 3 in seen and 31 in seen


def test_clear_all_objects_keeps_image_and_is_one_undoable_edit(screen, monkeypatch):
    mask = np.zeros_like(screen._canvas.mask)
    mask[10:20, 10:20] = 3
    mask[40:55, 40:55] = 8
    screen._canvas.mask = mask.copy()
    screen._history.clear()
    screen._history.push(mask)
    original = screen._canvas.image.copy()
    monkeypatch.setattr(screen, '_confirm', lambda *args: True)
    assert screen._btn_clear.text() == 'Clear all objects'
    screen._btn_clear.click()
    assert not screen._canvas.mask.any()
    np.testing.assert_array_equal(screen._canvas.image, original)
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, mask)
    screen._on_redo()
    assert not screen._canvas.mask.any()


def _wand_click(screen):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    screen._set_mode(mm.MODE_WAND_ADD)
    canvas = screen._canvas
    canvas.wand_trim_runaway = False
    canvas.wand_intensity_border = False
    canvas.wand_gradient_taper = False
    canvas.wand_max_pixels = 128*128
    canvas.wand_tol_pct = 2
    point = canvas._image_to_canvas(64.5, 64.5)
    QTest.mouseClick(canvas, Qt.LeftButton, pos=point)


def test_wand_uses_applied_pixels_and_toggling_off_restores_raw_flood(screen, qtbot):
    original = screen._canvas.image.copy()
    screen._enh_gamma.setValue(0.5)
    screen._btn_apply.click()
    canvas = screen._canvas
    qtbot.waitUntil(lambda: canvas._enhanced_picture is not None)
    _wand_click(screen)
    enhanced_mask = canvas.mask.copy()
    assert enhanced_mask.any()
    assert screen._log.edits[-1].detail['enhancement']['gamma'] == 0.5
    screen._on_undo()
    assert not canvas.mask.any()
    screen._btn_apply.click()
    _wand_click(screen)
    assert canvas.mask.any()
    assert not np.array_equal(canvas.mask, enhanced_mask)
    np.testing.assert_array_equal(canvas.image, original)


def test_wand_waits_without_editing_and_reapply_survives_an_obsolete_result(screen, qtbot, monkeypatch):
    import threading
    gate, started = threading.Event(), threading.Event()
    real = mm._enhanced_picture_for
    def slow(request):
        started.set()
        assert gate.wait(10)
        return real(request)
    monkeypatch.setattr(mm, '_enhanced_picture_for', slow)
    screen._enh_gamma.setValue(0.5)
    screen._btn_apply.click()
    canvas = screen._canvas
    try:
        qtbot.waitUntil(started.is_set)
        count = len(screen._log.edits)
        _wand_click(screen)
        assert not canvas.mask.any() and len(screen._log.edits) == count
        assert not canvas._stroke_in_progress
        assert 'updating' in screen._status_label.text()
        screen._btn_apply.click()
    finally:
        gate.set()
    qtbot.waitUntil(lambda: canvas._enhance_asked is None)
    assert canvas._enhanced_picture is None
    screen._btn_apply.click()
    qtbot.waitUntil(lambda: canvas._enhanced_picture is not None)
    _wand_click(screen)
    assert canvas.mask.any()


def test_failed_enhancement_blocks_wand_and_corrected_settings_retry(screen, qtbot, monkeypatch):
    real = mm._enhanced_picture_for
    def fails_once(request):
        if request.chain.gamma == 0.5:
            raise ValueError('controlled enhancement failure')
        return real(request)
    monkeypatch.setattr(mm, '_enhanced_picture_for', fails_once)
    screen._enh_gamma.setValue(0.5)
    screen._btn_apply.click()
    canvas = screen._canvas
    qtbot.waitUntil(lambda: canvas._enhance_failure is not None)
    _wand_click(screen)
    assert not canvas.mask.any() and not canvas._stroke_in_progress
    assert 'controlled enhancement failure' in screen._status_label.text()
    screen._enh_gamma.setValue(0.6)
    qtbot.waitUntil(lambda: canvas._enhanced_picture is not None)
    _wand_click(screen)
    assert canvas.mask.any() and canvas._enhance_failure is None
