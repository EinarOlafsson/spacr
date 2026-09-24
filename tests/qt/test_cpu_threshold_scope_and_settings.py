"""Named CPU thresholds must mean the same operation in both scopes."""
from __future__ import annotations

import imageio.v3 as imageio
import numpy as np
import pytest
from skimage import filters

from spacr.qt import detect_chain
from spacr.qt.screens import make_masks as mm


METHODS = ('li', 'yen', 'triangle', 'isodata', 'mean', 'minimum',
           'sauvola', 'niblack')


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    rng = np.random.default_rng(43)
    field = rng.normal(100, 5, (96, 96)).astype(np.float32)
    field[20:76, 20:76] += 100
    imageio.imwrite(tmp_path / 'field.tif', field)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(tmp_path))
    monkeypatch.setattr(made, '_detect_chain', lambda: detect_chain.NO_CHAIN)
    made._otsu_smoothing.setValue(0)
    made._otsu_fill_holes.setChecked(False)
    made._otsu_split.setChecked(False)
    made._min_area.setValue(0)
    made._otsu_window.setValue(15)
    made._otsu_local_k.setValue(0.35)
    made._magnifier.size = 96
    made._magnifier._cursor = (48, 48)
    made._magnifier.enabled = True
    made._magnifier.exclude_border = False
    made.region_requests = []
    made.image_requests = []
    monkeypatch.setattr(made._magnifier._worker, 'submit', made.region_requests.append)
    monkeypatch.setattr(made._magnifier._image_worker, 'submit', made.image_requests.append)
    yield made
    made._magnifier.close()
    made.close_folded()


def select(screen, mode):
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))


@pytest.mark.parametrize('mode', METHODS)
@pytest.mark.parametrize('bright', [True, False])
@pytest.mark.parametrize('scope', ['region', 'image'])
def test_both_scopes_use_the_selected_algorithm_on_unscaled_values(screen, mode, bright, scope):
    select(screen, mode)
    screen._otsu_bright.setChecked(bright)
    screen._otsu_correction.setValue(1.1)
    magnifier = screen._magnifier
    if scope == 'image':
        magnifier._start_image(magnifier._image_key_now())
        request = screen.image_requests[-1]
    else:
        request = magnifier.build_request()
    values = request.crop.astype(np.float32)
    function = getattr(filters, 'threshold_' + mode)
    if mode in ('sauvola', 'niblack'):
        level = function(values, window_size=15, k=0.35) * 1.1
    else:
        level = function(values)
        level = level * 1.1 if bright else values.max() - (values.max() - level) * 1.1
    expected = values > level if bright else values < level
    labels, used, note = mm._segment_region(request)
    assert used == mode and not note
    np.testing.assert_array_equal(labels > 0, expected)
    whole, _ = screen._cpu_detect(request.crop, mode, screen._otsu_settings())
    np.testing.assert_array_equal(labels, whole)


@pytest.mark.parametrize('mode', METHODS)
def test_switching_methods_ignores_saved_otsu_settings_and_restores_them_on_return(screen, mode):
    select(screen, 'multiotsu')
    screen._otsu_classes.setValue(4)
    screen._otsu_foreground.setValue(3)
    select(screen, 'otsu')
    screen._otsu_local.setChecked(True)
    select(screen, mode)
    settings = screen._otsu_settings()
    assert settings['classes'] == 2
    assert settings['foreground_class'] == 1
    assert settings['local'] is False
    assert not screen._otsu_classes.isEnabled()
    assert not screen._otsu_foreground.isEnabled()
    assert not screen._otsu_local.isEnabled()
    assert screen._otsu_bright.isEnabled()
    assert screen._otsu_window.isEnabled() is (mode in ('sauvola', 'niblack'))
    assert screen._otsu_description() == (
        'local 15 px, bright' if mode in ('sauvola', 'niblack') else 'bright')
    request = screen._magnifier.build_request()
    preview, _, _ = mm._segment_region(request)
    whole, _ = screen._cpu_detect(request.crop, mode, settings)
    np.testing.assert_array_equal(preview, whole)
    stale = dict(settings, classes=4, foreground_class=3, local=True)
    defensive, _ = screen._cpu_detect(request.crop, mode, stale)
    np.testing.assert_array_equal(defensive, whole)
    select(screen, 'otsu')
    assert screen._otsu_local.isChecked()
    assert screen._otsu_settings()['local'] is True
    screen._otsu_local.setChecked(False)
    assert screen._otsu_settings()['classes'] == 4
    assert screen._otsu_settings()['foreground_class'] == 3


def test_a_minimum_threshold_failure_does_not_substitute_or_disable_the_method(screen):
    select(screen, 'minimum')
    magnifier = screen._magnifier
    request = magnifier.build_request()
    bad = request._replace(crop=np.ones_like(request.crop))
    with pytest.raises(RuntimeError, match='maxima'):
        mm._segment_region(bad)
    magnifier._shown = magnifier._run(request)
    magnifier._requested_key = bad.key
    magnifier._on_delivered((bad, None, RuntimeError('Unable to find two maxima')))
    assert magnifier._shown is None
    assert not magnifier._unavailable
    recovered = magnifier._run(request)
    assert recovered.mode == 'minimum'
    assert recovered.count > 0


@pytest.mark.parametrize('mode', METHODS)
def test_completed_preview_matches_button_mask_with_shared_processing_settings(screen, mode):
    select(screen, mode)
    screen._otsu_smoothing.setValue(1.0)
    screen._otsu_fill_holes.setChecked(True)
    screen._otsu_split.setChecked(True)
    screen._min_area.setValue(8)
    request = screen._magnifier.build_request()
    completed = screen._magnifier._run(request)
    screen._combine_mode.setCurrentText('replace')
    screen._btn_otsu.click()
    np.testing.assert_array_equal(screen._canvas.mask > 0, completed.labels > 0)
    if completed.count:
        detail = screen._log.edits[-1].detail
        assert detail['method'] == mode
        assert detail['otsu_classes'] == 2
        assert detail['otsu_local'] is False
        screen._on_undo()
        assert not screen._canvas.mask.any()
