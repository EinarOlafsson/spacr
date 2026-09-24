"""Multi-Otsu must honor the selected bands in both magnifier scopes."""
from __future__ import annotations

import imageio.v3 as imageio
import numpy as np
import pytest
from skimage.filters import threshold_multiotsu

from spacr.qt import detect_chain
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    image = np.zeros((96, 96), dtype=np.uint16)
    image[12:84, 12:84] = 100
    image[24:72, 24:72] = 200
    image[36:60, 36:60] = 300
    image += np.random.default_rng(91).integers(0, 10, image.shape, dtype=np.uint16)
    imageio.imwrite(tmp_path / 'field.tif', image)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(tmp_path))
    monkeypatch.setattr(made, '_detect_chain', lambda: detect_chain.NO_CHAIN)
    made._otsu_smoothing.setValue(0)
    made._otsu_fill_holes.setChecked(False)
    made._otsu_split.setChecked(False)
    made._min_area.setValue(0)
    made._mag_mode.setCurrentIndex(made._mag_mode.findData('multiotsu'))
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


@pytest.mark.parametrize('scope', ['region', 'image'])
@pytest.mark.parametrize('classes,band', [(3, 1), (3, 2), (4, 1), (4, 3)])
def test_selected_band_matches_the_cpu_button_and_actual_thresholds(screen, scope, classes, band):
    screen._otsu_classes.setValue(classes)
    screen._otsu_foreground.setValue(band)
    magnifier = screen._magnifier
    if scope == 'region':
        request = magnifier.build_request()
    else:
        magnifier._start_image(magnifier._image_key_now())
        request = screen.image_requests[-1]
    result, used, note = mm._segment_region(request)
    assert used == 'multiotsu' and note == ''
    thresholds = threshold_multiotsu(request.crop.astype(np.float32), classes=classes)
    expected = np.digitize(request.crop, thresholds) == band
    np.testing.assert_array_equal(result > 0, expected)
    whole, _ = screen._cpu_detect(request.crop, 'multiotsu', screen._otsu_settings())
    np.testing.assert_array_equal(result, whole)


@pytest.mark.parametrize('attribute,value', [('_otsu_classes', 4), ('_otsu_foreground', 2)])
@pytest.mark.parametrize('scope', ['region', 'image'])
def test_band_edits_refresh_and_change_keys_without_moving_the_mouse(screen, attribute, value, scope):
    magnifier = screen._magnifier
    magnifier.scope = scope
    magnifier.refresh()
    submitted = screen.region_requests if scope == 'region' else screen.image_requests
    before = submitted[-1]
    getattr(screen, attribute).setValue(value)
    assert submitted[-1].key != before.key
    assert len(submitted) >= 2
    assert before.otsu_classes == 3
    assert before.otsu_foreground_class == 1


def test_switching_from_local_otsu_does_not_break_multiotsu(screen):
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData('otsu'))
    screen._otsu_local.setChecked(True)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData('multiotsu'))
    assert not screen._otsu_local.isEnabled()
    settings = screen._otsu_settings()
    assert settings['local'] is False
    assert settings['classes'] >= 3
    request = screen._magnifier.build_request()
    whole, _ = screen._cpu_detect(request.crop, 'multiotsu', settings)
    preview, used, note = mm._segment_region(request)
    assert used == 'multiotsu' and not note
    np.testing.assert_array_equal(preview, whole)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData('otsu'))
    assert screen._otsu_local.isEnabled()
    assert screen._otsu_settings()['local'] is True


@pytest.mark.parametrize('scope', ['region', 'image'])
def test_committed_mask_and_ledger_use_the_completed_request_band(screen, scope):
    screen._otsu_classes.setValue(4)
    screen._otsu_foreground.setValue(1)
    magnifier = screen._magnifier
    if scope == 'region':
        request = magnifier.build_request()
    else:
        magnifier._start_image(magnifier._image_key_now())
        request = screen.image_requests[-1]
    completed = magnifier._run(request)
    screen._otsu_classes.setValue(3)
    screen._otsu_foreground.setValue(2)
    clicked = mm._single_object(completed, 1) if scope == 'image' else completed
    assert screen._commit_magnifier_result(clicked)
    np.testing.assert_array_equal(screen._canvas.mask > 0, completed.labels > 0)
    detail = screen._log.edits[-1].detail
    assert (detail['mode'], detail['scope']) == ('multiotsu', scope)
    assert detail['otsu_classes'] == 4
    assert detail['otsu_foreground_class'] == 1
    screen._on_undo()
    assert not screen._canvas.mask.any()
    screen._otsu_classes.setValue(4)
    screen._otsu_foreground.setValue(1)
    screen._combine_mode.setCurrentText('replace')
    screen._btn_otsu.click()
    np.testing.assert_array_equal(screen._canvas.mask > 0, completed.labels > 0)


def test_insufficient_intensities_report_error_and_later_regions_can_succeed(screen):
    magnifier = screen._magnifier
    good = magnifier.build_request()
    bad = good._replace(crop=np.zeros_like(good.crop))
    with pytest.raises(ValueError):
        mm._segment_region(bad)
    magnifier._shown = magnifier._run(good)
    magnifier._requested_key = bad.key
    magnifier._on_delivered((bad, None, ValueError('too few intensity levels')))
    assert magnifier._shown is None
    assert not magnifier._unavailable
    recovered = magnifier._run(good)
    assert recovered.mode == 'multiotsu'
    assert recovered.count > 0
