"""Colony previews use run counts and original photo coordinates, without writes."""
import threading

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt

from spacr import plaque
from spacr.qt.widgets import plaque_preview as preview
from spacr.qt.widgets.segmentation_views import CELLPROB, FLOWS, MASKS


@pytest.fixture(autouse=True)
def offline_catalogue(monkeypatch):
    monkeypatch.setattr(preview, '_catalogue', lambda: [])


def _image(path, shape=(40, 60)):
    tifffile.imwrite(path, np.arange(np.prod(shape), dtype=np.uint16).reshape(shape))
    return path


def _count_result(image, *, well=None, **_kwargs):
    well = well or plaque.Well(0, 0, image.shape[1], image.shape[0])
    shape = plaque.crop_well(image, well).shape[:2]
    labels = np.zeros((shape[0] // 2, shape[1] // 2), np.int32)
    labels[1:3, 1:3] = 2
    return {'summary': {'colony_count': 2}, 'colonies': [{'area_px': 16}, {'area_px': 24}],
            'labels': labels, 'well': well, 'offset': (max(0, well.x0), max(0, well.y0))}


def test_raw_pixels_match_real_colony_run_and_source_is_unchanged(tmp_path, monkeypatch):
    from tests.test_colony_cfu_counting import _plate

    image, _, _, _ = _plate(12, 0, size=240, radius=(4, 6), seed=4)
    image = image.astype(np.uint16) * 128
    path = tmp_path / 'plate.tif'
    tifffile.imwrite(path, image)
    before = path.read_bytes()
    settings = {'colony_counting': True, 'colony_dilution': {'plate': 100},
                'colony_plated_volume_ul': 50, 'well_detection': False}
    expected = plaque._count_colony_plate(image, name=path.name, settings=settings)
    actual_count = plaque._count_colony_plate

    def observed(raw, **kwargs):
        np.testing.assert_array_equal(raw, image)
        assert raw.dtype == np.uint16
        return actual_count(raw, **kwargs)

    monkeypatch.setattr(plaque, '_count_colony_plate', observed)
    result = preview._colony_preview_pass(path, settings)
    assert result['count'] == expected['summary']['colony_count'] > 0
    assert result['areas'] == [row['area_px'] for row in expected['colonies']]
    assert result['colony_summaries'] == [expected['summary']]
    assert result['labels'].shape == image.shape[:2]
    assert result['flow_rgb'] is result['cellprob'] is None
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['plate.tif']


def test_multiple_wells_restore_offsets_and_keep_detector_box_counts(tmp_path, monkeypatch):
    path = _image(tmp_path / 'wells.tif')
    checkpoint = tmp_path / 'local.pt'
    checkpoint.write_bytes(b'fixture, never inferred')
    wells = [plaque.Well(-2, -2, 16, 16), plaque.Well(30, 20, 50, 40)]
    seen = []

    def detect(image, weights, **kwargs):
        assert image.dtype == np.uint16
        assert weights == str(checkpoint) and kwargs['confidence'] == 0.37
        return wells

    def count(image, **kwargs):
        seen.append(kwargs)
        assert kwargs['settings']['colony_detector'] == str(checkpoint)
        return _count_result(image, **kwargs)

    monkeypatch.setattr(plaque, 'detect_wells', detect)
    monkeypatch.setattr(plaque, '_count_colony_plate', count)
    result = preview._colony_preview_pass(path, {
        'well_detection': str(checkpoint), 'colony_detector': str(checkpoint),
        'well_confidence': 0.37})
    assert [entry['well'] for entry in seen] == wells
    expected = np.zeros((40, 60), np.int32)
    expected[2:6, 2:6] = 2
    expected[22:26, 32:36] = 4
    np.testing.assert_array_equal(result['labels'], expected)
    assert result['count'] == 4  # Two box rows per well, one visible ellipse per well.
    assert result['mean_area'] == 20
    assert result['areas'] == [16, 24, 16, 24]


@pytest.mark.parametrize('key', ['well_detection', 'colony_detector'])
def test_uncached_detector_is_offered_without_downloading_or_plaque_loading(tmp_path, monkeypatch, key):
    path = _image(tmp_path / 'plate.tif')
    entry = object()
    monkeypatch.setattr(preview, 'resolve_detector', lambda *_a: ('', 'not installed', entry))

    def forbidden(*_args, **_kwargs):
        pytest.fail('No model inference, plaque model resolution or download is allowed')

    monkeypatch.setattr(preview, 'resolve_plaque_model', forbidden)
    monkeypatch.setattr(plaque, '_count_colony_plate', forbidden)
    result = preview._colony_preview_pass(path, {key: 'missing-zoo-model'})
    assert result == {'error': 'not installed', 'entry': entry}


def test_actual_preview_button_uses_colonies_and_has_no_invented_flow_views(qtbot, tmp_path, monkeypatch):
    path = _image(tmp_path / 'plate.tif')
    monkeypatch.setattr(plaque, '_count_colony_plate', _count_result)

    def forbidden(*_args, **_kwargs):
        pytest.fail('Colony preview must not resolve a plaque model')

    monkeypatch.setattr(preview, 'resolve_plaque_model', forbidden)
    panel = preview.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({'colony_counting': True, 'plaque_model': 'missing-plaque.pt'})
    panel.load_source_async(path)
    qtbot.mouseClick(panel._run_btn, Qt.LeftButton)
    assert panel._plaque_result['colony']
    assert panel._plaque_result['count'] == 2
    assert panel._plaque_result['mean_area'] == 20
    assert panel.preview_status() == 'plate.tif: 2 colonies, mean area 20 px.'
    for mode in (FLOWS, CELLPROB):
        panel.views().set_view(mode)
        assert panel.views().is_showing_message()


def test_zero_colony_preview_uses_generic_empty_mask_message(qtbot, tmp_path, monkeypatch):
    path = _image(tmp_path / 'empty.tif')

    def empty(image, **kwargs):
        result = _count_result(image, **kwargs)
        result['labels'][:] = 0
        result['colonies'] = []
        result['summary']['colony_count'] = 0
        return result

    monkeypatch.setattr(plaque, '_count_colony_plate', empty)
    panel = preview.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({'colony_counting': True})
    panel.load_source_async(path)
    qtbot.mouseClick(panel._run_btn, Qt.LeftButton)
    assert panel.preview_status() == 'empty.tif: 0 colonies, mean area 0 px.'
    panel.views().set_view(MASKS)
    assert panel.views().is_showing_message()
    assert panel._render_masks_view({'labels': {'colony': panel._plaque_result['labels']}}) == 'Preview returned no masks.'


def test_missing_optional_detector_dependency_reports_failure_instead_of_different_method(tmp_path, monkeypatch):
    path = _image(tmp_path / 'plate.tif')
    checkpoint = tmp_path / 'colony.pt'
    checkpoint.write_bytes(b'fixture, never inferred')

    def missing(*_args, **_kwargs):
        raise ImportError('ultralytics is unavailable')

    monkeypatch.setattr(plaque, '_count_colony_plate', missing)
    result = preview._preview_call(lambda: preview._colony_preview_pass(
        path, {'colony_detector': str(checkpoint)}))
    assert 'ultralytics is unavailable' in result['error']
    assert 'labels' not in result and 'count' not in result


def test_colony_settings_edit_discards_blocked_preview_and_snapshots_nested_settings(qtbot, tmp_path, monkeypatch):
    path = _image(tmp_path / 'plate.tif')
    panel = preview.PlaquePreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    settings = {'colony_counting': True, 'colony_dilution': {'plate': 100}}
    panel.apply_settings(settings)
    panel.load_source_async(path)
    qtbot.waitUntil(lambda: panel.current_path() == path and not panel._load_jobs.is_busy(), timeout=5000)
    entered, release = threading.Event(), threading.Event()
    seen = []
    ready = []
    panel.preview_ready.connect(ready.append)

    def count(image, **kwargs):
        entered.set()
        assert release.wait(5)
        seen.append(kwargs['settings']['colony_dilution']['plate'])
        return _count_result(image, **kwargs)

    monkeypatch.setattr(plaque, '_count_colony_plate', count)
    try:
        qtbot.mouseClick(panel._run_btn, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        settings['colony_dilution']['plate'] = 999
        panel.apply_settings({'colony_counting': False})
        assert not panel._run_btn.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: panel._jobs.active_jobs() == 0, timeout=5000)
    qtbot.waitUntil(panel._run_btn.isEnabled, timeout=5000)
    assert seen == [100] and ready == [] and panel._plaque_result is None
    panel.shutdown()
