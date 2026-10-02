"""Optional second-model GUI wiring and stale-safe scientific-map exports."""
import json

import numpy as np
import pytest
import tifffile
from scipy import ndimage


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr import settings
    from spacr.qt import preferences
    from spacr.qt.prefs import _s
    from spacr.qt.screens.make_masks import MakeMasksScreen

    # Shared registry is integrated by the coordinating lane.
    entry = dict(settings.ALPHA_FEATURES[568])
    entry['widgets'] = tuple(entry['widgets']) + ('MakeMasksUncertaintyEnsembleSetting',)
    monkeypatch.setitem(settings.ALPHA_FEATURES, 568, entry)
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: True)
    _s().setValue('make_masks/uncertainty_second_model', '')
    for name in ('one', 'two'):
        image = np.zeros((32, 48), np.uint16)
        image[4:12, 5:14] = 2000
        tifffile.imwrite(tmp_path / f'{name}.tif', image)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(tmp_path))
    yield made
    _s().setValue('make_masks/uncertainty_second_model', '')


def segment(image):
    return ndimage.label(np.asarray(image) > 1000)[0]


def test_second_model_is_alpha_gated_and_persisted(screen, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.prefs import _s

    row = screen._uncertainty_ensemble.parentWidget()
    assert not row.isHidden()
    screen._uncertainty_ensemble.setEditText('/tmp/custom_checkpoint')
    assert screen._uncertainty_ensemble_model() == '/tmp/custom_checkpoint'
    assert _s().value('make_masks/uncertainty_second_model') == '/tmp/custom_checkpoint'
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: False)
    preferences._apply_alpha_widgets(screen)
    assert row.isHidden()


def test_snapshot_runs_both_models_with_captured_settings(screen, monkeypatch):
    from spacr.qt.screens import make_masks as mm

    calls = []

    def detect(request, models):
        calls.append((request['model'], dict(request['parameters'])))
        labels = segment(request['image'])
        if request['model'] == 'secondary':
            labels[:labels.shape[0] // 2] = 0
        return labels, None, None, None

    monkeypatch.setattr(mm, '_detect_cellpose_snapshot', detect)
    screen._uncertainty_ensemble.setEditText('secondary')
    assert screen._on_map_uncertainty(threaded=False)
    request, result = screen._uncertainty_result
    assert result['n_passes'] == 8
    assert [name for name, _settings in calls] == [request['detect']['model']] * 4 + ['secondary'] * 4
    assert all(parameters == request['detect']['parameters'] for _name, parameters in calls)


def test_saved_map_uses_completed_snapshot_not_changed_settings(screen, tmp_path):
    assert screen._on_map_uncertainty(segment=segment, threaded=False)
    request, result = screen._uncertainty_result
    screen._uncertainty_ensemble.setEditText('a-later-model')
    path = tmp_path / 'uncertainty' / 'map.tif'
    assert screen._on_save_uncertainty(str(path))
    assert np.array_equal(tifffile.imread(path), result['map'])
    with tifffile.TiffFile(path) as handle:
        metadata = json.loads(handle.pages[0].description)['spacr_uncertainty']
    assert metadata['primary_model'] == request['detect']['model']
    assert metadata['second_model'] is None
    assert metadata['source'].endswith('one.tif')


def test_navigation_and_blind_mode_prevent_stale_export(screen, tmp_path):
    screen._on_map_uncertainty(segment=segment, threaded=False)
    screen._blind = {'codes': {}}
    assert not screen._on_save_uncertainty(str(tmp_path / 'blocked.tif'))
    screen._blind = None
    screen._on_next()
    assert not screen._on_save_uncertainty(str(tmp_path / 'stale.tif'))
    assert not (tmp_path / 'stale.tif').exists()


def test_snapshot_rejects_same_model_twice(screen):
    from spacr.qt.screens.make_masks import _uncertainty_snapshot

    detect = screen._uncertainty_detect_request()
    request = dict(kind='field', image=screen._canvas.image,
                   segment=segment, detect=detect, second_model=detect['model'])
    with pytest.raises(ValueError, match='must differ'):
        _uncertainty_snapshot(request, {})
