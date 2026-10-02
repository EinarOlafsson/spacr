"""Opt-in unmixed Measure display preserves raw crops and source masks."""
import threading
from copy import deepcopy

import numpy as np
import pytest

from spacr.qt.widgets import measure_preview as mp


@pytest.fixture
def plate(tmp_path):
    folder = tmp_path / 'merged'
    folder.mkdir()
    first = np.zeros((32, 32))
    second = np.zeros_like(first)
    first[4:12, 4:12] = 120
    second[19:27, 18:26] = 80
    mask = np.zeros_like(first)
    mask[3:13, 3:13] = 1
    mask[18:28, 17:27] = 2
    paths = {}
    matrix = np.array([[1., .25], [.5, 1.]])
    for well, dyes in [('A01', (first, first * 0)),
                       ('B01', (second * 0, second)),
                       ('C01', (first, second))]:
        observed = np.stack(dyes, axis=-1) @ matrix.T + 10
        data = np.concatenate((observed, mask[..., None]), axis=-1).astype('uint16')
        path = folder / f'plate1_{well}_1.npy'
        np.save(path, data)
        paths[well] = path
    expected = np.dstack((first + 10, second + 10, mask))
    return paths, expected


def settings(channels=(0, 1)):
    return dict(unmix=True, unmix_controls='0:A01; 1:B01',
                unmix_background_percentile=5., channels=list(channels),
                _preview_mask_dims=[2])


@pytest.mark.parametrize('channels', [(0, 1), (1, 0)])
def test_actual_controls_independent_pixel_oracle_and_source_preservation(plate, channels):
    paths, expected = plate
    before = {path: path.read_bytes() for path in paths.values()}
    data = np.load(paths['C01'], mmap_mode='r')
    display, provenance = mp._unmixed_crop_source(
        data, paths['C01'], settings(channels), threading.Event(), {})
    np.testing.assert_array_equal(display, expected)
    assert display.dtype == data.dtype
    np.testing.assert_array_equal(display[..., 2], data[..., 2])
    assert provenance['channels'] == list(channels)
    assert provenance['input_modified'] is False
    assert provenance['display_only'] is True
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize('change,match', [
    ({'channels': [0, 2]}, 'mask planes'),
    ({'channels': [0, 0]}, 'distinct'),
    ({'channels': [0, 9]}, 'distinct'),
    ({'channels': [True, 1]}, 'distinct'),
    ({'unmix_controls': '0:D01; 1:E01'}, 'no field'),
])
def test_invalid_configuration_fails_without_source_writes(plate, change, match):
    paths, _ = plate
    before = paths['C01'].read_bytes()
    with pytest.raises(ValueError, match=match):
        mp._unmixed_crop_source(np.load(paths['C01']), paths['C01'],
                                dict(settings(), **change), threading.Event(), {})
    assert paths['C01'].read_bytes() == before


def panel(qtbot, plate, monkeypatch, threaded=False):
    monkeypatch.setattr('spacr.qt.preferences._is_alpha_visible', lambda *a: True)
    widget = mp.MeasurePreviewPanel(threaded=threaded)
    qtbot.addWidget(widget)
    options = settings()
    options.update(cell_mask_dim=2, nucleus_mask_dim=None, pathogen_mask_dim=None,
                   organelle_mask_dim=None, png_channel_mapping={'r': 0, 'g': 1, 'b': 0},
                   png_size=[16, 16], cell_min_size=1)
    widget.apply_settings(options)
    # None leaves numeric controls unchanged by the existing form contract.
    for name, control in widget._mask_dims.items():
        control.setValue(2 if name == 'cell' else -1)
    widget.load_array(str(plate[0]['C01']))
    qtbot.waitUntil(lambda: not widget._crop_running, timeout=10000)
    assert widget._crops
    return widget


def test_real_toggle_is_display_only_default_unchanged_and_failure_clears(qtbot, plate, monkeypatch):
    widget = panel(qtbot, plate, monkeypatch)
    assert not widget._unmix_btn.isChecked()
    raw = deepcopy(widget._crops)
    propagated = widget.settings_for_propagation()
    widget._unmix_btn.click()
    assert widget._crops and all('unmixing' in c for c in widget._crops)
    assert [(c['label'], c['phenotype'], c['included']) for c in widget._crops] == [
        (c['label'], c['phenotype'], c['included']) for c in raw]
    assert widget.settings_for_propagation() == propagated
    widget._unmix_btn.click()
    for actual, expected in zip(widget._crops, raw):
        np.testing.assert_array_equal(actual['crop'], expected['crop'])
        assert 'unmixing' not in actual
    widget.apply_settings({'unmix_controls': '0:D01; 1:E01'})
    widget._unmix_btn.click()
    assert widget._crops == []
    assert 'no field' in widget._status.text()


def test_unmeasured_display_channel_is_rejected(plate):
    path = plate[0]['C01']
    kwargs = dict(mask_dim=2, channels=[0, 1, 2], limit=10)
    result = mp._compute_checked_crops([path], path, np.load(path), kwargs,
                                       {}, threading.Event(), settings())
    assert not result['crops']
    assert 'every displayed colour' in result['warnings'][0]


def test_cancelled_load_never_reads_controls(plate, monkeypatch):
    from spacr.cancellation import PipelineCancelled
    stop = threading.Event()
    stop.set()
    path = plate[0]['C01']
    data = np.load(path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('read after cancel'))
    with pytest.raises(PipelineCancelled):
        mp._unmixed_crop_source(data, path, settings(), stop, {})


def test_threaded_old_unmix_cannot_replace_new_raw_display(qtbot, plate, monkeypatch):
    widget = panel(qtbot, plate, monkeypatch, threaded=True)
    raw = deepcopy(widget._crops)
    entered, release = threading.Event(), threading.Event()
    original = mp._unmixed_crop_source
    worker_threads = []

    def blocked(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(mp, '_unmixed_crop_source', blocked)
    widget._unmix_btn.click()
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert worker_threads[0] != threading.get_ident()
        widget._unmix_btn.click()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget._crop_running, timeout=10000)
    assert widget._crops
    for actual, expected in zip(widget._crops, raw):
        np.testing.assert_array_equal(actual['crop'], expected['crop'])
        assert 'unmixing' not in actual


def test_processed_crops_keep_integer_display_scaling_and_original_categories(plate):
    from spacr.measure import crop_objects_from_array
    paths, expected = plate
    path = paths['C01']
    source = np.load(path)
    kwargs = dict(mask_dim=2, channels=[1, 0, 1], normalize=False,
                  mask_background=False, limit=10)
    params = dict(object='cell', cell_dim=2, dims={}, minima={}, uninfected=True)
    actual = mp._compute_checked_crops([path], path, source, kwargs, params,
                                       threading.Event(), settings((1, 0)))
    oracle = crop_objects_from_array(expected.astype(source.dtype), **kwargs)
    assert len(actual['crops']) == len(oracle) == 2
    for entry, truth in zip(actual['crops'], oracle):
        np.testing.assert_array_equal(entry['crop'], truth['crop'])
        assert entry['label'] == truth['label']
        assert entry['included'] is True


def test_alpha_gate_disarms_processed_display(qtbot, plate, monkeypatch):
    widget = panel(qtbot, plate, monkeypatch)
    widget._unmix_btn.click()
    assert widget._unmix_btn.isChecked()
    monkeypatch.setattr('spacr.qt.preferences._is_alpha_visible', lambda *a: False)
    widget._refresh_alpha_visibility()
    assert not widget._unmix_btn.isChecked()
    assert widget._unmix_btn.isHidden()
    assert all('unmixing' not in entry for entry in widget._crops)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'pt', 'fr', 'zh_CN', 'hi', 'ko', 'is'])
def test_unmixed_display_control_uses_source_bound_locale(qtbot, monkeypatch, language):
    import hashlib
    import json
    from pathlib import Path

    from spacr.qt import i18n

    root = Path(__file__).resolve().parents[2]
    sources = [
        "Unmixed display",
        "Preview spectral unmixing using the run’s single-stain controls. "
        "This changes only the displayed crops; batch PNG exports and source files are unchanged.",
    ]
    active = [record
              for path in (root / 'docs/i18n/reviewed/runtime' / language).glob('*.json')
              for record in json.loads(path.read_text())['records']
              if record['table'] == 'ui' and record['source'] in sources]
    records = []
    for source in sources:
        matches = [record for record in active if record['source'] == source]
        assert matches, f"No current reviewed target for {source!r}"
        assert len({record['translation'] for record in matches}) == 1
        records.append(matches[0])
    for record in records:
        assert record['table'] == 'ui'
        assert record['key'] == record['source']
        assert record['source_sha256'] == hashlib.sha256(
            record['source'].encode()).hexdigest()
        assert record['translation'] != record['source']
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    widget = mp.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert widget._unmix_btn.text() == records[0]['translation']
    assert widget._unmix_btn.toolTip() == records[1]['translation']
    assert not widget._unmix_btn.isChecked()
